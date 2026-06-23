from __future__ import annotations

import os
import sys
import time
from typing import Any

from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from UtilsManager.UnifiedCacheManager import UnifiedCacheManager


class StockSyncEngine:

    def __init__(self, config_file: str = "config.ini", executor: Any = None, db_engine: Engine | None = None) -> None:

        self.config_file = config_file
        self.config = Config(config_file=config_file)

        if db_engine is not None:
            self.db = db_engine
        else:
            url_object = URL.create(
                "postgresql+psycopg2",
                username=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                database=self.config.DB_NAME,
            )
            self.db = create_engine(url_object, pool_pre_ping=True, pool_recycle=3600, echo=False, client_encoding="utf8")

            try:
                with self.db.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("[INFO]  数据库引擎初始化成功")
            except Exception as e:
                raise RuntimeError("数据库引擎初始化失败") from e

        self.calendar_mgr = TradingCalendarAnalyzer()
        self.today = self.calendar_mgr.get_last_trading_day()
        self.today_dt = pd.to_datetime(self.today).normalize()

        self.base_data_dir = self.config.TEMP_DATA_DIRECTORY
        os.makedirs(self.base_data_dir, exist_ok=True)

        self.cache_manager = UnifiedCacheManager(
            cache_dir=self.base_data_dir, today_str=self.today
        )

        self.main_report_cache_path = self.cache_manager.get_cache_path(
            "主力研报盈利预测_完整数据", cleaned=True, suffix=".csv"
        )
        self.raw_report_cache_path = self.cache_manager.get_cache_path("主力研报盈利预测", cleaned=True)

    def get_stock_pool_from_db(self) -> pd.DataFrame:
        """
        从 stock_basic_info_sw 表获取全量股票池
        返回包含 ts_code、name、industry
        """
        try:
            query = """
            SELECT 
               stock_code as  ts_code,
                stock_code as 股票代码,
                stock_name as name,
                industry_name as industry
            FROM stock_basic_info_sw
            ORDER BY stock_code
            """

            with self.db.connect() as conn:
                stock_index_df = pd.read_sql(text(query), conn)

            logger.info(f"[INFO] 从数据库获取 {len(stock_index_df)} 只股票。")

            if "股票代码" in stock_index_df.columns:
                stock_index_df["股票代码"] = stock_index_df["股票代码"].astype(str).str.zfill(6)

            required_cols = ["ts_code", "name", "industry", "股票代码"]
            for col in required_cols:
                if col not in stock_index_df.columns:
                    stock_index_df[col] = "N/A"

            return stock_index_df[required_cols]

        except Exception as e:
            logger.info(f"[ERROR] 从数据库获取股票池失败: {e}")
            return pd.DataFrame(columns=["ts_code", "name", "industry", "股票代码"])

    def _safe_ak_fetch(
        self, fetch_func: callable, description: str, cache_base_name: str | None = None, **kwargs: Any  # noqa: ANN401
    ) -> pd.DataFrame:
        """带重试、缓存、清洗的 Akshare 数据获取。"""

        def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df.columns = [col.strip() for col in df.columns]
            if "代码" in df.columns and "股票代码" not in df.columns:
                df.rename(columns={"代码": "股票代码"}, inplace=True)
            if "名称" in df.columns and "股票简称" not in df.columns:
                df.rename(columns={"名称": "股票简称"}, inplace=True)
            return df

        # 1. 尝试加载清洗后的缓存
        if cache_base_name:
            cached_df = self.cache_manager.load_cache(
                cache_base_name,
                cleaned=True,
                sep="|",
                encoding="utf-8-sig",
                dtype_mapping={
                    "股票代码": str,
                    "股票简称": str,
                    "机构投资评级(近六个月)-买入": float,
                    "2024预测每股收益": float,
                    "2025预测每股收益": float,
                },
            )
            if not cached_df.empty:
                cached_df = _standardize_columns(cached_df)
                logger.info(
                    f"  -  从缓存加载: {os.path.basename(self.cache_manager.get_cache_path(cache_base_name, cleaned=True))}"
                )
                return cached_df

        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                logger.info(f"  - 正在尝试第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次: {description}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    df = _standardize_columns(df)
                    break

                # 使用指数级退避递增重试延时
                wait_time = self.config.DATA_FETCH_DELAY * (2**i)
                logger.info(f"[WARN] 获取 {description} 返回空或无效，将在 {wait_time} 秒后重试")
                time.sleep(wait_time)
            except Exception as e:
                wait_time = self.config.DATA_FETCH_DELAY * (2**i)
                logger.info(f"[ERROR] 获取 {description} 失败: {e}，将在 {wait_time} 秒后重试")
                time.sleep(wait_time)

        if df.empty:
            logger.info(f"[CRITICAL] 所有重试失败，未能获取 {description}")
            return pd.DataFrame()

        # 清洗 + 保存到缓存
        if "股票代码" in df.columns:
            df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
            df = df.drop_duplicates(subset=["股票代码"])

        if cache_base_name and not df.empty:
            self.cache_manager.save_cache(df, cache_base_name, cleaned=True, sep="|", encoding="utf-8-sig")

        return df

    def _get_research_report_data(self) -> pd.DataFrame:
        """
        获取研报数据（不过滤），返回包含股票代码和买入评级的DataFrame。
        用于后续作为特征列添加到报告中。
        """
        logger.info("\n>>> 正在获取主力研报盈利预测数据...")

        # 获取原始数据
        report_df = self._safe_ak_fetch(
            fetch_func=ak.stock_profit_forecast_em, description="主力研报盈利预测", cache_base_name="主力研报盈利预测"
        )

        if report_df.empty:
            logger.info("[WARNING] 未获取到研报数据，返回空DataFrame")
            return pd.DataFrame()

        # 标准化列名
        if "股票代码" not in report_df.columns:
            report_df.rename(columns={"代码": "股票代码"}, inplace=True)
        logger.info(f"[DEBUG] 研报原始列名: {list(report_df.columns)}")

        # 买入评级列：尝试多个可能的列名
        rating_col_candidates = ["机构投资评级(近六个月)-买入", "买入 (次)", "买入评级次数", "买入评级", "买入次数", "机构买入评级"]
        found_rating_col = None
        for col in rating_col_candidates:
            if col in report_df.columns:
                found_rating_col = col
                break
        if found_rating_col and found_rating_col != "机构投资评级(近六个月)-买入":
            report_df.rename(columns={found_rating_col: "机构投资评级(近六个月)-买入"}, inplace=True)
        elif not found_rating_col:
            logger.warning(f"[WARNING] 无法识别研报买入评级列，可用列: {list(report_df.columns)}")
            return pd.DataFrame()

        # 清洗
        report_df = report_df.drop_duplicates(subset=["股票代码"])
        report_df["股票代码"] = report_df["股票代码"].astype(str).str.zfill(6)

        # 转换为数值型
        report_df["机构投资评级(近六个月)-买入"] = (
            pd.to_numeric(report_df["机构投资评级(近六个月)-买入"], errors="coerce").fillna(0).astype(int)
        )

        logger.info(f"[INFO] 获取到 {len(report_df)} 只股票的研报数据。")
        return report_df[["股票代码", "机构投资评级(近六个月)-买入"]]

    def backfill_close_normal(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """获取全市场最新不复权收盘价，委托给 IncrementalSyncEngine。"""
        if not hasattr(self, '_sync_engine') or self._sync_engine is None:
            logger.info("[close_normal] 尚未初始化同步引擎，跳过回填")
            return pd.DataFrame()
        return self._sync_engine.backfill_close_normal(symbols)

    def run_engine(self, target_date: str | None = None) -> set[str] | None:
        """主运行函数：研报过滤 + K线数据同步"""

        if target_date is None:
            try:
                target_date = TradingCalendarAnalyzer().get_last_trading_day()
            except (ValueError, ConnectionError, RuntimeError):
                target_date = self.today

        self.today_str = target_date
        self.today_dt = pd.to_datetime(target_date).normalize()

        logger.info(f"[DEBUG] 数据引擎运行日期: {self.today_str}")

        # Step 1: 获取数据库中的股票池
        stock_pool_df = self.get_stock_pool_from_db()
        if stock_pool_df.empty or "股票代码" not in stock_pool_df.columns:
            logger.info("[CRITICAL] 基础股票池无效")
            return set()

        # Step 1.5: 过滤ST股票
        if "name" in stock_pool_df.columns:
            st_pattern = r"(?:\s*(?:\*|★|※|•|·))?(?:[Ss][Tt])"
            before_count = len(stock_pool_df)
            stock_pool_df = stock_pool_df[~stock_pool_df["name"].astype(str).str.contains(st_pattern, na=False)].copy()
            after_count = len(stock_pool_df)
            filtered_count = before_count - after_count
            if filtered_count > 0:
                print(f"  过滤: 剔除 {filtered_count} 只ST → {after_count} 只", flush=True)
                logger.info(f"[FILTER] 已过滤 {filtered_count} 只ST股票，剩余 {after_count} 只正常股票。")
            else:
                print(f"  过滤: 无ST股票 ({after_count} 只)", flush=True)
                logger.info("[INFO] 无ST股票需要过滤。")

        pure_codes = set(stock_pool_df["股票代码"].unique().tolist())
        logger.info(f"[INFO] 从数据库获取 {len(pure_codes)} 只股票（已剔除ST）。")

        # Step 2: 获取研报数据（作为特征，不过滤）
        report_df = self._get_research_report_data()

        # 将研报数据保存到processed_data，供后续合并使用
        if not report_df.empty:
            # 保存到缓存文件，供 DataProcessingService 读取
            report_cache_path = self.cache_manager.get_cache_path("研报买入次数", cleaned=True)
            try:
                report_df.to_csv(report_cache_path, sep="|", index=False, encoding="utf-8-sig")
                logger.info(f"[INFO] 研报数据已保存至缓存: {os.path.basename(report_cache_path)}")
            except Exception as e:
                logger.info(f"[ERROR] 保存研报数据失败: {e}")

        # 代码过滤：全 A 股模式跳过所有过滤，否则按配置过滤
        full_a_share = self.config.app_config.backtest.FULL_A_SHARE_MODE
        if full_a_share:
            final_codes = pure_codes
            print(f"  过滤: 全A模式，跳过所有过滤 → {len(final_codes)} 只", flush=True)
            logger.info(f"[INFO] 全 A 股模式，跳过主板/研报过滤，分析池包含 {len(final_codes)} 只股票。")
        else:
            if self.config.MAIN_BOARD_ONLY:
                final_codes = {code for code in pure_codes if code.startswith(("60", "00"))}
                print(f"  过滤: 仅主板 → {len(final_codes)} 只", flush=True)
                logger.info(f"[INFO] 已开启主板过滤，分析池包含 {len(final_codes)} 只主板股票。")
            else:
                final_codes = pure_codes
                print(f"  过滤: 全市场 → {len(final_codes)} 只", flush=True)
                logger.info(f"[INFO] 全市场模式，分析池包含 {len(final_codes)} 只股票。")

            # 研报过滤（全 A 股模式下跳过）
            if self.config.ENABLE_RESEARCH_REPORT_FILTER and not report_df.empty:
                print(f"\n  [研报过滤] 启用 (阈值>{self.config.RESEARCH_REPORT_MIN_COUNT}次买入)", flush=True)
                logger.info(f"\n[研报过滤] 启用研报二次过滤，阈值: {self.config.RESEARCH_REPORT_MIN_COUNT} 次买入评级")
                filtered_report_df = report_df[
                    report_df["机构投资评级(近六个月)-买入"] > self.config.RESEARCH_REPORT_MIN_COUNT
                ]
                report_filtered_codes = set(filtered_report_df["股票代码"].unique().tolist())
                before_count = len(final_codes)
                final_codes = final_codes.intersection(report_filtered_codes)
                after_count = len(final_codes)
                msg = f"  [研报过滤] {before_count} → {after_count} (过滤掉 {before_count - after_count} 只)"
                print(msg, flush=True)
                logger.info(f"[研报过滤] 原始股票数: {before_count}, 过滤后: {after_count}, 过滤掉: {before_count - after_count}")
                if after_count == 0:
                    print("  [研报过滤] 过滤后无股票剩余，终止", flush=True)
                    logger.info("[警告] 研报过滤后无股票剩余")
                    return set()
            elif self.config.ENABLE_RESEARCH_REPORT_FILTER:
                print("  [研报过滤] 已启用但无研报数据，跳过过滤", flush=True)
                logger.info("[警告] 研报过滤已启用，但未获取到研报数据，跳过研报过滤")

        # 【统一数据同步】使用 IncrementalSyncEngine 做增量同步
        from DataManager.IncrementalSyncEngine import IncrementalSyncEngine
        from UtilsManager.CodeNormalizer import CodeNormalizer

        cache_dir = os.path.join(self.config.CACHE_DIRECTORY, "kline_batches")
        self._sync_engine = IncrementalSyncEngine(
            self.db,
            asharehub_api_key=getattr(self.config, 'ASHAREHUB_API_KEY', None),
            cache_dir=cache_dir,
        )
        self._sync_symbols = [CodeNormalizer.add_market_prefix(code) for code in sorted(final_codes)]
        logger.info(f"[INFO] 同步 {len(self._sync_symbols)} 只股票到 stock_daily_kline...")
        inserted = self._sync_engine.sync_all(self._sync_symbols)
        logger.info(f"[INFO] 同步完成，新增 {inserted} 行")

        # close_normal 回填推迟到 Step 4（不阻塞研报过滤等后续步骤），
        # 由 StockAnalysisCoordinator._step_4_get_kline_and_prices 按需触发

        # 保存最终股票列表
        final_output_path = os.path.join(self.base_data_dir, f"final_filtered_stocks_{self.today}.txt")
        try:
            with open(final_output_path, "w", encoding="utf-8") as f:
                for code in sorted(final_codes):
                    f.write(f"{code}\n")
            logger.info(f"[INFO] 最终股票列表已保存至: {final_output_path}")
        except Exception as e:
            logger.info(f"[ERROR] 保存最终代码列表失败: {e}")

        return final_codes

