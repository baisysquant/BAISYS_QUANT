import os
import sys

# 将项目根目录加入系统路径，支持直接运行此脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import akshare as ak
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError, OperationalError

from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from DataManager.ShareCodeFormatMgr import format_stock_code
from UtilsManager.UnifiedCacheManager import UnifiedCacheManager


class StockSyncEngine:

    def __init__(self, config_file: str = "config.ini", executor=None):

        self.config_file = config_file
        self.config = Config(config_file=config_file)
        self.executor = executor

        url_object = URL.create(
            "postgresql+psycopg2",
            username=self.config.DB_USER,
            password=self.config.DB_PASSWORD,
            host=self.config.DB_HOST,
            port=self.config.DB_PORT,
            database=self.config.DB_NAME,
        )

        #  初始化数据库引擎
        self.db = create_engine(url_object, pool_pre_ping=True, pool_recycle=3600, echo=False, client_encoding="utf8")

        #  测试数据库连接
        try:
            with self.db.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("[INFO]  数据库引擎初始化成功")
        except (DBAPIError, OperationalError) as e:
            raise RuntimeError("数据库引擎初始化失败") from e

        # 使用业务交易日而非物理时间
        self.calendar_mgr = TradingCalendarAnalyzer()
        self.today = self.calendar_mgr.get_last_trading_day()
        self.today_dt = pd.to_datetime(self.today).normalize()

        # 根据配置的天数动态计算起始日期
        self.global_start = self._calculate_start_date(self.calendar_mgr, self.config.KLINE_HISTORY_DAYS)
        print(f"[INFO] K线数据获取范围: {self.global_start} 至 {self.today} (共{self.config.KLINE_HISTORY_DAYS}天)")

        self.base_data_dir = self.config.TEMP_DATA_DIRECTORY
        os.makedirs(self.base_data_dir, exist_ok=True)

        # 初始化缓存管理器（向后兼容 API）
        self.cache_manager = UnifiedCacheManager(
            cache_dir=self.base_data_dir, today_str=self.today
        )

        # 缓存文件路径（使用 CacheManager 生成）
        self.main_report_cache_path = self.cache_manager.get_cache_path(
            "主力研报盈利预测_完整数据", cleaned=True, suffix=".csv"
        )
        self.raw_report_cache_path = self.cache_manager.get_cache_path("主力研报盈利预测", cleaned=True)
        self.kline_cache_path = self.cache_manager.get_cache_path("股票K线数据_已处理", cleaned=True, suffix=".csv")

        # 失败股票列表文件路径
        self.failed_symbols_file = os.path.join(self.base_data_dir, f"failed_symbols_{self.today}.json")

        # 已成功获取的股票列表文件路径（用于增量更新）
        self.success_symbols_file = os.path.join(self.base_data_dir, f"success_symbols_{self.today}.json")

    def _calculate_start_date(self, calendar_mgr, history_days: int) -> str:
        """
        根据配置的天数动态计算K线数据的起始日期

        Args:
            calendar_mgr: 交易日历管理器实例
            history_days: 历史天数（交易日）

        Returns:
            str: 格式为YYYYMMDD的起始日期字符串
        """
        try:
            # 获取当前交易日往前推history_days天的交易日
            start_date = calendar_mgr.get_trading_day_offset(-history_days)
            # 转换为YYYYMMDD格式
            start_date_str = start_date.replace("-", "")
            return start_date_str
        except Exception as e:
            print(f"[WARNING] 计算起始日期失败: {e}，使用默认值20250301")
            return "20250301"

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

            print(f"[INFO] 从数据库获取 {len(stock_index_df)} 只股票。")

            if "股票代码" in stock_index_df.columns:
                stock_index_df["股票代码"] = stock_index_df["股票代码"].astype(str).str.zfill(6)

            required_cols = ["ts_code", "name", "industry", "股票代码"]
            for col in required_cols:
                if col not in stock_index_df.columns:
                    stock_index_df[col] = "N/A"

            return stock_index_df[required_cols]

        except Exception as e:
            print(f"[ERROR] 从数据库获取股票池失败: {e}")
            return pd.DataFrame(columns=["ts_code", "name", "industry", "股票代码"])

    def _safe_ak_fetch(
        self, fetch_func: callable, description: str, cache_base_name: str = None, **kwargs
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
                print(
                    f"  -  从缓存加载: {os.path.basename(self.cache_manager.get_cache_path(cache_base_name, cleaned=True))}"
                )
                return cached_df

        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                print(f"  - 正在尝试第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次: {description}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    df = _standardize_columns(df)
                    break

                # 使用指数级退避递增重试延时
                wait_time = self.config.DATA_FETCH_DELAY * (2**i)
                print(f"[WARN] 获取 {description} 返回空或无效，将在 {wait_time} 秒后重试")
                time.sleep(wait_time)
            except Exception as e:
                wait_time = self.config.DATA_FETCH_DELAY * (2**i)
                print(f"[ERROR] 获取 {description} 失败: {e}，将在 {wait_time} 秒后重试")
                time.sleep(wait_time)

        if df.empty:
            print(f"[CRITICAL] 所有重试失败，未能获取 {description}")
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
        print("\n>>> 正在获取主力研报盈利预测数据...")

        # 获取原始数据
        report_df = self._safe_ak_fetch(
            fetch_func=ak.stock_profit_forecast_em, description="主力研报盈利预测", cache_base_name="主力研报盈利预测"
        )

        if report_df.empty:
            print("[WARNING] 未获取到研报数据，返回空DataFrame")
            return pd.DataFrame()

        # 标准化列名
        if "股票代码" not in report_df.columns:
            report_df.rename(columns={"代码": "股票代码"}, inplace=True)
        if "机构投资评级(近六个月)-买入" not in report_df.columns:
            report_df.rename(columns={"买入 (次)": "机构投资评级(近六个月)-买入"}, inplace=True)

        # 清洗
        report_df = report_df.drop_duplicates(subset=["股票代码"])
        report_df["股票代码"] = report_df["股票代码"].astype(str).str.zfill(6)

        # 转换为数值型
        report_df["机构投资评级(近六个月)-买入"] = (
            pd.to_numeric(report_df["机构投资评级(近六个月)-买入"], errors="coerce").fillna(0).astype(int)
        )

        print(f"[INFO] 获取到 {len(report_df)} 只股票的研报数据。")
        return report_df[["股票代码", "机构投资评级(近六个月)-买入"]]

    def _fetch_kline_for_symbol(self, symbol: str) -> pd.DataFrame:
        """获取单个股票的前复权 + 不复权数据，合并输出"""
        try:
            # 尝试获取前复权数据
            df_qfq = ak.stock_zh_a_hist_tx(
                symbol=symbol, start_date=self.global_start, end_date=self.today, adjust="qfq"
            )
            time.sleep(0.05)

            if df_qfq is None or df_qfq.empty:
                return None

            expected_cols = ["date", "open", "close", "high", "low", "amount"]
            missing = [c for c in expected_cols if c not in df_qfq.columns]
            if missing:
                # 北交所股票可能返回不同结构的数，记录并跳过
                print(f"[WARN] {symbol} QFQ 数据缺失列: {missing}，跳过。")
                return None

            # 尝试获取不复权数据
            df_norm = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=self.global_start, end_date=self.today, adjust="")
            time.sleep(0.05)

            if df_norm is None or df_norm.empty:
                return None

            if "close" not in df_norm.columns or "amount" not in df_norm.columns:
                print(f"[WARN] {symbol} 不复权数据缺失必要列，跳过。")
                return None

            df_norm = df_norm[["date", "close", "amount"]].rename(
                columns={"close": "close_normal", "amount": "volume_normal"}
            )

            # 合并数据
            df = pd.merge(df_qfq, df_norm, on="date", how="inner")
            if df.empty:
                return None

            # 数值转换与计算
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["close_normal"] = pd.to_numeric(df["close_normal"], errors="coerce")
            df["adj_ratio"] = df["close"] / df["close_normal"].replace(0, pd.NA)

            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["volume_normal"] = pd.to_numeric(df["volume_normal"], errors="coerce")
            df["volume_adj_ratio"] = df["amount"] / df["volume_normal"].replace(0, pd.NA)
            df["volume"] = df["volume_normal"] * df["volume_adj_ratio"]

            df.dropna(subset=["adj_ratio", "volume"], inplace=True)

            if df.empty:
                return None

            df["symbol"] = format_stock_code(symbol)
            df["date"] = pd.to_datetime(df["date"])
            df.rename(columns={"date": "trade_date"}, inplace=True)

            final_cols = [
                "trade_date",
                "symbol",
                "open",
                "close",
                "high",
                "low",
                "amount",
                "close_normal",
                "volume",
                "adj_ratio",
            ]
            return df[final_cols]
        except KeyError as e:
            # 专门捕获键错误（如 'day'），这通常意味着接口返回结构异常
            print(f"[ERROR] 获取 {symbol} 数据结构异常 (KeyError: {e})，可能是接口不支持该股票。")
            return None
        except Exception as e:
            print(f"[ERROR] 获取 {symbol} 数据失败: {e}")
            return None

    def _fetch_kline_with_delay(self, symbol: str, delay_seconds: float = 0) -> pd.DataFrame:
        """
        带错峰延迟的 K 线获取方法

        Args:
            symbol: 股票代码
            delay_seconds: 请求前的延迟时间（秒）
        """
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        return self._fetch_kline_for_symbol(symbol)

    def _load_failed_symbols(self) -> list[str]:
        """加载上次失败的股票列表"""
        if not os.path.exists(self.failed_symbols_file):
            return []

        try:
            with open(self.failed_symbols_file, encoding="utf-8") as f:
                failed_list = json.load(f)
            print(f"[断点续传] 加载失败列表: {len(failed_list)} 只股票")
            return failed_list
        except Exception as e:
            print(f"[WARN] 加载失败列表失败: {e}")
            return []

    def _save_failed_symbols(self, failed_symbols: list[str]):
        """保存失败的股票列表到本地"""
        try:
            with open(self.failed_symbols_file, "w", encoding="utf-8") as f:
                json.dump(failed_symbols, f, ensure_ascii=False, indent=2)
            print(f"[断点续传] 已保存 {len(failed_symbols)} 只失败股票至: {os.path.basename(self.failed_symbols_file)}")
        except Exception as e:
            print(f"[ERROR] 保存失败列表失败: {e}")

    def _clear_failed_symbols(self):
        """清除失败股票列表文件（全部成功时调用）"""
        if os.path.exists(self.failed_symbols_file):
            try:
                os.remove(self.failed_symbols_file)
                print("[断点续传] 已清除失败列表文件")
            except Exception as e:
                print(f"[WARN] 清除失败列表文件失败: {e}")

    def _load_success_symbols(self) -> set[str]:
        """加载已成功获取的股票列表（用于增量更新）"""
        if not os.path.exists(self.success_symbols_file):
            return set()

        try:
            with open(self.success_symbols_file, encoding="utf-8") as f:
                success_set = set(json.load(f))
            print(f"[增量更新] 发现今日已成功获取 {len(success_set)} 只股票，将跳过重复获取")
            return success_set
        except Exception as e:
            print(f"[WARN] 加载成功列表失败: {e}")
            return set()

    def _save_success_symbols(self, success_symbols: list[str], append: bool = True):
        """
        保存已成功获取的股票列表

        Args:
            success_symbols: 新成功的股票列表
            append: 是否追加到已有列表（True=追加，False=覆盖）
        """
        try:
            existing = set()
            if append and os.path.exists(self.success_symbols_file):
                with open(self.success_symbols_file, encoding="utf-8") as f:
                    existing = set(json.load(f))

            # 合并新旧成功列表
            all_success = existing | set(success_symbols)

            with open(self.success_symbols_file, "w", encoding="utf-8") as f:
                json.dump(list(all_success), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] 保存成功列表失败: {e}")

    def _clear_success_symbols(self):
        """清除成功股票列表文件"""
        if os.path.exists(self.success_symbols_file):
            try:
                os.remove(self.success_symbols_file)
                print("[增量更新] 已清除成功列表文件")
            except Exception as e:
                print(f"[WARN] 清除成功列表文件失败: {e}")

    def _has_today_batch_files(self, kline_cache_dir: str) -> bool:
        """
        检查是否有当前业务交易日的批次文件

        Args:
            kline_cache_dir: 批次缓存目录路径

        Returns:
            bool: 如果有当前交易日的批次文件返回True，否则返回False
        """
        if not os.path.exists(kline_cache_dir):
            return False

        try:
            # 获取最后一个交易日（YYYY-MM-DD），并转换为 YYYYMMDD
            trading_date_prefix = self.today.replace("-", "")

            for filename in os.listdir(kline_cache_dir):
                if filename.startswith("kline_batch_") and filename.endswith(".csv"):
                    # 提取时间戳部分（格式：kline_batch_YYYYMMDDHHMMSS.csv）
                    parts = filename.replace(".csv", "").split("_")
                    if len(parts) == 3:
                        file_timestamp = parts[2]
                        file_date = file_timestamp[:8]
                        # 如果找到当前交易日的批次文件，立即返回True
                        if file_date == trading_date_prefix:
                            return True

            return False
        except Exception as e:
            print(f"[WARN] 检查批次文件失败: {e}")
            return False

    def _cleanup_old_batch_files(self, kline_cache_dir: str):
        """清理旧的批次文件，只保留当前业务交易日的批次（通过时间戳判断）"""
        if not os.path.exists(kline_cache_dir):
            return

        try:
            # 获取最后一个交易日（YYYY-MM-DD），并转换为 YYYYMMDD
            trading_date_prefix = self.today.replace("-", "")
            cleaned_count = 0

            for filename in os.listdir(kline_cache_dir):
                if filename.startswith("kline_batch_") and filename.endswith(".csv"):
                    # 从文件名中提取时间戳部分（格式：kline_batch_YYYYMMDDHHMMSS.csv）
                    parts = filename.replace(".csv", "").split("_")
                    if len(parts) == 3:
                        file_timestamp = parts[2]  # 第3部分是时间戳（YYYYMMDDHHMMSS）
                        file_date = file_timestamp[:8]  # 提取前8位作为日期（YYYYMMDD）

                        # 如果不是当前交易日的批次文件，则删除
                        if file_date != trading_date_prefix:
                            file_path = os.path.join(kline_cache_dir, filename)
                            os.remove(file_path)
                            print(f"[清理] 已删除旧批次文件: {filename} (数据日期: {file_date})")
                            cleaned_count += 1
                    else:
                        # 兼容旧格式（带批次序号或无时间戳），直接删除
                        file_path = os.path.join(kline_cache_dir, filename)
                        os.remove(file_path)
                        print(f"[清理] 已删除旧批次文件: {filename} (旧格式)")
                        cleaned_count += 1

            if cleaned_count > 0:
                print(f"[清理] 共清理 {cleaned_count} 个旧批次文件")
        except Exception as e:
            print(f"[WARN] 清理旧批次文件失败: {e}")

    def _load_and_merge_cached_data(self, kline_cache_dir: str):
        """
        加载并合并今日已缓存的批次数据，直接写入数据库

        Args:
            kline_cache_dir: 批次缓存目录路径
        """
        if not os.path.exists(kline_cache_dir):
            print("[WARN] 批次缓存目录不存在")
            return

        # 查找所有批次文件（只加载当前交易日的）
        trading_date_prefix = self.today.replace("-", "")
        batch_files = []

        for f in os.listdir(kline_cache_dir):
            if f.startswith("kline_batch_") and f.endswith(".csv"):
                # 提取时间戳部分（格式：kline_batch_YYYYMMDDHHMMSS.csv）
                parts = f.replace(".csv", "").split("_")
                if len(parts) == 3:
                    file_timestamp = parts[2]  # 第3部分是时间戳
                    file_date = file_timestamp[:8]
                    # 只保留当前交易日的批次文件
                    if file_date == trading_date_prefix:
                        batch_files.append(f)
                else:
                    # 兼容旧格式（带批次序号或无时间戳），也加载
                    batch_files.append(f)

        # 按文件名排序（保证批次顺序）
        batch_files = sorted(batch_files)

        if not batch_files:
            print("[WARN] 未找到任何批次缓存文件")
            return

        print(f"[INFO] 发现 {len(batch_files)} 个批次缓存文件，开始合并...")

        # 加载并合并所有批次
        all_dfs = []
        total_records = 0

        for batch_file in batch_files:
            try:
                file_path = os.path.join(kline_cache_dir, batch_file)
                df = pd.read_csv(file_path, sep="|", encoding="utf-8-sig")
                all_dfs.append(df)
                total_records += len(df)
                print(f"  - 加载 {batch_file}: {len(df)} 条记录")
            except Exception as e:
                print(f"[ERROR] 加载 {batch_file} 失败: {e}")

        if not all_dfs:
            print("[WARNING] 所有批次文件加载失败")
            return

        # 合并所有批次
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[INFO] 成功合并 {total_records} 条 K 线记录。")

        # 保存到完整缓存文件
        try:
            combined_df.to_csv(self.kline_cache_path, sep="|", index=False, encoding="utf-8-sig")
            print(f"[INFO] K线数据已保存至本地缓存: {os.path.basename(self.kline_cache_path)}")
        except Exception as e:
            print(f"[ERROR] 保存 K线数据缓存失败: {e}")

        # 写入数据库
        try:
            combined_df.to_sql(
                name="stock_daily_kline", con=self.db, if_exists="replace", index=False, method="multi", chunksize=5000
            )
            print(f"[INFO] 成功将 {len(combined_df)} 条记录写入 'stock_daily_kline' 表。")
        except (DBAPIError, OperationalError) as e:
            print(f"[ERROR] 写入数据库失败: {e}")
            try:
                with self.db.connect() as conn:
                    conn.rollback()
            except (DBAPIError, OperationalError):
                pass
            raise

    def _clear_stock_daily_kline_table(self):
        """清空 stock_daily_kline 表"""
        if self.db is None:
            print("[CRITICAL] 数据库未初始化")
            return
        try:
            with self.db.connect() as conn:
                conn.execute(text("DELETE FROM stock_daily_kline;"))
                conn.commit()
            print("[INFO] 'stock_daily_kline' 表已清空。")
        except (DBAPIError, OperationalError) as e:
            print(f"[ERROR] 清空失败: {e}")
            try:
                with self.db.connect() as conn:
                    conn.rollback()
            except (DBAPIError, OperationalError):
                pass
            raise

    def run_engine(self, target_date: str = None):
        """主运行函数：研报过滤 + K线数据同步"""

        if target_date is None:
            try:
                target_date = TradingCalendarAnalyzer().get_last_trading_day()
            except (ValueError, ConnectionError, RuntimeError):
                target_date = self.today

        self.today_str = target_date
        self.today_dt = pd.to_datetime(target_date).normalize()

        print(f"[DEBUG] 数据引擎运行日期: {self.today_str}")

        # Step 1: 获取数据库中的股票池
        stock_pool_df = self.get_stock_pool_from_db()
        if stock_pool_df.empty or "股票代码" not in stock_pool_df.columns:
            print("[CRITICAL] 基础股票池无效")
            return set()

        # Step 1.5: 过滤ST股票
        if "name" in stock_pool_df.columns:
            st_pattern = r"(?:\s*(?:\*|★|※|•|·))?(?:[Ss][Tt])"
            before_count = len(stock_pool_df)
            stock_pool_df = stock_pool_df[~stock_pool_df["name"].astype(str).str.contains(st_pattern, na=False)].copy()
            after_count = len(stock_pool_df)
            filtered_count = before_count - after_count
            if filtered_count > 0:
                print(f"[FILTER] 已过滤 {filtered_count} 只ST股票，剩余 {after_count} 只正常股票。")
            else:
                print("[INFO] 无ST股票需要过滤。")

        pure_codes = set(stock_pool_df["股票代码"].unique().tolist())
        print(f"[INFO] 从数据库获取 {len(pure_codes)} 只股票（已剔除ST）。")

        # Step 2: 获取研报数据（作为特征，不过滤）
        report_df = self._get_research_report_data()

        # 将研报数据保存到processed_data，供后续合并使用
        if not report_df.empty:
            # 保存到缓存文件，供 DataProcessingService 读取
            report_cache_path = self.cache_manager.get_cache_path("研报买入次数", cleaned=True)
            try:
                report_df.to_csv(report_cache_path, sep="|", index=False, encoding="utf-8-sig")
                print(f"[INFO] 研报数据已保存至缓存: {os.path.basename(report_cache_path)}")
            except Exception as e:
                print(f"[ERROR] 保存研报数据失败: {e}")

        #  Step 3: 根据配置决定是否只保留主板股票
        if self.config.MAIN_BOARD_ONLY:
            final_codes = {code for code in pure_codes if code.startswith(("60", "00"))}
            print(f"[INFO] 已开启主板过滤，分析池包含 {len(final_codes)} 只主板股票。")
        else:
            final_codes = pure_codes
            print(f"[INFO] 全市场模式，分析池包含 {len(final_codes)} 只股票。")

        # 【新增】Step 3.5: 如果启用了研报过滤，则进行二次过滤
        if self.config.ENABLE_RESEARCH_REPORT_FILTER and not report_df.empty:
            print(f"\n[研报过滤] 启用研报二次过滤，阈值: {self.config.RESEARCH_REPORT_MIN_COUNT} 次买入评级")

            # 筛选出研报买入次数大于阈值的股票
            filtered_report_df = report_df[
                report_df["机构投资评级(近六个月)-买入"] > self.config.RESEARCH_REPORT_MIN_COUNT
            ]
            report_filtered_codes = set(filtered_report_df["股票代码"].unique().tolist())

            # 与final_codes取交集
            before_count = len(final_codes)
            final_codes = final_codes.intersection(report_filtered_codes)
            after_count = len(final_codes)
            filtered_count = before_count - after_count

            print(f"[研报过滤] 原始股票数: {before_count}, 研报过滤后: {after_count}, 过滤掉: {filtered_count}")

            if after_count == 0:
                print("[警告] 研报过滤后无股票剩余，请检查阈值设置或研报数据")
                return set()
        elif self.config.ENABLE_RESEARCH_REPORT_FILTER:
            print("[警告] 研报过滤已启用，但未获取到研报数据，跳过研报过滤")

        # 【测试模式】如果环境变量设置了 TEST_MODE，只取前10只股票测试
        import os as _os

        if _os.getenv("TEST_MODE", "").lower() == "true":
            test_count = int(_os.getenv("TEST_COUNT", "10"))
            final_codes = sorted(list(final_codes))[:test_count]
            print(f"[测试模式] 仅获取前 {test_count} 只股票进行流程验证")

        #  Step 4: 缓存检查 → 决定是否重取

        if os.path.exists(self.kline_cache_path):
            print(f" 缓存文件存在: {os.path.basename(self.kline_cache_path)}")
            try:
                df = pd.read_csv(self.kline_cache_path, sep="|", encoding="utf-8-sig")
                # 确保字段存在
                expected_cols = [
                    "trade_date",
                    "symbol",
                    "open",
                    "close",
                    "high",
                    "low",
                    "amount",
                    "close_normal",
                    "volume",
                    "adj_ratio",
                ]
                missing = [c for c in expected_cols if c not in df.columns]
                if missing:
                    print(f"[WARN] 缓存缺少列: {missing}，将重取")
                    df = pd.DataFrame()
                else:
                    # 将 symbol 明确转换为 sh/sz/bj 格式（可选）
                    df["symbol"] = df["symbol"].astype(str)
                    print(f" 成功加载缓存，共 {len(df)} 条记录。")

                    try:
                        df.to_sql(
                            name="stock_daily_kline",
                            con=self.db,
                            if_exists="replace",
                            index=False,
                            method="multi",
                            chunksize=5000,
                        )
                        print(f"成功将 {len(df)} 条记录写入 'stock_daily_kline' 表。")
                    except Exception as e:
                        print(f"[ERROR] 写入数据库失败: {e}")
                        raise

                    final_output_path = os.path.join(self.base_data_dir, f"final_filtered_stocks_{self.today}.txt")
                    try:
                        with open(final_output_path, "w", encoding="utf-8") as f:
                            for code in sorted(final_codes):
                                f.write(f"{code}\n")
                        print(f"[INFO] 最终筛选代码已保存至: {final_output_path}")
                    except Exception as e:
                        print(f"[ERROR] 保存最终代码列表失败: {e}")

                    print(f"  - 今日日期: {self.today}")
                    print(f"  - 筛选股票数: {len(final_codes)}")

                    return final_codes

            except Exception as e:
                print(f"[WARN] 缓存加载失败: {e}")

        filtered_codes = final_codes  # ←  这才是真正的"最终要处理的股票"
        print(f"[INFO]  将获取 {len(filtered_codes)} 只股票的 K 线（基于交集结果）。")

        #  Step 6: 获取最终 Akshare 格式代码（使用统一的格式化函数）
        from DataManager.ShareCodeFormatMgr import format_stock_code

        akshare_symbols = [format_stock_code(code) for code in filtered_codes]

        #  Step 7: 多线程并发获取 K 线数据（带断点续传 + 增量更新 + 进度条）
        from tqdm import tqdm

        print(f"[INFO] 正在获取 {len(akshare_symbols)} 只股票的 K 线数据（多线程并发模式）...")

        # 7.0 检查是否有今天的批次文件（判断是否需要全量获取）
        kline_cache_dir = os.path.join(self.base_data_dir, "kline_batches")
        os.makedirs(kline_cache_dir, exist_ok=True)
        has_today_batches = self._has_today_batch_files(kline_cache_dir)

        # 7.1 检查今日已成功获取的股票（增量更新）
        success_symbols = self._load_success_symbols()

        # 如果没有今天的批次文件，说明是历史数据，需要清空成功列表，从头开始
        if success_symbols and not has_today_batches:
            print("[警告] 检测到成功列表，但未找到今天的批次文件（可能是历史数据）")
            print("[清理] 清空成功列表和失败列表，将从头开始全量获取...")
            self._clear_success_symbols()
            self._clear_failed_symbols()
            success_symbols = set()

        if success_symbols:
            original_count = len(akshare_symbols)
            akshare_symbols = [s for s in akshare_symbols if s not in success_symbols]
            skipped_count = original_count - len(akshare_symbols)
            print(f"[增量更新] 跳过 {skipped_count} 只已成功的股票，本次需获取 {len(akshare_symbols)} 只")

            if not akshare_symbols:
                print("[INFO] 今日所有股票 K 线数据已成功获取，无需重复调用接口！")
                self._load_and_merge_cached_data(kline_cache_dir)
                return

        # 7.2 检查是否有上次失败的任务需要重试
        failed_symbols = self._load_failed_symbols()
        if failed_symbols:
            print(f"[断点续传] 发现上次失败的 {len(failed_symbols)} 只股票，优先重试...")
            akshare_symbols = failed_symbols + [s for s in akshare_symbols if s not in failed_symbols]

        # 分批并发获取（每500只股票为一批，每批内多线程并发）
        batch_size = 500
        max_workers = min(len(akshare_symbols), 15)

        total_new_batches = (len(akshare_symbols) + batch_size - 1) // batch_size
        all_success_dfs = []
        all_failed_symbols = []
        total_success = 0
        total_failed = 0

        for batch_idx in range(total_new_batches):
            last_trading_day = TradingCalendarAnalyzer().get_last_trading_day()
            trading_date_str = last_trading_day.replace("-", "")
            current_time_str = datetime.now().strftime("%H%M%S")
            batch_timestamp = f"{trading_date_str}{current_time_str}"

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(akshare_symbols))
            batch_symbols = akshare_symbols[start_idx:end_idx]

            batch_success_dfs = []
            batch_failed = []

            batch_desc = f"批次{batch_idx + 1}/{total_new_batches}"
            exec_hist = self.executor or ThreadPoolExecutor(max_workers=max_workers)
            try:
                with tqdm(total=len(batch_symbols), desc=batch_desc, unit="只", ncols=80, leave=False) as pbar:
                    futures = {exec_hist.submit(self._fetch_kline_for_symbol, s): s for s in batch_symbols}
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            result = future.result()
                            if result is not None and not result.empty:
                                batch_success_dfs.append(result)
                                total_success += 1
                            else:
                                batch_failed.append(symbol)
                                total_failed += 1
                        except (TimeoutError, ConnectionError, ValueError, TypeError):
                            batch_failed.append(symbol)
                            total_failed += 1
                        pbar.update(1)
                        pbar.set_postfix_str(f"成{total_success} 败{total_failed}")
            finally:
                if self.executor is None:
                    exec_hist.shutdown(wait=True)

                # 7.3 保存本批成功的数据到本地缓存
                if batch_success_dfs:
                    batch_df = pd.concat(batch_success_dfs, ignore_index=True)
                    batch_file = os.path.join(kline_cache_dir, f"kline_batch_{batch_timestamp}.csv")
                    batch_df.to_csv(batch_file, sep="|", index=False, encoding="utf-8-sig")
                    all_success_dfs.append(batch_df)
                    batch_success_codes = [df["symbol"].iloc[0] for df in batch_success_dfs if not df.empty]
                    self._save_success_symbols(batch_success_codes, append=True)

            all_failed_symbols.extend(batch_failed)

            # 批次间休息10秒
            if batch_idx < total_new_batches - 1:
                time.sleep(10)

        # 7.4 统计输出
        if all_failed_symbols:
            self._save_failed_symbols(all_failed_symbols)
            print(f"\n[统计] 总{len(akshare_symbols)}只 | 成功{total_success} | 失败{total_failed}")
        else:
            self._clear_failed_symbols()
            self._clear_success_symbols()
            self._cleanup_old_batch_files(kline_cache_dir)
            print(f"\n[统计] 总{len(akshare_symbols)}只 | 全部成功 ✓")

        # 7.5 合并所有批次的数据
        if not all_success_dfs:
            print("[WARNING] 所有股票 K 线获取失败，数据库将清空。")
            self._clear_stock_daily_kline_table()
            return

        combined_kline_df = pd.concat(all_success_dfs, ignore_index=True)
        print(f"[INFO] 成功合并 {len(combined_kline_df)} 条 K 线记录。")
        try:
            # 保存为 CSV，不包含 index
            combined_kline_df.to_csv(self.kline_cache_path, sep="|", index=False, encoding="utf-8-sig")
            print(f"[INFO]  K线数据已保存至本地缓存: {os.path.basename(self.kline_cache_path)}")
        except Exception as e:
            print(f"[ERROR] 保存 K线数据缓存失败: {e}")

        #  Step 8: 获取筹码分布数据（仅当启用时）
        if self.config.ENABLE_CHIP_DISTRIBUTION:
            from DataCollection.ChipDistributionFetcher import ChipDistributionFetcher

            chip_fetcher = ChipDistributionFetcher(self.config)
            chip_df = chip_fetcher.fetch_chip_data()
            if not chip_df.empty:
                print(f"[ChipDist] 筹码分布数据可用 {len(chip_df)} 条")
            else:
                print("[ChipDist] 未获取到筹码分布数据。")
        else:
            print("[ChipDist] 筹码分布获取未启用（config.ini 中 enable_chip_distribution = false）")

        #  Step 9: 写入数据库
        try:
            combined_kline_df.to_sql(
                name="stock_daily_kline",
                con=self.db,
                if_exists="replace",  # 替换已有表
                index=False,
                method="multi",
                chunksize=5000,
            )
            print(f"[INFO]  成功将 {len(combined_kline_df)} 条记录写入 'stock_daily_kline' 表。")
        except Exception as e:
            print(f"[ERROR] 写入数据库失败: {e}")
            raise

        print(f"  - 今日日期: {self.today}")
        print(f"  - 筛选股票数: {len(filtered_codes)}")

        #  可选：保存最终过滤列表
        final_output_path = os.path.join(self.base_data_dir, f"final_filtered_stocks_{self.today}.txt")
        try:
            with open(final_output_path, "w", encoding="utf-8") as f:
                for code in sorted(filtered_codes):
                    f.write(f"{code}\n")
            print(f"[INFO] 最终筛选代码已保存至: {final_output_path}")
        except Exception as e:
            print(f"[ERROR] 保存最终代码列表失败: {e}")

        return filtered_codes

