from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine


class FinancialQualityFetcher:
    """质量/成长因子采集器 — akShare（季度，缓存 90 天 + 文件缓存）。

    通过 akShare ``stock_financial_abstract`` 接口逐只获取财务摘要。
    文件缓存以交易日后缀命名，交易日匹配时直接读缓存跳过 API。
    """

    TABLE_NAME = "ods_financial_quality"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache_days = config.FINANCIAL_QUALITY_CACHE_DAYS
        self._engine = get_engine(config)

    def is_stale(self, symbol: str) -> bool:
        """检查指定股票的财务数据是否已过期。"""
        sql = (
            f"SELECT MAX(record_date) FROM {self.TABLE_NAME} "
            "WHERE symbol = :sym"
        )
        with self._engine.connect() as conn:
            result = conn.execute(
                __import__("sqlalchemy").text(sql), {"sym": symbol}
            ).scalar()
        if result is None:
            return True
        return (datetime.now().date() - result).days > self._cache_days

    def fetch_one(self, symbol: str) -> dict[str, Any] | None:
        """调用 akShare 获取单只股票的财务摘要，提取质量/成长指标。"""
        import akshare as ak

        try:
            df = ak.stock_financial_abstract(symbol=symbol)
        except Exception as e:
            logger.warning(f"[FinancialQuality] {symbol} akShare 调用失败: {e}")
            return None

        if df is None or df.empty:
            return None

        # stock_financial_abstract 返回格式：80 行指标 × N 列报告期
        # 列是日期字符串（如 "2023-12-31"），行是指标名
        # 取最近一个报告期
        date_cols = [c for c in df.columns if isinstance(c, str) and c[0].isdigit()]
        if not date_cols:
            return None
        latest_date = sorted(date_cols, reverse=True)[0]

        # 构建指标名 → 行索引映射
        # akShare 返回格式：col[0]="选项"（分类名）、col[1]="指标"（指标名）、col[2:]=报告日期
        name_to_idx = {}
        for idx, row in df.iterrows():
            raw = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            if raw:
                name_to_idx[raw] = idx

        def _find(prefix: str) -> int | None:
            for name, idx in name_to_idx.items():
                if name.startswith(prefix):
                    return idx
            return None

        result = {
            "symbol": symbol,
            "record_date": latest_date,
            "roe": self._safe_float(df.at[_find("净资产收益率"), latest_date]) if _find("净资产收益率") is not None else None,
            "gross_profit_margin": self._safe_float(df.at[_find("毛利率"), latest_date]) if _find("毛利率") is not None else None,
            "net_profit_margin": self._safe_float(df.at[_find("销售净利率"), latest_date]) if _find("销售净利率") is not None else None,
            "revenue_growth_rate": self._safe_float(df.at[_find("营业总收入增长率"), latest_date]) if _find("营业总收入增长率") is not None else None,
            "net_profit_growth_rate": self._safe_float(df.at[_find("归属母公司净利润增长率"), latest_date]) if _find("归属母公司净利润增长率") is not None else None,
        }

        return result

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        if val is None or (isinstance(val, float) and val != val):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def sync_one(self, symbol: str) -> bool:
        """采集单只股票并 UPSERT 到数据库。"""
        row = self.fetch_one(symbol)
        if row is None:
            return False

        from sqlalchemy import text

        sql = text(
            f"INSERT INTO {self.TABLE_NAME} "
            "(symbol, record_date, roe, gross_profit_margin, net_profit_margin, "
            "revenue_growth_rate, net_profit_growth_rate) "
            "VALUES (:symbol, :record_date, :roe, :gross_profit_margin, :net_profit_margin, "
            ":revenue_growth_rate, :net_profit_growth_rate) "
            "ON CONFLICT (symbol, record_date) DO UPDATE SET "
            "roe = EXCLUDED.roe, "
            "gross_profit_margin = EXCLUDED.gross_profit_margin, "
            "net_profit_margin = EXCLUDED.net_profit_margin, "
            "revenue_growth_rate = EXCLUDED.revenue_growth_rate, "
            "net_profit_growth_rate = EXCLUDED.net_profit_growth_rate"
        )
        with self._engine.begin() as conn:
            conn.execute(sql, row)
        return True

    # ── 文件缓存 ──────────────────────────────────────────────────

    @property
    def _cache_dir(self) -> str:
        return getattr(self.config, "CACHE_DIRECTORY", "cache")

    def _cache_path(self, today_str: str) -> str:
        key = today_str.replace("-", "")
        return os.path.join(self._cache_dir, f"financial_quality_{key}.parquet")

    def _load_cache(self, today_str: str) -> pd.DataFrame | None:
        path = self._cache_path(today_str)
        if not os.path.isfile(path):
            return None
        try:
            df = pd.read_parquet(path)
            logger.info(f"[FinancialQuality] 读取缓存 {path}，共 {len(df)} 只")
            return df
        except Exception as e:
            logger.warning(f"[FinancialQuality] 缓存读取失败: {e}")
            return None

    def _save_cache(self, today_str: str, rows: list[dict[str, Any]]) -> None:
        path = self._cache_path(today_str)
        os.makedirs(self._cache_dir, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        logger.info(f"[FinancialQuality] 缓存已保存 {path}，共 {len(df)} 只")

    def _bulk_upsert(self, rows: list[dict[str, Any]]) -> int:
        """批量 UPSERT 到数据库。"""
        from sqlalchemy import text

        sql = text(
            f"INSERT INTO {self.TABLE_NAME} "
            "(symbol, record_date, roe, gross_profit_margin, net_profit_margin, "
            "revenue_growth_rate, net_profit_growth_rate) "
            "VALUES (:symbol, :record_date, :roe, :gross_profit_margin, :net_profit_margin, "
            ":revenue_growth_rate, :net_profit_growth_rate) "
            "ON CONFLICT (symbol, record_date) DO UPDATE SET "
            "roe = EXCLUDED.roe, "
            "gross_profit_margin = EXCLUDED.gross_profit_margin, "
            "net_profit_margin = EXCLUDED.net_profit_margin, "
            "revenue_growth_rate = EXCLUDED.revenue_growth_rate, "
            "net_profit_growth_rate = EXCLUDED.net_profit_growth_rate"
        )
        with self._engine.begin() as conn:
            for row in rows:
                conn.execute(sql, row)
        return len(rows)

    def sync(self, stock_list: list[str], today_str: str | None = None,
             max_workers: int = 2) -> int:
        """并发采集全股池，跳过缓存未过期的，返回采集数量。

        若提供 ``today_str`` 且对应缓存文件存在，直接读缓存入库，跳过 API。
        ``max_workers`` 控制并发管道数，默认 2 路。
        先全部采集到内存 → 保存本地缓存 → 最后一次性批量写入 DB。
        """
        # ── 文件缓存命中 ──────────────────────────────────────
        if today_str:
            cached = self._load_cache(today_str)
            if cached is not None and not cached.empty:
                rows = cached.to_dict("records")
                written = self._bulk_upsert(rows)
                logger.info(f"[FinancialQuality] 缓存命中，写入 {written} 只到数据库，跳过 API")
                return written

        # ── 判断哪些需要采集 ──────────────────────────────────
        stale = [s for s in stock_list if self.is_stale(s)]
        if not stale:
            logger.info("[FinancialQuality] 无需采集，全部未过期")
            return 0

        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

        all_rows: list[dict[str, Any]] = []
        total = len(stale)
        done = 0

        logger.info(f"[FinancialQuality] 开始并发采集 {total} 只（{max_workers} 路）...")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_to_sym = {pool.submit(self.fetch_one, sym): sym for sym in stale}
            while fut_to_sym:
                done_set, not_done = wait(fut_to_sym, timeout=120, return_when=FIRST_COMPLETED)
                # ── 超时：120s 内无任何完成 → 跳过剩余全部 ──
                if not done_set and not_done:
                    for fut in not_done:
                        sym = fut_to_sym.get(fut, "?")
                        logger.warning(f"[FinancialQuality] {sym} 采集超时（120s），跳过")
                        fut.cancel()
                        done += 1
                    break
                for fut in done_set:
                    sym = fut_to_sym.pop(fut, "?")
                    done += 1
                    try:
                        row = fut.result(timeout=5)
                    except Exception:
                        logger.warning(f"[FinancialQuality] {sym} 采集失败，跳过")
                        continue
                    if row is None:
                        continue
                    all_rows.append(row)
                    if done % 50 == 0 or done == total:
                        logger.info(f"[FinancialQuality] 进度 {done}/{total}，已采集 {len(all_rows)} 只")

        skipped = total - len(all_rows)
        # ── 保存缓存文件 ──────────────────────────────────────
        if today_str and all_rows:
            self._save_cache(today_str, all_rows)

        # ── 一次性批量写入 DB ──────────────────────────────────
        if all_rows:
            self._bulk_upsert(all_rows)

        logger.info(f"[FinancialQuality] 完成，采集 {len(all_rows)}/{total} 只（跳过 {skipped}）")
        return len(all_rows)

    def load_quality(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """从数据库加载最近一期的质量因子数据。"""
        from sqlalchemy import text

        if symbols:
            placeholders = ", ".join(f":s{i}" for i in range(len(symbols)))
            params = {f"s{i}": s for i, s in enumerate(symbols)}
            sql = text(
                f"SELECT DISTINCT ON (symbol) symbol, record_date, "
                "roe, gross_profit_margin, net_profit_margin, "
                "revenue_growth_rate, net_profit_growth_rate "
                f"FROM {self.TABLE_NAME} "
                f"WHERE symbol IN ({placeholders}) "
                "ORDER BY symbol, record_date DESC"
            )
        else:
            sql = text(
                f"SELECT DISTINCT ON (symbol) symbol, record_date, "
                "roe, gross_profit_margin, net_profit_margin, "
                "revenue_growth_rate, net_profit_growth_rate "
                f"FROM {self.TABLE_NAME} "
                "ORDER BY symbol, record_date DESC"
            )
            params = {}

        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
