from __future__ import annotations

import os
import sys
import time
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
    采集按批进行（默认每批 500 只），批间休眠（默认 20 秒）以避免接口封禁。
    """

    TABLE_NAME = "ods_financial_quality"

    # 各报告期监管披露截止日（公告日最晚可能值，用于保守 as-of 回填，无前视）
    _DISCLOSURE_DEADLINES = {
        "03": "04-30", "06": "08-31", "09": "10-31", "12": "04-30",
    }

    @classmethod
    def _disclosure_deadline(cls, record_date: str) -> str:
        """按监管披露截止日推导披露日：公告日 <= 截止日恒成立，用截止日近似最保守（无前视）。

        年报（12-31）截止次年 4-30；一季报 4-30；中报 8-31；三季报 10-31。
        """
        y, m, _ = (int(x) for x in str(record_date).split("-"))
        md = cls._DISCLOSURE_DEADLINES.get(f"{m:02d}")
        if md is None:
            return str(record_date)
        year = y + 1 if m == 12 else y
        return f"{year}-{md}"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache_days = config.FINANCIAL_QUALITY_CACHE_DAYS
        self._batch_size = getattr(config, "FINANCIAL_QUALITY_BATCH_SIZE", 500)
        self._batch_sleep = getattr(config, "FINANCIAL_QUALITY_BATCH_SLEEP", 20)
        self._file_cache_days = getattr(config, "FINANCIAL_QUALITY_FILE_CACHE_DAYS", 30)
        self._engine = get_engine(config)


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
            "disclosure_date": self._disclosure_deadline(latest_date),
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


    # ── 文件缓存 ──────────────────────────────────────────────────

    @property
    def _cache_dir(self) -> str:
        return getattr(self.config, "CACHE_DIRECTORY", "cache")

    def _cache_path(self, today_str: str) -> str:
        key = today_str.replace("-", "")
        return os.path.join(self._cache_dir, f"financial_quality_{key}.parquet")

    def _find_recent_cache(self) -> pd.DataFrame | None:
        """查找最近 ``_file_cache_days`` 天内的缓存文件，命中则直接读取。

        财报非日更，只要离线文件在一个月内即可直接复用，避免频繁请求被封禁。
        """
        if not os.path.isdir(self._cache_dir):
            return None
        cutoff = time.time() - self._file_cache_days * 86400
        best_path: str | None = None
        best_mtime = 0.0
        for name in os.listdir(self._cache_dir):
            if not name.startswith("financial_quality_") or not name.endswith(".parquet"):
                continue
            path = os.path.join(self._cache_dir, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime >= cutoff and mtime > best_mtime:
                best_path, best_mtime = path, mtime
        if best_path is None:
            return None
        try:
            df = pd.read_parquet(best_path)
            logger.info(f"[FinancialQuality] 命中离线缓存 {best_path}（{self._file_cache_days} 天内），共 {len(df)} 只")
            return df
        except Exception as e:
            logger.warning(f"[FinancialQuality] 缓存读取失败 {best_path}: {e}")
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
            "(symbol, record_date, disclosure_date, roe, gross_profit_margin, net_profit_margin, "
            "revenue_growth_rate, net_profit_growth_rate) "
            "VALUES (:symbol, :record_date, :disclosure_date, :roe, :gross_profit_margin, :net_profit_margin, "
            ":revenue_growth_rate, :net_profit_growth_rate) "
            "ON CONFLICT (symbol, record_date) DO UPDATE SET "
            "disclosure_date = EXCLUDED.disclosure_date, "
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
        """并发采集全股池，增量补全，返回此次采集数量。

        流程：
        1. 合并已有数据：离线缓存（一个月内文件）∪ 数据库已有记录
        2. 仅采集 stock_list 中缺失的股票
        3. 按批采集（默认每批 500 只），批间休眠（默认 20 秒）避免接口封禁
        4. 新采集的追加到缓存文件（不覆盖）
        5. 批量写入 DB
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        from sqlalchemy import text

        target_set = set(stock_list)
        if not target_set:
            return 0

        already: set[str] = set()

        # ── 1a. 加载一个月内离线缓存（财报非日更，直接复用） ──
        cached_df: pd.DataFrame | None = self._find_recent_cache()
        if cached_df is not None and not cached_df.empty:
            cached_syms = set(cached_df["symbol"].unique())
            already |= cached_syms
            logger.info(f"[FinancialQuality] 离线缓存已有 {len(cached_syms)} 只")

        # ── 1b. 查询数据库最新 record_date，仅未过期的才跳过 ──
        cutoff = (datetime.now().date() - timedelta(days=self._cache_days))
        fresh_db_syms: set[str] = set()
        stale_db_count = 0
        try:
            placeholders = ", ".join(f":s{i}" for i in range(len(stock_list)))
            params = {f"s{i}": s for i, s in enumerate(stock_list)}
            sql = text(
                f"SELECT symbol, MAX(record_date) AS max_date FROM {self.TABLE_NAME} "
                f"WHERE symbol IN ({placeholders}) "
                "GROUP BY symbol"
            )
            with self._engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            for sym, max_date in rows:
                if max_date and max_date >= cutoff:
                    fresh_db_syms.add(sym)
                else:
                    stale_db_count += 1
            already |= fresh_db_syms
            logger.info(f"[FinancialQuality] 数据库最新数据未过期 {len(fresh_db_syms)} 只，过期需重采 {stale_db_count} 只")
        except Exception as e:
            logger.warning(f"[FinancialQuality] 查询数据库失败: {e}")

        # ── 2. 筛选需采集的股票 ──
        need = sorted(target_set - already)
        if not need:
            logger.info(f"[FinancialQuality] 全部 {len(target_set)} 只已采集，跳过")
            return 0

        logger.info(f"[FinancialQuality] 需采集 {len(need)}/{len(target_set)} 只（每批 {self._batch_size} 只，{max_workers} 路并发）...")

        all_rows: list[dict[str, Any]] = []
        total = len(need)
        done = 0
        failed = 0

        # ── 3. 按批采集，批间休眠避免接口封禁 ──
        for batch_start in range(0, total, self._batch_size):
            batch = need[batch_start:batch_start + self._batch_size]
            if batch_start > 0:
                logger.info(f"[FinancialQuality] 批间休眠 {self._batch_sleep} 秒，避免接口封禁...")
                time.sleep(self._batch_sleep)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_to_sym = {pool.submit(self.fetch_one, sym): sym for sym in batch}
                while fut_to_sym:
                    done_set, not_done = wait(fut_to_sym, timeout=120, return_when=FIRST_COMPLETED)
                    if not done_set and not_done:
                        for fut in not_done:
                            sym = fut_to_sym.get(fut, "?")
                            logger.warning(f"[FinancialQuality] {sym} 采集超时（120s），跳过")
                            fut.cancel()
                            done += 1
                            failed += 1
                        break
                    for fut in done_set:
                        sym = fut_to_sym.pop(fut, "?")
                        done += 1
                        try:
                            row = fut.result(timeout=5)
                        except Exception:
                            logger.warning(f"[FinancialQuality] {sym} 采集失败，跳过")
                            failed += 1
                            continue
                        if row is None:
                            failed += 1
                            continue
                        all_rows.append(row)
                        if done % 50 == 0 or done == total:
                            logger.info(f"[FinancialQuality] 进度 {done}/{total}，已采集 {len(all_rows)} 只，失败 {failed} 只")

        # ── 追加到缓存文件（不覆盖已有） ──
        if today_str and all_rows:
            new_df = pd.DataFrame(all_rows)
            if cached_df is not None and not cached_df.empty:
                combined = pd.concat([cached_df, new_df], ignore_index=True)
            else:
                combined = new_df
            self._save_cache(today_str, combined)

        # ── 批量写入 DB ──
        if all_rows:
            self._bulk_upsert(all_rows)

        logger.info(f"[FinancialQuality] 完成，此次采集 {len(all_rows)}/{total} 只，累计 {len(already | {r['symbol'] for r in all_rows})} 只")
        return len(all_rows)

    def load_quality(self, symbols: list[str] | None = None,
                     as_of: str | None = None) -> pd.DataFrame:
        """从数据库加载质量因子数据（PIT as-of 语义）。

        只取查询日 as_of 之前已披露的财报（disclosure_date <= as_of），
        每只股票取最新一个报告期 —— 避免历史复盘用到尚未披露的财报（前视）。
        旧数据无 disclosure_date 时回退为 record_date <= as_of 过滤。

        Args:
            symbols: 股票代码列表，None 表示全市场。
            as_of: 查询日 "YYYYMMDD" 或 "YYYY-MM-DD"，默认今天（当日数据可得性）。
        """
        from sqlalchemy import text

        if as_of:
            as_of_norm = str(as_of).replace("-", "")
            as_of_date = (f"{as_of_norm[:4]}-{as_of_norm[4:6]}-{as_of_norm[6:8]}")
        else:
            as_of_date = datetime.now().date().strftime("%Y-%m-%d")

        as_of_clause = (
            "((disclosure_date IS NOT NULL AND disclosure_date <= :as_of) "
            "OR (disclosure_date IS NULL AND record_date <= :as_of))"
        )

        if symbols:
            placeholders = ", ".join(f":s{i}" for i in range(len(symbols)))
            params: dict[str, Any] = {f"s{i}": s for i, s in enumerate(symbols)}
            params["as_of"] = as_of_date
            sql = text(
                f"SELECT DISTINCT ON (symbol) symbol, record_date, disclosure_date, "
                "roe, gross_profit_margin, net_profit_margin, "
                "revenue_growth_rate, net_profit_growth_rate "
                f"FROM {self.TABLE_NAME} "
                f"WHERE symbol IN ({placeholders}) AND {as_of_clause} "
                "ORDER BY symbol, record_date DESC"
            )
        else:
            sql = text(
                f"SELECT DISTINCT ON (symbol) symbol, record_date, disclosure_date, "
                "roe, gross_profit_margin, net_profit_margin, "
                "revenue_growth_rate, net_profit_growth_rate "
                f"FROM {self.TABLE_NAME} "
                f"WHERE {as_of_clause} "
                "ORDER BY symbol, record_date DESC"
            )
            params = {"as_of": as_of_date}

        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
