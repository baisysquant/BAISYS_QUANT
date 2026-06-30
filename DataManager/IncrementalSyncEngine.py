from __future__ import annotations

import os
import random
import time
from datetime import date, datetime, timedelta
from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger
from sqlalchemy import text
from tqdm import tqdm

from DataCollection.CalendarManager import TradingCalendarAnalyzer
from UtilsManager.AkshareConfig import ensure_akshare_timeout

ensure_akshare_timeout()


TABLE = "stock_daily_kline"
OVERLAP_DAYS = 15
BATCH_SIZE = 300          # 每批次处理 300 只
BATCH_INTERVAL = 10       # 批次间休息 10 秒
STAGGER_DELAY = 15        # 两管道错峰 15 秒
FETCH_BATCH = 10          # 每次子进程处理 10 只，降低 V8 踩线崩溃概率


def _fetch_kline_batch(symbols: list[str], start: str, end: str) -> list[dict]:
    """一批股票一次 subprocess 调用，分摊启动开销。"""
    import subprocess, json, os, sys
    script = os.path.join(os.path.dirname(__file__), "_fetch_one.py")
    payload = json.dumps({"symbols": symbols, "start": start, "end": end})
    try:
        r = subprocess.run(
            [sys.executable, script, payload],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            logger.warning(f"子进程退出码 {r.returncode}, stderr={r.stderr[:500]}")
            return []
        if not r.stdout.strip():
            logger.warning(f"子进程无输出, stderr={r.stderr[:500]}")
            return []
        return json.loads(r.stdout)
    except Exception as e:
        logger.warning(f"_fetch_kline_batch 异常: {e}")
        return []


def _sina_daily_worker(symbol: str, start: str, end: str, adjust: str = "") -> pd.DataFrame | None:
    """子进程 worker：获取日 K 线（腾讯 API，无 V8）。"""
    try:
        import akshare as ak  # noqa: F811
        import time, random
        time.sleep(random.uniform(0.05, 0.2))
        df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust=adjust)
        if df is None or df.empty:
            return None
        df["date"] = df["date"].astype(str)
        return df
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"  [{symbol}] _sina_daily_worker 异常: {e}")
        return None


class IncrementalSyncEngine:
    def __init__(
        self,
        db_engine: Any,
        default_start: str | None = None,
        cache_dir: str | None = None,
        main_board_only: bool = False,
        enable_research_report_filter: bool = False,
        research_report_min_count: int = 1,
    ) -> None:
        self._engine = db_engine
        self._default_start = self.align_to_trading_day(default_start) if default_start else None
        self._main_board_only = main_board_only
        self._enable_research_report_filter = enable_research_report_filter
        self._research_report_min_count = research_report_min_count
        self._cache_dir = cache_dir
        if not self._cache_dir:
            try:
                from ConfigParser import Config
                self._cache_dir = os.path.join(Config().CACHE_DIRECTORY, "kline_batches")
            except Exception:
                self._cache_dir = os.path.join(
                    os.environ.get("TEMP", "/tmp"), "opencode", "kline_batches"
                )
        os.makedirs(self._cache_dir, exist_ok=True)
        # 使用最新交易日而非 date.today()，避免周末/节假日误判
        try:
            _cal = TradingCalendarAnalyzer()
            _today_str = _cal.get_last_trading_day()
            self._trade_date = datetime.strptime(_today_str, "%Y-%m-%d").date()
        except Exception:
            self._trade_date = date.today()
        self._trade_date_str = self._trade_date.isoformat().replace("-", "")
        self._cleanup_old_cache()

    # ── public API ──────────────────────────────────────────────

    def _cleanup_old_cache(self) -> None:
        """清理超过 7 天的缓存文件，以及脏 close_normal_*.csv（日期不匹配当前交易日）。"""
        try:
            now = datetime.now()
            today_tag = f"close_normal_{self._trade_date_str}.csv"
            for fname in os.listdir(self._cache_dir):
                # 清理脏 close_normal 缓存（日期与当前交易日不一致）
                if fname.startswith("close_normal_") and fname != today_tag:
                    os.remove(os.path.join(self._cache_dir, fname))
                    continue
                # 清理超过 7 天的旧缓存
                fpath = os.path.join(self._cache_dir, fname)
                if os.path.isfile(fpath):
                    age = now - datetime.fromtimestamp(os.path.getmtime(fpath))
                    if age.days > 7:
                        os.remove(fpath)
        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")

    def sync_all(self, symbols_prefixed: list[str]) -> int:
        try:
            remaining = self._get_stale_symbols(symbols_prefixed)

            cached = self._load_failed_set()
            if cached:
                old_len = len(remaining)
                remaining = sorted(set(remaining) | cached)
                added = len(remaining) - old_len
                if added:
                    logger.info(f"加载 {len(cached)} 只缓存失败股票，待同步 {len(remaining)} 只（新增 {added} 只）")

            if not remaining:
                logger.info("所有股票已有最新交易日数据，无需同步")
                return 0

            start_iso = self._calc_start_iso(remaining)
            end_iso = self._trade_date.isoformat()
            logger.info(f"同步 {len(remaining)} 只, {start_iso} ~ {end_iso}")

            mid = len(remaining) // 2
            half_a = remaining[:mid]
            half_b = remaining[mid:]
            logger.info(f"双管道: 前半段 {len(half_a)} 只 | 后半段 {len(half_b)} 只")

            import threading
            results: dict = {"a": [0, []], "b": [0, []]}

            def run(label: str, half: list[str]):
                ins, fails = self._run_pipeline(half, start_iso, end_iso, label=label)
                results[label.lower()] = [ins, fails]

            ta = threading.Thread(target=run, args=("A", half_a), daemon=True)
            tb = threading.Thread(target=run, args=("B", half_b), daemon=True)

            ta.start()
            logger.info(f"管道 A 启动，{STAGGER_DELAY}s 后启动管道 B")
            time.sleep(STAGGER_DELAY)
            tb.start()

            ta.join()
            tb.join()

            all_failures = results["a"][1] + results["b"][1]
            total = results["a"][0] + results["b"][0]
            self._save_failed_set(set(all_failures))
            logger.info(f"同步完成，总写入 {total} 行（失败 {len(all_failures)} 只）")
            return total
        except BaseException as e:
            import traceback
            logger.error(f"sync_all 发生致命错误: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise

    def _calc_start_iso(self, symbols: list[str]) -> str:
        min_latest = self._get_min_latest_date(symbols)
        if min_latest is None:
            return (
                datetime.strptime(self._default_start, "%Y%m%d").strftime("%Y-%m-%d")
                if self._default_start else "2000-01-01"
            )
        return (min_latest - timedelta(days=OVERLAP_DAYS)).isoformat()

    # ── Dual Pipeline ───────────────────────────────────────────

    def _run_pipeline(self, symbols: list[str], start_iso: str, end_iso: str, label: str = "") -> tuple[int, list[str]]:
        start = start_iso.replace("-", "")
        end = end_iso.replace("-", "")
        total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
        inserted = 0
        all_failures: list[str] = []

        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_no = i // BATCH_SIZE + 1
            desc = f"  P{label} batch {batch_no}/{total_batches}"
            logger.info(f"管道{label} {desc}: {len(batch)} 只")
            count, failures = self._process_batch(batch, start, end, desc=desc)
            all_failures.extend(failures)
            inserted += count
            if i + BATCH_SIZE < len(symbols):
                time.sleep(BATCH_INTERVAL)

        logger.info(f"管道{label} 完成，写入 {inserted} 行（失败 {len(all_failures)} 只）")
        return inserted, all_failures

    def _process_batch(self, symbols: list[str], start: str, end: str, desc: str = "") -> tuple[pd.DataFrame | None, list[str]]:
        written_count = 0
        failed: list[str] = []
        with tqdm(total=len(symbols), desc=desc, unit="stk", leave=False) as pbar:
            for i in range(0, len(symbols), FETCH_BATCH):
                chunk = symbols[i:i + FETCH_BATCH]
                batch_results = _fetch_kline_batch(chunk, start, end)
                chunk_done: set[str] = set()
                for item in batch_results:
                    if item and item.get("symbol"):
                        df = pd.DataFrame(item)
                        if not df.empty:
                            self._write_with_split_detection(df)
                            written_count += df["symbol"].nunique()
                            chunk_done.update(df["symbol"].unique())
                for sym in chunk:
                    if sym not in chunk_done:
                        failed.append(sym)
                    pbar.update(1)
        logger.info(f"  {desc} 完成: 成功 {written_count} 只, 失败 {len(failed)} 只")
        return written_count, failed

    def _fetch_full_history(self, symbol: str) -> pd.DataFrame | None:
        """检测到除权除息时，从配置起始日到今日重新拉取全量数据。"""
        start = self._default_start.replace("-", "") if self._default_start else "20190101"
        end = self._trade_date.strftime("%Y%m%d")
        batch_results = _fetch_kline_batch([symbol], start, end)
        if not batch_results:
            return None
        for item in batch_results:
            if item and item.get("symbol"):
                return pd.DataFrame(item)
        return None

    def _write_with_split_detection(self, df: pd.DataFrame) -> int:
        """逐只股票：除权检测 + 幂等写入（无需前置 DELETE，(symbol, trade_date) 唯一约束自动处理）。"""
        total_inserted = 0
        to_write: list[pd.DataFrame] = []

        for sym, grp in df.groupby("symbol"):
            grp = grp.sort_values("trade_date")
            latest = self._get_latest_date(sym)

            if latest is None or self._has_no_close_normal(sym):
                to_write.append(grp)
                total_inserted += len(grp)
            elif self._detect_split_from_adj(sym, grp, latest):
                logger.info(f"  {sym} 除权除息，全量替换（拉取全量历史）")
                full = self._fetch_full_history(sym)
                if full is not None and not full.empty:
                    to_write.append(full)
                    total_inserted += len(full)
                else:
                    to_write.append(grp)
                    total_inserted += len(grp)
            else:
                new = grp[grp["trade_date"] > latest.isoformat()]
                if not new.empty:
                    to_write.append(new)
                    total_inserted += len(new)

        if not to_write:
            return 0

        final = pd.concat(to_write, ignore_index=True)
        self._write_batch(final)
        n_full = df["symbol"].nunique()
        logger.info(f"同步完成，写入 {total_inserted} 行（{n_full} 只）")
        return total_inserted

    def backfill_close_normal(self, symbols_prefixed: list[str] | None = None) -> pd.DataFrame:
        """双管道获取全市场最新不复权收盘价，写入本地缓存文件。

        通过 Tencent（stock_zh_a_hist_tx）双管道并行获取原始 close，
        写入 ``close_normal_{YYYYMMDD}.csv``（symbol, close 两列）。
        Args:
            symbols_prefixed: 股票代码列表（带前缀），None 表示从 DB 获取全部。
        Returns:
            DataFrame 包含 symbol, close 两列；失败返回空 DataFrame。
        """
        cache_file = self._close_normal_cache_path()
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"[close_normal] 从缓存加载 {len(df)} 只 → {os.path.basename(cache_file)}")
                return df
            except Exception as e:
                logger.warning(f"[close_normal] 缓存读取失败: {e}，重新拉取")

        if symbols_prefixed is None:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(f"SELECT DISTINCT symbol FROM {TABLE}")
                ).fetchall()
                symbols_prefixed = [r[0] for r in rows]

        if not symbols_prefixed:
            return pd.DataFrame()

        mid = len(symbols_prefixed) // 2
        half_a = symbols_prefixed[:mid]
        half_b = symbols_prefixed[mid:]
        end = self._trade_date_str
        results: dict[str, float] = {}

        def _fetch_one(sym: str) -> tuple[str, float | None]:
            try:
                with ProcessPoolExecutor(max_workers=1) as pool:
                    f = pool.submit(_sina_daily_worker, sym, end, end, "")
                    r = f.result(timeout=15)
                if r is not None and not r.empty:
                    close_val = pd.to_numeric(r["close"].iloc[-1], errors="coerce")
                    if pd.notna(close_val):
                        return sym, float(close_val)
            except Exception:
                pass
            return sym, None

        def _fetch_backfill_half(syms: list[str]) -> dict[str, float]:
            local: dict[str, float] = {}
            for sym in syms:
                sym_val = _fetch_one(sym)
                if sym_val[1] is not None:
                    local[sym_val[0]] = sym_val[1]
            return local

        results: dict[str, float] = {}
        for half in (half_a, half_b):
            results.update(_fetch_backfill_half(half))

        if not results:
            logger.warning("[close_normal] 双管道均无数据")
            return pd.DataFrame()

        df = pd.DataFrame(list(results.items()), columns=["symbol", "close"])
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_csv(cache_file, index=False)
        logger.info(f"[close_normal] 双管道获取完成: {len(df)} 只 → {os.path.basename(cache_file)}")
        return df

    # ── stock pool (merged from StockSyncEngine) ──────────────

    def get_stock_pool_from_db(self) -> pd.DataFrame:
        query = """
            SELECT stock_code AS ts_code, stock_code AS 股票代码,
                   stock_name AS name, industry_name AS industry
            FROM stock_basic_info_sw ORDER BY stock_code
        """
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if "股票代码" in df.columns:
            df["股票代码"] = df["股票代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
        for col in ("ts_code", "name", "industry", "股票代码"):
            if col not in df.columns:
                df[col] = "N/A"
        return df[["ts_code", "name", "industry", "股票代码"]]

    @staticmethod
    def filter_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
        if "name" not in df.columns:
            return df
        pattern = r"(?:\s*(?:\*|★|※|•|·))?(?:[Ss][Tt])|退市"
        return df[~df["name"].astype(str).str.contains(pattern, na=False)].copy()

    def filter_main_board(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._main_board_only:
            return df
        codes = df["股票代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
        return df[codes.str.match(r"^(60|00)", na=False)].copy()

    def _filter_by_research_report(self, pure_codes: set[str]) -> set[str]:
        if not self._enable_research_report_filter:
            return pure_codes
        try:
            for attempt in range(3):
                try:
                    raw = ak.stock_profit_forecast_em()
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        return pure_codes
            if raw is None or raw.empty:
                return pure_codes
            df = raw.copy()
            if "代码" in df.columns and "股票代码" not in df.columns:
                df.rename(columns={"代码": "股票代码"}, inplace=True)
            df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
            for col in df.columns:
                if "买入" in col:
                    rating_col = col
                    break
            else:
                return pure_codes
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0)
            qualified = set(df.loc[df[rating_col] > self._research_report_min_count, "股票代码"].unique())
            before = len(pure_codes)
            pure_codes &= qualified
            logger.info(f"研报过滤: {before} → {len(pure_codes)}（买入>{self._research_report_min_count}次）")
        except Exception as e:
            logger.warning(f"研报过滤异常: {e}，跳过研报过滤")
        return pure_codes

    def sync_stock_pool_and_kline(self, target_date: str | None = None) -> set[str]:
        from UtilsManager.CodeNormalizer import CodeNormalizer

        if target_date is None:
            target_date = TradingCalendarAnalyzer().get_last_trading_day()
        today_tag = target_date.replace("-", "")

        pool = self.get_stock_pool_from_db()
        before = len(pool)
        pool = self.filter_st_stocks(pool)
        pool = self.filter_main_board(pool)
        logger.info(f"股票池: {before} → {len(pool)}（过滤ST及非主板）")

        pure_codes = set(pool["股票代码"].unique())
        # K 线同步覆盖全 A 股（已过滤 ST/板块），保持数据完整供回测使用
        symbols = [CodeNormalizer.add_market_prefix(c) for c in sorted(pure_codes)]
        inserted = self.sync_all(symbols)
        logger.info(f"K线同步完成，新增 {inserted} 行")

        # 研报过滤仅影响分析池，不影响 K 线数据完整性
        analysis_pool = self._filter_by_research_report(pure_codes)

        save_dir = os.path.dirname(self._cache_dir) if os.path.isdir(self._cache_dir) else os.getcwd()
        out = os.path.join(save_dir, f"final_filtered_stocks_{today_tag}.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for c in sorted(analysis_pool):
                f.write(f"{c}\n")
        logger.info(f"最终股票列表已保存: {len(analysis_pool)} 只 → {out}")
        return analysis_pool

    def _get_min_latest_date(self, symbols: list[str]) -> date | None:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT MIN(latest) FROM (
                        SELECT MAX(trade_date::date) AS latest FROM {TABLE}
                        WHERE symbol = ANY(:symbols) GROUP BY symbol
                    ) sub
                """),
                {"symbols": symbols},
            ).scalar()
        return rows

    def _detect_split_from_adj(self, symbol: str, new_df: pd.DataFrame, latest: date) -> bool:
        """对比 DB 中最新的 adj_factor 与新批次数据是否一致。"""
        with self._engine.connect() as conn:
            old = conn.execute(
                text(f"SELECT adj_factor FROM {TABLE} WHERE symbol = :sym AND trade_date::date = :date"),
                {"sym": symbol, "date": latest},
            ).scalar()
        if old is None:
            return False
        new_adj = new_df.loc[new_df['trade_date'] == latest.isoformat(), 'adj_factor']
        if new_adj.empty:
            return False
        ratio = new_adj.iloc[0] / old
        return ratio > 1.01 or ratio < 0.99

    # ── stale filter (P0-1) ─────────────────────────────────────

    def _get_stale_symbols(self, symbols: list[str]) -> list[str]:
        if not symbols:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT symbol FROM {TABLE}
                    WHERE symbol = ANY(:symbols)
                      AND trade_date::date = :trade_date
                """),
                {"symbols": symbols, "trade_date": self._trade_date.isoformat()},
            ).fetchall()
        up_to_date = {row[0] for row in rows}
        stale = [s for s in symbols if s not in up_to_date]
        skipped = len(symbols) - len(stale)
        if skipped:
            logger.info(f"跳过 {skipped} 只（已有 {self._trade_date_str} 数据），需处理 {len(stale)} 只")
        return stale

    def _has_no_close_normal(self, symbol: str) -> bool:
        """DB 中 close_normal 为 NULL（从未写入原始不复权收盘价），需全量刷新。"""
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT close_normal FROM {TABLE} WHERE symbol = :sym ORDER BY trade_date DESC LIMIT 1"),
                {"sym": symbol},
            ).scalar()
        return row is None

    # ── trading day alignment ────────────────────────────────────

    @staticmethod
    def align_to_trading_day(date_str: str) -> str:
        """将 YYYYMMDD 对齐到当天或之后的首个交易日，返回 YYYYMMDD。"""
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            cal = TradingCalendarAnalyzer()
            dates = sorted(cal.get_official_trading_dates())
            dt = datetime.strptime(date_str, "%Y%m%d")
            formatted = dt.strftime("%Y-%m-%d")
            for d in dates:
                if d >= formatted:
                    return d.replace("-", "")
        except Exception:
            pass
        return date_str

    _align_to_trading_day = align_to_trading_day

    # ── database ────────────────────────────────────────────────

    def _get_latest_date(self, symbol: str) -> date | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT MAX(trade_date::date) FROM {TABLE} WHERE symbol = :sym"),
                {"sym": symbol},
            ).scalar()
        return row if row is not None else None

    def _write_batch(self, df: pd.DataFrame) -> None:
        """幂等写入：(symbol, trade_date) 唯一约束 + ON CONFLICT DO UPDATE。"""
        if df.empty:
            return
        records = df.rename(columns={}).to_dict("records")
        columns = ["symbol", "trade_date", "open", "close", "high", "low", "volume", "amount", "adj_factor", "close_normal"]
        placeholders = ", ".join(f":{c}" for c in columns)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in ("symbol", "trade_date"))
        with self._engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {TABLE} ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (symbol, trade_date) DO UPDATE SET {updates}
                """),
                records,
            )

    # ── failed-symbols cache ─────────────────────────────────────

    def _failed_cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"failed_symbols_{self._trade_date_str}.txt")

    def _load_failed_set(self) -> set[str]:
        path = self._failed_cache_path()
        if not os.path.exists(path):
            return set()
        try:
            with open(path, encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            logger.warning(f"读取失败股票缓存异常: {e}")
            return set()

    def _save_failed_set(self, symbols: set[str]) -> None:
        path = self._failed_cache_path()
        if not symbols:
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path, "w", encoding="utf-8") as f:
            for sym in sorted(symbols):
                f.write(f"{sym}\n")
        logger.warning(f"缓存 {len(symbols)} 只失败股票 → {os.path.basename(path)}")

    # ── helpers ─────────────────────────────────────────────────

    def _close_normal_cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"close_normal_{self._trade_date_str}.csv")


