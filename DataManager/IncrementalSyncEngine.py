from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import date, datetime, timedelta
from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger
from sqlalchemy import text

from DataCollection.CalendarManager import TradingCalendarAnalyzer
from UtilsManager.CodeNormalizer import CodeNormalizer


def _strip_prefix(symbol: str) -> str:
    for prefix in ("sh", "sz", "bj"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol


TABLE = "stock_daily_kline"
OVERLAP_DAYS = 15
SYNC_BATCH_SIZE = 500
SYNC_WORKERS = 5
STAGGER_DELAY = 0.5
RETRY_SLEEP = 10


class IncrementalSyncEngine:
    def __init__(
        self,
        db_engine: Any,
        default_start: str | None = None,
        asharehub_api_key: str | None = None,
    ) -> None:
        self._engine = db_engine
        self._default_start = self.align_to_trading_day(default_start) if default_start else None
        self._asharehub_api_key = asharehub_api_key
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
        from tqdm import tqdm

        total_inserted = 0
        all_success: list[str] = []
        all_failed: list[str] = []

        # skip delisted stocks before making any API call
        delisted = self._load_delisted()
        if delisted:
            before = len(symbols_prefixed)
            symbols_prefixed = [s for s in symbols_prefixed if s not in delisted]
            logger.info(f"跳过 {before - len(symbols_prefixed)} 只已退市股票")

        # load previous run state for resume
        resume_failed = self._load_failed()
        if resume_failed:
            logger.info(f"发现 {len(resume_failed)} 只股票待重试，优先处理")
            symbols_prefixed = resume_failed + [
                s for s in symbols_prefixed if s not in resume_failed
            ]

        # skip already synced today
        today_synced = self._load_success()
        remaining = [s for s in symbols_prefixed if s not in today_synced]
        logger.info(
            f"已跳过 {len(symbols_prefixed) - len(remaining)} 只（今日已同步），"
            f"本次需处理 {len(remaining)} 只"
        )
        if not remaining:
            return 0

        # 一次性加载回填确认缓存到内存
        self._backfill_cache = self._load_backfill_checked()

        total_batches = (len(remaining) + SYNC_BATCH_SIZE - 1) // SYNC_BATCH_SIZE
        pbar = tqdm(
            total=len(remaining),
            desc="增量同步 K 线",
            unit="只",
            ncols=80,
        )

        for batch_idx in range(total_batches):
            start = batch_idx * SYNC_BATCH_SIZE
            end = min(start + SYNC_BATCH_SIZE, len(remaining))
            batch_symbols = remaining[start:end]

            cache_files: list[str] = []
            batch_failed: list[str] = []

            # 初始错峰：前 SYNC_WORKERS 个任务间隔提交，避免 10 线程同时打 API
            with ThreadPoolExecutor(max_workers=SYNC_WORKERS) as pool:
                futures = {}
                for i, sym in enumerate(batch_symbols[:SYNC_WORKERS]):
                    cache_path = os.path.join(
                        self._cache_dir,
                        f"_sync_{batch_idx}_{i}_{sym.replace('/', '_')}.pkl",
                    )
                    fut = pool.submit(self._sync_one_to_cache, sym, cache_path)
                    futures[fut] = (sym, cache_path, time.time())
                    time.sleep(STAGGER_DELAY)
                # 后续任务自然排队（线程数固定即限流）
                for i, sym in enumerate(batch_symbols[SYNC_WORKERS:], SYNC_WORKERS):
                    cache_path = os.path.join(
                        self._cache_dir,
                        f"_sync_{batch_idx}_{i}_{sym.replace('/', '_')}.pkl",
                    )
                    futures[pool.submit(self._sync_one_to_cache, sym, cache_path)] = (sym, cache_path, time.time())

                # 轮询等待，单任务 120s 超时
                STOCK_TIMEOUT = 120
                pending = set(futures.keys())
                while pending:
                    done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    now = time.time()
                    for future in done:
                        sym, cache_path, _ = futures[future]
                        try:
                            n = future.result()
                            if n > 0:
                                cache_files.append(cache_path)
                                all_success.append(sym)
                                total_inserted += n
                        except Exception:
                            batch_failed.append(sym)
                            all_failed.append(sym)
                        finally:
                            pbar.update(1)
                    # 超时检查：超过 120s 未完成的任务强制取消
                    expired = [
                        f for f in pending
                        if now - futures[f][2] > STOCK_TIMEOUT
                    ]
                    for future in expired:
                        sym, cache_path, _ = futures[future]
                        logger.warning(f"  {sym} 同步超时 (>120s)，取消")
                        future.cancel()
                        pending.discard(future)
                        batch_failed.append(sym)
                        all_failed.append(sym)
                        pbar.update(1)

            # 从本地缓存文件合并写入 DB
            if cache_files:
                import pickle
                dfs = []
                valid_files = [f for f in cache_files if os.path.exists(f)]
                for f in valid_files:
                    try:
                        with open(f, "rb") as _fh:
                            dfs.append(pickle.load(_fh))
                    except Exception:
                        pass
                if dfs:
                    merged = pd.concat(dfs, ignore_index=True)
                    self._write_batch(merged)
                # 清理缓存文件
                for f in valid_files:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                # 增量保存成功列表（崩溃恢复）
                batch_ok = [sym for sym in batch_symbols if sym not in batch_failed]
                self._save_success(batch_ok)

            if batch_failed:
                self._save_failed(batch_failed)

            if batch_idx < total_batches - 1:
                time.sleep(RETRY_SLEEP)

        pbar.close()

        if all_failed:
            logger.info(
                f"\n[统计] 总{len(remaining)}只 | "
                f"成功{total_inserted}条 | 失败{len(all_failed)}只"
            )
        else:
            self._clear_failed()
            logger.info(f"\n[统计] 全部 {len(remaining)} 只股票同步完成 ✓")

        return total_inserted

    def _sync_one_to_cache(self, symbol: str, cache_path: str) -> int:
        """同步一只股票，结果写入本地 pickle 缓存文件。返回插入行数。"""
        try:
            n, df = self._sync_one(symbol)
            if n > 0 and df is not None:
                import pickle
                with open(cache_path, "wb") as f:
                    pickle.dump(df, f, protocol=4)
            return n
        except Exception as e:
            logger.warning(f"  {symbol} 同步失败: {e}")
            raise

    def backfill_close_normal(self, symbols_prefixed: list[str] | None = None) -> pd.DataFrame:
        """获取全市场最新不复权收盘价，写入本地缓存文件。

        AShareHub 批量获取后写入 ``close_normal_{YYYYMMDD}.csv``（ts_code, close 两列），
        各模块从此缓存文件读取，不再写 DB。

        Args:
            symbols_prefixed: 忽略（为了兼容旧调用签名）。
        Returns:
            DataFrame 包含 ts_code, close 两列；失败返回空 DataFrame。
        """
        cache_file = self._close_normal_cache_path()
        today_iso = self._trade_date.isoformat()

        # 当天缓存已存在，直接读取
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"[close_normal] 从缓存加载 {len(df)} 只 → {os.path.basename(cache_file)}")
                return df
            except Exception as e:
                logger.warning(f"[close_normal] 缓存读取失败: {e}，重新拉取")

        if not self._asharehub_api_key:
            logger.warning("[close_normal] ASHAREHUB_API_KEY 未配置，跳过")
            return pd.DataFrame()

        logger.info("[close_normal] AShareHub 批量获取全市场最新收盘价...")
        try:
            from asharehub import AShareHub
            client = AShareHub(api_key=self._asharehub_api_key)

            offset = 0
            limit = 5000
            all_rows = []
            while True:
                batch = client.market_daily(start_date=today_iso, end_date=today_iso, limit=limit, offset=offset)
                if batch is None or batch.empty:
                    break
                all_rows.extend(batch.to_dict("records"))
                if len(batch) < limit:
                    break
                offset += limit

            if not all_rows:
                logger.warning("[close_normal] AShareHub 返回空数据")
                return pd.DataFrame()

            df = pd.DataFrame(all_rows)[["ts_code", "close"]].copy()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"])
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_csv(cache_file, index=False)
            logger.info(f"[close_normal] 缓存已保存: {len(df)} 只 → {os.path.basename(cache_file)}")
            return df
        except Exception as e:
            logger.warning(f"[close_normal] AShareHub 批量请求失败: {e}")
            return pd.DataFrame()

    # ── single stock sync ───────────────────────────────────────

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

    def _sync_one(self, symbol: str) -> tuple[int, pd.DataFrame | None]:
        pure = _strip_prefix(symbol)
        latest = self._get_latest_date(symbol)

        if latest is None:
            return self._full_refresh(symbol, pure)

        # 如果 default_start 比股票最早日期还早，先拉取确认是否有更早的历史数据
        if self._default_start:
            earliest = self._get_earliest_date(symbol)
            default_date = datetime.strptime(self._default_start, "%Y%m%d").date()
            if earliest is not None and earliest > default_date:
                # 已确认过后上市股票，跳过全量拉取检查
                if symbol in self._backfill_cache:
                    df = self._fetch_from_tx(
                        pure, (latest - timedelta(days=OVERLAP_DAYS)).isoformat()
                    )
                    if df is None or df.empty:
                        return 0, None
                    new = df[df["trade_date"] > latest.isoformat()]
                    if new.empty:
                        return 0, None
                    return len(new), new

                df = self._fetch_from_tx(pure)
                if df is None or df.empty:
                    return 0, None
                df_earliest = datetime.strptime(df["trade_date"].min(), "%Y-%m-%d").date()
                if df_earliest < earliest:
                    logger.info(f"  {symbol} 发现更早数据 {df_earliest}，全量替换")
                    self._delete_stock(symbol)
                    return len(df), df
                # 没有更早数据（后上市股票），记录到缓存，下次跳过拉取检查
                self._backfill_cache.add(symbol)
                self._save_backfill_checked([symbol])
                new = df[df["trade_date"] > latest.isoformat()]
                if new.empty:
                    return 0, None
                return len(new), new

        df = self._fetch_from_tx(
            pure, (latest - timedelta(days=OVERLAP_DAYS)).isoformat()
        )
        if df is None or df.empty:
            return 0, None

        if self._detect_split(symbol, df, latest):
            logger.info(f"  {symbol} 检测到除权除息，全量替换")
            self._delete_stock(symbol)
            return len(df), df

        new = df[df["trade_date"] > latest.isoformat()]
        if new.empty:
            return 0, None

        return len(new), new

    def _full_refresh(self, symbol: str, pure_code: str) -> tuple[int, pd.DataFrame | None]:
        self._delete_stock(symbol)
        df = self._fetch_from_tx(pure_code)
        if df is None or df.empty:
            return 0, None
        return len(df), df

    # ── database ────────────────────────────────────────────────

    def _get_latest_date(self, symbol: str) -> date | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT MAX(trade_date::date) FROM {TABLE} WHERE symbol = :sym"),
                {"sym": symbol},
            ).scalar()
        return row if row is not None else None

    def _get_earliest_date(self, symbol: str) -> date | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT MIN(trade_date::date) FROM {TABLE} WHERE symbol = :sym"),
                {"sym": symbol},
            ).scalar()
        return row if row is not None else None

    def _delete_stock(self, symbol: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {TABLE} WHERE symbol = :sym"),
                {"sym": symbol},
            )

    def _write_batch(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        symbols = df["symbol"].unique().tolist()
        with self._engine.begin() as conn:
            for sym in symbols:
                sym_dates = df[df["symbol"] == sym]["trade_date"].tolist()
                conn.execute(
                    text(
                        f"DELETE FROM {TABLE} WHERE symbol = :sym AND trade_date IN :dates"
                    ),
                    {"sym": sym, "dates": tuple(sym_dates)},
                )
            conn.execute(
                text(f"""
                    INSERT INTO {TABLE}
                        (symbol, trade_date, open, close, high, low, volume, amount, adj_factor, close_normal)
                    VALUES (:symbol, :trade_date, :open, :close, :high, :low, :volume, :amount, :adj_factor, :close_normal)
                """),
                df.rename(columns={}).to_dict("records"),
            )

    # ── Tencent API ────────────────────────────────────────────

    def _fetch_from_tx(
        self, pure_code: str, start: str | None = None
    ) -> pd.DataFrame | None:
        prefixed = CodeNormalizer.add_market_prefix(pure_code)
        start_str = (start or self._default_start or "20000101").replace("-", "")
        end_str = self._trade_date_str

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                raw = ak.stock_zh_a_hist_tx(
                    symbol=prefixed,
                    start_date=start_str,
                    end_date=end_str,
                    adjust="hfq",
                    timeout=30,
                )
                if raw is not None and not raw.empty:
                    break
                return None  # empty response → no data, skip
            except Exception as e:
                err_str = str(e)
                # 连接/超时类错误才重试，NoData 类直接跳过
                is_retryable = any(
                    kw in err_str.lower()
                    for kw in (
                        "timeout", "reset", "connection", "max retries",
                        "remote end", "getaddrinfo", "eof", "broken",
                    )
                ) or isinstance(e, (ConnectionError, TimeoutError))
                if is_retryable and attempt < max_attempts:
                    delay = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(
                        f"  {pure_code} 第{attempt}次连接失败({err_str[:60]}), "
                        f"{delay:.0f}s 后重试"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(f"  {pure_code} 腾讯源下载失败: {err_str[:120]}")
                    return None

        df = raw.rename(columns={"date": "trade_date"}).copy()
        df["symbol"] = CodeNormalizer.add_market_prefix(pure_code)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        for col in ("open", "close", "high", "low"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = (
            pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype("int64") * 100
        )
        df["amount"] = df["close"] * df["volume"]
        df["adj_factor"] = 1.0
        df["close_normal"] = None

        # close_normal 不再在此拉取，由 backfill_close_normal（AShareHub 批量）统一写入 DB
        return df

    # ── split detection ─────────────────────────────────────────

    def _detect_split(self, symbol: str, new_df: pd.DataFrame, latest: date) -> bool:
        old_df = self._read_overlap(symbol, latest)
        if old_df is None or old_df.empty:
            return False

        merged = old_df.merge(
            new_df[["trade_date", "close"]],
            on="trade_date",
            suffixes=("_old", "_new"),
        )
        if merged.empty:
            return False

        ratio = (merged["close_new"] / merged["close_old"]).abs()
        return (ratio.max() > 1.01) or (ratio.min() < 0.99)

    def _read_overlap(self, symbol: str, latest: date) -> pd.DataFrame | None:
        start = latest - timedelta(days=OVERLAP_DAYS)
        with self._engine.connect() as conn:
            return pd.read_sql(
                text(
                    f"""
                    SELECT trade_date, close FROM {TABLE}
                    WHERE symbol = :sym AND trade_date::date >= :start
                    ORDER BY trade_date
                """
                ),
                conn,
                params={"sym": symbol, "start": start},
            )

    # ── delisted filter ─────────────────────────────────────────

    def _load_delisted(self) -> set[str]:
        """占位：stock_basic_info_sw 不含退市日期字段，暂无法过滤。"""
        return set()

    # ── resume helpers ──────────────────────────────────────────

    def _close_normal_cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"close_normal_{self._trade_date_str}.csv")

    @property
    def _failed_file(self) -> str:
        return os.path.join(self._cache_dir, f"failed_{self._trade_date_str}.json")

    @property
    def _success_file(self) -> str:
        return os.path.join(self._cache_dir, f"success_{self._trade_date_str}.json")

    @property
    def _backfill_file(self) -> str:
        key = self._default_start or "default"
        return os.path.join(self._cache_dir, f"backfill_checked_{key}.json")

    def _load_failed(self) -> list[str]:
        if not os.path.exists(self._failed_file):
            return []
        with open(self._failed_file, encoding="utf-8") as f:
            return json.load(f)

    def _save_failed(self, symbols: list[str]) -> None:
        existing = set(self._load_failed())
        all_f = existing | set(symbols)
        with open(self._failed_file, "w", encoding="utf-8") as f:
            json.dump(list(all_f), f, ensure_ascii=False)

    def _clear_failed(self) -> None:
        if os.path.exists(self._failed_file):
            os.remove(self._failed_file)

    def _load_success(self) -> set[str]:
        if not os.path.exists(self._success_file):
            return set()
        with open(self._success_file, encoding="utf-8") as f:
            return set(json.load(f))

    def _save_success(self, symbols: list[str]) -> None:
        existing = self._load_success()
        all_s = existing | set(symbols)
        with open(self._success_file, "w", encoding="utf-8") as f:
            json.dump(list(all_s), f, ensure_ascii=False)

    def _load_backfill_checked(self) -> set[str]:
        if not os.path.exists(self._backfill_file):
            return set()
        with open(self._backfill_file, encoding="utf-8") as f:
            return set(json.load(f))

    def _save_backfill_checked(self, symbols: list[str]) -> None:
        existing = self._load_backfill_checked()
        all_c = existing | set(symbols)
        with open(self._backfill_file, "w", encoding="utf-8") as f:
            json.dump(list(all_c), f, ensure_ascii=False)
