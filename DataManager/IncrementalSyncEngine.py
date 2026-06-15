from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger
from sqlalchemy import text

from UtilsManager.CodeNormalizer import CodeNormalizer


def _strip_prefix(symbol: str) -> str:
    for prefix in ("sh", "sz", "bj"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol


TABLE = "stock_daily_kline"
OVERLAP_DAYS = 20
BATCH_SIZE = 500
MAX_WORKERS = 15
RETRY_SLEEP = 10


class IncrementalSyncEngine:
    def __init__(self, db_engine: Any) -> None:
        self._engine = db_engine
        self._cache_dir = os.path.join(
            os.environ.get("TEMP", "/tmp"), "opencode", "kline_batches"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

    # ── public API ──────────────────────────────────────────────

    def sync_all(self, symbols_prefixed: list[str]) -> int:
        from tqdm import tqdm

        total_inserted = 0
        all_success: list[str] = []
        all_failed: list[str] = []

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
            self._merge_and_write()
            return 0

        total_new_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        pbar = tqdm(
            total=len(remaining),
            desc="增量同步 K 线",
            unit="只",
            ncols=80,
        )

        for batch_idx in range(total_new_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(remaining))
            batch_symbols = remaining[start:end]

            batch_success: list[pd.DataFrame] = []
            batch_failed: list[str] = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(self._sync_one, sym): sym for sym in batch_symbols
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        n, df = future.result()
                        if n > 0 and df is not None:
                            batch_success.append(df)
                            all_success.append(sym)
                            total_inserted += n
                        else:
                            # None return means full skip (no data / error)
                            pass
                    except Exception:
                        batch_failed.append(sym)
                        all_failed.append(sym)
                    finally:
                        pbar.update(1)

            # write this batch to DB + cache
            if batch_success:
                merged = pd.concat(batch_success, ignore_index=True)
                self._write_batch(merged)
                cache_file = os.path.join(
                    self._cache_dir,
                    f"kline_batch_{date.today().isoformat().replace('-', '')}_{batch_idx}.csv",
                )
                merged.to_csv(cache_file, sep="|", index=False, encoding="utf-8-sig")

            if batch_failed:
                self._save_failed(batch_failed)

            if batch_idx < total_new_batches - 1:
                time.sleep(RETRY_SLEEP)

        pbar.close()

        if all_failed:
            logger.info(
                f"\n[统计] 总{len(remaining)}只 | "
                f"成功{total_inserted}条 | 失败{len(all_failed)}只"
            )
        else:
            self._clear_failed()
            self._save_success(all_success)
            logger.info(f"\n[统计] 全部 {len(remaining)} 只股票同步完成 ✓")

        return total_inserted

    # ── single stock sync ───────────────────────────────────────

    def _sync_one(self, symbol: str) -> tuple[int, pd.DataFrame | None]:
        pure = _strip_prefix(symbol)
        latest = self._get_latest_date(symbol)

        if latest is None:
            return self._full_refresh(symbol, pure)

        df = self._fetch_from_tx(
            pure, (latest - timedelta(days=OVERLAP_DAYS * 2)).isoformat()
        )
        if df is None or df.empty:
            return 0, None

        if self._detect_split(symbol, df, latest):
            logger.info(f"  {symbol} 检测到除权除息，全量重拉")
            return self._full_refresh(symbol, pure)

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
                        (symbol, trade_date, open, close, high, low, volume, amount, adj_factor)
                    VALUES (:symbol, :trade_date, :open, :close, :high, :low, :volume, :amount, :adj_factor)
                """),
                df.rename(columns={}).to_dict("records"),
            )

    def _merge_and_write(self) -> None:
        """加载今日缓存文件并写入 DB（重跑或增量场景）"""
        dfs = []
        for fname in sorted(os.listdir(self._cache_dir)):
            if not fname.endswith(".csv"):
                continue
            dfs.append(
                pd.read_csv(
                    os.path.join(self._cache_dir, fname), sep="|", encoding="utf-8-sig"
                )
            )
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            self._write_batch(merged)
            logger.info(f"合并写入 {len(merged)} 条记录")

    # ── Tencent API ────────────────────────────────────────────

    def _fetch_from_tx(
        self, pure_code: str, start: str | None = None
    ) -> pd.DataFrame | None:
        prefixed = CodeNormalizer.add_market_prefix(pure_code)
        start_str = (start or "20000101").replace("-", "")
        end_str = date.today().isoformat().replace("-", "")

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

    # ── resume helpers ──────────────────────────────────────────

    @property
    def _failed_file(self) -> str:
        td = date.today().isoformat().replace("-", "")
        return os.path.join(self._cache_dir, f"failed_{td}.json")

    @property
    def _success_file(self) -> str:
        td = date.today().isoformat().replace("-", "")
        return os.path.join(self._cache_dir, f"success_{td}.json")

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
