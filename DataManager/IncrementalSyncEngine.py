from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger
from sqlalchemy import text

from UtilsManager.CodeNormalizer import CodeNormalizer


def _strip_prefix(symbol: str) -> str:
    """去掉 sh/sz/bj 前缀，返回纯数字代码。"""
    for prefix in ("sh", "sz", "bj"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol

TABLE = "stock_daily_kline"
OVERLAP_DAYS = 20


class IncrementalSyncEngine:
    """增量同步引擎 — 每日只拉新增行，检测到除权除息时全量重拉。

    统一写入 ``stock_daily_kline``，复盘和回测都读同一张表。
    """

    def __init__(self, db_engine: Any) -> None:
        self._engine = db_engine

    def sync_all(self, symbols_prefixed: list[str]) -> int:
        """增量同步全量股票，返回本次新增行数。"""
        from tqdm import tqdm

        total = 0
        for sym in tqdm(symbols_prefixed, desc="增量同步 K 线", unit="只", ncols=80):
            total += self._sync_one(sym)
        return total

    # ── 单只股票同步 ──────────────────────────────────────────

    def _sync_one(self, symbol: str) -> int:
        pure = _strip_prefix(symbol)
        latest = self._get_latest_date(symbol)

        if latest is None:
            return self._full_refresh(symbol, pure)

        df = self._fetch_from_tx(pure, (latest - timedelta(days=OVERLAP_DAYS * 2)).isoformat())
        if df is None or df.empty:
            return 0

        if self._detect_split(symbol, df, latest):
            logger.info(f"  {symbol} 检测到除权除息，全量重拉")
            return self._full_refresh(symbol, pure)

        new = df[df["trade_date"] > latest]
        if new.empty:
            return 0
        self._write_batch(symbol, new)
        return len(new)

    def _full_refresh(self, symbol: str, pure_code: str) -> int:
        self._delete_stock(symbol)
        df = self._fetch_from_tx(pure_code)
        if df is None or df.empty:
            return 0
        self._write_batch(symbol, df)
        return len(df)

    # ── 数据库读写 ────────────────────────────────────────────

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

    def _write_batch(self, symbol: str, df: pd.DataFrame) -> None:
        """写数据前先删掉该股可能冲突的行，避免 stock_daily_kline 无主键导致重复。"""
        dates = df["trade_date"].tolist()
        if not dates:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {TABLE} WHERE symbol = :sym AND trade_date IN :dates"),
                {"sym": symbol, "dates": tuple(dates)},
            )
            conn.execute(
                text(f"""
                    INSERT INTO {TABLE}
                        (symbol, trade_date, open, close, high, low, volume, amount, adj_factor)
                    VALUES (:symbol, :trade_date, :open, :close, :high, :low, :volume, :amount, :adj_factor)
                """),
                df.rename(columns={}).to_dict("records"),
            )

    # ── akshare 获取 ─────────────────────────────────────────

    def _fetch_from_tx(self, pure_code: str, start: str | None = None) -> pd.DataFrame | None:
        """
        从腾讯源获取K线数据。
        
        注意: akshare.stock_zh_a_hist_tx 要求小写前缀 + 数字格式，如 "sh600006" 或 "sz000001"
        不接受纯数字格式。
        """
        # 使用 CodeNormalizer.add_market_prefix() 规范化为小写前缀格式
        tx_symbol = CodeNormalizer.add_market_prefix(pure_code).lower()
        
        try:
            raw = ak.stock_zh_a_hist_tx(
                symbol=tx_symbol,
                start_date=(start or "20000101").replace("-", ""),
                end_date=date.today().isoformat().replace("-", ""),
                adjust="hfq",
            )
            if raw is None or raw.empty:
                return None
        except Exception as e:
            logger.warning(f"  {tx_symbol} 腾讯源下载失败: {e}")
            return None

        df = raw.rename(columns={"date": "trade_date"}).copy()
        df["symbol"] = CodeNormalizer.add_market_prefix(pure_code)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        
        # 强化数据类型转换 - 用 errors="coerce" 把无效值转成 NaN
        for col in ("open", "close", "high", "low"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 修复 volume 计算 - 确保 amount 先转成数字再乘以 100
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["volume"] = (df["amount"].fillna(0) * 100).astype("int64")
        
        # 重新计算 amount = close * volume
        df["amount"] = df["close"] * df["volume"]
        df["adj_factor"] = 1.0
        
        # 删除数据异常的行（包含 NaN 的关键列）
        df = df.dropna(subset=["trade_date", "close", "volume"], how="any")
        
        if df.empty:
            logger.warning(f"  {tx_symbol} 数据清洗后为空")
            return None
        
        return df

    # ── 除权检测 ─────────────────────────────────────────────

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
                text(f"""
                    SELECT trade_date, close FROM {TABLE}
                    WHERE symbol = :sym AND trade_date::date >= :start
                    ORDER BY trade_date
                """),
                conn,
                params={"sym": symbol, "start": start},
            )
