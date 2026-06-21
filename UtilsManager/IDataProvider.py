from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

TABLE = "stock_daily_kline"


class IDataProvider(ABC):
    @abstractmethod
    def get_kline(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取 K 线数据。"""


class LiveDataProvider(IDataProvider):
    """实时/日更模式 — 从 stock_daily_kline 读取后复权数据。"""

    def __init__(self, db_engine: Engine) -> None:
        self._db_engine = db_engine

    def get_kline(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        where = ["symbol = ANY(:symbols)"]
        if start_date:
            where.append("trade_date >= :start_date")
        if end_date:
            where.append("trade_date <= :end_date")

        sql = text(f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount, close_normal
            FROM {TABLE}
            WHERE {' AND '.join(where)}
            ORDER BY symbol, trade_date
        """)
        params = {"symbols": list(symbols)}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        with self._db_engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)

        if df.empty:
            logger.warning(f"{TABLE} 无数据 (symbols={len(symbols)}, start={start_date}, end={end_date})")
        return df


class BacktestDataProvider(IDataProvider):
    """回测模式 — 从 stock_daily_kline 读取，end_date 截断到 replay_date。"""

    def __init__(self, db_engine: Engine, replay_date: str | None = None) -> None:
        self._db_engine = db_engine
        self._replay_date = replay_date

    def get_kline(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        actual_end = end_date
        if self._replay_date is not None:
            actual_end = self._replay_date if end_date is None else min(end_date, self._replay_date)

        where = ["symbol = ANY(:symbols)"]
        if start_date:
            where.append("trade_date >= :start_date")
        where.append("trade_date <= :end_date")

        sql = text(f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount, close_normal
            FROM {TABLE}
            WHERE {' AND '.join(where)}
            ORDER BY symbol, trade_date
        """)
        params = {"symbols": list(symbols), "end_date": actual_end}
        if start_date:
            params["start_date"] = start_date

        with self._db_engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
