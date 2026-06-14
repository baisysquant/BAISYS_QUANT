from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import text

from UtilsManager.IDataProvider import IDataProvider

TABLE = "stock_daily_kline"


class BacktestDataProvider(IDataProvider):
    """回测数据提供者 — 从 stock_daily_kline 读取，end_date 截断到 replay_date。"""

    def __init__(self, engine: Any, replay_date: str | None = None) -> None:
        self._engine = engine
        self._replay_date = replay_date

    def get_kline(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        end = self._replay_date or end_date or date.today().isoformat()

        sql = text(f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount
            FROM {TABLE}
            WHERE symbol = ANY(:symbols)
              AND trade_date >= :start
              AND trade_date <= :end
            ORDER BY symbol, trade_date
        """)
        params = {"symbols": list(symbols), "start": start_date or "2000-01-01", "end": end}
        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
