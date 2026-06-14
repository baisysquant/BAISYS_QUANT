"""
数据提供者接口 + 实时/回测两种实现

解耦 ``StockAnalysisCoordinator`` 与具体的数据获取逻辑，使回测时
只需替换数据提供者即可模拟历史日期的数据状态。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy.engine import Engine

from DataManager.KlineAdjuster import get_adjusted_kline


class IDataProvider(ABC):
    """数据提供者接口 — 所有数据获取方法在此定义。"""

    @abstractmethod
    def get_kline(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        adj_type: str = "qfq",
    ) -> pd.DataFrame:
        """获取 K 线数据。"""


class LiveDataProvider(IDataProvider):
    """实时/日更模式 — 直接查询数据库的全量历史数据。"""

    def __init__(self, db_engine: Engine) -> None:
        self._db_engine = db_engine

    def get_kline(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        adj_type: str = "qfq",
    ) -> pd.DataFrame:
        return get_adjusted_kline(self._db_engine, symbols, start_date, end_date, adj_type)


class BacktestDataProvider(IDataProvider):
    """回测模式 — 将 end_date 截断到 replay_date，防止看到未来数据。

    Args:
        db_engine: SQLAlchemy 数据库引擎
        replay_date: 回测锚定日期，所有查询的 end_date 不超过此日期
    """

    def __init__(self, db_engine: Engine, replay_date: str) -> None:
        self._db_engine = db_engine
        self._replay_date = replay_date

    def get_kline(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        adj_type: str = "qfq",
    ) -> pd.DataFrame:
        actual_end = self._replay_date
        if end_date is not None:
            actual_end = min(end_date, self._replay_date)
        return get_adjusted_kline(self._db_engine, symbols, start_date, actual_end, adj_type)
