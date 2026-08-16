from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

TABLE = "stock_daily_kline"

# 回测会话级 advisory lock 键：回测全程持有，外部数据同步检测到后必须让路，
# 避免运行中改写 K 线导致信号缓存内容漂移（与 runner._BACKTEST_LOCK_KEY 共用）。
BACKTEST_ADVISORY_LOCK_KEY = 987654321


def backtest_lock_held(engine: Engine) -> bool:
    """P3.3 制度化探测：回测进程是否持有会话级 advisory lock。

    同步引擎在写 K 线前统一调用本函数；返回 True 表示回测运行中，
    必须整体跳过本次同步（探测后立即释放锁，本进程无需持有）。
    """
    try:
        with engine.connect() as conn:
            acquired = conn.execute(
                text(f"SELECT pg_try_advisory_lock({BACKTEST_ADVISORY_LOCK_KEY})")
            ).scalar()
            if not acquired:
                return True
            conn.execute(text(f"SELECT pg_advisory_unlock({BACKTEST_ADVISORY_LOCK_KEY})"))
            return False
    except Exception:
        return False


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
    """实时/日更模式 — 从 stock_daily_kline 读取，返回不复权价和后复权价。

    列名规范（P0-12 价格空间审计修复）：
    - close/open/high/low       → 不复权原始价（交易所真实成交价，用于涨跌停模型/撮合）
    - close_normal/open_normal/… → 后复权价（跨除权日连续，用于信号/止损/估值）
    （P3 审计修复：实际列名为 close_normal/open_normal（见 sync.py 列定义），
    原 docstring 误写 close_adj/open_adj 已修正）
    """

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
            SELECT symbol, trade_date,
                   open, high, low, close,
                   open AS open_raw,
                   high AS high_raw,
                   low AS low_raw,
                   close AS close_raw,
                   open_normal, high_normal, low_normal, close_normal,
                   volume, amount, adj_factor
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
    """回测模式 — 从 stock_daily_kline 读取，end_date 截断到 replay_date，返回双价格。"""

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
            SELECT symbol, trade_date,
                   open, high, low, close,
                   open AS open_raw,
                   high AS high_raw,
                   low AS low_raw,
                   close AS close_raw,
                   open_normal, high_normal, low_normal, close_normal,
                   volume, amount, adj_factor
            FROM {TABLE}
            WHERE {' AND '.join(where)}
            ORDER BY symbol, trade_date
        """)
        params = {"symbols": list(symbols), "end_date": actual_end}
        if start_date:
            params["start_date"] = start_date

        with self._db_engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
