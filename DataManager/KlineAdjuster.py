"""
K线复权调整工具 — 支持前/后/等比复权的动态切换

依赖 stock_daily_kline 表中已有的 adj_ratio / adj_factor 列。

调整类型：
    - qfq (前复权): prices = close_normal * adj_factor  (默认)
    - hfq (后复权): prices = close_normal * max(adj_factor) / adj_factor
    - pfq (等比复权): prices = close_normal * max(adj_factor) / adj_factor * first_adj / adj_factor
                     （简化：等比等于后复权 * 首日因子归一化）
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text


def get_adjusted_kline(
    engine: Any,  # noqa: ANN401
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    adj_type: str = "hfq",
) -> pd.DataFrame:
    """从 stock_daily_kline 查询 K 线数据并按要求调整价格。

    Args:
        engine: SQLAlchemy 数据库引擎
        symbols: 股票代码列表（带市场前缀，如 ``sh600000``）
        start_date: 起始日期（含，格式 ``YYYY-MM-DD``）
        end_date: 截止日期（含，格式 ``YYYY-MM-DD``）
        adj_type: 复权类型 --- ``qfq``（前复权，默认）/ ``hfq``（后复权）/ ``pfq``（等比复权）

    Returns:
        pd.DataFrame:
            包含 ``trade_date``, ``symbol``, ``open``, ``high``, ``low``, ``close``, ``volume``, ``amount``
            以及原始 ``close_normal``, ``adj_factor`` 列。
            价格列已按要求调整，``close_normal`` 始终为不复权收盘价。
    """
    where_clauses = ["symbol = ANY(:symbols)"]
    if start_date:
        where_clauses.append("trade_date >= :start_date")
    if end_date:
        where_clauses.append("trade_date <= :end_date")

    sql = text(f"""
        SELECT trade_date, symbol, "open", "close", high, low, volume, amount,
               close_normal, adj_factor
        FROM stock_daily_kline
        WHERE {' AND '.join(where_clauses)}
        ORDER BY symbol, trade_date
    """)

    params = {"symbols": list(symbols)}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        return df

    _apply_adjustment(df, adj_type)
    return df


def _apply_adjustment(df: pd.DataFrame, adj_type: str) -> None:
    """就地调整 DataFrame 中的价格列。"""
    if adj_type == "qfq":
        # 前复权：已经是 close / adj_factor * adj_factor = close
        pass

    elif adj_type == "hfq":
        # 后复权：每只股票独立计算
        _adjust_hfq(df)

    elif adj_type == "pfq":
        # 等比复权
        _adjust_pfq(df)

    else:
        raise ValueError(f"不支持的复权类型: {adj_type}")


def _adjust_hfq(df: pd.DataFrame) -> None:
    """后复权：close_normal * max(adj_factor) / adj_factor"""
    for symbol in df["symbol"].unique():
        mask = df["symbol"] == symbol
        indices = df.index[mask]
        factors = df.loc[indices, "adj_factor"].fillna(1.0)
        max_factor = factors.max()
        if max_factor <= 0:
            continue
        ratio = max_factor / factors
        for col in ["open", "high", "low", "close"]:
            raw = df.loc[indices, col]
            df.loc[indices, col] = raw / ratio.values


def _adjust_pfq(df: pd.DataFrame) -> None:
    """等比复权：close_normal * prod(1 + ratio_i) 简化版本 = hfq 再归一化"""
    _adjust_hfq(df)
    for symbol in df["symbol"].unique():
        mask = df["symbol"] == symbol
        indices = df.index[mask]
        first_close = df.loc[indices[0], "close"]
        if first_close <= 0:
            continue
        hfq_close = df.loc[indices, "close"].values
        scale = first_close / hfq_close[0]
        for col in ["open", "high", "low", "close"]:
            df.loc[indices, col] = df.loc[indices, col].values * scale
