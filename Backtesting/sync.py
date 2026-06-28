from __future__ import annotations

from typing import Any

from sqlalchemy import text


_UNIQUE_INDEX_NAME = "uq_stock_daily_kline_symbol_trade_date"


def ensure_table(
    engine: Any,
    table: str = "stock_daily_kline",
) -> None:
    """确保 stock_daily_kline 有 adj_factor 列 + (symbol, trade_date) 唯一索引。"""
    with engine.connect() as conn:
        conn.execute(
            text(f"""
                ALTER TABLE {table}
                ADD COLUMN IF NOT EXISTS adj_factor DOUBLE PRECISION DEFAULT 1.0
            """)
        )
        conn.execute(
            text(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {_UNIQUE_INDEX_NAME}
                ON {table} (symbol, trade_date)
            """)
        )
        conn.commit()
