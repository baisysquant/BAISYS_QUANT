from __future__ import annotations

from typing import Any

from sqlalchemy import text


def ensure_table(
    engine: Any,
    table: str = "stock_daily_kline",
) -> None:
    """确保 stock_daily_kline 有 adj_factor 列。"""
    with engine.connect() as conn:
        conn.execute(
            text(f"""
                ALTER TABLE {table}
                ADD COLUMN IF NOT EXISTS adj_factor DOUBLE PRECISION DEFAULT 1.0
            """)
        )
        conn.commit()
