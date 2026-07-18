from __future__ import annotations

from typing import Any

from sqlalchemy import text


_UNIQUE_INDEX_NAME = "uq_stock_daily_kline_symbol_trade_date"


def ensure_table(
    engine: Any,
    table: str = "stock_daily_kline",
) -> None:
    """确保 stock_daily_kline 有 adj_factor 列 + (symbol, trade_date) 唯一索引，
    同时确保因子数据表存在。"""
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

    _ensure_factor_tables(engine)


def _ensure_factor_tables(engine: Any) -> None:
    """确保多因子 Alpha 所需的数据表存在。"""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.ods_factor_ic_history (
                id SERIAL PRIMARY KEY,
                factor_name VARCHAR(20) NOT NULL,
                check_date DATE NOT NULL,
                rolling_ic_mean FLOAT,
                rolling_ic_std FLOAT,
                icir FLOAT,
                rank_ic FLOAT,
                normal_ic FLOAT,
                decile_spread FLOAT,
                decile_monotonicity FLOAT,
                is_decayed BOOLEAN DEFAULT FALSE,
                current_weight FLOAT,
                suggested_weight FLOAT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.ods_index_daily (
                index_code VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume FLOAT,
                amount FLOAT,
                PRIMARY KEY (index_code, trade_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.ods_financial_quality (
                symbol VARCHAR(20) NOT NULL,
                record_date DATE NOT NULL,
                roe FLOAT,
                gross_profit_margin FLOAT,
                net_profit_margin FLOAT,
                revenue_growth_rate FLOAT,
                net_profit_growth_rate FLOAT,
                PRIMARY KEY (symbol, record_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.ods_financial_valuation (
                symbol VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                pe FLOAT,
                pe_ttm FLOAT,
                pb FLOAT,
                total_mv FLOAT,
                circ_mv FLOAT,
                PRIMARY KEY (symbol, trade_date)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_fv_date
            ON public.ods_financial_valuation (trade_date)
        """))
