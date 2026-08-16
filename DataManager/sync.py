from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import text


_UNIQUE_INDEX_NAME = "uq_stock_daily_kline_symbol_trade_date"

# P0-12 审计修复：后复权价列（open/close/high/low 为不复权原始价，
# *_normal 为后复权价，adj_factor = close_normal / close）。
_ADJ_COLUMNS = ("open_normal", "high_normal", "low_normal")


def ensure_table(
    engine: Any,
    table: str = "stock_daily_kline",
) -> None:
    """确保 stock_daily_kline 有复权列 + (symbol, trade_date) 唯一索引，
    同时确保因子数据表存在。"""
    # 注意：必须用 begin()（自动提交）——connect() 未显式 commit 会在退出时回滚，
    # 导致 ALTER TABLE 的列从未真正创建（管线曾因此 UndefinedColumn 崩溃）。
    with engine.begin() as conn:
        conn.execute(
            text(f"""
                ALTER TABLE {table}
                ADD COLUMN IF NOT EXISTS adj_factor DOUBLE PRECISION DEFAULT 1.0
            """)
        )
        for _col in _ADJ_COLUMNS:
            conn.execute(
                text(f"""
                    ALTER TABLE {table}
                    ADD COLUMN IF NOT EXISTS {_col} DOUBLE PRECISION
                """)
            )
        conn.execute(
            text(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {_UNIQUE_INDEX_NAME}
                ON {table} (symbol, trade_date)
            """)
        )

    # P0-12 审计修复：历史存量数据复权语义检测与迁移（幂等）
    migrate_price_semantics(engine, table)

    _ensure_factor_tables(engine)


def migrate_price_semantics(engine: Any, table: str = "stock_daily_kline") -> int:
    """检测并修复 stock_daily_kline 的复权语义（幂等，可重复执行）。

    契约（引擎/消费层，见 IDataProvider、prepare、engine/core）：
        open/close/high/low = 不复权原始价；
        open_normal/close_normal/high_normal/low_normal = 后复权价；
        adj_factor = close_normal / close（累计因子）。
    旧版写入层曾把 close 写成后复权价、close_normal 写成原始价（语义反转）。
    本函数抽样检测存量行：

    - 反转态（close / close_normal ≈ adj_factor，占样本 ≥ 80% 且契约态 < 20%）：
      单条 UPDATE 原位互换并回算 *_normal（PG/SQLite 的 SET 右侧均取旧值）：
        open/high/low 除以 adj_factor 还原为原始价；
        close ← 旧 close_normal（原始价）；close_normal ← 旧 close（后复权价）；
        open_normal ← 旧 open（后复权价）；high_normal/low_normal 同理。
      adj_factor 语义不变（后复权 ÷ 原始），无需改动。
    - 契约态（close_normal / close ≈ adj_factor，占样本 ≥ 80%）：仅补全
      *_normal 空列（历史表无该列或未写入时）为 raw × adj_factor。
    - 无法判定：告警并跳过，绝不盲目改写。

    Returns: 受影响行数（仅 UPDATE 行；契约态补列计为 UPDATE 行数）。
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT adj_factor, close, close_normal
                    FROM {table}
                    WHERE adj_factor IS NOT NULL
                      AND adj_factor <> 0
                      AND close > 0
                    LIMIT 500
                """)
            ).fetchall()
    except Exception as e:
        logger.warning(f"[sync] 复权语义检测跳过（查询失败）: {type(e).__name__}: {e}")
        return 0
    if not rows:
        return 0

    # adj_factor ≈ 1 的行对语义判别无区分力（契约/反转两个检验都近似成立），
    # 必须排除，否则会虚增对侧命中数导致真实反转态被误判为"无法判定"。
    classifiable = [(_af, _c, _cn) for _af, _c, _cn in rows
                    if _cn is not None and _cn > 0 and abs(_af - 1.0) > 1e-6]
    contract_hits = sum(
        1 for _af, _c, _cn in classifiable
        if abs(_cn / _c - _af) / max(abs(_af), 1e-9) < 1e-6
    )
    inverted_hits = sum(
        1 for _af, _c, _cn in classifiable
        if abs(_c / _cn - _af) / max(abs(_af), 1e-9) < 1e-6
    )
    classified = len(classifiable)

    if classified > 0 and inverted_hits >= classified * 0.8 and contract_hits < classified * 0.2:
        with engine.begin() as conn:
            n = conn.execute(
                text(f"""
                    UPDATE {table}
                    SET open = open / NULLIF(adj_factor, 0),
                        high = high / NULLIF(adj_factor, 0),
                        low = low / NULLIF(adj_factor, 0),
                        close = close_normal,
                        close_normal = close,
                        open_normal = open,
                        high_normal = high,
                        low_normal = low
                    WHERE adj_factor IS NOT NULL AND adj_factor <> 0
                      AND close_normal IS NOT NULL
                      AND abs(adj_factor - 1.0) > 1e-6
                """)
            ).rowcount
        logger.warning(f"[sync] 复权语义迁移完成（反转态→契约态），影响 {n} 行")
        return n

    if classified == 0 or contract_hits >= classified * 0.8:
        # 契约态（含 close_normal 全为 NULL 的存量行：旧写入层必写 close_normal，
        # 故 NULL 行只能来自其他写入路径，视为契约态原始价）→ 仅补全 *_normal 列
        with engine.begin() as conn:
            n = conn.execute(
                text(f"""
                    UPDATE {table}
                    SET open_normal = COALESCE(open_normal, open * NULLIF(adj_factor, 0)),
                        high_normal = COALESCE(high_normal, high * NULLIF(adj_factor, 0)),
                        low_normal = COALESCE(low_normal, low * NULLIF(adj_factor, 0)),
                        close_normal = COALESCE(close_normal, close * NULLIF(adj_factor, 0))
                    WHERE adj_factor IS NOT NULL AND adj_factor <> 0
                """)
            ).rowcount
        logger.info(f"[sync] 复权语义已为契约态，补全 *_normal 列 {n} 行")
        return n

    logger.warning(
        f"[sync] 复权语义无法判定（契约 {contract_hits}/{classified}，反转 {inverted_hits}/{classified}），"
        "跳过迁移；请人工核查 stock_daily_kline 数据语义"
    )
    return 0


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
                disclosure_date DATE,
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
