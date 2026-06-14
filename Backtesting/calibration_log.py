from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from loguru import logger
from sqlalchemy import text

TABLE = "backtest_calibration_log"


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              SERIAL PRIMARY KEY,
    run_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    frequency       VARCHAR(16) NOT NULL,
    lookback_days   INT         NOT NULL,
    out_of_sample_days INT      NOT NULL,
    initial_cash    NUMERIC(14,2) NOT NULL,
    params          JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    sharpe          NUMERIC(8,4),
    total_return    NUMERIC(8,4),
    max_drawdown    NUMERIC(8,4),
    status          VARCHAR(16) NOT NULL DEFAULT 'success'
);

CREATE INDEX IF NOT EXISTS idx_{TABLE}_run_time ON {TABLE} (run_time DESC);
"""


def ensure_table(engine: Any) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))


def get_last_run(engine: Any) -> dict[str, Any] | None:
    sql = text(f"""
        SELECT run_time, frequency, lookback_days, out_of_sample_days,
               initial_cash, params, sharpe, total_return, max_drawdown, status
        FROM {TABLE}
        WHERE status = 'success'
        ORDER BY run_time DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).mappings().fetchone()
    if row is None:
        return None
    result = dict(row)
    if isinstance(result.get("params"), str):
        result["params"] = json.loads(result["params"])
    return result


def should_rerun(last_run: dict[str, Any] | None, frequency: str, today: date | None = None) -> tuple[bool, str]:
    """判断是否需要重新执行回测。

    Returns:
        (should_run, reason)
    """
    if today is None:
        today = date.today()

    if last_run is None:
        return True, "从未执行过回测"

    last_time: datetime = last_run["run_time"]
    if isinstance(last_time, str):
        last_time = datetime.fromisoformat(last_time)
    last_date = last_time.date()

    if frequency == "initial":
        return False, f"频率=initial，上次执行于 {last_date}，不再自动重跑"

    if frequency == "monthly":
        if last_date.year == today.year and last_date.month == today.month:
            return False, f"本月已于 {last_date} 执行过回测"
        return True, f"上月回测于 {last_date}，本月未执行"

    if frequency == "quarterly":
        last_q = (last_date.month - 1) // 3
        cur_q = (today.month - 1) // 3
        if last_date.year == today.year and last_q == cur_q:
            return False, f"本季度已于 {last_date} 执行过回测"
        return True, f"上季度回测于 {last_date}，本季度未执行"

    return True, f"未知频率 {frequency}，执行回测"


def record_run(
    engine: Any,
    frequency: str,
    lookback_days: int,
    out_of_sample_days: int,
    initial_cash: float,
    params: dict[str, float],
    sharpe: float,
    total_return: float,
    max_drawdown: float,
    status: str = "success",
) -> None:
    sql = text(f"""
        INSERT INTO {TABLE}
            (run_time, frequency, lookback_days, out_of_sample_days,
             initial_cash, params, sharpe, total_return, max_drawdown, status)
        VALUES
            (NOW(), :frequency, :lookback_days, :out_of_sample_days,
             :initial_cash, :params::jsonb, :sharpe, :total_return, :max_drawdown, :status)
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "frequency": frequency,
            "lookback_days": lookback_days,
            "out_of_sample_days": out_of_sample_days,
            "initial_cash": initial_cash,
            "params": json.dumps(params, ensure_ascii=False),
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "status": status,
        })
    logger.info(f"回测记录已写入 {TABLE}")
