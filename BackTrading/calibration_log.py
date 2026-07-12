from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
from loguru import logger
from sqlalchemy import text

TABLE = "backtest_calibration_log"


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              SERIAL PRIMARY KEY,
    run_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    frequency       VARCHAR(16) NOT NULL,
    backtest_start_date VARCHAR(8) NOT NULL,
    out_of_sample_days INT        NOT NULL,
    initial_cash    NUMERIC(14,2) NOT NULL,
    params          JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    sharpe          NUMERIC(8,4),
    sortino         NUMERIC(8,4),
    calmar          NUMERIC(8,4),
    total_return    NUMERIC(8,4),
    annual_return   NUMERIC(8,4),
    annual_vol      NUMERIC(8,4),
    max_drawdown    NUMERIC(8,4),
    max_drawdown_duration INT DEFAULT 0,
    var_95          NUMERIC(8,4),
    cvar_95         NUMERIC(8,4),
    win_rate        NUMERIC(6,4),
    profit_factor   NUMERIC(10,4),
    total_trades    INT DEFAULT 0,
    status          VARCHAR(16) NOT NULL DEFAULT 'success',
    git_commit      VARCHAR(12) DEFAULT '',
    config_hash     VARCHAR(8)  DEFAULT '',
    pbo             NUMERIC(6,4) DEFAULT 0.0,
    dsr             NUMERIC(6,4) DEFAULT 0.0,
    num_trials      INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_{TABLE}_run_time ON {TABLE} (run_time DESC);
"""


def ensure_table(engine: Any) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    # 迁移：兼容旧表
    for col, typ in [
        ("lookback_days", None),
        ("sortino", "NUMERIC(8,4)"),
        ("calmar", "NUMERIC(8,4)"),
        ("annual_return", "NUMERIC(8,4)"),
        ("annual_vol", "NUMERIC(8,4)"),
        ("max_drawdown_duration", "INT DEFAULT 0"),
        ("var_95", "NUMERIC(8,4)"),
        ("cvar_95", "NUMERIC(8,4)"),
        ("win_rate", "NUMERIC(6,4)"),
        ("profit_factor", "NUMERIC(10,4)"),
        ("total_trades", "INT DEFAULT 0"),
        ("git_commit", "VARCHAR(12) DEFAULT ''"),
        ("config_hash", "VARCHAR(8) DEFAULT ''"),
        ("pbo", "NUMERIC(6,4) DEFAULT 0.0"),
        ("dsr", "NUMERIC(6,4) DEFAULT 0.0"),
        ("num_trials", "INT DEFAULT 0"),
    ]:
        try:
            if col == "lookback_days":
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {TABLE} RENAME COLUMN lookback_days TO lookback_days_old"))
                    conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN backtest_start_date VARCHAR(8)"))
                    conn.execute(text(f"UPDATE {TABLE} SET backtest_start_date = lookback_days_old::TEXT"))
                    conn.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN lookback_days_old"))
            else:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS {col} {typ}"))
        except Exception:
            pass


def get_last_run(engine: Any) -> dict[str, Any] | None:
    sql = text(f"""
        SELECT run_time, frequency, backtest_start_date, out_of_sample_days,
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


def _pyval(v: Any) -> Any:
    """numpy → 原生 Python 类型，避免 psycopg2 序列化成 np.float64(...) 导致 SQL 报错。"""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {k: _pyval(v) for k, v in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_pyval(x) for x in v)
    return v


def record_run(
    engine: Any,
    frequency: str,
    backtest_start_date: str,
    out_of_sample_days: int,
    initial_cash: float,
    params: dict[str, float],
    sharpe: float,
    total_return: float,
    max_drawdown: float,
    status: str = "success",
    extra_metrics: dict[str, Any] | None = None,
    git_commit: str = "",
    config_hash: str = "",
) -> None:
    metrics = dict(extra_metrics or {})
    sortino = metrics.pop("sortino_ratio", None) or metrics.get("sortino", 0)
    calmar = metrics.pop("calmar_ratio", None) or metrics.get("calmar", 0)
    var_95 = metrics.pop("var_95", 0)
    cvar_95 = metrics.pop("cvar_95", 0)
    win_rate = metrics.pop("win_rate", 0)
    profit_factor = metrics.pop("profit_factor", 0)
    total_trades = metrics.pop("total_trades", 0)
    pbo = metrics.pop("pbo", 0.0)
    dsr = metrics.pop("dsr", 0.0)
    num_trials = metrics.pop("num_trials", 0)

    sql = text(f"""
        INSERT INTO {TABLE}
            (run_time, frequency, backtest_start_date, out_of_sample_days,
             initial_cash, params, sharpe, total_return, max_drawdown, status,
             sortino, calmar, var_95, cvar_95, win_rate, profit_factor, total_trades,
             git_commit, config_hash, pbo, dsr, num_trials)
        VALUES
            (NOW(), :frequency, :backtest_start_date, :out_of_sample_days,
             :initial_cash, CAST(:params AS jsonb), :sharpe, :total_return, :max_drawdown, :status,
             :sortino, :calmar, :var_95, :cvar_95, :win_rate, :profit_factor, :total_trades,
             :git_commit, :config_hash, :pbo, :dsr, :num_trials)
    """)
    with engine.begin() as conn:
        conn.execute(sql, _pyval({
            "frequency": frequency,
            "backtest_start_date": backtest_start_date,
            "out_of_sample_days": out_of_sample_days,
            "initial_cash": initial_cash,
            "params": json.dumps(_pyval(params), ensure_ascii=False),
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "status": status,
            "sortino": sortino,
            "calmar": calmar,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "git_commit": git_commit,
            "config_hash": config_hash,
            "pbo": pbo,
            "dsr": dsr,
            "num_trials": num_trials,
        }))
    logger.info(f"回测记录已写入 {TABLE}")
