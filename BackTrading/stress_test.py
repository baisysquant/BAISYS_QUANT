from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from BackTrading._engine_legacy import _run_single_backtest
from BackTrading.domain.models import EngineConfig
from BackTrading.prepare import _build_params, prepare_backtest_data

# 历史极端场景（基于已知 A 股危机时段）
CRISIS_SCENARIOS = {
    "2015_崩盘": {
        "start": "2015-06-12",
        "end": "2015-08-26",
        "description": "2015 年股灾（去杠杆 + 熔断前）",
    },
    "2016_熔断": {
        "start": "2016-01-04",
        "end": "2016-01-28",
        "description": "2016 年熔断",
    },
    "2018_单边熊": {
        "start": "2018-01-29",
        "end": "2018-12-28",
        "description": "2018 年贸易摩擦单边下跌",
    },
    "2020_疫情": {
        "start": "2020-01-20",
        "end": "2020-03-19",
        "description": "2020 年新冠冲击",
    },
    "2022_三月": {
        "start": "2022-03-01",
        "end": "2022-04-26",
        "description": "2022 年 3 月流动性危机",
    },
    "2024_微盘": {
        "start": "2024-01-02",
        "end": "2024-02-05",
        "description": "2024 年微盘股崩盘（量化踩踏）",
    },
}


def run_stress_tests(
    kline_df: pd.DataFrame,
    engine_cfg: EngineConfig,
    params: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """对历史极端场景逐一回测，报告回撤和 Sharpe。

    Args:
        kline_df: 全量 K 线数据
        engine_cfg: 引擎配置
        params: WFO 最佳参数

    Returns:
        {scenario_name: {sharpe, max_drawdown, total_return, ...}}
    """
    # 准备信号
    structured = _build_params(type("cfg", (), {})(), params)
    prepared = prepare_backtest_data(kline_df, params=structured, compute_exit_strategy=True, vectorized=True)

    stop_mult = params.get("atr_stop_mult", 2.0)
    if "ATR" in prepared.columns:
        prepared["止损价"] = prepared["close"] - prepared["ATR"] * stop_mult
    else:
        prepared["止损价"] = 0.0

    date_range = prepared["trade_date"].astype(str)

    results = {}
    for name, scenario in CRISIS_SCENARIOS.items():
        mask = (date_range >= scenario["start"]) & (date_range <= scenario["end"])
        scene_data = prepared[mask].copy()
        if scene_data.empty:
            logger.info(f"  压力测试 [{name}] 无数据，跳过")
            continue

        tl: list[dict[str, Any]] = []
        ec: list[dict[str, Any]] = []
        _run_single_backtest(scene_data, params, engine_cfg, tl, ec)

        from LogicAnalyzer.backtest_metrics import compute_risk_metrics
        risk = compute_risk_metrics(ec) or {}

        results[name] = {
            "sharpe": risk.get("sharpe_ratio", 0),
            "total_return": risk.get("total_return", 0),
            "annual_return": risk.get("annual_return", 0),
            "max_drawdown": risk.get("max_drawdown", 0),
            "max_drawdown_duration": risk.get("max_drawdown_duration", 0),
            "annual_vol": risk.get("annual_vol", 0),
            "trading_days": len(scene_data["trade_date"].unique()),
        }
        logger.info(
            f"  压力测试 [{name}]: "
            f"Return={results[name]['total_return']:.2%}, "
            f"MaxDD={results[name]['max_drawdown']:.2%}, "
            f"Vol={results[name]['annual_vol']:.2%}, "
            f"Days={results[name]['trading_days']}"
        )

    return results
