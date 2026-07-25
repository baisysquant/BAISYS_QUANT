from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from BackTrading._engine_legacy import EngineConfig, run_full_backtest, _run_single_backtest
from BackTrading.prepare import _build_params, prepare_backtest_data
from LogicAnalyzer.backtest_metrics import compute_risk_metrics


@dataclass
class SimTradeVerdict:
    """模拟交易验证结果与决策。"""

    sim_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    degradation: float = 0.0     # 1 - sim/oos，负值表示 sim 优于 oos
    promote: bool = False
    reason: str = ""


def validate_params(
    kline_df: pd.DataFrame,
    best_params: dict[str, float],
    oos_sharpe: float,
    sim_days: int = 20,
    config: Any | None = None,
    engine_cfg: EngineConfig | None = None,
) -> SimTradeVerdict:
    """用最近 sim_days 个交易日验证 best_params 的稳定性。

    Args:
        kline_df: 全量 K 线数据（含信号列或原始数据均可）。
        best_params: WFO 选出的最佳参数（flat dict，至少含 atr_stop_mult）。
        oos_sharpe: WFO 在样本外窗口上的 Sharpe。
        sim_days: 用于验证的最近交易日数。
        config: Config 实例（可选，用于构建结构化 params）。
        engine_cfg: EngineConfig 实例（可选，构建最终回测引擎）。

    Returns:
        SimTradeVerdict 包含决策和原因。
    """
    if best_params is None or oos_sharpe is None:
        return SimTradeVerdict(promote=False, reason="WFO 结果为空，跳过模拟验证")

    # 取最近 sim_days 个交易日
    dates = sorted(kline_df["trade_date"].unique())
    if len(dates) < sim_days + 20:
        return SimTradeVerdict(
            promote=True,
            reason=f"数据不足（{len(dates)} 个交易日），无法做模拟验证，直接放行",
        )
    sim_start_idx = len(dates) - sim_days
    sim_dates_set = set(dates[sim_start_idx:])

    # 准备信号 + 止损价
    if config is not None:
        from UtilsManager.ConfigParser import Config as _Cfg
        cfg = config if isinstance(config, _Cfg) else _Cfg()
        structured = _build_params(cfg)
        structured["scoring"].update(
            {k: v for k, v in best_params.items() if k in (
                "atr_stop_mult", "atr_t1_mult", "atr_t2_mult",
            )}
        )
        prepared = prepare_backtest_data(kline_df, params=structured, compute_exit_strategy=False, vectorized=True)
    else:
        prepared = prepare_backtest_data(kline_df, params=best_params, compute_exit_strategy=False, vectorized=True)

    # 验证期切片
    import pandas as _pd
    _prep = prepared
    if _pd.api.types.is_datetime64_any_dtype(_prep["trade_date"]):
        _prep = _prep.copy()
        _prep["trade_date"] = _prep["trade_date"].dt.strftime("%Y-%m-%d")
    sim_dates_str = {d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in sim_dates_set}
    sim_data = _prep[_prep["trade_date"].isin(sim_dates_str)].copy()
    if sim_data.empty:
        return SimTradeVerdict(promote=True, reason="模拟期数据为空，直接放行")

    # 按 best_params 的 atr_stop_mult 计算止损价
    stop_mult = best_params.get("atr_stop_mult")
    if stop_mult is not None and "ATR" in sim_data.columns:
        sim_data["止损价"] = sim_data["close"] - sim_data["ATR"] * stop_mult
    elif "止损价" not in sim_data.columns:
        sim_data["止损价"] = 0.0

    if engine_cfg is None:
        engine_cfg = EngineConfig(
            kelly_fraction=best_params.get("kelly_fraction", 0.25),
            position_a=best_params.get("position_a", 0.3),
            risk_none_multiplier=best_params.get("risk_none_multiplier", 1.0),
            atr_stop_mult=best_params.get("atr_stop_mult", 1.5),
        )

    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(sim_data, best_params, engine_cfg, tl, ec)
    risk = compute_risk_metrics(ec) or {}
    sim_sharpe = risk.get("sharpe_ratio", 0.0) or 0.0

    degradation = 1.0 - (sim_sharpe / oos_sharpe) if oos_sharpe > 0.01 else 0.0
    promote = sim_sharpe > 0.0 and degradation < 0.5

    if promote:
        reason = (
            f"模拟验证通过: sim_Sharpe={sim_sharpe:.2f} / oos_Sharpe={oos_sharpe:.2f}"
            f" 退化率={degradation:.0%} < 50%"
        )
    else:
        reason = (
            f"模拟验证失败: sim_Sharpe={sim_sharpe:.2f} / oos_Sharpe={oos_sharpe:.2f}"
            f" 退化率={degradation:.0%} >= 50%（或 sim Sharpe ≤ 0）"
        )

    logger.info(f"  {reason}")
    return SimTradeVerdict(
        sim_sharpe=sim_sharpe,
        oos_sharpe=oos_sharpe,
        degradation=degradation,
        promote=promote,
        reason=reason,
    )
