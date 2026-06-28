from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def risk_parity_weights(cov: np.ndarray, max_weight: float = 0.1) -> np.ndarray:
    """风险平价权重 — 每项资产对组合风险的贡献相等。"""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([min(1.0, max_weight)])

    x = np.ones(n) / n
    for _ in range(100):
        sigma = np.sqrt(x @ cov @ x)
        if sigma < 1e-12:
            break
        mrc = cov @ x / sigma
        target = np.mean(mrc)
        x = x * (target / mrc)
        x = np.clip(x, 0, max_weight)
        x /= x.sum()
    return x


def min_variance_weights(cov: np.ndarray, max_weight: float = 0.1) -> np.ndarray:
    """最小方差组合。"""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([min(1.0, max_weight)])

    from scipy.optimize import minimize

    def objective(w: np.ndarray) -> float:
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, max_weight)] * n
    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    return np.full(n, 1.0 / n)


def mean_variance_weights(
    cov: np.ndarray, expected_returns: np.ndarray, max_weight: float = 0.1, risk_aversion: float = 1.0
) -> np.ndarray:
    """均值方差优化 — 最大化 Sharpe。"""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([min(1.0, max_weight)])

    from scipy.optimize import minimize

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        port_ret = w @ expected_returns
        return -(port_ret - 0.5 * risk_aversion * port_var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, max_weight)] * n
    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    return np.full(n, 1.0 / n)


def allocate_weights(
    bars: pd.DataFrame,
    method: str = "risk_parity",
    max_weight: float = 0.1,
    lookback: int = 60,
    entry_col: str = "进场评分",
    risk_col: str = "风险等级",
) -> dict[str, float]:
    """在给定评分下分配组合权重。

    Args:
        bars: 当日所有股票的 bar 数据（含历史），已预计算指标。
        method: risk_parity / min_variance / mean_variance / score_weighted
        max_weight: 单票权重上限
        lookback: 协方差估计的回看天数
    """
    if bars.empty:
        return {}

    candidates = bars[bars[entry_col] >= 60].copy()
    if candidates.empty:
        return {}

    if method == "score_weighted":
        risk_map = {"NONE": 1.0, "LOW": 1.5, "MEDIUM": 3.0, "HIGH": 5.0, "D": 8.0}
        weights = {}
        for _, r in candidates.iterrows():
            risk = risk_map.get(str(r.get(risk_col, "MEDIUM")).upper(), 3.0)
            weights[str(r["symbol"])] = min(1.0 / risk, max_weight)
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
        return weights

    symbols = candidates["symbol"].tolist()
    n = len(symbols)

    close_data = bars.pivot_table(index="trade_date", columns="symbol", values="close")

    ret = close_data.pct_change().tail(lookback).dropna(how="all")
    common_syms = [s for s in symbols if s in ret.columns]
    if not common_syms:
        return {s: 1.0 / n for s in symbols}

    ret = ret[common_syms].dropna()
    if ret.empty or ret.shape[1] < 2:
        return {s: 1.0 / len(common_syms) for s in common_syms}

    cov = ret.cov().values
    n_assets = len(common_syms)

    if method == "risk_parity":
        w = risk_parity_weights(cov, max_weight)
    elif method == "min_variance":
        w = min_variance_weights(cov, max_weight)
    elif method == "mean_variance":
        expected_ret = ret.mean().values
        w = mean_variance_weights(cov, expected_ret, max_weight)
    else:
        w = np.full(n_assets, 1.0 / n_assets)

    return dict(zip(common_syms, w))
