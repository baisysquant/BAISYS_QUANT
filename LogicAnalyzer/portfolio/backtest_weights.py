from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ewma_cov(returns: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance matrix (RiskMetrics standard λ=0.94).

    Args:
        returns: (T, N) return array, more recent rows are later in time.
        lambda_: decay factor; lower = faster decay.

    Returns:
        (N, N) covariance matrix.
    """
    T, N = returns.shape
    weights = np.array([(1 - lambda_) * lambda_ ** (T - 1 - t) for t in range(T)])
    weights /= weights.sum()
    mean = np.average(returns, axis=0, weights=weights)
    centered = returns - mean
    cov = np.zeros((N, N))
    for t in range(T):
        cov += weights[t] * np.outer(centered[t], centered[t])
    return cov


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
    cov: np.ndarray, expected_returns: np.ndarray, max_weight: float = 0.1,
    risk_aversion: float = 1.0, turnover_penalty: float = 0.0,
    prev_weights: np.ndarray | None = None,
    industry_groups: list[list[int]] | None = None,
    max_industry_weight: float = 0.3,
) -> np.ndarray:
    """均值方差优化 — 最大化 Sharpe，支持行业约束和换手率惩罚。

    Args:
        cov: 协方差矩阵 (n x n)
        expected_returns: 预期收益向量 (n,)
        max_weight: 单票权重上限
        risk_aversion: 风险厌恶系数
        turnover_penalty: 换手率惩罚系数
        prev_weights: 上一期权重，用于计算换手率
        industry_groups: [[行业A的索引列表], [行业B的索引列表], ...]
        max_industry_weight: 单行业权重上限

    Returns:
        优化后的权重向量
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([min(1.0, max_weight)])

    from scipy.optimize import minimize

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        port_ret = w @ expected_returns
        obj = -(port_ret - 0.5 * risk_aversion * port_var)
        if turnover_penalty > 0 and prev_weights is not None:
            turnover = np.sum(np.abs(w - prev_weights))
            obj += turnover_penalty * turnover
        return obj

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if industry_groups:
        for grp in industry_groups:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, g=grp: max_industry_weight - np.sum(w[g]),
            })
    bounds = [(0, max_weight)] * n
    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 500})
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
    risk_none_multiplier: float = 1.0,
    prev_weights: dict[str, float] | None = None,
    industry_col: str | None = None,
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
        risk_map = {"NONE": risk_none_multiplier, "LOW": 1.5, "MEDIUM": 3.0, "HIGH": 5.0, "D": 8.0}
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

    cov = _ewma_cov(ret.values, lambda_=0.94)
    n_assets = len(common_syms)

    if method == "risk_parity":
        w = risk_parity_weights(cov, max_weight)
    elif method == "min_variance":
        w = min_variance_weights(cov, max_weight)
    elif method == "mean_variance":
        expected_ret = ret.mean().values
        # 行业分组约束
        ind_groups: list[list[int]] | None = None
        if industry_col and industry_col in bars.columns:
            last_bar = bars[bars["symbol"].isin(common_syms)].drop_duplicates(subset="symbol")
            ind_map: dict[str, list[int]] = {}
            for idx, sym in enumerate(common_syms):
                row = last_bar[last_bar["symbol"] == sym]
                if not row.empty:
                    ind = str(row.iloc[0].get(industry_col, "未知"))
                    ind_map.setdefault(ind, []).append(idx)
            ind_groups = list(ind_map.values())
        # 上一期权重（用于换手率惩罚）
        _prev_np = None
        if prev_weights:
            _prev_np = np.array([prev_weights.get(s, 0.0) for s in common_syms])
        w = mean_variance_weights(
            cov, expected_ret, max_weight,
            turnover_penalty=0.001, prev_weights=_prev_np,
            industry_groups=ind_groups,
        )
    else:
        w = np.full(n_assets, 1.0 / n_assets)

    return dict(zip(common_syms, w))
