from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RiskExposure:
    """组合风险暴露分解。"""

    industry_exposure: dict[str, float]          # 行业 → 权重
    style_exposure: dict[str, float]             # 风格因子 → 暴露值
    tracking_error: float = 0.0                  # 年化跟踪误差
    active_risk: float = 0.0                     # 主动风险
    industry_concentration: float = 0.0          # HHI 行业集中度


_STYLE_FACTORS = ["size", "beta", "momentum", "volatility", "value", "liquidity"]


def compute_industry_exposure(
    portfolio_weights: pd.Series,
    industry_labels: pd.Series,
) -> dict[str, float]:
    """计算组合的行业暴露（各行业权重之和）。"""
    df = pd.DataFrame({"weight": portfolio_weights, "industry": industry_labels})
    return df.groupby("industry")["weight"].sum().to_dict()


def compute_style_exposure(
    portfolio_weights: pd.Series,
    stock_features: pd.DataFrame,
) -> dict[str, float]:
    """计算组合在风格因子上的加权平均暴露。

    Args:
        portfolio_weights: index=symbol, values=weight
        stock_features: index=symbol, columns=style_factor_names
    """
    common = portfolio_weights.index.intersection(stock_features.index)
    if common.empty:
        return {f: 0.0 for f in _STYLE_FACTORS}
    w = portfolio_weights[common]
    w = w / w.sum()
    exposures = {}
    for col in stock_features.columns:
        exposures[col] = float((w * stock_features.loc[common, col]).sum())
    return exposures


def compute_tracking_error(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> float:
    """年化跟踪误差 = sqrt((w_p - w_b)^T Σ (w_p - w_b)) * sqrt(252)。"""
    active = portfolio_weights.subtract(benchmark_weights, fill_value=0)
    common = active.index.intersection(cov_matrix.index).intersection(cov_matrix.columns)
    if len(common) < 2:
        return 0.0
    active = active[common].values
    sub_cov = cov_matrix.loc[common, common].values
    te = float(np.sqrt(active @ sub_cov @ active)) * np.sqrt(252)
    return te


def industry_hhi(industry_exposure: dict[str, float]) -> float:
    """行业 HHI 集中度（Herfindahl Index），越高越集中。"""
    weights = np.array(list(industry_exposure.values()))
    return float((weights ** 2).sum())


def apply_tracking_error_constraint(
    raw_weights: pd.Series,
    benchmark_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    max_te: float = 0.05,
    max_iter: int = 10,
) -> pd.Series:
    """通过逐步向基准收缩，将跟踪误差控制在 max_te 以内。"""
    if compute_tracking_error(raw_weights, benchmark_weights, cov_matrix) <= max_te:
        return raw_weights
    w = raw_weights.copy()
    lam = 0.0
    for _ in range(max_iter):
        lam += 0.1
        w = (1 - lam) * raw_weights + lam * benchmark_weights
        w = w.clip(0, 1)
        w = w / w.sum()
        te = compute_tracking_error(w, benchmark_weights, cov_matrix)
        if te <= max_te:
            break
    return w


def portfolio_risk_decomposition(
    portfolio_weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> dict[str, float]:
    """组合风险分解：边际贡献 + 总风险。"""
    common = portfolio_weights.index.intersection(cov_matrix.index).intersection(cov_matrix.columns)
    if len(common) < 2:
        return {}
    w = portfolio_weights[common].values
    sub_cov = cov_matrix.loc[common, common].values
    port_var = w @ sub_cov @ w
    port_vol = np.sqrt(port_var)
    mrc = sub_cov @ w / port_vol
    rc = w * mrc
    return {
        "portfolio_vol": float(port_vol * np.sqrt(252)),
        "marginal_risk_contrib": dict(zip(common, mrc)),
        "risk_contrib": dict(zip(common, rc)),
    }
