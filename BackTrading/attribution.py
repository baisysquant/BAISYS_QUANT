from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ── Brinson 归因 ──


# ── 因子暴露归因（截面回归） ──


@dataclass
class FactorExposureResult:
    """因子暴露与显著性。"""

    exposures: dict[str, float]       # 因子名 → 暴露系数
    t_stats: dict[str, float]         # 因子名 → t 统计量
    p_values: dict[str, float]        # 因子名 → p 值
    rsquared: float = 0.0
    adj_rsquared: float = 0.0


def factor_exposure(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    add_constant: bool = True,
) -> FactorExposureResult:
    """通过 OLS 时间序列回归估算组合对因子的暴露。

    r_p = alpha + beta_1 * f_1 + beta_2 * f_2 + ... + epsilon

    Args:
        portfolio_returns: 组合日收益率序列，index=trade_date
        factor_returns: 因子日收益率，index=trade_date, columns=factor_name
        add_constant: 是否加入截距项（alpha）

    Returns:
        FactorExposureResult 包含各因子暴露系数及显著性。
    """
    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    if len(common_idx) < 10:
        return FactorExposureResult(exposures={}, t_stats={}, p_values={})

    y = portfolio_returns.reindex(common_idx).values
    X = factor_returns.reindex(common_idx).values
    names = list(factor_returns.columns)

    if add_constant:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["Alpha"] + names

    n, k = X.shape
    if n <= k:
        return FactorExposureResult(exposures={}, t_stats={}, p_values={})

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        mse = resid @ resid / (n - k)
        se = np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))
        t_vals = beta / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=n - k))

        ss_res = resid @ resid
        ss_tot = (y - y.mean()) @ (y - y.mean())
        rsq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        adj_rsq = 1.0 - (1.0 - rsq) * (n - 1) / (n - k)

        return FactorExposureResult(
            exposures=dict(zip(names, beta)),
            t_stats=dict(zip(names, t_vals)),
            p_values=dict(zip(names, p_vals)),
            rsquared=rsq,
            adj_rsquared=adj_rsq,
        )
    except np.linalg.LinAlgError:
        return FactorExposureResult(exposures={}, t_stats={}, p_values={})
