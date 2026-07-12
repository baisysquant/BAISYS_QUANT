from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _ann_factor() -> int:
    return 252


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    target_sr: float = 0.0,
) -> float:
    if n_obs <= 1:
        return 0.5

    sd = sharpe / math.sqrt(_ann_factor())
    td = target_sr / math.sqrt(_ann_factor())

    num = (sd - td) * math.sqrt(n_obs - 1)
    den = math.sqrt(1.0 - skew * sd + ((kurt - 1.0) / 4.0) * sd * sd)
    if den <= 1e-12:
        return 0.5

    return float(stats.norm.cdf(num / den))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    num_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    if num_trials <= 1:
        return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, 0.0)

    sigma_sr = 1.0 / math.sqrt(n_obs - 1) if n_obs > 1 else 1.0
    gamma_euler = 0.5772156649

    inv_n = 1.0 / num_trials
    try:
        z1 = float(stats.norm.ppf(1.0 - inv_n))
        z2 = float(stats.norm.ppf(1.0 - inv_n / math.e))
    except Exception:
        return 0.5

    e_max_sr = sigma_sr * ((1.0 - gamma_euler) * z1 + gamma_euler * z2)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, e_max_sr)


def compute_pbo(
    window_results: list[dict[Any, Any]],
    top_m: int = 5,
) -> float:
    if not window_results:
        return 0.5

    violations = 0
    total_windows = 0

    for w in window_results:
        ocs = w.get("oos_combos", [])
        if len(ocs) < 2:
            continue

        oos_sharpes = [
            c["oos_sharpe"]
            for c in ocs
            if c.get("oos_sharpe") is not None and not (isinstance(c["oos_sharpe"], float) and math.isnan(c["oos_sharpe"]))
        ]
        if len(oos_sharpes) < 2:
            continue

        median_oos = float(np.median(oos_sharpes))
        rank1 = next((c for c in ocs if c.get("is_rank") == 1), None)
        if rank1 is not None:
            sr = rank1.get("oos_sharpe")
            if sr is not None and not (isinstance(sr, float) and math.isnan(sr)):
                total_windows += 1
                if sr < median_oos:
                    violations += 1

    if total_windows == 0:
        return 0.5
    return violations / total_windows


def compute_dsr_from_equity_curve(
    equity_curve: list[dict[Any, Any]],
    num_trials: int,
) -> float:
    if len(equity_curve) < 2:
        return 0.5

    vals = pd.Series([e.get("portfolio_value", 0) for e in equity_curve]).values.astype(float)
    if len(vals) < 2:
        return 0.5

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    n = len(returns)
    if n < 2:
        return 0.5

    sharpe = float(returns.mean() / returns.std()) * math.sqrt(_ann_factor()) if returns.std() > 0 else 0.0
    skew = float(pd.Series(returns).skew())  # type: ignore[arg-type]
    kurt = float(pd.Series(returns).kurtosis()) + 3.0  # type: ignore[arg-type]

    return deflated_sharpe_ratio(sharpe, n, num_trials, skew, kurt)
