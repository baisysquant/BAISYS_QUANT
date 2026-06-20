from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def compute_risk_metrics(equity_curve: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(equity_curve)
    if df.empty or "portfolio_value" not in df.columns or len(df) < 2:
        return {}

    vals = df["portfolio_value"].values.astype(float)
    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    n = len(returns)
    if n < 2:
        return {}

    total_ret = vals[-1] / vals[0] - 1
    ann_factor = 252
    mu = returns.mean() * ann_factor
    sigma = returns.std() * math.sqrt(ann_factor)

    sharpe = mu / sigma if sigma > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() * math.sqrt(ann_factor) if len(downside) > 0 else 1e-10
    sortino = mu / downside_std if downside_std > 0 else 0.0

    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    max_dd = float(dd.min())

    peak_idx = np.argmax(peak)
    trough_idx = np.argmin(vals[peak_idx:]) + peak_idx if peak_idx < len(vals) - 1 else peak_idx
    dd_duration = int(trough_idx - peak_idx) if trough_idx > peak_idx else 0

    sorted_ret = np.sort(returns)
    var_95 = float(np.percentile(sorted_ret, 5))
    cvar_95 = float(sorted_ret[sorted_ret <= var_95].mean()) if np.any(sorted_ret <= var_95) else var_95

    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": round(total_ret, 6),
        "annual_return": round(mu, 6),
        "annual_vol": round(sigma, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration": dd_duration,
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
    }


def compute_trade_metrics(trade_log: list[dict[str, Any]]) -> dict[str, Any]:
    buys = [t for t in trade_log if t.get("action") == "buy"]
    sells = [t for t in trade_log if t.get("action") == "sell"]

    if not buys or not sells:
        return {"total_trades": 0}

    total = len(buys) + len(sells)
    paired = min(len(buys), len(sells))

    pnl = [s.get("price", 0) - b.get("price", 0) for b, s in zip(buys[:paired], sells[:paired])]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]

    win_rate = len(wins) / len(pnl) if pnl else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1e-10
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float("inf")

    return {
        "total_trades": total,
        "buy_trades": len(buys),
        "sell_trades": len(sells),
        "win_rate": round(win_rate, 4),
        "avg_win": round(float(avg_win), 4),
        "avg_loss": round(float(avg_loss), 4),
        "profit_factor": round(float(profit_factor), 4),
    }
