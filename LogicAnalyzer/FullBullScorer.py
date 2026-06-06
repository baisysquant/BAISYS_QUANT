"""
完全多头评分模块

从 MACDAnalyzer.py 提取，负责完全多头排列评分、K线形态评分等。
"""

import numpy as np
import pandas as pd

from LogicAnalyzer.MACDDivergence import slope_analysis as calc_slope, calculate_momentum_score
from LogicAnalyzer.SignalConstants import (
    MACDSignals, MACDMomentum, Divergence, TrendLevels,
    Conclusion, FullBullScoring, CombinedSignal, KLineLevels
)


def score_kline_pattern(kline_signals: list[dict]) -> int:
    score = 0
    if not kline_signals:
        return 0
    for d in kline_signals:
        if d.get("direction") == "看涨":
            if d.get("level") == "强反转":
                score += 5
            elif d.get("level") == "中反转":
                score += 2
            else:
                score += 1
        else:
            if d.get("level") == "强反转":
                score -= 3
            elif d.get("level") == "中反转":
                score -= 1
    return max(-10, min(10, score))


def volume_price_trend_score(df: pd.DataFrame, window: int = 5) -> str:
    if len(df) < window + 1:
        return "数据不足"
    price = df["close"].iloc[-window:].values
    volume = df["volume"].iloc[-window:].values
    pct_change = (price[-1] / price[0] - 1) * 100
    vol_slope = np.polyfit(np.arange(window), volume, 1)[0]
    vol_trend = vol_slope / (volume.mean() + 1e-9) * 100
    if pct_change > 0 and vol_trend > 0:
        return f"量价齐升 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})"
    elif pct_change > 0:
        return f"价涨量缩 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})"
    elif vol_trend > 0:
        return f"放量下跌 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})"
    else:
        return f"缩量下跌 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})"


def backtest_signal_winrate(df: pd.DataFrame, signal_col: str, signal_value: str,
                            forward_bars: int = 5) -> dict:
    result = {"win_count": 0, "loss_count": 0, "total": 0, "winrate": 0.0}
    if signal_col not in df.columns or "close" not in df.columns:
        return result
    series = df[signal_col]
    close = df["close"].values
    indices = series[series == signal_value].index
    wins = 0
    losses = 0
    for idx in indices:
        pos = df.index.get_loc(idx)
        if pos + forward_bars < len(close):
            ret = (close[pos + forward_bars] / close[pos] - 1) * 100
            if ret > 0:
                wins += 1
            else:
                losses += 1
    total = wins + losses
    if total > 0:
        result.update(win_count=wins, loss_count=losses, total=total, winrate=round(wins / total, 3))
    return result


def detect_market_regime(df: pd.DataFrame, dif_col: str, dea_col: str) -> dict:
    if len(df) < 20 or dif_col not in df.columns or dea_col not in df.columns:
        return {"regime": "unknown", "description": "数据不足", "score_adjust": 1.0}
    dif = df[dif_col].values
    dea = df[dea_col].values
    macd_hist = 2 * (dif - dea)
    above_zero = np.mean(dif[-10:] > 0)
    hist_trend = np.mean(np.diff(macd_hist[-10:]))
    strength = np.std(macd_hist[-20:]) / (np.mean(np.abs(macd_hist[-20:])) + 1e-9)

    if above_zero > 0.8 and hist_trend > 0:
        regime = "strong_bull"
        desc = "强势多头"
        adjust = 1.2
    elif above_zero > 0.6:
        regime = "bull"
        desc = "偏多"
        adjust = 1.1
    elif above_zero < 0.2:
        regime = "bear"
        desc = "偏空"
        adjust = 0.9
    else:
        regime = "oscillation"
        desc = "震荡"
        adjust = 1.0
    if strength > 2.0:
        desc += " (高波动)"
        adjust *= 0.95
    return {"regime": regime, "description": desc, "score_adjust": adjust}


def analyze_full_bull(
    df: pd.DataFrame,
    second_params: tuple[int, int, int],
    second_period_name: str,
    custom_macd_func: callable,
    divergence_func: callable,
    config,
    weights: dict[str, int] | None = None,
    thresholds: dict[str, int] | None = None,
) -> dict:
    if weights is None:
        weights = {"零轴条件": 20, "战略金叉": 20, "战术金叉": 15,
                   "动能": 20, "DIF斜率": 15, "背离信号": 10, "量价配合": 10}
    if thresholds is None:
        thresholds = {}

    full_bull_th = thresholds.get("full_bull", config.FULL_BULL_THRESHOLD)
    accel_th = thresholds.get("trend_acceleration", config.TREND_ACCELERATION_THRESHOLD)
    osc_th = thresholds.get("trend_oscillation", config.TREND_OSCILLATION_THRESHOLD)

    scores: dict[str, tuple[str, int]] = {}
    state = {"detail": ""}

    # 零轴条件
    dif_12269_val = df["DIF_12269"].iloc[-1]
    w_zero = weights.get("零轴条件", 20)
    if dif_12269_val > 0:
        scores["零轴条件"] = ("DIF > 0（多头主导）", w_zero)
    else:
        scores["零轴条件"] = ("DIF < 0（空头主导）", -w_zero)

    # 慢速 MACD
    slow_detail = df.get(f"MACD_12269_SIGNAL_DETAIL", pd.Series(dtype=str)).iloc[-1] if f"MACD_12269_SIGNAL_DETAIL" in df.columns else ""
    slow_dif = df.get("DIF_12269", pd.Series(dtype=float))
    slow_dea = df.get("DEA_12269", pd.Series(dtype=float))
    w_strat = weights.get("战略金叉", 20)

    if slow_detail == MACDSignals.GOLDEN_CROSS_ABOVE_ZERO:
        scores["战略金叉"] = ("12269 零轴上金叉（最强信号）", w_strat)
    elif slow_detail == MACDSignals.GOLDEN_CROSS_BELOW_ZERO:
        scores["战略金叉"] = ("12269 零轴下金叉（注意假突破）", w_strat // 2)
    elif len(slow_dif) > 0 and len(slow_dea) > 0 and slow_dif.iloc[-1] > slow_dea.iloc[-1]:
        scores["战略金叉"] = ("12269 多头持续（DIF > DEA）", w_strat // 3)
    else:
        scores["战略金叉"] = ("12269 空头/死叉", 0)

    # 快速 MACD
    fast_detail_col = f"MACD_{second_period_name}_SIGNAL_DETAIL"
    fast_detail = df.get(fast_detail_col, pd.Series(dtype=str)).iloc[-1] if fast_detail_col in df.columns else ""
    w_tact = weights.get("战术金叉", 15)

    if fast_detail == MACDSignals.GOLDEN_CROSS_ABOVE_ZERO:
        scores["战术金叉"] = (f"{second_period_name} 零轴上金叉", w_tact)
    elif fast_detail == MACDSignals.GOLDEN_CROSS_BELOW_ZERO:
        scores["战术金叉"] = (f"{second_period_name} 零轴下金叉（注意假突破）", w_tact // 2)
    else:
        fast_dif_col = f"DIF_{second_period_name}"
        fast_dea_col = f"DEA_{second_period_name}"
        if fast_dif_col in df.columns and fast_dea_col in df.columns:
            if df[fast_dif_col].iloc[-1] > df[fast_dea_col].iloc[-1]:
                scores["战术金叉"] = (f"{second_period_name} 多头持续", w_tact // 3)
            else:
                scores["战术金叉"] = (f"{second_period_name} 空头/死叉", 0)

    # 动能评分
    mom_score_val = calculate_momentum_score(df, f"DIF_12269", f"DEA_12269",
                                              f"DIF_{second_period_name}", f"DEA_{second_period_name}")
    scores["动能"] = (f"动能评分: {mom_score_val}", mom_score_val)

    # DIF 斜率
    slope_val = calc_slope(df["DIF_12269"], window=5)
    w_slope = weights.get("DIF斜率", 15)
    slope_score = int(min(15, max(-15, slope_val["slope"] * 100)))
    scores["DIF斜率"] = (f"DIF斜率: {slope_val['slope']:.4f} (R²={slope_val['r2']:.2f}, {slope_val['trend']})", slope_score)

    # 背离信号
    w_div = weights.get("背离信号", 10)
    div_result = divergence_func(df, distance_slow=25, distance_fast=12,
                                 second_period_name=second_period_name)
    div_score = 0
    if Divergence.BOTTOM_DIVERGENCE in str(div_result.get("combined_signal", "")):
        div_score = w_div
    elif Divergence.TOP_DIVERGENCE in str(div_result.get("combined_signal", "")):
        div_score = -w_div
    scores["背离信号"] = (div_result.get("combined_signal", "无背离"), div_score)

    # 量价配合
    vp_desc = volume_price_trend_score(df)
    w_vp = weights.get("量价配合", 10)
    vp_score = int(w_vp * 0.6) if "量价齐升" in vp_desc else -int(w_vp * 0.5)
    scores["量价配合"] = (vp_desc, vp_score)

    total = sum(v[1] for v in scores.values())
    max_possible = sum(max(0, v) for k, v in weights.items())
    base_score = max(0, min(100, int(total / max_possible * 100))) if max_possible > 0 else 0

    if base_score >= full_bull_th:
        conclusion = "完全多头 (强烈买入)"
    elif base_score >= accel_th:
        conclusion = "偏多 (可逢低布局)"
    elif base_score >= osc_th:
        conclusion = "多空拉锯 (观望为主)"
    else:
        conclusion = "空头/弱势 (回避)"

    return {
        "score": base_score,
        "details": {k: {"desc": v[0], "score": v[1]} for k, v in scores.items()},
        "conclusion": conclusion,
    }



