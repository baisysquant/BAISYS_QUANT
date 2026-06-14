"""
MACD 背离检测模块

从 MACDAnalyzer.py 提取，负责顶/底背离检测相关的纯计算逻辑。
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from LogicAnalyzer.SignalConstants import Divergence


def find_peaks_troughs(series: pd.Series, distance: int = 5) -> tuple[np.ndarray, np.ndarray]:
    peaks, _ = find_peaks(series, distance=distance)
    neg_series = -series
    troughs, _ = find_peaks(neg_series, distance=distance)
    return peaks, troughs


def adaptive_distance(series: pd.Series, base_distance: int = 10) -> int:
    n = len(series)
    if n < 20:
        return max(3, n // 4)
    price_range = series.max() - series.min()
    if price_range == 0:
        return base_distance
    volatility = series.diff().abs().mean() / price_range
    dynamic = max(3, int(base_distance * (1 + volatility * 10)))
    return min(dynamic, max(10, n // 5))


def calc_slope_linear(series: pd.Series, window: int = 3) -> float:
    if len(series) < 2:
        return 0.0
    y = series.iloc[-window:].values
    x = np.arange(len(y))
    if len(y) < 2:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return slope


def signal_with_decay(signal_type: str | None, signal_idx: int | None,
                       current_idx: int, half_life: int = 8) -> float:
    if signal_type is None or signal_idx is None:
        return 0.0
    bars_ago = max(current_idx - signal_idx, 0)
    decay = 0.5 ** (bars_ago / half_life)
    return decay


def detect_divergence_single_param(
    df: pd.DataFrame, price: pd.Series, indicator: pd.Series, distance: int = 25
) -> tuple[str | None, int | None, float]:
    current_idx = len(df) - 1
    adj_dist = adaptive_distance(indicator, base_distance=distance)
    peaks, troughs = find_peaks_troughs(indicator, distance=adj_dist)

    strength = 0.0
    # 顶背离：价格创新高，指标未创新高
    for p in reversed(peaks):
        if p < current_idx - adj_dist * 2:
            continue
        if price.iloc[p] > price.iloc[current_idx] * 0.98:
            continue
        if indicator.iloc[p] > indicator.iloc[current_idx]:
            continue
        price_ratio = price.iloc[current_idx] / price.iloc[p] - 1
        ind_ratio = 1 - indicator.iloc[current_idx] / indicator.iloc[p]
        strength = min(1.0, max(0, (price_ratio + ind_ratio) / 2))
        if strength > 0.15:
            return Divergence.TOP_DIVERGENCE, p, strength

    # 底背离：价格创新低，指标未创新低
    for t in reversed(troughs):
        if t < current_idx - adj_dist * 2:
            continue
        if price.iloc[t] < price.iloc[current_idx] * 1.02:
            continue
        if indicator.iloc[t] < indicator.iloc[current_idx]:
            continue
        price_ratio = 1 - price.iloc[current_idx] / price.iloc[t]
        ind_ratio = indicator.iloc[current_idx] / indicator.iloc[t] - 1
        strength = min(1.0, max(0, (price_ratio + ind_ratio) / 2))
        if strength > 0.15:
            return Divergence.BOTTOM_DIVERGENCE, t, strength

    return None, None, 0.0


def volume_confirmation(df: pd.DataFrame, signal_type: str | None, signal_idx: int | None) -> str:
    if signal_type is None or signal_idx is None:
        return "量价正常"
    recent_vol = df["volume"].iloc[-5:].mean()
    hist_vol = df["volume"].iloc[signal_idx:signal_idx + 5].mean() if signal_idx < len(df) - 5 else recent_vol
    if hist_vol == 0:
        return "量价正常"
    vol_ratio = recent_vol / hist_vol
    if signal_type == Divergence.BOTTOM_DIVERGENCE:
        return "底背离：量能放大（vol_ratio >= 1.2）→ 确认买入" if vol_ratio >= 1.2 else f"底背离：量能不足（vol_ratio={vol_ratio:.2f}）→ 需等待"
    else:
        return "顶背离：量能萎缩（vol_ratio <= 0.8）→ 确认卖出" if vol_ratio <= 0.8 else f"顶背离：量能正常（vol_ratio={vol_ratio:.2f}）→ 需观察"





# ── 共享工具函数（原 MACDHelpers.py） ────────────────────────────────────


def slope_analysis(series: pd.Series, window: int = 5) -> dict:
    y = series.iloc[-window:].values
    x = np.arange(len(y), dtype=float)
    if len(y) < 3:
        return {"slope": 0.0, "r2": 0.0, "trend": "N/A"}
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    if r2 > 0.7 and slope > 0:
        trend = "明确上行"
    elif r2 > 0.7 and slope < 0:
        trend = "明确下行"
    else:
        trend = "震荡"
    return {"slope": slope, "r2": r2, "trend": trend}

