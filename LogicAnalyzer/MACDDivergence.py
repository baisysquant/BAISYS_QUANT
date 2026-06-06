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
        return f"底背离：量能放大（vol_ratio >= 1.2）→ 确认买入" if vol_ratio >= 1.2 else f"底背离：量能不足（vol_ratio={vol_ratio:.2f}）→ 需等待"
    else:
        return f"顶背离：量能萎缩（vol_ratio <= 0.8）→ 确认卖出" if vol_ratio <= 0.8 else f"顶背离：量能正常（vol_ratio={vol_ratio:.2f}）→ 需观察"


def detect_combined_divergence(
    df: pd.DataFrame,
    distance_slow: int = 25,
    distance_fast: int = 12,
    recent_window: int = 5,
    decay_half_life: int = 8,
    second_period_name: str = "6135",
) -> dict:
    current_idx = len(df) - 1
    slow_result = detect_divergence_single_param(df, df["close"], df["DIF_12269"], distance=distance_slow)
    div_12269, idx_12269, str_12269 = slow_result

    dif_col_second = f"DIF_{second_period_name}"
    if dif_col_second in df.columns:
        fast_result = detect_divergence_single_param(df, df["close"], df[dif_col_second], distance=distance_fast)
        div_second, idx_second, str_second = fast_result
    else:
        div_second, idx_second, str_second = None, None, 0.0

    decay_12269 = signal_with_decay(div_12269, idx_12269, current_idx, decay_half_life)
    decay_second = signal_with_decay(div_second, idx_second, current_idx, decay_half_life)
    threshold = 0.15
    eff_12269 = decay_12269 * str_12269
    eff_second = decay_second * str_second
    top_12269 = div_12269 == Divergence.TOP_DIVERGENCE and eff_12269 >= threshold
    bot_12269 = div_12269 == Divergence.BOTTOM_DIVERGENCE and eff_12269 >= threshold
    top_second = div_second == Divergence.TOP_DIVERGENCE and eff_second >= threshold
    bot_second = div_second == Divergence.BOTTOM_DIVERGENCE and eff_second >= threshold

    if dif_col_second in df.columns:
        fast_golden = df[dif_col_second].iloc[-1] > df[f"DEA_{second_period_name}"].iloc[-1]
        fast_dead = df[dif_col_second].iloc[-1] < df[f"DEA_{second_period_name}"].iloc[-1]
    else:
        fast_golden = False
        fast_dead = False

    if bot_12269 and fast_golden:
        combined = "战略底背离 + 战术金叉确认 (强烈买入信号)"
    elif top_12269 and fast_dead:
        combined = "战略顶背离 + 战术死叉确认 (强烈卖出信号)"
    elif bot_12269 and bot_second:
        combined = "双重底背离 (强烈买入关注)"
    elif top_12269 and top_second:
        combined = "双重顶背离 (强烈卖出预警)"
    elif bot_12269:
        combined = "12269 底背离 (战略买入预警)"
    elif top_12269:
        combined = "12269 顶背离 (战略卖出预警)"
    elif bot_second:
        combined = f"{second_period_name} 底背离 (可考虑买入)" if div_12269 is None else \
                   f"{second_period_name} 底背离 (大趋势偏空，谨慎)"
    elif top_second:
        combined = f"{second_period_name} 顶背离 (需结合大趋势)" if div_12269 is None else \
                   f"{second_period_name} 顶背离 (可考虑卖出)"
    else:
        combined = "无明显背离信号"

    div_signal = ""
    if bot_12269:
        div_signal += "底背离 "
    if top_12269:
        div_signal += "顶背离 "
    if bot_second:
        div_signal += f"{second_period_name}底背离 "
    if top_second:
        div_signal += f"{second_period_name}顶背离 "
    div_signal = div_signal.strip() or "无背离"

    return {
        "combined_signal": combined,
        "div_12269": div_12269,
        "idx_12269": idx_12269,
        "strength_12269": str_12269,
        "decay_12269": decay_12269,
        f"div_{second_period_name}": div_second,
        f"idx_{second_period_name}": idx_second,
        f"strength_{second_period_name}": str_second,
        f"decay_{second_period_name}": decay_second,
    }


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


def calculate_momentum_score(
    df: pd.DataFrame,
    dif_12269_col: str = "DIF_12269",
    dea_12269_col: str = "DEA_12269",
    dif_second_col: str = "DIF_6135",
    dea_second_col: str = "DEA_6135",
) -> int:
    max_score = 100
    if len(df) < 5:
        return max_score // 2
    score = 0
    for col, weight in [(dif_12269_col, 0.6), (dif_second_col, 0.4)]:
        if col not in df.columns:
            continue
        hist = df[col].diff().fillna(0)
        recent = hist.iloc[-5:]
        if recent.std() == 0:
            continue
        z = recent.iloc[-1] / recent.std()
        raw = max(-1, min(1, z / 3))
        score += int(50 * (raw + 1) * weight)
    return min(max_score, max(0, score))
