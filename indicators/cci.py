from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CCIResult:
    cci: pd.Series
    current_cci: float
    zone: str
    divergence_type: str
    divergence_strength: float
    divergence_desc: str
    centerline: str
    trend: str
    signal: str


def _find_pivot_highs(series: pd.Series, window: int = 5) -> np.ndarray:
    if len(series) < window * 2 + 1:
        return np.zeros(len(series), dtype=bool)
    highs = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i - window : i + window + 1].max():
            highs[i] = True
    return highs


def _find_pivot_lows(series: pd.Series, window: int = 5) -> np.ndarray:
    if len(series) < window * 2 + 1:
        return np.zeros(len(series), dtype=bool)
    lows = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i - window : i + window + 1].min():
            lows[i] = True
    return lows


def _pivot_indices(series: pd.Series, window: int = 5) -> tuple[list[int], list[int]]:
    highs_mask = _find_pivot_highs(series, window)
    lows_mask = _find_pivot_lows(series, window)
    return list(np.where(highs_mask)[0]), list(np.where(lows_mask)[0])


def _detect_divergence(
    price: pd.Series, cci: pd.Series, window: int = 5
) -> tuple[str, float, str]:
    if len(price) < window * 3:
        return "none", 0.0, ""

    hi_idx, lo_idx = _pivot_indices(price, window)
    cci_hi_idx, cci_lo_idx = _pivot_indices(cci, window)

    bearish_candidates: list[tuple[float, str]] = []
    bullish_candidates: list[tuple[float, str]] = []

    if len(hi_idx) >= 2 and len(cci_hi_idx) >= 2:
        p_last_i = hi_idx[-1]
        p_prev_i = hi_idx[-2]
        i_last = max((i for i in cci_hi_idx if i < p_last_i), default=None)
        i_prev = max((i for i in cci_hi_idx if i < i_last), default=None)
        if i_prev is not None and i_last is not None:
            p_last, p_prev = price.iloc[p_last_i], price.iloc[p_prev_i]
            c_last, c_prev = cci.iloc[i_last], cci.iloc[i_prev]
            if p_last > p_prev and c_last < c_prev:
                strength = min(1.0, abs(c_last - c_prev) / 100)
                bearish_candidates.append((
                    strength,
                    f"顶背离 (价格新高{p_last:.2f}>{p_prev:.2f}, CCI回落{c_last:.1f}<{c_prev:.1f})",
                ))

    if len(lo_idx) >= 2 and len(cci_lo_idx) >= 2:
        p_last_i = lo_idx[-1]
        p_prev_i = lo_idx[-2]
        i_last = max((i for i in cci_lo_idx if i < p_last_i), default=None)
        i_prev = max((i for i in cci_lo_idx if i < i_last), default=None)
        if i_prev is not None and i_last is not None:
            p_last, p_prev = price.iloc[p_last_i], price.iloc[p_prev_i]
            c_last, c_prev = cci.iloc[i_last], cci.iloc[i_prev]
            if p_last < p_prev and c_last > c_prev:
                strength = min(1.0, abs(c_last - c_prev) / 100)
                bullish_candidates.append((
                    strength,
                    f"底背离 (价格新低{p_last:.2f}<{p_prev:.2f}, CCI回升{c_last:.1f}>{c_prev:.1f})",
                ))

    bearish_candidates.sort(key=lambda x: x[0], reverse=True)
    bullish_candidates.sort(key=lambda x: x[0], reverse=True)

    # 返回最强的一种
    bear_str = bearish_candidates[0][0] if bearish_candidates else 0.0
    bull_str = bullish_candidates[0][0] if bullish_candidates else 0.0

    if bear_str >= bull_str and bear_str > 0:
        return ("regular_bearish", round(bear_str, 2), bearish_candidates[0][1])
    elif bull_str > 0:
        return ("regular_bullish", round(bull_str, 2), bullish_candidates[0][1])
    return "none", 0.0, ""


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> CCIResult:
    typical_price = (high + low + close) / 3

    sma = typical_price.rolling(window=period, min_periods=period).mean()

    def _mean_deviation(x: pd.Series) -> float:
        return float(np.abs(x - x.mean()).mean())

    mad = typical_price.rolling(window=period, min_periods=period).apply(
        _mean_deviation, raw=False
    )
    mad = mad.replace(0, 1e-9)

    cci_series: pd.Series = (typical_price - sma) / (0.015 * mad)
    cci_series.name = f"CCI_{period}"

    valid = cci_series.dropna()
    if valid.empty:
        return CCIResult(
            cci=cci_series, current_cci=0.0, zone="neutral",
            divergence_type="none", divergence_strength=0.0, divergence_desc="",
            centerline="neutral", trend="neutral", signal="数据不足",
        )

    curr = float(valid.iloc[-1])

    if curr > 200:
        zone = "极度超买"
    elif curr > 100:
        zone = "强势超买"
    elif curr > -100:
        zone = "常态波动"
    elif curr > -200:
        zone = "弱势超卖"
    else:
        zone = "极度超卖"

    price = close.dropna()
    price_aligned = price.loc[valid.index].dropna()
    div_type, div_str, div_desc = _detect_divergence(price_aligned, valid, window=5)

    if curr > 100:
        centerline = "above"
    elif curr < -100:
        centerline = "below"
    else:
        centerline = "neutral"

    short = valid.iloc[-5:].mean() if len(valid) >= 5 else curr
    long = valid.iloc[-20:].mean() if len(valid) >= 20 else curr
    trend = "rising" if short > long else "falling"

    if div_type == "regular_bearish":
        signal = f"顶背离! ({curr:.2f})"
    elif div_type == "regular_bullish":
        signal = f"底背离! ({curr:.2f})"
    elif curr > 200:
        signal = f"极度超买 ({curr:.2f})"
    elif curr > 100:
        signal = f"强势超买 ({curr:.2f})"
    elif curr < -200:
        signal = f"极度超卖 ({curr:.2f})"
    elif curr < -100:
        signal = f"弱势超卖 ({curr:.2f})"
    else:
        signal = f"常态 ({curr:.2f})"

    return CCIResult(
        cci=cci_series,
        current_cci=round(curr, 2),
        zone=zone,
        divergence_type=div_type,
        divergence_strength=div_str,
        divergence_desc=div_desc,
        centerline=centerline,
        trend=trend,
        signal=signal,
    )
