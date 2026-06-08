from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RSIResult:
    rsi: pd.Series
    current_rsi: float
    zone: str
    divergence_type: str
    divergence_strength: float
    divergence_desc: str
    failure_swing: str
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
    price: pd.Series, rsi: pd.Series, window: int = 5
) -> tuple[str, float, str]:
    if len(price) < window * 3:
        return "none", 0.0, ""

    hi_idx, lo_idx = _pivot_indices(price, window)
    rsi_hi_idx, rsi_lo_idx = _pivot_indices(rsi, window)

    candidates: list[tuple[str, float, str]] = []

    # bearish divergences
    if len(hi_idx) >= 2 and len(rsi_hi_idx) >= 2:
        p_last_i = hi_idx[-1]
        p_prev_i = hi_idx[-2]
        i_last = max((i for i in rsi_hi_idx if i < p_last_i), default=None)
        i_prev = max((i for i in rsi_hi_idx if i < i_last), default=None)
        if i_prev is not None and i_last is not None:
            p_last, p_prev = price.iloc[p_last_i], price.iloc[p_prev_i]
            r_last, r_prev = rsi.iloc[i_last], rsi.iloc[i_prev]
            if p_last > p_prev and r_last < r_prev:
                candidates.append((
                    "regular_bearish",
                    min(1.0, abs(r_last - r_prev) / 20),
                    f"顶背离 (价格新高{p_last:.2f}>{p_prev:.2f}, RSI回落{r_last:.1f}<{r_prev:.1f})",
                ))
            elif p_last < p_prev and r_last > r_prev:
                candidates.append(("hidden_bearish", 0.5, "隐藏顶背离"))

    # bullish divergences
    if len(lo_idx) >= 2 and len(rsi_lo_idx) >= 2:
        p_last_i = lo_idx[-1]
        p_prev_i = lo_idx[-2]
        i_last = max((i for i in rsi_lo_idx if i < p_last_i), default=None)
        i_prev = max((i for i in rsi_lo_idx if i < i_last), default=None)
        if i_prev is not None and i_last is not None:
            p_last, p_prev = price.iloc[p_last_i], price.iloc[p_prev_i]
            r_last, r_prev = rsi.iloc[i_last], rsi.iloc[i_prev]
            if p_last < p_prev and r_last > r_prev:
                candidates.append((
                    "regular_bullish",
                    min(1.0, abs(r_last - r_prev) / 20),
                    f"底背离 (价格新低{p_last:.2f}<{p_prev:.2f}, RSI回升{r_last:.1f}>{r_prev:.1f})",
                ))
            elif p_last > p_prev and r_last < r_prev:
                candidates.append(("hidden_bullish", 0.5, "隐藏底背离"))

    if not candidates:
        return "none", 0.0, ""

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def _detect_failure_swing(rsi_series: pd.Series, lookback: int = 20) -> str:
    if len(rsi_series) < lookback * 2:
        return "none"
    rsi = rsi_series.iloc[-lookback:].values
    if len(rsi) < 15:
        return "none"

    peak_idx = int(np.argmax(rsi[:10]))
    if rsi[peak_idx] > 70:
        trough = rsi[peak_idx:].min()
        recovery = rsi[-5:].max()
        if trough < 65 and recovery < rsi[peak_idx] * 0.98 and rsi[-1] < 65:
            return "bearish_failure"

    trough_idx = int(np.argmin(rsi[:10]))
    if rsi[trough_idx] < 30:
        peak = rsi[trough_idx:].max()
        pullback = rsi[-5:].min()
        if peak > 35 and pullback > rsi[trough_idx] * 0.98 and rsi[-1] > 35:
            return "bullish_failure"

    return "none"


def rsi(close: pd.Series, period: int = 14) -> RSIResult:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    avg_loss = avg_loss.replace(0, 1e-9)

    rs = avg_gain / avg_loss
    rsi_series: pd.Series = 100 - (100 / (1 + rs))
    rsi_series.name = f"RSI_{period}"

    valid = rsi_series.dropna()
    if valid.empty:
        return RSIResult(
            rsi=rsi_series, current_rsi=50.0, zone="neutral",
            divergence_type="none", divergence_strength=0.0, divergence_desc="",
            failure_swing="none", trend="neutral", signal="数据不足",
        )

    curr = float(valid.iloc[-1])

    if curr >= 70:
        zone = "overbought"
    elif curr >= 60:
        zone = "strong"
    elif curr >= 40:
        zone = "neutral"
    elif curr >= 30:
        zone = "weak"
    else:
        zone = "oversold"

    div_type, div_str, div_desc = _detect_divergence(close.dropna(), valid, window=5)

    fs = _detect_failure_swing(valid, lookback=20)

    short = valid.iloc[-5:].mean() if len(valid) >= 5 else curr
    long = valid.iloc[-20:].mean() if len(valid) >= 20 else curr
    trend = "rising" if short > long else "falling"

    if fs == "bearish_failure":
        signal = f"衰竭顶! ({curr:.1f})"
    elif fs == "bullish_failure":
        signal = f"衰竭底! ({curr:.1f})"
    elif div_type == "regular_bearish":
        signal = f"顶背离! ({curr:.1f})"
    elif div_type == "regular_bullish":
        signal = f"底背离! ({curr:.1f})"
    elif zone == "oversold":
        signal = f"超卖 ({curr:.1f})"
    elif zone == "overbought":
        signal = f"超买 ({curr:.1f})"
    else:
        signal = f"RSI={curr:.1f}"

    return RSIResult(
        rsi=rsi_series,
        current_rsi=round(curr, 2),
        zone=zone,
        divergence_type=div_type,
        divergence_strength=round(div_str, 2),
        divergence_desc=div_desc,
        failure_swing=fs,
        trend=trend,
        signal=signal,
    )
