"""
Professional-grade technical indicator analysis module.

Provides institutional-quality RSI, Bollinger Bands, and CCI analysis including:
  - Multi-period divergence (regular + hidden)
  - Failure swings, zone persistence
  - Squeeze detection, band walk, %B analysis
  - Centerline crossovers, trend analysis

All outputs are dicts with numeric + categorical fields,
plus backward-compatible 'simple_signal' string field.
"""

import numpy as np
import pandas as pd
import pandas_ta  # noqa: F401

# ── helper ─────────────────────────────────────────────────────────


def _find_pivot_highs(series: pd.Series, window: int = 5) -> np.ndarray:
    """返回局部高点的布尔掩码。"""
    if len(series) < window * 2 + 1:
        return np.zeros(len(series), dtype=bool)
    highs = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i - window:i + window + 1].max():
            highs[i] = True
    return highs


def _find_pivot_lows(series: pd.Series, window: int = 5) -> np.ndarray:
    """返回局部低点的布尔掩码。"""
    if len(series) < window * 2 + 1:
        return np.zeros(len(series), dtype=bool)
    lows = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i - window:i + window + 1].min():
            lows[i] = True
    return lows


def _pivot_indices(series: pd.Series, window: int = 5) -> tuple[list[int], list[int]]:
    """返回 (高点索引列表, 低点索引列表)。"""
    highs_mask = _find_pivot_highs(series, window)
    lows_mask = _find_pivot_lows(series, window)
    return (
        list(np.where(highs_mask)[0]),
        list(np.where(lows_mask)[0]),
    )


# ── RSI 分析 ───────────────────────────────────────────────────────


class RSI_DivergenceType:
    NONE = "none"
    REGULAR_BULLISH = "regular_bullish"
    REGULAR_BEARISH = "regular_bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"


BOLL_ZONE_OVERBOUGHT = "overbought"
BOLL_ZONE_NEUTRAL = "neutral"
BOLL_ZONE_OVERSOLD = "oversold"


def _detect_rsi_divergence(
    price: pd.Series, rsi: pd.Series, window: int = 5
) -> dict:
    """检测全部 4 种 RSI 背离。

    Returns:
        best: 最强背离类型
        strength: [0, 1]
        description: 中文描述
    """
    if len(price) < window * 3:
        return {"best": RSI_DivergenceType.NONE, "strength": 0.0, "description": ""}

    hi_idx, lo_idx = _pivot_indices(price, window)
    rsi_hi_idx, rsi_lo_idx = _pivot_indices(rsi, window)

    candidates = []

    for direction, price_indices, indicator_indices, label in [
        ("bearish", hi_idx, rsi_hi_idx, "顶"),
        ("bullish", lo_idx, rsi_lo_idx, "底"),
    ]:
        if len(price_indices) < 2 or len(indicator_indices) < 2:
            continue
        p_last = price_indices[-1]
        p_prev = price_indices[-2]
        i_last = max((i for i in indicator_indices if i < p_last), default=None)
        i_prev = max((i for i in indicator_indices if i < i_last), default=None)
        if i_prev is None or i_last is None:
            continue

        price_prev_val = price.iloc[p_prev]
        price_last_val = price.iloc[p_last]
        ind_prev_val = rsi.iloc[i_prev]
        ind_last_val = rsi.iloc[i_last]

        if direction == "bearish":
            if price_last_val > price_prev_val and ind_last_val < ind_prev_val:
                t = RSI_DivergenceType.REGULAR_BEARISH
                desc = f"RSI顶背离 (价格新高{price_last_val:.2f} > {price_prev_val:.2f}，RSI回落{ind_last_val:.1f} < {ind_prev_val:.1f})"
                candidates.append((t, min(1.0, abs(ind_last_val - ind_prev_val) / 20), desc))
            elif price_last_val < price_prev_val and ind_last_val > ind_prev_val:
                t = RSI_DivergenceType.HIDDEN_BEARISH
                desc = "RSI隐藏顶背离 (价格回落但RSI走强)"
                candidates.append((t, 0.5, desc))
        else:
            if price_last_val < price_prev_val and ind_last_val > ind_prev_val:
                t = RSI_DivergenceType.REGULAR_BULLISH
                desc = f"RSI底背离 (价格新低{price_last_val:.2f} < {price_prev_val:.2f}，RSI回升{ind_last_val:.1f} > {ind_prev_val:.1f})"
                candidates.append((t, min(1.0, abs(ind_last_val - ind_prev_val) / 20), desc))
            elif price_last_val > price_prev_val and ind_last_val < ind_prev_val:
                t = RSI_DivergenceType.HIDDEN_BULLISH
                desc = "RSI隐藏底背离 (价格上涨但RSI走弱)"
                candidates.append((t, 0.5, desc))

    if not candidates:
        return {"best": RSI_DivergenceType.NONE, "strength": 0.0, "description": ""}

    candidates.sort(key=lambda x: x[1], reverse=True)
    return {"best": candidates[0][0], "strength": candidates[0][1], "description": candidates[0][2]}


def _detect_failure_swing(rsi_series: pd.Series, lookback: int = 20) -> str:
    """检测 RSI 衰竭摆动。

    * 看跌衰竭: RSI > 70 → 回落到 70 以下 → 反弹不破前高 → 跌破 70 以下回落低点
    * 看涨衰竭: RSI < 30 → 回升到 30 以上 → 回踩不破前低 → 突破 30 以上回升高点
    """
    if len(rsi_series) < lookback * 2:
        return "none"
    rsi = rsi_series.iloc[-lookback:].values
    if len(rsi) < 15:
        return "none"

    # 看跌衰竭
    peak_idx = np.argmax(rsi[:10])
    if rsi[peak_idx] > 70:
        trough = rsi[peak_idx:].min()
        recovery = rsi[-5:].max()
        if trough < 65 and recovery < rsi[peak_idx] * 0.98 and rsi[-1] < 65:
            return "bearish_failure"

    # 看涨衰竭
    trough_idx = np.argmin(rsi[:10])
    if rsi[trough_idx] < 30:
        peak = rsi[trough_idx:].max()
        pullback = rsi[-5:].min()
        if peak > 35 and pullback > rsi[trough_idx] * 0.98 and rsi[-1] > 35:
            return "bullish_failure"

    return "none"


def _rsi_zone_persistence(rsi_series: pd.Series) -> dict:
    """统计在当前区间停留的天数。"""
    rsi_val = rsi_series.iloc[-1]
    if pd.isna(rsi_val):
        return {"zone": "unknown", "days": 0}

    if rsi_val >= 70:
        zone = BOLL_ZONE_OVERBOUGHT
    elif rsi_val <= 30:
        zone = BOLL_ZONE_OVERSOLD
    else:
        zone = BOLL_ZONE_NEUTRAL

    days = 0
    for val in rsi_series[::-1]:
        if (zone == BOLL_ZONE_OVERBOUGHT and val >= 70) or \
           (zone == BOLL_ZONE_OVERSOLD and val <= 30) or \
           (zone == BOLL_ZONE_NEUTRAL and 30 < val < 70):
            days += 1
        else:
            break
    return {"zone": zone, "days": days}


def analyze_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    """Professional-grade RSI analysis.

    Args:
        df: OHLC DataFrame with 'close' column
        period: RSI period (default 14)

    Returns dict with keys:
        rsi_value, zone, divergence, divergence_strength, divergence_desc,
        failure_swing, zone_days, rsi_trend, simple_signal
    """
    required = ['close']
    if not all(c in df.columns for c in required):
        return {}

    if not any(c.startswith('RSI_') for c in df.columns):
        df.ta.rsi(append=True, close='close', length=period)
    rsi_col = next((c for c in df.columns if c.startswith('RSI_') and str(period) in c), None)
    if rsi_col is None:
        rsi_col = next((c for c in df.columns if c.startswith('RSI_')), None)
    if rsi_col is None:
        return {}

    rsi_series = df[rsi_col].dropna()
    if rsi_series.empty:
        return {}

    curr_rsi = float(rsi_series.iloc[-1])
    price = df['close'].dropna()

    # diverence
    div = _detect_rsi_divergence(price, rsi_series, window=5)

    # zone persistence
    zp = _rsi_zone_persistence(rsi_series)

    # failure swing
    fs = _detect_failure_swing(rsi_series, lookback=20)

    # trend
    short = rsi_series.iloc[-5:].mean() if len(rsi_series) >= 5 else curr_rsi
    long = rsi_series.iloc[-20:].mean() if len(rsi_series) >= 20 else curr_rsi
    rsi_trend = "rising" if short > long else "falling"

    # zone label
    if curr_rsi >= 70:
        zone_label = "超买"
    elif curr_rsi >= 60:
        zone_label = "偏强"
    elif curr_rsi >= 40:
        zone_label = "中性"
    elif curr_rsi >= 30:
        zone_label = "偏弱"
    else:
        zone_label = "超卖"

    # simple signal (backward-compatible)
    if div["best"] == RSI_DivergenceType.REGULAR_BULLISH:
        simple = f"RSI底背离! ({curr_rsi:.1f})"
    elif div["best"] == RSI_DivergenceType.REGULAR_BEARISH:
        simple = f"RSI顶背离! ({curr_rsi:.1f})"
    else:
        simple = f"RSI={curr_rsi:.1f}"

    return {
        "rsi_value": round(curr_rsi, 2),
        "rsi_zone": zone_label,
        "rsi_divergence_type": div["best"],
        "rsi_divergence_strength": round(div["strength"], 2),
        "rsi_divergence_desc": div["description"],
        "rsi_failure_swing": fs,
        "rsi_zone_days": zp["days"],
        "rsi_trend": rsi_trend,
        "rsi_trend_diff": round(short - long, 2),
        "simple_signal": simple,
    }


# ── 布林带分析 ─────────────────────────────────────────────────────


def _detect_boll_walk(df: pd.DataFrame, upper_col: str, lower_col: str, period: int = 5) -> str:
    """检测价格沿轨运行（Band Walk）。

    * upper_walk: 连续 N 根 K 线收盘贴近上轨
    * lower_walk: 连续 N 根 K 线收盘贴近下轨
    """
    close = df['close']
    upper = df[upper_col]
    lower = df[lower_col]
    bw = (upper - lower)

    upper_touch = (close >= upper - bw * 0.1) & (close <= upper + bw * 0.05)
    lower_touch = (close <= lower + bw * 0.1) & (close >= lower - bw * 0.05)

    if upper_touch.iloc[-period:].all() and close.iloc[-1] > close.iloc[-period]:
        return "upper_walk"
    if lower_touch.iloc[-period:].all() and close.iloc[-1] < close.iloc[-period]:
        return "lower_walk"
    return "none"


def _detect_m_top_w_bottom(df: pd.DataFrame, close: pd.Series, upper_col: str, lower_col: str, mid_col: str) -> tuple[bool, bool]:
    """检测 M 顶 / W 底 布林带形态。"""
    m_top = False
    w_bottom = False
    if len(close) < 40:
        return m_top, w_bottom

    upper = df[upper_col]
    lower = df[lower_col]
    mid = df[mid_col]

    # M 顶: 价格上穿上轨后回落至中轨, 再次上攻未创新高
    recent = close.iloc[-20:]
    if recent.max() > upper.iloc[-20:].max() * 0.98:
        peak_idx = recent.idxmax()
        post_peak = close.loc[peak_idx:]
        if len(post_peak) > 3 and post_peak.iloc[0] > mid.loc[peak_idx] and post_peak.iloc[-1] <= mid.loc[peak_idx] * 1.02:
            m_top = True

    # W 底: 价格下穿下轨后回升至中轨, 再次回踩未创新低
    if recent.min() < lower.iloc[-20:].min() * 1.02:
        trough_idx = recent.idxmin()
        post_trough = close.loc[trough_idx:]
        if len(post_trough) > 3 and post_trough.iloc[0] < mid.loc[trough_idx] and post_trough.iloc[-1] >= mid.loc[trough_idx] * 0.98:
            w_bottom = True

    return m_top, w_bottom


def analyze_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> dict:
    """Professional-grade Bollinger Bands analysis.

    Returns dict with keys:
        percent_b, bandwidth, bandwidth_pctile, squeeze, walk,
        m_top, w_bottom, zone, simple_signal
    """
    required = ['close']
    if not all(c in df.columns for c in required):
        return {}

    if not any(c.startswith('BBU_') for c in df.columns):
        df.ta.bbands(append=True, length=period, std=std, close='close')

    upper_col = next((c for c in df.columns if c.startswith('BBU_') and str(period) in c), None)
    mid_col = next((c for c in df.columns if c.startswith('BBM_') and str(period) in c), None)
    lower_col = next((c for c in df.columns if c.startswith('BBL_') and str(period) in c), None)

    if not all([upper_col, mid_col, lower_col]):
        # 容错 fallback
        upper_col = next((c for c in df.columns if c.startswith('BBU_')), None)
        mid_col = next((c for c in df.columns if c.startswith('BBM_')), None)
        lower_col = next((c for c in df.columns if c.startswith('BBL_')), None)
    if not all([upper_col, mid_col, lower_col]):
        return {}

    close = df['close'].dropna()
    upper = df[upper_col].dropna()
    lower = df[lower_col].dropna()
    mid = df[mid_col].dropna()

    # %B
    bbw = (upper - lower)
    percent_b = ((close - lower) / bbw.replace(0, np.nan)).iloc[-1]
    percent_b = round(float(percent_b), 3) if not pd.isna(percent_b) else 0.5

    # BandWidth (normalized)
    bandwidth = float((bbw / mid).iloc[-1]) if mid.iloc[-1] != 0 else 0.0
    bw_series = (bbw / mid).dropna()
    bandwidth_pctile = float(bw_series.rank(pct=True).iloc[-1]) if len(bw_series) > 1 else 0.5

    # Squeeze: bandwidth at multi-month low
    squeeze = bandwidth_pctile < 0.10 if len(bw_series) > 20 else False

    # Walk
    walk = _detect_boll_walk(df, upper_col, lower_col, period=5)

    # M-top / W-bottom
    m_top, w_bottom = _detect_m_top_w_bottom(df, close, upper_col, lower_col, mid_col)

    # Zone
    if float(close.iloc[-1]) > float(upper.iloc[-1]):
        zone = BOLL_ZONE_OVERBOUGHT
    elif float(close.iloc[-1]) < float(lower.iloc[-1]):
        zone = BOLL_ZONE_OVERSOLD
    else:
        zone = BOLL_ZONE_NEUTRAL

    # Simple signal
    if squeeze:
        simple = "低波/缩口"
    elif walk == "upper_walk":
        simple = "沿上轨攀升"
    elif walk == "lower_walk":
        simple = "沿下轨下跌"
    elif zone == BOLL_ZONE_OVERBOUGHT:
        simple = "突破上轨"
    elif zone == BOLL_ZONE_OVERSOLD:
        simple = "跌破下轨"
    else:
        simple = "常态/张口"

    return {
        "percent_b": percent_b,
        "bandwidth": round(bandwidth, 4),
        "bandwidth_pctile": round(bandwidth_pctile, 3),
        "squeeze": squeeze,
        "walk": walk,
        "m_top": m_top,
        "w_bottom": w_bottom,
        "boll_zone": zone,
        "simple_signal": simple,
    }


# ── CCI 分析 ────────────────────────────────────────────────────────


CCI_DIVERGENCE_NONE = "none"
CCI_DIVERGENCE_REGULAR_BULLISH = "regular_bullish"
CCI_DIVERGENCE_REGULAR_BEARISH = "regular_bearish"


def _detect_cci_divergence(
    price: pd.Series, cci: pd.Series, window: int = 5
) -> dict:
    hi_idx, lo_idx = _pivot_indices(price, window)
    cci_hi_idx, cci_lo_idx = _pivot_indices(cci, window)

    if len(hi_idx) >= 2 and len(cci_hi_idx) >= 2:
        p_last = price.iloc[hi_idx[-1]]
        p_prev = price.iloc[hi_idx[-2]]
        ci_last = cci.iloc[cci_hi_idx[-1]]
        ci_prev = cci.iloc[cci_hi_idx[-2]]
        if p_last > p_prev and ci_last < ci_prev:
            return {"best": CCI_DIVERGENCE_REGULAR_BEARISH, "strength": min(1.0, abs(ci_last - ci_prev) / 100), "description": "CCI顶背离"}

    if len(lo_idx) >= 2 and len(cci_lo_idx) >= 2:
        p_last = price.iloc[lo_idx[-1]]
        p_prev = price.iloc[lo_idx[-2]]
        ci_last = cci.iloc[cci_lo_idx[-1]]
        ci_prev = cci.iloc[cci_lo_idx[-2]]
        if p_last < p_prev and ci_last > ci_prev:
            return {"best": CCI_DIVERGENCE_REGULAR_BULLISH, "strength": min(1.0, abs(ci_last - ci_prev) / 100), "description": "CCI底背离"}

    return {"best": CCI_DIVERGENCE_NONE, "strength": 0.0, "description": ""}


def analyze_cci(df: pd.DataFrame, period: int = 20) -> dict:
    """Professional-grade CCI analysis.

    Returns dict with keys:
        cci_value, zone, divergence_type, divergence_strength, divergence_desc,
        centerline, cci_trend, simple_signal
    """
    required = ['close', 'high', 'low']
    if not all(c in df.columns for c in required):
        return {}

    if not any(c.startswith('CCI_') for c in df.columns):
        df.ta.cci(append=True, close='close', high='high', low='low', length=period)
    cci_col = next((c for c in df.columns if c.startswith('CCI_') and str(period) in c), None)
    if cci_col is None:
        cci_col = next((c for c in df.columns if c.startswith('CCI_')), None)
    if cci_col is None:
        return {}

    cci_series = df[cci_col].dropna()
    if cci_series.empty:
        return {}

    cci_val = float(cci_series.iloc[-1])

    # Zone
    if cci_val > 200:
        zone = "极度超买"
    elif cci_val > 100:
        zone = "强势超买"
    elif cci_val > -100:
        zone = "常态波动"
    elif cci_val > -200:
        zone = "弱势超卖"
    else:
        zone = "极度超卖"

    # Divergence
    price = df['close'].dropna()
    div = _detect_cci_divergence(price, cci_series, window=5)

    # Centerline
    centerline = "above" if cci_val > 100 else ("below" if cci_val < -100 else "neutral")

    # Trend
    short = cci_series.iloc[-5:].mean() if len(cci_series) >= 5 else cci_val
    long = cci_series.iloc[-20:].mean() if len(cci_series) >= 20 else cci_val
    cci_trend = "rising" if short > long else "falling"

    # Simple signal
    if div["best"] != CCI_DIVERGENCE_NONE:
        simple = f"{div['description']} ({cci_val:.2f})"
    else:
        simple = f"{zone} ({cci_val:.2f})"

    return {
        "cci_value": round(cci_val, 2),
        "cci_zone": zone,
        "cci_divergence_type": div["best"],
        "cci_divergence_strength": round(div["strength"], 2),
        "cci_divergence_desc": div["description"],
        "centerline": centerline,
        "cci_trend": cci_trend,
        "cci_trend_diff": round(float(short) - float(long), 2),
        "simple_signal": simple,
    }
