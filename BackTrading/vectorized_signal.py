"""全向量化信号计算 — 无 per-bar Python 循环。

入口: compute_signals(stock_df, params, compute_exit_strategy=False) -> pd.DataFrame
输出字段与 _stock_worker 的 rows 条目完全一致。
"""

from typing import Any

import numpy as np
import pandas as pd

from LogicAnalyzer.SignalConstants import Divergence, MACDSignals, MACDTrend


# ═══════════════════════════════════════════════════════════════════
# 1. MACD 趋势分类
# ═══════════════════════════════════════════════════════════════════

def macd_trend(dif: pd.Series, dea: pd.Series) -> np.ndarray:
    """逐 bar 的 MACD 趋势分类 (SUPER_STRONG/STRONG/WEAK/SUPER_WEAK)。"""
    return np.select(
        [
            (dif > dea) & (dea > 0),
            (dif > dea),
            (dif < dea) & (dea < 0),
        ],
        [
            MACDTrend.SUPER_STRONG,
            MACDTrend.STRONG,
            MACDTrend.SUPER_WEAK,
        ],
        default=MACDTrend.WEAK,
    )


# ═══════════════════════════════════════════════════════════════════
# 2. 市场状态检测
# ═══════════════════════════════════════════════════════════════════

def _regime_series(
    df: pd.DataFrame,
    boll_bw_col: str | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """全向量化的市场状态检测，返回逐 bar 的 regime 字符串数组。"""
    if params is None:
        params = {}
    close = df["close"]
    ma5 = df["MA_5"]
    ma10 = df["MA_10"]
    ma20 = df["MA_20"]
    ma30 = df["MA_30"]
    ma60 = df["MA_60"]
    dif = df["DIF"]
    dea = df["DEA"] if "DEA" in df.columns else pd.Series(0.0, index=df.index)
    hist = dif - dea
    slope_window = int(params.get("slope_window", 5))

    ma_bullish = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma30) & (ma30 > ma60)
    ma_bearish = (ma5 < ma10) & (ma10 < ma20) & (ma20 < ma30) & (ma30 < ma60)
    momentum_positive = hist > 0

    # DIF 斜率
    slope = dif.diff().rolling(slope_window, min_periods=3).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 3 else 0,
        raw=True,
    )
    slope_positive = slope > 0

    # Bollinger 带宽
    is_narrow = pd.Series(False, index=df.index)
    if boll_bw_col and boll_bw_col in df.columns:
        bw = df[boll_bw_col]
        hist_bw = bw.expanding().mean().shift(1)
        narrow_ratio = float(params.get("boll_narrow_ratio", 0.8))
        is_narrow = bw < hist_bw * narrow_ratio

    oscillation = pd.Series(False, index=df.index)
    osc_min_bars = int(params.get("oscillation_min_bars", 30))
    if len(df) > osc_min_bars:
        hist_std_ratio = float(params.get("oscillation_hist_std_ratio", 0.1))
        recent_hist_std = hist.rolling(osc_min_bars).std()
        close_std = close.rolling(osc_min_bars).std()
        oscillation = is_narrow & (hist.abs() < hist_std_ratio * close_std)

    # 反转检测
    reversal_lookback = int(params.get("reversal_lookback", 10))
    dif_positive = dif > 0
    bottom_reversal = (
        ~ma_bullish
        & (dif < 0)
        & (dif  > dif.shift(reversal_lookback))
        & (hist > hist.shift(reversal_lookback))
    )
    close_ma20_ratio = (close - ma20) / ma20.replace(0, np.nan)
    top_risk_dev = float(params.get("top_risk_ma20_deviation", 0.15))
    top_risk = (
        ma_bullish
        & (close_ma20_ratio > top_risk_dev)
        & (dif < dif.shift(reversal_lookback))
        & (hist < hist.shift(reversal_lookback))
    )

    return np.select(
        [
            ma_bullish & slope_positive & momentum_positive,
            ma_bearish & ~dif_positive & ~momentum_positive,
            oscillation,
            bottom_reversal,
            top_risk,
        ],
        [
            "STRONG_TREND",
            "WEAK_TREND",
            "OSCILLATION",
            "BOTTOM_REVERSAL",
            "TOP_RISK",
        ],
        default="UNCLEAR",
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Divergence（使用预计算的 peak/trough）
# ═══════════════════════════════════════════════════════════════════

def _divergence_scores(
    df: pd.DataFrame,
    peaks: np.ndarray,
    troughs: np.ndarray,
    distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐 bar divergence 类型/强度/衰减 — 全向量化。

    对每个 peak，向前广播 distance*2 根 bar 检查顶背离条件。
    对每个 trough，向前广播 distance*2 根 bar 检查底背离条件。
    取最近的（最后写入的）最强信号。
    """
    n = len(df)
    div_type = np.full(n, None, dtype=object)
    div_idx = np.full(n, -1, dtype=np.int32)
    div_strength = np.zeros(n, dtype=np.float64)
    close_arr = df["close"].values
    indicator_arr = df["DIF"].values
    max_lookahead = distance * 2

    # 顶背离：从每个 peak 向前广播（确保 peak 已被 distance 根 bar 确认，避免前瞻偏差）
    for p in peaks:
        start = p + distance
        end = min(p + max_lookahead + 1, n)
        if start >= end:
            continue
        idx = np.arange(start, end)
        cond = (
            (close_arr[p] > close_arr[idx] * 0.98)
            & (indicator_arr[p] > indicator_arr[idx])
        )
        valid = idx[cond]
        if len(valid) == 0:
            continue
        price_ratio = close_arr[valid] / close_arr[p] - 1
        ind_ratio = 1 - indicator_arr[valid] / indicator_arr[p]
        strengths = np.clip((price_ratio + ind_ratio) / 2, 0.0, 1.0)
        strong = strengths > 0.15
        for v, s in zip(valid[strong], strengths[strong]):
            if s > div_strength[v]:
                div_type[v] = Divergence.TOP_DIVERGENCE
                div_idx[v] = p
                div_strength[v] = s

    # 底背离：从每个 trough 向前广播
    for t in troughs:
        start = t + 1
        end = min(t + max_lookahead + 1, n)
        if start >= end:
            continue
        idx = np.arange(start, end)
        cond = (
            (close_arr[t] < close_arr[idx] * 1.02)
            & (indicator_arr[t] < indicator_arr[idx])
        )
        valid = idx[cond]
        if len(valid) == 0:
            continue
        price_ratio = 1 - close_arr[valid] / close_arr[t]
        ind_ratio = indicator_arr[valid] / indicator_arr[t] - 1
        strengths = np.clip((price_ratio + ind_ratio) / 2, 0.0, 1.0)
        strong = strengths > 0.15
        for v, s in zip(valid[strong], strengths[strong]):
            if s > div_strength[v]:
                div_type[v] = Divergence.BOTTOM_DIVERGENCE
                div_idx[v] = t
                div_strength[v] = s

    return div_type, div_idx, div_strength


def _divergence_decay(
    div_type: np.ndarray,
    div_idx: np.ndarray,
    half_life: int = 8,
) -> np.ndarray:
    """计算衰减值：decay = 0.5 ** (bars_ago / half_life)"""
    n = len(div_type)
    idx_arr = np.arange(n, dtype=np.int32)
    bars_ago = np.where(div_idx >= 0, idx_arr - div_idx, 0).astype(np.float64)
    valid = div_idx >= 0
    decay = np.zeros(n, dtype=np.float64)
    decay[valid] = 0.5 ** (bars_ago[valid] / half_life)
    return decay


# ═══════════════════════════════════════════════════════════════════
# 4. 动量分
# ═══════════════════════════════════════════════════════════════════

def _momentum(dif: pd.Series, dea: pd.Series, max_score: int = 15) -> np.ndarray:
    """逐 bar 的动能量化得分 (rolling 5)。"""
    hist = dif - dea
    hist_change = hist.diff()
    hist_vol = hist.rolling(5, min_periods=3).std().replace(0, 1e-9)
    norm_change = (hist_change / hist_vol).fillna(0).to_numpy()
    is_bull = (hist > 0).to_numpy()

    score = np.zeros(len(hist), dtype=np.int32)
    bull_mask = is_bull & (norm_change >= 0)
    score[bull_mask] = np.clip(
        (max_score * (0.5 + 0.5 * norm_change[bull_mask] / (norm_change[bull_mask] + 1))).astype(int),
        0, max_score,
    )
    bull_dec = is_bull & (norm_change < 0)
    score[bull_dec] = np.clip(
        (max_score * (0.5 + 0.5 * norm_change[bull_dec] / (norm_change[bull_dec] - 1))).astype(int),
        0, max_score,
    )
    bear = ~is_bull
    max_bear = max(8, max_score * 2 // 5)
    abs_norm = np.abs(norm_change)
    score[bear] = np.clip(
        (max_bear * abs_norm[bear] / (abs_norm[bear] + 1)).astype(int),
        0, max_bear,
    )
    score[:6] = 0
    return score


# ═══════════════════════════════════════════════════════════════════
# 5. DIF 斜率分
# ═══════════════════════════════════════════════════════════════════

def _dif_slope(dif: pd.Series, window: int = 5, max_score: int = 10) -> np.ndarray:
    """逐 bar 的 DIF 斜率得分 — 纯向量化 (np.correlate)。"""
    arr = dif.values.astype(np.float64)
    n = len(arr)
    score = np.zeros(n, dtype=np.int32)
    if n < window:
        return score

    w = window
    x = np.arange(w, dtype=np.float64)
    kernel = x - x.mean()
    denom = np.sum(kernel ** 2)

    # 每个窗口的斜率 slope = kernel · y / denom
    slopes = np.correlate(arr, kernel, mode='valid') / denom

    # 每个窗口的 ss_tot = Σy² - (Σy)²/w
    ones = np.ones(w)
    sum_y = np.correlate(arr, ones, mode='valid')
    sum_y2 = np.correlate(arr ** 2, ones, mode='valid')
    ss_tot = np.maximum(sum_y2 - sum_y ** 2 / w, 1e-12)

    r2 = np.clip(slopes ** 2 * denom / ss_tot, 0.0, 1.0)

    bar_idx = np.arange(w - 1, w - 1 + len(slopes))
    mask_pos = (slopes > 0) & (r2 > 0.7)
    mask_mid = (slopes > 0) & (r2 <= 0.7)
    score[bar_idx[mask_pos]] = max_score
    score[bar_idx[mask_mid]] = int(max_score * 0.55)
    return score


# ═══════════════════════════════════════════════════════════════════
# 6. 量价配合分
# ═══════════════════════════════════════════════════════════════════

def _volume_price(df: pd.DataFrame, lookback: int = 5, max_score: int = 10) -> np.ndarray:
    """逐 bar 量价配合得分 — 全向量化。"""
    close = df["close"].values
    volume = df["volume"].values
    n = len(df)
    half = max_score // 2
    score = np.zeros(n, dtype=np.int32)

    if n <= lookback:
        return score

    # 价格涨跌幅 (close[i] - close[i-lookback+1]) / close[i-lookback+1]
    pct = np.zeros(n)
    pct_idx = np.arange(lookback - 1, n)
    prev = pct_idx - lookback + 1
    pct[pct_idx] = (close[pct_idx] - close[prev]) / np.maximum(close[prev], 1e-9)

    # 量早/量晚: 窗口前2根均值 / 窗口最后1根
    vol_early = np.zeros(n)
    idx_lookback = np.arange(lookback, n)
    vol_early[idx_lookback] = (volume[idx_lookback - lookback + 1] + volume[idx_lookback - lookback + 2]) / 2.0

    vol_trend = np.divide(
        volume - vol_early, vol_early,
        out=np.zeros(n, dtype=np.float64),
        where=vol_early > 1e-9,
    )

    cond_qsq = (pct > 0.02) & (vol_trend > 0.1)
    score[cond_qsq] = max_score
    cond_jz = (pct > 0.02) & ~(vol_trend > 0.1)
    score[cond_jz] = half
    cond_fd = (pct < -0.02) & (vol_trend > 0.1)
    score[cond_fd] = -half
    score[:lookback] = 0
    return score


# ═══════════════════════════════════════════════════════════════════
# 7. K 线形态分
# ═══════════════════════════════════════════════════════════════════

def _kline_pattern(df: pd.DataFrame, max_score: int = 10) -> np.ndarray:
    """逐 bar K 线形态得分 — 全向量化。"""
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    if n < 5:
        return np.zeros(n, dtype=np.int32)

    # ── 单 bar 特征 ──
    body = np.abs(close - open_)
    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    bullish = close > open_

    # Hammer / Shooting star
    raw_bar = np.zeros(n)
    cond_body = body > 0
    cond_hammer = cond_body & (lower_shadow > body * 2) & (upper_shadow < body * 0.5)
    cond_shooting = cond_body & (upper_shadow > body * 2) & (lower_shadow < body * 0.5)
    raw_bar[cond_hammer] = np.where(bullish[cond_hammer], 1.0, -1.0)
    raw_bar[cond_shooting] = np.where(bullish[cond_shooting], 1.0, -1.0)

    # 窗口最后 5 根 bar 的 hammer/shooting 贡献的滚动和
    bar_acc = pd.Series(raw_bar).rolling(5, min_periods=1).sum().fillna(0).values

    # ── 三连阳/三连阴 ──
    bull_int = bullish.astype(np.int32)
    bull_sum = pd.Series(bull_int).rolling(3, min_periods=3).sum().values
    triple_raw = np.zeros(n)
    triple_raw[bull_sum == 3] = 1.0
    triple_raw[bull_sum == 0] = -1.0

    # 前缀和实现 O(1) 区间查询
    prefix = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(triple_raw, out=prefix[1:])

    i_arr = np.arange(n)
    starts = np.maximum(0, i_arr - 19)
    triple_acc = np.zeros(n)
    triple_acc[1:] = prefix[0:n - 1] - prefix[starts[1:]]

    # ── 吞没形态（engulfing） ──
    shift_v = lambda a, n: pd.Series(a).shift(n).fillna(False).values.astype(a.dtype)
    prev_close = shift_v(close, 1)
    prev_open = shift_v(open_, 1)
    prev_bullish = shift_v(bullish, 1).astype(bool)
    engulfing = np.zeros(n)
    # 看涨吞没：前阴后阳，阳吞阴
    cond_be = (~prev_bullish) & bullish & (open_ < prev_close) & (close > prev_open)
    engulfing[cond_be] = 1.0
    # 看跌吞没：前阳后阴，阴吞阳
    cond_se = prev_bullish & (~bullish) & (open_ > prev_close) & (close < prev_open)
    engulfing[cond_se] = -1.0

    # ── 晨星/夜星（3 bar） ──
    prev2_close = shift_v(close, 2)
    prev2_open = shift_v(open_, 2)
    prev_body = shift_v(body, 1)
    prev2_bullish = shift_v(bullish, 2).astype(bool)
    body_ma20 = pd.Series(body).rolling(20, min_periods=5).mean().values
    mid_body_small = prev_body < (np.where(body_ma20 > 0, body_ma20, np.nanmean(body)) * 0.3)
    # 晨星：长阴 → 小实体（跳空低开） → 长阳（收过第一根中点）
    cond_ms = (~prev2_bullish) & mid_body_small & bullish & (close > (prev2_open + prev2_close) / 2.0)
    engulfing[cond_ms] = 2.0
    # 夜星：长阳 → 小实体（跳空高开） → 长阴（收过第一根中点）
    cond_es = prev2_bullish & mid_body_small & (~bullish) & (close < (prev2_open + prev2_close) / 2.0)
    engulfing[cond_es] = -2.0

    eng_acc = pd.Series(engulfing).rolling(5, min_periods=1).sum().fillna(0).values

    # ── 合并 + 归一化 ──
    raw_total = bar_acc + triple_acc + eng_acc
    norm = np.clip(raw_total / 10.0, -1.0, 1.0)
    scores = ((norm + 1.0) / 2.0 * max_score).astype(np.int32)
    scores[:5] = 0
    return scores


# ═══════════════════════════════════════════════════════════════════
# 8. 金叉评分
# ═══════════════════════════════════════════════════════════════════

def golden_cross_score(
    df: pd.DataFrame,
    macd_cross: pd.Series,
    dif: pd.Series,
    dea: pd.Series,
    w_cross: int,
    vol_norm_denom: float,
    cross_decay_days: int,
    cross_decay_min: float,
) -> np.ndarray:
    """逐 bar 金叉评分 — 衰减部分全向量化。"""
    n = len(df)
    atr = df["ATR"]
    detail = df.get("MACD_SIGNAL_DETAIL", pd.Series("", index=df.index))
    is_bull = dif > dea

    detail_str = detail.astype(str)
    golden_zero_above = detail_str.str.contains("零轴上金叉", na=False)
    golden_zero_below = detail_str.str.contains("零轴下金叉", na=False)

    golden_strength = (dif - dea).abs() / atr.replace(0, np.nan)
    vol_factor = np.where(
        (~pd.isna(golden_strength)) & (golden_strength > 0),
        np.minimum(1.0, golden_strength / vol_norm_denom),
        1.0,
    )

    score = np.zeros(n, dtype=np.int32)
    mask_za = golden_zero_above.values
    score[mask_za] = (w_cross * vol_factor[mask_za]).astype(int)
    mask_zb = golden_zero_below.values
    score[mask_zb] = (w_cross // 2 * vol_factor[mask_zb]).astype(int)
    mask_bull = is_bull.values & ~mask_za & ~mask_zb
    score[mask_bull] = (w_cross * 0.75 * vol_factor[mask_bull]).astype(int)

    # 衰减向量化: 对每个 cross 位置，线性衰减向后传播 cross_decay_days 根 bar
    cross_positions = np.where(macd_cross.values == 1)[0]
    if len(cross_positions) == 0:
        return score

    # 为每根 bar 计算最近 cross 的距离，取最近 cross 的衰减
    # 反向扫描：从最近到最远 cross
    decay_mult = np.ones(n, dtype=np.float64)
    for idx in cross_positions:
        end = min(idx + cross_decay_days, n)
        length = end - idx
        # 线性衰减: 1.0 → cross_decay_min, 跨度 length 根 bar
        decay_curve = np.maximum(
            cross_decay_min,
            1.0 - np.arange(length, dtype=np.float64) / cross_decay_days,
        )
        # 仅当新衰减更小（更严格）时才覆盖
        mask_update = decay_curve < decay_mult[idx:end]
        decay_mult[idx:end][mask_update] = decay_curve[mask_update]

    score = (score.astype(np.float64) * decay_mult).astype(np.int32)
    return score


# ═══════════════════════════════════════════════════════════════════
# 9. 风险等级
# ═══════════════════════════════════════════════════════════════════

def _risk_level(
    regime: np.ndarray,
    macd_trend_arr: np.ndarray,
    div_type: np.ndarray,
    div_strength: np.ndarray,
    has_top_div: np.ndarray,
) -> np.ndarray:
    """逐 bar 风险等级 (HIGH/LOW/MEDIUM/NONE)。"""
    n = len(macd_trend_arr)
    rl = np.full(n, "LOW", dtype=object)
    rl[macd_trend_arr == MACDTrend.SUPER_WEAK] = "HIGH"
    rl[regime == "WEAK_TREND"] = "HIGH"
    rl[has_top_div & (div_strength > 0.3)] = "HIGH"
    return rl


# ═══════════════════════════════════════════════════════════════════
# 10. 退出评分
# ═══════════════════════════════════════════════════════════════════

def _exit_score(
    risk_level: np.ndarray,
    close: pd.Series,
    atr: pd.Series,
    atr_stop_mult: float,
) -> np.ndarray:
    """逐 bar 退出评分。"""
    n = len(risk_level)
    es = np.zeros(n, dtype=np.float64)
    es[risk_level == "HIGH"] = 100.0
    es[risk_level == "D"] = 100.0
    stop = close.shift(1) - atr.shift(1) * atr_stop_mult
    stop_hit = (stop > 0) & (close < stop)
    es[stop_hit] = np.maximum(es[stop_hit], 90.0)
    return es


# ═══════════════════════════════════════════════════════════════════
# 11. 综合评分
# ═══════════════════════════════════════════════════════════════════

def _composite_score(
    macd_trend_arr: np.ndarray,
    golden_score: np.ndarray,
    mom_score: np.ndarray,
    slope_score: np.ndarray,
    div_type: np.ndarray,
    div_strength: np.ndarray,
    div_decay: np.ndarray,
    vol_score: np.ndarray,
    kp_score: np.ndarray,
    regime: np.ndarray,
    has_top_div: np.ndarray,
    weights: dict[str, int],
    thresholds: dict[str, int],
    w_cross: int,
    w_mom: int,
    w_slope: int,
    w_div: int,
    w_vol: int,
    w_kp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐 bar 综合评分和 level/conclusion。"""
    n = len(macd_trend_arr)
    trend_score_map = {
        MACDTrend.SUPER_STRONG: int(weights["MACD趋势"]),
        MACDTrend.STRONG: int(weights["MACD趋势"] * 3 // 5),
        MACDTrend.WEAK: int(weights["MACD趋势"] * 2 // 5),
        MACDTrend.SUPER_WEAK: 0,
    }
    trend_scores = np.vectorize(trend_score_map.get)(macd_trend_arr)

    # divergence 评分
    div_score = np.zeros(n, dtype=np.int32)
    bot_div = np.char.find(div_type.astype(str), Divergence.BOTTOM_DIVERGENCE) >= 0
    eff = div_strength * div_decay
    div_score[bot_div] = (w_div * (0.5 + 0.5 * eff[bot_div])).astype(int)

    # 量价
    vol_bonus = np.where(has_top_div, 0, vol_score)
    vol_bonus = np.clip(vol_bonus, -w_vol, w_vol)

    total_base = trend_scores + golden_score + mom_score + slope_score + div_score + kp_score
    total_max_base = sum(weights.values())
    total_base = np.clip(total_base, 0, total_max_base)
    total = np.clip(total_base + vol_bonus, 0, total_max_base + w_vol)

    # level
    level = np.full(n, "C", dtype=object)
    is_high_risk = np.zeros(n, dtype=bool)
    rl_high = macd_trend_arr == MACDTrend.SUPER_WEAK
    top_div_strong = has_top_div & (div_strength > 0.3)
    is_high_risk = rl_high | top_div_strong
    level[is_high_risk] = "D"

    fb = thresholds["fully_bull"]
    bl = thresholds["bullish"]
    osc = thresholds["oscillate"]
    not_hr = ~is_high_risk
    level[not_hr & (total_base >= fb)] = "A"
    level[not_hr & (total_base >= bl) & (total_base < fb)] = "B"
    level[not_hr & (total_base >= osc) & (total_base < bl)] = "C"
    level[not_hr & (total_base < osc)] = "C"

    # 简化 conclusion
    conclusion = np.where(is_high_risk, "D: 顶部风险", "C: 正常")
    conclusion[not_hr & (total_base >= fb)] = "A: 综合多头"
    conclusion[not_hr & (total_base >= bl) & (total_base < fb)] = "B: 偏多"
    return total, level, conclusion


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

def compute_signals(
    stock_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    compute_exit_strategy: bool = False,
    precomputed_peaks: np.ndarray | None = None,
    precomputed_troughs: np.ndarray | None = None,
    diverge_distance: int = 11,
) -> pd.DataFrame:
    """全向量化信号计算。

    Args:
        stock_df: 已通过 _compute_indicators 处理的全量 K 线 DataFrame。
        params: 信号参数 dict（含 divergence, scoring, thresholds 等）。
        compute_exit_strategy: 是否计算止损价。
        precomputed_peaks: 预计算的 DIF peak 索引数组。
        precomputed_troughs: 预计算的 DIF trough 索引数组。
        diverge_distance: 背离检测距离参数。

    Returns:
        DataFrame，字段与 _stock_worker 的 rows 条目一致。
    """
    if params is None:
        params = {}
    div_p = params.get("divergence", {})
    score_p = params.get("scoring", {})
    th_p = params.get("thresholds", {})
    weights = {
        "MACD趋势": 20, "金叉信号": 15, "柱状动能": 15,
        "DIF斜率": 10, "背离信号": 10, "量价配合": 10, "K线形态": 10,
    }

    close = stock_df["close"]
    dif = stock_df["DIF"]
    dea = stock_df["DEA"] if "DEA" in stock_df.columns else pd.Series(0.0, index=stock_df.index)
    atr = stock_df["ATR"]
    macd_cross = stock_df.get("MACD_CROSS", pd.Series(0, index=stock_df.index))

    n = len(stock_df)
    # 强制数值类型，兼容配置解析为字符串的情况
    decay_half_life = int(div_p.get("decay_half_life", 8))
    slope_window = int(div_p.get("slope_window", 5))
    vol_norm_denom = float(score_p.get("vol_norm_denominator", 0.15))
    cross_decay_days = int(score_p.get("cross_decay_days", 30))
    cross_decay_min = float(score_p.get("cross_decay_min", 0.3))
    atr_stop_mult = float(
        params.get("atr_stop_mult") or score_p.get("atr_stop_mult", 1.5)
    )
    thresholds = {
        "fully_bull": int(th_p.get("fully_bull", 80)),
        "bullish": int(th_p.get("bullish", 60)),
        "oscillate": int(th_p.get("oscillate", 40)),
    }

    # ── 1. MACD 趋势 ──
    trend_arr = macd_trend(dif, dea)

    # ── 2. Divergence ──
    div_type = np.full(n, None, dtype=object)
    div_idx = np.full(n, -1, dtype=np.int32)
    div_strength = np.zeros(n, dtype=np.float64)
    if precomputed_peaks is not None and precomputed_troughs is not None:
        dt, di, ds = _divergence_scores(
            stock_df, precomputed_peaks, precomputed_troughs, diverge_distance,
        )
        div_type, div_idx, div_strength = dt, di, ds
    div_decay = _divergence_decay(div_type, div_idx, decay_half_life)
    has_top_div = np.array(
        [t == Divergence.TOP_DIVERGENCE for t in div_type], dtype=bool,
    )

    # ── 3. 市场状态 ──
    boll_bw = "BOLL_BANDWIDTH" if "BOLL_BANDWIDTH" in stock_df.columns else None
    regime = _regime_series(stock_df, boll_bw_col=boll_bw, params=params.get("regime"))

    # ── 4. 动量分 ──
    mom_score = _momentum(dif, dea, max_score=15)

    # ── 5. 斜率分 ──
    slope_score = _dif_slope(dif, window=slope_window, max_score=10)

    # ── 6. 量价分 ──
    vol_score = _volume_price(stock_df, max_score=10)

    # ── 7. K 线形态分 ──
    kp_score = _kline_pattern(stock_df, max_score=10)

    # ── 8. 金叉评分 ──
    w_cross = weights["金叉信号"]
    golden_score = golden_cross_score(
        stock_df, macd_cross, dif, dea,
        w_cross, vol_norm_denom, cross_decay_days, cross_decay_min,
    )

    # ── 9. 风险等级 ──
    risk_level = _risk_level(regime, trend_arr, div_type, div_strength, has_top_div)

    # ── 10. 退出评分 ──
    exit_score_arr = _exit_score(risk_level, close, atr, atr_stop_mult)

    # ── 11. 综合评分 ──
    w_mom = weights["柱状动能"]
    w_slope = weights["DIF斜率"]
    w_div = weights["背离信号"]
    w_vol = weights["量价配合"]
    w_kp = weights["K线形态"]
    score_arr, level_arr, conclusion_arr = _composite_score(
        trend_arr, golden_score, mom_score, slope_score,
        div_type, div_strength, div_decay,
        vol_score, kp_score, regime, has_top_div,
        weights, thresholds,
        w_cross, w_mom, w_slope, w_div, w_vol, w_kp,
    )

    # ── 止损价 ──
    stop_loss = np.where(
        (atr > 0) & (~pd.isna(atr)),
        close - atr * atr_stop_mult,
        0.0,
    ).astype(np.float64)

    result = pd.DataFrame({
        "trade_date": stock_df["trade_date"],
        "entry_score": score_arr.astype(np.float64),
        "exit_score": exit_score_arr,
        "risk_level": risk_level,
        "score": score_arr.astype(np.float64),
        "atr": atr.values,
        "macd_trend": trend_scores(dif, dea).astype(np.float64),
        "golden_cross": golden_score.astype(np.float64),
        "hist_momentum": mom_score.astype(np.float64),
        "dif_slope": slope_score.astype(np.float64),
        "divergence": (div_strength * 100).astype(np.float64),
        "vol_price": vol_score.astype(np.float64),
        "kline": kp_score.astype(np.float64),
        "stop_loss": stop_loss,
        "level": level_arr,
        "conclusion": conclusion_arr,
    })
    for c in ["entry_score", "exit_score", "score"]:
        result[c] = result[c].fillna(0.0)
    return result


def trend_scores(dif: pd.Series, dea: pd.Series) -> np.ndarray:
    """MACD 趋势分数列（兼容 _details 格式）。"""
    t = macd_trend(dif, dea)
    trend_score_map = {
        MACDTrend.SUPER_STRONG: 20,
        MACDTrend.STRONG: 12,
        MACDTrend.WEAK: 8,
        MACDTrend.SUPER_WEAK: 0,
    }
    return np.vectorize(trend_score_map.get)(t)
