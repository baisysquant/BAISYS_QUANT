"""
ScoringRules 基础函数库：条件函数 + 动作函数。

与 rules.py 互为依赖（rules 中的 RULES 列表通过 lambda 引用本模块函数）。
"""

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 条件函数（引用 gate 0-4 的规则）
# ═══════════════════════════════════════════════════════════════

def _has_top_divergence(state: dict) -> bool:
    """顶背离存在"""
    div = state.get('divergence') or {}
    return div.get('has_top_div', False)


def _has_bearish_kline_strong(state: dict) -> bool:
    """强看跌K线：三只乌鸦/乌云盖顶/看跌吞没"""
    kd = state.get('kline_data') or {}
    return kd.get('strong_bearish', False)


def _has_bearish_kline_medium(state: dict) -> bool:
    """中等级别看跌K线"""
    kd = state.get('kline_data') or {}
    return kd.get('medium_bearish', False) or kd.get('strong_bearish', False)


def _has_bullish_kline_strong(state: dict) -> bool:
    """强看涨K线"""
    kd = state.get('kline_data') or {}
    return kd.get('strong_bullish', False)


def _volume_shrinking(state: dict) -> bool:
    """缩量"""
    vt = state.get('volume_trend')
    if not vt or len(vt) < 2:
        return False
    return vt[1] < 0


def _volume_expanding(state: dict) -> bool:
    """放量"""
    vt = state.get('volume_trend')
    if not vt or len(vt) < 2:
        return False
    return vt[1] > 0


def _regime_is(state: dict, tag: str) -> bool:
    """当前情景是否匹配指定标签"""
    return state.get('regime') == tag


def _macd_golden_cross(state: dict) -> bool:
    """MACD金叉（DIF上穿DEA）"""
    sl = state.get('signal_list', [])
    return any(s.get('signal') == 'golden_cross' for s in sl)


def _macd_above_zero(state: dict) -> bool:
    """DIF/DEA在零轴上方"""
    for p in (state.get('config') or {}).get('macd_params', (12, 26, 9)):
        pass
    sl = state.get('signal_list', [])
    return any(s.get('signal') == 'macd_above_zero' for s in sl)


def _volume_positive(state: dict) -> bool:
    """量价得分 > 0"""
    vt = state.get('volume_trend')
    if not vt or len(vt) < 2:
        return False
    return vt[1] > 0


def _price_new_high(state: dict) -> bool:
    """价格创近期新高"""
    df = state.get('df')
    if df is None or df.empty:
        return False
    lookback = (state.get('config') or {}).get('divergence_lookback', 20)
    close = df['close'].dropna().values
    if len(close) < lookback + 1:
        return False
    return close[-1] >= close[-lookback - 1:-1].max()


def _rsi_not_new_high(state: dict) -> bool:
    """RSI未创新高（顶背离辅助）"""
    df = state.get('df')
    if df is None or df.empty:
        return False
    rsi_cols = [c for c in df.columns if 'rsi' in c.lower()]
    if not rsi_cols:
        return False
    rsi = df[rsi_cols[0]].dropna().values
    if len(rsi) < 20:
        return False
    return rsi[-1] <= rsi[-20:-1].max()


def _volume_not_new_high(state: dict) -> bool:
    """成交量未创新高"""
    df = state.get('df')
    if df is None or df.empty:
        return False
    vol = df['volume'].dropna().values if 'volume' in df.columns else df['amount'].dropna().values
    if len(vol) < 20:
        return False
    return vol[-1] <= vol[-20:-1].max()


def _momentum_decreasing(state: dict) -> bool:
    """MACD柱状图连续3期递减"""
    hist = state.get('histogram')
    if not hist or len(hist) < 3:
        return False
    return hist[-1] < hist[-2] < hist[-3]


def _dif_slope_turning(state: dict) -> bool:
    """DIF斜率拐头向下"""
    slope = state.get('slope')
    if not slope:
        return False
    return slope.get('slope', 0) < 0


def _kline_unconfirmed_bullish(state: dict) -> bool:
    """K线形态待确认看涨"""
    kd = state.get('kline_data') or {}
    return kd.get('unconfirmed_bullish', False)


def _has_bot_divergence(state: dict) -> bool:
    """底背离存在"""
    div = state.get('divergence') or {}
    return div.get('has_bot_div', False)


def _regime_transition_to_strong(state: dict) -> bool:
    """情景从弱势/震荡切换到强势"""
    regime = state.get('regime', '')
    prev = state.get('prev_regime', '')
    return regime in ('STRONG_TREND', 'ACCELERATION') and prev not in ('STRONG_TREND', 'ACCELERATION')


def _chip_winner_rate_high(state: dict, threshold: float | None = None) -> bool:
    """获利比例高"""
    chip = state.get('chip_data') or {}
    rate = chip.get('winner_rate', 0)
    thr = threshold if threshold is not None else 80
    return rate > thr


def _chip_winner_rate_low(state: dict, threshold: float | None = None) -> bool:
    """获利比例低"""
    chip = state.get('chip_data') or {}
    rate = chip.get('winner_rate', 0)
    thr = threshold if threshold is not None else 20
    return rate < thr


def _chip_price_at_resistance(state: dict) -> bool:
    """股价在筹码密集峰阻力位"""
    chip = state.get('chip_data') or {}
    if not chip:
        return False
    cost_95 = chip.get('cost_95pct', 0)
    cost_5 = chip.get('cost_5pct', 0)
    close = (state.get('spot_data') or {}).get('close', 0)
    if not cost_95 or not close:
        return False
    upper = cost_95 * 1.02
    lower = cost_95 * 0.98
    return lower <= close <= upper


def _chip_cost_concentrated(state: dict) -> bool:
    """筹码集中度高"""
    chip = state.get('chip_data') or {}
    cost_95 = chip.get('cost_95pct', 0)
    cost_5 = chip.get('cost_5pct', 0)
    if not cost_95 or not cost_5:
        return False
    range_pct = (cost_95 - cost_5) / cost_5 * 100
    return range_pct < 20


def _kline_too_short(state: dict) -> bool:
    """K线数据不够"""
    df = state.get('df')
    if df is None:
        return True
    min_len = (state.get('config') or {}).get('min_kline_length', 20)
    return len(df) < min_len


def _atr_missing(state: dict) -> bool:
    """ATR数据缺失"""
    df = state.get('df')
    if df is None:
        return True
    return 'atr' not in df.columns or df['atr'].dropna().empty


def _ma60_missing(state: dict) -> bool:
    """60日均线缺失"""
    df = state.get('df')
    if df is None:
        return True
    for col in df.columns:
        if 'ma60' in col.lower() or 'ma_60' in col.lower():
            return df[col].dropna().empty
    return True


def _volume_empty(state: dict) -> bool:
    """成交量数据为空"""
    df = state.get('df')
    if df is None:
        return True
    vol = df.get('volume', df.get('amount', pd.Series(dtype=float)))
    return vol.dropna().empty


def _macro_weak_with_high_score(state: dict) -> bool:
    """宏观弱但评分高"""
    macro = state.get('macro_data') or {}
    score = state.get('score', 0)
    return macro.get('regime') == 'WEAK' and score > 60


def _macro_strong_with_high_score(state: dict) -> bool:
    """宏观强且评分高"""
    macro = state.get('macro_data') or {}
    score = state.get('score', 0)
    return macro.get('regime') == 'STRONG' and score > 60


def _multi_indicator_aligned(state: dict, required: int = 3) -> bool:
    """多指标共振"""
    count = 0
    if _macd_golden_cross(state):
        count += 1
    sl = state.get('signal_list', [])
    if any(s.get('signal') == 'kdj_golden_cross' for s in sl):
        count += 1
    if any(s.get('signal') == 'rsi_bullish' for s in sl):
        count += 1
    if any(s.get('signal') == 'boll_bullish' for s in sl):
        count += 1
    if state.get('volume_trend') and state['volume_trend'][1] > 0:
        count += 1
    return count >= required


def _multitimeframe_aligned_bull(state: dict) -> bool:
    """多周期共振看涨（周线+日线）"""
    multi = state.get('multitimeframe') or {}
    return multi.get('weekly_bull') and multi.get('daily_bull')


def _macd_kdj_rsi_all_bullish(state: dict) -> bool:
    """MACD+KDJ+RSI全部看涨"""
    return (_macd_golden_cross(state)
            and any(s.get('signal') == 'kdj_golden_cross' for s in state.get('signal_list', []))
            and any(s.get('signal') == 'rsi_bullish' for s in state.get('signal_list', [])))


def _extreme_volatility(state: dict) -> bool:
    """极端波动率"""
    df = state.get('df')
    if df is None:
        return False
    atr = df.get('atr', pd.Series(dtype=float))
    close = df.get('close', pd.Series(dtype=float))
    if atr.dropna().empty or close.dropna().empty:
        return False
    atr_pct = (atr / close).dropna()
    return atr_pct.iloc[-1] > 0.08 if not atr_pct.empty else False


def _top_divergence_volume_down(state: dict) -> bool:
    """顶背离+缩量"""
    return _has_top_divergence(state) and _volume_shrinking(state)


def _momentum_exhausting(state: dict) -> bool:
    """动量衰竭"""
    hist = state.get('histogram')
    if not hist or len(hist) < 5:
        return False
    return all(hist[-i] < hist[-i - 1] for i in range(1, 5))


def _price_below_ma20_ma60(state: dict) -> bool:
    """价格在MA20和MA60下方"""
    df = state.get('df')
    if df is None:
        return True
    close = df['close'].dropna().values
    if len(close) < 2:
        return True
    for col in df.columns:
        if 'ma20' in col.lower() or 'ma_20' in col.lower():
            ma20 = df[col].dropna().values
            if len(ma20) > 0 and close[-1] < ma20[-1]:
                return True
    return False


def _liquidity_crisis(state: dict) -> bool:
    """流动性危机"""
    df = state.get('df')
    if df is None:
        return False
    amount_col = df.get('amount', df.get('volume', pd.Series(dtype=float)))
    if amount_col.dropna().empty:
        return False
    recent = amount_col.dropna().tail(5)
    if len(recent) < 5:
        return False
    ma20 = amount_col.dropna().tail(20).mean()
    return recent.mean() < ma20 * 0.3 if ma20 > 0 else False


def _amplitude_extreme_99(state: dict) -> bool:
    """振幅99分位异常"""
    df = state.get('df')
    if df is None:
        return False
    high = df.get('high', pd.Series(dtype=float))
    low = df.get('low', pd.Series(dtype=float))
    if high.dropna().empty or low.dropna().empty:
        return False
    amp = ((high - low) / low * 100).dropna()
    return amp.iloc[-1] > amp.quantile(0.99) if not amp.empty else False


def _moneyflow_positive_with_macd_bullish(state: dict) -> bool:
    """资金净流入+MACD看涨"""
    mf = state.get('moneyflow_data') or {}
    return mf.get('net_flow', 0) > 0 and _macd_golden_cross(state)


def _moneyflow_negative_with_macd_bearish(state: dict) -> bool:
    """资金净流出+MACD看跌"""
    mf = state.get('moneyflow_data') or {}
    return mf.get('net_flow', 0) < 0 and state.get('signal_list') and any(
        s.get('signal') in ('dead_cross', 'macd_bearish') for s in state['signal_list']
    )


def _volume_price_healthy(state: dict) -> bool:
    """量价健康"""
    vt = state.get('volume_trend')
    if not vt or len(vt) < 2:
        return False
    return vt[1] > 0.05


def _chip_bottom_with_divergence(state: dict) -> bool:
    """筹码底部+底背离"""
    chip = state.get('chip_data') or {}
    winner_rate = chip.get('winner_rate', 100)
    return winner_rate < 30 and _has_bot_divergence(state)


def _three_green_candles_volume_up(state: dict) -> bool:
    """三连阳+放量"""
    df = state.get('df')
    if df is None:
        return False
    close = df['close'].dropna().values
    open_ = df['open'].dropna().values if 'open' in df.columns else None
    if open_ is None or len(close) < 3:
        return False
    return all(close[-i] > open_[-i] for i in range(1, 4)) and _volume_expanding(state)


def _risk_high_pos(state: dict) -> bool:
    s = state.get('risk_level', '')
    return s == 'HIGH'


def _medium_score_low_pos(state: dict) -> bool:
    return state.get('risk_level') == 'MEDIUM' and (state.get('score', 100) < 40)


def _oscillation_score_low_pos(state: dict) -> bool:
    regime = state.get('regime', '')
    return regime in ('OSCILLATION', 'WEAK') and (state.get('score', 100) < 50)


def _bot_div_bottom_reversal_pos(state: dict) -> bool:
    return _has_bot_divergence(state) and _regime_is(state, 'BOTTOM_REVERSAL')


def _kline_strong_reversal_volume_pos(state: dict) -> bool:
    return _has_bullish_kline_strong(state) and _volume_expanding(state)


def _high_vol_atr_pos(state: dict) -> bool:
    df = state.get('df')
    if df is None:
        return False
    atr = df.get('atr', pd.Series(dtype=float))
    close = df.get('close', pd.Series(dtype=float))
    if atr.dropna().empty or close.dropna().empty:
        return False
    atr_pct = (atr / close).dropna()
    return atr_pct.iloc[-1] > 0.05 if not atr_pct.empty else False


def _vol_regime_is(state: dict, regime: str) -> bool:
    return state.get('vol_regime') == regime


def _ma_bearish_alignment(state: dict) -> bool:
    """均线空头排列"""
    df = state.get('df')
    if df is None:
        return False
    ma_cols = sorted([c for c in df.columns if c.lower().startswith('ma') and c[2:].isdigit()],
                     key=lambda x: int(x[2:]))
    if len(ma_cols) < 3:
        return False
    vals = [df[c].dropna().values[-1] if not df[c].dropna().empty else 0 for c in ma_cols]
    return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def _ma_bullish_alignment(state: dict) -> bool:
    """均线多头排列"""
    df = state.get('df')
    if df is None:
        return False
    ma_cols = sorted([c for c in df.columns if c.lower().startswith('ma') and c[2:].isdigit()],
                     key=lambda x: int(x[2:]))
    if len(ma_cols) < 3:
        return False
    vals = [df[c].dropna().values[-1] if not df[c].dropna().empty else 0 for c in ma_cols]
    return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def _score_above_oscillate(state: dict) -> bool:
    return state.get('score', 0) >= 40


def _adx_below_threshold(state: dict) -> bool:
    df = state.get('df')
    if df is None:
        return True
    adx_cols = [c for c in df.columns if 'adx' in c.lower()]
    if not adx_cols:
        return True
    adx = df[adx_cols[0]].dropna()
    return adx.iloc[-1] < 25 if not adx.empty else True


def _far_from_ma200(state: dict) -> bool:
    """远离MA200（乖离率过大）"""
    df = state.get('df')
    if df is None:
        return False
    close = df['close'].dropna().values
    if len(close) < 2:
        return False
    for col in df.columns:
        if 'ma200' in col.lower() or 'ma_200' in col.lower():
            ma200 = df[col].dropna().values
            if len(ma200) > 0:
                bias = abs(close[-1] - ma200[-1]) / ma200[-1]
                return bias > 0.3
    return False


def _high_volatility_atr(state: dict) -> bool:
    df = state.get('df')
    if df is None:
        return False
    atr = df.get('atr', pd.Series(dtype=float))
    close = df.get('close', pd.Series(dtype=float))
    if atr.dropna().empty or close.dropna().empty:
        return False
    return (atr / close).dropna().iloc[-1] > 0.06


def _abnormal_amplitude(state: dict) -> bool:
    df = state.get('df')
    if df is None:
        return False
    high = df.get('high', pd.Series(dtype=float))
    low = df.get('low', pd.Series(dtype=float))
    if high.dropna().empty or low.dropna().empty:
        return False
    amp = ((high - low) / low * 100).dropna()
    return amp.iloc[-1] > 15 if not amp.empty else False


def _boll_bandwidth_narrowing_then_expanding(state: dict) -> bool:
    """布林带宽先缩后扩（突破信号）"""
    df = state.get('df')
    if df is None:
        return False
    upper = df.get('boll_upper', pd.Series(dtype=float))
    lower = df.get('boll_lower', pd.Series(dtype=float))
    if upper.dropna().empty or lower.dropna().empty:
        return False
    bw = ((upper - lower) / ((upper + lower) / 2)).dropna()
    if len(bw) < 10:
        return False
    recent_max = bw.tail(5).max()
    prior_min = bw.head(len(bw) - 5).min()
    return recent_max > prior_min * 1.5


def _low_volume(state: dict) -> bool:
    df = state.get('df')
    if df is None:
        return False
    vol = df.get('volume', df.get('amount', pd.Series(dtype=float)))
    if vol.dropna().empty:
        return True
    ma5 = vol.dropna().tail(5).mean()
    ma20 = vol.dropna().tail(20).mean()
    return ma5 < ma20 * 0.5 if ma20 > 0 else True


def _kdj_golden_cross(state: dict) -> bool:
    sl = state.get('signal_list', [])
    return any(s.get('signal') == 'kdj_golden_cross' for s in sl)


def _rsi_not_overbought(state: dict) -> bool:
    df = state.get('df')
    if df is None:
        return False
    rsi_cols = [c for c in df.columns if 'rsi' in c.lower()]
    if not rsi_cols:
        return False
    rsi = df[rsi_cols[0]].dropna()
    return rsi.iloc[-1] < 70 if not rsi.empty else False


def _has_bot_divergence_with_volume(state: dict) -> bool:
    return _has_bot_divergence(state) and _volume_expanding(state)


def _golden_cross_stagnant(state: dict) -> bool:
    """金叉后DIF/DEA走平"""
    sl = state.get('signal_list', [])
    cross_found = any(s.get('signal') == 'golden_cross' for s in sl)
    if not cross_found:
        return False
    df = state.get('df')
    if df is None:
        return False
    dif = df.get('dif', pd.Series(dtype=float))
    if dif.dropna().empty:
        return False
    recent = dif.dropna().tail(5)
    return recent.max() - recent.min() < dif.dropna().std() * 0.2


def _has_forecast(state: dict) -> bool:
    fd = state.get('forecast_data')
    return fd is not None and bool(fd)


# ═══════════════════════════════════════════════════════════════
# 动作函数
# ═══════════════════════════════════════════════════════════════

def _act_terminate_top_risk(state: dict) -> None:
    state['conclusion'] = '见顶风险'
    state['level'] = 'D'
    state['risk_level'] = 'HIGH'
    state['risk_desc'] = '顶背离+强看跌K线+缩量'
    state['triggered_rules'].append('R01')


def _act_discount_kline_in_trend(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 15)
    state['triggered_rules'].append('R02')


def _act_boost_bottom_resonance(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R03')


def _act_boost_golden_volume(state: dict) -> None:
    bonus = state.get('config', {}).get('golden_cross_bonus', 10)
    state['score'] = min(100, state.get('score', 0) + bonus)
    state['triggered_rules'].append('R04')


def _act_fake_breakout_warning(state: dict) -> None:
    state['conclusion'] = '假突破预警'
    state['risk_desc'] = '价创新高但RSI/量未新高'
    state['score'] = max(0, state.get('score', 0) - 20)
    state['triggered_rules'].append('R05')


def _act_breakout_start(state: dict) -> None:
    state['conclusion'] = '横盘突破'
    state['triggered_rules'].append('R06')


def _act_momentum_decay(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) // 2)
    state['triggered_rules'].append('R07')


def _act_wait_kline_confirm(state: dict) -> None:
    state['triggered_rules'].append('R08')


def _act_chip_high_winner_risk(state: dict) -> None:
    if state.get('regime') in ('WEAK', 'TOP_RISK'):
        state['risk_level'] = 'MEDIUM'
        state['risk_desc'] = '高位获利盘风险'
        state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R09')


def _act_chip_resistance_risk(state: dict) -> None:
    state['risk_level'] = 'MEDIUM'
    state['risk_desc'] = '筹码密集峰阻力位'
    state['score'] = max(0, state.get('score', 0) - 5)
    state['triggered_rules'].append('R10')


def _act_ma_conflict_downgrade(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 15)
    state['level'] = 'D'
    state['triggered_rules'].append('R11')


def _act_fake_breakout_warning_adx(state: dict) -> None:
    state['conclusion'] = 'ADX低值假突破预警'
    state['risk_desc'] = '评分≥40但ADX<25，趋势不明朗'
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R12')


def _act_far_from_ma200(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R13')


def _act_high_volatility(state: dict) -> None:
    state['risk_desc'] = '高波动率'
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R14')


def _act_abnormal_amplitude(state: dict) -> None:
    state['risk_level'] = 'HIGH'
    state['risk_desc'] = '异常振幅'
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R15')


def _act_boll_breakout_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R16')


def _act_low_liquidity(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R17')


def _act_triple_resonance_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 15)
    state['triggered_rules'].append('R19')


def _act_bottom_divergence_volume_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R20')


def _act_force_level_a(state: dict) -> None:
    if state.get('score', 0) >= 60:
        state['level'] = 'A'
        state['triggered_rules'].append('R21')


def _act_golden_cross_stagnant(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R22')


def _act_forecast_note(state: dict) -> None:
    state['triggered_rules'].append('R25')


def _act_vol_trend_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R26')


def _act_vol_reversal_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R27')


def _act_data_insufficient(state: dict) -> None:
    state['conclusion'] = '数据不足'
    state['score'] = 0
    state['level'] = 'D'
    state['risk_level'] = 'HIGH'
    state['triggered_rules'].append('R30')


def _act_no_volatility(state: dict) -> None:
    state['triggered_rules'].append('R31')


def _act_no_long_term_trend(state: dict) -> None:
    state['triggered_rules'].append('R32')


def _act_no_volume(state: dict) -> None:
    state['triggered_rules'].append('R33')


def _act_macro_weak_downgrade(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 20)
    state['triggered_rules'].append('R34')


def _act_macro_strong_upgrade(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R35')


def _act_multi_factor_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R36')


def _act_kdj_rsi_volume_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R37')


def _act_multitimeframe_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R38')


def _act_four_indicator_majority(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 5)
    state['triggered_rules'].append('R39')


def _act_extreme_vol_risk(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 15)
    state['risk_level'] = 'MEDIUM'
    state['triggered_rules'].append('R40')


def _act_top_divergence_volume_down(state: dict) -> None:
    penalty = state.get('config', {}).get('divergence_penalty', 20)
    state['score'] = max(0, state.get('score', 0) - penalty)
    state['triggered_rules'].append('R41')


def _act_momentum_exhaustion(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R42')


def _act_price_below_ma20_ma60(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 20)
    state['triggered_rules'].append('R43')


def _act_liquidity_crisis(state: dict) -> None:
    state['risk_level'] = 'HIGH'
    state['risk_desc'] = '流动性枯竭'
    state['score'] = max(0, state.get('score', 0) - 30)
    state['triggered_rules'].append('R44')


def _act_amplitude_extreme_delay(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R45')


def _act_moneyflow_confirm_bullish(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 5)
    state['triggered_rules'].append('R46')


def _act_moneyflow_confirm_bearish(state: dict) -> None:
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R47')


def _act_volume_price_healthy_boost(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 5)
    state['triggered_rules'].append('R48')


def _act_chip_bottom_confirm(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R49')


def _act_three_green_strength(state: dict) -> None:
    state['score'] = min(100, state.get('score', 0) + 10)
    state['triggered_rules'].append('R50')


def _act_position_zero(state: dict) -> None:
    state['position_adjust'] = -1.0
    state['triggered_rules'].append('R51')


def _act_position_half(state: dict) -> None:
    state['position_adjust'] = -0.5
    state['triggered_rules'].append('R52')


def _act_position_reduce_30(state: dict) -> None:
    state['position_adjust'] = -0.3
    state['triggered_rules'].append('R53')


def _act_position_add_20(state: dict) -> None:
    state['position_adjust'] = 0.2
    state['triggered_rules'].append('R54')


def _act_position_add_15(state: dict) -> None:
    state['position_adjust'] = 0.15
    state['triggered_rules'].append('R55')


def _act_position_reduce_25(state: dict) -> None:
    state['position_adjust'] = -0.25
    state['triggered_rules'].append('R56')


def state_highest_risk_not_high(s: dict) -> bool:
    return s.get('risk_level') != 'HIGH'
