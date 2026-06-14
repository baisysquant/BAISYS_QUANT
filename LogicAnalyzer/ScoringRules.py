"""
规则引擎 — 因子联动规则定义。

每条规则包含：
  - id / priority / name          标识信息
  - condition(state) → bool       触发条件（从当前 state 中读取所需数据）
  - action(state) → None           动作：修改 state 中的 conclusion / level / risk 等

state 结构:
  {
    'df': DataFrame,              完整 K 线数据
    'regime': str,                情景标签
    'signal_list': list[dict],    Gate 1 信号列表
    'risk_level': str,            'HIGH'/'MEDIUM'/'LOW'/'NONE'
    'risk_desc': str,
    'conclusion': str,            最终结论文本
    'level': str,                 'A'/'B'/'C'/'D'
    'score': int,                 综合评分
    'triggered_rules': list[str], 已触发规则 ID
    'divergence': dict,           背离检测结果
    'kline_data': dict | None,    K 线形态检测结果
    'volume_trend': tuple,        量价趋势 (desc, score)
    'momentum': dict,             动能数据
    'slope': dict,                DIF 斜率数据
    'moneyflow_data': dict|None,  资金流向原始数据
    'forecast_data': dict|None,   业绩预告原始数据
    'config': dict,               规则阈值配置
    'vol_regime': str,            波动率情景 (HIGH_VOL_TREND/LOW_VOL_REVERSAL/NORMAL)
  }
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from loguru import logger

# ── 规则数据类型 ──────────────────────────────────────────────────────────────

@dataclass
class Rule:
    id: str
    priority: int                         # 1=最高, 5=最低
    name: str
    description: str
    condition: Callable[[dict], bool]     # 接收 state，返回 True/False
    action: Callable[[dict], None]        # 接收 state，修改之
    gate: int = 1                         # 所属 Gate (0~3)


# ── 条件函数（返回 True/False） ───────────────────────────────────────────────

def _has_top_divergence(state: dict) -> bool:
    """顶背离 + 强度 > threshold"""
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    threshold = state.get('config', {}).get('divergence', 0.3)
    strength = div.get('strength', 0)
    return '顶背离' in cs and strength > threshold


def _has_bearish_kline_strong(state: dict) -> bool:
    """K 线强看跌反转"""
    kd = state.get('kline_data')
    if not kd or not kd.get('details'):
        return False
    for d in kd['details']:
        if d['level'] == '强反转' and d['direction'] == '看跌':
            return True
    return False


def _has_bearish_kline_medium(state: dict) -> bool:
    """K 线中看跌反转"""
    kd = state.get('kline_data')
    if not kd or not kd.get('details'):
        return False
    for d in kd['details']:
        if d['level'] in ('强反转', '中反转') and d['direction'] == '看跌':
            return True
    return False


def _has_bullish_kline_strong(state: dict) -> bool:
    """K 线强看涨反转"""
    kd = state.get('kline_data')
    if not kd or not kd.get('details'):
        return False
    for d in kd['details']:
        if d['level'] == '强反转' and d['direction'] == '看涨':
            return True
    return False


def _volume_shrinking(state: dict) -> bool:
    """成交量萎缩"""
    vt = state.get('volume_trend')
    if not vt:
        return False
    return '缩量' in vt[0] or '量缩' in vt[0]


def _volume_expanding(state: dict) -> bool:
    """成交量放大"""
    vt = state.get('volume_trend')
    if not vt:
        return False
    return '放量' in vt[0] or '量增' in vt[0] or '量价齐升' in vt[0]


def _regime_is(state: dict, tag: str) -> bool:
    """情景是否为指定标签"""
    return state.get('regime') == tag


def _macd_golden_cross(state: dict) -> bool:
    """MACD 12269 金叉"""
    df = state.get('df')
    if df is None or len(df) < 2:
        return False
    detail = df['MACD_SIGNAL_DETAIL'].iloc[-1]
    return '金叉' in str(detail)


def _macd_above_zero(state: dict) -> bool:
    """DIF 在零轴上"""
    df = state.get('df')
    if df is None:
        return False
    return df['DIF'].iloc[-1] > 0


def _volume_positive(state: dict) -> bool:
    """量价得分 > 0"""
    vt = state.get('volume_trend')
    if not vt or len(vt) < 2:
        return False
    return vt[1] > 0


def _price_new_high(state: dict) -> bool:
    """股价创 N 日新高"""
    df = state.get('df')
    cfg = state.get('config', {})
    ndays = cfg.get('price_new_high_days', 20)
    if df is None or len(df) < ndays:
        return False
    recent = df['close'].iloc[-ndays:]
    return recent.iloc[-1] >= recent.max()


def _rsi_not_new_high(state: dict) -> bool:
    """RSI 未创新高"""
    df = state.get('df')
    cfg = state.get('config', {})
    ndays = cfg.get('price_new_high_days', 20)
    if df is None or len(df) < ndays:
        return False
    rsi_cols = [c for c in df.columns if c.startswith('RSI_')]
    if not rsi_cols:
        return False
    rsi = df[rsi_cols[0]].iloc[-ndays:]
    return rsi.iloc[-1] < rsi.max()


def _volume_not_new_high(state: dict) -> bool:
    """成交量未创新高"""
    df = state.get('df')
    cfg = state.get('config', {})
    ndays = cfg.get('price_new_high_days', 20)
    if df is None or len(df) < ndays or 'volume' not in df.columns:
        return False
    vol = df['volume'].iloc[-ndays:]
    return vol.iloc[-1] < vol.max()


def _momentum_decreasing(state: dict) -> bool:
    """动能连续 3 期递减"""
    df = state.get('df')
    if df is None or len(df) < 5:
        return False
    hist = df['DIF'] - df['DEA']
    recent = hist.iloc[-5:]
    diffs = recent.diff().iloc[1:]
    return len(diffs) >= 3 and all(diffs.iloc[-i] < 0 for i in range(1, 4) if i <= len(diffs))


def _dif_slope_turning(state: dict) -> bool:
    """DIF 斜率拐头向下"""
    slope = state.get('slope')
    if not slope:
        return False
    return slope.get('slope', 0) < 0


def _kline_unconfirmed_bullish(state: dict) -> bool:
    """K 线看涨信号尚未确认（待确认状态 > 0 根前）"""
    kd = state.get('kline_data')
    if not kd or not kd.get('details'):
        return False
    for d in kd['details']:
        if d['direction'] == '看涨' and not d['confirmed']:
            return True
    return False


def _has_bot_divergence(state: dict) -> bool:
    """底背离 + 强度 > threshold"""
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    threshold = state.get('config', {}).get('divergence', 0.3)
    strength = div.get('strength', 0)
    return '底背离' in cs and strength > threshold


def _regime_transition_to_strong(state: dict) -> bool:
    """情景从震荡切换为强势趋势"""
    # 简化：当前是强势趋势且之前是震荡（由外部调用者判断）
    return state.get('regime') == 'STRONG_TREND'


def _chip_winner_rate_high(state: dict, threshold: float | None = None) -> bool:
    """获利比例 > threshold"""
    cfg = state.get('config', {})
    if threshold is None:
        threshold = cfg.get('winner_rate_high', 80)
    chip = state.get('chip_data')
    if not chip:
        return False
    return chip.get('winner_rate', 0) > threshold

def _chip_winner_rate_low(state: dict, threshold: float | None = None) -> bool:
    """获利比例 < threshold"""
    cfg = state.get('config', {})
    if threshold is None:
        threshold = cfg.get('winner_rate_low', 15)
    chip = state.get('chip_data')
    if not chip:
        return False
    return chip.get('winner_rate', 0) < threshold

def _chip_price_at_resistance(state: dict) -> bool:
    """收盘价接近成本上沿 (cost_95pct)"""
    chip = state.get('chip_data')
    df = state.get('df')
    cfg = state.get('config', {})
    if not chip or df is None:
        return False
    close = df['close'].iloc[-1]
    cost_95 = chip.get('cost_95pct', 0)
    ratio = cfg.get('cost_resistance_ratio', 0.95)
    return cost_95 > 0 and close >= cost_95 * ratio

def _chip_cost_concentrated(state: dict) -> bool:
    """筹码高度集中"""
    chip = state.get('chip_data')
    cfg = state.get('config', {})
    if not chip:
        return False
    c5 = chip.get('cost_5pct', 0)
    c95 = chip.get('cost_95pct', 0)
    if c5 <= 0 or c95 <= 0:
        return False
    ratio = cfg.get('chip_concentrated_ratio', 0.15)
    return (c95 - c5) / c5 < ratio


# ── Gate 0: 数据质量条件 ───────────────────────────────────────────────────────

def _kline_too_short(state: dict) -> bool:
    df = state.get('df')
    return df is None or len(df) < 60


def _atr_missing(state: dict) -> bool:
    df = state.get('df')
    if df is None or 'ATR' not in df.columns:
        return True
    atr_val = df['ATR'].iloc[-1]
    return pd.isna(atr_val) or atr_val <= 0


def _ma60_missing(state: dict) -> bool:
    df = state.get('df')
    if df is None or 'MA_60' not in df.columns:
        return True
    v = df['MA_60'].iloc[-1]
    return pd.isna(v) or v <= 0


def _volume_empty(state: dict) -> bool:
    df = state.get('df')
    if df is None or 'volume' not in df.columns:
        return True
    return df['volume'].iloc[-5:].sum() <= 0


# ── Gate 0.5: 宏观情景条件 ──────────────────────────────────────────────────────

def _macro_weak_with_high_score(state: dict) -> bool:
    return state.get('regime') == 'WEAK_TREND' and state.get('score', 0) > 60


def _macro_strong_with_high_score(state: dict) -> bool:
    return state.get('regime') == 'STRONG_TREND' and state.get('score', 0) > 80


# ── Gate 1: 多因子共振条件 ──────────────────────────────────────────────────────

def _multi_indicator_aligned(state: dict, required: int = 3) -> bool:
    df = state.get('df')
    if df is None or len(df) < 3:
        return False
    bullish = 0
    detail = str(df['MACD_SIGNAL_DETAIL'].iloc[-1]) if 'MACD_SIGNAL_DETAIL' in df.columns else ''
    if '金叉' in detail:
        bullish += 1
    k_col = next((c for c in df.columns if c.startswith('STOCHk')), None)
    d_col = next((c for c in df.columns if c.startswith('STOCHd')), None)
    if k_col and d_col and df[k_col].iloc[-1] > df[d_col].iloc[-1]:
        bullish += 1
    cci_col = next((c for c in df.columns if c.startswith('CCI_')), None)
    if cci_col and not pd.isna(df[cci_col].iloc[-1]) and df[cci_col].iloc[-1] > 100:
        bullish += 1
    rsi_col = next((c for c in df.columns if c.startswith('RSI_')), None)
    if rsi_col and not pd.isna(df[rsi_col].iloc[-1]) and df[rsi_col].iloc[-1] > 50:
        bullish += 1
    return bullish >= required


def _multitimeframe_aligned_bull(state: dict) -> bool:
    return state.get('mtf_alignment') == 'ALIGNED_BULL'


def _macd_kdj_rsi_all_bullish(state: dict) -> bool:
    """MACD+KDJ+RSI 三金叉（与 R19 同逻辑，用于 Gate 1 加分）"""
    df = state.get('df')
    if df is None or len(df) < 3:
        return False
    detail = str(df['MACD_SIGNAL_DETAIL'].iloc[-1]) if 'MACD_SIGNAL_DETAIL' in df.columns else ''
    if '金叉' not in detail:
        return False
    k_col = next((c for c in df.columns if c.startswith('STOCHk')), None)
    d_col = next((c for c in df.columns if c.startswith('STOCHd')), None)
    if k_col is None or d_col is None:
        return False
    if df[k_col].iloc[-1] <= df[d_col].iloc[-1]:
        return False
    rsi_col = next((c for c in df.columns if c.startswith('RSI_')), None)
    if rsi_col is None:
        return False
    rsi_val = df[rsi_col].iloc[-1]
    return not pd.isna(rsi_val) and rsi_val < 70


# ── Gate 2: 风险加强条件 ────────────────────────────────────────────────────────

def _extreme_volatility(state: dict) -> bool:
    df = state.get('df')
    if df is None or 'ATR' not in df.columns:
        return False
    cfg = state.get('config', {})
    threshold = cfg.get('atr_extreme', 0.08)
    atr = df['ATR'].iloc[-1]
    close = df['close'].iloc[-1]
    if pd.isna(atr) or close <= 0:
        return False
    return atr / close > threshold


def _top_divergence_volume_down(state: dict) -> bool:
    """顶背离 + 放量下跌（与 R01 区分：放量恐慌 vs 缩量见顶）"""
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    if '顶背离' not in cs:
        return False
    vt = state.get('volume_trend')
    return vt is not None and ('放量' in vt[0] or '量增' in vt[0]) and '下跌' in vt[0]


def _momentum_exhausting(state: dict) -> bool:
    """柱状线连续 5 期缩短 + DIF 斜率拐头"""
    df = state.get('df')
    if df is None or len(df) < 7:
        return False
    if 'DIF' not in df.columns or 'DEA' not in df.columns:
        return False
    hist = df['DIF'] - df['DEA']
    recent = hist.iloc[-6:]
    if len(recent) < 6:
        return False
    diffs = recent.diff().iloc[1:]
    if len(diffs) < 5:
        return False
    all_decreasing = all(diffs.iloc[-i] < 0 for i in range(1, 6) if i <= len(diffs))
    if not all_decreasing:
        return False
    slope = state.get('slope')
    return slope is not None and slope.get('slope', 0) < 0


def _price_below_ma20_ma60(state: dict) -> bool:
    df = state.get('df')
    if df is None:
        return False
    try:
        close = df['close'].iloc[-1]
        ma20 = df['MA_20'].iloc[-1]
        ma60 = df['MA_60'].iloc[-1]
        return close < ma20 < ma60
    except (KeyError, IndexError):
        return False


def _liquidity_crisis(state: dict) -> bool:
    """成交额 / 自身MA20 < liq_veto_ratio → 流动性枯竭"""
    df = state.get('df')
    if df is None or 'AMOUNT' not in df.columns or 'AMOUNT_MA20' not in df.columns:
        return False
    amount = df['AMOUNT'].iloc[-1]
    amount_ma20 = df['AMOUNT_MA20'].iloc[-1]
    if pd.isna(amount_ma20) or amount_ma20 <= 0:
        return False
    config = state.get('config', {})
    veto_ratio = config.get('liq_veto_ratio', 0.05)
    return (amount / amount_ma20) < veto_ratio


def _amplitude_extreme_99(state: dict) -> bool:
    """振幅 > 近 60 日 99 分位数"""
    df = state.get('df')
    if df is None or 'AMPLITUDE_PCT' not in df.columns or len(df) < 61:
        return False
    recent = df['AMPLITUDE_PCT'].iloc[-61:-1].dropna()
    if len(recent) < 10:
        return False
    threshold = recent.quantile(0.99)
    current = df['AMPLITUDE_PCT'].iloc[-1]
    return not pd.isna(current) and current > threshold


# ── Gate 3: 评分修饰条件 ───────────────────────────────────────────────────────

def _moneyflow_positive_with_macd_bullish(state: dict) -> bool:
    """资金净流入 > 0 且 MACD 看涨"""
    mf = state.get('moneyflow_data')
    if not mf:
        return False
    net = float(mf.get('net_mf_amount', 0))
    if net <= 0:
        return False
    df = state.get('df')
    if df is None or 'DIF' not in df.columns or 'DEA' not in df.columns:
        return False
    return df['DIF'].iloc[-1] > df['DEA'].iloc[-1]


def _moneyflow_negative_with_macd_bearish(state: dict) -> bool:
    """资金净流出 > 0 且 MACD 看跌"""
    mf = state.get('moneyflow_data')
    if not mf:
        return False
    net = float(mf.get('net_mf_amount', 0))
    if net >= 0:
        return False
    df = state.get('df')
    if df is None or 'DIF' not in df.columns or 'DEA' not in df.columns:
        return False
    return df['DIF'].iloc[-1] < df['DEA'].iloc[-1]


def _volume_price_healthy(state: dict) -> bool:
    """close > MA60 且 volume > MA5 均量"""
    df = state.get('df')
    if df is None or 'volume' not in df.columns:
        return False
    try:
        close = df['close'].iloc[-1]
        ma60 = df['MA_60'].iloc[-1]
        if pd.isna(ma60) or close <= ma60:
            return False
        vol = df['volume'].iloc[-1]
        vol_ma5 = df['volume'].iloc[-5:].mean()
        return not pd.isna(vol_ma5) and vol > vol_ma5
    except (KeyError, IndexError):
        return False


def _chip_bottom_with_divergence(state: dict) -> bool:
    """筹码获利 < 20% 且 底背离"""
    chip = state.get('chip_data')
    if not chip:
        return False
    wr = float(chip.get('winner_rate', 100))
    if wr >= 20:
        return False
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    return '底背离' in cs


def _three_green_candles_volume_up(state: dict) -> bool:
    """K 线 3 连阳（收盘>开盘）且成交量依次递增"""
    df = state.get('df')
    if df is None or len(df) < 4 or 'volume' not in df.columns:
        return False
    for i in range(1, 4):
        if df['close'].iloc[-i] <= df['open'].iloc[-i]:
            return False
    recent_vol = df['volume'].iloc[-3:].values
    return all(recent_vol[i] < recent_vol[i + 1] for i in range(len(recent_vol) - 1))


# ── Gate 4: 仓位调整条件 ────────────────────────────────────────────────────────

def _risk_high_pos(state: dict) -> bool:
    return state.get('risk_level') == 'HIGH'


def _risk_medium_score_low_pos(state: dict) -> bool:
    return state.get('risk_level') == 'MEDIUM' and (state.get('score', 100) < 60)


def _oscillation_score_low_pos(state: dict) -> bool:
    return state.get('regime') == 'OSCILLATION' and (state.get('score', 100) < 50)


def _bot_div_bottom_reversal_pos(state: dict) -> bool:
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    return '底背离' in cs and state.get('regime') == 'BOTTOM_REVERSAL'


def _kline_strong_reversal_volume_pos(state: dict) -> bool:
    kd = state.get('kline_data')
    if not kd or not kd.get('details'):
        return False
    has_strong_bullish = any(d['level'] == '强反转' and d['direction'] == '看涨' for d in kd['details'])
    if not has_strong_bullish:
        return False
    vt = state.get('volume_trend')
    return vt is not None and ('放量' in vt[0] or '量增' in vt[0])


def _high_vol_atr_pos(state: dict) -> bool:
    df = state.get('df')
    if df is None or 'ATR' not in df.columns:
        return False
    atr = df['ATR'].iloc[-1]
    close = df['close'].iloc[-1]
    if pd.isna(atr) or close <= 0:
        return False
    return atr / close > 0.05


# ── 波动率情景条件 ─────────────────────────────────────────────────────────────

def _vol_regime_is(state: dict, regime: str) -> bool:
    """波动率情景是否为指定类型"""
    return state.get('vol_regime') == regime


# ── 趋势过滤族条件 ─────────────────────────────────────────────────────────────

def _ma_bearish_alignment(state: dict) -> bool:
    """均线空头排列：MA5 < MA10 < MA20 < MA30 < MA60"""
    df = state.get('df')
    if df is None or len(df) < 60:
        return False
    try:
        return (df['MA_5'].iloc[-1] < df['MA_10'].iloc[-1] < df['MA_20'].iloc[-1]
                < df['MA_30'].iloc[-1] < df['MA_60'].iloc[-1])
    except (KeyError, IndexError):
        return False


def _ma_bullish_alignment(state: dict) -> bool:
    """均线多头排列：MA5 > MA10 > MA20 > MA30 > MA60"""
    df = state.get('df')
    if df is None or len(df) < 60:
        return False
    try:
        return (df['MA_5'].iloc[-1] > df['MA_10'].iloc[-1] > df['MA_20'].iloc[-1]
                > df['MA_30'].iloc[-1] > df['MA_60'].iloc[-1])
    except (KeyError, IndexError):
        return False


def _score_above_oscillate(state: dict) -> bool:
    """综合评分 >= oscilate 阈值"""
    cfg = state.get('config', {})
    threshold = cfg.get('oscillate', 40)
    return state.get('score', 0) >= threshold


def _adx_below_threshold(state: dict) -> bool:
    """ADX 低于阈值，表示趋势弱"""
    df = state.get('df')
    if df is None or len(df) < 20:
        return False
    cfg = state.get('config', {})
    threshold = cfg.get('adx_fake_breakout', 20)
    adx_col = next((c for c in df.columns if c.startswith('ADX_')), None)
    if adx_col is None:
        return False
    return df[adx_col].iloc[-1] < threshold


def _far_from_ma200(state: dict) -> bool:
    """价格距 MA200 过远"""
    df = state.get('df')
    if df is None or 'MA_200' not in df.columns:
        return False
    cfg = state.get('config', {})
    threshold = cfg.get('ma200_distance', 0.30)
    close = df['close'].iloc[-1]
    ma200 = df['MA_200'].iloc[-1]
    if pd.isna(ma200) or ma200 <= 0:
        return False
    return (close - ma200) / ma200 > threshold


def _high_volatility_atr(state: dict) -> bool:
    """ATR/价格 > 阈值，高波动率"""
    df = state.get('df')
    if df is None or 'ATR' not in df.columns:
        return False
    cfg = state.get('config', {})
    threshold = cfg.get('atr_volatility', 0.05)
    atr = df['ATR'].iloc[-1]
    close = df['close'].iloc[-1]
    if pd.isna(atr) or close <= 0:
        return False
    return atr / close > threshold


def _abnormal_amplitude(state: dict) -> bool:
    """日内振幅超过近 20 日 95 分位数"""
    df = state.get('df')
    if df is None or 'AMPLITUDE_PCT' not in df.columns or len(df) < 21:
        return False
    cfg = state.get('config', {})
    percentile = cfg.get('amplitude_percentile', 0.95)
    recent = df['AMPLITUDE_PCT'].iloc[-21:-1].dropna()
    if len(recent) < 5:
        return False
    threshold = recent.quantile(percentile)
    current = df['AMPLITUDE_PCT'].iloc[-1]
    return not pd.isna(current) and current > threshold


def _boll_bandwidth_narrowing_then_expanding(state: dict) -> bool:
    """BOLL 带宽先缩口后张口（突破信号）"""
    df = state.get('df')
    bw_col = next((c for c in ('BOLL_BANDWIDTH',) if c in df.columns), None)
    if df is None or bw_col is None or len(df) < 25:
        return False
    bw = df[bw_col].dropna()
    if len(bw) < 25:
        return False
    recent_10 = bw.iloc[-10:].mean()
    hist_mean = bw.mean()
    if recent_10 >= hist_mean * 0.8:
        return False
    before_10 = bw.iloc[-20:-10].mean()
    return bw.iloc[-1] > before_10 * 1.05


def _low_volume(state: dict) -> bool:
    """成交额（volume × close）低于阈值"""
    df = state.get('df')
    if df is None or 'volume' not in df.columns:
        return False
    cfg = state.get('config', {})
    threshold = cfg.get('volume_threshold', 1e7)
    close = df['close'].iloc[-1]
    volume = df['volume'].iloc[-1]
    return close * volume < threshold


def _kdj_golden_cross(state: dict) -> bool:
    """KDJ 金叉：STOCHk 上穿 STOCHd"""
    df = state.get('df')
    if df is None or len(df) < 3:
        return False
    k_col = next((c for c in df.columns if c.startswith('STOCHk')), None)
    d_col = next((c for c in df.columns if c.startswith('STOCHd')), None)
    if k_col is None or d_col is None:
        return False
    k_series = df[k_col]
    d_series = df[d_col]
    if len(k_series) < 2:
        return False
    return k_series.iloc[-1] > d_series.iloc[-1] and k_series.iloc[-2] <= d_series.iloc[-2]


def _rsi_not_overbought(state: dict) -> bool:
    """RSI 未超买（< 70）"""
    df = state.get('df')
    if df is None:
        return False
    rsi_col = next((c for c in df.columns if c.startswith('RSI_')), None)
    if rsi_col is None:
        return False
    rsi_val = df[rsi_col].iloc[-1]
    return not pd.isna(rsi_val) and rsi_val < 70


def _has_bot_divergence_with_volume(state: dict) -> bool:
    """底背离存在且量价得分 > 0"""
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    vt = state.get('volume_trend')
    return '底背离' in cs and vt is not None and len(vt) >= 2 and vt[1] > 0


def _golden_cross_stagnant(state: dict) -> bool:
    """金叉已发生 N 天，价格无明显上涨"""
    df = state.get('df')
    if df is None or 'MACD_SIGNAL_DETAIL' not in df.columns or len(df) < 30:
        return False
    cfg = state.get('config', {})
    stagnant_days = cfg.get('golden_cross_stagnant_days', 10)
    stagnant_pct = cfg.get('golden_cross_stagnant_pct', 0.02)
    detail = df['MACD_SIGNAL_DETAIL']
    for i in range(len(detail) - 2, max(len(detail) - stagnant_days - 5, 0), -1):
        if '金叉' in str(detail.iloc[i]):
            cross_idx = i
            cross_close = df['close'].iloc[cross_idx]
            current_close = df['close'].iloc[-1]
            days_since = len(df) - 1 - cross_idx
            if days_since >= stagnant_days:
                return (current_close - cross_close) / cross_close < stagnant_pct
            return False
    return False


def _has_forecast(state: dict) -> bool:
    """有业绩预告数据"""
    fc = state.get('forecast_data')
    return fc is not None and bool(fc.get('type', ''))


# ── 动作函数（修改 state） ────────────────────────────────────────────────────

def _act_terminate_top_risk(state: dict) -> None:
    """R01: 顶部否决——跳过所有后续评分"""
    state['level'] = 'D'
    state['conclusion'] = 'D: 顶部风险: 顶背离+见顶形态+缩量'
    state['risk_level'] = 'HIGH'
    state['score'] = 0
    state['triggered_rules'].append('R01')


def _act_discount_kline_in_trend(state: dict) -> None:
    """R02: 趋势中 K 线形态权重打折"""
    state.setdefault('_notes', [])
    state['_notes'].append('(趋势中注意回调)')


def _act_boost_bottom_resonance(state: dict) -> None:
    """R03: 底部共振——提升信号可信度"""
    state.setdefault('_notes', [])
    state['_notes'].append('共振')


def _act_boost_golden_volume(state: dict) -> None:
    """R04: 金叉量价确认"""
    state.setdefault('_notes', [])
    if '(趋势中注意回调)' not in state.get('_notes', []):
        state['_notes'].append('量价确认')


def _act_fake_breakout_warning(state: dict) -> None:
    """R05: 假突破预警"""
    state['level'] = 'B' if state.get('level', 'C') != 'D' else 'D'
    state.setdefault('_notes', [])
    state['_notes'].append('假突破预警')


def _act_breakout_start(state: dict) -> None:
    """R06: 横盘突破"""
    state.setdefault('_notes', [])
    state['_notes'].append('横盘突破趋势启动')


def _act_momentum_decay(state: dict) -> None:
    """R07: 力度衰减"""
    state.setdefault('_notes', [])
    state['_notes'].append('上涨力度衰减')


def _act_wait_kline_confirm(state: dict) -> None:
    """R08: 底部待确认"""
    state.setdefault('_notes', [])
    state['_notes'].append('等待K线确认')


def _act_chip_high_winner_risk(state: dict) -> None:
    """R09: 高位获利盘风险"""
    wr = state.get('chip_data', {}).get('winner_rate', 0)
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
    state.setdefault('_notes', [])
    state['_notes'].append(f'筹码风险:获利{wr:.0f}%')
    state['triggered_rules'].append('R09')

def _act_chip_resistance_risk(state: dict) -> None:
    """R10: 筹码阻力位风险"""
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
    state.setdefault('_notes', [])
    state['_notes'].append('筹码阻力位')
    state['triggered_rules'].append('R10')


# ── 趋势过滤族动作 ─────────────────────────────────────────────────────────────

def _act_ma_conflict_downgrade(state: dict) -> None:
    """R11: 均线方向与评分方向冲突 → 降级"""
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
    state.setdefault('_notes', [])
    state['_notes'].append('均线排列不支持当前方向')
    state['triggered_rules'].append('R11')


def _act_fake_breakout_warning_adx(state: dict) -> None:
    """R12: ADX 低位 + 高分 → 假突破预警"""
    state.setdefault('_notes', [])
    state['_notes'].append('ADX偏低假突破风险')
    if state.get('level', 'C') not in ('D',):
        state['level'] = 'C'
    state['triggered_rules'].append('R12')


def _act_far_from_ma200(state: dict) -> None:
    """R13: 距 MA200 过远 → 均值回归预警"""
    state.setdefault('_notes', [])
    state['_notes'].append('偏离MA200过远均值回归风险')
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
    state['triggered_rules'].append('R13')


def _act_high_volatility(state: dict) -> None:
    """R14: 高波动率 → 缩小仓位提示"""
    state.setdefault('_notes', [])
    state['_notes'].append('高波动率注意仓位控制')
    state['triggered_rules'].append('R14')


def _act_abnormal_amplitude(state: dict) -> None:
    """R15: 振幅异常 → 延迟入场提示"""
    state.setdefault('_notes', [])
    state['_notes'].append('异动振幅建议延迟入场')
    state['triggered_rules'].append('R15')


def _act_boll_breakout_boost(state: dict) -> None:
    """R16: BOLL 缩口后张口 → 突破加分"""
    state.setdefault('_notes', [])
    state['_notes'].append('BOLL缩口突破')
    state['triggered_rules'].append('R16')


def _act_low_liquidity(state: dict) -> None:
    """R17: 低流动性 → 不可交易标记"""
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
    state.setdefault('_notes', [])
    state['_notes'].append('低流动性注意')
    state['triggered_rules'].append('R17')


def _act_triple_resonance_boost(state: dict) -> None:
    """R19: MACD+KDJ+RSI 三金叉共振 → 加分"""
    state.setdefault('_notes', [])
    if '三金叉共振' not in state.get('_notes', []):
        state['_notes'].append('三金叉共振')
    state['triggered_rules'].append('R19')


def _act_bottom_divergence_volume_boost(state: dict) -> None:
    """R20: 底背离 + 量价齐升 → 加分"""
    state.setdefault('_notes', [])
    state['_notes'].append('底背离量价共振')
    state['triggered_rules'].append('R20')


def _act_force_level_a(state: dict) -> None:
    """R21: 多头排列 + MACD 超强 → 强制 A 级"""
    state['level'] = 'A'
    state.setdefault('_notes', [])
    state['_notes'].append('多头共振推A级')
    state['triggered_rules'].append('R21')


def _act_golden_cross_stagnant(state: dict) -> None:
    """R22: 金叉后未涨 → 信号衰减"""
    state.setdefault('_notes', [])
    state['_notes'].append('金叉钝化迟迟未涨')
    state['score'] = max(0, state.get('score', 0) - 10)
    state['triggered_rules'].append('R22')


def _act_forecast_note(state: dict) -> None:
    """R25: 业绩预告期间 → 延迟入场提示"""
    state.setdefault('_notes', [])
    fc = state.get('forecast_data', {})
    fc_type = fc.get('type', '未知')
    state['_notes'].append(f'业绩预告{fc_type}')
    state['triggered_rules'].append('R25')


# ── 波动率情景动作 ─────────────────────────────────────────────────────────────

def _act_vol_trend_boost(state: dict) -> None:
    """R26: 高波动趋势市 → 强化趋势/突破信号"""
    state.setdefault('_notes', [])
    act_notes = ['趋势市优先']
    if '横盘突破趋势启动' in state.get('_notes', []):
        act_notes.append('突破确认')
    if state.get('level') == 'A':
        act_notes.append('趋势共振')
    state['_notes'].extend(act_notes)
    state['triggered_rules'].append('R26')


def _act_vol_reversal_boost(state: dict) -> None:
    """R27: 低波动震荡市 → 强化反转/底部信号"""
    state.setdefault('_notes', [])
    act_notes = ['反转市优先']
    if '共振' in state.get('_notes', []):
        act_notes.append('底部共振确认')
    if '底背离' in str(state.get('divergence', {}).get('combined_signal', '')):
        act_notes.append('底背离强化')
    state['_notes'].extend(act_notes)
    state['triggered_rules'].append('R27')


# ── Gate 0: 数据质量动作 ───────────────────────────────────────────────────────

def _act_data_insufficient(state: dict) -> None:
    """R30: 数据不足 → 否决"""
    state['level'] = 'D'
    state['score'] = 0
    state['risk_level'] = 'HIGH'
    state['conclusion'] = 'D: 数据不足(K线<60)'
    state['triggered_rules'].append('R30')


def _act_no_volatility(state: dict) -> None:
    """R31: 无波动率数据 → 标记，score=0"""
    state['score'] = 0
    state.setdefault('_notes', [])
    state['_notes'].append('无ATR波动率数据')
    state['triggered_rules'].append('R31')


def _act_no_long_term_trend(state: dict) -> None:
    """R32: 无长期趋势参考 → 标记"""
    state.setdefault('_notes', [])
    state['_notes'].append('MA60缺失无长期趋势')
    state['triggered_rules'].append('R32')


def _act_no_volume(state: dict) -> None:
    """R33: 无成交量数据 → 中等风险"""
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
    state.setdefault('_notes', [])
    state['_notes'].append('无成交量数据')
    state['triggered_rules'].append('R33')


# ── Gate 0.5: 宏观情景动作 ─────────────────────────────────────────────────────

def _act_macro_weak_downgrade(state: dict) -> None:
    """R34: 大盘弱势 + 高分 → 降级"""
    if state.get('level', 'C') == 'B':
        state['level'] = 'C'
    state.setdefault('_notes', [])
    state['_notes'].append('大盘弱势，降级')
    state['triggered_rules'].append('R34')


def _act_macro_strong_upgrade(state: dict) -> None:
    """R35: 大盘强势 + 高分 → 升级"""
    if state.get('level', 'C') == 'B':
        state['level'] = 'A'
        state.setdefault('_notes', [])
        state['_notes'].append('大盘共振升级')
    elif state.get('level', 'C') in ('C',) and state.get('score', 0) > 70:
        state['level'] = 'B'
        state.setdefault('_notes', [])
        state['_notes'].append('大盘共振升B级')
    state['triggered_rules'].append('R35')


# ── Gate 1: 多因子共振动作 ─────────────────────────────────────────────────────

def _act_multi_factor_boost(state: dict) -> None:
    """R36: 多指标同向 → 加分"""
    state['score'] = min(100, state.get('score', 0) + 5)
    state.setdefault('_notes', [])
    if '多因子共振' not in state['_notes']:
        state['_notes'].append('多因子共振+5')
    state['triggered_rules'].append('R36')


def _act_kdj_rsi_volume_boost(state: dict) -> None:
    """R37: KDJ+RSI+量价 → 加分"""
    state['score'] = min(100, state.get('score', 0) + 5)
    state.setdefault('_notes', [])
    if 'KDJ+RSI共振' not in state.get('_notes', []):
        state['_notes'].append('KDJ+RSI共振+5')
    state['triggered_rules'].append('R37')


def _act_multitimeframe_boost(state: dict) -> None:
    """R38: 日周共振多头 → 加分"""
    state['score'] = min(100, state.get('score', 0) + 10)
    state.setdefault('_notes', [])
    state['_notes'].append('日周共振多头+10')
    state['triggered_rules'].append('R38')


def _act_four_indicator_majority(state: dict) -> None:
    """R39: 四指标多数同向 → 强制升级"""
    if state['risk_level'] not in ('HIGH',):
        if state.get('level', 'C') == 'C':
            state['level'] = 'B'
            state.setdefault('_notes', [])
            state['_notes'].append('多指标共振升B级')
    state['triggered_rules'].append('R39')


# ── Gate 2: 风险加强动作 ───────────────────────────────────────────────────────

def _act_extreme_vol_risk(state: dict) -> None:
    """R40: 极端波动 → 中等风险"""
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
    state.setdefault('_notes', [])
    state['_notes'].append('极端波动(ATR>8%)')
    state['triggered_rules'].append('R40')


def _act_top_divergence_volume_down(state: dict) -> None:
    """R41: 顶背离+放量下跌 → 否决"""
    state['level'] = 'D'
    state['conclusion'] = 'D: 顶背离+放量下跌'
    state['risk_level'] = 'HIGH'
    state['score'] = 0
    state['triggered_rules'].append('R41')


def _act_momentum_exhaustion(state: dict) -> None:
    """R42: 动能衰竭 → 降级"""
    if state.get('level', 'C') == 'B':
        state['level'] = 'C'
    state.setdefault('_notes', [])
    state['_notes'].append('动能衰竭(柱状5期缩短)')
    state['triggered_rules'].append('R42')


def _act_price_below_ma20_ma60(state: dict) -> None:
    """R43: close<MA20<MA60 → 中等风险"""
    if state['risk_level'] in ('NONE', 'LOW'):
        state['risk_level'] = 'MEDIUM'
        state.setdefault('_notes', [])
        state['_notes'].append('均线空头排列(MA20>MA60)')
    state['triggered_rules'].append('R43')


def _act_liquidity_crisis(state: dict) -> None:
    """R44: 流动性枯竭 → 否决"""
    state['level'] = 'D'
    state['conclusion'] = 'D: 流动性枯竭'
    state['risk_level'] = 'HIGH'
    state['score'] = 0
    state['triggered_rules'].append('R44')


def _act_amplitude_extreme_delay(state: dict) -> None:
    """R45: 振幅极端 → 延迟入场"""
    state.setdefault('_notes', [])
    state['_notes'].append('异动振幅(99分位)建议延迟入场')
    state['triggered_rules'].append('R45')


# ── Gate 3: 评分修饰动作 ───────────────────────────────────────────────────────

def _act_moneyflow_confirm_bullish(state: dict) -> None:
    """R46: 资金净流入确认上涨 → 加分"""
    state['score'] = min(100, state.get('score', 0) + 3)
    state.setdefault('_notes', [])
    state['_notes'].append('资金确认+3')
    state['triggered_rules'].append('R46')


def _act_moneyflow_confirm_bearish(state: dict) -> None:
    """R47: 资金净流出确认下跌 → 扣分"""
    state['score'] = max(0, state.get('score', 0) - 3)
    state.setdefault('_notes', [])
    state['_notes'].append('资金流出-3')
    state['triggered_rules'].append('R47')


def _act_volume_price_healthy_boost(state: dict) -> None:
    """R48: 量价趋势健康 → 加分"""
    state['score'] = min(100, state.get('score', 0) + 5)
    state.setdefault('_notes', [])
    state['_notes'].append('量价健康+5')
    state['triggered_rules'].append('R48')


def _act_chip_bottom_confirm(state: dict) -> None:
    """R49: 筹码底部+底背离 → 加分"""
    state['score'] = min(100, state.get('score', 0) + 8)
    state.setdefault('_notes', [])
    state['_notes'].append('筹码底部确认+8')
    state['triggered_rules'].append('R49')


def _act_three_green_strength(state: dict) -> None:
    """R50: 三连阳量递增 → 加分"""
    state['score'] = min(100, state.get('score', 0) + 5)
    state.setdefault('_notes', [])
    state['_notes'].append('三连阳量增+5')
    state['triggered_rules'].append('R50')


# ── Gate 4: 仓位调整动作 ────────────────────────────────────────────────────────

def _act_position_zero(state: dict) -> None:
    """R51: HIGH 风险 → 仓位归零"""
    state['position_adjust'] = state.get('position_adjust', 0.0) - 1.0
    state['triggered_rules'].append('R51')


def _act_position_half(state: dict) -> None:
    """R52: MEDIUM风险+低分 → 仓位砍半"""
    state['position_adjust'] = state.get('position_adjust', 0.0) - 0.5
    state['triggered_rules'].append('R52')


def _act_position_reduce_30(state: dict) -> None:
    """R53: 震荡+低分 → 减仓"""
    state['position_adjust'] = state.get('position_adjust', 0.0) - 0.3
    state['triggered_rules'].append('R53')


def _act_position_add_20(state: dict) -> None:
    """R54: 底背离+底部反转 → 加仓"""
    state['position_adjust'] = state.get('position_adjust', 0.0) + 0.2
    state['triggered_rules'].append('R54')


def _act_position_add_15(state: dict) -> None:
    """R55: K线强反转+放量 → 加仓"""
    state['position_adjust'] = state.get('position_adjust', 0.0) + 0.15
    state['triggered_rules'].append('R55')


def _act_position_reduce_25(state: dict) -> None:
    """R56: 高波动率 → 减仓"""
    state['position_adjust'] = state.get('position_adjust', 0.0) - 0.25
    state['triggered_rules'].append('R56')


# ── 规则库 ────────────────────────────────────────────────────────────────────

RULES: list[Rule] = [
    # ── R01: 顶部否决（最高优先级） ──────────────────────────────────────────
    Rule(
        id='R01', priority=1, name='顶部否决',
        description='顶背离+强看跌K线+缩量 → 直接否决',
        condition=lambda s: _has_top_divergence(s) and _has_bearish_kline_strong(s) and _volume_shrinking(s),
        action=_act_terminate_top_risk,
        gate=2,
    ),
    # ── R02: 趋势见顶预警 ────────────────────────────────────────────────────
    Rule(
        id='R02', priority=2, name='趋势见顶预警',
        description='强势趋势+单根见顶K线(无顶背离) → K线权重打折+备注',
        condition=lambda s: _regime_is(s, 'STRONG_TREND') and _has_bearish_kline_medium(s) and not _has_top_divergence(s),
        action=_act_discount_kline_in_trend,
        gate=3,
    ),
    # ── R03: 底部共振 ────────────────────────────────────────────────────────
    Rule(
        id='R03', priority=2, name='底部共振',
        description='底部反转+零轴下金叉 → 信号可信度+1',
        condition=lambda s: _regime_is(s, 'BOTTOM_REVERSAL') and _macd_golden_cross(s) and not _macd_above_zero(s),
        action=_act_boost_bottom_resonance,
        gate=1,
    ),
    # ── R04: 金叉量价确认 ────────────────────────────────────────────────────
    Rule(
        id='R04', priority=3, name='金叉量价确认',
        description='MACD金叉+量价得分>0 → 可信度+1',
        condition=lambda s: _macd_golden_cross(s) and _volume_positive(s),
        action=_act_boost_golden_volume,
        gate=1,
    ),
    # ── R05: 假突破预警 ──────────────────────────────────────────────────────
    Rule(
        id='R05', priority=3, name='假突破预警',
        description='新高+RSI未新高+量未新高 → 标记假突破',
        condition=lambda s: _price_new_high(s) and _rsi_not_new_high(s) and _volume_not_new_high(s),
        action=_act_fake_breakout_warning,
        gate=2,
    ),
    # ── R06: 横盘突破 ────────────────────────────────────────────────────────
    Rule(
        id='R06', priority=4, name='横盘突破',
        description='震荡→趋势切换+放量 → 标记趋势启动',
        condition=lambda s: _regime_transition_to_strong(s) and _volume_expanding(s),
        action=_act_breakout_start,
        gate=3,
    ),
    # ── R07: 力度衰减 ────────────────────────────────────────────────────────
    Rule(
        id='R07', priority=4, name='力度衰减',
        description='动能3期递减+DIF斜率拐头 → 动能分减半',
        condition=lambda s: _momentum_decreasing(s) and _dif_slope_turning(s),
        action=_act_momentum_decay,
        gate=3,
    ),
    # ── R08: 底部二次确认 ────────────────────────────────────────────────────
    Rule(
        id='R08', priority=5, name='底部二次确认',
        description='底背离+K线待确认 → 加备注',
        condition=lambda s: _has_bot_divergence(s) and _kline_unconfirmed_bullish(s),
        action=_act_wait_kline_confirm,
        gate=1,
    ),
    # ── R09: 高位获利盘风险 ───────────────────────────────────────────────
    Rule(
        id='R09', priority=2, name='高位获利盘风险',
        description='获利比例>80% + 弱势/顶部情景 → 中等风险',
        condition=lambda s: _chip_winner_rate_high(s) and (
            s.get('regime') in ('WEAK_TREND', 'TOP_RISK', 'UNCLEAR')
        ),
        action=_act_chip_high_winner_risk,
        gate=2,
    ),
    # ── R10: 筹码阻力位 ───────────────────────────────────────────────────
    Rule(
        id='R10', priority=3, name='筹码阻力位',
        description='价格接近成本上沿 + 高位获利 → 中等风险',
        condition=lambda s: _chip_price_at_resistance(s) and _chip_winner_rate_high(s, threshold=70),
        action=_act_chip_resistance_risk,
        gate=2,
    ),
    # ── R11: 均线方向冲突 ──────────────────────────────────────────────────
    Rule(
        id='R11', priority=2, name='均线方向冲突',
        description='空头排列 + 评分 ≥ 震荡阈值 → 降级',
        condition=lambda s: _ma_bearish_alignment(s) and _score_above_oscillate(s),
        action=_act_ma_conflict_downgrade,
        gate=2,
    ),
    # ── R12: ADX 假突破预警 ───────────────────────────────────────────────
    Rule(
        id='R12', priority=2, name='ADX假突破预警',
        description='ADX < 20 + 高分 → 趋势弱，假突破风险',
        condition=lambda s: _adx_below_threshold(s) and s.get('score', 0) > 70,
        action=_act_fake_breakout_warning_adx,
        gate=2,
    ),
    # ── R13: 距 MA200 过远 ────────────────────────────────────────────────
    Rule(
        id='R13', priority=3, name='偏离MA200',
        description='价格距 MA200 超过 30% → 均值回归风险',
        condition=_far_from_ma200,
        action=_act_far_from_ma200,
        gate=3,
    ),
    # ── R14: 高波动率 ──────────────────────────────────────────────────────
    Rule(
        id='R14', priority=4, name='高波动率',
        description='ATR/价格 > 5% → 高波动率提示',
        condition=_high_volatility_atr,
        action=_act_high_volatility,
        gate=3,
    ),
    # ── R15: 振幅异常 ─────────────────────────────────────────────────────
    Rule(
        id='R15', priority=3, name='振幅异常',
        description='日内振幅 > 近20日95分位 → 延迟入场',
        condition=_abnormal_amplitude,
        action=_act_abnormal_amplitude,
        gate=2,
    ),
    # ── R16: BOLL 缩口突破 ────────────────────────────────────────────────
    Rule(
        id='R16', priority=4, name='BOLL缩口突破',
        description='带宽缩口后张口 → 突破信号加分',
        condition=_boll_bandwidth_narrowing_then_expanding,
        action=_act_boll_breakout_boost,
        gate=3,
    ),
    # ── R17: 低流动性 ─────────────────────────────────────────────────────
    Rule(
        id='R17', priority=2, name='低流动性',
        description='成交额（vol×close）低于阈值 → 不可交易',
        condition=_low_volume,
        action=_act_low_liquidity,
        gate=2,
    ),
    # ── R19: 三金叉共振 ────────────────────────────────────────────────────
    Rule(
        id='R19', priority=2, name='MACD+KDJ+RSI共振',
        description='MACD金叉 + KDJ金叉 + RSI未超买 → 共振加分',
        condition=lambda s: _macd_golden_cross(s) and _kdj_golden_cross(s) and _rsi_not_overbought(s),
        action=_act_triple_resonance_boost,
        gate=1,
    ),
    # ── R20: 底背离量价共振 ───────────────────────────────────────────────
    Rule(
        id='R20', priority=3, name='底背离量价共振',
        description='底背离存在 + 量价得分 > 0 → 共振加分',
        condition=_has_bot_divergence_with_volume,
        action=_act_bottom_divergence_volume_boost,
        gate=3,
    ),
    # ── R21: 多头排列 + MACD 超强 → A 级 ──────────────────────────────────
    Rule(
        id='R21', priority=2, name='多头共振推A级',
        description='均线多头排列 + MACD超强 → 强制A级',
        condition=lambda s: _ma_bullish_alignment(s) and s.get('macd_trend', '') == '指标超强',
        action=_act_force_level_a,
        gate=3,
    ),
    # ── R22: 金叉滞涨衰减 ─────────────────────────────────────────────────
    Rule(
        id='R22', priority=3, name='金叉钝化',
        description='金叉发生 N 天后价格无明显上涨 → 信号衰减扣分',
        condition=_golden_cross_stagnant,
        action=_act_golden_cross_stagnant,
        gate=2,
    ),
    # ── R25: 业绩预告期间 ─────────────────────────────────────────────────
    Rule(
        id='R25', priority=3, name='业绩预告',
        description='有业绩预告 → 延迟入场提示',
        condition=_has_forecast,
        action=_act_forecast_note,
        gate=2,
    ),
    # ── R26: 高波动趋势市 → 趋势强化 ──────────────────────────────────────
    Rule(
        id='R26', priority=4, name='趋势市优先',
        description='ATR↑30%+ADX>25 → 优先趋势突破信号',
        condition=lambda s: _vol_regime_is(s, 'HIGH_VOL_TREND'),
        action=_act_vol_trend_boost,
        gate=3,
    ),
    # ── R27: 低波动震荡市 → 反转强化 ──────────────────────────────────────
    Rule(
        id='R27', priority=4, name='反转市优先',
        description='ATR↓30%+ADX<20 → 优先反转底部信号',
        condition=lambda s: _vol_regime_is(s, 'LOW_VOL_REVERSAL'),
        action=_act_vol_reversal_boost,
        gate=3,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Gate 0:  数据质量预筛
    # Gate 0.5: 宏观情景注入
    # Gate 4:  仓位联动
    # ═══════════════════════════════════════════════════════════════════════

    # ── R30: 数据不足 ──────────────────────────────────────────────────────
    Rule(
        id='R30', priority=1, name='数据不足',
        description='K线长度 < 60 → 直接否决',
        condition=_kline_too_short,
        action=_act_data_insufficient,
        gate=0,
    ),
    # ── R31: 无波动率 ─────────────────────────────────────────────────────
    Rule(
        id='R31', priority=1, name='无波动率',
        description='ATR/close 缺失或为零 → score=0',
        condition=_atr_missing,
        action=_act_no_volatility,
        gate=0,
    ),
    # ── R32: 无长期趋势 ─────────────────────────────────────────────────
    Rule(
        id='R32', priority=5, name='无长期趋势',
        description='MA_60 缺失 → 标记备注',
        condition=_ma60_missing,
        action=_act_no_long_term_trend,
        gate=0,
    ),
    # ── R33: 无成交量 ─────────────────────────────────────────────────────
    Rule(
        id='R33', priority=3, name='无成交量',
        description='近5日成交量全零 → 中等风险',
        condition=_volume_empty,
        action=_act_no_volume,
        gate=0,
    ),
    # ── R34: 大盘弱势降级 ────────────────────────────────────────────────
    Rule(
        id='R34', priority=3, name='大盘弱势降级',
        description='WEAK_TREND + score>60 → B降C',
        condition=_macro_weak_with_high_score,
        action=_act_macro_weak_downgrade,
        gate=0.5,
    ),
    # ── R35: 大盘强势升级 ────────────────────────────────────────────────
    Rule(
        id='R35', priority=3, name='大盘强势升级',
        description='STRONG_TREND + score>80 → B升A',
        condition=_macro_strong_with_high_score,
        action=_act_macro_strong_upgrade,
        gate=0.5,
    ),
    # ── R36: 多因子共振 ────────────────────────────────────────────────────
    Rule(
        id='R36', priority=4, name='多因子共振',
        description='MACD/KDJ/CCI/RSI 中任意3个看涨 → +5分',
        condition=lambda s: _multi_indicator_aligned(s, required=3),
        action=_act_multi_factor_boost,
        gate=1,
    ),
    # ── R37: KDJ+RSI+量价共振 ────────────────────────────────────────────
    Rule(
        id='R37', priority=4, name='KDJ+RSI共振',
        description='KDJ金叉 + RSI>50 + 放量 → +5分',
        condition=_macd_kdj_rsi_all_bullish,
        action=_act_kdj_rsi_volume_boost,
        gate=1,
    ),
    # ── R38: 日周共振 ──────────────────────────────────────────────────────
    Rule(
        id='R38', priority=3, name='日周共振多头',
        description='日线+周线MACD同步多头 → +10分',
        condition=_multitimeframe_aligned_bull,
        action=_act_multitimeframe_boost,
        gate=1,
    ),
    # ── R39: 四指标多数同向 → 升级 ────────────────────────────────────────
    Rule(
        id='R39', priority=3, name='多指标升级',
        description='MACD/KDJ/CCI/RSI 中任意3个看涨 → B级',
        condition=lambda s: _multi_indicator_aligned(s, required=3) and state_highest_risk_not_high(s),
        action=_act_four_indicator_majority,
        gate=1,
    ),
    # ── R40: 极端波动 ──────────────────────────────────────────────────────
    Rule(
        id='R40', priority=3, name='极端波动',
        description='ATR/价格 > 8% → 中等风险',
        condition=_extreme_volatility,
        action=_act_extreme_vol_risk,
        gate=2,
    ),
    # ── R41: 顶背离+放量下跌 ──────────────────────────────────────────────
    Rule(
        id='R41', priority=2, name='顶背离放量下跌',
        description='顶背离 + 放量下跌 → 直接否决',
        condition=_top_divergence_volume_down,
        action=_act_top_divergence_volume_down,
        gate=2,
    ),
    # ── R42: 动能衰竭 ─────────────────────────────────────────────────────
    Rule(
        id='R42', priority=3, name='动能衰竭',
        description='柱状连续5期缩短 + DIF斜率拐头 → 降级',
        condition=_momentum_exhausting,
        action=_act_momentum_exhaustion,
        gate=2,
    ),
    # ── R43: 均线空头排列 ────────────────────────────────────────────────
    Rule(
        id='R43', priority=3, name='均线空头排列',
        description='close < MA20 < MA60 → 中等风险',
        condition=_price_below_ma20_ma60,
        action=_act_price_below_ma20_ma60,
        gate=2,
    ),
    # ── R44: 流动性枯竭 ─────────────────────────────────────────────────
    Rule(
        id='R44', priority=2, name='流动性枯竭',
        description='AMOUNT/AMOUNT_MA20 < liq_veto_ratio → 直接否决',
        condition=_liquidity_crisis,
        action=_act_liquidity_crisis,
        gate=2,
    ),
    # ── R45: 振幅极端 ────────────────────────────────────────────────────
    Rule(
        id='R45', priority=3, name='振幅极端',
        description='日内振幅 > 近60日99分位 → 延迟入场',
        condition=_amplitude_extreme_99,
        action=_act_amplitude_extreme_delay,
        gate=2,
    ),
    # ── R46: 资金确认看涨 ────────────────────────────────────────────────
    Rule(
        id='R46', priority=4, name='资金确认看涨',
        description='资金净流入 + MACD看涨 → +3分',
        condition=_moneyflow_positive_with_macd_bullish,
        action=_act_moneyflow_confirm_bullish,
        gate=3,
    ),
    # ── R47: 资金确认看跌 ────────────────────────────────────────────────
    Rule(
        id='R47', priority=4, name='资金确认看跌',
        description='资金净流出 + MACD看跌 → -3分',
        condition=_moneyflow_negative_with_macd_bearish,
        action=_act_moneyflow_confirm_bearish,
        gate=3,
    ),
    # ── R48: 量价趋势健康 ────────────────────────────────────────────────
    Rule(
        id='R48', priority=4, name='量价健康',
        description='close>MA60 + vol>MA5均量 → +5分',
        condition=_volume_price_healthy,
        action=_act_volume_price_healthy_boost,
        gate=3,
    ),
    # ── R49: 筹码底部确认 ────────────────────────────────────────────────
    Rule(
        id='R49', priority=3, name='筹码底部确认',
        description='获利<20% + 底背离 → +8分',
        condition=_chip_bottom_with_divergence,
        action=_act_chip_bottom_confirm,
        gate=3,
    ),
    # ── R50: 三连阳量增 ──────────────────────────────────────────────────
    Rule(
        id='R50', priority=4, name='三连阳量增',
        description='3连阳 + 成交量依次递增 → +5分',
        condition=_three_green_candles_volume_up,
        action=_act_three_green_strength,
        gate=3,
    ),
    # ── R51: HIGH风险 → 仓位归零 ─────────────────────────────────────────
    Rule(
        id='R51', priority=1, name='仓位归零',
        description='risk_level=HIGH → position_adjust=-1.0',
        condition=_risk_high_pos,
        action=_act_position_zero,
        gate=4,
    ),
    # ── R52: MEDIUM风险低分 → 仓位砍半 ──────────────────────────────────
    Rule(
        id='R52', priority=2, name='仓位砍半',
        description='risk_level=MEDIUM + score<60 → position_adjust=-0.5',
        condition=_risk_medium_score_low_pos,
        action=_act_position_half,
        gate=4,
    ),
    # ── R53: 震荡低分 → 减仓 ────────────────────────────────────────────
    Rule(
        id='R53', priority=3, name='震荡减仓',
        description='OSCILLATION + score<50 → position_adjust=-0.3',
        condition=_oscillation_score_low_pos,
        action=_act_position_reduce_30,
        gate=4,
    ),
    # ── R54: 底背离底部反转 → 加仓 ─────────────────────────────────────
    Rule(
        id='R54', priority=4, name='底背离加仓',
        description='底背离 + BOTTOM_REVERSAL → position_adjust=+0.2',
        condition=_bot_div_bottom_reversal_pos,
        action=_act_position_add_20,
        gate=4,
    ),
    # ── R55: K线强反转+放量 → 加仓 ─────────────────────────────────────
    Rule(
        id='R55', priority=4, name='反转加仓',
        description='K线强反转 + 放量 → position_adjust=+0.15',
        condition=_kline_strong_reversal_volume_pos,
        action=_act_position_add_15,
        gate=4,
    ),
    # ── R56: 高波动 → 减仓 ──────────────────────────────────────────────
    Rule(
        id='R56', priority=4, name='高波动减仓',
        description='ATR/价格 > 5% → position_adjust=-0.25',
        condition=_high_vol_atr_pos,
        action=_act_position_reduce_25,
        gate=4,
    ),
]


def state_highest_risk_not_high(s: dict) -> bool:
    """辅助条件：风险不为 HIGH"""
    return s.get('risk_level') != 'HIGH'


def get_rules_by_gate(gate: int) -> list[Rule]:
    """获取指定 Gate 的规则，按优先级排序。"""
    return sorted([r for r in RULES if r.gate == gate], key=lambda x: x.priority)


def execute_rules(state: dict, gate: int) -> None:
    """
    执行指定 Gate 的所有规则。

    规则按优先级执行：优先级 1 先执行。
    高优先级规则触发的 action 可能会影响 state，进而影响后续规则的条件判断。
    """
    for rule in get_rules_by_gate(gate):
        try:
            if rule.condition(state):
                rule.action(state)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
            logger.warning(f"规则 {rule.name}(id={rule.id}) 执行失败: {e}")
