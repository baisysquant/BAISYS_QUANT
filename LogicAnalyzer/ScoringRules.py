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
  }
"""

from dataclasses import dataclass, field
from typing import Any, Callable


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
    """顶背离 + 强度 > 0.3"""
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    strength = div.get('strength_12269', 0) or div.get('strength_6135', 0)
    return '顶背离' in cs and strength > 0.3


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
    detail = df['MACD_12269_SIGNAL_DETAIL'].iloc[-1]
    return '金叉' in str(detail)


def _macd_above_zero(state: dict) -> bool:
    """DIF 在零轴上"""
    df = state.get('df')
    if df is None:
        return False
    return df['DIF_12269'].iloc[-1] > 0


def _volume_positive(state: dict) -> bool:
    """量价得分 > 0"""
    vt = state.get('volume_trend')
    if not vt or len(vt) < 2:
        return False
    return vt[1] > 0


def _price_new_high(state: dict) -> bool:
    """股价创 N 日新高（20 日）"""
    df = state.get('df')
    if df is None or len(df) < 20:
        return False
    recent = df['close'].iloc[-20:]
    return recent.iloc[-1] >= recent.max()


def _rsi_not_new_high(state: dict) -> bool:
    """RSI 未创新高"""
    df = state.get('df')
    if df is None or len(df) < 20:
        return False
    rsi_cols = [c for c in df.columns if c.startswith('RSI_')]
    if not rsi_cols:
        return False
    rsi = df[rsi_cols[0]].iloc[-20:]
    return rsi.iloc[-1] < rsi.max()


def _volume_not_new_high(state: dict) -> bool:
    """成交量未创新高"""
    df = state.get('df')
    if df is None or len(df) < 20 or 'volume' not in df.columns:
        return False
    vol = df['volume'].iloc[-20:]
    return vol.iloc[-1] < vol.max()


def _momentum_decreasing(state: dict) -> bool:
    """动能连续 3 期递减"""
    df = state.get('df')
    if df is None or len(df) < 5:
        return False
    hist = df['DIF_12269'] - df['DEA_12269']
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
    """底背离 + 强度 > 0.3"""
    div = state.get('divergence') or {}
    cs = div.get('combined_signal', '')
    strength = div.get('strength_12269', 0) or div.get('strength_6135', 0)
    return '底背离' in cs and strength > 0.3


def _regime_transition_to_strong(state: dict) -> bool:
    """情景从震荡切换为强势趋势"""
    # 简化：当前是强势趋势且之前是震荡（由外部调用者判断）
    return state.get('regime') == 'STRONG_TREND'


def _chip_winner_rate_high(state: dict, threshold: float = 80) -> bool:
    """获利比例 > threshold"""
    chip = state.get('chip_data')
    if not chip:
        return False
    return chip.get('winner_rate', 0) > threshold

def _chip_winner_rate_low(state: dict, threshold: float = 15) -> bool:
    """获利比例 < threshold"""
    chip = state.get('chip_data')
    if not chip:
        return False
    return chip.get('winner_rate', 0) < threshold

def _chip_price_at_resistance(state: dict) -> bool:
    """收盘价接近成本上沿 (cost_95pct)"""
    chip = state.get('chip_data')
    df = state.get('df')
    if not chip or df is None:
        return False
    close = df['close'].iloc[-1]
    cost_95 = chip.get('cost_95pct', 0)
    return cost_95 > 0 and close >= cost_95 * 0.95

def _chip_cost_concentrated(state: dict) -> bool:
    """筹码高度集中（(cost_95pct - cost_5pct) / cost_5pct < 0.15）"""
    chip = state.get('chip_data')
    if not chip:
        return False
    c5 = chip.get('cost_5pct', 0)
    c95 = chip.get('cost_95pct', 0)
    if c5 <= 0 or c95 <= 0:
        return False
    return (c95 - c5) / c5 < 0.15


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
]


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
        except Exception:
            pass
