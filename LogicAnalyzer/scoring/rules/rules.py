"""
规则定义：Rule 数据类型 + RULES 列表 + 执行入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from LogicAnalyzer.scoring.rules.base import (  # noqa: F401, F403
    _abnormal_amplitude, _adx_below_threshold, _amplitude_extreme_99,
    _atr_missing, _boll_bandwidth_narrowing_then_expanding,
    _chip_bottom_with_divergence, _chip_price_at_resistance,
    _chip_winner_rate_high, _extreme_volatility, _far_from_ma200,
    _golden_cross_stagnant, _has_bearish_kline_medium, _has_bearish_kline_strong,
    _has_bot_divergence, _has_bot_divergence_with_volume, _has_forecast,
    _has_top_divergence, _high_volatility_atr, _kdj_golden_cross,
    _kline_too_short, _kline_unconfirmed_bullish, _liquidity_crisis,
    _low_volume, _ma60_missing, _ma_bearish_alignment, _ma_bullish_alignment,
    _macd_above_zero, _macd_golden_cross, _macd_kdj_rsi_all_bullish,
    _macro_strong_with_high_score, _macro_weak_with_high_score,
    _momentum_decreasing, _momentum_exhausting, _moneyflow_negative_with_macd_bearish,
    _moneyflow_positive_with_macd_bullish, _multi_indicator_aligned,
    _multitimeframe_aligned_bull, _price_below_ma20_ma60, _price_new_high,
    _regime_is, _regime_transition_to_strong, _rsi_not_new_high,
    _rsi_not_overbought, _score_above_oscillate, _three_green_candles_volume_up,
    _top_divergence_volume_down, _vol_regime_is, _volume_empty, _volume_expanding,
    _volume_not_new_high, _volume_positive, _volume_price_healthy, _volume_shrinking,
    _act_terminate_top_risk, _act_discount_kline_in_trend, _act_boost_bottom_resonance,
    _act_boost_golden_volume, _act_fake_breakout_warning, _act_breakout_start,
    _act_momentum_decay, _act_wait_kline_confirm, _act_chip_high_winner_risk,
    _act_chip_resistance_risk, _act_ma_conflict_downgrade,
    _act_fake_breakout_warning_adx, _act_far_from_ma200, _act_high_volatility,
    _act_abnormal_amplitude, _act_boll_breakout_boost, _act_low_liquidity,
    _act_triple_resonance_boost, _act_bottom_divergence_volume_boost,
    _act_force_level_a, _act_golden_cross_stagnant, _act_forecast_note,
    _act_vol_trend_boost, _act_vol_reversal_boost, _act_data_insufficient,
    _act_no_volatility, _act_no_long_term_trend, _act_no_volume,
    _act_macro_weak_downgrade, _act_macro_strong_upgrade, _act_multi_factor_boost,
    _act_kdj_rsi_volume_boost, _act_multitimeframe_boost, _act_four_indicator_majority,
    _act_extreme_vol_risk, _act_top_divergence_volume_down, _act_momentum_exhaustion,
    _act_price_below_ma20_ma60, _act_liquidity_crisis, _act_amplitude_extreme_delay,
    _act_moneyflow_confirm_bullish, _act_moneyflow_confirm_bearish,
    _act_volume_price_healthy_boost, _act_chip_bottom_confirm,
    _act_three_green_strength, _act_position_zero, _act_position_half,
    _act_position_reduce_30, _act_position_add_20, _act_position_add_15,
    _act_position_reduce_25, _dif_slope_turning, _has_bullish_kline_strong,
    _chip_winner_rate_low, _chip_cost_concentrated, _risk_high_pos,
    _medium_score_low_pos, _oscillation_score_low_pos, _bot_div_bottom_reversal_pos,
    _kline_strong_reversal_volume_pos, _high_vol_atr_pos, state_highest_risk_not_high,
)


# ═══════════════════════════════════════════════════════════════
# Rule 数据类型
# ═══════════════════════════════════════════════════════════════

@dataclass
class Rule:
    id: str
    priority: int                         # 1=最高, 5=最低
    name: str
    description: str
    condition: Callable[[dict], bool]     # 接收 state，返回 True/False
    action: Callable[[dict], None]        # 接收 state，修改之
    gate: int = 1                         # 所属 Gate (0~4)


# ═══════════════════════════════════════════════════════════════
# RULES — 全量规则列表
# ═══════════════════════════════════════════════════════════════

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
            _regime_is(s, 'WEAK') or _regime_is(s, 'TOP_RISK')
        ),
        action=_act_chip_high_winner_risk,
        gate=2,
    ),
    # ── R10: 筹码密集峰阻力 ─────────────────────────────────────────────────
    Rule(
        id='R10', priority=3, name='筹码密集峰阻力',
        description='价格触及95%筹码线+获利比例>70%',
        condition=lambda s: _chip_price_at_resistance(s) and _chip_winner_rate_high(s, 70),
        action=_act_chip_resistance_risk,
        gate=2,
    ),
    # ── R11: MA空头+评分≥40 → D级 ─────────────────────────────────────────
    Rule(
        id='R11', priority=1, name='MA空头评分较高',
        description='均线空头排列+评分≥40 → 直接降为D级',
        condition=lambda s: _ma_bearish_alignment(s) and _score_above_oscillate(s),
        action=_act_ma_conflict_downgrade,
        gate=2,
    ),
    # ── R12: ADX低值假突破 ─────────────────────────────────────────────────
    Rule(
        id='R12', priority=3, name='ADX低值假突破',
        description='评分≥40但ADX<25，趋势强度不足→预警',
        condition=lambda s: _score_above_oscillate(s) and _adx_below_threshold(s),
        action=_act_fake_breakout_warning_adx,
        gate=2,
    ),
    # ── R13: 远离MA200 → 回调预警 ──────────────────────────────────────────
    Rule(
        id='R13', priority=4, name='远离MA200',
        description='价格偏离MA200超过30%',
        condition=_far_from_ma200,
        action=_act_far_from_ma200,
        gate=3,
    ),
    # ── R14: 高波动率 → 中等风险 ───────────────────────────────────────────
    Rule(
        id='R14', priority=5, name='高波动率',
        description='ATR/close > 6%',
        condition=_high_volatility_atr,
        action=_act_high_volatility,
        gate=3,
    ),
    # ── R15: 异常振幅 ──────────────────────────────────────────────────────
    Rule(
        id='R15', priority=5, name='异常振幅',
        description='当日振幅>15%',
        condition=_abnormal_amplitude,
        action=_act_abnormal_amplitude,
        gate=2,
    ),
    # ── R16: 布林突破 ──────────────────────────────────────────────────────
    Rule(
        id='R16', priority=4, name='布林带突破',
        description='带宽先缩后扩 → 突破启动',
        condition=_boll_bandwidth_narrowing_then_expanding,
        action=_act_boll_breakout_boost,
        gate=3,
    ),
    # ── R17: 低流动性 ──────────────────────────────────────────────────────
    Rule(
        id='R17', priority=5, name='低流动性',
        description='5日均量<20日均量50%',
        condition=_low_volume,
        action=_act_low_liquidity,
        gate=2,
    ),
    # ── R19: 三指标共振 ────────────────────────────────────────────────────
    Rule(
        id='R19', priority=3, name='三指标共振',
        description='KDJ金叉+RSI未超买+MACD金叉 → 可信度大幅提升',
        condition=lambda s: _kdj_golden_cross(s) and _rsi_not_overbought(s) and _macd_golden_cross(s),
        action=_act_triple_resonance_boost,
        gate=1,
    ),
    # ── R20: 底背离+放量 ────────────────────────────────────────────────────
    Rule(
        id='R20', priority=4, name='底背离放量',
        description='底背离+当日放量 → 反转可信度+1',
        condition=_has_bot_divergence_with_volume,
        action=_act_bottom_divergence_volume_boost,
        gate=3,
    ),
    # ── R21: 均线多头+评分≥60 → A级 ─────────────────────────────────────────
    Rule(
        id='R21', priority=3, name='均线多头助推',
        description='均线多头排列+评分≥60 → 升A级',
        condition=lambda s: _ma_bullish_alignment(s) and state_highest_risk_not_high(s),
        action=_act_force_level_a,
        gate=3,
    ),
    # ── R22: 金叉僵化 ────────────────────────────────────────────────────
    Rule(
        id='R22', priority=4, name='金叉僵化',
        description='金叉后DIF/DEA走平 → 动能衰减',
        condition=_golden_cross_stagnant,
        action=_act_golden_cross_stagnant,
        gate=2,
    ),
    # ── R25: 业绩预告 ────────────────────────────────────────────────────────
    Rule(
        id='R25', priority=5, name='业绩预告',
        description='有业绩预告',
        condition=_has_forecast,
        action=_act_forecast_note,
        gate=2,
    ),
    # ── R26: 高波动趋势 → 加分 ────────────────────────────────────────────
    Rule(
        id='R26', priority=4, name='高波动趋势加分',
        description='高波动率趋势(趋势跟踪策略)',
        condition=lambda s: _vol_regime_is(s, 'HIGH_VOL_TREND'),
        action=_act_vol_trend_boost,
        gate=3,
    ),
    # ── R27: 低波动反转 → 加分 ─────────────────────────────────────────────
    Rule(
        id='R27', priority=4, name='低波动反转加分',
        description='低波动率反转(均值回归策略)',
        condition=lambda s: _vol_regime_is(s, 'LOW_VOL_REVERSAL'),
        action=_act_vol_reversal_boost,
        gate=3,
    ),
    # ── R30: 数据不足 ──────────────────────────────────────────────────────
    Rule(
        id='R30', priority=1, name='数据不足',
        description='K线数据太少 → 直接否决',
        condition=_kline_too_short,
        action=_act_data_insufficient,
        gate=0,
    ),
    # ── R31: 无ATR ───────────────────────────────────────────────────────
    Rule(
        id='R31', priority=2, name='无ATR',
        description='ATR数据缺失 → 标记无波动',
        condition=_atr_missing,
        action=_act_no_volatility,
        gate=0,
    ),
    # ── R32: 无MA60 ─────────────────────────────────────────────────────
    Rule(
        id='R32', priority=2, name='无MA60',
        description='60日均线缺失 → 标记无长期趋势',
        condition=_ma60_missing,
        action=_act_no_long_term_trend,
        gate=0,
    ),
    # ── R33: 无量 ─────────────────────────────────────────────────────────
    Rule(
        id='R33', priority=2, name='无成交量',
        description='成交量数据为空 → 标记无交易',
        condition=_volume_empty,
        action=_act_no_volume,
        gate=0,
    ),
    # ── R34: 宏观弱 + 评分高 → 降级 ───────────────────────────────────────
    Rule(
        id='R34', priority=3, name='宏观弱评分高',
        description='宏观环境弱但评分高 → 降分',
        condition=_macro_weak_with_high_score,
        action=_act_macro_weak_downgrade,
        gate=0,
    ),
    # ── R35: 宏观强 + 评分高 → 加分 ─────────────────────────────────────
    Rule(
        id='R35', priority=3, name='宏观强评分高',
        description='宏观环境强且评分高 → 加分',
        condition=_macro_strong_with_high_score,
        action=_act_macro_strong_upgrade,
        gate=0,
    ),
    # ── R36: 多因子共振 → 加分 ──────────────────────────────────────────────
    Rule(
        id='R36', priority=3, name='多因子共振',
        description='3项指标共振',
        condition=lambda s: _multi_indicator_aligned(s, 3),
        action=_act_multi_factor_boost,
        gate=1,
    ),
    # ── R37: MACD+KDJ+RSI全看涨 → 加分 ─────────────────────────────────
    Rule(
        id='R37', priority=4, name='三指标全看涨',
        description='MACD金叉+KDJ金叉+RSI看涨',
        condition=_macd_kdj_rsi_all_bullish,
        action=_act_kdj_rsi_volume_boost,
        gate=1,
    ),
    # ── R38: 多周期共振 → 加分 ────────────────────────────────────────────
    Rule(
        id='R38', priority=4, name='多周期共振',
        description='周线+日线均看涨',
        condition=_multitimeframe_aligned_bull,
        action=_act_multitimeframe_boost,
        gate=1,
    ),
    # ── R39: 四指标共振(再加5分) ─────────────────────────────────────────
    Rule(
        id='R39', priority=4, name='四指标共振',
        description='4项指标共振+风险非HIGH → 再加5分',
        condition=lambda s: _multi_indicator_aligned(s, 4) and state_highest_risk_not_high(s),
        action=_act_four_indicator_majority,
        gate=1,
    ),
    # ── R40: 极端波动率 → 降级 ──────────────────────────────────────────────
    Rule(
        id='R40', priority=3, name='极端波动率',
        description='ATR/Close > 8% → 中等风险降分',
        condition=_extreme_volatility,
        action=_act_extreme_vol_risk,
        gate=2,
    ),
    # ── R41: 顶背离缩量 → 降分 ─────────────────────────────────────────────
    Rule(
        id='R41', priority=3, name='顶背离缩量',
        description='顶背离+缩量 → 见顶信号降分',
        condition=_top_divergence_volume_down,
        action=_act_top_divergence_volume_down,
        gate=2,
    ),
    # ── R42: 动量衰竭 → 降分 ───────────────────────────────────────────────
    Rule(
        id='R42', priority=4, name='动量衰竭',
        description='MACD柱连续4期递减 → 动量衰减',
        condition=_momentum_exhausting,
        action=_act_momentum_exhaustion,
        gate=2,
    ),
    # ── R43: 价格在MA20/MA60下方 → 降分 ───────────────────────────────────
    Rule(
        id='R43', priority=2, name='价格在均线下方',
        description='价格低于MA20或MA60 → 趋势偏空',
        condition=_price_below_ma20_ma60,
        action=_act_price_below_ma20_ma60,
        gate=2,
    ),
    # ── R44: 流动性枯竭 → 否决 ────────────────────────────────────────────
    Rule(
        id='R44', priority=1, name='流动性枯竭',
        description='近5日均量<20日均量30% → 直接否决',
        condition=_liquidity_crisis,
        action=_act_liquidity_crisis,
        gate=2,
    ),
    # ── R45: 振幅极端 → 降分 ────────────────────────────────────────────────
    Rule(
        id='R45', priority=4, name='振幅极端',
        description='当日振幅超过99分位线',
        condition=_amplitude_extreme_99,
        action=_act_amplitude_extreme_delay,
        gate=2,
    ),
    # ── R46: 资金净流入+MACD看涨 → 加分 ─────────────────────────────────
    Rule(
        id='R46', priority=4, name='资金流入确认',
        description='资金净流入+MACD金叉 → 看涨确认',
        condition=_moneyflow_positive_with_macd_bullish,
        action=_act_moneyflow_confirm_bullish,
        gate=3,
    ),
    # ── R47: 资金净流出+MACD看跌 → 降分 ─────────────────────────────────
    Rule(
        id='R47', priority=4, name='资金流出确认',
        description='资金净流出+MACD死叉 → 看跌确认',
        condition=_moneyflow_negative_with_macd_bearish,
        action=_act_moneyflow_confirm_bearish,
        gate=3,
    ),
    # ── R48: 量价健康 → 加分 ──────────────────────────────────────────────
    Rule(
        id='R48', priority=5, name='量价健康加分',
        description='量价得分>5% → 资金认可',
        condition=_volume_price_healthy,
        action=_act_volume_price_healthy_boost,
        gate=3,
    ),
    # ── R49: 筹码底部+底背离 → 加分 ──────────────────────────────────────
    Rule(
        id='R49', priority=4, name='筹码底部背离',
        description='获利比例<30%+底背离 → 底部确认',
        condition=_chip_bottom_with_divergence,
        action=_act_chip_bottom_confirm,
        gate=3,
    ),
    # ── R50: 三连阳放量 → 加分 ──────────────────────────────────────────────
    Rule(
        id='R50', priority=4, name='三连阳放量',
        description='三连阳+当日放量 → 多方强势',
        condition=_three_green_candles_volume_up,
        action=_act_three_green_strength,
        gate=3,
    ),
    # ── R51: 高风险 → 零仓位 ────────────────────────────────────────────────
    Rule(
        id='R51', priority=1, name='高风险零仓位',
        description='风险等级=HIGH → 不持仓',
        condition=_risk_high_pos,
        action=_act_position_zero,
        gate=4,
    ),
    # ── R52: 中风险+低评分 → 半仓 ──────────────────────────────────────────
    Rule(
        id='R52', priority=2, name='中风险减仓',
        description='风险MEDIUM+评分<40 → 半仓',
        condition=_medium_score_low_pos,
        action=_act_position_half,
        gate=4,
    ),
    # ── R53: 震荡+低评分 → 减仓30% ───────────────────────────────────────
    Rule(
        id='R53', priority=3, name='震荡减仓',
        description='震荡/弱势+评分<50 → 减30%',
        condition=_oscillation_score_low_pos,
        action=_act_position_reduce_30,
        gate=4,
    ),
    # ── R54: 底背离+底部反转 → 加仓20% ─────────────────────────────────
    Rule(
        id='R54', priority=3, name='底背离加仓',
        description='底背离+底部反转情景 → 加20%',
        condition=_bot_div_bottom_reversal_pos,
        action=_act_position_add_20,
        gate=4,
    ),
    # ── R55: 强反转K线+放量 → 加仓15% ──────────────────────────────────
    Rule(
        id='R55', priority=4, name='强反转加仓',
        description='强看涨K线+放量 → 加15%',
        condition=_kline_strong_reversal_volume_pos,
        action=_act_position_add_15,
        gate=4,
    ),
    # ── R56: 高波动率 → 减仓25% ─────────────────────────────────────────
    Rule(
        id='R56', priority=4, name='高波动减仓',
        description='ATR/close>5% → 减25%',
        condition=_high_vol_atr_pos,
        action=_act_position_reduce_25,
        gate=4,
    ),
]


# ═══════════════════════════════════════════════════════════════
# 执行入口
# ═══════════════════════════════════════════════════════════════

def get_rules_by_gate(gate: int) -> list[Rule]:
    """获取指定 Gate 的规则，按优先级排序。"""
    return sorted([r for r in RULES if r.gate == gate], key=lambda x: x.priority)


def execute_rules(state: dict, gate: int) -> None:
    """执行指定 Gate 的所有规则。"""
    for rule in get_rules_by_gate(gate):
        try:
            if rule.condition(state):
                rule.action(state)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
            logger.warning(f"规则 {rule.name}(id={rule.id}) 执行失败: {e}")
