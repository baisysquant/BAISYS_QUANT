import numpy as np
import pandas as pd

from LogicAnalyzer.MACDDivergence import (
    adaptive_distance, detect_combined_divergence, slope_analysis,
)
from LogicAnalyzer.SignalConstants import (
    MACDSignals, MACDMomentum, Divergence, TrendLevels,
    CombinedSignal, FullBullScoring, Conclusion, KLineLevels
)


class MACDAnalyzer:
    """
    双参数 MACD 分析器（标准 12-26-9 + 可配置第二周期）。

    子模块委托：
      - MACDDivergence: 背离检测
      - FullBullScorer: 完全多头评分
      - MACDHelpers: 斜率分析、动能评分等共享工具
    """

    def _custom_macd(self, df: pd.DataFrame, second_params: tuple[int, int, int] = (6, 13, 5)) -> pd.DataFrame:
        """
        同时计算标准 (12, 26, 9) 和第二周期两套 MACD。

        参数：
          df: 包含 close 列的 DataFrame
          second_params: 第二周期的 (快线, 慢线, 信号线)，默认 (6, 13, 5)
                        此参数为必填项，不可为 None 或 (0,0,0)

        新增：
          - MACD_HIST_{name}  柱状值（×2 标准显示）
          - MACD_{name}_CROSS  1=金叉 / -1=死叉 / 0=无
          - MACD_{name}_SIGNAL_DETAIL  零轴上/下金叉/死叉文字
        """
        if "close" not in df.columns:
            return df

        close = df["close"]

        # 标准周期（强制保留）
        macd_periods = {
            "12269": (12, 26, 9),
        }

        # 添加第二周期（必填）
        fast, slow, signal = second_params
        name = f"{fast}{slow}{signal}"
        macd_periods[name] = second_params

        for name, (fast, slow, signal) in macd_periods.items():
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=signal, adjust=False).mean()

            df[f"DIF_{name}"] = dif
            df[f"DEA_{name}"] = dea
            df[f"MACD_HIST_{name}"] = 2 * (dif - dea)

            prev_dif = dif.shift(1).fillna(dea.shift(1).fillna(0))
            prev_dea = dea.shift(1).fillna(0)

            golden = (dif > dea) & (prev_dif <= prev_dea)
            dead = (dif < dea) & (prev_dif >= prev_dea)

            df[f"MACD_{name}_SIGNAL_DETAIL"] = np.where(
                golden,
                MACDSignals.golden_cross_label(dif, dea),
                MACDSignals.death_cross_label(dead, dif, dea),
            )
            df[f"MACD_{name}_CROSS"] = np.where(golden, 1, np.where(dead, -1, 0))

        return df

    @staticmethod
    def _calculate_macd_momentum(df: pd.DataFrame, dif_col: str, dea_col: str) -> str:
        if len(df) < 2:
            return "N/A (数据不足)"
        latest_dif = df[dif_col].iloc[-1]
        latest_dea = df[dea_col].iloc[-1]
        dif_change = latest_dif - df[dif_col].iloc[-2]
        if latest_dif >= latest_dea:
            return MACDMomentum.ACCELERATE_UP if dif_change > 0 else MACDMomentum.DECELERATE_UP
        return MACDMomentum.ACCELERATE_DOWN if dif_change < 0 else MACDMomentum.DECELERATE_DOWN

    def pipeline_analysis(
        self,
        df: pd.DataFrame,
        second_params: tuple[int, int, int] = (6, 13, 5),
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
        rule_thresholds: dict | None = None,
    ) -> dict:
        from LogicAnalyzer.ScoringRules import execute_rules

        if weights is None:
            weights = {"零轴条件": 20, "战略金叉": 15, "战术金叉": 10, "动能": 15,
                       "DIF斜率": 10, "背离信号": 10, "量价配合": 10, "K线形态": 10}
        if thresholds is None:
            thresholds = {"fully_bull": 80, "bullish": 60, "oscillate": 40}
        if rule_thresholds is None:
            rule_thresholds = {"divergence": 0.3, "winner_rate_high": 80,
                               "cost_resistance_ratio": 0.95, "chip_concentrated_ratio": 0.15,
                               "price_new_high_days": 20}

        fast, slow, signal = second_params
        second_period_name = f"{fast}{slow}{signal}"

        if 'MA_5' not in df.columns:
            for p in [5, 10, 20, 30, 60]:
                df[f'MA_{p}'] = df['close'].rolling(p).mean()

        boll_bw_col = None
        boll_upper_cols = [c for c in df.columns if c.startswith('BBU_')]
        boll_lower_cols = [c for c in df.columns if c.startswith('BBL_')]
        if boll_upper_cols and boll_lower_cols:
            bw_col = 'BOLL_BANDWIDTH'
            if bw_col not in df.columns:
                df[bw_col] = (df[boll_upper_cols[0]] - df[boll_lower_cols[0]]) / df['close']
            boll_bw_col = bw_col

        dist_slow = adaptive_distance(df, base=10)
        dist_fast = adaptive_distance(df, base=5)
        div_signals = detect_combined_divergence(
            df, distance_slow=dist_slow, distance_fast=dist_fast,
            second_period_name=second_period_name,
        )
        mom_desc, mom_score = _calc_momentum_desc(df, 'DIF_12269', 'DEA_12269', max_score=20)
        slope_info = slope_analysis(df['DIF_12269'], window=5)
        vol_desc, vol_score = _volume_price_trend_score(df, max_bonus=10)
        kp_result = _score_kline_pattern(df, max_score=10)

        state = {
            'df': df, 'regime': None, 'signal_list': [],
            'risk_level': 'NONE', 'risk_desc': '', 'conclusion': '',
            'level': 'C', 'score': 0, 'triggered_rules': [], '_notes': [],
            'divergence': div_signals,
            'kline_data': df.attrs.get('_kline_pattern_details', None),
            'volume_trend': (vol_desc, vol_score),
            'momentum': {'desc': mom_desc, 'score': mom_score},
            'slope': slope_info,
            'chip_data': df.attrs.get('chip_data', None),
            'config': rule_thresholds,
        }

        regime = _detect_market_regime(df, boll_col=boll_bw_col if boll_bw_col else None)
        state['regime'] = regime

        if regime == 'WEAK_TREND':
            state['level'] = 'C'
            state['conclusion'] = 'C: 弱势趋势，环境不配合'
            state['score'] = 0
            return _pipeline_output(state)

        signal_list = []
        detail_12269 = df['MACD_12269_SIGNAL_DETAIL'].iloc[-1] if 'MACD_12269_SIGNAL_DETAIL' in df.columns else ''
        if '金叉' in str(detail_12269):
            if '零轴上' in str(detail_12269):
                signal_list.append({'type': 'MACD_金叉', 'confidence': 'high', 'desc': '零轴上金叉'})
            else:
                signal_list.append({'type': 'MACD_金叉', 'confidence': 'medium', 'desc': '零轴下金叉'})

        cs = div_signals.get('combined_signal', '')
        if '底背离' in cs:
            strength = div_signals.get('strength_12269', 0) or div_signals.get(f'strength_{second_period_name}', 0)
            confidence = 'high' if strength > 0.6 else 'medium'
            signal_list.append({'type': '底背离', 'confidence': confidence, 'desc': cs})

        kd = state.get('kline_data')
        if kd and kd.get('details'):
            for d in kd['details']:
                if d['direction'] == '看涨' and d['level'] in ('强反转', '中反转'):
                    confidence_p = 'high' if d['level'] == '强反转' else 'medium'
                    signal_list.append({'type': 'K线反转', 'confidence': confidence_p, 'desc': d['label']})
                    break

        if not signal_list:
            r = state.get('regime', '')
            last_dif = df['MACD_12269_DIF'].iloc[-1] if 'MACD_12269_DIF' in df.columns else 0
            if r == 'STRONG_TREND' and last_dif > 0:
                signal_list.append({'type': '强势延续', 'confidence': 'high', 'desc': 'MACD多头趋势持续'})

        state['signal_list'] = signal_list
        execute_rules(state, gate=1)

        if not signal_list:
            state['level'] = 'C'
            state['conclusion'] = 'C: 无明确入场信号'
            return _pipeline_output(state)

        execute_rules(state, gate=2)
        if state['risk_level'] == 'HIGH':
            return _pipeline_output(state)

        has_top_div = '顶背离' in cs or '卖出' in cs
        if has_top_div and 'R01' not in state['triggered_rules']:
            state['risk_level'] = 'MEDIUM'
            state['risk_desc'] = '顶背离(未达R01阈值)'

        chip = state.get('chip_data')
        if chip:
            winner_rate = chip.get('winner_rate', 0)
            close_val = df['close'].iloc[-1]
            cost_95pct = chip.get('cost_95pct', 0)
            cost_5pct = chip.get('cost_5pct', 0)
            if winner_rate > 80 and state['regime'] in ('WEAK_TREND', 'TOP_RISK', 'UNCLEAR'):
                if state['risk_level'] in ('NONE', 'LOW'):
                    state['risk_level'] = 'MEDIUM'
                    state['risk_desc'] += ' 筹码高位获利[>80%]'
            if cost_95pct > 0 and close_val >= cost_95pct * 0.95 and winner_rate > 70:
                if state['risk_level'] in ('NONE', 'LOW'):
                    state['risk_level'] = 'MEDIUM'
                    state['risk_desc'] += ' 筹码阻力位'
            if cost_95pct > 0 and cost_5pct > 0:
                chip_range = (cost_95pct - cost_5pct) / max(cost_5pct, 0.01)
                if chip_range < 0.15:
                    cost_50pct = chip.get('cost_50pct', 0)
                    if close_val > cost_50pct and winner_rate > 70:
                        if state['risk_level'] not in ('HIGH',):
                            state['risk_level'] = 'HIGH' if state['regime'] in ('WEAK_TREND', 'TOP_RISK') else 'MEDIUM'
                            state['risk_desc'] += ' 筹码密集区高位'

        execute_rules(state, gate=3)

        regime_mult = {
            'STRONG_TREND':   {'zero': 1.2, 'strat': 1.0, 'tact': 1.0, 'mom': 1.5, 'slope': 1.5, 'div': 1.0, 'vol': 1.0, 'kp': 0.3},
            'BOTTOM_REVERSAL':{'zero': 0.5, 'strat': 1.0, 'tact': 1.0, 'mom': 0.8, 'slope': 0.8, 'div': 2.0, 'vol': 1.5, 'kp': 1.5},
            'TOP_RISK':       {'zero': 0.3, 'strat': 0.5, 'tact': 0.5, 'mom': 0.5, 'slope': 0.5, 'div': 2.0, 'vol': 1.5, 'kp': 2.0},
            'OSCILLATION':    {'zero': 0.5, 'strat': 0.5, 'tact': 0.5, 'mom': 0.3, 'slope': 0.3, 'div': 1.5, 'vol': 0.5, 'kp': 1.2},
            'UNCLEAR':        {'zero': 1.0, 'strat': 1.0, 'tact': 1.0, 'mom': 1.0, 'slope': 1.0, 'div': 1.0, 'vol': 1.0, 'kp': 1.0},
        }
        mult = regime_mult.get(regime, regime_mult['UNCLEAR'])
        w_zero = int(weights['零轴条件'] * mult['zero'])
        w_strat = int(weights['战略金叉'] * mult['strat'])
        w_tact = int(weights['战术金叉'] * mult['tact'])
        w_mom = int(weights['动能'] * mult['mom'])
        w_slope = int(weights['DIF斜率'] * mult['slope'])
        w_div = int(weights['背离信号'] * mult['div'])
        w_vol = int(weights['量价配合'] * mult['vol'])
        w_kp = int(weights['K线形态'] * mult['kp'])

        scores = {}
        dif_above = df['DIF_12269'].iloc[-1] > 0
        dea_above = df['DEA_12269'].iloc[-1] > 0
        if dif_above and dea_above:
            scores['零轴条件'] = ('DIF/DEA 均在零轴上', w_zero)
        elif dif_above:
            scores['零轴条件'] = ('DIF 在零轴上，DEA 仍在下方', w_zero // 2)
        else:
            scores['零轴条件'] = ('DIF 在零轴下', 0)

        slow_detail = df['MACD_12269_SIGNAL_DETAIL'].iloc[-1] if 'MACD_12269_SIGNAL_DETAIL' in df.columns else ''
        slow_bull = df['DIF_12269'].iloc[-1] > df['DEA_12269'].iloc[-1]
        if '零轴上金叉' in str(slow_detail):
            scores['战略金叉'] = ('12269 零轴上金叉', w_strat)
        elif '零轴下金叉' in str(slow_detail):
            scores['战略金叉'] = ('12269 零轴下金叉', w_strat // 2)
        elif slow_bull:
            scores['战略金叉'] = ('12269 多头持续', int(w_strat * 0.75))
        else:
            scores['战略金叉'] = ('12269 空头排列', 0)

        fast_detail_col = f'MACD_{second_period_name}_SIGNAL_DETAIL'
        fast_dif_col = f'DIF_{second_period_name}'
        fast_dea_col = f'DEA_{second_period_name}'
        if fast_detail_col in df.columns:
            fast_detail_s = df[fast_detail_col].iloc[-1]
            fast_bull_s = df[fast_dif_col].iloc[-1] > df[fast_dea_col].iloc[-1]
            if '零轴上金叉' in str(fast_detail_s):
                scores['战术金叉'] = (f'{second_period_name} 零轴上金叉', w_tact)
            elif fast_bull_s:
                scores['战术金叉'] = (f'{second_period_name} 多头持续', int(w_tact * 0.65))
            else:
                scores['战术金叉'] = (f'{second_period_name} 空头/死叉', 0)

        prefix = '强势' if mom_score >= w_mom // 2 else ('关注' if mom_score >= w_mom // 4 else '弱势')
        scores['动能'] = (f'{prefix}: {mom_desc}', min(mom_score, w_mom))
        if slope_info['trend'] == '明确上行':
            scores['DIF斜率'] = (f"确认 {slope_info['trend']} (R²={slope_info['r2']})", w_slope)
        elif '上行' in slope_info['trend']:
            scores['DIF斜率'] = (f"关注 {slope_info['trend']} (R²={slope_info['r2']})", int(w_slope * 0.55))
        else:
            scores['DIF斜率'] = (f"弱势 {slope_info['trend']} (R²={slope_info['r2']})", 0)

        div_str = div_signals.get('strength_12269') or div_signals.get(f'strength_{second_period_name}', 0.0)
        div_decay = div_signals.get('decay_12269') or div_signals.get(f'decay_{second_period_name}', 0.0)
        eff = round(div_str * div_decay, 3)
        if '底背离' in cs:
            scores['背离信号'] = (f'确认 {cs} (强度={div_str}, 衰减={div_decay}, 有效={eff})', int(w_div * (0.5 + 0.5 * eff)))
        elif '顶背离' in cs or '卖出' in cs:
            scores['背离信号'] = (f'否定 {cs}（一票否决）', 0)
        else:
            scores['背离信号'] = ('中性: 无背离信号', 0)

        vol_bonus = 0 if has_top_div else vol_score
        scores['量价配合'] = (vol_desc, min(vol_bonus, w_vol))
        scores['K线形态'] = kp_result

        base_keys = [k for k in scores if k != '量价配合']
        total_base = sum(scores[k][1] for k in base_keys)
        total_max_base = w_zero + w_strat + w_tact + w_mom + w_slope + w_div + w_kp
        total_base = max(0, min(total_max_base, total_base))
        total = max(0, min(total_max_base + w_vol, total_base + scores['量价配合'][1]))

        notes = state.get('_notes', [])
        notes_str = '+'.join(notes) if notes else ''
        is_high_risk = state['risk_level'] == 'HIGH' or has_top_div

        if is_high_risk:
            state['level'] = 'D'
            conclusion_parts = ['D: 顶部风险']
        elif total_base >= thresholds['fully_bull']:
            state['level'] = 'A'
            conclusion_parts = ['A: 综合多头']
        elif total_base >= thresholds['bullish'] and has_top_div:
            state['level'] = 'B'
            conclusion_parts = ['B: 偏多(注意顶部风险)']
        elif total_base >= thresholds['bullish']:
            state['level'] = 'B'
            conclusion_parts = ['B: 偏多']
        elif total_base >= thresholds['oscillate']:
            state['level'] = 'C'
            conclusion_parts = ['C: 多空拉锯']
        else:
            state['level'] = 'C'
            conclusion_parts = ['C: 偏空']

        regime_labels = {
            'STRONG_TREND': '强势趋势', 'WEAK_TREND': '弱势趋势',
            'BOTTOM_REVERSAL': '底部反转', 'TOP_RISK': '顶部风险',
            'OSCILLATION': '震荡', 'UNCLEAR': '方向不明',
        }
        regime_label = regime_labels.get(regime, '方向不明')
        signal_desc = '+'.join(s['desc'] for s in signal_list[:2]) if signal_list else ''
        if notes_str:
            conclusion_parts.append(notes_str)
        if regime_label != '方向不明':
            conclusion_parts.insert(1, regime_label)
        if signal_desc:
            conclusion_parts.insert(2, signal_desc)

        state['conclusion'] = ' | '.join(filter(None, conclusion_parts))
        state['score'] = total
        state['risk_desc'] = state.get('risk_desc', '')
        df.attrs['pipeline_level'] = state['level']
        df.attrs['pipeline_conclusion'] = state['conclusion']
        df.attrs['pipeline_regime'] = state['regime']

        close_v = df['close'].iloc[-1]
        lookback = 20
        support = df['close'].iloc[-lookback:].min()
        resistance = df['close'].iloc[-lookback:].max()
        risk_pct = max((close_v - support) / close_v, 0.005)
        reward_pct = max((resistance - close_v) / close_v, 0)
        state['expected_return'] = round(reward_pct * 100, 1)
        state['risk_reward_ratio'] = round(reward_pct / risk_pct, 2) if risk_pct > 0 else 0
        state['_scores'] = scores
        return _pipeline_output(state)

    def analyze_full_bull(
        self,
        df: pd.DataFrame,
        decay_half_life: int = 8,
        slope_window: int = 5,
        second_params: tuple[int, int, int] = (6, 13, 5),
        recalc_macd: bool = True,
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
    ) -> dict:
        if weights is None:
            weights = {"零轴条件": 20, "战略金叉": 20, "战术金叉": 15, "动能": 20, "DIF斜率": 15, "背离信号": 10, "量价配合": 10}
        if thresholds is None:
            thresholds = {"fully_bull": 80, "bullish": 60, "oscillate": 40}
        if recalc_macd:
            df = self._custom_macd(df, second_params=second_params)
        else:
            required = {"DIF_12269", "DEA_12269", "MACD_12269", "MACD_12269_SIGNAL_DETAIL"}
            if not required.issubset(df.columns):
                df = self._custom_macd(df, second_params=second_params)

        fast, slow, signal = second_params
        second_period_name = f"{fast}{slow}{signal}"

        dist_slow = adaptive_distance(df, base=10)
        dist_fast = adaptive_distance(df, base=5)
        div_signals = detect_combined_divergence(
            df, distance_slow=dist_slow, distance_fast=dist_fast,
            decay_half_life=decay_half_life, second_period_name=second_period_name,
        )

        scores: dict[str, tuple[str, int]] = {}
        w_zero = weights["零轴条件"]
        w_strat = weights["战略金叉"]
        w_tact = weights["战术金叉"]
        w_mom = weights["动能"]
        w_slope = weights["DIF斜率"]
        w_div = weights["背离信号"]
        w_vol = weights["量价配合"]
        w_kp = weights.get("K线形态", 10)

        dif_above = df["DIF_12269"].iloc[-1] > 0
        dea_above = df["DEA_12269"].iloc[-1] > 0
        if dif_above and dea_above:
            scores["零轴条件"] = ("DIF/DEA 均在零轴上", w_zero)
        elif dif_above:
            scores["零轴条件"] = ("DIF 在零轴上，DEA 仍在下方（金叉进行中）", w_zero // 2)
        else:
            scores["零轴条件"] = ("DIF 在零轴下", 0)

        slow_detail = df["MACD_12269_SIGNAL_DETAIL"].iloc[-1]
        slow_bull = df["DIF_12269"].iloc[-1] > df["DEA_12269"].iloc[-1]
        if slow_detail == "零轴上金叉":
            scores["战略金叉"] = ("12269 零轴上金叉（最强信号）", w_strat)
        elif slow_detail == "零轴下金叉":
            scores["战略金叉"] = ("12269 零轴下金叉（注意假突破）", w_strat // 2)
        elif slow_bull:
            scores["战略金叉"] = ("12269 多头持续（DIF > DEA）", int(w_strat * 0.75))
        else:
            scores["战略金叉"] = ("12269 空头排列", 0)

        fast_detail_col = f"MACD_{second_period_name}_SIGNAL_DETAIL"
        fast_dif_col = f"DIF_{second_period_name}"
        fast_dea_col = f"DEA_{second_period_name}"
        fast_detail = df[fast_detail_col].iloc[-1]
        fast_bull = df[fast_dif_col].iloc[-1] > df[fast_dea_col].iloc[-1]
        if fast_detail == "零轴上金叉":
            scores["战术金叉"] = (f"{second_period_name} 零轴上金叉", w_tact)
        elif fast_bull:
            scores["战术金叉"] = (f"{second_period_name} 多头持续", int(w_tact * 0.65))
        else:
            scores["战术金叉"] = (f"{second_period_name} 空头/死叉", 0)

        mom_desc, mom_score = _calc_momentum_desc(df, "DIF_12269", "DEA_12269", max_score=w_mom)
        prefix = "强势" if mom_score >= w_mom // 2 else ("关注" if mom_score >= w_mom // 4 else "弱势")
        scores["动能"] = (f"{prefix}: {mom_desc}", mom_score)

        slope_info = slope_analysis(df["DIF_12269"], window=slope_window)
        if slope_info["trend"] == "明确上行":
            scores["DIF斜率"] = (f"确认 {slope_info['trend']} (R²={slope_info['r2']})", w_slope)
        elif "上行" in slope_info["trend"]:
            scores["DIF斜率"] = (f"关注 {slope_info['trend']} (R²={slope_info['r2']})", int(w_slope * 0.55))
        else:
            scores["DIF斜率"] = (f"弱势 {slope_info['trend']} (R²={slope_info['r2']})", 0)

        cs = div_signals["combined_signal"]
        div_str = (
            div_signals.get("strength_12269") if div_signals.get("div_12269")
            else div_signals.get(f"strength_{second_period_name}", 0.0)
        )
        div_decay = (
            div_signals.get("decay_12269") if div_signals.get("div_12269")
            else div_signals.get(f"decay_{second_period_name}", 0.0)
        )
        eff = round(div_str * div_decay, 3)
        has_top_div = "顶背离" in cs or "卖出" in cs
        if "底背离" in cs:
            scores["背离信号"] = (f"确认 {cs} (强度={div_str}, 衰减={div_decay}, 有效={eff})", int(w_div * (0.5 + 0.5 * eff)))
        elif has_top_div:
            scores["背离信号"] = (f"否定 {cs}（一票否决）", 0)
        else:
            scores["背离信号"] = ("中性: 无背离信号", 0)

        vol_desc, vol_bonus = _volume_price_trend_score(df, max_bonus=w_vol)
        if has_top_div:
            vol_bonus = 0
            vol_desc = "顶背离压制，跳过量价奖励"
        scores["量价配合"] = (vol_desc, vol_bonus)

        kp_result = _score_kline_pattern(df, max_score=w_kp)
        if has_top_div:
            kp_result = (kp_result[0], max(0, kp_result[1]))
        scores["K线形态"] = kp_result

        winrate = _backtest_signal_winrate(df, "MACD_12269_SIGNAL_DETAIL", "零轴上金叉", forward_bars=5)
        winrate_str = (
            f"样本 {winrate['sample_count']} 次，胜率 {winrate['win_rate']}，均收益 {winrate['avg_return']}"
            if winrate["sample_count"] > 0 else "样本不足"
        )

        base_keys = [k for k in scores if k != "量价配合"]
        total_base = sum(scores[k][1] for k in base_keys)
        bonus = scores["量价配合"][1]
        if has_top_div:
            total_base = min(total_base, w_div * 4)
        total_max_base = w_zero + w_strat + w_tact + w_mom + w_slope + w_div + w_kp
        total_base = max(0, min(total_max_base, total_base))
        total = max(0, min(total_max_base + w_vol, total_base + bonus))

        if total_base >= thresholds["fully_bull"]:
            conclusion = "完全多头 (强烈买入)"
        elif total_base >= thresholds["bullish"]:
            conclusion = "偏多 (可逢低布局)"
        elif total_base >= thresholds["oscillate"]:
            conclusion = "多空拉锯 (观望为主)"
        else:
            conclusion = "偏空 (回避或做空)"

        return {
            "score": total, "score_base": total_base, "conclusion": conclusion,
            "details": {k: {"desc": v[0], "score": v[1]} for k, v in scores.items()},
            "divergence": div_signals, "momentum": mom_desc, "slope": slope_info,
            "winrate_ref": winrate_str,
        }


# ── 模块级辅助函数（由 pipeline_analysis / analyze_full_bull 调用） ──────


def _calc_momentum_desc(df: pd.DataFrame, dif_col: str, dea_col: str, max_score: int = 20) -> tuple[str, int]:
    if len(df) < 7:
        return "数据不足", 0
    hist = df[dif_col] - df[dea_col]
    cur_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    hist_change = cur_hist - prev_hist
    recent_hist = hist.iloc[-5:]
    hist_vol = max(recent_hist.std(), 1e-9)
    norm_change = hist_change / hist_vol
    is_bull = cur_hist > 0
    if is_bull:
        ratio = norm_change / (norm_change + 1) if norm_change >= 0 else norm_change / (norm_change - 1)
        score = min(max_score, max(0, int(max_score * (0.5 + 0.5 * ratio))))
        desc = f"加速上涨 (红柱加长, z={norm_change:.2f})" if norm_change > 0 else f"减速上涨 (红柱缩短, z={norm_change:.2f})"
    else:
        max_bear = max(8, max_score * 2 // 5)
        ratio = abs(norm_change) / (abs(norm_change) + 1)
        score = min(max_bear, max(0, int(max_bear * ratio)))
        desc = f"加速下跌 (绿柱加长, z={norm_change:.2f})" if norm_change < 0 else f"减速下跌 (绿柱缩短, z={norm_change:.2f})"
    return desc, score


def _volume_price_trend_score(df: pd.DataFrame, lookback: int = 5, max_bonus: int = 10) -> tuple[str, int]:
    if len(df) < lookback + 1 or "volume" not in df.columns:
        return "数据不足", 0
    closes = df["close"].iloc[-lookback:].values
    volumes = df["volume"].iloc[-lookback:].values
    pct_change = (closes[-1] - closes[0]) / (closes[0] + 1e-9)
    vol_early = volumes[:2].mean()
    vol_trend = (volumes[-1] - vol_early) / (vol_early + 1e-9)
    half = max_bonus // 2
    if pct_change > 0.02 and vol_trend > 0.1:
        return f"量价齐升 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", max_bonus
    if pct_change > 0.02:
        return f"价涨量缩 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", half
    if pct_change < -0.02 and vol_trend > 0.1:
        return f"放量下跌 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", -half
    if pct_change < -0.02:
        return f"缩量下跌 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", 0
    return f"量价平淡 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", 0


def _score_kline_pattern(df: pd.DataFrame, max_score: int = 10) -> tuple[str, int]:
    attrs_score = df.attrs.get('kline_pattern_score', None)
    if attrs_score is not None:
        raw = attrs_score
    else:
        recent = df.tail(20)
        raw = 0.0
        for i in range(len(recent) - 1, max(len(recent) - 6, -1), -1):
            body = abs(recent['close'].iloc[i] - recent['open'].iloc[i])
            lower = min(recent['open'].iloc[i], recent['close'].iloc[i]) - recent['low'].iloc[i]
            upper = recent['high'].iloc[i] - max(recent['open'].iloc[i], recent['close'].iloc[i])
            if body > 0 and lower > body * 2 and upper < body * 0.5:
                raw += 1 if recent['close'].iloc[i] > recent['open'].iloc[i] else -1
            if body > 0 and upper > body * 2 and lower < body * 0.5:
                raw += 1 if recent['close'].iloc[i] > recent['open'].iloc[i] else -1
        for i in range(len(recent) - 2):
            if all(recent['close'].iloc[i + j] > recent['open'].iloc[i + j] for j in range(3)):
                raw += 1
            if all(recent['close'].iloc[i + j] < recent['open'].iloc[i + j] for j in range(3)):
                raw -= 1
    raw_norm = max(-1.0, min(1.0, raw / 10.0))
    score = int((raw_norm + 1.0) / 2.0 * max_score)
    if raw_norm > 0.3:
        desc = f"偏多形态 (score={raw:.1f})"
    elif raw_norm < -0.3:
        desc = f"偏空形态 (score={raw:.1f})"
    else:
        desc = f"中性形态 (score={raw:.1f})"
    return desc, score


def _backtest_signal_winrate(df: pd.DataFrame, signal_col: str, target_signal: str, forward_bars: int = 5) -> dict:
    if signal_col not in df.columns:
        return {"sample_count": 0, "win_rate": None, "avg_return": None, "max_gain": None, "max_loss": None}
    hit_locs = [df.index.get_loc(i) for i in df.index[df[signal_col] == target_signal]]
    results = []
    for loc in hit_locs:
        end_loc = min(loc + forward_bars, len(df) - 1)
        if end_loc <= loc:
            continue
        entry = df["close"].iloc[loc]
        exit_ = df["close"].iloc[end_loc]
        results.append((exit_ - entry) / (entry + 1e-9))
    if not results:
        return {"sample_count": 0, "win_rate": None, "avg_return": None, "max_gain": None, "max_loss": None}
    arr = np.array(results)
    return {
        "sample_count": len(arr),
        "win_rate": round(float((arr > 0).mean()), 3),
        "avg_return": round(float(arr.mean()), 4),
        "max_gain": round(float(arr.max()), 4),
        "max_loss": round(float(arr.min()), 4),
    }


def _detect_market_regime(df: pd.DataFrame, boll_col: str | None = None) -> str:
    close = df['close'].astype(float)
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma30 = df['close'].rolling(30).mean().iloc[-1]
    ma60 = df['close'].rolling(60).mean().iloc[-1]
    ma_bullish = ma5 > ma10 > ma20 > ma30 > ma60
    ma_bearish = ma5 < ma10 < ma20 < ma30 < ma60
    dif = df['DIF_12269'].iloc[-1] if 'DIF_12269' in df.columns else 0
    hist = df['DIF_12269'] - df['DEA_12269'] if 'DIF_12269' in df.columns and 'DEA_12269' in df.columns else None
    momentum_positive = hist.iloc[-1] > 0 if hist is not None else False
    slope_info = slope_analysis(df['DIF_12269'] if 'DIF_12269' in df.columns else df['close'])
    slope_positive = slope_info['slope'] > 0
    is_narrow_boll = False
    if boll_col and boll_col in df.columns:
        recent_bw = df[boll_col].iloc[-5:].mean()
        hist_bw = df[boll_col].mean()
        is_narrow_boll = recent_bw < hist_bw * 0.8
    if ma_bullish and slope_positive and momentum_positive:
        return 'STRONG_TREND'
    if ma_bearish and dif < 0 and not momentum_positive:
        return 'WEAK_TREND'
    if is_narrow_boll and hist is not None and len(df) > 30 and abs(hist.iloc[-1]) < 0.1 * close.std():
        return 'OSCILLATION'
    if not ma_bullish and 'DIF_12269' in df.columns and len(df) >= 10:
        dif_series = df['DIF_12269']
        if dif < 0 and dif_series.iloc[-1] > dif_series.iloc[-5] and hist is not None:
            if hist.iloc[-1] > hist.iloc[-5]:
                return 'BOTTOM_REVERSAL'
    if ma_bullish and 'DIF_12269' in df.columns and len(df) >= 10:
        dif_series = df['DIF_12269']
        close_vs_ma20 = (close - ma20) / ma20 if ma20 > 0 else 0
        if close_vs_ma20 > 0.15 and dif_series.iloc[-1] < dif_series.iloc[-5] and hist is not None:
            if hist.iloc[-1] < hist.iloc[-5]:
                return 'TOP_RISK'
    regime_from_attrs = df.attrs.get('_regime_hint', None)
    return regime_from_attrs if regime_from_attrs else 'UNCLEAR'


# ── 级联流水线输出构建 ────────────────────────────────────────────────────


def _pipeline_output(state: dict) -> dict:
    """
    从 pipeline state 构建统一返回 dict。

    保持与 analyze_full_bull 相同的顶层 key，以便 SignalManager 兼容处理。
    """
    notes = state.get('_notes', [])
    details = {}
    # 如果 state 中有 scores 则包含（取自 Gate 3 计算）
    if '_scores' in state:
        details = {k: {'desc': v[0], 'score': v[1]} for k, v in state['_scores'].items()}

    return {
        "score": state.get('score', 0),
        "score_base": state.get('score', 0),
        "conclusion": state.get('conclusion', ''),
        "level": state.get('level', 'C'),
        "regime": state.get('regime', 'UNCLEAR'),
        "risk_level": state.get('risk_level', 'NONE'),
        "risk_desc": state.get('risk_desc', ''),
        "triggered_rules": state.get('triggered_rules', []),
        "notes": notes,
        "signal_count": len(state.get('signal_list', [])),
        "details": details if details else {},
        "divergence": state.get('divergence', {}),
        "momentum": state.get('momentum', {}).get('desc', ''),
        "slope": state.get('slope', {}),
        "winrate_ref": '参考 pipeline',
        "expected_return": state.get('expected_return', 0),
        "risk_reward_ratio": state.get('risk_reward_ratio', 0),
    }
