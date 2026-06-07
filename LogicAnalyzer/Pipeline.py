import numpy as np
import pandas as pd
import pandas_ta as ta

from LogicAnalyzer.MACDDivergence import (
    adaptive_distance, detect_divergence_single_param, slope_analysis, signal_with_decay,
)
from LogicAnalyzer.SignalConstants import (
    MACDSignals, Divergence, TrendLevels,
    FullBullScoring, Conclusion, KLineLevels, MACDTrend
)
from LogicAnalyzer.PipelineScoring import (
    _calc_momentum_desc, _volume_price_trend_score, _score_kline_pattern,
    _backtest_signal_winrate, _calc_moneyflow_score,
)
from LogicAnalyzer.PipelineState import (
    _make_state, _get_regime_multiplier, _get_macd_trend_mult,
    _apply_chip_risk, _detect_market_regime, _calc_exit_strategy, _pipeline_output,
)


class MACDAnalyzer:
    """
    单参数 MACD 分析器。

    子模块委托：
      - MACDDivergence: 背离检测
      - ScoringRules: 门控规则
    """

    @staticmethod
    def _classify_macd_trend(dif: float, dea: float) -> str:
        if dif > dea > 0:
            return MACDTrend.SUPER_STRONG
        if dif > dea:
            return MACDTrend.STRONG
        if dif < dea < 0:
            return MACDTrend.SUPER_WEAK
        return MACDTrend.WEAK

    def _custom_macd(self, df: pd.DataFrame, params: tuple[int, int, int] = (12, 26, 9)) -> pd.DataFrame:
        if "close" not in df.columns:
            return df
        fast, slow, signal = params
        close = df["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()

        df["DIF"] = dif
        df["DEA"] = dea
        df["MACD_HIST"] = 2 * (dif - dea)

        prev_dif = dif.shift(1).fillna(dea.shift(1).fillna(0))
        prev_dea = dea.shift(1).fillna(0)
        golden = (dif > dea) & (prev_dif <= prev_dea)
        dead = (dif < dea) & (prev_dif >= prev_dea)
        df["MACD_SIGNAL_DETAIL"] = np.where(
            golden,
            MACDSignals.golden_cross_label(dif, dea),
            MACDSignals.death_cross_label(dead, dif, dea),
        )
        df["MACD_CROSS"] = np.where(golden, 1, np.where(dead, -1, 0))
        return df

    @staticmethod
    def _check_multitimeframe(df: pd.DataFrame) -> dict:
        if len(df) < 30 or 'DIF' not in df.columns or 'DEA' not in df.columns:
            return {'alignment': 'UNKNOWN', 'multiplier': 1.0, 'desc': ''}

        weekly = df.rolling(5).agg({
            'close': 'last', 'DIF': 'last', 'DEA': 'last',
        }).dropna()

        if weekly.empty:
            return {'alignment': 'UNKNOWN', 'multiplier': 1.0, 'desc': ''}

        d_bull = df['DIF'].iloc[-1] > df['DEA'].iloc[-1]
        w_bull = weekly['DIF'].iloc[-1] > weekly['DEA'].iloc[-1]

        if d_bull and w_bull:
            return {'alignment': 'ALIGNED_BULL', 'multiplier': 1.1, 'desc': '日周共振多头'}
        if not d_bull and not w_bull:
            return {'alignment': 'ALIGNED_BEAR', 'multiplier': 1.0, 'desc': '日周共振空头'}
        if d_bull and not w_bull:
            return {'alignment': 'WEEKLY_BEAR_DAILY_BULL', 'multiplier': 0.5, 'desc': '周线空头日线金叉(反弹诱多)'}
        return {'alignment': 'WEEKLY_BULL_DAILY_BEAR', 'multiplier': 0.6, 'desc': '周线多头日线死叉(回调买入)'}

    @staticmethod
    def _detect_volatility_regime(df: pd.DataFrame) -> str:
        atr = df['ATR'].dropna()
        if len(atr) < 30:
            return 'NORMAL'
        hist_mean = atr.iloc[-30:].mean()
        current_atr = atr.iloc[-1]
        vol_ratio = current_atr / hist_mean if hist_mean > 0 else 1.0

        adx_col = next((c for c in df.columns if c.startswith('ADX_')), None)
        adx_val = df[adx_col].iloc[-1] if adx_col else 0

        if vol_ratio > 1.3 and adx_val > 25:
            return 'HIGH_VOL_TREND'
        if vol_ratio < 0.7 and adx_val < 20:
            return 'LOW_VOL_REVERSAL'
        return 'NORMAL'

    @staticmethod
    def _precompute_rule_indicators(df: pd.DataFrame) -> None:
        if 'ATR' not in df.columns:
            try:
                df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            except Exception:
                df['ATR'] = float('nan')

        if 'ADX' not in df.columns:
            try:
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                for c in adx_df.columns:
                    df[c] = adx_df[c]
            except Exception:
                df['ADX_14'] = float('nan')

        if 'MA_200' not in df.columns:
            df['MA_200'] = df['close'].rolling(200).mean()

        if not any(c.startswith('RSI_') for c in df.columns):
            try:
                df.ta.rsi(append=True, close='close', length=14)
            except Exception:
                pass

        if not any(c.startswith('STOCHk') for c in df.columns):
            try:
                stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=9, d=3)
                for c in stoch_df.columns:
                    df[c] = stoch_df[c]
            except Exception:
                pass

        if 'AMPLITUDE_PCT' not in df.columns and 'close' in df.columns and len(df) >= 2:
            df['AMPLITUDE_PCT'] = (df['high'] - df['low']) / df['close'].shift(1) * 100

        if not any(c.startswith('CCI_') for c in df.columns):
            try:
                df.ta.cci(append=True, high='high', low='low', close='close')
            except Exception:
                pass

        if not any(c.startswith('BBU_') for c in df.columns):
            try:
                df.ta.bbands(append=True, close='close', length=20, std=2)
            except Exception:
                pass

    def pipeline_analysis(
        self,
        df: pd.DataFrame,
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
        rule_thresholds: dict | None = None,
        decay_half_life: int = 8,
        slope_window: int = 5,
        params: dict | None = None,
    ) -> dict:
        """全流水线分析。

        Args:
            df: K 线 DataFrame
            weights: 7 维评分权重
            thresholds: 结论阈值 (fully_bull/bullish/oscillate)
            rule_thresholds: ScoringRules 门控参数
            decay_half_life: 背离信号半衰期（天），会被 params.divergence 覆盖
            slope_window: DIF 斜率回归窗口，会被 params.divergence 覆盖
            params: 集中参数配置（来自 ConfigParser），含 regime/divergence/scoring/technical 子字典。
                    不传时使用方法内部默认值（同旧 magic number）。
        """
        # 从集中参数提取子配置
        _p = params or {}
        _regime_p = _p.get('regime', {})
        _div_p = _p.get('divergence', {})
        _score_p = _p.get('scoring', {})
        _tech_p = _p.get('technical', {})

        # 参数合并：显式参数优先 > 集中参数 > 方法默认值
        decay_half_life = _div_p.get('decay_half_life', decay_half_life)
        slope_window = _div_p.get('slope_window', slope_window)
        div_strength_threshold = _div_p.get('strength_threshold', 0.15)
        vol_norm_denom = _score_p.get('vol_norm_denominator', 0.15)
        cross_decay_days = _score_p.get('cross_decay_days', 30)
        cross_decay_min = _score_p.get('cross_decay_min', 0.3)
        kline_decay_days = _score_p.get('kline_decay_days', 10)
        kline_decay_min = _score_p.get('kline_decay_min', 0.2)
        expected_return_lookback = _score_p.get('expected_return_lookback', 20)

        from LogicAnalyzer.ScoringRules import execute_rules

        if weights is None:
            weights = {"MACD趋势": 20, "金叉信号": 15, "柱状动能": 15,
                       "DIF斜率": 10, "背离信号": 10, "量价配合": 10, "K线形态": 10}
        if thresholds is None:
            thresholds = {"fully_bull": 80, "bullish": 60, "oscillate": 40}
        if rule_thresholds is None:
            rule_thresholds = {"divergence": 0.3, "winner_rate_high": 80,
                               "cost_resistance_ratio": 0.95, "chip_concentrated_ratio": 0.15,
                               "price_new_high_days": 20,
                               "adx_fake_breakout": 20, "ma200_distance": 0.30,
                               "atr_volatility": 0.05, "volume_threshold": 1e7,
                               "amplitude_percentile": 0.95, "golden_cross_stagnant_days": 10,
                               "golden_cross_stagnant_pct": 0.02}

        if 'MA_5' not in df.columns:
            for p in [5, 10, 20, 30, 60]:
                df[f'MA_{p}'] = df['close'].rolling(p).mean()

        self._precompute_rule_indicators(df)

        boll_bw_col = None
        boll_upper_cols = [c for c in df.columns if c.startswith('BBU_')]
        boll_lower_cols = [c for c in df.columns if c.startswith('BBL_')]
        if boll_upper_cols and boll_lower_cols:
            bw_col = 'BOLL_BANDWIDTH'
            if bw_col not in df.columns:
                df[bw_col] = (df[boll_upper_cols[0]] - df[boll_lower_cols[0]]) / df['close']
            boll_bw_col = bw_col

        dif_val = df['DIF'].iloc[-1]
        dea_val = df['DEA'].iloc[-1]
        macd_trend = self._classify_macd_trend(dif_val, dea_val)

        if macd_trend == MACDTrend.SUPER_WEAK:
            return _pipeline_output({
                'level': 'D', 'score': 0, 'conclusion': 'D: 中长期空头，回避',
                'regime': 'WEAK_TREND', 'risk_level': 'HIGH', 'risk_desc': 'DIF<DEA<0',
                'signal_list': [], 'triggered_rules': [], '_notes': [],
                'divergence': {}, 'momentum': {}, 'slope': {}, 'chip_data': None,
            })

        regime = _detect_market_regime(df, boll_col=boll_bw_col if boll_bw_col else None,
                                        params=_regime_p)
        if regime in ('WEAK_TREND',) and macd_trend != MACDTrend.SUPER_STRONG:
            state = _make_state(df, regime, rule_thresholds)
            state.update({'level': 'C', 'score': 0, 'conclusion': 'C: 弱势趋势，环境不配合'})
            return _pipeline_output(state)

        if macd_trend == MACDTrend.SUPER_STRONG and regime in ('OSCILLATION', 'UNCLEAR'):
            regime = 'STRONG_TREND'

        # Gate 0: 数据质量预筛（在耗费计算资源的背离检测之前）
        if len(df) < 60:
            return _pipeline_output({
                'level': 'D', 'score': 0, 'conclusion': 'D: 数据不足(K线<60)',
                'regime': regime, 'risk_level': 'HIGH', 'risk_desc': '',
                'signal_list': [], 'triggered_rules': ['R30'], '_notes': [],
                'divergence': {}, 'momentum': {}, 'slope': {}, 'chip_data': None,
                'exit_strategy': {}, 'macd_trend': macd_trend, 'position_adjust': 0.0,
            })
        if 'ATR' in df.columns and (pd.isna(df['ATR'].iloc[-1]) or df['ATR'].iloc[-1] <= 0):
            # ATR 缺失继续运行但 score 会偏低
            pass  # R31 不会触发 fatal，继续
        if 'MA_60' not in df.columns or pd.isna(df['MA_60'].iloc[-1]):
            return _pipeline_output({
                'level': 'D', 'score': 0, 'conclusion': 'D: MA60缺失无长期趋势',
                'regime': regime, 'risk_level': 'HIGH', 'risk_desc': '',
                'signal_list': [], 'triggered_rules': ['R32'], '_notes': [],
                'divergence': {}, 'momentum': {}, 'slope': {}, 'chip_data': None,
                'exit_strategy': {}, 'macd_trend': macd_trend, 'position_adjust': 0.0,
            })

        dist_div = adaptive_distance(df['DIF'], base_distance=_div_p.get('base_distance', 10))
        div_type, div_idx, div_strength = detect_divergence_single_param(df, df['close'], df['DIF'], distance=dist_div)
        div_decay = signal_with_decay(div_type, div_idx, len(df) - 1, half_life=decay_half_life)
        div_combined = '无明显背离信号'
        if div_type == Divergence.BOTTOM_DIVERGENCE and div_strength > div_strength_threshold:
            div_combined = f"底背离 (强度={div_strength:.2f}, 衰减={div_decay:.2f})"
        elif div_type == Divergence.TOP_DIVERGENCE and div_strength > div_strength_threshold:
            div_combined = f"顶背离 (强度={div_strength:.2f}, 衰减={div_decay:.2f})"
        days_since_div = len(df) - 1 - div_idx if div_type != '' and div_idx is not None else -1
        divergence_price = round(float(df['close'].iloc[div_idx]), 2) if div_type != '' and div_idx is not None and 0 <= div_idx < len(df) else None
        divergence_info = {
            'combined_signal': div_combined,
            'signal_type': div_type, 'idx': div_idx,
            'strength': div_strength, 'decay': div_decay,
            'days_since': days_since_div if days_since_div >= 0 else None,
            'position_price': divergence_price,
        }

        mom_desc, mom_score = _calc_momentum_desc(df, 'DIF', 'DEA', max_score=15)
        slope_info = slope_analysis(df['DIF'], window=slope_window)

        vol_desc, vol_score = _volume_price_trend_score(df, max_bonus=10)
        kp_result = _score_kline_pattern(df, max_score=10)
        kd = df.attrs.get('_kline_pattern_details')
        if kd and kd.get('details'):
            bars_ago = min(d['bars_ago'] for d in kd['details'])
            kp_decay = max(kline_decay_min, 1.0 - bars_ago / kline_decay_days)
            kp_result = (kp_result[0] + f' (衰减{bars_ago}天)', int(kp_result[1] * kp_decay))

        state = _make_state(df, regime, rule_thresholds)
        state.update({
            'divergence': divergence_info,
            'kline_data': df.attrs.get('_kline_pattern_details', None),
            'volume_trend': (vol_desc, vol_score),
            'momentum': {'desc': mom_desc, 'score': mom_score},
            'slope': slope_info,
            'chip_data': df.attrs.get('chip_data', None),
            'macd_trend': macd_trend,
            'current_dif': float(df['DIF'].iloc[-1]) if 'DIF' in df.columns else 0,
            'moneyflow_data': df.attrs.get('moneyflow_data', None),
            'forecast_data': df.attrs.get('forecast_data', None),
            'vol_regime': self._detect_volatility_regime(df),
            'position_adjust': 0.0,
        })

        signal_list = []
        detail_str = df['MACD_SIGNAL_DETAIL'].iloc[-1] if 'MACD_SIGNAL_DETAIL' in df.columns else ''
        if '金叉' in str(detail_str):
            confidence = 'high' if '零轴上' in str(detail_str) else 'medium'
            signal_list.append({'type': 'MACD_金叉', 'confidence': confidence, 'desc': detail_str})
        if '底背离' in div_combined:
            signal_list.append({'type': '底背离', 'confidence': 'high' if div_strength > 0.6 else 'medium', 'desc': div_combined})
        kd = state.get('kline_data')
        if kd and kd.get('details'):
            for d in kd['details']:
                if d['direction'] == '看涨' and d['level'] in ('强反转', '中反转'):
                    signal_list.append({'type': 'K线反转', 'confidence': 'high' if d['level'] == '强反转' else 'medium', 'desc': d['label']})
                    break
        if not signal_list and regime == 'STRONG_TREND' and dif_val > 0:
            signal_list.append({'type': '强势延续', 'confidence': 'high', 'desc': 'MACD多头趋势持续'})
        state['signal_list'] = signal_list

        execute_rules(state, gate=1)
        if not signal_list:
            state.update({'level': 'C', 'score': 0, 'conclusion': 'C: 无明确入场信号'})
            return _pipeline_output(state)

        execute_rules(state, gate=2)
        if state['risk_level'] == 'HIGH':
            return _pipeline_output(state)

        _apply_chip_risk(state, df)
        execute_rules(state, gate=3)

        mult = _get_regime_multiplier(regime)
        trend_mult = _get_macd_trend_mult(macd_trend)

        w_trend = int(weights['MACD趋势'])
        w_cross = int(weights['金叉信号'] * mult['cross'] * trend_mult)
        w_mom = int(weights['柱状动能'] * mult['mom'] * trend_mult)
        w_slope_dim = int(weights['DIF斜率'] * mult['slope'] * trend_mult)
        w_div = int(weights['背离信号'] * mult['div'])
        w_vol = int(weights['量价配合'] * mult['vol'])
        w_kp = int(weights['K线形态'] * mult['kp'])

        scores = {}

        trend_score_map = {MACDTrend.SUPER_STRONG: w_trend, MACDTrend.STRONG: w_trend * 3 // 5,
                           MACDTrend.WEAK: w_trend * 2 // 5, MACDTrend.SUPER_WEAK: 0}
        scores['MACD趋势'] = (f"{macd_trend} (DIF={dif_val:.2f}, DEA={dea_val:.2f})", trend_score_map.get(macd_trend, 0))

        detail_str = df['MACD_SIGNAL_DETAIL'].iloc[-1] if 'MACD_SIGNAL_DETAIL' in df.columns else ''
        is_bull = dif_val > dea_val
        golden_cross_positions = df.index[df['MACD_CROSS'] == 1] if 'MACD_CROSS' in df.columns else []
        atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else float('nan')
        golden_strength = abs(dif_val - dea_val) / atr_val if (not pd.isna(atr_val) and atr_val > 0) else 999
        vol_factor = min(1.0, golden_strength / vol_norm_denom)

        if '零轴上金叉' in str(detail_str):
            base_score = int(w_cross * vol_factor)
            desc = '零轴上金叉'
        elif '零轴下金叉' in str(detail_str):
            base_score = int(w_cross // 2 * vol_factor)
            desc = '零轴下金叉'
        elif is_bull:
            base_score = int(w_cross * 0.75 * vol_factor)
            desc = '多头持续'
        else:
            scores['金叉信号'] = ('空头/死叉', 0)
            base_score = None

        if base_score is not None:
            if len(golden_cross_positions) > 0:
                days_since = len(df) - 1 - golden_cross_positions[-1]
                decay = max(cross_decay_min, 1.0 - days_since / cross_decay_days)
                base_score = int(base_score * decay)
                desc += f' (衰减{days_since}天×{decay:.2f})'
            if vol_factor < 0.8:
                desc += f' [vol_norm={vol_factor:.2f}]'
            scores['金叉信号'] = (desc, base_score)

        prefix = '强势' if mom_score >= w_mom // 2 else ('关注' if mom_score >= w_mom // 4 else '弱势')
        scores['柱状动能'] = (f'{prefix}: {mom_desc}', min(mom_score, w_mom))

        if slope_info['trend'] == '明确上行':
            scores['DIF斜率'] = (f"确认 {slope_info['trend']} (R²={slope_info['r2']:.4f})", w_slope_dim)
        elif '上行' in slope_info['trend']:
            scores['DIF斜率'] = (f"关注 {slope_info['trend']} (R²={slope_info['r2']:.4f})", int(w_slope_dim * 0.55))
        else:
            scores['DIF斜率'] = (f"弱势 {slope_info['trend']} (R²={slope_info['r2']:.4f})", 0)

        has_top_div = div_type == Divergence.TOP_DIVERGENCE
        eff = round(div_strength * div_decay, 3)
        if '底背离' in div_combined:
            scores['背离信号'] = (f'确认 底背离 (强度={div_strength:.2f}, 有效={eff})', int(w_div * (0.5 + 0.5 * eff)))
        elif has_top_div:
            scores['背离信号'] = (f'否定 顶背离（一票否决）', 0)
        else:
            scores['背离信号'] = ('中性: 无背离信号', 0)

        vol_bonus = 0 if has_top_div else vol_score
        scores['量价配合'] = (vol_desc, min(vol_bonus, w_vol))
        scores['K线形态'] = kp_result

        base_keys = [k for k in scores if k != '量价配合']
        total_base = sum(scores[k][1] for k in base_keys)
        total_max_base = w_trend + w_cross + w_mom + w_slope_dim + w_div + w_kp
        total_base = max(0, min(total_max_base, total_base))
        total = max(0, min(total_max_base + w_vol, total_base + scores['量价配合'][1]))

        mf_bonus, mf_desc = _calc_moneyflow_score(state.get('moneyflow_data'))
        if mf_bonus != 0:
            total = max(0, min(100, total + mf_bonus))
            state.setdefault('_notes', [])
            state['_notes'].append(mf_desc)

        mtf = self._check_multitimeframe(df)
        total = max(0, min(100, int(total * mtf['multiplier'])))
        if mtf['multiplier'] != 1.0:
            state.setdefault('_notes', [])
            state['_notes'].append(mtf['desc'])
        state['mtf_alignment'] = mtf['alignment']

        state['exit_strategy'] = _calc_exit_strategy(df, params=_score_p)

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
        support = df['close'].iloc[-expected_return_lookback:].min()
        resistance = df['close'].iloc[-expected_return_lookback:].max()
        risk_pct = max((close_v - support) / close_v, 0.005)
        reward_pct = max((resistance - close_v) / close_v, 0)
        state['expected_return'] = round(reward_pct * 100, 1)
        state['risk_reward_ratio'] = round(reward_pct / risk_pct, 2) if risk_pct > 0 else 0
        state['_scores'] = scores

        # Gate 0.5: 宏观情景注入（基于最终评分 + 级别）
        execute_rules(state, gate=0.5)

        # Gate 4: 仓位调整（基于最终评分 + 级别 + 风控）
        execute_rules(state, gate=4)

        return _pipeline_output(state)

    def analyze_full_bull(
        self,
        df: pd.DataFrame,
        decay_half_life: int = 8,
        slope_window: int = 5,
        recalc_macd: bool = True,
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
    ) -> dict:
        if recalc_macd:
            df = self._custom_macd(df)
        else:
            required = {"DIF", "DEA", "MACD_HIST", "MACD_SIGNAL_DETAIL"}
            if not required.issubset(df.columns):
                df = self._custom_macd(df)

        result = self.pipeline_analysis(
            df, weights=weights, thresholds=thresholds,
            decay_half_life=decay_half_life, slope_window=slope_window,
        )
        return {
            "score": result.get("score", 0),
            "score_base": result.get("score_base", 0),
            "conclusion": result.get("conclusion", ""),
            "details": result.get("details", {}),
            "divergence": result.get("divergence", {}),
            "momentum": result.get("momentum", ""),
            "slope": result.get("slope", {}),
            "winrate_ref": result.get("winrate_ref", ""),
            "macd_trend": result.get("macd_trend", ""),
        }
