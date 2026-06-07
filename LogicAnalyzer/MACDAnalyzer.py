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
        """多周期对齐检查：用 rolling(5) 模拟周线，对比日线-周线方向。

        Returns:
            {'alignment': str, 'multiplier': float, 'desc': str}
        """
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
            return {'alignment': 'ALIGNED_BULL', 'multiplier': 1.1,
                    'desc': '日周共振多头'}
        if not d_bull and not w_bull:
            return {'alignment': 'ALIGNED_BEAR', 'multiplier': 1.0,
                    'desc': '日周共振空头'}
        if d_bull and not w_bull:
            return {'alignment': 'WEEKLY_BEAR_DAILY_BULL', 'multiplier': 0.5,
                    'desc': '周线空头日线金叉(反弹诱多)'}
        return {'alignment': 'WEEKLY_BULL_DAILY_BEAR', 'multiplier': 0.6,
                'desc': '周线多头日线死叉(回调买入)'}

    @staticmethod
    def _detect_volatility_regime(df: pd.DataFrame) -> str:
        """波动率情景分类。

        机构做法：
          HIGH_VOL_TREND  ← ATR 高于自身均值30%+且ADX>25（高波动趋势市）
          LOW_VOL_REVERSAL ← ATR 低于自身均值30%+且ADX<20（低波动震荡/反转市）
          NORMAL           ← 其他

        影响：Gate 3 规则根据情景切换倾向（趋势规则 vs 反转规则）。
        """
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
        """预计算规则引擎所需的衍生指标（ADX/ATR/MA200/KDJ/RSI）。"""
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

    def pipeline_analysis(
        self,
        df: pd.DataFrame,
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
        rule_thresholds: dict | None = None,
        decay_half_life: int = 8,
        slope_window: int = 5,
    ) -> dict:
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

        # Step 1: MACD趋势分级
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

        # Step 1: 情景检测（以MACD趋势辅助修正）
        regime = _detect_market_regime(df, boll_col=boll_bw_col if boll_bw_col else None)
        if regime in ('WEAK_TREND',) and macd_trend != MACDTrend.SUPER_STRONG:
            state = _make_state(df, regime, rule_thresholds)
            state.update({'level': 'C', 'score': 0, 'conclusion': 'C: 弱势趋势，环境不配合'})
            return _pipeline_output(state)

        # MACD趋势对情景的修正
        if macd_trend == MACDTrend.SUPER_STRONG and regime in ('OSCILLATION', 'UNCLEAR'):
            regime = 'STRONG_TREND'

        # Step 2: 背离检测（单参数，半衰期=decay_half_life）
        dist_div = adaptive_distance(df['DIF'], base_distance=10)
        div_type, div_idx, div_strength = detect_divergence_single_param(df, df['close'], df['DIF'], distance=dist_div)
        div_decay = signal_with_decay(div_type, div_idx, len(df) - 1, half_life=decay_half_life)
        div_combined = '无明显背离信号'
        if div_type == Divergence.BOTTOM_DIVERGENCE and div_strength > 0.15:
            div_combined = f"底背离 (强度={div_strength:.2f}, 衰减={div_decay:.2f})"
        elif div_type == Divergence.TOP_DIVERGENCE and div_strength > 0.15:
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

        # Step 3: 动能
        mom_desc, mom_score = _calc_momentum_desc(df, 'DIF', 'DEA', max_score=15)

        # Step 4: 斜率
        slope_info = slope_analysis(df['DIF'], window=slope_window)

        # Step 5: 量价配合 + K线形态
        vol_desc, vol_score = _volume_price_trend_score(df, max_bonus=10)
        kp_result = _score_kline_pattern(df, max_score=10)
        # K线形态时间衰减（半衰期=5）：根据最强形态的 bars_ago 调整
        kd = df.attrs.get('_kline_pattern_details')
        if kd and kd.get('details'):
            bars_ago = min(d['bars_ago'] for d in kd['details'])
            kp_decay = max(0.2, 1.0 - bars_ago / 10)
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
        })

        # 信号列表
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

        # 筹码风险
        _apply_chip_risk(state, df)
        execute_rules(state, gate=3)

        # 情景倍数 & MACD趋势倍数
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

        # MACD趋势（自身分）
        trend_score_map = {MACDTrend.SUPER_STRONG: w_trend, MACDTrend.STRONG: w_trend * 3 // 5,
                           MACDTrend.WEAK: w_trend * 2 // 5, MACDTrend.SUPER_WEAK: 0}
        scores['MACD趋势'] = (f"{macd_trend} (DIF={dif_val:.2f}, DEA={dea_val:.2f})", trend_score_map.get(macd_trend, 0))

        # 金叉信号（含时间衰减半衰期15 + 波动率归一化）
        detail_str = df['MACD_SIGNAL_DETAIL'].iloc[-1] if 'MACD_SIGNAL_DETAIL' in df.columns else ''
        is_bull = dif_val > dea_val
        golden_cross_positions = df.index[df['MACD_CROSS'] == 1] if 'MACD_CROSS' in df.columns else []
        # 波动率归一化：金叉强度 = (DIF-DEA)/ATR，低于阈值(0.15)则降权
        atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else float('nan')
        golden_strength = abs(dif_val - dea_val) / atr_val if (not pd.isna(atr_val) and atr_val > 0) else 999
        vol_factor = min(1.0, golden_strength / 0.15)

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
                decay = max(0.3, 1.0 - days_since / 30)
                base_score = int(base_score * decay)
                desc += f' (衰减{days_since}天×{decay:.2f})'
            if vol_factor < 0.8:
                desc += f' [vol_norm={vol_factor:.2f}]'
            scores['金叉信号'] = (desc, base_score)

        # 柱状动能
        prefix = '强势' if mom_score >= w_mom // 2 else ('关注' if mom_score >= w_mom // 4 else '弱势')
        scores['柱状动能'] = (f'{prefix}: {mom_desc}', min(mom_score, w_mom))

        # DIF斜率
        if slope_info['trend'] == '明确上行':
            scores['DIF斜率'] = (f"确认 {slope_info['trend']} (R²={slope_info['r2']:.4f})", w_slope_dim)
        elif '上行' in slope_info['trend']:
            scores['DIF斜率'] = (f"关注 {slope_info['trend']} (R²={slope_info['r2']:.4f})", int(w_slope_dim * 0.55))
        else:
            scores['DIF斜率'] = (f"弱势 {slope_info['trend']} (R²={slope_info['r2']:.4f})", 0)

        # 背离信号
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

        # ── 资金流向评分（加/扣分，-5 ~ +5） ────────────────────────────────
        mf_bonus, mf_desc = _calc_moneyflow_score(state.get('moneyflow_data'))
        if mf_bonus != 0:
            total = max(0, min(100, total + mf_bonus))
            state.setdefault('_notes', [])
            state['_notes'].append(mf_desc)

        # ── 多周期对齐修正（日线vs周线，0.5~1.1×） ──────────────────────────
        mtf = self._check_multitimeframe(df)
        total = max(0, min(100, int(total * mtf['multiplier'])))
        if mtf['multiplier'] != 1.0:
            state.setdefault('_notes', [])
            state['_notes'].append(mtf['desc'])
        state['mtf_alignment'] = mtf['alignment']

        # ── 退出策略（纯输出层） ────────────────────────────────────────────────
        state['exit_strategy'] = _calc_exit_strategy(df)

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
        recalc_macd: bool = True,
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
    ) -> dict:
        """统一入口，委托给 pipeline_analysis。

        保持相同返回格式以兼容下游调用者，避免双轨输出不一致。
        """
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


def _make_state(df: pd.DataFrame, regime: str, rule_thresholds: dict) -> dict:
    return {
        'df': df, 'regime': regime, 'signal_list': [],
        'risk_level': 'NONE', 'risk_desc': '', 'conclusion': '',
        'level': 'C', 'score': 0, 'triggered_rules': [], '_notes': [],
        'divergence': {}, 'config': rule_thresholds,
    }


def _get_regime_multiplier(regime: str) -> dict[str, float]:
    regime_mult = {
        'STRONG_TREND':    {'cross': 1.0, 'mom': 1.5, 'slope': 1.5, 'div': 1.0, 'vol': 1.0, 'kp': 0.3},
        'BOTTOM_REVERSAL': {'cross': 1.0, 'mom': 0.8, 'slope': 0.8, 'div': 2.0, 'vol': 1.5, 'kp': 1.5},
        'TOP_RISK':        {'cross': 0.5, 'mom': 0.5, 'slope': 0.5, 'div': 2.0, 'vol': 1.5, 'kp': 2.0},
        'OSCILLATION':     {'cross': 0.5, 'mom': 0.3, 'slope': 0.3, 'div': 1.5, 'vol': 0.5, 'kp': 1.2},
        'UNCLEAR':         {'cross': 1.0, 'mom': 1.0, 'slope': 1.0, 'div': 1.0, 'vol': 1.0, 'kp': 1.0},
    }
    return regime_mult.get(regime, regime_mult['UNCLEAR'])


def _get_macd_trend_mult(macd_trend: str) -> float:
    return {
        MACDTrend.SUPER_STRONG: 1.3,
        MACDTrend.STRONG: 1.0,
        MACDTrend.WEAK: 0.5,
        MACDTrend.SUPER_WEAK: 0.0,
    }.get(macd_trend, 1.0)


def _calc_moneyflow_score(mf_data: dict | None) -> tuple[int, str]:
    """计算资金流向评分（-5 ~ +5）。

    指标：
      主力净买入率 = (buy_lg + buy_elg - sell_lg - sell_elg) / 总成交额
      散户净买入率 = (buy_sm - sell_sm) / 总成交额
      聪明钱背离度 = 主力净买入率 - 散户净买入率

    Args:
        mf_data: moneyflow 原始数据字典（来自 moneyflow API 的一行）

    Returns:
        (bonus, description): 加分/扣分 + 描述文本
    """
    if not mf_data:
        return 0, ''

    lg_buy = float(mf_data.get('buy_lg_amount', 0))
    lg_sell = float(mf_data.get('sell_lg_amount', 0))
    elg_buy = float(mf_data.get('buy_elg_amount', 0))
    elg_sell = float(mf_data.get('sell_elg_amount', 0))
    md_buy = float(mf_data.get('buy_md_amount', 0))
    md_sell = float(mf_data.get('sell_md_amount', 0))
    sm_buy = float(mf_data.get('buy_sm_amount', 0))
    sm_sell = float(mf_data.get('sell_sm_amount', 0))

    total_amount = lg_buy + lg_sell + elg_buy + elg_sell + md_buy + md_sell + sm_buy + sm_sell
    if total_amount <= 0:
        return 0, ''

    net_lg_inflow = (lg_buy + elg_buy - lg_sell - elg_sell) / total_amount * 100
    net_sm_inflow = (sm_buy - sm_sell) / total_amount * 100
    divergence = net_lg_inflow - net_sm_inflow

    if net_lg_inflow > 0:
        desc = f'主力净买{net_lg_inflow:.0f}% '
    else:
        desc = f'主力净卖{abs(net_lg_inflow):.0f}% '

    if divergence > 15:
        return 5, desc + '(大户进、散户出，健康做多)'
    if divergence > 5:
        return 2, desc + '(主力温和流入)'
    if divergence < -15:
        return -5, desc + '(派发: 大户出、散户接)'
    if divergence < -5:
        return -2, desc + '(主力温和流出)'
    return 0, desc


def _apply_chip_risk(state: dict, df: pd.DataFrame) -> None:
    chip = state.get('chip_data')
    if not chip:
        return
    winner_rate = chip.get('winner_rate', 0)
    close_val = df['close'].iloc[-1]
    cost_95pct = chip.get('cost_95pct', 0)
    cost_5pct = chip.get('cost_5pct', 0)
    regime = state.get('regime', 'UNCLEAR')
    risk = state['risk_level']
    if winner_rate > 80 and regime in ('WEAK_TREND', 'TOP_RISK', 'UNCLEAR'):
        if risk in ('NONE', 'LOW'):
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
                    state['risk_level'] = 'HIGH' if regime in ('WEAK_TREND', 'TOP_RISK') else 'MEDIUM'
                    state['risk_desc'] += ' 筹码密集区高位'


def _detect_market_regime(df: pd.DataFrame, boll_col: str | None = None) -> str:
    close = float(df['close'].iloc[-1])
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma30 = df['close'].rolling(30).mean().iloc[-1]
    ma60 = df['close'].rolling(60).mean().iloc[-1]
    ma_bullish = ma5 > ma10 > ma20 > ma30 > ma60
    ma_bearish = ma5 < ma10 < ma20 < ma30 < ma60
    dif = df['DIF'].iloc[-1] if 'DIF' in df.columns else 0
    hist = df['DIF'] - df['DEA'] if 'DIF' in df.columns and 'DEA' in df.columns else None
    momentum_positive = hist.iloc[-1] > 0 if hist is not None else False
    slope_info = slope_analysis(df['DIF'] if 'DIF' in df.columns else df['close'])
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
    if is_narrow_boll and hist is not None and len(df) > 30 and abs(hist.iloc[-1]) < 0.1 * df['close'].std():
        return 'OSCILLATION'
    if not ma_bullish and 'DIF' in df.columns and len(df) >= 10:
        dif_series = df['DIF']
        if dif < 0 and dif_series.iloc[-1] > dif_series.iloc[-5] and hist is not None:
            if hist.iloc[-1] > hist.iloc[-5]:
                return 'BOTTOM_REVERSAL'
    if ma_bullish and 'DIF' in df.columns and len(df) >= 10:
        dif_series = df['DIF']
        close_vs_ma20 = (close - ma20) / ma20 if ma20 > 0 else 0
        if close_vs_ma20 > 0.15 and dif_series.iloc[-1] < dif_series.iloc[-5] and hist is not None:
            if hist.iloc[-1] < hist.iloc[-5]:
                return 'TOP_RISK'
    regime_from_attrs = df.attrs.get('_regime_hint', None)
    return regime_from_attrs if regime_from_attrs else 'UNCLEAR'


# ── 退出策略计算 ──────────────────────────────────────────────────────────────


def _calc_exit_strategy(df: pd.DataFrame) -> dict:
    """基于 ATR 的止损/目标价计算（纯输出层，不参与评分）。"""
    close = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else float('nan')

    if pd.isna(atr) or atr <= 0:
        return {'stop_loss': None, 't1_target': None, 't2_target': None,
                'trailing_stop': None, 'exit_rrr': None}

    stop_loss = round(float(close - atr * 1.5), 2)
    t1 = round(float(close + atr * 3.0), 2)
    t2 = round(float(close + atr * 5.0), 2)

    trailing_stop = None
    if len(df) >= 20:
        recent_high = df['high'].iloc[-20:].max()
        if close >= recent_high * 0.98:
            trailing_stop = round(float(df['low'].iloc[-10:].min()), 2)

    risk = float(close - stop_loss) if stop_loss and close > stop_loss else 0.01
    reward = float(t1 - close)
    rrr = round(reward / risk, 2) if risk > 0 else 0

    return {
        'stop_loss': stop_loss, 't1_target': t1, 't2_target': t2,
        'trailing_stop': trailing_stop, 'exit_rrr': rrr,
    }


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
    else:
        # 早期返回路径：用 state 已有信息推一个最小化的 MACD趋势
        macd_desc = state.get('conclusion', '') or state.get('risk_desc', '') or ''
        details = {
            'MACD趋势': {'desc': macd_desc or '指标偏弱', 'score': state.get('score', 0)},
        }

    exit_strat = state.get('exit_strategy', {})

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
        "divergence_days": state.get('divergence', {}).get('days_since'),
        "divergence_price": state.get('divergence', {}).get('position_price'),
        "current_dif": state.get('current_dif', 0),
        "momentum": state.get('momentum', {}).get('desc', ''),
        "slope": state.get('slope', {}),
        "winrate_ref": '参考 pipeline',
        "expected_return": state.get('expected_return', 0),
        "risk_reward_ratio": state.get('risk_reward_ratio', 0),
        "stop_loss": exit_strat.get('stop_loss'),
        "t1_target": exit_strat.get('t1_target'),
        "t2_target": exit_strat.get('t2_target'),
        "trailing_stop": exit_strat.get('trailing_stop'),
        "exit_rrr": exit_strat.get('exit_rrr'),
    }
