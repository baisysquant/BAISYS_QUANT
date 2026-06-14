import pandas as pd

from LogicAnalyzer.MACDDivergence import slope_analysis
from LogicAnalyzer.SignalConstants import MACDTrend


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


def _detect_market_regime(df: pd.DataFrame, boll_col: str | None = None,
                          params: dict | None = None) -> str:
    """检测市场状态。

    Args:
        df: K 线 DataFrame
        boll_col: BOLL 带宽列名（用于窄布林判定）
        params: 可选参数字典，键名见 ConfigParser.REGIME_DETECTION。
                不传则使用默认值（同 magic number 旧值）。
    """
    if params is None:
        params = {}
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
    if boll_col and boll_col in df.columns and len(df) > 5:
        recent_bw = df[boll_col].iloc[-5:].mean()
        hist_bw = df[boll_col].iloc[:-5].mean()
        boll_narrow_ratio = params.get('boll_narrow_ratio', 0.8)
        is_narrow_boll = recent_bw < hist_bw * boll_narrow_ratio
    if ma_bullish and slope_positive and momentum_positive:
        return 'STRONG_TREND'
    if ma_bearish and dif < 0 and not momentum_positive:
        return 'WEAK_TREND'
    if is_narrow_boll and hist is not None:
        osc_min_bars = params.get('oscillation_min_bars', 30)
        hist_std_ratio = params.get('oscillation_hist_std_ratio', 0.1)
        if len(df) > osc_min_bars and abs(hist.iloc[-1]) < hist_std_ratio * df['close'].std():
            return 'OSCILLATION'
    reversal_lookback = params.get('reversal_lookback', 10)
    if not ma_bullish and 'DIF' in df.columns and len(df) >= reversal_lookback:
        dif_series = df['DIF']
        if dif < 0 and dif_series.iloc[-1] > dif_series.iloc[-reversal_lookback] and hist is not None:
            if hist.iloc[-1] > hist.iloc[-reversal_lookback]:
                return 'BOTTOM_REVERSAL'
    if ma_bullish and 'DIF' in df.columns and len(df) >= reversal_lookback:
        dif_series = df['DIF']
        close_vs_ma20 = (close - ma20) / ma20 if ma20 > 0 else 0
        top_risk_dev = params.get('top_risk_ma20_deviation', 0.15)
        if close_vs_ma20 > top_risk_dev and dif_series.iloc[-1] < dif_series.iloc[-reversal_lookback] and hist is not None:
            if hist.iloc[-1] < hist.iloc[-reversal_lookback]:
                return 'TOP_RISK'
    regime_from_attrs = df.attrs.get('_regime_hint', None) if hasattr(df, 'attrs') else None
    return regime_from_attrs if regime_from_attrs else 'UNCLEAR'


def _calc_exit_strategy(df: pd.DataFrame, params: dict | None = None) -> dict:
    """计算退出策略（止损/目标价/移动止损/盈亏比）。

    Args:
        df: K 线 DataFrame
        params: 可选参数字典，键名见 ConfigParser.SCORING_PARAMS。
                不传则使用默认值。
    """
    if params is None:
        params = {}
    close = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else float('nan')

    if pd.isna(atr) or atr <= 0:
        return {'stop_loss': None, 't1_target': None, 't2_target': None,
                'trailing_stop': None, 'exit_rrr': None}

    atr_stop = params.get('atr_stop_mult', 1.5)
    atr_t1 = params.get('atr_t1_mult', 3.0)
    atr_t2 = params.get('atr_t2_mult', 5.0)
    stop_loss = round(float(close - atr * atr_stop), 2)
    t1 = round(float(close + atr * atr_t1), 2)
    t2 = round(float(close + atr * atr_t2), 2)

    trailing_stop = None
    high_lookback = params.get('trailing_stop_high_lookback', 20)
    high_ratio = params.get('trailing_stop_high_ratio', 0.98)
    stop_lookback = params.get('trailing_stop_lookback', 10)
    if len(df) >= high_lookback:
        recent_high = df['high'].iloc[-high_lookback:].max()
        if close >= recent_high * high_ratio:
            trailing_stop = round(float(df['low'].iloc[-stop_lookback:].min()), 2)

    risk = float(close - stop_loss) if stop_loss and close > stop_loss else 0.01
    reward = float(t1 - close)
    rrr = round(reward / risk, 2) if risk > 0 else 0

    return {
        'stop_loss': stop_loss, 't1_target': t1, 't2_target': t2,
        'trailing_stop': trailing_stop, 'exit_rrr': rrr,
    }


def _pipeline_output(state: dict) -> dict:
    notes = state.get('_notes', [])
    details = {}
    if '_scores' in state:
        details = {k: {'desc': v[0], 'score': v[1]} for k, v in state['_scores'].items()}
    else:
        macd_desc = state.get('conclusion', '') or state.get('risk_desc', '') or ''
        default_score = state.get('score', 0)
        details = {
            'MACD趋势': {'desc': macd_desc or '指标偏弱', 'score': default_score},
            '金叉信号': {'desc': '未触发', 'score': 0},
            '柱状动能': {'desc': '未触发', 'score': 0},
            'DIF斜率': {'desc': '未触发', 'score': 0},
            '背离信号': {'desc': '未触发', 'score': 0},
            '量价配合': {'desc': '未触发', 'score': 0},
            'K线形态': {'desc': '未触发', 'score': 0},
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
        "macd_trend": state.get('macd_trend', ''),
        "position_adjust": state.get('position_adjust', 0.0),
    }
