import numpy as np
import pandas as pd


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


def _calc_moneyflow_score(mf_data: dict | None) -> tuple[int, str]:
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


def calc_entry_signal(df: pd.DataFrame, min_score: float = 60.0) -> pd.Series:
    """按日入场信号判断。

    基于 DataFrame 中已预计算的评分列（进场评分、综合评分、风险等级）
    逐行判断是否应买入。

    Args:
        df: 多股票 DataFrame，每行一个股票一个交易日。
             必须至少包含列：进场评分 或 综合评分。
        min_score: 最低入场评分阈值（默认 60）。

    Returns:
        pd.Series[bool]: True 表示该股票当日应买入。
    """
    entry_series = pd.Series(False, index=df.index)

    # 综合评分入场
    score_col = None
    for candidate in ("进场评分", "综合评分", "score"):
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col:
        score = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
        entry_series |= score >= min_score

    # 风控过滤：高风险/弱势不买入
    if "风险等级" in df.columns:
        risk = df["风险等级"].astype(str).str.upper()
        entry_series &= ~risk.isin({"HIGH", "D", "E"})

    return entry_series
