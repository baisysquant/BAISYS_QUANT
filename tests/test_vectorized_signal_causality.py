"""vectorized_signal 因果性（防前视泄漏）测试。

核心性质：任意 bar 的信号只能依赖"截至该 bar 收盘"的数据。
用完整性不变量验证：对同一数据的前缀（截断到 m）单独计算，
与在全量数据上计算的同一前缀相比，逐 bar 结果必须逐位相等。
若某 bar 借用了未来 bar 的数据（例如全样本 mean），两条路径必然不一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BackTrading.vectorized_signal import _kline_pattern


def _df(o, c, h, l) -> pd.DataFrame:
    n = len(o)
    return pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": np.asarray(o, dtype=float),
        "high": np.asarray(h, dtype=float),
        "low": np.asarray(l, dtype=float),
        "close": np.asarray(c, dtype=float),
    })


def test_kline_pattern_prefix_invariance_no_future_leak() -> None:
    """完整长度 vs 裁剪前缀：重叠区间的 K 线形态分必须逐位一致。

    旧实现 body_ma20 早期窗口回退到全样本 np.nanmean(body)，
    使前 5 根 bar 附近（经 eng_acc 滚动窗口传播到 bar 5/6）
    的信号被未来 bar 污染 → 该测试在旧代码上必然失败。
    """
    # bar0: 长阴（晨星第一腿，body=2）; bar1: 十字小实体（body=0.1）;
    # bar2: 长阳确认，收过 bar0 中点 11 → 若 mid 判定成立则为晨星
    o = [12.0, 10.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7]
    c = [10.0, 10.0, 11.4, 11.35, 11.45, 11.55, 11.65, 11.75]
    h = [12.1, 10.15, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1]
    l = [9.9, 9.95, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6]
    # 未来巨实体（body=60）：全样本 mean 会被它抬高 → 旧代码晨星阈值失真
    o = o + [1.0]
    c = c + [61.0]
    h = h + [61.0]
    l = l + [0.9]

    full = _df(o, c, h, l)
    prefix = full.iloc[:8].reset_index(drop=True)

    k_full = _kline_pattern(full, max_score=10)
    k_trunc = _kline_pattern(prefix, max_score=10)

    assert k_full[:8].tolist() == k_trunc.tolist()


def test_kline_pattern_prefix_invariance_random_walk() -> None:
    """随机游走去向前缀不变量：多裁剪点全部一致。"""
    rng = np.random.default_rng(42)
    n = 120
    close = 10.0 + np.cumsum(rng.normal(0, 0.2, n))
    prev = np.concatenate([[close[0]], close[:-1]])
    o = prev * (1 + rng.uniform(-0.01, 0.01, n))
    h = np.maximum(o, close) * 1.01
    l = np.minimum(o, close) * 0.99
    full = _df(o, close, h, l)
    k_full = _kline_pattern(full, max_score=10)

    for m in (9, 15, 30, 60):
        pre = _df(o[:m], close[:m], h[:m], l[:m])
        k_trunc = _kline_pattern(pre, max_score=10)
        assert k_full[:m].tolist() == k_trunc.tolist(), f"mismatch at m={m}"


def test_kline_pattern_causal_fallback_small_sample() -> None:
    """早期不足 20 根时回退必须是因果的：禁止出现 future-influenced 全样本均值。"""
    o = [12.0, 10.1, 11.2, 11.3]
    c = [10.0, 10.0, 11.4, 11.35]
    h = [12.1, 10.15, 11.6, 11.7]
    l = [9.9, 9.95, 11.1, 11.2]
    # 追加极端尾部使全样本均值含未来信息
    for _ in range(6):
        o.append(1.0)
        c.append(60.0)
        h.append(60.0)
        l.append(0.9)
    full = _df(o, c, h, l)
    prefix = _df(o[:4], c[:4], h[:4], l[:4])

    k_full = _kline_pattern(full, max_score=10)
    k_trunc = _kline_pattern(prefix, max_score=10)
    # n<5 → 全零；前 4 bar 不允许因"未来均值"出现非零
    assert (k_trunc == 0).all()
    assert k_full[:4].tolist() == k_trunc.tolist()


def _indicator_df(n: int = 200) -> pd.DataFrame:
    """构造带 DIF/DEA/ATR/MA 的指标 DataFrame（与 conftest 逻辑一致）。"""
    rng = np.random.default_rng(7)
    close = pd.Series(10.0 * (1 + np.linspace(0, 0.3, n)) + rng.normal(0, 0.2, n)).clip(lower=5.0)
    high = close + np.abs(rng.normal(0, 0.3, n)).reshape(-1)
    low = close - np.abs(rng.normal(0, 0.3, n)).reshape(-1)
    opn = close + rng.normal(0, 0.1, n).reshape(-1)
    df = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n),
        "amount": close * 5_000_000,
    })
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["ATR"] = (high - low).rolling(14).mean()
    for p in (5, 10, 20, 30, 60):
        df[f"MA_{p}"] = close.rolling(p).mean()
    return df


def test_compute_signals_prefix_invariance() -> None:
    """整条评分管线（compute_signals）前缀不变量：裁剪截断不得改变重叠信号。"""
    from BackTrading.vectorized_signal import compute_signals

    full = _indicator_df(n=200)
    sig_full = compute_signals(full, params={})

    for m in (40, 80, 120):
        pre = full.iloc[:m].reset_index(drop=True)
        sig_pre = compute_signals(pre, params={})
        for col in ("entry_score", "exit_score", "kline", "score"):
            a = sig_full[col].iloc[:m].fillna(0.0).to_numpy()
            b = sig_pre[col].fillna(0.0).to_numpy()
            assert np.array_equal(a, b, equal_nan=True), f"m={m} col={col} 前视偏差"
        assert sig_full["risk_level"].iloc[:m].tolist() == sig_pre["risk_level"].tolist()