"""指标计算降级（degradation）+ 置信度标签 — 测试。

验收:
1. STRICT: 短序列按原周期全窗计算，头部 NaN，不抛异常（等价原行为）。
2. RELAX: 短序列缩窗计算，结果标 low_confidence，start_bar 正确。
3. SKIP: 短序列整段 low_confidence，策略层跳过信号。
4. _compute_indicators 输出 bar 级 _IND_CONF 列 + attrs _confidence 汇总。
5. apply_confidence_consumption: skip 归零 / low_weight 按系数降权。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BackTrading import degradation as deg
from BackTrading.degradation import (
    Confidence,
    DegradeMode,
    IndicatorResult,
    apply_confidence_consumption,
    low_confidence_mask,
    resolve_period,
)
from BackTrading.prepare import _compute_indicators


def _valid_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.maximum(close, 5.0)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "trade_date": pd.date_range("2023-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "symbol": "sh600000",
        "open": close - 0.05,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": vol,
    })


# ── resolve_period 决策 ─────────────────────────────────────────────────

def test_resolve_period_sufficient():
    res = resolve_period("ma_20", 20, n=300, mode=DegradeMode.RELAX)
    assert res.value == 20 and res.confidence.is_low is False and res.start_bar == 19


def test_resolve_period_strict_short():
    res = resolve_period("ma_200", 200, n=50, mode=DegradeMode.STRICT)
    assert res.value == 200 and res.confidence.is_low is False and res.start_bar == 199


def test_resolve_period_relax_short():
    res = resolve_period("ma_200", 200, n=50, mode=DegradeMode.RELAX)
    assert res.value < 200 and res.confidence.is_low and res.start_bar == res.value - 1
    assert "degraded" in res.confidence.reasons[0]


def test_resolve_period_skip_short():
    res = resolve_period("ma_200", 200, n=50, mode=DegradeMode.SKIP)
    assert res.value == 200 and res.confidence.is_low and res.start_bar == 0
    assert "SKIP" in res.confidence.reasons[0]


def test_resolve_period_relax_min_periods_floor():
    res = resolve_period("ma_200", 200, n=50, mode=DegradeMode.RELAX, min_periods=40)
    assert res.value >= 40


# ── safe wrappers 不抛异常 ──────────────────────────────────────────────

@pytest.mark.parametrize("mode", [DegradeMode.STRICT, DegradeMode.RELAX, DegradeMode.SKIP])
def test_safe_macd_never_raises_short(mode):
    df = _valid_df(n=10)
    res = deg.safe_macd(df["close"], mode=mode)
    assert isinstance(res, IndicatorResult)
    assert res.value is not None


def test_safe_ma_relax_valid_from_start_bar():
    df = _valid_df(n=50)
    res = deg.safe_ma(df["close"], 200, mode=DegradeMode.RELAX)
    vals = res.value.to_numpy()
    assert np.isnan(vals[: res.start_bar]).all()
    assert not np.isnan(vals[res.start_bar :]).any()


def test_safe_stoch_relax_short():
    df = _valid_df(n=5)
    res = deg.safe_stoch(df["high"], df["low"], df["close"], mode=DegradeMode.RELAX)
    assert res.value is not None and res.confidence.is_low


# ── _compute_indicators 端到端：置信度列 + attrs ───────────────────────

def test_compute_indicators_long_series_high_confidence():
    df = _compute_indicators(_valid_df(n=300))
    assert (df["_IND_CONF"] == "high").all()
    assert df.attrs.get("_confidence") is None


def test_compute_indicators_short_series_relax_marks_low_confidence():
    df = _compute_indicators(_valid_df(n=40))
    conf = df.attrs.get("_confidence")
    assert conf is not None and conf["level"] == "low"
    start = conf["start_bar"]
    assert start >= 0
    assert conf["mode"] == "RELAX"
    assert len(conf["reasons"]) > 0
    assert (df["_IND_CONF"].iloc[:start] == "high").all()
    assert (df["_IND_CONF"].iloc[start:] == "low").all()


def test_compute_indicators_short_series_strict_no_low_flag():
    from unittest.mock import patch

    with patch.object(deg, "degrade_mode", return_value=DegradeMode.STRICT):
        df2 = _compute_indicators(_valid_df(n=40))
    assert (df2["_IND_CONF"] == "high").all()
    assert df2.attrs.get("_confidence") is None
    # STRICT 保持原周期全窗：n=40 < 60 → MA_60 全 NaN（等价原行为）
    assert df2["MA_60"].isna().all()


def test_compute_indicators_confidence_flag_off():
    df = _compute_indicators(_valid_df(n=40), confidence_flag=False)
    assert "_IND_CONF" not in df.columns


# ── 策略层消费 ─────────────────────────────────────────────────────────

def test_low_confidence_mask_fallback():
    assert low_confidence_mask(_valid_df()).sum() == 0


def test_low_confidence_mask_detects_column():
    df = _valid_df(n=100)
    df["_IND_CONF"] = "high"
    df.loc[50:, "_IND_CONF"] = "low"
    mask = low_confidence_mask(df)
    assert mask[:50].sum() == 0 and mask[50:].sum() == 50


def test_apply_consumption_skip_zeroes_scores():
    scores = pd.DataFrame({"entry_score": np.full(100, 80.0), "score": np.full(100, 80.0)})
    df = _valid_df(n=100)
    df["_IND_CONF"] = "low"
    df.loc[:49, "_IND_CONF"] = "high"
    mask = apply_confidence_consumption(scores, df, params={"indicator_degradation": {}})
    assert mask.sum() == 50
    assert (scores.loc[50:, "entry_score"] == 0.0).all()
    assert (scores.loc[:49, "entry_score"] == 80.0).all()


def test_apply_consumption_low_weight():
    scores = pd.DataFrame({"entry_score": np.full(100, 80.0), "golden_cross": np.full(100, 10.0)})
    df = _valid_df(n=100)
    df["_IND_CONF"] = "low"
    conf_p = {"indicator_degradation": {"low_confidence_action": "low_weight",
                                        "low_confidence_weight": 0.5}}
    apply_confidence_consumption(scores, df, params=conf_p)
    assert scores.loc[0, "entry_score"] == pytest.approx(40.0)
    assert scores.loc[0, "golden_cross"] == pytest.approx(5.0)


# ── 配置兜底 ───────────────────────────────────────────────────────────

def test_helpers_have_safe_defaults():
    assert deg.degrade_mode() in (DegradeMode.STRICT, DegradeMode.RELAX, DegradeMode.SKIP)
    assert deg.low_confidence_action() in ("skip", "low_weight")
    assert 0.01 <= deg.low_confidence_weight() <= 1.0


def test_confidence_dataclass():
    c = Confidence.high()
    assert c.level == "high" and not c.is_low
    c2 = Confidence.low(["x"])
    assert c2.is_low and c2.to_dict()["reasons"] == ["x"]
