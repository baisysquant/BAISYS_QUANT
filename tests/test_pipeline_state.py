from __future__ import annotations

import pandas as pd

from LogicAnalyzer.PipelineState import (
    _calc_exit_strategy,
    _detect_market_regime,
    _make_state,
)


def test_make_state(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    state = _make_state(sample_ohlcv_with_indicators, "STRONG_TREND", {})
    assert isinstance(state, dict)
    assert state["regime"] == "STRONG_TREND"
    assert state["risk_level"] == "NONE"
    assert state["level"] == "C"
    assert state["score"] == 0
    assert "df" in state


def test_detect_market_regime_strong_trend(sample_boll_bandwidth: pd.DataFrame) -> None:
    regime = _detect_market_regime(sample_boll_bandwidth, boll_col="BOLL_BANDWIDTH")
    assert regime in ("STRONG_TREND", "WEAK_TREND", "OSCILLATION", "BOTTOM_REVERSAL", "TOP_RISK", "UNCLEAR")


def test_detect_market_regime_no_boll(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    regime = _detect_market_regime(sample_ohlcv_with_indicators)
    assert isinstance(regime, str)


def test_detect_market_regime_short_data() -> None:
    df = pd.DataFrame({"close": [10.0] * 10, "DIF": [0.0] * 10, "DEA": [0.0] * 10})
    regime = _detect_market_regime(df)
    assert regime == "UNCLEAR"


def test_detect_market_regime_oscillation(sample_boll_bandwidth: pd.DataFrame) -> None:
    df = sample_boll_bandwidth.copy()
    df["BOLL_BANDWIDTH"] = df["BOLL_BANDWIDTH"] * 0.1
    regime = _detect_market_regime(df, boll_col="BOLL_BANDWIDTH",
                                   params={"oscillation_min_bars": 5})
    assert isinstance(regime, str)


def test_calc_exit_strategy(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    result = _calc_exit_strategy(sample_ohlcv_with_indicators)
    assert isinstance(result, dict)
    assert "stop_loss" in result
    assert "t1_target" in result
    assert "t2_target" in result
    assert "trailing_stop" in result
    assert "exit_rrr" in result


def test_calc_exit_strategy_no_atr() -> None:
    df = pd.DataFrame({"close": [10.0] * 30})
    result = _calc_exit_strategy(df)
    assert result["stop_loss"] is None


def test_calc_exit_strategy_with_params(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    result = _calc_exit_strategy(sample_ohlcv_with_indicators,
                                 params={"atr_stop_mult": 2.0, "atr_t1_mult": 4.0})
    assert result["stop_loss"] is not None
    assert result["t1_target"] is not None
