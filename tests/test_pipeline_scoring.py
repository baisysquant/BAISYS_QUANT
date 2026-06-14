from __future__ import annotations

import pandas as pd

from LogicAnalyzer.PipelineScoring import (
    _calc_momentum_desc,
    _calc_moneyflow_score,
    _score_kline_pattern,
    _volume_price_trend_score,
)


def test_calc_momentum_desc_bull(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    df = sample_ohlcv_with_indicators.copy()
    df["DIF"] = df["DIF"] + 0.5
    desc, score = _calc_momentum_desc(df, dif_col="DIF", dea_col="DEA")
    assert isinstance(desc, str)
    assert isinstance(score, int)
    assert 0 <= score <= 20


def test_calc_momentum_desc_bear(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    df = sample_ohlcv_with_indicators.copy()
    df["DIF"] = df["DIF"] - 0.5
    desc, score = _calc_momentum_desc(df, dif_col="DIF", dea_col="DEA")
    assert isinstance(desc, str)
    assert isinstance(score, int)


def test_calc_momentum_desc_missing_cols() -> None:
    import pandas as pd
    df = pd.DataFrame({"close": [10.0] * 30})
    import pytest
    with pytest.raises(KeyError, match="DIF"):
        _calc_momentum_desc(df, dif_col="DIF", dea_col="DEA")


def test_volume_price_trend_score(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    df = sample_ohlcv_with_indicators.copy()
    desc, score = _volume_price_trend_score(df, lookback=5)
    assert isinstance(desc, str)
    assert isinstance(score, int)
    assert 0 <= score <= 10


def test_volume_price_trend_score_short_data() -> None:
    df = pd.DataFrame({"close": [10] * 3, "volume": [100] * 3})
    desc, score = _volume_price_trend_score(df, lookback=5)
    assert isinstance(desc, str)
    assert isinstance(score, int)


def test_score_kline_pattern(sample_ohlcv_with_indicators: pd.DataFrame) -> None:
    desc, score = _score_kline_pattern(sample_ohlcv_with_indicators, max_score=10)
    assert isinstance(desc, str)
    assert isinstance(score, int)
    assert 0 <= score <= 10


def test_score_kline_pattern_short_data() -> None:
    df = pd.DataFrame({"close": [10] * 2, "high": [11] * 2, "low": [9] * 2, "open": [10] * 2})
    desc, score = _score_kline_pattern(df, max_score=10)
    assert isinstance(desc, str)


def test_calc_moneyflow_score_none() -> None:
    score, desc = _calc_moneyflow_score(None)
    assert score == 0
    assert isinstance(desc, str)


def test_calc_moneyflow_score_neutral() -> None:
    mf = {"净流入额": 0, "main_net_amount_5d": 0, "main_net_amount_10d": 0}
    score, desc = _calc_moneyflow_score(mf)
    assert isinstance(score, int)
    assert isinstance(desc, str)


def test_calc_moneyflow_score_positive() -> None:
    mf = {"净流入额": 20_000_000, "main_net_amount_5d": 10_000_000, "main_net_amount_10d": 5_000_000}
    score, desc = _calc_moneyflow_score(mf)
    assert isinstance(score, int)
    assert isinstance(desc, str)
