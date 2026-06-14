from __future__ import annotations

import numpy as np
import pandas as pd

from LogicAnalyzer.MACDDivergence import (
    adaptive_distance,
    find_peaks_troughs,
    slope_analysis,
)


def test_find_peaks_troughs() -> None:
    series = pd.Series([1, 3, 2, 4, 3, 5, 4, 6, 5, 4, 3, 2, 1])
    peaks, troughs = find_peaks_troughs(series, distance=2)
    assert isinstance(peaks, np.ndarray)
    assert isinstance(troughs, np.ndarray)


def test_find_peaks_troughs_monotonic() -> None:
    series = pd.Series(np.linspace(1, 10, 20))
    peaks, troughs = find_peaks_troughs(series, distance=3)
    assert len(peaks) >= 0
    assert len(troughs) >= 0


def test_adaptive_distance_low_volatility() -> None:
    series = pd.Series([10.0] * 30 + [10.5, 10.3, 10.6, 10.2, 10.7])
    dist = adaptive_distance(series, base_distance=10)
    assert isinstance(dist, int)
    assert dist > 0


def test_adaptive_distance_high_volatility() -> None:
    np.random.seed(42)
    series = pd.Series(10 + np.random.randn(50) * 5)
    dist = adaptive_distance(series, base_distance=10)
    assert isinstance(dist, int)
    assert dist > 0


def test_adaptive_distance_short_series() -> None:
    series = pd.Series([1, 2, 3])
    dist = adaptive_distance(series, base_distance=10)
    assert dist == 3


def test_slope_analysis_rising() -> None:
    series = pd.Series(np.linspace(1, 10, 30))
    result = slope_analysis(series, window=5)
    assert result["slope"] > 0
    assert result["r2"] > 0.5
    assert result["trend"] == "明确上行"


def test_slope_analysis_falling() -> None:
    series = pd.Series(np.linspace(10, 1, 30))
    result = slope_analysis(series, window=5)
    assert result["slope"] < 0
    assert result["trend"] == "明确下行"


def test_slope_analysis_flat() -> None:
    series = pd.Series([5.0] * 30 + [5.1, 4.9, 5.0, 5.05, 4.95])
    result = slope_analysis(series, window=5)
    assert isinstance(result["slope"], float)
    assert isinstance(result["trend"], str)


def test_slope_analysis_short_data() -> None:
    series = pd.Series([1, 2])
    result = slope_analysis(series, window=5)
    assert result["slope"] == 0.0
    assert result["r2"] == 0.0
