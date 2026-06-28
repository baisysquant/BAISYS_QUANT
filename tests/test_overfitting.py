from __future__ import annotations

import math

import numpy as np
import pytest

from Backtesting.overfitting import (
    compute_pbo,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    compute_dsr_from_equity_curve,
)


class TestProbabilisticSharpeRatio:
    def test_psr_nominal(self):
        sr = probabilistic_sharpe_ratio(1.0, 252)
        assert 0.5 < sr < 1.0

    def test_psr_zero_returns_0p5(self):
        sr = probabilistic_sharpe_ratio(0.0, 252)
        assert abs(sr - 0.5) < 0.01

    def test_psr_negative(self):
        sr = probabilistic_sharpe_ratio(-0.5, 252)
        assert sr < 0.5

    def test_psr_short_series(self):
        sr = probabilistic_sharpe_ratio(1.0, 1)
        assert sr == 0.5

    def test_psr_high_sharpe(self):
        sr = probabilistic_sharpe_ratio(5.0, 252)
        assert sr > 0.90

    def test_psr_with_target(self):
        sr = probabilistic_sharpe_ratio(0.5, 252, target_sr=1.0)
        assert sr < 0.5

    def test_psr_with_skew(self):
        sr_skew = probabilistic_sharpe_ratio(1.0, 252, skew=-1.0)
        sr_base = probabilistic_sharpe_ratio(1.0, 252)
        assert sr_skew < sr_base


class TestDeflatedSharpeRatio:
    def test_dsr_single_trial(self):
        dsr = deflated_sharpe_ratio(1.0, 252, num_trials=1)
        psr = probabilistic_sharpe_ratio(1.0, 252)
        assert abs(dsr - psr) < 0.001

    def test_dsr_many_trials(self):
        dsr_1 = deflated_sharpe_ratio(1.0, 252, num_trials=1)
        dsr_100 = deflated_sharpe_ratio(1.0, 252, num_trials=100)
        assert dsr_100 < dsr_1

    def test_dsr_very_many_trials(self):
        dsr = deflated_sharpe_ratio(1.0, 252, num_trials=1000)
        assert 0 < dsr < 1

    def test_dsr_high_sharpe_survives(self):
        dsr = deflated_sharpe_ratio(3.0, 252 * 3, num_trials=100)
        assert dsr > 0.90

    def test_dsr_many_trials_low_sharpe(self):
        dsr = deflated_sharpe_ratio(0.2, 63, num_trials=1000)
        assert dsr < 0.5


class TestComputePBO:
    def test_pbo_no_overfitting(self):
        windows = []
        for w in range(5):
            ocs = [
                {"is_rank": 1, "oos_sharpe": 2.0},
                {"is_rank": 2, "oos_sharpe": 1.5},
                {"is_rank": 3, "oos_sharpe": 1.0},
            ]
            windows.append({"oos_combos": ocs})
        pbo = compute_pbo(windows)
        assert pbo == 0.0

    def test_pbo_all_overfitting(self):
        windows = []
        for w in range(5):
            ocs = [
                {"is_rank": 1, "oos_sharpe": -1.0},
                {"is_rank": 2, "oos_sharpe": 1.0},
                {"is_rank": 3, "oos_sharpe": 0.5},
            ]
            windows.append({"oos_combos": ocs})
        pbo = compute_pbo(windows)
        assert pbo == 1.0

    def test_pbo_partial(self):
        windows = [
            {"oos_combos": [
                {"is_rank": 1, "oos_sharpe": 2.0},
                {"is_rank": 2, "oos_sharpe": 1.0},
                {"is_rank": 3, "oos_sharpe": 0.5},
            ]},
            {"oos_combos": [
                {"is_rank": 1, "oos_sharpe": 0.3},
                {"is_rank": 2, "oos_sharpe": 1.5},
                {"is_rank": 3, "oos_sharpe": 0.8},
            ]},
        ]
        pbo = compute_pbo(windows)
        assert pbo == 0.5

    def test_pbo_empty_windows(self):
        pbo = compute_pbo([])
        assert pbo == 0.5

    def test_pbo_nan_handling(self):
        windows = [{
            "oos_combos": [
                {"is_rank": 1, "oos_sharpe": None},
                {"is_rank": 2, "oos_sharpe": 1.0},
            ]
        }]
        pbo = compute_pbo(windows, top_m=2)
        assert pbo == 0.5

    def test_pbo_single_combo(self):
        windows = [{
            "oos_combos": [
                {"is_rank": 1, "oos_sharpe": 0.5},
            ]
        }]
        pbo = compute_pbo(windows)
        assert pbo == 0.5


class TestComputeDsrFromEquityCurve:
    def test_dsr_positive(self):
        vals = [1.0]
        for i in range(1, 253):
            vals.append(vals[-1] * (1 + np.random.normal(0.001, 0.02)))
        ec = [{"portfolio_value": v} for v in vals]
        dsr = compute_dsr_from_equity_curve(ec, num_trials=10)
        assert 0 < dsr < 1

    def test_dsr_short_curve(self):
        dsr = compute_dsr_from_equity_curve([{"portfolio_value": 100}], num_trials=10)
        assert dsr == 0.5

    def test_dsr_when_flat(self):
        ec = [{"portfolio_value": 100.0}] * 10
        dsr = compute_dsr_from_equity_curve(ec, num_trials=1)
        assert dsr == 0.5
