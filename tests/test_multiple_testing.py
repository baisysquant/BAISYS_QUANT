"""多重测试惩罚 & 参数稳健性检查单元测试"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from BackTrading.calibration_log import (
    apply_multiple_testing_penalty,
    MAX_TUNING_ATTEMPTS,
    MULTIPLE_TESTING_PENALTY,
)


class TestMultipleTestingPenalty:
    def test_no_penalty_within_limit(self):
        """尝试次数 ≤ 10 时不惩罚。"""
        s, so, level = apply_multiple_testing_penalty(
            1.5, 2.0, 5, "20230101", 60,
        )
        assert s == 1.5
        assert so == 2.0
        assert level == "INFO"

    def test_no_penalty_at_limit(self):
        """恰好在 10 次时不惩罚。"""
        s, so, level = apply_multiple_testing_penalty(
            1.5, 2.0, MAX_TUNING_ATTEMPTS, "20230101", 60,
        )
        assert s == 1.5
        assert so == 2.0
        assert level == "INFO"

    def test_penalty_exceeds_limit(self):
        """超过 10 次时 Sharpe 和 Sortino 各扣减 20%。"""
        s, so, level = apply_multiple_testing_penalty(
            1.5, 2.0, 11, "20230101", 60,
        )
        expected_sharpe = 1.5 * (1.0 - MULTIPLE_TESTING_PENALTY)
        expected_sortino = 2.0 * (1.0 - MULTIPLE_TESTING_PENALTY)
        assert abs(s - expected_sharpe) < 1e-6
        assert abs(so - expected_sortino) < 1e-6
        assert level == "WARNING"

    def test_critical_at_31_attempts(self):
        """超过 30 次标记为 CRITICAL。"""
        s, so, level = apply_multiple_testing_penalty(
            2.0, 3.0, 31, "20230101", 60,
        )
        assert abs(s - 2.0 * (1.0 - MULTIPLE_TESTING_PENALTY)) < 1e-6
        assert level == "CRITICAL"

    def test_penalty_on_zero_sharpe(self):
        """Sharpe = 0 时惩罚后仍为 0。"""
        s, so, level = apply_multiple_testing_penalty(
            0.0, 0.0, 15, "20230101", 60,
        )
        assert s == 0.0
        assert so == 0.0

    def test_penalty_on_negative_sharpe(self):
        """负 Sharpe 也惩罚（变得更负）。"""
        s, so, level = apply_multiple_testing_penalty(
            -1.0, -1.5, 12, "20230101", 60,
        )
        assert abs(s - -1.0 * (1.0 - MULTIPLE_TESTING_PENALTY)) < 1e-6
        assert level == "WARNING"


class TestMultipleTestingConstants:
    def test_constants_defined(self):
        """确保常量合理。"""
        assert MAX_TUNING_ATTEMPTS == 10
        assert 0 < MULTIPLE_TESTING_PENALTY <= 0.5
        assert MULTIPLE_TESTING_PENALTY == 0.20
