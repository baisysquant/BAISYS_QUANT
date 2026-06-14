from __future__ import annotations

from datetime import date, datetime


from Backtesting.calibration_log import should_rerun


class TestShouldRerun:
    def test_no_last_run(self) -> None:
        should, _ = should_rerun(None, "monthly")
        assert should is True

    def test_initial_frequency(self) -> None:
        last = {"run_time": datetime(2025, 1, 1)}
        should, _ = should_rerun(last, "initial")
        assert should is False

    def test_monthly_same_month(self) -> None:
        last = {"run_time": datetime(2025, 6, 15)}
        should, _ = should_rerun(last, "monthly", today=date(2025, 6, 20))
        assert should is False

    def test_monthly_different_month(self) -> None:
        last = {"run_time": datetime(2025, 5, 1)}
        should, _ = should_rerun(last, "monthly", today=date(2025, 6, 1))
        assert should is True

    def test_monthly_new_year(self) -> None:
        last = {"run_time": datetime(2025, 12, 1)}
        should, _ = should_rerun(last, "monthly", today=date(2026, 1, 1))
        assert should is True

    def test_quarterly_same_quarter(self) -> None:
        last = {"run_time": datetime(2025, 2, 15)}
        should, _ = should_rerun(last, "quarterly", today=date(2025, 3, 20))
        assert should is False

    def test_quarterly_new_quarter(self) -> None:
        last = {"run_time": datetime(2025, 3, 1)}
        should, _ = should_rerun(last, "quarterly", today=date(2025, 4, 1))
        assert should is True

    def test_quarterly_new_year(self) -> None:
        last = {"run_time": datetime(2025, 12, 1)}
        should, _ = should_rerun(last, "quarterly", today=date(2026, 1, 15))
        assert should is True
