from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from textwrap import dedent

import pytest

from Backtesting.alert import BacktestAlert
from Backtesting.calibration import CalibrationResult
from Backtesting.calibration_log import should_rerun
from ConfigParser import Config


class TestShouldRun:
    def test_initial_returns_false(self) -> None:
        should, _ = should_rerun({"run_time": "2025-01-01T00:00:00"}, "initial", today=date(2025, 6, 1))
        assert not should

    def test_monthly_same_month(self) -> None:
        should, _ = should_rerun({"run_time": "2025-01-15T00:00:00"}, "monthly", today=date(2025, 1, 25))
        assert not should

    def test_monthly_different_month(self) -> None:
        should, _ = should_rerun({"run_time": "2025-01-01T00:00:00"}, "monthly", today=date(2025, 2, 1))
        assert should

    def test_quarterly_same_quarter(self) -> None:
        should, _ = should_rerun({"run_time": "2025-02-01T00:00:00"}, "quarterly", today=date(2025, 3, 15))
        assert not should

    def test_quarterly_new_quarter(self) -> None:
        should, _ = should_rerun({"run_time": "2025-03-15T00:00:00"}, "quarterly", today=date(2025, 4, 1))
        assert should

    def test_quarterly_new_year(self) -> None:
        should, _ = should_rerun({"run_time": "2025-12-01T00:00:00"}, "quarterly", today=date(2026, 1, 15))
        assert should


class TestBacktestAlert:
    @pytest.fixture
    def alert(self) -> BacktestAlert:
        return BacktestAlert(Config())

    def test_on_success_logs(self, alert: BacktestAlert) -> None:
        result = CalibrationResult(params={"atr_stop_mult": 1.5}, sharpe=0.8, total_return=0.1)
        alert.on_success(result)  # 不应抛出异常

    def test_on_failure_logs(self, alert: BacktestAlert) -> None:
        alert.on_failure(ValueError("test error"))  # 不应抛出异常

    def test_drift_detection(self, alert: BacktestAlert, tmp_path: Path) -> None:
        from Backtesting.calibration import save_calibration

        old = CalibrationResult(params={"atr_stop_mult": 1.0}, sharpe=0.5)
        save_calibration(old)

        new_params = {"atr_stop_mult": 2.0}  # 100% 变化 > 15% 阈值
        alert._check_drift(new_params)

        assert alert.DRIFT_LOG.exists()
        records = json.loads(alert.DRIFT_LOG.read_text(encoding="utf-8"))
        assert any(r.get("type") == "drift" for r in records)

        # 清理
        if alert.DRIFT_LOG.exists():
            alert.DRIFT_LOG.unlink()

    def test_no_drift(self, alert: BacktestAlert) -> None:
        from Backtesting.calibration import save_calibration

        old = CalibrationResult(params={"atr_stop_mult": 1.0})
        save_calibration(old)

        new_params = {"atr_stop_mult": 1.05}  # 5% < 15%
        alert._check_drift(new_params)

        records = []
        if alert.DRIFT_LOG.exists():
            records = json.loads(alert.DRIFT_LOG.read_text(encoding="utf-8"))
        drifts = [r for r in records if r.get("type") == "drift"]
        assert len(drifts) == 0
        if alert.DRIFT_LOG.exists():
            alert.DRIFT_LOG.unlink()


class TestWriteCalibrationToIni:
    def test_updates_config_ini_values(self, tmp_path: Path) -> None:
        from Backtesting.calibration import write_calibration_to_ini, CONFIG_INI
        import importlib

        ini_content = dedent("""\
            [SCORING_PARAMS]
            # ATR 止损倍数
            atr_stop_mult = 1.5
            # 金叉衰减半衰期
            cross_decay_days = 30

            [POSITION_SIZING]
            # 半凯利系数
            kelly_fraction = 0.25
            position_a = 0.30

            [FILTER_RULES]
            # 流动性否决阈值
            liq_veto_ratio = 0.05
        """)
        original = CONFIG_INI.read_text(encoding="utf-8") if CONFIG_INI.exists() else None

        try:
            CONFIG_INI.write_text(ini_content, encoding="utf-8")
            importlib.reload(__import__("Backtesting.calibration"))

            params = {
                "atr_stop_mult": 2.0,
                "cross_decay_days": 20,
                "kelly_fraction": 0.15,
                "liq_veto_ratio": 0.03,
            }
            write_calibration_to_ini(params)

            updated = CONFIG_INI.read_text(encoding="utf-8")
            assert "atr_stop_mult = 2" in updated or "atr_stop_mult = 2.0" in updated
            assert "cross_decay_days = 20" in updated
            assert "kelly_fraction = 0.15" in updated
            assert "liq_veto_ratio = 0.03" in updated
            assert "ATR 止损倍数" in updated
        finally:
            if original:
                CONFIG_INI.write_text(original, encoding="utf-8")
            elif CONFIG_INI.exists():
                CONFIG_INI.unlink()
