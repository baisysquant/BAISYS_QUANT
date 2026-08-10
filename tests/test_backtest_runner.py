from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from BackTrading.alert import BacktestAlert
from BackTrading.calibration import CalibrationResult
from BackTrading.calibration_log import should_rerun
from UtilsManager.ConfigParser import Config


class TestKlineDataVersion:
    def test_returns_version_with_max_date_and_hash(self) -> None:
        from BackTrading.runner import _compute_kline_data_version

        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = ("2025-06-30", 1234)
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn

        version = _compute_kline_data_version(engine)
        assert version.startswith("2025-06-30_")
        assert len(version) > len("2025-06-30_")

    def test_changes_when_row_count_changes(self) -> None:
        from BackTrading.runner import _compute_kline_data_version

        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = ("2025-06-30", 1234)
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        v1 = _compute_kline_data_version(engine)

        conn2 = MagicMock()
        conn2.execute.return_value.fetchone.return_value = ("2025-06-30", 5678)
        engine2 = MagicMock()
        engine2.connect.return_value.__enter__.return_value = conn2
        v2 = _compute_kline_data_version(engine2)
        assert v1 != v2

    def test_changes_when_max_date_changes(self) -> None:
        from BackTrading.runner import _compute_kline_data_version

        def _mk(max_date: str):
            conn = MagicMock()
            conn.execute.return_value.fetchone.return_value = (max_date, 1234)
            engine = MagicMock()
            engine.connect.return_value.__enter__.return_value = conn
            return _compute_kline_data_version(engine)

        assert _mk("2025-06-30") != _mk("2025-07-01")

    def test_empty_table_returns_empty_string(self) -> None:
        from BackTrading.runner import _compute_kline_data_version

        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (None, 0)
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        assert _compute_kline_data_version(engine) == ""

    def test_db_error_returns_empty_string(self) -> None:
        from BackTrading.runner import _compute_kline_data_version

        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("db down")
        assert _compute_kline_data_version(engine) == ""


class TestBacktestLockHeld:
    def _engine_with_lock(self, acquired: bool) -> MagicMock:
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = acquired
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        return engine

    def test_lock_held_when_try_fails(self) -> None:
        from UtilsManager.IDataProvider import backtest_lock_held

        assert backtest_lock_held(self._engine_with_lock(False)) is True

    def test_lock_free_when_try_succeeds(self) -> None:
        from UtilsManager.IDataProvider import backtest_lock_held

        engine = self._engine_with_lock(True)
        assert backtest_lock_held(engine) is False
        # 探测后必须释放，避免自身会话持有锁
        conn = engine.connect.return_value.__enter__.return_value
        unlock_calls = [c for c in conn.execute.call_args_list
                        if "pg_advisory_unlock" in str(c.args[0])]
        assert len(unlock_calls) == 1

    def test_error_returns_false(self) -> None:
        from UtilsManager.IDataProvider import backtest_lock_held

        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("db down")
        assert backtest_lock_held(engine) is False

    def test_shared_key_with_runner(self) -> None:
        # runner 持有的会话锁 key 与 IDataProvider 探测的 key 必须一致
        import BackTrading.runner as runner_mod
        from UtilsManager import IDataProvider

        assert runner_mod._BACKTEST_LOCK_KEY == IDataProvider.BACKTEST_ADVISORY_LOCK_KEY



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

    def test_data_version_change_forces_rerun(self) -> None:
        last = {"run_time": "2025-01-15T00:00:00", "data_version": "2025-01-14_abc123"}
        should, reason = should_rerun(
            last, "monthly", today=date(2025, 1, 25), data_version="2025-01-15_def456"
        )
        assert should
        assert "数据版本变化" in reason

    def test_data_version_same_no_rerun(self) -> None:
        last = {"run_time": "2025-01-15T00:00:00", "data_version": "2025-01-14_abc123"}
        should, _ = should_rerun(
            last, "monthly", today=date(2025, 1, 25), data_version="2025-01-14_abc123"
        )
        assert not should

    def test_config_hash_change_forces_rerun(self) -> None:
        last = {"run_time": "2025-01-15T00:00:00", "config_hash": "oldhash"}
        should, reason = should_rerun(
            last, "monthly", today=date(2025, 1, 25), config_hash="newhash"
        )
        assert should
        assert "配置哈希变化" in reason

    def test_old_record_without_data_version_no_force(self) -> None:
        # 历史记录无 data_version 列 → 不因版本比较强制重跑（向前兼容）
        last = {"run_time": "2025-01-15T00:00:00"}
        should, _ = should_rerun(
            last, "monthly", today=date(2025, 1, 25), data_version="2025-01-15_def456"
        )
        assert not should


class TestExtractBestParams:
    def _make_wf(self, rows: list[dict]) -> pytest.DataFrame:
        import pandas as pd
        return pd.DataFrame(rows)

    def test_median_path_preferred_over_weighted(self) -> None:
        import pandas as pd
        wf = pd.DataFrame([
            {"window": 0, "params": {"a": 1.0}, "sharpe_ratio": 1.0,
             "dm_p_value": 0.01, "train_sharpe": 1.2, "num_combos": 10},
            {"window": 1, "params": {"a": 3.0}, "sharpe_ratio": 2.0,
             "dm_p_value": 0.01, "train_sharpe": 2.5, "num_combos": 10},
            {"window": 2, "params": {"a": 9.0}, "sharpe_ratio": 3.0,
             "dm_p_value": 0.01, "train_sharpe": 3.5, "num_combos": 10},
            {"window": 3, "params": {"a": 2.0}, "sharpe_ratio": 0.5,
             "dm_p_value": 0.30, "train_sharpe": 1.0, "num_combos": 10},
        ])
        from BackTrading.runner import _extract_best_params
        result = _extract_best_params(wf)
        # DM 显著窗口 (0,1,2) 的中位数 = 3.0（Sharpe 加权会偏向 9）
        assert result["a"] == 3.0

    def test_median_fallback_when_no_dm_column(self) -> None:
        import pandas as pd
        wf = pd.DataFrame([
            {"window": 0, "params": {"a": 1.0}, "sharpe_ratio": 1.0,
             "train_sharpe": 1.2, "num_combos": 10},
            {"window": 1, "params": {"a": 2.0}, "sharpe_ratio": 2.0,
             "train_sharpe": 2.5, "num_combos": 10},
            {"window": 2, "params": {"a": 5.0}, "sharpe_ratio": 1.5,
             "train_sharpe": 1.8, "num_combos": 10},
            {"window": 3, "params": {"a": 8.0}, "sharpe_ratio": -0.5,
             "train_sharpe": -1.0, "num_combos": 10},
        ])
        from BackTrading.runner import _extract_best_params
        result = _extract_best_params(wf)
        # OOS>0 窗口 (0,1,2) 中位数 = 2.0
        assert result["a"] == 2.0

    def test_weighted_fallback_when_few_windows(self) -> None:
        import pandas as pd
        wf = pd.DataFrame([
            {"window": 0, "params": {"a": 1.0, "b": 10.0}, "sharpe_ratio": 1.0,
             "train_sharpe": 1.2, "num_combos": 10},
        ])
        from BackTrading.runner import _extract_best_params
        result = _extract_best_params(wf)
        assert result["a"] == 1.0 and result["b"] == 10.0

    def test_empty_result_uses_config_midpoint_fallback(self) -> None:
        import pandas as pd
        wf = pd.DataFrame()
        from BackTrading.runner import _extract_best_params
        result = _extract_best_params(wf)
        assert "atr_stop_mult" in result

    def test_median_robust_to_outlier_window(self) -> None:
        import pandas as pd
        wf = pd.DataFrame([
            {"window": 0, "params": {"a": 10.0}, "sharpe_ratio": 1.0,
             "dm_p_value": 0.01, "train_sharpe": 1.2, "num_combos": 10},
            {"window": 1, "params": {"a": 12.0}, "sharpe_ratio": 1.1,
             "dm_p_value": 0.02, "train_sharpe": 1.3, "num_combos": 10},
            {"window": 2, "params": {"a": 500.0}, "sharpe_ratio": 5.0,
             "dm_p_value": 0.04, "train_sharpe": 6.0, "num_combos": 10},
        ])
        from BackTrading.runner import _extract_best_params
        result = _extract_best_params(wf)
        # 中位数 = 12，不被 500 拖偏
        assert result["a"] == 12.0

    def test_fallback_frame_columns_safe(self) -> None:
        # 兜底帧（无有效窗口时 meta_optimizer 返回）只有核心列，
        # runner 的绩效平均逻辑必须容忍缺失的 total_return / max_drawdown
        import pandas as pd
        wf = pd.DataFrame([{
            "window": 0, "params": {"atr_stop_mult": 2.0}, "sharpe_ratio": 0.0,
            "num_combos": 1, "num_paths": 3,
        }])
        from BackTrading.runner import _extract_best_params
        result = _extract_best_params(wf)
        # 无 DM 列、OOS=0 无正收益窗口 → 退回配置中位数兜底（不应抛异常）
        assert isinstance(result, dict) and len(result) > 0
        # 模拟 runner.py 447 行的防御逻辑
        top = wf.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False).head(5)
        total_return_avg = float(top["total_return"].mean()) if "total_return" in top.columns and not top.empty else 0.0
        max_dd_avg = float(top["max_drawdown"].mean()) if "max_drawdown" in top.columns and not top.empty else 0.0
        assert total_return_avg == 0.0
        assert max_dd_avg == 0.0


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
        from BackTrading.calibration import save_calibration

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
        from BackTrading.calibration import save_calibration

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
        from BackTrading.calibration import write_calibration_to_ini, CONFIG_INI
        import importlib

        ini_content = dedent("""\
            [BACKTEST_CALIBRATED]
            # ATR 止损倍数
            atr_stop_mult = 1.5
            # 金叉衰减半衰期
            cross_decay_days = 30
            # 半凯利系数
            kelly_fraction = 0.25
            # 流动性否决阈值
            liq_veto_ratio = 0.05
            # A 级评分阈值
            conclusion_full_bull = 80
            # R04 金叉加分
            golden_cross_bonus = 10
            # R41 顶背离扣分
            divergence_penalty = 20
            # NONE 风险仓位系数
            risk_none_multiplier = 1.0
        """)
        original = CONFIG_INI.read_text(encoding="utf-8") if CONFIG_INI.exists() else None

        try:
            CONFIG_INI.write_text(ini_content, encoding="utf-8")
            importlib.reload(__import__("BackTrading.calibration"))

            params = {
                "atr_stop_mult": 2.0,
                "cross_decay_days": 20,
                "kelly_fraction": 0.15,
                "liq_veto_ratio": 0.03,
                "conclusion_full_bull": 85,
                "golden_cross_bonus": 15,
                "divergence_penalty": 25,
                "risk_none_multiplier": 1.5,
            }
            write_calibration_to_ini(params)

            updated = CONFIG_INI.read_text(encoding="utf-8")
            assert "atr_stop_mult = 2" in updated or "atr_stop_mult = 2.0" in updated
            assert "cross_decay_days = 20" in updated
            assert "kelly_fraction = 0.15" in updated
            assert "liq_veto_ratio = 0.03" in updated
            assert "conclusion_full_bull = 85" in updated
            assert "golden_cross_bonus = 15" in updated
            assert "divergence_penalty = 25" in updated
            assert "risk_none_multiplier = 1.5" in updated
        finally:
            if original:
                CONFIG_INI.write_text(original, encoding="utf-8")
            elif CONFIG_INI.exists():
                CONFIG_INI.unlink()
