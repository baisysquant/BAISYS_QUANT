"""P0-7 ②：校准参数写回闭环断裂修复测试。

覆盖：CALIB_PARAM_MAP 与加载逻辑一致（8 键全读全写）、_INT_KEYS 补齐
（buy_threshold/max_holdings 整值落盘）、落盘前类型断言（非整值拒绝写）、
[BACKTEST_CALIBRATED] 覆写读取、apply_calibration_to_config 覆盖全部键。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from UtilsManager.ConfigParser import Config


def _make_ini(base: Path, extra: str = "") -> Path:
    cfg_path = base / "config.ini"
    cfg_path.write_text(
        "[DATABASE]\nUSER = u\nPASSWORD = p\nHOST = localhost\nPORT = 5432\nDB_NAME = db\n\n"
        "[SYSTEM]\nHOME_DIRECTORY = ~/test_baisys\n\n"
        "[LOGGING]\nLOG_LEVEL = DEBUG\n\n"
        "[BACKTEST]\n"
        "ENABLED = true\n"
        "OPTIMIZE_FREQUENCY = monthly\n"
        "BACKTEST_START_DATE = 20200101\n"
        "OUT_OF_SAMPLE_DAYS = 120\n"
        "INITIAL_CASH = 1000000\n\n"
        "[BACKTEST_CALIBRATED]\n"
        "atr_stop_mult = 2.0\n"
        "boll_narrow_ratio = 0.9\n"
        "cross_decay_days = 40\n"
        "conclusion_full_bull = 85\n"
        "golden_cross_bonus = 15\n"
        "divergence_penalty = 25\n"
        "buy_threshold = 17\n"
        "max_holdings = 11\n"
        f"{extra}",
        encoding="utf-8",
    )
    return cfg_path


class TestCalibrationClosure:
    def test_int_keys_cover_buy_threshold_and_max_holdings(self) -> None:
        from BackTrading.calibration import _INT_KEYS

        assert {"buy_threshold", "max_holdings"} <= _INT_KEYS

    def test_calib_param_map_matches_expected_keys(self) -> None:
        from BackTrading.calibration import CALIB_PARAM_MAP

        assert set(CALIB_PARAM_MAP) == {
            "atr_stop_mult", "boll_narrow_ratio", "cross_decay_days",
            "conclusion_full_bull", "golden_cross_bonus", "divergence_penalty",
            "buy_threshold", "max_holdings",
        }

    def test_write_writes_int_keys_as_integers(self, tmp_path: Path, monkeypatch) -> None:
        import BackTrading.calibration as cal

        ini = _make_ini(tmp_path)
        monkeypatch.setattr(cal, "CONFIG_INI", ini)

        cal.write_calibration_to_ini({"buy_threshold": 17.0, "max_holdings": 11.0})

        text = ini.read_text(encoding="utf-8")
        assert "buy_threshold = 17" in text and "buy_threshold = 17.0" not in text
        assert "max_holdings = 11" in text and "max_holdings = 11.0" not in text

    def test_write_raises_on_non_integral_int_key(self, tmp_path: Path, monkeypatch) -> None:
        import BackTrading.calibration as cal

        ini = tmp_path / "config.ini"
        ini.write_text("[BACKTEST_CALIBRATED]\nbuy_threshold = 17\n", encoding="utf-8")
        monkeypatch.setattr(cal, "CONFIG_INI", ini)

        with pytest.raises(ValueError, match="整数参数 buy_threshold"):
            cal.write_calibration_to_ini({"buy_threshold": 17.5, "max_holdings": 11.0})

        # 失败即中止，config.ini 未被污染
        assert "17.5" not in ini.read_text(encoding="utf-8")

    def test_config_parser_overlay_applies_calibrated_keys(self, tmp_path: Path) -> None:
        cfg = Config(config_file=str(_make_ini(tmp_path)))

        assert cfg.app_config.backtest.BUY_THRESHOLD == 17
        assert cfg.app_config.backtest.MAX_HOLDINGS == 11
        # 既有 6 键覆写不受影响
        assert cfg.app_config.scoring_params.ATR_STOP_MULT == 2.0
        assert cfg.app_config.regime_detection.BOLL_NARROW_RATIO == 0.9
        assert cfg.app_config.scoring_params.CROSS_DECAY_DAYS == 40
        assert cfg.app_config.full_bull_scoring.CONCLUSION_FULL_BULL == 85
        assert cfg.app_config.scoring_params.GOLDEN_CROSS_BONUS == 15
        assert cfg.app_config.scoring_params.DIVERGENCE_PENALTY == 25

    def test_overlay_tolerates_legacy_float_format(self, tmp_path: Path) -> None:
        """历史版本以 "17.0" 浮点落盘：加载端容错取整，不崩溃。"""
        cfg = Config(config_file=str(_make_ini(tmp_path, extra="")))

        # 覆写为历史污染格式后重载
        ini = tmp_path / "config.ini"
        text = ini.read_text(encoding="utf-8").replace("buy_threshold = 17", "buy_threshold = 17.0")
        text = text.replace("max_holdings = 11", "max_holdings = 11.0")
        ini.write_text(text, encoding="utf-8")

        cfg2 = Config(config_file=str(ini))
        assert cfg2.app_config.backtest.BUY_THRESHOLD == 17
        assert cfg2.app_config.backtest.MAX_HOLDINGS == 11

    def test_apply_calibration_to_config_covers_all_map_keys(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import BackTrading.calibration as cal
        from BackTrading.calibration import (
            CalibrationResult,
            apply_calibration_to_config,
            save_calibration,
        )

        ini = _make_ini(tmp_path)
        cfg = Config(config_file=str(ini))
        monkeypatch.setattr(cal, "CALIBRATION_FILE", tmp_path / "calibration_result.json")

        params = {
            "atr_stop_mult": 2.5,
            "boll_narrow_ratio": 0.7,
            "cross_decay_days": 45,
            "conclusion_full_bull": 88,
            "golden_cross_bonus": 12,
            "divergence_penalty": 30,
            "buy_threshold": 19.0,
            "max_holdings": 8.0,
        }
        save_calibration(CalibrationResult(params=params))
        apply_calibration_to_config(cfg)

        sc = cfg.app_config.scoring_params
        assert sc.ATR_STOP_MULT == 2.5
        assert cfg.app_config.regime_detection.BOLL_NARROW_RATIO == 0.7
        assert sc.CROSS_DECAY_DAYS == 45
        assert cfg.app_config.full_bull_scoring.CONCLUSION_FULL_BULL == 88
        assert sc.GOLDEN_CROSS_BONUS == 12
        assert sc.DIVERGENCE_PENALTY == 30
        assert cfg.app_config.backtest.BUY_THRESHOLD == 19
        assert cfg.app_config.backtest.MAX_HOLDINGS == 8

    def test_engine_config_fallback_uses_calibrated_values(self, tmp_path: Path) -> None:
        """日频路径 EngineConfig 兜底读 bt.BUY_THRESHOLD / bt.MAX_HOLDINGS。"""
        cfg = Config(config_file=str(_make_ini(tmp_path)))
        bt = cfg.app_config.backtest

        from BackTrading.engine import EngineConfig

        ecfg = EngineConfig(
            buy_threshold=int(bt.BUY_THRESHOLD),
            max_holdings=int(bt.MAX_HOLDINGS),
        )
        assert ecfg.buy_threshold == 17
        assert ecfg.max_holdings == 11
        assert isinstance(ecfg.buy_threshold, int)
        assert isinstance(ecfg.max_holdings, int)