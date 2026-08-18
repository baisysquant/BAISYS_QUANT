from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from UtilsManager.ConfigParser import Config
from UtilsManager.ConfigCipher import ConfigCipher


class TestStripInline:
    @pytest.mark.unit
    def test_strips_hash_comment_and_trailing_space(self):
        assert Config._strip_inline("12,26,9  # MACD fast") == "12,26,9"

    @pytest.mark.unit
    def test_strips_semicolon_comment_and_trailing_space(self):
        assert Config._strip_inline("true  ; enable flag") == "true"

    @pytest.mark.unit
    def test_no_comment_unchanged(self):
        assert Config._strip_inline("some_value") == "some_value"

    @pytest.mark.unit
    def test_empty_string(self):
        assert Config._strip_inline("") == ""


class TestParseAliases:
    @pytest.mark.unit
    def test_parses_single_alias(self):
        from UtilsManager.ConfigParser import parse_aliases
        result = parse_aliases("代码=ts_code")
        assert result == {"代码": "ts_code"}

    @pytest.mark.unit
    def test_parses_multi_alias(self):
        from UtilsManager.ConfigParser import parse_aliases
        result = parse_aliases("代码=ts_code,名称=name")
        assert result == {"代码": "ts_code", "名称": "name"}

    @pytest.mark.unit
    def test_empty_string(self):
        from UtilsManager.ConfigParser import parse_aliases
        assert parse_aliases("") == {}


class TestConfigInit:
    @pytest.mark.unit
    def test_loads_from_temp_config(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.DB_USER == "test_user"
        assert cfg.DB_HOST == "localhost"
        assert cfg.BACKTEST_START_DATE == "20200101"
        assert cfg.OUT_OF_SAMPLE_DAYS == 20

    @pytest.mark.unit
    def test_backtest_calibrated_overrides(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        scoring = cfg.app_config.scoring_params
        assert scoring.ATR_STOP_MULT == 1.5
        assert scoring.CROSS_DECAY_DAYS == 30
        # atr_t1_mult / atr_t2_mult 已从配置中移除（死键清理），下游采用 .get() 硬编码默认值降级

    @pytest.mark.unit
    def test_type_conversion_bool_on_filter(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.ENABLE_WEAK_STOCK_FILTER is True

    @pytest.mark.unit
    def test_type_conversion_int(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.OUT_OF_SAMPLE_DAYS == 20

    @pytest.mark.unit
    def test_type_conversion_float_on_backtest(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.app_config.backtest.INITIAL_CASH == 1_000_000.0

    @pytest.mark.unit
    def test_resume_gap_defaults(self, temp_config_ini):
        """0.6 复牌跳空阈值：默认 0.05，且 0=关闭 合法。"""
        cfg = Config(str(temp_config_ini))
        bt = cfg.app_config.backtest
        assert bt.RESUME_GAP_UP == pytest.approx(0.05)
        assert bt.RESUME_GAP_DOWN == pytest.approx(0.05)

    @pytest.mark.unit
    def test_macd_params_parsed_as_tuple(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.MACD_PARAMS == (12, 26, 9)

    @pytest.mark.unit
    def test_moving_average_periods(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.MOVING_AVERAGE_PERIODS == [10, 20, 30]

    @pytest.mark.unit
    def test_exempt_levels(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.EXEMPT_LEVELS == ["完全主升", "趋势加速"]

    @pytest.mark.unit
    def test_bayesian_cpcv_defaults(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        bt = cfg.app_config.backtest
        assert bt.BAYESIAN_CPCV_PURGE_DAYS >= 0
        assert bt.BAYESIAN_CPCV_EMBARGO_DAYS >= 0
        assert bt.BAYESIAN_TIME_BUDGET_SECONDS >= 600
        assert bt.BAYESIAN_MAX_NO_IMPROVE_WINDOWS >= 1

    @pytest.mark.unit
    def test_bayesian_cpcv_parsed_from_ini(self, tmp_path):
        ini = tmp_path / "config.ini"
        ini.write_text(
            "[DATABASE]\nUSER=u\nPASSWORD=p\nHOST=h\nPORT=5432\nDB_NAME=d\n\n"
            "[BACKTEST]\n"
            "bayesian_cpcv_purge_days = 10\n"
            "bayesian_cpcv_embargo_days = 7\n"
            "bayesian_time_budget_seconds = 7200\n"
            "bayesian_max_no_improve_windows = 2\n",
            encoding="utf-8",
        )
        cfg = Config(str(ini))
        bt = cfg.app_config.backtest
        assert bt.BAYESIAN_CPCV_PURGE_DAYS == 10
        assert bt.BAYESIAN_CPCV_EMBARGO_DAYS == 7
        assert bt.BAYESIAN_TIME_BUDGET_SECONDS == 7200
        assert bt.BAYESIAN_MAX_NO_IMPROVE_WINDOWS == 2


class TestHotReload:
    @pytest.mark.unit
    def test_reload_detects_changes(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        assert cfg.app_config.backtest.INITIAL_CASH == 1_000_000
        content = temp_config_ini.read_text(encoding="utf-8")
        content = content.replace("INITIAL_CASH = 1000000", "INITIAL_CASH = 2000000")
        temp_config_ini.write_text(content, encoding="utf-8")
        cfg.reload()
        assert cfg.app_config.backtest.INITIAL_CASH == 2_000_000

    @pytest.mark.unit
    def test_watch_detects_change(self, temp_config_ini):
        cfg = Config(str(temp_config_ini))
        import threading, time
        changed = []
        def callback(c):
            changed.append(True)
        t = threading.Thread(target=cfg.watch, args=(0.05, callback), daemon=True)
        t.start()
        time.sleep(0.1)
        content = temp_config_ini.read_text(encoding="utf-8")
        content = content.replace("INITIAL_CASH = 1000000", "INITIAL_CASH = 3000000")
        temp_config_ini.write_text(content, encoding="utf-8")
        time.sleep(0.15)
        assert len(changed) >= 1


class TestConfigCipher:
    @pytest.mark.unit
    def test_encrypt_decrypt_roundtrip(self):
        c = ConfigCipher()
        plain = "sensitive_password"
        token = c.encrypt(plain)
        assert token != plain
        assert c.decrypt(token) == plain

    @pytest.mark.unit
    def test_is_encrypted_detects_prefix(self):
        assert ConfigCipher.is_encrypted("ENC:gAAAAA...") is True
        assert ConfigCipher.is_encrypted("plaintext") is False

    @pytest.mark.unit
    def test_strip_prefix(self):
        assert ConfigCipher.strip_prefix("ENC:value") == "value"
        assert ConfigCipher.strip_prefix("plain") == "plain"

    @pytest.mark.unit
    def test_looks_like_fernet_token(self):
        assert ConfigCipher.looks_like_fernet_token("gAAAAAabc123") is True
        assert ConfigCipher.looks_like_fernet_token("plain_text") is False

    @pytest.mark.unit
    def test_maybe_decrypt_encrypted_value(self):
        c = ConfigCipher()
        token = c.encrypt("secret")
        prefixed = f"ENC:{token}"
        result = ConfigCipher.maybe_decrypt(prefixed)
        assert result == "secret"

    @pytest.mark.unit
    def test_maybe_decrypt_plaintext_passthrough(self):
        assert ConfigCipher.maybe_decrypt("not_encrypted") == "not_encrypted"
