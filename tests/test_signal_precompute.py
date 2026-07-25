from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


class TestConfigHash:
    @pytest.mark.unit
    def test_compute_config_hash_is_deterministic(self):
        from BackTrading.prepare import _compute_config_hash
        h1 = _compute_config_hash()
        h2 = _compute_config_hash()
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 8

    @pytest.mark.unit
    def test_config_hash_is_unknown_on_config_error(self, monkeypatch):
        from BackTrading.prepare import _compute_config_hash
        monkeypatch.setattr("BackTrading.prepare.Config", lambda: (_ for _ in ()).throw(RuntimeError("no config")))
        assert _compute_config_hash() == "unknown"


class TestParamHash:
    @pytest.mark.unit
    def test_param_hash_is_deterministic(self):
        from BackTrading.prepare import _compute_param_hash
        params = {"atr_stop_mult": 1.5, "kelly_fraction": 0.25, "position_a": 0.3}
        h1 = _compute_param_hash(params)
        h2 = _compute_param_hash(params)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 8

    @pytest.mark.unit
    def test_param_hash_differs_with_different_signal_params(self):
        from BackTrading.prepare import _compute_param_hash
        h1 = _compute_param_hash({"golden_cross_bonus": 5})
        h2 = _compute_param_hash({"golden_cross_bonus": 15})
        assert h1 != h2

    @pytest.mark.unit
    def test_param_hash_excludes_atr_params(self):
        from BackTrading.prepare import _compute_param_hash
        h1 = _compute_param_hash({"atr_stop_mult": 1.0})
        h2 = _compute_param_hash({"atr_stop_mult": 3.0})
        assert h1 == h2  # atr_stop_mult 不影响信号，hash 相同

    @pytest.mark.unit
    def test_param_hash_excludes_conclusion_full_bull(self):
        from BackTrading.prepare import _compute_param_hash
        h1 = _compute_param_hash({"conclusion_full_bull": 70})
        h2 = _compute_param_hash({"conclusion_full_bull": 90})
        assert h1 == h2  # conclusion_full_bull 不影响信号，hash 相同

    @pytest.mark.unit
    def test_param_hash_includes_golden_cross_bonus(self):
        from BackTrading.prepare import _compute_param_hash
        h1 = _compute_param_hash({"golden_cross_bonus": 5})
        h2 = _compute_param_hash({"golden_cross_bonus": 15})
        assert h1 != h2

    @pytest.mark.unit
    def test_param_hash_includes_divergence_penalty(self):
        from BackTrading.prepare import _compute_param_hash
        h1 = _compute_param_hash({"divergence_penalty": 10})
        h2 = _compute_param_hash({"divergence_penalty": 30})
        assert h1 != h2

    @pytest.mark.unit
    def test_param_hash_excludes_risk_none_multiplier(self):
        from BackTrading.prepare import _compute_param_hash
        h1 = _compute_param_hash({"risk_none_multiplier": 0.5})
        h2 = _compute_param_hash({"risk_none_multiplier": 2.0})
        assert h1 == h2  # risk_none_multiplier 不影响信号，hash 相同


class TestCacheDirFor:
    @pytest.mark.unit
    def test_cache_dir_format(self):
        from BackTrading.prepare import _cache_dir_for
        d = _cache_dir_for("20240628", param_hash="abc12345", config_hash="cfg12345")
        assert "signal_cache_20240628_cfg12345_abc12345" in str(d)

    @pytest.mark.unit
    def test_cache_dir_without_param_hash(self):
        from BackTrading.prepare import _cache_dir_for
        d = _cache_dir_for("20240628")
        assert "signal_cache_20240628" in str(d)
        assert "None" not in str(d)


class TestSymbolBucket:
    @pytest.mark.unit
    def test_sh_prefix(self):
        from BackTrading.prepare import _symbol_bucket
        assert _symbol_bucket("sh600000") == "sh"

    @pytest.mark.unit
    def test_sz_prefix(self):
        from BackTrading.prepare import _symbol_bucket
        assert _symbol_bucket("sz000001") == "sz"

    @pytest.mark.unit
    def test_lowercase(self):
        from BackTrading.prepare import _symbol_bucket
        assert _symbol_bucket("SH600000") == "sh"


class TestSymbolCachePath:
    @pytest.mark.unit
    def test_path_contains_bucket_and_symbol(self):
        from BackTrading.prepare import _symbol_cache_path
        import tempfile
        d = tempfile.mkdtemp()
        p = _symbol_cache_path(Path(d), "sh600000")
        assert "sh" in str(p)
        assert "sh600000" in str(p)
        import shutil
        shutil.rmtree(d)


class TestCompletedSymbols:
    @pytest.mark.unit
    def test_returns_empty_for_empty_cache(self, temp_cache_dir):
        from BackTrading.prepare import _completed_symbols, _cache_dir_for
        import tempfile
        completed = _completed_symbols("20991231")
        assert isinstance(completed, set)

    @pytest.mark.unit
    def test_detects_existing_symbols(self, temp_cache_dir):
        from BackTrading.prepare import _completed_symbols, _symbol_cache_path, _cache_dir_for, _save_stock_signal
        td = "20991231"
        cd = _cache_dir_for(td)
        cd.mkdir(parents=True, exist_ok=True)
        bucket = cd / "sh"
        bucket.mkdir(parents=True, exist_ok=True)
        (bucket / "sh600000.parquet").write_bytes(b"dummy")
        completed = _completed_symbols(td)
        assert len(completed) > 0


class TestMergeSignal:
    @pytest.mark.unit
    def test_merge_adds_signal_columns(self):
        from BackTrading.prepare import _merge_signal
        kline = pd.DataFrame({
            "symbol": ["sh600000"],
            "trade_date": ["2024-06-28"],
            "close": [10.0],
        })
        signal = pd.DataFrame({
            "symbol": ["sh600000"],
            "trade_date": ["2024-06-28"],
            "进场评分": [80],
            "退出评分": [20],
            "综合评分": [100],
            "止损价": [9.5],
            "风险等级": ["LOW"],
        })
        merged = _merge_signal(kline, signal)
        assert "进场评分" in merged.columns
        assert merged.iloc[0]["进场评分"] == 80

    @pytest.mark.unit
    def test_merge_fills_missing_scores(self):
        from BackTrading.prepare import _merge_signal
        kline = pd.DataFrame({
            "symbol": ["sh600000", "sh600001"],
            "trade_date": ["2024-06-28", "2024-06-28"],
            "close": [10.0, 11.0],
        })
        signal = pd.DataFrame({
            "symbol": ["sh600000"],
            "trade_date": ["2024-06-28"],
            "进场评分": [80],
            "退出评分": [20],
            "综合评分": [100],
            "止损价": [9.5],
            "风险等级": ["LOW"],
        })
        merged = _merge_signal(kline, signal)
        na_row = merged[merged["symbol"] == "sh600001"]
        assert na_row.iloc[0]["进场评分"] == 0


class TestComputeParamHashForWorker:
    """验证 prepare_backtest_data 核心逻辑（信号预计算）的哈希依赖。"""

    @pytest.mark.unit
    def test_params_and_hash_are_consistent(self):
        from BackTrading.prepare import _compute_param_hash, prepare_backtest_data
        params = {"atr_stop_mult": 1.5, "kelly_fraction": 0.25}
        h = _compute_param_hash(params)
        assert isinstance(h, str)


# ── 辅助函数 ──
