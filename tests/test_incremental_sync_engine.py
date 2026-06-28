from __future__ import annotations

import datetime
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from DataManager.IncrementalSyncEngine import IncrementalSyncEngine


class TestAlignToTradingDay:
    @pytest.mark.unit
    def test_align_returns_same_if_trading_day(self, monkeypatch):
        monkeypatch.setattr(
            "DataManager.IncrementalSyncEngine.TradingCalendarAnalyzer",
            _mock_calendar(dates={"2024-06-28"}),
        )
        result = IncrementalSyncEngine.align_to_trading_day("20240628")
        assert result == "20240628"

    @pytest.mark.unit
    def test_align_shifts_to_next_trading_day(self, monkeypatch):
        monkeypatch.setattr(
            "DataManager.IncrementalSyncEngine.TradingCalendarAnalyzer",
            _mock_calendar(dates={"2024-06-28", "2024-07-01"}),
        )
        result = IncrementalSyncEngine.align_to_trading_day("20240629")
        assert result == "20240701"

    @pytest.mark.unit
    def test_align_fallback_on_error(self, monkeypatch):
        monkeypatch.setattr(
            "DataManager.IncrementalSyncEngine.TradingCalendarAnalyzer",
            _mock_calendar(raise_on_get=True),
        )
        result = IncrementalSyncEngine.align_to_trading_day("20240628")
        assert result == "20240628"


class TestFilterStStocks:
    @pytest.mark.unit
    def test_filters_st(self):
        df = pd.DataFrame({"name": ["正常股票", "*ST 风险", "★ST 警示", "S ST 处理"]})
        result = IncrementalSyncEngine.filter_st_stocks(df)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "正常股票"

    @pytest.mark.unit
    def test_keeps_non_st(self):
        df = pd.DataFrame({"name": ["贵州茅台", "宁德时代"]})
        result = IncrementalSyncEngine.filter_st_stocks(df)
        assert len(result) == 2

    @pytest.mark.unit
    def test_handles_empty(self):
        df = pd.DataFrame({"name": []})
        result = IncrementalSyncEngine.filter_st_stocks(df)
        assert result.empty


class TestCleanupOldCache:
    @pytest.mark.unit
    def test_removes_old_cache_files(self, temp_cache_dir, monkeypatch):
        engine = _make_engine(cache_dir=str(temp_cache_dir))
        engine._trade_date_str = "20991231"
        old = temp_cache_dir / "close_normal_20000101.csv"
        old.write_text("dummy")
        fresh = temp_cache_dir / "close_normal_20991231.csv"
        fresh.write_text("dummy")
        engine._cleanup_old_cache()
        assert not old.exists()
        assert fresh.exists()

    @pytest.mark.unit
    def test_leaves_non_close_normal_files(self, temp_cache_dir, monkeypatch):
        orphan = temp_cache_dir / "some_orphan_file.csv"
        orphan.write_text("dummy")
        monkeypatch.setattr(IncrementalSyncEngine, "_cleanup_old_cache", lambda self: None)
        engine = _make_engine(cache_dir=str(temp_cache_dir))
        engine._cache_dir = str(temp_cache_dir)
        engine._cleanup_old_cache()
        assert orphan.exists()


class TestFailedSetCache:
    @pytest.mark.unit
    def test_save_and_load_roundtrip(self, temp_cache_dir):
        engine = _make_engine(cache_dir=str(temp_cache_dir), trade_date="20240628")
        engine._save_failed_set({"sym1", "sym2"})
        loaded = engine._load_failed_set()
        assert loaded == {"sym1", "sym2"}

    @pytest.mark.unit
    def test_save_empty_removes_file(self, temp_cache_dir):
        engine = _make_engine(cache_dir=str(temp_cache_dir), trade_date="20240628")
        engine._save_failed_set({"sym1"})
        engine._save_failed_set(set())
        assert not os.path.exists(engine._failed_cache_path())

    @pytest.mark.unit
    def test_load_missing_file(self, temp_cache_dir):
        engine = _make_engine(cache_dir=str(temp_cache_dir), trade_date="20240628")
        loaded = engine._load_failed_set()
        assert loaded == set()


class TestDetectSplitFromAdj:
    @pytest.mark.unit
    def test_detects_split_when_adj_changes(self, monkeypatch):
        engine = _make_engine()
        new_df = pd.DataFrame({
            "trade_date": ["2024-06-28", "2024-07-01"],
            "adj_factor": [1.05, 1.05],
        })
        monkeypatch.setattr(engine, "_get_latest_date", lambda sym: datetime.date(2024, 6, 28))
        monkeypatch.setattr(engine, "_engine", _mock_db_engine({"adj_factor": 1.0}))
        assert engine._detect_split_from_adj("sh600000", new_df, datetime.date(2024, 6, 28)) == True

    @pytest.mark.unit
    def test_no_split_when_adj_stable(self, monkeypatch):
        engine = _make_engine()
        new_df = pd.DataFrame({
            "trade_date": ["2024-06-28", "2024-07-01"],
            "adj_factor": [1.0, 1.0],
        })
        monkeypatch.setattr(engine, "_get_latest_date", lambda sym: datetime.date(2024, 6, 28))
        monkeypatch.setattr(engine, "_engine", _mock_db_engine({"adj_factor": 1.0}))
        assert engine._detect_split_from_adj("sh600000", new_df, datetime.date(2024, 6, 28)) == False


class TestCalcStartIso:
    @pytest.mark.unit
    def test_uses_min_latest_date_with_overlap(self, monkeypatch):
        engine = _make_engine(default_start="20200101")
        monkeypatch.setattr(engine, "_get_min_latest_date", lambda syms: datetime.date(2024, 6, 1))
        result = engine._calc_start_iso(["sh600000"])
        expected = (datetime.date(2024, 6, 1) - datetime.timedelta(days=15)).isoformat()
        assert result == expected

    @pytest.mark.unit
    def test_falls_back_to_default(self, monkeypatch):
        engine = _make_engine(default_start="20200101")
        engine._default_start = "20200101"
        monkeypatch.setattr(engine, "_get_min_latest_date", lambda syms: None)
        result = engine._calc_start_iso(["sh600000"])
        assert result == "2020-01-01"


# ── 辅助函数 ──

def _mock_calendar(dates=None, raise_on_get=False):
    class Fake:
        def get_official_trading_dates(self):
            return dates or set()

        def get_last_trading_day(self, *a, **kw):
            if raise_on_get:
                raise RuntimeError("模拟异常")
            if dates:
                return max(dates)
            return "2024-06-28"
    return Fake


def _mock_db_engine(row: dict):
    import sqlalchemy
    from sqlalchemy import text
    class FakeResult:
        def __init__(self, rows):
            self._rows = rows
            self._idx = 0
        def fetchone(self):
            if self._idx < len(self._rows):
                r = self._rows[self._idx]
                self._idx += 1
                return r
            return None
        def scalar(self):
            if self._rows:
                return list(self._rows[0].values())[0] if isinstance(self._rows[0], dict) else self._rows[0]
            return None
        def fetchall(self):
            return []
        def keys(self):
            return list(row.keys())
        def __iter__(self):
            return iter(self._rows)
    class FakeConnection:
        def execute(self, stmt, parameters=None, **kw):
            if isinstance(stmt, sqlalchemy.sql.elements.TextClause):
                sql = str(stmt)
                if "MAX" in sql or "SELECT adj_factor" in sql or "SELECT close_normal" in sql:
                    return FakeResult([row])
            return FakeResult([])
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
    class FakeEngine:
        def connect(self):
            return FakeConnection()
        def begin(self):
            return FakeConnection()
    return FakeEngine()


def _make_engine(cache_dir=None, default_start=None, trade_date=None):
    db = _mock_db_engine({})
    eng = IncrementalSyncEngine(
        db_engine=db,
        default_start=default_start or "20200101",
        cache_dir=cache_dir or tempfile.mkdtemp(),
    )
    if trade_date:
        eng._trade_date = trade_date
    return eng
