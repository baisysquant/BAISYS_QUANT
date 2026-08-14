"""Task F: 交易日历与停牌标志对齐 — 测试。

覆盖：日历维护（只读/拉取分离、注入钩子）、对齐标志列、停牌统计（日历口径）、
precheck 日历口径 SKIP（回退启发式不变）、Phase 0 预检线程化、worker 线程化、
prepare 输出标志、引擎日轴对齐（补全日结转）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BackTrading import calendar_align as _ca


def _reset_cal(monkeypatch: pytest.MonkeyPatch) -> None:
    _ca.set_official_calendar(None)
    _ca._mem_dates = None
    _ca._mem_loaded_at = None


@pytest.fixture(autouse=True)
def _cal_state_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    yield
    _reset_cal(monkeypatch)


def _ohlcv(n: int = 60, start: str = "2024-01-02") -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start, periods=n, freq="B").strftime("%Y-%m-%d")
    close = 10.0 * (1 + np.linspace(0, 0.2, n)) + np.random.randn(n) * 0.1
    close = np.maximum(close, 5.0)
    return pd.DataFrame({
        "trade_date": dates,
        "open": close + np.random.randn(n) * 0.05,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })


# ── 日历维护 ──────────────────────────────────────────────────────────

class TestCalendarMaintenance:
    @pytest.mark.unit
    def test_set_and_get_injected_calendar(self, monkeypatch):
        _reset_cal(monkeypatch)
        dates = {"2024-01-01", "2024-01-02"}
        _ca.set_official_calendar(dates)
        assert _ca.get_official_calendar() == dates
        assert _ca.get_official_calendar() is not dates  # 返回副本

    @pytest.mark.unit
    def test_get_from_disk_cache_no_fetch(self, monkeypatch):
        _reset_cal(monkeypatch)
        monkeypatch.setattr(_ca, "_read_disk_cache", lambda: {"2024-01-01", "2024-01-02"})
        assert _ca.get_official_calendar() == {"2024-01-01", "2024-01-02"}

    @pytest.mark.unit
    def test_get_empty_when_nothing_available(self, monkeypatch):
        _reset_cal(monkeypatch)
        monkeypatch.setattr(_ca, "_read_disk_cache", lambda: None)
        assert _ca.get_official_calendar() == set()

    @pytest.mark.unit
    def test_maintain_uses_override_without_fetch(self, monkeypatch):
        _reset_cal(monkeypatch)
        dates = {f"2024-01-{d:02d}" for d in range(1, 11)}
        _ca.set_official_calendar(dates)
        assert _ca.maintain_calendar() == 10

    @pytest.mark.unit
    def test_maintain_disk_fresh_without_fetch(self, monkeypatch):
        _reset_cal(monkeypatch)
        dates = {f"2024-02-{d:02d}" for d in range(1, 5)}
        monkeypatch.setattr(_ca, "_read_disk_cache", lambda: dates)
        assert _ca.maintain_calendar() == 4

    @pytest.mark.unit
    def test_maintain_fallback_to_zero(self, monkeypatch):
        _reset_cal(monkeypatch)
        monkeypatch.setattr(_ca, "_read_disk_cache", lambda: None)

        class _BoomAnalyzer:
            def get_official_trading_dates(self):
                raise RuntimeError("fetch failed")

        monkeypatch.setattr("DataCollection.CalendarManager.TradingCalendarAnalyzer", _BoomAnalyzer)
        assert _ca.maintain_calendar() == 0

    @pytest.mark.unit
    def test_align_enabled_from_config(self, monkeypatch):
        class _Cfg:
            CALENDAR_ALIGN_MODE = "off"

        monkeypatch.setattr("UtilsManager.ConfigParser.Config", lambda: _Cfg())
        assert _ca.align_enabled() is False
        _Cfg.CALENDAR_ALIGN_MODE = "on"
        assert _ca.align_enabled() is True
        _Cfg.CALENDAR_ALIGN_MODE = "ON"
        assert _ca.align_enabled() is True

    @pytest.mark.unit
    def test_align_enabled_safe_on_config_error(self, monkeypatch):
        def _boom():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("UtilsManager.ConfigParser.Config", _boom)
        assert _ca.align_enabled() is False


# ── 对齐标志列 ────────────────────────────────────────────────────────

class TestAlignmentFlags:
    @pytest.mark.unit
    def test_adds_bool_flags(self):
        df = pd.DataFrame({"symbol": ["a"], "trade_date": ["2024-01-02"]})
        out = _ca.add_alignment_flags(df)
        assert "is_trading" in out.columns and "is_suspended" in out.columns
        assert out["is_trading"].dtype == bool
        assert bool(out.iloc[0]["is_trading"]) is True
        assert bool(out.iloc[0]["is_suspended"]) is False

    @pytest.mark.unit
    def test_idempotent(self):
        df = _ca.add_alignment_flags(pd.DataFrame({"trade_date": ["2024-01-02"]}))
        df2 = _ca.add_alignment_flags(df)
        assert list(df2.columns).count("is_trading") == 1
        assert list(df2.columns).count("is_suspended") == 1


# ── 停牌统计（日历口径） ───────────────────────────────────────────────

class TestSuspensionStats:
    def _cal(self, n: int = 10) -> set[str]:
        return {f"2024-01-{d:02d}" for d in range(1, n + 1)}

    @pytest.mark.unit
    def test_detects_missing_calendar_days(self):
        cal = self._cal()
        df = pd.DataFrame({
            "symbol": ["sh600000"] * 8,
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03",
                           "2024-01-06", "2024-01-07", "2024-01-08",
                           "2024-01-09", "2024-01-10"],
        })
        stats = _ca.compute_suspension_stats(df, official_dates=cal)
        assert set(stats) == {"sh600000"}
        s = stats["sh600000"]
        assert s["span_trading_days"] == 10
        assert s["suspended_days"] == ["2024-01-04", "2024-01-05"]
        assert s["suspension_ratio"] == pytest.approx(0.2)

    @pytest.mark.unit
    def test_all_days_present_ratio_zero(self):
        cal = self._cal()
        df = pd.DataFrame({
            "symbol": ["sh600000"] * 10,
            "trade_date": [f"2024-01-{d:02d}" for d in range(1, 11)],
        })
        s = _ca.compute_suspension_stats(df, official_dates=cal)["sh600000"]
        assert s["suspended_days"] == []
        assert s["suspension_ratio"] == 0.0

    @pytest.mark.unit
    def test_span_limited_to_listed_range(self):
        cal = self._cal()
        df = pd.DataFrame({
            "symbol": ["sh600000"] * 4,
            "trade_date": ["2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10"],
        })
        s = _ca.compute_suspension_stats(df, official_dates=cal)["sh600000"]
        assert s["span_trading_days"] == 4
        assert s["suspension_ratio"] == 0.0

    @pytest.mark.unit
    def test_non_calendar_dates_not_counted(self):
        cal = {d for d in pd.date_range("2024-01-01", periods=8, freq="B").strftime("%Y-%m-%d")}
        df = pd.DataFrame({
            "symbol": ["sh600000"] * 7,
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03",
                           "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"],
        })
        # 01-06/01-07 为周六日（非官方交易日）：不计入 span（该日也不构成停牌）
        s = _ca.compute_suspension_stats(df, official_dates=cal)["sh600000"]
        assert s["span_trading_days"] == 5
        assert s["suspended_days"] == []
        assert s["suspension_ratio"] == 0.0

    @pytest.mark.unit
    def test_datetime_dates_normalized(self):
        cal = self._cal(3)
        df = pd.DataFrame({
            "symbol": ["sh600000"] * 2,
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        })
        s = _ca.compute_suspension_stats(df, official_dates=cal)["sh600000"]
        assert s["suspended_days"] == ["2024-01-02"]

    @pytest.mark.unit
    def test_empty_calendar_returns_empty(self, monkeypatch):
        _reset_cal(monkeypatch)
        monkeypatch.setattr(_ca, "_read_disk_cache", lambda: None)
        df = pd.DataFrame({"symbol": ["a"], "trade_date": ["2024-01-01"]})
        assert _ca.compute_suspension_stats(df) == {}

    @pytest.mark.unit
    def test_missing_columns_returns_empty(self):
        assert _ca.compute_suspension_stats(pd.DataFrame({"x": [1]})) == {}
        assert _ca.compute_suspension_stats(pd.DataFrame()) == {}

    @pytest.mark.unit
    def test_interior_missing_blocks_and_tail(self):
        """span 内缺失=interior（两端有成交）；末日之后到全池最末日缺失=tail。"""
        cal = {f"2024-01-{d:02d}" for d in range(1, 11)}  # 10 个官方交易日
        # sh600000: 01-01..01-02 与 01-08..01-10 有数据 → interior 缺失 01-03..01-07（5 天）
        # sh600001: 01-01..01-05 有数据，全池最末日 01-10 → tail 缺失 01-06..01-10（5 天）
        rows = []
        for d in ("2024-01-01", "2024-01-02", "2024-01-08", "2024-01-09", "2024-01-10"):
            rows.append({"symbol": "sh600000", "trade_date": d})
        for d in ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"):
            rows.append({"symbol": "sh600001", "trade_date": d})
        df = pd.DataFrame(rows)
        stats = _ca.compute_suspension_stats(df, official_dates=cal)
        s0 = stats["sh600000"]
        assert s0["interior_missing_days"] == ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]
        assert s0["suspension_ratio"] == pytest.approx(0.5)
        assert s0["missing_blocks"] == [{"start": "2024-01-03", "end": "2024-01-07", "days": 5}]
        assert s0["tail_missing_days"] == []  # 自身覆盖到全池最末日
        assert s0["cross_validated"] is False
        s1 = stats["sh600001"]
        assert s1["interior_missing_days"] == []
        assert s1["tail_missing_days"] == ["2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10"]
        # span 截止于自身最末日 01-05 → 停牌占比不因"末日缺失"虚高（老口径不变）
        assert s1["span_trading_days"] == 5
        assert s1["suspension_ratio"] == 0.0

    @pytest.mark.unit
    def test_cross_validation_confirmed_vs_under_collected(self):
        cal = {f"2024-01-{d:02d}" for d in range(1, 11)}
        df = pd.DataFrame({
            "symbol": ["sh600000"] * 6,
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03",
                           "2024-01-09", "2024-01-10", "2024-01-11"],
        })
        confirmed = {"2024-01-05", "2024-01-06"}  # 公告确认停牌 2 天
        s = _ca.compute_suspension_stats(df, official_dates=cal,
                                         confirmed_suspension_days=confirmed)["sh600000"]
        assert s["cross_validated"] is True
        # span=01-01..01-10（10 天），缺失=04..08（5 天）；确认停牌=05,06；漏采嫌疑=04,07,08
        assert s["confirmed_days"] == ["2024-01-05", "2024-01-06"]
        assert s["under_collected_days"] == ["2024-01-04", "2024-01-07", "2024-01-08"]
        assert s["suspension_ratio_confirmed"] == pytest.approx(0.2)
        assert s["under_collection_ratio"] == pytest.approx(0.3)


# ── precheck 日历口径 SKIP ────────────────────────────────────────────

class TestPrecheckCalendarSuspension:
    @pytest.mark.unit
    def test_high_ratio_skips(self):
        from BackTrading.precheck import PrecheckStatus, precheck
        df = _ohlcv()
        res = precheck(df, {"mode": "RELAX"}, suspension_stats={
            "span_trading_days": 10, "suspended_days": ["2024-01-03", "2024-01-04", "2024-01-05"],
        })
        assert res.status == PrecheckStatus.SKIP
        assert "SUSPENSION_RATIO_HIGH_CAL" in res.reasons
        assert res.metrics["SUSPENSION_RATIO_HIGH_CAL"]["ratio"] == pytest.approx(0.3)

    @pytest.mark.unit
    def test_low_ratio_ok(self):
        from BackTrading.precheck import PrecheckStatus, precheck
        df = _ohlcv()
        res = precheck(df, {"mode": "RELAX"}, suspension_stats={
            "span_trading_days": 100, "suspended_days": ["2024-01-03"],
        })
        assert res.status == PrecheckStatus.OK
        assert "SUSPENSION_RATIO_HIGH_CAL" not in res.reasons

    @pytest.mark.unit
    def test_legacy_heuristic_unchanged_without_stats(self):
        from BackTrading.precheck import PrecheckStatus, precheck
        df = _ohlcv(60)
        df.loc[df.index < 20, "volume"] = 0.0
        df.loc[df.index < 20, "close"] = df["close"].iloc[20]
        res = precheck(df, {"mode": "RELAX"})
        assert res.status == PrecheckStatus.LOW_CONFIDENCE
        assert "SUSPENSION_RATIO_HIGH" in res.reasons
        assert "SUSPENSION_RATIO_HIGH_CAL" not in res.reasons

    @pytest.mark.unit
    def test_apply_precheck_calendar_skip_returns_empty(self):
        from BackTrading.precheck import apply_precheck
        out, res = apply_precheck("zz000001", _ohlcv(), context="test",
                                  suspension_stats={
                                      "span_trading_days": 10,
                                      "suspended_days": ["2024-01-03", "2024-01-04", "2024-01-05"],
                                  })
        assert out.empty
        assert "SUSPENSION_RATIO_HIGH_CAL" in res.reasons


# ── 指标计算 / worker 线程化 ──────────────────────────────────────────

class TestWorkerThreading:
    @pytest.mark.unit
    def test_stock_worker_calendar_skip(self, tmp_path, monkeypatch):
        """P0-10 ②：循环路径已删除。日历口径 SKIP 在 Phase 0 预计算阶段拦截，
        _stock_worker_vectorized 对 SKIP 股票返回 []（不进入信号计算）。"""
        import BackTrading.indicator_cache as ic
        ic._reset_memory_caches()
        monkeypatch.setattr(ic, "_cache_root", lambda: tmp_path / "icache")
        from BackTrading.prepare import _stock_worker_vectorized, precompute_all_indicators
        stock_dir = tmp_path / "stocks"
        stock_dir.mkdir()
        df = _ohlcv(60)
        df["symbol"] = "sh600000"
        df.to_parquet(stock_dir / "sh600000.parquet", index=False, engine="fastparquet")
        stats = {"sh600000": {
            "span_trading_days": 10, "suspended_days": ["2024-01-03", "2024-01-04", "2024-01-05"],
        }}
        precompute_all_indicators(str(stock_dir), suspension_stats=stats, shard_mode="off")
        rows = _stock_worker_vectorized("sh600000", str(stock_dir), {}, False, stats)
        assert rows == []

    @pytest.mark.unit
    def test_stock_worker_without_stats_still_runs_legacy_path(self, tmp_path, monkeypatch):
        # 无统计时回退启发式：正常数据不应被误杀（此处仅验证不因新参数崩溃，
        # 信号计算失败返回空属预期；stats 缺省分支应走旧逻辑）
        import BackTrading.indicator_cache as ic
        ic._reset_memory_caches()
        monkeypatch.setattr(ic, "_cache_root", lambda: tmp_path / "icache")
        from BackTrading.prepare import _stock_worker_vectorized, precompute_all_indicators
        stock_dir = tmp_path / "stocks"
        stock_dir.mkdir()
        df = _ohlcv(60)
        df["symbol"] = "sh600000"
        df.to_parquet(stock_dir / "sh600000.parquet", index=False, engine="fastparquet")
        precompute_all_indicators(str(stock_dir), shard_mode="off")
        _stock_worker_vectorized("sh600000", str(stock_dir), {})  # 不抛异常即可

    @pytest.mark.unit
    def test_phase0_calendar_skip(self, tmp_path, monkeypatch):
        import BackTrading.indicator_cache as ic
        monkeypatch.setattr(ic, "_cache_root", lambda: tmp_path / "icache")
        stock_dir = tmp_path / "stocks"
        stock_dir.mkdir()
        df = _ohlcv(60)
        df.to_parquet(stock_dir / "zz000001.parquet", index=False, engine="fastparquet")
        stats = {"zz000001": {
            "span_trading_days": 10, "suspended_days": ["2024-01-03", "2024-01-04", "2024-01-05"],
        }}
        res = ic._precompute_one_symbol(stock_dir / "zz000001.parquet", "zz000001", stats)
        assert res == "empty"


# ── prepare 集成 ──────────────────────────────────────────────────────

class TestPrepareIntegration:
    def _kline(self, n: int = 70) -> pd.DataFrame:
        df = _ohlcv(n)
        frames = []
        for sym in ("sh600000", "sh600001", "sh600002"):
            g = df.copy()
            g["symbol"] = sym
            if sym == "sh600002":
                g = g.iloc[::2].copy()  # 隔日停牌 → 日历口径停牌占比 ~50%
            frames.append(g)
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "trade_date"])

    def _run_prepare(self, monkeypatch, tmp_path):
        from BackTrading import prepare as _prep
        captured: dict[str, dict] = {}

        def _fake_worker(symbol, stock_dir, params, compute_exit_strategy=False, susp_stats=None):
            captured[symbol] = (susp_stats or {}).get(symbol, {}).copy()
            return [{
                "symbol": symbol,
                "trade_date": "2024-01-02",
                "进场评分": 0, "退出评分": 0, "综合评分": 0,
                "止损价": 0.0, "风险等级": "LOW",
            }]

        monkeypatch.setattr(_prep, "_stock_worker_vectorized", _fake_worker)
        monkeypatch.setattr(_prep, "apply_ml_signal", lambda df: df)
        monkeypatch.setattr(_prep, "_trade_day_str", lambda: "2024-03-31")
        monkeypatch.setattr(_prep, "CACHE_DIR", tmp_path / "signal_cache")
        return _prep, captured

    @pytest.mark.unit
    def test_prepare_adds_flags_and_passes_stats(self, monkeypatch, tmp_path):
        cal = {d for d in pd.date_range("2024-01-01", periods=200, freq="B").strftime("%Y-%m-%d")}
        _ca.set_official_calendar(cal)
        monkeypatch.setattr(_ca, "align_enabled", lambda: True)
        monkeypatch.setattr(_ca, "maintain_calendar", lambda force=False: len(cal))
        _prep, captured = self._run_prepare(monkeypatch, tmp_path)
        kline = self._kline()
        out = _prep.prepare_backtest_data(kline, params={"atr_stop_mult": 1.5}, vectorized=True)
        assert "is_trading" in out.columns and "is_suspended" in out.columns
        assert bool(out["is_trading"].all())
        assert not bool(out["is_suspended"].any())
        assert set(captured) == {"sh600000", "sh600001", "sh600002"}
        assert captured["sh600000"]["suspension_ratio"] == 0.0
        assert captured["sh600002"]["suspension_ratio"] > 0.4

    @pytest.mark.unit
    def test_prepare_align_off_no_flags(self, monkeypatch, tmp_path):
        from BackTrading import prepare as _prep
        _ca.set_official_calendar({d for d in pd.date_range("2024-01-01", periods=70).strftime("%Y-%m-%d")})
        monkeypatch.setattr(_ca, "align_enabled", lambda: False)
        _prep, captured = self._run_prepare(monkeypatch, tmp_path)
        out = _prep.prepare_backtest_data(self._kline(), params={"atr_stop_mult": 1.5}, vectorized=True)
        assert "is_trading" not in out.columns
        assert "is_suspended" not in out.columns


# ── 引擎日轴对齐 ──────────────────────────────────────────────────────

class TestEngineCalendarAxis:
    def _frame(self, days: list[str], with_flags: bool = True) -> pd.DataFrame:
        rows = []
        for d in days:
            rows.append({
                "symbol": "sh600000",
                "trade_date": d,
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.1,
                "close_adj": 10.1, "volume": 1_000_000.0,
                "进场评分": 0, "退出评分": 0, "风险等级": "LOW", "止损价": 0.0,
            })
        df = pd.DataFrame(rows)
        if with_flags:
            df["is_trading"] = True
            df["is_suspended"] = False
        return df

    @pytest.mark.unit
    def test_missing_calendar_day_appears_in_equity_curve(self, monkeypatch):
        from BackTrading.engine import run_full_backtest
        from BackTrading.engine import EngineConfig
        data = self._frame(["2024-01-02", "2024-01-04"])  # 01-03 缺失
        _ca.set_official_calendar({"2024-01-02", "2024-01-03", "2024-01-04"})
        _, curve = run_full_backtest(data, {}, EngineConfig(point_in_time=False))
        times = [str(e["time"]) for e in curve]
        assert times == ["2024-01-02", "2024-01-03", "2024-01-04"]
        assert curve[1]["portfolio_value"] == curve[0]["portfolio_value"]  # 结转
        assert curve[1]["turnover"] == 0.0

    @pytest.mark.unit
    def test_no_flags_keeps_data_axis(self, monkeypatch):
        from BackTrading.engine import run_full_backtest
        from BackTrading.engine import EngineConfig
        data = self._frame(["2024-01-02", "2024-01-04"], with_flags=False)
        _ca.set_official_calendar({"2024-01-02", "2024-01-03", "2024-01-04"})
        _, curve = run_full_backtest(data, {}, EngineConfig(point_in_time=False))
        assert [str(e["time"]) for e in curve] == ["2024-01-02", "2024-01-04"]

    @pytest.mark.unit
    def test_calendar_unavailable_falls_back_to_data_axis(self, monkeypatch):
        from BackTrading.engine import run_full_backtest
        from BackTrading.engine import EngineConfig
        data = self._frame(["2024-01-02", "2024-01-04"])
        _reset_cal(monkeypatch)
        monkeypatch.setattr("BackTrading.engine.core._cal_get", lambda: set())
        _, curve = run_full_backtest(data, {}, EngineConfig(point_in_time=False))
        assert [str(e["time"]) for e in curve] == ["2024-01-02", "2024-01-04"]

    @pytest.mark.unit
    def test_empty_calendar_day_in_legacy_portfolio_path(self, monkeypatch):
        from BackTrading.engine import run_full_backtest
        from BackTrading.engine import EngineConfig
        data = self._frame(["2024-01-02", "2024-01-04"])
        _ca.set_official_calendar({"2024-01-02", "2024-01-03", "2024-01-04"})
        _, curve = run_full_backtest(
            data, {}, EngineConfig(point_in_time=False, portfolio_method="equal_weight")
        )
        assert [str(e["time"]) for e in curve] == ["2024-01-02", "2024-01-03", "2024-01-04"]
        assert curve[1]["portfolio_value"] == curve[0]["portfolio_value"]


# ── 配置校验 ──────────────────────────────────────────────────────────

class TestConfigValidation:
    @pytest.mark.unit
    def test_calendar_align_mode_validator(self):
        from UtilsManager.ConfigParser import BacktestConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(CALENDAR_ALIGN_MODE="sideways")
        assert BacktestConfig(CALENDAR_ALIGN_MODE="On").CALENDAR_ALIGN_MODE == "on"
        assert BacktestConfig(CALENDAR_ALIGN_MODE="OFF").CALENDAR_ALIGN_MODE == "off"
        assert BacktestConfig().CALENDAR_ALIGN_MODE == "on"
        assert BacktestConfig().CALENDAR_TTL_HOURS == 24.0

    @pytest.mark.unit
    def test_calendar_ttl_bounds(self):
        from UtilsManager.ConfigParser import BacktestConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(CALENDAR_TTL_HOURS=0.5)
        assert BacktestConfig(CALENDAR_TTL_HOURS=6).CALENDAR_TTL_HOURS == 6.0

    @pytest.mark.unit
    def test_temp_config_defaults(self, temp_config_ini):
        from UtilsManager.ConfigParser import Config
        cfg = Config(str(temp_config_ini))
        assert cfg.CALENDAR_ALIGN_MODE == "off"  # 测试隔离：默认关闭对齐
        assert cfg.CALENDAR_TTL_HOURS == 24.0
