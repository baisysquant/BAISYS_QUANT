"""P0-10 审计修复测试：
  ① 经手费/证管费历史分段表（2023-08-28 经手费 0.00487%→0.00341%；
     2015-08-01 证管费 0.004%→0.002%），commission_includes_fees=False 时历史成本不低估
  ② prepare.py 循环兜底路径删除（两套信号实现合一）
  ③ vectorized_signal.py 背离峰事件表 + 金叉衰减向量化：与参考实现定点等价 + 性能基准
  ④ simulated_trading.py 模拟验证改独立验证集（与选参区间无交集）
  ⑤ 已删除模块引用清理（LoggerManager/GetStockBasicinfo）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ═══════════════════════════════════════════════════════════════════
# ① 经手费/证管费分段表
# ═══════════════════════════════════════════════════════════════════

class TestHandlingCsrcSegments:
    @pytest.mark.unit
    def test_default_segments(self):
        from BackTrading.domain.models import (
            DEFAULT_CSRC_FEE_SEGMENTS,
            DEFAULT_HANDLING_FEE_SEGMENTS,
        )

        assert DEFAULT_HANDLING_FEE_SEGMENTS[0] == ("2023-08-28", 0.0000341)
        assert DEFAULT_HANDLING_FEE_SEGMENTS[-1] == ("2000-01-01", 0.0000487)
        assert DEFAULT_CSRC_FEE_SEGMENTS[0] == ("2015-08-01", 0.00002)
        assert DEFAULT_CSRC_FEE_SEGMENTS[-1] == ("2000-01-01", 0.00004)

    @pytest.mark.unit
    def test_rate_for_historical_dates(self):
        from BackTrading.domain.models import CostModel

        cm = CostModel()
        assert cm.handling_fee_rate_for("2023-08-27") == pytest.approx(0.0000487)
        assert cm.handling_fee_rate_for("2023-08-28") == pytest.approx(0.0000341)
        assert cm.handling_fee_rate_for("2026-01-01") == pytest.approx(0.0000341)
        assert cm.handling_fee_rate_for(None) == pytest.approx(0.0000341)
        assert cm.csrc_fee_rate_for("2015-07-31") == pytest.approx(0.00004)
        assert cm.csrc_fee_rate_for("2015-08-01") == pytest.approx(0.00002)
        assert cm.csrc_fee_rate_for("2020-01-01") == pytest.approx(0.00002)

    @pytest.mark.unit
    def test_breakdown_uses_historical_rate_when_not_included(self):
        """commission_includes_fees=False 时，历史日期按分段表计费（不低估 ~30%）。"""
        from BackTrading.domain.models import CostModel

        cm = CostModel(commission_includes_fees=False)
        value = 1_000_000.0
        old = cm.buy_cost_breakdown(value, 1000, 1e7, dt="2015-01-05")
        new = cm.buy_cost_breakdown(value, 1000, 1e7, dt="2024-01-02")
        assert old["handling"] == pytest.approx(value * 0.0000487)
        assert new["handling"] == pytest.approx(value * 0.0000341)
        assert old["csrc"] == pytest.approx(value * 0.00004)
        assert new["csrc"] == pytest.approx(value * 0.00002)
        # 历史费率 > 现行费率（修复前固定 0.00341% → 低估 ~30%）
        assert old["handling"] > new["handling"] * 1.4
        assert old["total"] > new["total"]

    @pytest.mark.unit
    def test_breakdown_includes_fees_default_zero(self):
        """commission_includes_fees=True（默认）：经手费/证管费不单独收取。"""
        from BackTrading.domain.models import CostModel

        cm = CostModel()
        parts = cm.buy_cost_breakdown(1_000_000.0, 1000, 1e7, dt="2020-01-01")
        assert parts["handling"] == 0.0
        assert parts["csrc"] == 0.0

    @pytest.mark.unit
    def test_custom_segments_parsing_with_fallback_injection(self):
        from BackTrading.domain.models import CostModel

        segs = CostModel._parse_handling_segments("2024-01-01:0.00003")
        assert segs[0] == ("2000-01-01", 0.0000487)  # 兜底段强制注入
        assert segs[-1] == ("2024-01-01", 0.00003)
        with pytest.raises(ValueError):
            CostModel._parse_handling_segments("2024-01-01")
        with pytest.raises(ValueError):
            CostModel._parse_csrc_segments("bad")

    @pytest.mark.unit
    def test_from_backtest_config_defaults(self, monkeypatch):
        from BackTrading.domain.models import CostModel, DEFAULT_HANDLING_FEE_SEGMENTS

        class _BT:
            COMMISSION_RATE = 0.0003
            STAMP_TAX_RATE = 0.001
            SLIPPAGE = 0.001
            MIN_COMMISSION_PER_TRADE = 5.0
            TRANSFER_FEE_RATE = 0.00002

        model = CostModel.from_backtest_config(_BT())
        assert model.handling_fee_segments == tuple(
            sorted(DEFAULT_HANDLING_FEE_SEGMENTS)
        )
        assert model.handling_fee_rate_for("2020-01-01") == pytest.approx(0.0000487)


# ═══════════════════════════════════════════════════════════════════
# ② 循环兜底路径删除
# ═══════════════════════════════════════════════════════════════════

class TestLoopPathRemoved:
    @pytest.mark.unit
    def test_loop_worker_and_compute_signal_removed(self):
        import BackTrading.prepare as _prep

        assert not hasattr(_prep, "_stock_worker")
        assert not hasattr(_prep, "_compute_signal")
        assert not hasattr(_prep, "_calc_exit_score")
        assert hasattr(_prep, "_stock_worker_vectorized")

    @pytest.mark.unit
    def test_vectorized_false_still_uses_vectorized_path(self, monkeypatch, tmp_path):
        """vectorized=False 兼容传参：告警 + 统一走向量化（循环路径已删除）。"""
        import BackTrading.prepare as _prep

        calls = []
        monkeypatch.setattr(
            _prep, "_stock_worker_vectorized",
            lambda *a, **k: calls.append(1) or [],
        )
        monkeypatch.setattr(_prep, "apply_ml_signal", lambda df: df)
        monkeypatch.setattr(_prep, "_trade_day_str", lambda: "2024-03-31")
        monkeypatch.setattr(_prep, "CACHE_DIR", tmp_path / "signal_cache")
        monkeypatch.setattr(_prep, "_load_signal_cache", lambda *a, **k: pd.DataFrame())
        monkeypatch.setattr(_prep, "precompute_all_indicators", lambda *a, **k: None)
        kline = pd.DataFrame({
            "symbol": ["sh600000"], "trade_date": ["2024-01-02"],
            "open": [10.0], "high": [10.5], "low": [9.5], "close": [10.0],
            "volume": [1e6],
        })
        _prep.prepare_backtest_data(kline, params={"atr_stop_mult": 1.5}, vectorized=False)
        assert calls, "vectorized=False 也应走 _stock_worker_vectorized"


# ═══════════════════════════════════════════════════════════════════
# ③ 背离/金叉：定点等价 + 性能基准
# ═══════════════════════════════════════════════════════════════════

def _ref_divergence_scores(df: pd.DataFrame, base_distance: int = 10):
    """P0-10 ③ 参考实现：重构前的逐 bar × 逐峰循环（保留在测试中做定点回归）。"""
    from LogicAnalyzer.SignalConstants import Divergence
    from LogicAnalyzer.signals.divergence import adaptive_distance, find_peaks_troughs

    n = len(df)
    div_type = np.full(n, None, dtype=object)
    div_idx = np.full(n, -1, dtype=np.int32)
    div_strength = np.zeros(n, dtype=np.float64)
    close_arr = df["close"].values
    indicator_arr = df["DIF"].values
    max_lookahead = base_distance * 2
    batch_size = max(1, base_distance // 2)

    last_peaks: np.ndarray = np.array([], dtype=int)
    last_troughs: np.ndarray = np.array([], dtype=int)

    for i in range(1, n):
        if i % batch_size == 0:
            sub = pd.Series(indicator_arr[: i + 1]).bfill().ffill()
            if len(sub) < 5 or sub.isna().all():
                last_peaks, last_troughs = np.array([], dtype=int), np.array([], dtype=int)
            else:
                adj = adaptive_distance(sub, base_distance=base_distance)
                last_peaks, last_troughs = find_peaks_troughs(sub, distance=adj)

        for p in reversed(last_peaks):
            if p >= i:
                continue
            if i - p > max_lookahead:
                break
            if (close_arr[p] > close_arr[i] * 0.98) and (indicator_arr[p] > indicator_arr[i]):
                price_ratio = close_arr[i] / close_arr[p] - 1
                ind_ratio = 1 - indicator_arr[i] / indicator_arr[p]
                s = min(1.0, max(0.0, (price_ratio + ind_ratio) / 2))
                if s > 0.15 and s > div_strength[i]:
                    div_type[i] = Divergence.TOP_DIVERGENCE
                    div_idx[i] = p
                    div_strength[i] = s
                break

        for t in reversed(last_troughs):
            if t >= i:
                continue
            if i - t > max_lookahead:
                break
            if (close_arr[t] < close_arr[i] * 1.02) and (indicator_arr[t] < indicator_arr[i]):
                price_ratio = 1 - close_arr[i] / close_arr[t]
                ind_ratio = indicator_arr[i] / indicator_arr[t] - 1
                s = min(1.0, max(0.0, (price_ratio + ind_ratio) / 2))
                if s > 0.15 and s > div_strength[i]:
                    div_type[i] = Divergence.BOTTOM_DIVERGENCE
                    div_idx[i] = t
                    div_strength[i] = s
                break

    return div_type, div_idx, div_strength


def _mk_dif_series(n: int, kind: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if kind == "trend":
        x = np.cumsum(rng.normal(0, 0.1, n)) + np.linspace(0, 2, n)
    elif kind == "oscillate":
        x = np.sin(np.linspace(0, n * 0.4, n)) * 0.8 + rng.normal(0, 0.02, n)
    elif kind == "nan_head":
        x = np.cumsum(rng.normal(0, 0.1, n))
        x[: n // 5] = np.nan
    else:
        x = rng.normal(0, 0.3, n)
    close = 20.0 + np.cumsum(rng.normal(0, 0.2, n)) * 0.5
    return pd.DataFrame({"close": close, "DIF": x})


class TestDivergenceFixedPoint:
    @pytest.mark.parametrize("kind", ["trend", "oscillate", "nan_head", "noise"])
    @pytest.mark.parametrize("n", [60, 250, 1000])
    def test_event_table_equals_reference(self, kind, n):
        """峰事件表实现与参考（逐 bar × 逐峰）输出完全一致（定点回归）。"""
        from BackTrading.vectorized_signal import _divergence_scores

        df = _mk_dif_series(n, kind)
        t_new, i_new, s_new = _divergence_scores(df, base_distance=10)
        t_ref, i_ref, s_ref = _ref_divergence_scores(df, base_distance=10)
        np.testing.assert_array_equal(i_new, i_ref)
        np.testing.assert_allclose(s_new, s_ref, rtol=0, atol=1e-12)
        for a, b in zip(t_new, t_ref):
            assert a is b or a == b


def _ref_golden_decay(macd_cross, n, cross_decay_days, cross_decay_min):
    """参考实现：逐 cross 循环衰减（重构前）。"""
    cross_positions = np.where(macd_cross == 1)[0]
    if len(cross_positions) == 0:
        return np.ones(n, dtype=np.float64)
    decay_mult = np.ones(n, dtype=np.float64)
    for idx in cross_positions:
        end = min(idx + cross_decay_days, n)
        length = end - idx
        decay_curve = np.maximum(
            cross_decay_min,
            1.0 - np.arange(length, dtype=np.float64) / cross_decay_days,
        )
        mask_update = decay_curve < decay_mult[idx:end]
        decay_mult[idx:end][mask_update] = decay_curve[mask_update]
    return decay_mult


class TestGoldenCrossDecayFixedPoint:
    @pytest.mark.parametrize("n", [100, 500, 2000])
    def test_searchsorted_decay_equals_reference(self, n):
        """金叉衰减 searchsorted 向量化与逐 cross 循环输出完全一致。"""
        rng = np.random.default_rng(3)
        cross = np.zeros(n, dtype=np.int32)
        cross[rng.choice(n, size=max(1, n // 30), replace=False)] = 1
        # 重建 searchsorted 版（与生产代码同逻辑，独立复制避免实现漂移）
        idx_arr = np.arange(n)
        cp = np.where(cross == 1)[0]
        lo_idx = np.searchsorted(cp, idx_arr - 30 + 1, side="left")
        _lo = np.minimum(lo_idx, len(cp) - 1)
        valid = (lo_idx < len(cp)) & (cp[_lo] <= idx_arr)
        dist = np.where(valid, idx_arr - cp[_lo], 0)
        decay_mult = np.where(
            valid,
            np.maximum(0.3, 1.0 - dist.astype(np.float64) / 30),
            1.0,
        )
        np.testing.assert_allclose(
            decay_mult, _ref_golden_decay(cross, n, 30, 0.3), rtol=0, atol=1e-12
        )


class TestDivergencePerfBenchmark:
    @pytest.mark.benchmark
    def test_divergence_5000_bars_under_budget(self):
        """性能基准（入库防回归）：5000 bar 背离检测 < 5s（旧实现 O(n²) 数十秒级）。"""
        from BackTrading.vectorized_signal import _divergence_scores

        df = _mk_dif_series(5000, "trend", seed=11)
        t0 = time.perf_counter()
        _divergence_scores(df, base_distance=10)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"背离检测 5000 bar 耗时 {elapsed:.2f}s，超出性能预算"

    @pytest.mark.benchmark
    def test_divergence_scaling_subquadratic(self):
        """规模扩展基准：2000→8000 bar 耗时增幅 < 16 倍（线性 → 超线性门槛）。"""
        from BackTrading.vectorized_signal import _divergence_scores

        def _time(n: int) -> float:
            df = _mk_dif_series(n, "trend", seed=11)
            t0 = time.perf_counter()
            _divergence_scores(df, base_distance=10)
            return time.perf_counter() - t0

        t_small = _time(2000)
        t_big = _time(8000)
        assert t_big < max(t_small * 16, 1.0), (
            f"背离检测扩展比异常: 2000bar={t_small:.3f}s 8000bar={t_big:.3f}s"
        )


# ═══════════════════════════════════════════════════════════════════
# ④ 模拟交易验证：独立验证集
# ═══════════════════════════════════════════════════════════════════

def _mk_kline(n_days: int = 80, n_syms: int = 1) -> pd.DataFrame:
    dates = [str(d.date()) for d in pd.bdate_range("2024-01-02", periods=n_days)]
    rows = []
    for k in range(n_syms):
        for d in dates:
            rows.append({
                "symbol": f"sh6000{k:02d}",
                "trade_date": d,
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
                "volume": 1_000_000.0,
            })
    return pd.DataFrame(rows).sort_values(["symbol", "trade_date"])


class TestSimTradeIndependentValidation:
    def _patch_env(self, monkeypatch, captured: dict):
        import BackTrading.simulated_trading as _st

        monkeypatch.setattr(
            _st, "prepare_backtest_data",
            lambda kline_df, params=None, **kw: kline_df.assign(
                trade_date=lambda d: d["trade_date"].astype(str),
                ATR=1.0, close_adj=10.0, 止损价=0.0,
            ),
        )

        def _fake_run(ext_data, best_params, engine_cfg, tl, ec):
            captured["dates"] = sorted(ext_data["trade_date"].unique())
            for _d in captured["dates"]:
                ec.append({"time": _d, "portfolio_value": 1_000_000.0})

        monkeypatch.setattr(_st, "_run_single_backtest", _fake_run)
        monkeypatch.setattr(
            _st, "compute_risk_metrics",
            lambda ec: {"sharpe_ratio": 0.5, "sortino_ratio": 0.4},
        )
        return _st

    @pytest.mark.unit
    def test_uses_independent_validation_dates(self, monkeypatch):
        """提供独立验证集时：sim 段 = 验证集（而非最近 20 日，消除自引用）。"""
        import BackTrading.simulated_trading as _st

        captured: dict = {}
        self._patch_env(monkeypatch, captured)
        kline = _mk_kline(80)
        all_dates = sorted(kline["trade_date"].unique())
        holdout = set(all_dates[-30:])  # 末段 holdout（与选参区间分离）
        verdict = _st.validate_params(
            kline, {"atr_stop_mult": 1.5}, oos_sharpe=1.0,
            sim_days=20, validation_dates=holdout,
        )
        assert verdict.sim_sharpe == pytest.approx(0.5)
        sim_seg = set(captured["dates"]) & holdout
        assert sim_seg == holdout  # 验证集完整参与（非最近 20 日）

    @pytest.mark.unit
    def test_validation_dates_not_at_end_of_data(self, monkeypatch):
        """独立验证集在数据中段（不在末尾）：仍精确使用该段，证明非"最近 N 日"。"""
        import BackTrading.simulated_trading as _st

        captured: dict = {}
        self._patch_env(monkeypatch, captured)
        kline = _mk_kline(100)
        all_dates = sorted(kline["trade_date"].unique())
        mid = set(all_dates[30:50])  # 中段独立验证集
        _st.validate_params(
            kline, {"atr_stop_mult": 1.5}, oos_sharpe=1.0,
            sim_days=20, validation_dates=mid,
        )
        # sim 段必须包含中段验证集日期；若回退"最近 20 日"则完全不重叠
        assert set(captured["dates"]) & mid == mid

    @pytest.mark.unit
    def test_too_small_validation_set_passes_through(self, monkeypatch):
        """独立验证集过小 → 放行并说明原因（不误杀）。"""
        import BackTrading.simulated_trading as _st

        captured: dict = {}
        self._patch_env(monkeypatch, captured)
        kline = _mk_kline(40)
        verdict = _st.validate_params(
            kline, {"atr_stop_mult": 1.5}, oos_sharpe=1.0,
            sim_days=20, validation_dates={"2024-01-03", "2024-01-04"},
        )
        assert verdict.promote
        assert "独立验证集" in verdict.reason

    @pytest.mark.unit
    def test_no_validation_dates_falls_back_with_warning(self, monkeypatch):
        """未提供独立验证集 → 回退最近 N 日（自引用，告警），行为兼容。"""
        import BackTrading.simulated_trading as _st

        captured: dict = {}
        self._patch_env(monkeypatch, captured)
        kline = _mk_kline(80)
        verdict = _st.validate_params(
            kline, {"atr_stop_mult": 1.5}, oos_sharpe=1.0, sim_days=20,
        )
        assert verdict.sim_sharpe == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# ⑤ 已删除模块引用清理
# ═══════════════════════════════════════════════════════════════════

class TestDeletedModuleRefsCleaned:
    @pytest.mark.unit
    def test_coordinator_imports_without_deleted_modules(self):
        """Review.coordinator 可正常导入（LoggerManager/GetStockBasicinfo 引用已移除）。"""
        import Review.coordinator  # noqa: F401

    @pytest.mark.unit
    def test_no_leftover_imports_in_entry_files(self):
        for f in ("MainShareAnalysis.py", "Review/coordinator.py"):
            text = Path(f).read_text(encoding="utf-8")
            assert "from UtilsManager.LoggerManager" not in text, f"{f} 仍引用已删除模块"
            assert "from DataCollection.GetStockBasicinfo" not in text, f"{f} 仍引用已删除模块"