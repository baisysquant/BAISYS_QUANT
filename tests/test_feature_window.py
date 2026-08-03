from __future__ import annotations

import numpy as np
import pandas as pd

from LogicAnalyzer.ml.feature_window import (
    FeatureWindowReport,
    check_feature_window,
    check_no_global_statistics,
    run_feature_window_check,
)
from LogicAnalyzer.ml.signal_model import (
    _FEATURE_ALL,
    _compute_feature_matrix,
    _compute_features,
    _compute_features_raw,
)


def _make_panel(n_stocks: int = 20, n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """构造合成面板：close/open/high/low/volume/amount。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days).strftime("%Y-%m-%d")
    symbols = [f"600{100 + i:03d}" for i in range(n_stocks)]
    rows = []
    for sym in symbols:
        close = 10.0 + np.cumsum(rng.normal(0, 0.1, n_days))
        prev = np.concatenate([[close[0]], close[:-1]])
        opn = prev * (1 + rng.normal(0, 0.002, n_days))
        rows.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "trade_date": dates,
                    "close": close,
                    "open": opn,
                    "high": np.maximum(close, opn) * 1.005,
                    "low": np.minimum(close, opn) * 0.995,
                    "volume": rng.integers(1_000_000, 5_000_000, n_days),
                    "amount": rng.integers(1e8, 1e9, n_days),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _sorted_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _shifted_group(panel: pd.DataFrame, col: str, f) -> pd.Series:
    """T-1 闭合参考实现：f(逐 symbol 序列) 后整体后移一日。"""
    out = panel.copy()
    out[col] = out.groupby("symbol")[col].transform(f)
    return out.groupby("symbol")[col].shift(1)


# ── T-1 窗口闭合检验（扰动重构） ───────────────────────────


def test_closed_features_pass_perturbation_check() -> None:
    panel = _sorted_panel(_make_panel())
    report = check_feature_window(_compute_feature_matrix, panel, _FEATURE_ALL)
    assert isinstance(report, FeatureWindowReport)
    assert report.passed
    assert report.n_violations == 0


def test_today_close_leak_detected_in_raw_features() -> None:
    """回归保护：未闭合（旧式）特征（ret_1d = C_t/C_{t-1} - 1 等）必须被检出。"""
    panel = _sorted_panel(_make_panel())
    report = check_feature_window(_compute_features_raw, panel, _FEATURE_ALL)
    assert not report.passed
    assert report.n_violations > 0
    assert any("ret_1d" in d for d in report.details)
    assert any("hl_ratio" in d for d in report.details)


def test_perturbation_check_skips_missing_feature_columns() -> None:
    panel = _sorted_panel(_make_panel())
    report = check_feature_window(_compute_feature_matrix, panel, ["不存在的列"])
    assert report.passed


# ── 全时段统计量检测（极端行追加） ─────────────────────────


def test_global_statistic_detected() -> None:
    """依赖全时段统计量（full-sample mean）的特征必须被检出。"""
    panel = _sorted_panel(_make_panel())

    def leaky_global(df: pd.DataFrame) -> pd.DataFrame:
        out = _compute_feature_matrix(df)
        out["global_ret"] = out["ret_1d"] / out["ret_1d"].mean()
        return out

    report = check_no_global_statistics(leaky_global, panel, ["global_ret"])
    assert not report.passed
    assert report.n_violations > 0
    assert any("global_ret" in d for d in report.details)

    clean = check_no_global_statistics(leaky_global, panel, ["ret_1d"])
    assert clean.passed


def test_append_robustness_closed_features_pass() -> None:
    panel = _sorted_panel(_make_panel())
    report = check_no_global_statistics(_compute_feature_matrix, panel, _FEATURE_ALL)
    assert report.passed
    assert report.n_violations == 0


# ── 一站式自检 ─────────────────────────────────────────────


def test_run_feature_window_check() -> None:
    panel = _sorted_panel(_make_panel())
    result = run_feature_window_check(_compute_feature_matrix, panel, _FEATURE_ALL)
    assert result["passed"]
    assert len(result["reports"]) == 2
    assert list(result["summary"].columns) == ["检查项", "通过", "样本数", "违规数", "最大偏差", "说明"]
    assert (result["summary"]["通过"] == "PASS").all()


def test_run_feature_window_check_fails_on_raw_features() -> None:
    panel = _sorted_panel(_make_panel())
    result = run_feature_window_check(_compute_features_raw, panel, _FEATURE_ALL)
    assert not result["passed"]
    assert result["reports"][0].n_violations > 0


# ── signal_model 集成（特征已 T-1 窗口闭合） ────────────────


def test_signal_model_features_are_closed() -> None:
    """闭合后特征 T 行 = 以 T-1 及以前数据计算的原始值（抽查代表性特征）。"""
    panel = _sorted_panel(_make_panel())
    out = _compute_feature_matrix(panel.copy())

    pd.testing.assert_series_equal(
        out["ret_1d"].astype(float),
        _shifted_group(panel, "close", lambda s: s / s.shift(1) - 1).astype(float),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        out["ret_20d"].astype(float),
        _shifted_group(panel, "close", lambda s: s / s.shift(20) - 1).astype(float),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        out["amt_20d"].astype(float),
        _shifted_group(panel, "amount", lambda s: s.rolling(20).mean()).astype(float),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        out["price_pos_ma20"].astype(float),
        _shifted_group(panel, "close", lambda s: s / s.rolling(20).mean() - 1).astype(float),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )

    panel2 = panel.copy()
    panel2["_hl"] = (panel2["high"] - panel2["low"]) / (panel2["close"] + 1e-8)
    pd.testing.assert_series_equal(
        out["hl_ratio"].astype(float),
        _shifted_group(panel2, "_hl", lambda s: s).astype(float),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_signal_model_end_to_end_window_check_passes() -> None:
    """_compute_features 端到端：输出特征通过窗口自检（运行期自检不告警）。"""
    panel = _sorted_panel(_make_panel())
    _compute_features(panel.copy())
    result = run_feature_window_check(_compute_feature_matrix, panel, _FEATURE_ALL)
    assert result["passed"]
