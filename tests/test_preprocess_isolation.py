"""
1.5 预处理信息隔离 测试

覆盖：
  - check_train_features_invariant：无参数管线 PASS / 全局（含测试期）分布
    拟合必被检出 / 生产特征管线 _compute_feature_matrix 重构一致
  - check_param_fit_within_train：参数拟合区间必须完全落在训练集内
    （起点早于训练 / 终点晚于训练 / 触及验证测试期 均违规）
  - run_preprocess_check 一站式入口
  - 集成：signal_model._PREPROCESS_PARAMS 登记为空、_compute_features 冒烟
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from LogicAnalyzer.ml.preprocess_isolation import (
    PARAM_KINDS,
    PreprocessParam,
    check_param_fit_within_train,
    check_train_features_invariant,
    run_preprocess_check,
)
from LogicAnalyzer.ml.signal_model import (
    _FEATURE_ALL,
    _PREPROCESS_PARAMS,
    _compute_feature_matrix,
    _compute_features,
)


def _make_panel(n_stocks: int = 20, n_days: int = 120, seed: int = 7) -> pd.DataFrame:
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


def _stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """逐行计算的特征（无任何时间维分布参数）。"""
    out = df.copy()
    out["f_a"] = df["close"] / df["low"]
    out["f_b"] = (df["high"] - df["low"]) / df["close"]
    return out


def _leaky_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """违规管线：用「全样本（含测试期）」均值/方差标准化（1.5 绝对禁止）。"""
    out = _stateless_features(df)
    for c in ("f_a", "f_b"):
        mu = out[c].mean()
        sd = out[c].std()
        out[c] = (out[c] - mu) / (sd + 1e-12)
    return out


_FIT_BOUNDARY = "2024-05-14"  # 120 交易日面板的 80% 分界（= 检查的测试期起点）


def _train_only_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """合规管线：仅在固定的训练期历史（边界前）拟合参数，再对全部行被动套用。"""
    out = _stateless_features(df)
    fit = out[out["trade_date"] < _FIT_BOUNDARY]
    for c in ("f_a", "f_b"):
        mu = fit[c].mean()
        sd = fit[c].std()
        out[c] = (out[c] - mu) / (sd + 1e-12)
    return out


def _bdays(n: int, start: str = "2024-01-01") -> list[str]:
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()


# ── 训练行不变性重构验证 ──────────────────────────────────


def test_stateless_features_pass_invariant() -> None:
    panel = _make_panel()
    report = check_train_features_invariant(_stateless_features, panel, ["f_a", "f_b"])
    assert report.passed is True
    assert report.n_violations == 0


def test_global_standardization_detected() -> None:
    """全局（含测试期）均值/方差标准化必被检出：训练行特征随测试行漂移。"""
    panel = _make_panel()
    report = check_train_features_invariant(_leaky_standardize, panel, ["f_a", "f_b"])
    assert report.passed is False
    assert report.n_violations > 0
    assert "漂移" in "；".join(report.details)


def test_train_only_standardization_pass() -> None:
    """仅训练期拟合 + 被动套用的标准化不被误伤。"""
    panel = _make_panel()
    report = check_train_features_invariant(_train_only_standardize, panel, ["f_a", "f_b"])
    assert report.passed is True


def test_production_feature_matrix_invariant() -> None:
    """生产特征管线 _compute_feature_matrix 无时间维分布参数 ⇒ 重构一致。"""
    panel = _make_panel(n_days=140)
    report = check_train_features_invariant(_compute_feature_matrix, panel, _FEATURE_ALL)
    assert report.passed is True, "；".join(report.details)
    assert report.n_violations == 0


def test_too_few_test_dates_skips() -> None:
    panel = _make_panel(n_days=10)
    report = check_train_features_invariant(
        _stateless_features, panel, ["f_a", "f_b"], min_test_rows=5
    )
    assert report.passed is True
    assert "跳过" in "；".join(report.details)


def test_empty_panel_fails() -> None:
    report = check_train_features_invariant(_stateless_features, pd.DataFrame(), ["f_a"])
    assert report.passed is False


# ── 参数来源登记核查 ──────────────────────────────────────


def test_param_fit_within_train_passes() -> None:
    dates = _bdays(60)
    params = [
        PreprocessParam("std_close", "standardize", dates[5], dates[45]),
        PreprocessParam("impute_vol", "impute", dates[10], dates[40], n_cols=3),
    ]
    report = check_param_fit_within_train(params, dates[:50], test_dates=dates[50:])
    assert report.passed is True
    assert report.n_checked == 2


def test_param_fit_end_beyond_train_fails() -> None:
    dates = _bdays(60)
    param = PreprocessParam("std_close", "standardize", dates[5], dates[52])
    report = check_param_fit_within_train([param], dates[:50])
    assert report.passed is False
    assert "晚于训练终点" in "；".join(report.details)


def test_param_fit_start_before_train_fails() -> None:
    dates = _bdays(60)
    earlier = (pd.Timestamp(dates[0]) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    param = PreprocessParam("winsor_ret", "winsorize", earlier, dates[20])
    report = check_param_fit_within_train([param], dates[:50])
    assert report.passed is False
    assert "早于训练起点" in "；".join(report.details)


def test_param_overlapping_test_period_fails() -> None:
    dates = _bdays(60)
    param = PreprocessParam("pca_m", "pca", dates[40], dates[55])  # 覆盖测试期
    report = check_param_fit_within_train([param], dates[:50], test_dates=dates[50:])
    assert report.passed is False
    assert "验证/测试期" in "；".join(report.details)


def test_empty_params_pass() -> None:
    dates = _bdays(60)
    report = check_param_fit_within_train([], dates[:50])
    assert report.passed is True
    assert "无登记" in "；".join(report.details)


def test_empty_train_fails() -> None:
    param = PreprocessParam("std", "standardize", None, None)
    report = check_param_fit_within_train([param], [])
    assert report.passed is False


def test_invalid_param_kind_rejected() -> None:
    with pytest.raises(ValueError):
        PreprocessParam("bad", "quantile")
    assert "standardize" in PARAM_KINDS


def test_param_to_dict() -> None:
    p = PreprocessParam("std_close", "standardize", "2024-01-05", "2024-02-02", n_cols=2)
    d = p.to_dict()
    assert d["参数"] == "std_close"
    assert d["类型"] == "standardize"
    assert d["拟合起点"] == "2024-01-05"
    assert d["列数"] == 2


# ── 一站式入口 ────────────────────────────────────────────


def test_run_preprocess_check_combined() -> None:
    panel = _make_panel()
    dates = _bdays(120)
    params = [PreprocessParam("std", "standardize", dates[5], dates[90])]
    result = run_preprocess_check(
        panel, _stateless_features, ["f_a", "f_b"],
        params=params, train_dates=dates[:96], test_dates=dates[96:],
    )
    assert result["passed"] is True
    assert len(result["reports"]) == 2
    assert isinstance(result["summary"], pd.DataFrame)
    assert len(result["summary"]) == 2


def test_run_preprocess_check_catches_global_fit() -> None:
    panel = _make_panel()
    result = run_preprocess_check(panel, _leaky_standardize, ["f_a", "f_b"])
    assert result["passed"] is False
    assert result["summary"].iloc[0]["通过"] == "FAIL"


# ── 集成：signal_model ────────────────────────────────────


def test_preprocess_params_registry_empty() -> None:
    assert _PREPROCESS_PARAMS == []


def test_compute_features_smoke_no_regression() -> None:
    panel = _make_panel(n_days=140)
    panel["ATR"] = panel["high"] - panel["low"]
    rng = np.random.default_rng(1)
    for c in ("MACD趋势分", "金叉信号分", "DIF斜评分", "量价配合分"):
        panel[c] = rng.uniform(0, 100, len(panel))
    df = _compute_features(panel)
    assert "fwd_5d" in df.columns
    assert len(df) == len(panel)
    assert all(c in df.columns for c in _FEATURE_ALL)
