from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from LogicAnalyzer.ml.label_integrity import (
    ComplianceReport,
    LabelConvention,
    build_forward_return,
    check_feature_leakage,
    check_train_val_purge,
    run_label_integrity_check,
    validate_labels,
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


# ── 合规标签构建 ───────────────────────────────────────────


def test_build_tail_close_formula() -> None:
    panel = _sorted_panel(_make_panel())
    label = build_forward_return(panel, 5, LabelConvention.TAIL_CLOSE)
    expected = (
        panel.groupby("symbol")["close"].shift(-5)
        / panel.groupby("symbol")["close"].shift(-1)
        - 1
    )
    pd.testing.assert_series_equal(
        label.reset_index(drop=True), expected.reset_index(drop=True), check_names=False
    )


def test_build_next_open_formula() -> None:
    panel = _sorted_panel(_make_panel())
    label = build_forward_return(panel, 5, LabelConvention.NEXT_OPEN)
    expected = (
        panel.groupby("symbol")["close"].shift(-5)
        / panel.groupby("symbol")["open"].shift(-1)
        - 1
    )
    pd.testing.assert_series_equal(
        label.reset_index(drop=True), expected.reset_index(drop=True), check_names=False
    )


def test_build_next_open_prefers_vwap() -> None:
    panel = _sorted_panel(_make_panel())
    panel["vwap"] = panel["open"] * 0.99
    label = build_forward_return(
        panel, 5, LabelConvention.NEXT_OPEN, vwap_col="vwap"
    )
    expected = (
        panel.groupby("symbol")["close"].shift(-5)
        / panel.groupby("symbol")["vwap"].shift(-1)
        - 1
    )
    pd.testing.assert_series_equal(
        label.reset_index(drop=True), expected.reset_index(drop=True), check_names=False
    )


def test_build_next_open_missing_open_raises() -> None:
    panel = _make_panel().drop(columns=["open"])
    with pytest.raises(ValueError):
        build_forward_return(panel, 5, LabelConvention.NEXT_OPEN)


def test_build_rejects_bad_horizon() -> None:
    panel = _make_panel()
    with pytest.raises(ValueError):
        build_forward_return(panel, 0, LabelConvention.TAIL_CLOSE)


# ── 标签前瞻性清除验证 ─────────────────────────────────────


def test_validate_passes_compliant_label() -> None:
    panel = _make_panel()
    panel["fwd_5d"] = build_forward_return(panel, 5, LabelConvention.TAIL_CLOSE).values
    report = validate_labels(panel, "fwd_5d", LabelConvention.TAIL_CLOSE, 5)
    assert report.passed
    assert report.n_violations == 0
    assert isinstance(report, ComplianceReport)


def test_validate_detects_today_close_leak() -> None:
    """旧式标签 C_{t+h}/C_t - 1（起点=今日收盘价）必须被检出。"""
    panel = _make_panel()
    leaky = (
        panel.groupby("symbol")["close"].shift(-5) / panel["close"] - 1
    )
    panel["fwd_5d"] = leaky.values
    report = validate_labels(panel, "fwd_5d", LabelConvention.TAIL_CLOSE, 5)
    assert not report.passed
    assert report.n_violations > 0
    assert report.max_deviation > 1e-3
    assert any("违规日期示例" in d for d in report.details)


def test_validate_next_open_rejects_close_to_close() -> None:
    """NEXT_OPEN 约定下，收盘→收盘 涨跌幅标签（含今日收盘→明日收盘段）必须被检出。"""
    panel = _make_panel()
    legacy = panel.groupby("symbol")["close"].shift(-5) / panel["close"] - 1
    panel["fwd_5d"] = legacy.values
    report = validate_labels(panel, "fwd_5d", LabelConvention.NEXT_OPEN, 5)
    assert not report.passed
    assert report.n_violations > 0


def test_validate_missing_tail_rows_are_not_violations() -> None:
    panel = _make_panel()
    label = build_forward_return(panel, 5, LabelConvention.TAIL_CLOSE)
    panel["fwd_5d"] = label.values  # 尾部天然 NaN
    report = validate_labels(panel, "fwd_5d", LabelConvention.TAIL_CLOSE, 5)
    assert report.passed
    assert report.n_violations == 0


def test_validate_missing_label_column() -> None:
    report = validate_labels(_make_panel(), "nope", LabelConvention.TAIL_CLOSE, 5)
    assert not report.passed
    assert any("标签列不存在" in d for d in report.details)


# ── 特征前移泄露检测 ───────────────────────────────────────


def test_feature_leakage_detects_tomorrow_return() -> None:
    """特征 = 明日收益率（未来数据计算的常见泄露形态）必须被检出。"""
    panel = _sorted_panel(_make_panel())
    panel["mom5"] = panel.groupby("symbol")["close"].pct_change(5)
    panel["leaky"] = panel.groupby("symbol")["close"].shift(-1) / panel["close"] - 1
    report = check_feature_leakage(panel, ["mom5", "leaky"])
    assert not report.passed
    assert report.n_violations == 1
    assert any("leaky" in d and "未来量" in d for d in report.details)


def test_feature_leakage_detects_tomorrow_close() -> None:
    """特征 = 明日收盘价（未来价格直接入特征）必须被检出。"""
    panel = _sorted_panel(_make_panel())
    panel["leaky"] = panel.groupby("symbol")["close"].shift(-1)
    report = check_feature_leakage(panel, ["leaky"])
    assert not report.passed
    assert any("明日收盘价" in d for d in report.details)


def test_feature_leakage_clean_features_pass() -> None:
    panel = _sorted_panel(_make_panel())
    panel["mom5"] = panel.groupby("symbol")["close"].pct_change(5)
    panel["vol20"] = panel.groupby("symbol")["close"].pct_change().rolling(20).std()
    report = check_feature_leakage(panel, ["mom5", "vol20"])
    assert report.passed
    assert report.n_violations == 0


# ── 训练/验证标签重叠检查 ──────────────────────────────────


def test_purge_new_convention_sufficient() -> None:
    dates = list(pd.bdate_range("2024-01-01", periods=100))
    train, val = dates[:60], dates[65:]  # 间隔 5 天 = horizon
    report = check_train_val_purge(train, val, 5, LabelConvention.TAIL_CLOSE)
    assert report.passed
    assert report.n_violations == 0


def test_purge_new_convention_insufficient() -> None:
    dates = list(pd.bdate_range("2024-01-01", periods=100))
    train, val = dates[:60], dates[63:]  # 间隔 3 天 < horizon 5
    report = check_train_val_purge(train, val, 5, LabelConvention.NEXT_OPEN)
    assert not report.passed
    assert report.n_violations > 0


def test_purge_legacy_close_to_close_needs_extra_day() -> None:
    dates = list(pd.bdate_range("2024-01-01", periods=100))
    train, val = dates[:60], dates[64:]  # 间隔 5 交易日
    # 旧式约定（base_offset=0，起点=当日收盘）：标签价格区间 [t, t+5]，
    # 5 交易日间隔仍与 1 个价格日重叠（v = t_last + 5 ≤ t_last + h）
    report = check_train_val_purge(train, val, 5, LabelConvention.TAIL_CLOSE, base_offset=0)
    assert not report.passed
    assert report.n_violations == 1


# ── 一站式自检 ─────────────────────────────────────────────


def test_run_label_integrity_check() -> None:
    panel = _make_panel()
    panel["fwd_5d"] = build_forward_return(panel, 5, LabelConvention.TAIL_CLOSE).values
    panel["mom5"] = panel.groupby("symbol")["close"].pct_change(5)
    dates = list(pd.bdate_range("2024-01-01", periods=120))
    result = run_label_integrity_check(
        panel,
        "fwd_5d",
        LabelConvention.TAIL_CLOSE,
        5,
        feature_cols=["mom5"],
        train_dates=dates[:70],
        val_dates=dates[75:],
    )
    assert result["passed"]
    assert len(result["reports"]) == 3
    assert list(result["summary"].columns) == ["检查项", "通过", "样本数", "违规数", "最大偏差", "说明"]
    assert (result["summary"]["通过"] == "PASS").all()


def test_run_label_integrity_check_fails_on_leaky_label() -> None:
    panel = _make_panel()
    panel["fwd_5d"] = panel.groupby("symbol")["close"].shift(-5) / panel["close"] - 1
    result = run_label_integrity_check(panel, "fwd_5d", LabelConvention.TAIL_CLOSE, 5)
    assert not result["passed"]
    assert result["reports"][0].n_violations > 0


# ── signal_model 集成（标签已对齐合规约定） ────────────────


def test_signal_model_label_is_compliant() -> None:
    """_compute_features 生成的 fwd_5d 必须等于 C_{t+5}/C_{t+1}-1 的截面排名。"""
    from LogicAnalyzer.ml.signal_model import _compute_features

    panel = _make_panel()
    out = _compute_features(panel.copy())

    raw = panel.groupby("symbol")["close"].shift(-5) / panel.groupby("symbol")["close"].shift(-1) - 1
    panel["raw"] = raw.values
    expected_rank = panel.groupby("trade_date")["raw"].rank(pct=True)
    expected_rank.index = panel.index

    pd.testing.assert_series_equal(
        out["fwd_5d"].astype(float),
        expected_rank.astype(float),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_signal_model_label_would_have_failed_old_formula() -> None:
    """回归保护：旧式 C_{t+5}/C_t - 1 标签应被自检查出（防止倒退）。"""
    panel = _make_panel()
    old = panel.groupby("symbol")["close"].shift(-5) / panel["close"] - 1
    panel["fwd_5d"] = old.values
    report = validate_labels(panel, "fwd_5d", LabelConvention.TAIL_CLOSE, 5)
    assert not report.passed
