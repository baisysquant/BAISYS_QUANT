"""
1.4 数据重叠泄漏隔离 测试

覆盖：
  - check_purging_band：特征隔离带（N 天）
  - check_embargo_band：标签隔离带（M 天）
  - validate_purge_embargo：单折组合
  - check_walk_forward_bands：多折
  - run_overlap_check 一站式入口
  - 回归：旧 purge=5 天切分必被检出隔离带不足（修复前 bug 证明）
  - 集成：signal_model._split_train_val 修复后隔离带 ≥ 60 交易日
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from LogicAnalyzer.ml.overlap_integrity import (
    check_embargo_band,
    check_purging_band,
    check_walk_forward_bands,
    run_overlap_check,
    validate_purge_embargo,
)
from LogicAnalyzer.ml.signal_model import (
    _FEATURE_WINDOW_DAYS,
    _ISOLATION_DAYS,
    _LABEL_HORIZON,
    _PURGE_DAYS,
    _RETRAIN_EVERY,
    _RETRAIN_FREQ,
    _TRAIN_WINDOW,
    _split_train_val,
)

_N = _FEATURE_WINDOW_DAYS   # 特征最大滚动窗口（Purging）
_M = _LABEL_HORIZON         # 标签持有期（Embargo）


def _bdays(n: int, start: str = "2024-01-01") -> list[str]:
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()


def _gap(dates: list[str], i: int, j: int) -> int:
    """dates[i] 与 dates[j] 之间的交易日间隔（bdate 序列下 = j - i - 1）。"""
    return j - i - 1


# ── 特征隔离带（Purging）──────────────────────────────────


def test_purging_band_sufficient_passes() -> None:
    dates = _bdays(160)
    train, val = dates[:40], dates[100:]  # 隔离带 60 天
    assert _gap(dates, 39, 100) == 60
    report = check_purging_band(train, val, feature_window_days=_N)
    assert report.passed is True
    assert report.n_violations == 0


def test_purging_band_insufficient_fails() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[45:]  # 隔离带仅 5 天 < 60
    report = check_purging_band(train, val, feature_window_days=_N)
    assert report.passed is False
    assert report.n_violations == 1
    assert "Purging" in report.check_name


def test_purging_band_zero_window_rejected() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[45:]
    report = check_purging_band(train, val, feature_window_days=0)
    assert report.passed is False


# ── 标签隔离带（Embargo）──────────────────────────────────


def test_embargo_band_sufficient_passes() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[45:]  # 隔离带 5 天 = M
    assert _gap(dates, 39, 45) == 5
    report = check_embargo_band(train, val, horizon=_M)
    assert report.passed is True


def test_embargo_band_insufficient_fails() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[41:50]  # 隔离带 1 天 < 5
    report = check_embargo_band(train, val, horizon=_M)
    assert report.passed is False
    assert report.n_violations == 1
    assert "Embargo" in report.check_name


# ── 单折组合 ──────────────────────────────────────────────


def test_combined_passes_when_gap_ge_max() -> None:
    dates = _bdays(160)
    train, val = dates[:40], dates[100:]
    report = validate_purge_embargo(
        train, val, feature_window_days=_N, horizon=_M
    )
    assert report.passed is True
    assert report.n_violations == 0


def test_combined_fails_when_only_embargo_satisfied() -> None:
    """旧行为（仅 purge 5 天）：满足 Embargo 但违反 Purging。"""
    dates = _bdays(120)
    train, val = dates[:40], dates[45:]  # gap=5 ≥ M=5，但 < N=60
    report = validate_purge_embargo(
        train, val, feature_window_days=_N, horizon=_M
    )
    assert report.passed is False
    assert report.n_violations == 1  # 仅特征隔离带违规


# ── 多折 ──────────────────────────────────────────────────


def _loop_folds(dates: list[str]) -> list[tuple[list[str], list[str]]]:
    """复刻 apply_ml_signal 重训循环的折叠生成逻辑（含 P0-4 尾部 purge）。"""
    folds: list[tuple[list[str], list[str]]] = []
    for i in range(_RETRAIN_FREQ, len(dates), _RETRAIN_EVERY):
        cut_idx = max(0, i - _TRAIN_WINDOW)
        window_dates = dates[cut_idx : max(cut_idx, i - _PURGE_DAYS)]
        train_dates, val_dates = _split_train_val(window_dates)
        if train_dates and val_dates:
            folds.append((train_dates, val_dates))
    return folds


def test_walk_forward_bands_pass() -> None:
    dates = _bdays(400)
    folds = _loop_folds(dates)
    assert len(folds) >= 10
    report = check_walk_forward_bands(folds, feature_window_days=_N, horizon=_M)
    assert report.passed is True
    assert report.n_violations == 0


def test_walk_forward_bands_catches_insufficient_gap() -> None:
    dates = _bdays(300)
    folds = [
        (dates[:40], dates[100:120]),                 # gap=60 ✓
        (dates[:140], dates[150:180]),                # gap=9 ✗（< 60）
    ]
    report = check_walk_forward_bands(folds, feature_window_days=_N, horizon=_M)
    assert report.passed is False
    assert report.n_violations >= 1
    assert "隔离带不足" in "；".join(report.details)


def test_walk_forward_bands_empty_fails() -> None:
    report = check_walk_forward_bands([], feature_window_days=_N, horizon=_M)
    assert report.passed is False


# ── 一站式入口 ────────────────────────────────────────────


def test_run_overlap_check_single_fold() -> None:
    dates = _bdays(160)
    result = run_overlap_check(
        dates[:40], dates[100:], feature_window_days=_N, horizon=_M
    )
    assert result["passed"] is True
    assert len(result["reports"]) == 1
    assert isinstance(result["summary"], pd.DataFrame)
    assert result["summary"].iloc[0]["通过"] == "PASS"


def test_run_overlap_check_with_folds() -> None:
    dates = _bdays(300)
    folds = [(dates[:40], dates[100:120]), (dates[:140], dates[220:240])]
    result = run_overlap_check(
        dates[:40], dates[100:120],
        feature_window_days=_N, horizon=_M, folds=folds,
    )
    assert result["passed"] is True
    assert len(result["reports"]) == 2


def test_run_overlap_check_catches_violation() -> None:
    dates = _bdays(120)
    result = run_overlap_check(
        dates[:40], dates[45:], feature_window_days=_N, horizon=_M
    )
    assert result["passed"] is False


# ── 回归：修复前旧切分必被检出 ─────────────────────────────


def test_regression_old_purge_5_split_fails_purging() -> None:
    """修复前：训练窗口尾端仅 purge 5 天 ⇒ 特征隔离带 5 < 60，必 FAIL。"""
    window = _bdays(180)
    split = int(len(window) * 0.8)
    old_train = window[:split - 5]
    old_val = window[split:]
    assert _gap(window, split - 6, split) == 5
    report = check_purging_band(old_train, old_val, feature_window_days=_N)
    assert report.passed is False
    combined = validate_purge_embargo(
        old_train, old_val, feature_window_days=_N, horizon=_M
    )
    assert combined.passed is False


# ── 集成：修复后的 _split_train_val ────────────────────────


def test_split_train_val_isolates_60_days() -> None:
    window = _bdays(180)
    train, val = _split_train_val(window)
    t_dates = [pd.Timestamp(d) for d in train]
    v_dates = [pd.Timestamp(d) for d in val]
    gap = int(np.busday_count(t_dates[-1].date(), v_dates[0].date()))
    assert gap >= _ISOLATION_DAYS


def test_split_train_val_train_and_val_cover_window() -> None:
    window = _bdays(180)
    train, val = _split_train_val(window)
    assert len(train) + len(val) <= len(window)
    assert len(val) > 0
    assert max(train) < min(val)


def test_retrain_loop_folds_isolated() -> None:
    dates = _bdays(400)
    folds = _loop_folds(dates)
    for tr, va in folds:
        sub = validate_purge_embargo(tr, va, feature_window_days=_N, horizon=_M)
        assert sub.passed is True, f"fold 隔离带不足: {'；'.join(sub.details)}"
    report = check_walk_forward_bands(folds, feature_window_days=_N, horizon=_M)
    assert report.passed is True


def test_constants_consistent() -> None:
    assert _FEATURE_WINDOW_DAYS >= 60
    assert max(_FEATURE_WINDOW_DAYS, 5) == _ISOLATION_DAYS
    assert _ISOLATION_DAYS >= _LABEL_HORIZON
