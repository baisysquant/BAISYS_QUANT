"""
1.6 动态重训时间戳锚定 测试

覆盖：
  - check_train_cutoff：训练集截止线（max ≤ 锚点 T）
  - check_label_window_within_anchor（P0-4）：标签价格窗口 ≤ 锚点断言——
    通过 / 尾部泄漏检出（窗口未 purge）/ 边界恰好 = horizon / 空窗口
  - check_first_signal_after_train：重训当天信号废弃
  - run_anchor_time_check 一站式入口（含标签价格窗口检查）
  - 集成回归：signal_model 重训循环窗口尾部 purge 后全部折叠通过
    标签价格窗口锚定；修复前（无 purge）窗口必被检出泄漏
"""

from __future__ import annotations

import pandas as pd

from LogicAnalyzer.ml.anchor_integrity import (
    check_first_signal_after_train,
    check_label_window_within_anchor,
    check_train_cutoff,
    run_anchor_time_check,
)
from LogicAnalyzer.ml.signal_model import (
    _LABEL_HORIZON,
    _PURGE_DAYS,
    _RETRAIN_EVERY,
    _RETRAIN_FREQ,
    _TRAIN_WINDOW,
)

_HORIZON = _LABEL_HORIZON


def _bdays(n: int, start: str = "2024-01-01") -> list[str]:
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()


# ── check_train_cutoff ──────────────────────────────────


def test_train_cutoff_within_anchor_passes() -> None:
    dates = _bdays(120)
    report = check_train_cutoff(dates[:100], dates[105])
    assert report.passed is True
    assert report.n_violations == 0


def test_train_cutoff_future_sample_fails() -> None:
    dates = _bdays(120)
    report = check_train_cutoff(dates[50:110], dates[100])
    assert report.passed is False
    assert report.n_violations == 9


# ── check_label_window_within_anchor（P0-4）───────────────


def test_label_window_purged_passes() -> None:
    """窗口截止 anchor−6：标签最远引用 anchor−1 < anchor，无前视泄漏。"""
    dates = _bdays(130)
    window = dates[60:124]  # 截止 index 123 = anchor(129) − 6
    report = check_label_window_within_anchor(window, dates[129], horizon=_HORIZON)
    assert report.passed is True
    assert report.n_violations == 0


def test_label_window_unpurged_fails() -> None:
    """修复前窗口（截止 anchor−1）尾部标签引用锚点当日及以后价格，必被检出。"""
    dates = _bdays(200)
    leaky = dates[20:119]  # 截止 index 118 = anchor(120) − 2 → 标签引用 T+3
    report = check_label_window_within_anchor(leaky, dates[120], horizon=_HORIZON)
    assert report.passed is False
    assert report.n_violations == 1
    assert "越过锚点" in "；".join(report.details)


def test_label_window_exact_horizon_boundary_passes() -> None:
    """边界：窗口截止 anchor−5（gap = horizon）→ 标签最远引用 = 锚点当日，≤ 锚点合规。"""
    dates = _bdays(130)
    window = dates[60:125]  # 截止 index 124 = anchor(129) − 5
    report = check_label_window_within_anchor(window, dates[129], horizon=_HORIZON)
    assert report.passed is True


def test_label_window_empty_fails() -> None:
    report = check_label_window_within_anchor([], "2024-06-30", horizon=_HORIZON)
    assert report.passed is False


def test_label_window_horizon_shift_changes_verdict() -> None:
    """同一窗口下 horizon 越大越易越锚（标签引用更远价格）。"""
    dates = _bdays(130)
    window = dates[60:124]  # 截止 index 123 = anchor(129) − 6
    assert check_label_window_within_anchor(
        window, dates[129], horizon=5
    ).passed is True
    assert check_label_window_within_anchor(
        window, dates[129], horizon=7
    ).passed is False


# ── check_first_signal_after_train ──────────────────────


def test_first_signal_after_train_next_window_passes() -> None:
    dates = _bdays(60)
    report = check_first_signal_after_train(dates[40], dates[41])
    assert report.passed is True


def test_first_signal_on_train_day_fails() -> None:
    dates = _bdays(60)
    report = check_first_signal_after_train(dates[40], dates[40])
    assert report.passed is False
    assert report.n_violations == 1


# ── run_anchor_time_check 一站式入口 ────────────────────


def test_run_anchor_time_check_includes_label_window() -> None:
    dates = _bdays(130)
    window = dates[60:124]  # 截止 anchor−6，标签窗口 ≤ anchor ✓
    result = run_anchor_time_check(window, dates[129])
    assert result["passed"] is True
    assert len(result["reports"]) == 2
    assert result["summary"].iloc[0]["通过"] == "PASS"
    assert result["summary"].iloc[1]["通过"] == "PASS"


def test_run_anchor_time_check_catches_leaky_window() -> None:
    dates = _bdays(130)
    leaky = dates[60:126]  # 截止 anchor−3 → 标签引用 T+2 > T
    result = run_anchor_time_check(leaky, dates[129])
    assert result["passed"] is False
    assert result["summary"].iloc[1]["通过"] == "FAIL"


# ── 集成回归：signal_model 重训循环窗口（P0-4）─────────────


def _loop_windows(dates: list[str]) -> list[tuple[list[str], str]]:
    """复刻 apply_ml_signal 重训循环的窗口（含 P0-4 尾部 purge）。"""
    windows: list[tuple[list[str], str]] = []
    for i in range(_RETRAIN_FREQ, len(dates), _RETRAIN_EVERY):
        cut_idx = max(0, i - _TRAIN_WINDOW)
        window = dates[cut_idx : max(cut_idx, i - _PURGE_DAYS)]
        if window:
            windows.append((window, dates[i]))
    return windows


def test_retrain_loop_windows_label_window_anchor_pass() -> None:
    """修复后重训循环：全部折叠窗口的标签价格窗口 ≤ 锚点（严格 anchor−1）。"""
    dates = _bdays(400)
    windows = _loop_windows(dates)
    assert len(windows) >= 10
    for window, anchor in windows:
        report = check_label_window_within_anchor(
            window, anchor, horizon=_HORIZON
        )
        assert report.passed is True, f"折叠窗口泄漏: {'；'.join(report.details)}"


def test_retrain_loop_window_max_label_price_below_anchor() -> None:
    """窗口最晚样本 + 标签持有期 严格 < 锚点（修复 = 截止 anchor−6）。"""
    dates = _bdays(400)
    for window, anchor in _loop_windows(dates):
        gap = len(pd.bdate_range(pd.Timestamp(window[-1]), pd.Timestamp(anchor)))
        assert gap >= _HORIZON + 1, f"窗口 {window[-1]} 距锚点 {anchor} 不足 {_HORIZON + 1} 交易日"


def test_unpurged_loop_window_is_leaky() -> None:
    """回归证明：修复前（窗口截止 anchor−1）的折叠窗口必被新断言检出。"""
    dates = _bdays(200)
    for i in range(_RETRAIN_FREQ, len(dates), _RETRAIN_EVERY):
        cut_idx = max(0, i - _TRAIN_WINDOW)
        leaky = dates[cut_idx:i]  # 修复前逻辑：无尾部 purge
        report = check_label_window_within_anchor(leaky, dates[i], horizon=_HORIZON)
        assert report.passed is False, f"修复前窗口 {i} 应被检出泄漏"