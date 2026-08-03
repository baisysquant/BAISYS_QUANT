"""
1.3 样本切平时序合规 测试

覆盖：
  - validate_time_split：合规切分 / 随机打乱 / 时间交错（未来样本）/
    日期重叠 / purge 不足 / 空输入
  - check_walk_forward_folds：滚动窗口合规 / 扩展窗口合规 /
    窗口回退回归 / 空折叠
  - run_split_integrity_check 一站式入口
  - signal_model._split_train_val 与真实重训循环折叠的集成回归
"""

from __future__ import annotations

import pandas as pd

from LogicAnalyzer.ml.signal_model import (
    _RETRAIN_EVERY,
    _RETRAIN_FREQ,
    _TRAIN_WINDOW,
    _split_train_val,
)
from LogicAnalyzer.ml.split_integrity import (
    SplitReport,
    check_walk_forward_folds,
    run_split_integrity_check,
    validate_time_split,
)

_HORIZON = 5
_PURGE = 5


def _bdays(n: int, start: str = "2024-01-01") -> list[str]:
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()


# ── 单折：validate_time_split ──────────────────────────────


def test_single_fold_clean_split_passes() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[45:]
    report = validate_time_split(train, val, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is True
    assert report.n_violations == 0
    assert report.n_checked == len(val)


def test_single_fold_default_purge_equals_horizon() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[45:]
    report = validate_time_split(train, val, horizon=5)
    assert report.passed is True


def test_shuffled_val_dates_detected() -> None:
    dates = _bdays(120)
    train = dates[:40]
    val = dates[38:]  # 前 2 天与训练集时间交错（打乱残留）
    report = validate_time_split(train, val, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False
    assert report.n_violations >= 2
    assert "随机打乱禁用" in "；".join(report.details)


def test_future_sample_into_train_detected() -> None:
    """未来样本进入训练集：验证集整段早于训练集（反向切分）。"""
    dates = _bdays(120)
    train, val = dates[50:], dates[:30]
    report = validate_time_split(train, val, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False
    assert report.n_violations >= len(val)
    assert "随机打乱禁用" in "；".join(report.details)


def test_overlapping_dates_detected() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[20:45]
    report = validate_time_split(train, val, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False
    assert report.n_violations >= 1


def test_insufficient_purge_detected() -> None:
    """purge 不足：训练末日与验证首日间隔 < 标签持有期。"""
    dates = _bdays(120)
    train, val = dates[:40], dates[41:50]
    report = validate_time_split(train, val, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False
    assert "purge 不足" in "；".join(report.details)
    assert report.n_violations == 1


def test_sufficient_purge_with_zero_ordering_violations() -> None:
    dates = _bdays(120)
    train, val = dates[:40], dates[45:]
    report = validate_time_split(train, val, horizon=_HORIZON, purge_days=_PURGE)
    assert report.n_violations == 0
    assert report.passed is True


def test_empty_inputs_fail() -> None:
    dates = _bdays(30)
    report = validate_time_split([], dates, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False
    assert "为空" in "；".join(report.details)
    report2 = validate_time_split(dates, [], horizon=_HORIZON, purge_days=_PURGE)
    assert report2.passed is False


def test_duplicate_and_unsorted_dates_normalized() -> None:
    dates = _bdays(120)
    train = [dates[30], dates[10], dates[10], dates[10]]
    val = [dates[45], dates[45]]
    report = validate_time_split(train, val, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is True
    assert report.n_checked == 1


def test_report_to_dict_and_dataframe() -> None:
    dates = _bdays(60)
    report = validate_time_split(dates[:20], dates[25:], horizon=_HORIZON, purge_days=_PURGE)
    d = report.to_dict()
    assert d["通过"] == "PASS"
    assert d["样本数"] == 35
    df = report.to_dataframe()
    assert list(df.columns) == ["检查项", "通过", "样本数", "违规数", "说明"]


# ── 多折：check_walk_forward_folds ─────────────────────────


def _clean_folds(n: int = 200, gap: int = 10) -> list[tuple[list[str], list[str]]]:
    """构造严格推进的滚动窗口折叠（train/val 间隔 gap 个交易日）。"""
    dates = _bdays(n * 2)
    return [
        (dates[i : i + 60], dates[i + 60 + gap : i + 60 + gap + 30])
        for i in range(0, 200, 40)
    ]


def test_walk_forward_rolling_windows_pass() -> None:
    folds = _clean_folds()
    report = check_walk_forward_folds(folds, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is True
    assert report.n_violations == 0


def test_walk_forward_expanding_windows_pass() -> None:
    """扩展窗口：训练起点固定为最早日期，验证窗口仍严格推进。"""
    dates = _bdays(300)
    folds = [
        (dates[:80], dates[90:110]),
        (dates[:120], dates[130:150]),
        (dates[:160], dates[170:190]),
    ]
    report = check_walk_forward_folds(folds, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is True


def test_window_regression_detected() -> None:
    """窗口回退：后一折验证终点早于前一折验证终点。"""
    dates = _bdays(300)
    folds = [
        (dates[:80], dates[90:140]),     # val end = dates[139]
        (dates[:100], dates[105:137]),   # val end = dates[136] < 139 → 回退
    ]
    report = check_walk_forward_folds(folds, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False
    assert "窗口回退" in "；".join(report.details)
    assert report.n_violations >= 1


def test_random_cv_folds_detected() -> None:
    """随机折叠（非时序）：折叠乱序，后续折叠整体早于前一折。"""
    dates = _bdays(200)
    folds = [
        (dates[60:120], dates[120:140]),
        (dates[:40], dates[40:60]),  # 整体早于 fold0
    ]
    report = check_walk_forward_folds(folds, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False


def test_empty_folds_fail() -> None:
    report = check_walk_forward_folds([], horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is False


# ── 一站式入口 ─────────────────────────────────────────────


def test_run_split_integrity_check_single_fold() -> None:
    dates = _bdays(120)
    result = run_split_integrity_check(
        dates[:40], dates[45:], horizon=_HORIZON, purge_days=_PURGE
    )
    assert result["passed"] is True
    assert len(result["reports"]) == 1
    assert isinstance(result["summary"], pd.DataFrame)
    assert result["summary"].iloc[0]["通过"] == "PASS"


def test_run_split_integrity_check_with_folds() -> None:
    dates = _bdays(300)
    result = run_split_integrity_check(
        dates[:80], dates[90:110],
        horizon=_HORIZON, purge_days=_PURGE,
        folds=[(dates[:80], dates[90:110]), (dates[:120], dates[130:150])],
    )
    assert result["passed"] is True
    assert len(result["reports"]) == 2


def test_run_split_integrity_check_catches_violation() -> None:
    dates = _bdays(120)
    result = run_split_integrity_check(
        dates[:40], dates[38:50], horizon=_HORIZON, purge_days=_PURGE
    )
    assert result["passed"] is False


# ── 集成回归：signal_model 重训循环折叠 ─────────────────────


def _loop_folds(dates: list[str]) -> list[tuple[list[str], list[str]]]:
    """复刻 apply_ml_signal 重训循环的折叠生成逻辑（使用模块常量）。"""
    folds: list[tuple[list[str], list[str]]] = []
    for i in range(_RETRAIN_FREQ, len(dates), _RETRAIN_EVERY):
        cut_idx = max(0, i - _TRAIN_WINDOW)
        window_dates = dates[cut_idx:i]
        train_dates, val_dates = _split_train_val(window_dates)
        if train_dates and val_dates:
            folds.append((train_dates, val_dates))
    return folds


def test_split_train_val_chronological_purge() -> None:
    window = _bdays(120)
    train, val = _split_train_val(window)
    assert len(train) + _PURGE + len(val) <= len(window)
    assert max(train) < min(val)
    assert pd.Timestamp(train[-1]) < pd.Timestamp(val[0])


def test_retrain_loop_folds_compliant() -> None:
    dates = _bdays(400)
    folds = _loop_folds(dates)
    assert len(folds) >= 10
    for tr, va in folds:
        sub = validate_time_split(tr, va, horizon=_HORIZON, purge_days=_PURGE)
        assert sub.passed is True, f"fold 不合规: {'；'.join(sub.details)}"
    report = check_walk_forward_folds(folds, horizon=_HORIZON, purge_days=_PURGE)
    assert report.passed is True


def test_retrain_loop_folds_advance_chronologically() -> None:
    dates = _bdays(400)
    folds = _loop_folds(dates)
    val_ends = [pd.Timestamp(va[-1]) for _, va in folds]
    assert val_ends == sorted(val_ends)
    assert len(set(val_ends)) == len(val_ends)
    for i in range(1, len(folds)):
        assert pd.Timestamp(folds[i][1][0]) > pd.Timestamp(folds[i - 1][1][0])


def test_split_report_defaults() -> None:
    report = SplitReport(check_name="x", passed=False)
    assert report.n_checked == 0
    assert report.n_violations == 0
    assert report.details == []
    assert report.to_dict()["通过"] == "FAIL"
