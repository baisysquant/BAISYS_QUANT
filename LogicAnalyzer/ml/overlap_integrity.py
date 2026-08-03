"""
机器学习数据重叠泄漏隔离自检模块

1.4 数据重叠泄漏隔离（Data Overlap Purging & Embargo）

业务定义：消除由于特征滚动计算或长期标签导致的「时间跨度重叠」引起的
信息泄露——训练集与验证集/测试集之间必须留出足够的时间真空隔离带。

自检内容：
  1. 特征隔离带（Purging）：若特征包含 N 天的滚动计算窗口，训练集与
     验证集/测试集之间必须留出至少 N 天的时间真空隔离带——否则验证集
     头部样本的特征窗口覆盖训练期价格，模型并未真正做到样本外。
  2. 标签隔离带（Embargo）：若标签为未来 M 天持仓收益，训练集末尾样本
     的标签价格区间 [t+1, t+M] 不得进入验证集头部——训练集结束与验证集
     开始之间必须剔除前 M 天的所有样本（训练末 M 天样本不得入训）。

两者都等价于要求：训练末日 → 验证首日的交易日间隔 ≥ max(N, M)。

模块功能：
  - check_purging_band      —— 特征隔离带检查（N 天）
  - check_embargo_band      —— 标签隔离带检查（M 天）
  - validate_purge_embargo  —— 单折组合检查
  - check_walk_forward_bands —— 多折 Walk-Forward 组合检查
  - run_overlap_check       —— 一站式入口

用法:
    from LogicAnalyzer.ml.overlap_integrity import validate_purge_embargo
    report = validate_purge_embargo(
        train_dates, val_dates, feature_window_days=60, horizon=5
    )
    if not report.passed:
        logger.warning(f"[重叠泄漏] {'；'.join(report.details)}")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from LogicAnalyzer.ml.split_integrity import SplitReport


def _norm_dates(dates: Sequence[str | pd.Timestamp]) -> list[pd.Timestamp]:
    """规范化日期：转为 Timestamp、剔除 NaT、去重、按时间升序。"""
    dt = pd.to_datetime(list(dates), errors="coerce")
    return [pd.Timestamp(d) for d in dt.dropna().drop_duplicates().sort_values()]


def _gap_bdays(t: list[pd.Timestamp], v: list[pd.Timestamp]) -> int:
    return int(np.busday_count(t[-1].date(), v[0].date()))


def check_purging_band(
    train_dates: Sequence[str | pd.Timestamp],
    val_dates: Sequence[str | pd.Timestamp],
    *,
    feature_window_days: int,
    _log: bool = True,
) -> SplitReport:
    """特征隔离带（Purging）：训练/验证间真空隔离带必须 ≥ N 天。

    N = 特征滚动计算窗口天数。验证集头部样本的特征窗口必须整体晚于
    训练期结束（模型不得在训练期价格上做过拟合的样本外承诺）。

    Args:
        train_dates / val_dates: 训练/验证日期序列。
        feature_window_days: 特征最大滚动窗口 N（天）。

    Returns:
        SplitReport: passed = 隔离带 ≥ N。
    """
    check = SplitReport(check_name=f"特征隔离带(Purging N={feature_window_days})", passed=False)
    t = _norm_dates(train_dates)
    v = _norm_dates(val_dates)
    if not t or not v:
        check.details.append("训练/验证日期为空")
        if _log:
            check.log()
        return check

    n = int(feature_window_days)
    if n < 1:
        check.details.append(f"特征窗口 N={n} 非法（必须 ≥ 1）")
        if _log:
            check.log()
        return check
    gap = _gap_bdays(t, v)
    check.n_checked = len(v)
    check.n_violations = 0 if gap >= n else 1
    check.passed = gap >= n
    check.details.append(
        f"特征窗口 N={n} 天：训练末日 {t[-1].date()} → 验证首日 {v[0].date()}，"
        f"隔离带 {gap} 交易日（需 ≥ {n}）"
    )
    if not check.passed:
        check.details.append(f"需再隔 {n - gap} 个交易日的特征真空带")
    if _log:
        check.log()
    return check


def check_embargo_band(
    train_dates: Sequence[str | pd.Timestamp],
    val_dates: Sequence[str | pd.Timestamp],
    *,
    horizon: int,
    _log: bool = True,
) -> SplitReport:
    """标签隔离带（Embargo）：训练结束与验证开始之间的隔离带必须 ≥ M 天。

    M = 标签持有期（未来 M 天收益）。训练集末尾样本的标签价格区间
    [t+1, t+M] 不得进入验证集头部 —— 训练末 M 天样本必须剔除。

    Args:
        train_dates / val_dates: 训练/验证日期序列。
        horizon: 标签持有期 M（天）。

    Returns:
        SplitReport: passed = 隔离带 ≥ M。
    """
    check = SplitReport(check_name=f"标签隔离带(Embargo M={horizon})", passed=False)
    t = _norm_dates(train_dates)
    v = _norm_dates(val_dates)
    if not t or not v:
        check.details.append("训练/验证日期为空")
        if _log:
            check.log()
        return check

    m = int(horizon)
    if m < 1:
        check.details.append(f"标签持有期 M={m} 非法（必须 ≥ 1）")
        if _log:
            check.log()
        return check
    gap = _gap_bdays(t, v)
    check.n_checked = len(v)
    check.n_violations = 0 if gap >= m else 1
    check.passed = gap >= m
    check.details.append(
        f"标签持有期 M={m} 天：训练末日 {t[-1].date()} → 验证首日 {v[0].date()}，"
        f"隔离带 {gap} 交易日（需 ≥ {m}）"
    )
    if not check.passed:
        check.details.append(f"训练末 {m - gap} 天样本的标签价格区间进入验证集，需剔除")
    if _log:
        check.log()
    return check


def validate_purge_embargo(
    train_dates: Sequence[str | pd.Timestamp],
    val_dates: Sequence[str | pd.Timestamp],
    *,
    feature_window_days: int,
    horizon: int = 5,
    _log: bool = True,
) -> SplitReport:
    """单折数据重叠泄漏隔离组合检查（1.4）。

    等价于要求训练末日 → 验证首日的交易日间隔 ≥ max(N, M)。

    Returns:
        SplitReport: passed = 特征隔离带与标签隔离带均满足。
    """
    purging = check_purging_band(
        train_dates, val_dates, feature_window_days=feature_window_days, _log=False
    )
    embargo = check_embargo_band(
        train_dates, val_dates, horizon=horizon, _log=False
    )
    check = SplitReport(check_name="数据重叠泄漏隔离(单折)", passed=False)
    check.details.extend(purging.details)
    check.details.extend(embargo.details)
    check.n_checked = purging.n_checked
    check.n_violations = purging.n_violations + embargo.n_violations
    check.passed = purging.passed and embargo.passed
    if _log:
        check.log()
    return check


def check_walk_forward_bands(
    folds: Sequence[tuple[Sequence[str | pd.Timestamp], Sequence[str | pd.Timestamp]]],
    *,
    feature_window_days: int,
    horizon: int = 5,
) -> SplitReport:
    """多折 Walk-Forward 数据重叠泄漏隔离组合检查（1.4）。

    逐折执行 validate_purge_embargo（特征隔离带 N 天 + 标签隔离带 M 天）。

    Returns:
        SplitReport: passed = 全部折叠隔离带合规。
    """
    check = SplitReport(check_name="Walk-Forward 重叠隔离合规", passed=True)
    if not folds:
        check.details.append("无折叠")
        check.passed = False
        check.log()
        return check

    for k, (tr, va) in enumerate(folds):
        sub = validate_purge_embargo(
            tr, va, feature_window_days=feature_window_days, horizon=horizon, _log=False
        )
        check.n_checked += sub.n_checked
        check.n_violations += sub.n_violations
        if not sub.passed:
            check.passed = False
            check.details.append(
                f"fold{k}: 隔离带不足（{sub.details[0] if sub.details else '未知'}）"
            )
        t = _norm_dates(tr)
        v = _norm_dates(va)
        if t and v:
            check.details.append(
                f"fold{k}: train {t[-1].date()} → val {v[0].date()} 隔离 {_gap_bdays(t, v)} 交易日"
            )

    check.log()
    return check


def run_overlap_check(
    train_dates: Sequence[str | pd.Timestamp],
    val_dates: Sequence[str | pd.Timestamp],
    *,
    feature_window_days: int,
    horizon: int = 5,
    folds: Sequence[tuple[Sequence[str | pd.Timestamp], Sequence[str | pd.Timestamp]]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """一站式数据重叠泄漏隔离自检（1.4 数据重叠泄漏隔离）。

    Args:
        train_dates / val_dates: 单折训练/验证日期（必须提供）。
        feature_window_days: 特征最大滚动窗口天数 N。
        horizon: 标签持有期 M 天。
        folds: 可选多折序列，执行多折组合检查。
        **kwargs: 透传各检查项参数。

    Returns:
        dict: {
            "passed": bool（全部通过）,
            "reports": [SplitReport, ...],
            "summary": pd.DataFrame（Excel 友好）,
        }
    """
    reports = [
        validate_purge_embargo(
            train_dates, val_dates,
            feature_window_days=feature_window_days, horizon=horizon,
            **{k: v for k, v in kwargs.items() if k in {"_log"}},
        )
    ]
    if folds:
        reports.append(
            check_walk_forward_bands(
                folds, feature_window_days=feature_window_days, horizon=horizon,
            )
        )
    return {
        "passed": all(r.passed for r in reports),
        "reports": reports,
        "summary": pd.DataFrame([r.to_dict() for r in reports]),
    }