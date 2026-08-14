"""
机器学习动态重训时间戳锚定自检模块

1.6 动态重训时间戳锚定（Anchor-Time Verification）

业务定义：应对用户随时调参、强制即时回测的不确定习惯，锁死模型可见的
数据边界——任何一次训练都必须以「当前锚点」为可见性上限，并且当天刚完成
训练的模型绝不允许去撮合当天的成交价。

自检内容：
  1. 动态训练集截止线（check_train_cutoff）：训练集（特征 X 与标签 Y）样本
     的「最大时间戳」必须 ≤ 锚点 T，绝对禁止任何时间戳 > T 的样本进入模型
     的 Fit 逻辑；验证集同样不得早于训练集（另见 1.3）。
  2. 重训当天信号废弃（check_first_signal_after_train）：若用户在 T 日（盘中）
     强制推出重训，因训练耗时，该模型的首次信号只能落在 T 之后的下一个有效
     交易窗口（不得 = T 日）；T 日信号只能由 ≥ 上一代、训练时点在 T 之前的
     旧模型产出。
  3. 标签价格窗口锚定（check_label_window_within_anchor）：窗口（训练+验证）
     内任意样本的标签 fwd_5d = C_{t+5}/C_{t+1}−1 最远引用价格 C_{t+5} 必须
     ≤ 锚点 T。窗口尾部样本若引用 T 及以后的价格，将驱动 XGBoost 早停、
     验证集 Rank-IC 与显著性门控（决定 ML 是否覆写评分）——模型选择被未来
     价格影响（前视泄漏）。训练窗口必须尾部 purge 标签持有期（截止 anchor−6，
     全部标签价格窗口 ≤ anchor−1）。

模块功能：
  - check_train_cutoff            —— 训练集截止线（max ≤ 锚点 T）
  - check_label_window_within_anchor —— 标签价格窗口 ≤ 锚点（P0-4 前视泄漏断言）
  - check_first_signal_after_train —— 首个信号必须位于重训锚点之后
  - run_anchor_time_check         —— 一站式入口

用法：
    from LogicAnalyzer.ml.anchor_integrity import (
        check_train_cutoff, check_label_window_within_anchor,
        check_first_signal_after_train, run_anchor_time_check,
    )
    c1 = check_train_cutoff(train_dates, anchor_date=T)
    c2 = check_label_window_within_anchor(window_dates, anchor_date=T)
    c3 = check_first_signal_after_train(anchor_date=T, first_signal_date=T1)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from LogicAnalyzer.ml.split_integrity import SplitReport


def _norm_dates(dates: Sequence[str | pd.Timestamp]) -> list[pd.Timestamp]:
    """规范化日期：升序、去重、剔除 NaT。"""
    dt = pd.to_datetime(list(dates), errors="coerce")
    return [pd.Timestamp(d) for d in dt.dropna().drop_duplicates().sort_values()]


def check_train_cutoff(
    train_dates: Sequence[str | pd.Timestamp],
    anchor_date: str | pd.Timestamp,
    *,
    _log: bool = True,
) -> SplitReport:
    """动态训练集截止线：训练样本最大时间戳必须 ≤ 锚点 T。

    验证「当前时刻 T」强制重训时，fit 输入流里没有任何时间戳 > T 的样本
    （特征 X 与标签 Y 的样本行时间戳一律 ≤ T）。

    Args:
        train_dates: 参与本次 Fit 的训练样本日期序列。
        anchor_date: 锚点 T（重训触发时间 / 强制回测起始时间）。
        _log: 是否输出日志。

    Returns:
        SplitReport: passed = 训练集最大时间戳 ≤ T。
    """
    check = SplitReport(check_name=f"训练集截止线(≤{pd.Timestamp(anchor_date).date()})", passed=True)
    t = _norm_dates(train_dates)
    if not t:
        check.details.append("训练集日期为空")
        check.passed = False
        if _log:
            check.log()
        return check

    a = pd.Timestamp(anchor_date)
    late = [d for d in t if d > a]
    check.n_checked = len(t)
    check.n_violations = len(late)
    check.passed = len(late) == 0
    check.details.append(
        f"训练时间戳范围 {t[0].date()} ~ {t[-1].date()}，锚点 T={a.date()}"
    )
    if check.passed:
        check.details.append(f"max(训练样本)={t[-1].date()} ≤ T，无越界样本")
    else:
        check.details.append(
            f"越界样本 {len(late)} 个（时间戳 > T，绝对禁止进入 Fit）: "
            + ", ".join(d.date().isoformat() for d in late[:5])
        )
    if _log:
        check.log()
    return check


def check_label_window_within_anchor(
    window_dates: Sequence[str | pd.Timestamp],
    anchor_date: str | pd.Timestamp,
    *,
    horizon: int = 5,
    _log: bool = True,
) -> SplitReport:
    """标签价格窗口锚定：窗口内样本标签所引用的最远价格必须 ≤ 锚点 T。

    P0-4 前视泄漏断言：标签 fwd_5d = C_{t+5} / C_{t+1} - 1 的行 t 最远引用
    价格 C_{t+5}。窗口内最晚样本 t_max 若满足 t_max + horizon > T，则该行
    标签引用锚点当日及以后的价格——这些尾部行落入验证集，驱动 XGBoost 早停、
    验证集 Rank-IC 与显著性门控（决定 ML 是否覆写评分），模型选择被未来价格
    影响。修复要求：训练窗口尾部 purge 标签持有期（窗口截止 anchor−6 时
    标签最远引用 anchor−1 < anchor）。

    Args:
        window_dates: 参与本次 Fit 的窗口（训练+验证）样本日期序列。
        anchor_date: 锚点 T（重训触发时间）。
        horizon: 标签持有期（fwd_5d 的 h，默认 5）。
        _log: 是否输出日志。

    Returns:
        SplitReport: passed = max(窗口) + horizon ≤ T（标签价格窗口不越过锚点）。
    """
    t = _norm_dates(window_dates)
    a = pd.Timestamp(anchor_date)
    check = SplitReport(
        check_name=f"标签价格窗口锚定(≤{a.date()})", passed=True
    )
    if not t:
        check.passed = False
        check.details.append("窗口日期为空")
        if _log:
            check.log()
        return check

    t_max = t[-1]
    gap = int(np.busday_count(t_max.date(), a.date()))  # [t_max, T) 内交易日数
    check.n_checked = 1
    check.passed = gap >= horizon  # t_max + horizon ≤ T ⟺ 标签最远引用价格不越过锚点
    check.n_violations = 0 if check.passed else 1
    check.details.append(
        f"窗口最晚样本 {t_max.date()}（标签最远引用价格 = t+{horizon}），"
        f"距锚点 {a.date()} 相隔 {gap} 交易日"
    )
    if check.passed:
        check.details.append(
            f"标签价格窗口（≤ {t_max.date()} + {horizon}）不越过锚点 {a.date()}，"
            "无前视泄漏"
        )
    else:
        check.details.append(
            f"标签价格窗口越过锚点：窗口尾部 {horizon - gap} 行标签引用锚点当日及"
            "以后价格，将驱动早停/IC/显著性门控——必须将训练窗口尾部 purge "
            "标签持有期（截止 anchor−6，使标签价格窗口 ≤ anchor−1）"
        )
    if _log:
        check.log()
    return check


def check_first_signal_after_train(
    anchor_date: str | pd.Timestamp,
    first_signal_date: str | pd.Timestamp,
    *,
    _log: bool = True,
) -> SplitReport:
    """重训当天信号废弃：该模型的首个信号必须位于重训锚点之后。

    模型在锚点 T 日完成训练（训练耗时），其最新信号只能在 T 之后的下一个
    有效交易窗口执行；T 当日的信号只能来自时点早于 T 的更早模型，本模型
    严禁产出 T 日信号（不得撮合 T 日已开盘价/VWAP）。

    Args:
        anchor_date: 模型重训锚点 T。
        first_signal_date: 该模型首次被应用的信号日期。

    Returns:
        SplitReport: passed = first_signal_date 严格晚于 anchor_date。
    """
    a = pd.Timestamp(anchor_date)
    s = pd.Timestamp(first_signal_date)
    gap = int(np.busday_count(a.date(), s.date()))
    check = SplitReport(
        check_name=f"重训当天信号废弃(锚点 {a.date()})",
        passed=s > a and gap >= 1,
    )
    check.n_checked = 1
    check.n_violations = 0 if check.passed else 1
    check.details.append(
        f"训练锚点 {a.date()} → 首个信号 {s.date()}，相隔 {max(gap, 0)} 交易日"
    )
    if check.passed:
        check.details.append("首个信号位于下一交易窗口及之后，合规")
    else:
        check.details.append(
            "重训当日信号废弃违规：该模型不得产出锚点日信号"
            "（T 日信号只可由更早的模型产出）"
        )
    if _log:
        check.log()
    return check


def run_anchor_time_check(
    train_dates: Sequence[str | pd.Timestamp],
    anchor_date: str | pd.Timestamp,
    first_signal_date: str | pd.Timestamp | None = None,
    horizon: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """一站式动态重训时间戳锚定自检（1.6）.

    Args:
        train_dates: 参与本次 Fit 的训练样本日期。
        anchor_date: 重训锚点 T.
        first_signal_date: 该模型的首个信号日期（可选；提供后执行当天废弃检查）。
        horizon: 标签持有期（标签价格窗口锚定检查用）。
        **kwargs: 透传各检查项参数。

    Returns:
        dict: {"passed", "reports", "summary"}
    """
    reports = [check_train_cutoff(train_dates, anchor_date)]
    reports.append(
        check_label_window_within_anchor(train_dates, anchor_date, horizon=horizon)
    )
    if first_signal_date is not None:
        reports.append(
            check_first_signal_after_train(anchor_date, first_signal_date)
        )
    return {
        "passed": all(r.passed for r in reports),
        "reports": reports,
        "summary": pd.DataFrame([r.to_dict() for r in reports]),
    }