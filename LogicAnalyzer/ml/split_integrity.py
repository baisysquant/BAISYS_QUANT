"""
机器学习样本切分合规自检模块

1.3 样本切平时序合规（Temporal Split Integrity）

业务定义：防止「未来样本」进入训练集影响「过去样本」的预测——训练/验证/
测试划分必须严格沿时间轴线性切分（先过去后未来），任何验证集时间段必须
严格在对应训练集时间段之后。

自检内容：
  1. 随机打乱禁用：train/val/test 划分关闭任何随机 Shuffle——验证集中不得
     出现早于训练集的日期（时间交错 = 打乱残留，未来样本进入训练集）。
  2. 时序线性切分：训练集与验证集日期无重叠，训练集全部日期严格早于
     验证集全部日期（先过去后未来）。
  3. 标签窗口 purge：训练末样本与验证首样本的标签价格区间不得重叠——
     训练末日与验证首日的交易日间隔必须 ≥ purge_days（默认 = 标签持有期）。
  4. Walk-Forward 交叉验证：多折必须为滚动窗口或扩展窗口、逐折严格后继
     推进——后一折验证终点必须晚于前一折验证终点（检出窗口回退/随机折叠）。

模块功能：
  - validate_time_split        —— 单折检查（打乱 / 重叠 / purge）
  - check_walk_forward_folds   —— 多折 Walk-Forward 合规检查
  - run_split_integrity_check  —— 一站式入口
  - SplitReport                —— PASS/FAIL 自检报告（可转 Excel）

用法:
    from LogicAnalyzer.ml.split_integrity import (
        validate_time_split, check_walk_forward_folds, run_split_integrity_check,
    )
    report = validate_time_split(train_dates, val_dates, horizon=5, purge_days=5)
    if not report.passed:
        logger.warning(f"[切分合规] 单折切分违规: {'；'.join(report.details)}")
    result = run_split_integrity_check(train_dates, val_dates, folds=[(tr, va), ...])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SplitReport:
    """样本切分合规自检报告。"""

    check_name: str
    passed: bool
    n_checked: int = 0
    n_violations: int = 0
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "检查项": self.check_name,
            "通过": "PASS" if self.passed else "FAIL",
            "样本数": self.n_checked,
            "违规数": self.n_violations,
            "说明": "；".join(self.details[:6]) or "",
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def log(self) -> None:
        if self.passed:
            logger.info(f"[切分合规] {self.check_name} PASS（样本 {self.n_checked}）")
        else:
            logger.warning(
                f"[切分合规] {self.check_name} FAIL（违规 {self.n_violations}/{self.n_checked}）"
            )


def _norm_dates(dates: Sequence[str | pd.Timestamp]) -> list[pd.Timestamp]:
    """规范化日期：转为 Timestamp、剔除 NaT、去重、按时间升序。"""
    dt = pd.to_datetime(list(dates), errors="coerce")
    return [pd.Timestamp(d) for d in dt.dropna().drop_duplicates().sort_values()]


# ── 单折时序切分检查 ───────────────────────────────────────

def validate_time_split(
    train_dates: Sequence[str | pd.Timestamp],
    val_dates: Sequence[str | pd.Timestamp],
    *,
    horizon: int = 5,
    purge_days: int | None = None,
    _log: bool = True,
) -> SplitReport:
    """单折时序切分合规检查（1.3 样本切平时序合规）。

    检查：
      1. 随机打乱禁用：验证集中任何日期早于或等于训练末日 ⇒ 时间交错
         （未来样本进入训练集 / 打乱残留），计为违规行。
      2. 时序线性切分：训练集与验证集无重叠（去重后的公共日期）。
      3. purge 充足：训练末日与验证首日交易日间隔 ≥ purge_days
         （默认 = horizon，保证训练/验证标签价格区间不重叠）。

    Args:
        train_dates: 训练集日期序列。
        val_dates: 验证集日期序列。
        horizon: 标签持有期（交易日），默认 purge 要求取该值。
        purge_days: 训练末日与验证首日所需最小交易日间隔；
            不传时 = horizon（标签价格区间跨度）。
        _log: 是否输出日志（多折检查内部调用时传 False）。

    Returns:
        SplitReport: passed = 无打乱/重叠且 purge 充足。
    """
    check = SplitReport(check_name="时序切分合规(单折)", passed=False)
    t = _norm_dates(train_dates)
    v = _norm_dates(val_dates)
    if not t or not v:
        check.details.append("训练/验证日期为空")
        if _log:
            check.log()
        return check

    purge = int(purge_days) if purge_days is not None else int(horizon)
    train_last = t[-1]
    val_first = v[0]

    overlap = sorted(set(t) & set(v))
    interleaved = [d for d in v if d <= train_last]
    violating = sorted(set(interleaved) | set(overlap))

    check.n_checked = len(v)
    check.n_violations = len(violating)
    check.passed = check.n_violations == 0

    if check.n_violations:
        check.details.append(
            f"随机打乱禁用违规：{check.n_violations}/{len(v)} 个验证日早于或等于训练末日 "
            f"{train_last.date()}（时间交错，未来样本进入训练集）"
        )
        check.details.append(
            "示例: " + ", ".join(d.date().isoformat() for d in violating[:5])
        )

    gap_bdays = int(np.busday_count(train_last.date(), val_first.date()))
    check.details.append(
        f"训练末日 {train_last.date()} → 验证首日 {val_first.date()}，"
        f"间隔 {gap_bdays} 交易日（需 ≥ {purge} 交易日）"
    )
    if gap_bdays < purge:
        check.passed = False
        check.n_violations += 1
        check.details.append(
            f"purge 不足：需再清洗 {purge - gap_bdays} 个交易日"
            "（否则训练/验证标签价格区间重叠）"
        )

    if _log:
        check.log()
    return check


# ── 多折 Walk-Forward 检查 ─────────────────────────────────

def check_walk_forward_folds(
    folds: Sequence[tuple[Sequence[str | pd.Timestamp], Sequence[str | pd.Timestamp]]],
    *,
    horizon: int = 5,
    purge_days: int | None = None,
) -> SplitReport:
    """多折 Walk-Forward 合规检查（1.3 样本切平时序合规）。

    逐折执行 validate_time_split（单折合规：无打乱、线性切分、purge 充足），
    并额外检查折叠推进：
      - 后一折验证终点必须严格晚于前一折验证终点（滚动/扩展窗口推进，
        检出窗口回退、随机折叠、乱序折叠）。

    Args:
        folds: [(train_dates, val_dates), ...]，按训练时间先后排列。
        horizon: 标签持有期（交易日）。
        purge_days: 每折训练末日与验证首日所需最小交易日间隔，
            不传时 = horizon。

    Returns:
        SplitReport: passed = 全部折叠合规。
    """
    check = SplitReport(check_name="Walk-Forward 折叠合规", passed=True)
    if not folds:
        check.details.append("无折叠")
        check.passed = False
        check.log()
        return check

    purge = int(purge_days) if purge_days is not None else int(horizon)
    prev_v_end: pd.Timestamp | None = None

    for k, (tr, va) in enumerate(folds):
        sub = validate_time_split(tr, va, horizon=horizon, purge_days=purge, _log=False)
        check.n_checked += sub.n_checked
        if not sub.passed:
            check.passed = False
            check.details.append(f"fold{k}: {sub.details[0] if sub.details else '切分不合规'}")

        t = _norm_dates(tr)
        v = _norm_dates(va)
        if not t or not v:
            check.n_violations += sub.n_violations
            continue
        check.details.append(
            f"fold{k}: train {t[0].date()}~{t[-1].date()} "
            f"val {v[0].date()}~{v[-1].date()}"
        )
        if prev_v_end is not None and v[-1] <= prev_v_end:
            check.passed = False
            check.n_violations += 1
            check.details.append(
                f"fold{k} 验证终点 {v[-1].date()} 不晚于 fold{k-1} 验证终点 "
                f"{prev_v_end.date()}（窗口回退/随机折叠）"
            )
        prev_v_end = v[-1]

    check.log()
    return check


# ── 组合自检 ───────────────────────────────────────────────

def run_split_integrity_check(
    train_dates: Sequence[str | pd.Timestamp],
    val_dates: Sequence[str | pd.Timestamp],
    *,
    horizon: int = 5,
    purge_days: int | None = None,
    folds: Sequence[tuple[Sequence[str | pd.Timestamp], Sequence[str | pd.Timestamp]]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """一站式样本切分合规自检（1.3 样本切平时序合规）。

    Args:
        train_dates / val_dates: 单折训练/验证日期（必须提供）。
        horizon / purge_days: 透传 validate_time_split。
        folds: 可选多折序列，执行 Walk-Forward 折叠合规检查。
        **kwargs: 透传各检查项参数。

    Returns:
        dict: {
            "passed": bool（全部通过）,
            "reports": [SplitReport, ...],
            "summary": pd.DataFrame（Excel 友好）,
        }
    """
    reports = [
        validate_time_split(
            train_dates, val_dates,
            **{k: v for k, v in kwargs.items() if k in {"horizon", "purge_days"}},
        )
    ]
    if folds:
        reports.append(
            check_walk_forward_folds(
                folds,
                **{k: v for k, v in kwargs.items() if k in {"horizon", "purge_days"}},
            )
        )
    return {
        "passed": all(r.passed for r in reports),
        "reports": reports,
        "summary": pd.DataFrame([r.to_dict() for r in reports]),
    }
