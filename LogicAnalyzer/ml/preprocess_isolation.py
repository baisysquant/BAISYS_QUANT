"""
机器学习预处理信息隔离自检模块

1.5 预处理信息隔离（Preprocessing Isolation）

业务定义：确保模型在测试阶段面对的特征分布变换参数完全来自历史训练集——
所有涉及数据分布调整的预处理（缺失值填充值、标准化均值与方差、极值截断
阈值、PCA 降维矩阵）的计算基准必须且只能来自「当前训练集」；在验证集与
测试集上只能被动套用该历史参数，绝对禁止在测试集上重算分布参数。

自检内容：
  1. 参数来源登记核查（check_param_fit_within_train）：任何分布参数预处理
     必须显式登记为 PreprocessParam，其拟合时间区间必须完全落在训练集内；
     不得早于训练起点、不得触及验证/测试期。
  2. 训练行不变性重构验证（check_train_features_invariant）：若分布参数在
     「训练+测试」全样本上拟合，则移除测试行后同一批训练行的特征取值必然
     改变。将管线在（a）全样本 与（b）仅训练期历史 上算得的特征，在
     训练行上逐点比对，任何不一致即证明存在全局/含测试期分布拟合——
     对当前无参数管线恒等于 PASS，并守护今后特性不引入全局统计回归。

当前管线合规结论：
  _cross_sectional_rank（逐日横截面排名：各日互不独立、与训练和未来无关）
  与逐行 NaN 掩码均无时间维分布参数；不存在 imputation / 标准化 /
  winsorize / PCA ⇒ 满足 1.5。

用法:
    from LogicAnalyzer.ml.preprocess_isolation import (
        PreprocessParam,
        check_param_fit_within_train,
        check_train_features_invariant,
        run_preprocess_check,
    )
    ok = check_train_features_invariant(feature_fn, panel, feature_cols)
    report = check_param_fit_within_train(params, train_dates, test_dates=val_dates)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from LogicAnalyzer.ml.split_integrity import SplitReport

# 支持的分布参数预处理类型
PARAM_KINDS = ("impute", "standardize", "winsorize", "pca")


@dataclass
class PreprocessParam:
    """已登记的分布参数预处理（含其拟合数据来源的时间区间）。

    Attributes:
        name: 参数名（如 "标准化_close_mean/std"）。
        kind: 类型，须为 PARAM_KINDS 之一。
        fit_start / fit_end: 拟合基准的数据日期范围（含端点）。
        n_cols: 涉及的列数。
    """

    name: str
    kind: str
    fit_start: pd.Timestamp | None = None
    fit_end: pd.Timestamp | None = None
    n_cols: int = 0

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"kind 必须属于 {PARAM_KINDS}，当前 {self.kind!r}")
        if self.fit_start is not None:
            self.fit_start = pd.Timestamp(self.fit_start)
        if self.fit_end is not None:
            self.fit_end = pd.Timestamp(self.fit_end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "参数": self.name,
            "类型": self.kind,
            "拟合起点": None if self.fit_start is None else self.fit_start.date().isoformat(),
            "拟合终点": None if self.fit_end is None else self.fit_end.date().isoformat(),
            "列数": self.n_cols,
        }


def _norm_dates(dates: Sequence[str | pd.Timestamp]) -> list[pd.Timestamp]:
    """规范化日期：转为 Timestamp、剔除 NaT、去重、按时间升序。"""
    dt = pd.to_datetime(list(dates), errors="coerce")
    return [pd.Timestamp(d) for d in dt.dropna().drop_duplicates().sort_values()]


# ── 1. 参数来源登记核查 ────────────────────────────────────

def check_param_fit_within_train(
    params: Sequence[PreprocessParam],
    train_dates: Sequence[str | pd.Timestamp],
    *,
    test_dates: Sequence[str | pd.Timestamp] | None = None,
) -> SplitReport:
    """参数来源登记核查：分布参数拟合基准必须完全落在当前训练集内。

    违规类型：
      - 拟合起点早于训练集起点（利用更早/测试之前数据）；
      - 拟合终点晚于训练集终点（含验证/测试期分布信息）；
      - 拟合区间与验证/测试期相交。

    Args:
        params: 已登记的分布参数预处理序列。
        train_dates: 当前训练集日期。
        test_dates: 可选验证/测试集日期（用于检查相交）。

    Returns:
        SplitReport: passed = 全部参数来源合规。
    """
    check = SplitReport(check_name="预处理参数来源(仅训练集)", passed=True)
    t = _norm_dates(train_dates)
    if not t:
        check.details.append("训练集日期为空")
        check.passed = False
        check.log()
        return check
    if not params:
        check.details.append(f"无登记分布参数（共 {len(params)} 个）")
        check.passed = True
        check.n_checked = 0
        check.log()
        return check

    v = _norm_dates(test_dates) if test_dates is not None else []
    check.n_checked = len(params)
    for p in params:
        bad = []
        if p.fit_start is not None and p.fit_start < t[0]:
            bad.append(f"起点 {p.fit_start.date()} 早于训练起点 {t[0].date()}")
        if p.fit_end is not None and p.fit_end > t[-1]:
            bad.append(f"终点 {p.fit_end.date()} 晚于训练终点 {t[-1].date()}")
        if (
            v
            and p.fit_start is not None and p.fit_start <= v[-1]
            and p.fit_end is not None and p.fit_end >= v[0]
        ):
            bad.append("拟合区间与验证/测试期相交")
        if bad:
            check.passed = False
            check.n_violations += 1
            check.details.append(f"{p.name}[{p.kind}]: {'；'.join(bad)}（来源非训练集）")
        else:
            check.details.append(f"{p.name}[{p.kind}] 拟合区间在训练集内")
    check.log()
    return check


# ── 2. 训练行不变性重构验证 ───────────────────────────────

def check_train_features_invariant(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    test_frac: float = 0.2,
    tol: float = 1e-9,
    min_test_rows: int = 5,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> SplitReport:
    """训练行特征不变性重构验证（1.5 预处理信息隔离）。

    逻辑：
      全样本 = feature_fn(完整面板)（生产管线取值，分布参数若存在则含测试期）。
      历史样本 = feature_fn(仅训练期历史)（不含测试行）。
      对「训练行」（两遍都出现的行）逐点比对特征取值——
      若测试行参与过分布拟合，训练行取值必然随之漂移 ⇒ 违规。
      无时间维分布参数的管线（含逐日横截面变换）两遍逐点相等 ⇒ PASS。

    Args:
        feature_fn: 特征管线函数（入参保留 index 的面板，返回同 index）。
        panel: 原始面板（需含 symbol_col / date_col）。
        feature_cols: 参与比对的特征列。
        test_frac: 末尾该比例日期视为「测试期」参与截断比较（默认 20%）。
        tol: 并列定等价容差。
        min_test_rows: 测试期日期数低于该值时跳过（视同通过）。
        symbol_col / date_col: 列名。

    Returns:
        SplitReport: passed = 训练行特征不受测试行存在与否影响。
    """
    check = SplitReport(check_name="参数来源隔离(训练行重构)", passed=True)
    if panel is None or panel.empty:
        check.details.append("面板为空")
        check.passed = False
        check.log()
        return check

    df = panel.copy()
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_dt"]).sort_values([symbol_col, "_dt"])
    if df.empty:
        check.details.append("面板无有效日期")
        check.passed = False
        check.log()
        return check

    uniq_dates = df["_dt"].drop_duplicates().sort_values()
    n_test = max(1, int(round(len(uniq_dates) * test_frac)))
    if n_test < min_test_rows:
        check.details.append(f"测试期日期数 {n_test} < {min_test_rows}，跳过重构比对")
        check.n_checked = 0
        check.log()
        return check
    test_start = uniq_dates.iloc[-n_test]

    train_mask = df["_dt"] < test_start
    train_idx = df.index[train_mask]
    if len(train_idx) < 1:
        check.details.append("无可比对的训练行")
        check.passed = False
        check.log()
        return check

    full = feature_fn(df)
    hist = feature_fn(df.loc[train_idx])

    cols = [c for c in feature_cols if c in full.columns]
    if not cols:
        check.details.append("特征管线未输出任何待检特征列")
        check.passed = False
        check.log()
        return check

    a = pd.DataFrame(full.loc[train_idx, cols]).apply(pd.to_numeric, errors="coerce")
    b = pd.DataFrame(hist.loc[train_idx, cols]).apply(pd.to_numeric, errors="coerce")

    both = a.notna() & b.notna()
    dev = (a - b).abs()
    mismatch = both & (dev > tol)

    check.n_checked = int(both.to_numpy().sum())
    check.n_violations = int(mismatch.to_numpy().sum())
    check.passed = check.n_violations == 0

    if not check.n_checked:
        check.details.append("训练行上无有效比对单元格")
        check.passed = False
    elif check.passed:
        check.details.append(
            f"训练行 {len(train_idx)} 行 × {len(cols)} 特征与「仅训练期历史」重算逐点一致"
            "（无全局/含测试期分布参数）"
        )
    else:
        per_col = mismatch.sum()
        worst = per_col[per_col > 0].sort_values(ascending=False)
        for c, n in worst.head(5).items():
            check.details.append(f"'{c}' 有 {int(n)} 个训练单元格因测试行存在而漂移")
        check.details.append(f"测试期由末尾 {n_test} 个交易日构成")
    check.log()
    return check


# ── 3. 组合自检 ──────────────────────────────────────────

def run_preprocess_check(
    panel: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    feature_cols: Sequence[str],
    *,
    params: Sequence[PreprocessParam] | None = None,
    train_dates: Sequence[str | pd.Timestamp] | None = None,
    test_dates: Sequence[str | pd.Timestamp] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """一站式预处理信息隔离自检（1.5 预处理信息隔离）。

    - 提供 train_dates 时执行参数来源登记核查；
    - 始终执行训练行不变性重构验证。

    Returns:
        dict: {"passed", "reports", "summary(pd.DataFrame)"}
    """
    reports: list[SplitReport] = []
    if params is not None and train_dates is not None:
        reports.append(
            check_param_fit_within_train(params, train_dates, test_dates=test_dates)
        )
    reports.append(
        check_train_features_invariant(
            feature_fn, panel, feature_cols,
            **{k: v for k, v in kwargs.items() if k in {
                "test_frac", "tol", "min_test_rows", "symbol_col", "date_col",
            }},
        )
    )
    return {
        "passed": all(r.passed for r in reports),
        "reports": reports,
        "summary": pd.DataFrame([r.to_dict() for r in reports]),
    }