"""
ML 特征未来函数阻断自检模块（1.2 Feature Window Validation）

业务定义：特征在 T 行的取值只能使用截止 T-1 收盘的信息——
  - 特征 T 日窗口必须截止 T-1 收盘（T 行不得使用当日收盘/最高/最低/量/额）
  - 禁止全时段全局统计量（full-sample mean/max 等，隐含未来数据）
  - 横截面特征（rel_strength_20d / regime_*）须逐日对齐并同样 T-1 闭合
  - 基本面特征须以公告日次日（ann_date + 1）对齐（本模块针对技术面特征）
  - 停牌日不得错位挪用复牌数据（逐 symbol 序列计算）

检测原理（无需特征公式，纯黑盒）：
  1. 扰动重构（check_feature_window）：将每只股票最后一天的 OHLCV 篡改后
     重建特征——T-1 闭合特征在 T 行的取值不变；任何使用了当日数据的特征
     （如 ret_1d = C_t / C_{t-1} - 1）必然改变。
  2. 极端行追加（check_no_global_statistics）：在每只股票末尾追加一天极端
     价格行后重建特征——依赖全时段统计量的特征会在全部既有行上整体改变。

模块功能：
  - check_feature_window          —— T-1 窗口闭合检验（扰动重构）
  - check_no_global_statistics    —— 全时段统计量检测（极端行追加）
  - run_feature_window_check      —— 一站式特征窗口自检（可转 Excel）
  - FeatureWindowReport           —— PASS/FAIL 自检报告

用法:
    from LogicAnalyzer.ml.feature_window import run_feature_window_check
    result = run_feature_window_check(_compute_feature_matrix, panel, _FEATURE_ALL)
    if not result["passed"]:
        logger.warning("[特征窗口] 特征存在未来函数")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# 特征计算实际读取的当日行情列（signal_model._compute_features 的输入）
_PRICE_COLS = ("close", "high", "low", "volume", "amount", "ATR")


@dataclass
class FeatureWindowReport:
    """特征窗口合规自检报告。"""

    check_name: str
    passed: bool
    n_checked: int = 0
    n_violations: int = 0
    max_deviation: float = 0.0
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "检查项": self.check_name,
            "通过": "PASS" if self.passed else "FAIL",
            "样本数": self.n_checked,
            "违规数": self.n_violations,
            "最大偏差": round(self.max_deviation, 10),
            "说明": "；".join(self.details[:6]) or "",
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def log(self) -> None:
        if self.passed:
            logger.info(f"[特征窗口] {self.check_name} PASS（样本 {self.n_checked}）")
        else:
            logger.warning(
                f"[特征窗口] {self.check_name} FAIL（违规 {self.n_violations}/{self.n_checked}）"
            )


# ── 重构输入构造 ───────────────────────────────────────────

def _perturb_last_rows(panel: pd.DataFrame, *, symbol_col: str = "symbol") -> pd.DataFrame:
    """篡改每只股票最后一行的 OHLCV，其余行原样保留。

    各列采用不同变换（避免缩放不变特征如 hl_ratio / amt_ratio_5_20 逃检）：
    close 按前收方向反向篡改（涨→×0.5、跌→×2，保证 (c > prev) 类布尔特征
    必然翻转）、high 放大 / low 缩小 / volume 放大 / amount 放大后加平移。
    """
    df = panel.copy()
    last_idx = df.groupby(symbol_col).tail(1).index
    for col in _PRICE_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(float)
        vals = df.loc[last_idx, col]
        if col == "close":
            prev = df.groupby(symbol_col)["close"].shift(1).loc[last_idx]
            up = vals > prev
            df.loc[last_idx, col] = np.where(up, vals * 0.5, vals * 2.0)
        elif col == "amount":
            df.loc[last_idx, col] = vals * 1.5 + 1e6
        elif col == "high":
            df.loc[last_idx, col] = vals * 1.3
        elif col == "low":
            df.loc[last_idx, col] = vals * 0.7
        else:  # volume / ATR
            df.loc[last_idx, col] = np.where(vals == 0, 1.0, vals * 1.5)
    return df


def _extreme_junk_rows(
    panel: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """构造每只股票末尾追加的极端价格行（末日 +1 天），用于全时段统计量检测。"""
    is_dt = pd.api.types.is_datetime64_any_dtype(panel[date_col])
    rows = []
    for _, grp in panel.groupby(symbol_col, sort=False):
        last = grp.iloc[[-1]].copy()
        for col in _PRICE_COLS:
            if col in last.columns:
                last[col] = float(1e-9 if col == "low" else 1e9)
        d = pd.to_datetime(last[date_col], errors="coerce") + pd.Timedelta(days=1)
        if bool(d.notna().all()):
            last[date_col] = d if is_dt else d.dt.strftime("%Y-%m-%d")
        rows.append(last)
    junk = pd.concat(rows)
    junk.index = pd.RangeIndex(len(panel), len(panel) + len(junk))
    return junk


# ── T-1 窗口闭合检验 ───────────────────────────────────────

def check_feature_window(
    compute_features: Callable[[pd.DataFrame], pd.DataFrame],
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    tol: float = 1e-9,
    symbol_col: str = "symbol",
) -> FeatureWindowReport:
    """T-1 窗口闭合检验（扰动重构，检出使用当日收盘/最高/最低/量/额的特征）。

    原理：篡改每只股票最后一行（T 行）的 OHLCV 后重建特征。T-1 闭合的
    特征在 T 行的取值 = 以 T-1 及以前数据计算的原始值，不受 T 行行情影响；
    任何使用了当日数据的特征（如 ret_1d = C_t / C_{t-1} - 1）都会改变。

    Args:
        compute_features: 特征计算函数（输入原始 OHLCV 面板，输出含
            feature_cols 的特征面板，须保持 index 不变）。
        panel: 原始行情长表面板。
        feature_cols: 待检测特征列。
        tol / symbol_col: 容差 / 股票代码列。

    Returns:
        FeatureWindowReport: passed = 全部特征 T-1 闭合。
    """
    check = FeatureWindowReport(check_name="特征 T-1 窗口闭合检验", passed=True)
    try:
        base = compute_features(panel.copy())
        perturbed = compute_features(
            _perturb_last_rows(panel, symbol_col=symbol_col)
        )
    except Exception as e:
        check.passed = False
        check.details.append(f"特征重建失败: {e}")
        check.log()
        return check

    last_idx = panel.groupby(symbol_col).tail(1).index
    valid = [c for c in feature_cols if c in base.columns]
    if not valid:
        check.details.append("无可检测特征列")
        check.log()
        return check

    for col in valid:
        b = pd.to_numeric(base.loc[last_idx, col], errors="coerce").astype(float)
        p = pd.to_numeric(perturbed.loc[last_idx, col], errors="coerce").astype(float)
        diff = (b - p).abs()
        bad = ~(diff <= tol) & ~(b.isna() & p.isna())
        check.n_checked += int(b.notna().sum())
        if bad.any():
            check.passed = False
            check.n_violations += int(bad.sum())
            check.max_deviation = max(check.max_deviation, float(diff.max()))
            check.details.append(
                f"'{col}' 依赖当日行情：篡改 T 行后 {int(bad.sum())} 行取值改变"
            )
    check.log()
    return check


# ── 全时段统计量检测 ───────────────────────────────────────

def check_no_global_statistics(
    compute_features: Callable[[pd.DataFrame], pd.DataFrame],
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    tol: float = 1e-9,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> FeatureWindowReport:
    """全时段统计量检测（极端行追加，检出依赖 full-sample mean/max 的特征）。

    原理：在每只股票末尾追加一天极端价格行后重建特征。滚动窗口/滞后类
    特征不受追加行影响；依赖全时段统计量的特征会在全部既有行上整体改变。

    Args:
        compute_features / panel / feature_cols: 见 check_feature_window。
        tol / symbol_col / date_col: 容差 / 股票代码列 / 日期列。

    Returns:
        FeatureWindowReport: passed = 无特征依赖全时段统计量。
    """
    check = FeatureWindowReport(check_name="全时段统计量检测", passed=True)
    junk = _extreme_junk_rows(panel, symbol_col=symbol_col, date_col=date_col)
    extended = pd.concat([panel.copy(), junk])
    marker = np.zeros(len(extended), dtype=bool)
    marker[len(panel):] = True
    extended["__feature_window_junk__"] = marker
    try:
        base = compute_features(panel.copy())
        rebuilt = compute_features(extended)
    except Exception as e:
        check.passed = False
        check.details.append(f"特征重建失败: {e}")
        check.log()
        return check

    rebuilt = rebuilt.loc[~rebuilt["__feature_window_junk__"]].drop(
        columns=["__feature_window_junk__"]
    )
    valid = [c for c in feature_cols if c in base.columns]
    if not valid:
        check.details.append("无可检测特征列")
        check.log()
        return check

    for col in valid:
        b = pd.to_numeric(base[col], errors="coerce").astype(float)
        p = pd.to_numeric(rebuilt[col], errors="coerce").astype(float)
        diff = (b - p).abs()
        bad = ~(diff <= tol) & ~(b.isna() & p.isna())
        check.n_checked += int(b.notna().sum())
        if bad.any():
            check.passed = False
            check.n_violations += int(bad.sum())
            check.max_deviation = max(check.max_deviation, float(diff.max()))
            check.details.append(
                f"'{col}' 依赖全时段统计量：追加极端行后 {int(bad.sum())} 行取值改变"
            )
    check.log()
    return check


# ── 组合自检 ───────────────────────────────────────────────

def run_feature_window_check(
    compute_features: Callable[[pd.DataFrame], pd.DataFrame],
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """一站式特征窗口自检（1.2 特征未来函数阻断）。

    Args:
        compute_features / panel / feature_cols: 见 check_feature_window。
        **kwargs: 透传各检查项参数（tol / symbol_col / date_col）。

    Returns:
        dict: {
            "passed": bool（全部通过）,
            "reports": [FeatureWindowReport, ...],
            "summary": pd.DataFrame（Excel 友好）,
        }
    """
    reports = [
        check_feature_window(
            compute_features, panel, feature_cols,
            **{k: v for k, v in kwargs.items() if k in {"tol", "symbol_col"}},
        ),
        check_no_global_statistics(
            compute_features, panel, feature_cols,
            **{k: v for k, v in kwargs.items() if k in {"tol", "symbol_col", "date_col"}},
        ),
    ]
    return {
        "passed": all(r.passed for r in reports),
        "reports": reports,
        "summary": pd.DataFrame([r.to_dict() for r in reports]),
    }
