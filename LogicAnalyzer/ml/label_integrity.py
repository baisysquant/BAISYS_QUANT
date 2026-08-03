"""
机器学习特征与标签合规自检模块

1.1 标签前瞻性清除（Label Look-Ahead Validation）

业务定义：确保标签（Y）的计算在业务执行时处于「绝对不可知」状态——
标签收益率所基于的价格，必须全部晚于信号的产生时刻。

两种业务对齐约定（LabelConvention）：

  - TAIL_CLOSE（尾盘下单）：
      今日尾盘（14:50-14:56）生成信号，以当日收盘价（Close）成交
      ⇒ 标签起点 = 明日收盘价：  Y_t = C_{t+h} / C_{t+1} - 1
      绝对禁止：以今日收盘价及更早价格作为标签的起点/除数。
      （违规示例：C_{t+h}/C_t - 1 —— 隐含"信号时刻已知当日收盘价"）

  - NEXT_OPEN（次日早盘下单）：
      今日收盘后生成信号，明日开盘价（Open）或早盘前 15 分钟 VWAP 成交
      ⇒ 标签起点 = 明日成交价：  Y_t = C_{t+h} / O_{t+1} - 1（或 VWAP_{t+1}）
      绝对禁止：以今日收盘价 → 明日收盘价的涨跌幅作为标签
      （违规示例：C_{t+h}/C_t - 1 —— 成交价与标签起点不符）

模块功能：
  - build_forward_return      —— 按约定构建合规标签（标签对齐/修复用）
  - validate_labels           —— 重构验证：用价格数据按约定重建标签并与现有
                                 标签逐点比对，检出任何"使用今日及以前价格"的泄露
  - check_feature_leakage     —— 特征前移泄露检测（feature[t] == feature[t+k]）
  - check_train_val_purge     —— 训练/验证标签价格重叠检查（purge ≥ 标签跨度）
  - ComplianceReport          —— PASS/FAIL 自检报告（可转 Excel）

用法:
    from LogicAnalyzer.ml.label_integrity import (
        LabelConvention, build_forward_return, validate_labels, run_label_integrity_check,
    )
    compliant = build_forward_return(panel, horizon=5, convention=LabelConvention.TAIL_CLOSE)
    report = validate_labels(panel, label_col="fwd_5d", convention=LabelConvention.TAIL_CLOSE, horizon=5)
    if not report.passed:
        logger.warning(f"[标签合规] {report.n_violations} 行标签存在前瞻性泄露")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class LabelConvention(str, Enum):
    """标签对齐的业务约定。"""

    TAIL_CLOSE = "tail_close"  # 尾盘下单：信号当日生成、当日收盘价成交 → 起点 = 明日收盘价
    NEXT_OPEN = "next_open"    # 次日早盘下单：信号次日开盘价/VWAP 成交 → 起点 = 明日成交价


# 各约定的标签起点偏移（t + base_offset 为收益率起点日期）
_BASE_OFFSET = {
    LabelConvention.TAIL_CLOSE: 1,
    LabelConvention.NEXT_OPEN: 1,
}


@dataclass
class ComplianceReport:
    """合规自检报告。"""

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
            logger.info(f"[标签合规] {self.check_name} PASS（样本 {self.n_checked}）")
        else:
            logger.warning(
                f"[标签合规] {self.check_name} FAIL（违规 {self.n_violations}/{self.n_checked}）"
            )


# ── 合规标签构建 ───────────────────────────────────────────

def build_forward_return(
    panel: pd.DataFrame,
    horizon: int,
    convention: LabelConvention | str,
    *,
    close_col: str = "close",
    open_col: str = "open",
    vwap_col: str | None = None,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> pd.Series:
    """按业务约定构建合规的前瞻收益率标签（起点 = 明日价格，不含今日及以前）。

    约定与公式：
      - TAIL_CLOSE：Y_t = C_{t+h} / C_{t+1} - 1（明日收盘价为起点）
      - NEXT_OPEN ：Y_t = C_{t+h} / P_{t+1} - 1（P 为明日开盘价，或提供
        vwap_col 时用明日 VWAP）

    Args:
        panel: 长表面板，需含 symbol_col / date_col / close_col。
        horizon: 持有期（交易日）。
        convention: LabelConvention 或其字符串值。
        close_col: 收盘价列。
        open_col: 开盘价列（NEXT_OPEN 且未提供 vwap_col 时必需）。
        vwap_col: 可选 VWAP 列（NEXT_OPEN 优先用作成交价）。
        symbol_col / date_col: 股票代码 / 日期列。

    Returns:
        pd.Series: 标签收益率，index 与排序后的 panel 一致（调用方可用
        panel.index 对齐）。

    Raises:
        ValueError: NEXT_OPEN 缺少 open/vwap 列，或约定非法。
    """
    if panel is None or panel.empty:
        return pd.Series(dtype=float)
    if close_col not in panel.columns:
        raise ValueError(f"面板缺少收盘价列: {close_col}")

    conv = LabelConvention(convention)
    if conv is LabelConvention.NEXT_OPEN:
        base_col = vwap_col if vwap_col and vwap_col in panel.columns else open_col
        if base_col is None or base_col not in panel.columns:
            raise ValueError(
                f"NEXT_OPEN 约定需要成交价列（{vwap_col or 'vwap_col'} 或 {open_col}）"
            )
    else:
        base_col = close_col

    df = panel.copy()
    df["_sym"] = df[symbol_col].astype(str)
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_date"])
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[base_col] = pd.to_numeric(df[base_col], errors="coerce")
    df = df.sort_values(["_sym", "_date"])

    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon 必须 ≥ 1（当前 {h}）")
    g = df.groupby("_sym", sort=False)
    fwd = g[close_col].shift(-h) / g[base_col].shift(-1) - 1
    fwd = fwd.replace([np.inf, -np.inf], np.nan)
    return pd.Series(fwd.to_numpy(), index=df.index)


# ── 标签前瞻性清除验证 ─────────────────────────────────────

def validate_labels(
    panel: pd.DataFrame,
    label_col: str,
    convention: LabelConvention | str,
    horizon: int,
    *,
    tol: float = 1e-8,
    close_col: str = "close",
    open_col: str = "open",
    vwap_col: str | None = None,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> ComplianceReport:
    """重构验证标签是否包含前瞻性泄露。

    用价格数据按约定重建标签 Y_expected = build_forward_return(...)，
    与面板中现有标签逐点比对：
      - |Y - Y_expected| > tol 的行 → 违规（标签起点使用了今日及以前的价格）
      - 标签缺失（NaN）→ 不视为违规（窗口尾部天然缺失）
      - 标签存在但重建为 NaN → 标签超出数据范围，视为不一致

    Returns:
        ComplianceReport: passed = 违规数为 0。
    """
    check = ComplianceReport(check_name=f"标签前瞻性清除({label_col})", passed=False)
    if label_col not in panel.columns:
        check.details.append("标签列不存在")
        return check
    try:
        expected = build_forward_return(
            panel, horizon, convention,
            close_col=close_col, open_col=open_col, vwap_col=vwap_col,
            symbol_col=symbol_col, date_col=date_col,
        )
    except ValueError as e:
        check.details.append(str(e))
        return check

    expected = expected.reindex(panel.index)
    y = pd.to_numeric(panel[label_col], errors="coerce")
    exp = pd.to_numeric(expected, errors="coerce")

    both = y.notna() & exp.notna()
    missing = y.isna() & exp.notna()
    overflow = y.notna() & exp.isna()

    dev = (y - exp).abs().where(both)
    violations = both & (dev > tol)

    check.n_checked = int(both.sum())
    check.n_violations = int(violations.sum())
    check.max_deviation = float(dev.max()) if check.n_checked else 0.0
    check.passed = check.n_violations == 0

    if not check.n_checked:
        check.details.append("标签与价格无有效对齐样本")
    if check.n_violations:
        viol_dates = panel.loc[violations, date_col].astype(str).drop_duplicates().head(5)
        check.details.append(f"违规日期示例: {', '.join(viol_dates)}")
        top_syms = (
            panel.loc[violations, symbol_col].astype(str).value_counts().head(3)
        )
        check.details.append("违规最多标的: " + ", ".join(f"{s}({c})" for s, c in top_syms.items()))
    if int(missing.sum()):
        check.details.append(f"标签缺失 {int(missing.sum())} 行（窗口尾部，正常）")
    if int(overflow.sum()):
        check.details.append(f"标签超出数据范围 {int(overflow.sum())} 行（不一致）")
        check.passed = False

    check.log()
    return check


# ── 特征前移泄露检测 ───────────────────────────────────────

# 未来量：显示名 → (源价格列, 形态)；形态 price=次日价, ret1=次日收益率, ret5=未来5日收益率
_FUTURE_QUANTITIES: dict[str, tuple[str, str]] = {
    "明日收盘价": ("close", "price"),
    "明日开盘价": ("open", "price"),
    "明日最高价": ("high", "price"),
    "明日最低价": ("low", "price"),
    "明日收益率": ("close", "ret1"),
    "未来5日收益率": ("close", "ret5"),
}


def check_feature_leakage(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    tol: float = 1e-12,
    lookahead: Sequence[int] = (1, 2, 3),
    max_leak_ratio: float = 0.01,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    close_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
) -> ComplianceReport:
    """检测特征是否引用了未来数据（前瞻性泄露），两种模式：

    1. 未来量重合：特征列与「明日收盘/开盘/最高/最低、明日收益率、
       未来5日收益率」等未来量逐行完全相等（t 时刻特征由未来数据计算）。
    2. 行重复回填：feature[t] == feature[t+k]（未来值被前移/回填到过去行）。

    Args:
        panel: 长表面板（需含 symbol_col / date_col）。
        feature_cols: 待检测特征列。
        tol: 判定"相等"的容差（默认精确相等）。
        lookahead: 行重复回填检测的滞后期序列。
        max_leak_ratio: 允许的最大重合行占比（超出即违规）。
        symbol_col / date_col / close_col / open_col / high_col / low_col: 列名。

    Returns:
        ComplianceReport: passed = 无特征泄露。details 列出嫌疑特征及占比。
    """
    check = ComplianceReport(check_name="特征前移泄露检测", passed=True)
    valid_cols = [c for c in feature_cols if c in panel.columns]
    if not valid_cols:
        check.details.append("无可检测特征列")
        return check

    df = panel.copy()
    df["_sym"] = df[symbol_col].astype(str)
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_date"]).sort_values(["_sym", "_date"])
    g = df.groupby("_sym", sort=False)

    # 未来量字典：显示名 → 序列（与 df 对齐）
    future_series: dict[str, pd.Series] = {}
    for name, (col, form) in _FUTURE_QUANTITIES.items():
        if col not in df.columns:
            continue
        src = pd.to_numeric(df[col], errors="coerce")
        if form == "ret1":
            future_series[name] = g[col].shift(-1) / src - 1
        elif form == "ret5":
            future_series[name] = g[col].shift(-5) / src - 1
        else:
            future_series[name] = g[col].shift(-1)
        future_series[name] = future_series[name].replace([np.inf, -np.inf], np.nan)

    for col in valid_cols:
        vals = pd.to_numeric(df[col], errors="coerce")

        # 1) 未来量重合
        matched = []
        for name, fut in future_series.items():
            mask = vals.notna() & fut.notna()
            if mask.sum() == 0:
                continue
            ratio = float(((vals - fut).abs() <= tol).sum() / mask.sum())
            if ratio > max_leak_ratio:
                matched.append(f"{name}({ratio:.2%})")
        # 2) 行重复回填
        worst_ratio = 0.0
        worst_k = 0
        for k in lookahead:
            fut = g[col].shift(-int(k))
            fut = pd.to_numeric(fut, errors="coerce")
            mask = vals.notna() & fut.notna()
            if mask.sum() == 0:
                continue
            ratio = float(((vals - fut).abs() <= tol).sum() / mask.sum())
            if ratio > worst_ratio:
                worst_ratio, worst_k = ratio, int(k)

        check.n_checked += 1
        if matched:
            check.passed = False
            check.n_violations += 1
            check.details.append(f"'{col}' 与未来量重合: " + ", ".join(matched))
        elif worst_ratio > max_leak_ratio:
            check.passed = False
            check.n_violations += 1
            check.details.append(
                f"'{col}' 与 t+{worst_k} 日取值完全相同占比 {worst_ratio:.2%}（疑似前移回填）"
            )
        else:
            check.details.append(f"'{col}' 最大前移重合 {worst_ratio:.4%}（t+{worst_k}）")

    check.log()
    return check


# ── 训练/验证标签价格重叠检查 ──────────────────────────────

def check_train_val_purge(
    train_dates: Sequence[str | pd.Timestamp],
    val_dates: Sequence[str | pd.Timestamp],
    horizon: int,
    convention: LabelConvention | str,
    *,
    base_offset: int | None = None,
) -> ComplianceReport:
    """检查训练集与验证集标签所引用的价格区间是否重叠。

    标签 Y_t 引用价格区间 [t + base_offset, t + horizon]。
    训练末样本与验证首样本区间相交 ⟺ 存在价格行同时参与训练与验证标签
    （即训练集"看到"了验证集的未来），应通过 purge（日期间隔）避免。

    Args:
        train_dates: 训练集日期序列。
        val_dates: 验证集日期序列。
        horizon: 标签持有期（天）。
        convention: 业务约定（决定 base_offset，默认 1）。
        base_offset: 显式指定起点偏移（0 = 旧式 close-to-close）。

    Returns:
        ComplianceReport: passed = 无重叠。details 给出所需 purge 天数。
    """
    check = ComplianceReport(check_name="训练/验证标签重叠检查", passed=False)
    t = sorted(pd.to_datetime(list(train_dates)))
    v = sorted(pd.to_datetime(list(val_dates)))
    if not t or not v:
        check.details.append("训练/验证日期为空")
        return check

    conv = LabelConvention(convention)
    offset = int(base_offset) if base_offset is not None else _BASE_OFFSET[conv]
    h = int(horizon)
    t_last = t[-1]
    v_first = v[0]

    gap_bdays = int(np.busday_count(t_last.date(), v_first.date()))
    overlap_limit = t_last + pd.offsets.BDay(n=h - offset)
    overlapping = [d for d in v if d <= overlap_limit]

    check.n_checked = len(v)
    check.n_violations = len(overlapping)
    check.passed = check.n_violations == 0
    required_purge = h - offset + 1
    check.details.append(f"约定={conv.value} base_offset={offset} horizon={h}")
    check.details.append(
        f"训练末日 {t_last.date()} → 验证首日 {v_first.date()}，"
        f"间隔 {gap_bdays} 交易日（需 ≥ {required_purge} 交易日）"
    )
    if check.n_violations:
        check.details.append(f"前 {min(3, len(overlapping))} 个重叠验证日: " +
                             ", ".join(d.date().isoformat() for d in overlapping[:3]))
    check.log()
    return check


# ── 组合自检 ───────────────────────────────────────────────

def run_label_integrity_check(
    panel: pd.DataFrame,
    label_col: str,
    convention: LabelConvention | str,
    horizon: int,
    *,
    feature_cols: Sequence[str] | None = None,
    train_dates: Sequence[str | pd.Timestamp] | None = None,
    val_dates: Sequence[str | pd.Timestamp] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """一站式特征与标签合规自检（1.1 标签前瞻性清除）。

    Args:
        panel / label_col / convention / horizon: 见 validate_labels。
        feature_cols: 特征列（可选，执行特征前移泄露检测）。
        train_dates / val_dates: 训练/验证日期（可选，执行重叠检查）。
        **kwargs: 透传各检查项参数。

    Returns:
        dict: {
            "passed": bool（全部通过）,
            "reports": [ComplianceReport, ...],
            "summary": pd.DataFrame（Excel 友好）,
        }
    """
    reports = [
        validate_labels(
            panel, label_col, convention, horizon,
            **{k: v for k, v in kwargs.items() if k in {
                "tol", "close_col", "open_col", "vwap_col", "symbol_col", "date_col",
            }},
        )
    ]
    if feature_cols:
        reports.append(
            check_feature_leakage(
                panel, feature_cols,
                **{k: v for k, v in kwargs.items() if k in {
                    "tol", "lookahead", "max_leak_ratio", "symbol_col", "date_col",
                }},
            )
        )
    if train_dates is not None and val_dates is not None:
        reports.append(
            check_train_val_purge(
                train_dates, val_dates, horizon, convention,
                **{k: v for k, v in kwargs.items() if k in {"base_offset"}},
            )
        )
    return {
        "passed": all(r.passed for r in reports),
        "reports": reports,
        "summary": pd.DataFrame([r.to_dict() for r in reports]),
    }
