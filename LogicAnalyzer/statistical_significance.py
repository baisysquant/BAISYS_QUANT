"""
统计显著性基础校验（Statistical Significance Validation）

业务定义：排除偶发性大额盈利带来的指标虚高，确保策略具有可重复性。

自检内容：
  1. 最小样本量约束：总交易回合数必须 > 100
  2. 平均持仓周期健康度：持仓<2天 + 胜率<55% → 废弃
  3. 多重牛熊覆盖：回测必须覆盖至少一个完整牛熊周期

用法:
    from LogicAnalyzer.statistical_significance import run_significance_check
    report = run_significance_check(trade_log, kline_df)
    if not report.passed:
        logger.warning(f"统计显著性未通过: {report.reason}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ── 阈值常量 ──
MIN_TOTAL_TRADES = 100                    # 最小交易回合数
MIN_AVG_HOLDING_DAYS = 2.0                # 最小平均持仓周期（交易日）
MIN_DAILY_STRATEGY_WINRATE = 0.55         # 日频策略最低胜率
MIN_MARKET_CYCLE_YEARS = 3.0              # 最小回测跨度（年），用于覆盖牛熊周期
MIN_BULL_MARKET_DAYS = 60                 # 牛市中至少 60 个上涨交易日
MIN_BEAR_MARKET_DAYS = 30                 # 熊市中至少 30 个下跌交易日
MAX_CONTINUOUS_UP_DAYS = 120              # 若连续上涨超过 120 天无回调，视为未覆盖熊市


@dataclass
class SignificanceReport:
    """统计显著性自检报告。"""

    check_name: str = ""
    passed: bool = True
    reason: str = ""
    details: list[str] = field(default_factory=list)

    @property
    def warning_level(self) -> str:
        return "FAIL" if not self.passed else "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": "PASS" if self.passed else "FAIL",
            "reason": self.reason,
            "details": self.details,
        }

    def log(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        level = "INFO" if self.passed else "WARNING"
        getattr(logger, level.lower())(
            f"[统计显著性] {self.check_name}: {status} — {self.reason}"
        )
        for detail in self.details:
            logger.info(f"  {detail}")


@dataclass
class SignificanceSummary:
    """综合统计显著性报告（三项合一）。"""

    min_sample_check: SignificanceReport = field(default_factory=SignificanceReport)
    holding_period_check: SignificanceReport = field(default_factory=SignificanceReport)
    market_cycle_check: SignificanceReport = field(default_factory=SignificanceReport)

    @property
    def passed(self) -> bool:
        return all([
            self.min_sample_check.passed,
            self.holding_period_check.passed,
            self.market_cycle_check.passed,
        ])

    @property
    def reason(self) -> str:
        failed = [
            c.check_name for c in [
                self.min_sample_check,
                self.holding_period_check,
                self.market_cycle_check,
            ] if not c.passed
        ]
        if failed:
            return f"以下校验未通过: {'，'.join(failed)}"
        return "全部通过"

    def log(self) -> None:
        for check in [self.min_sample_check, self.holding_period_check, self.market_cycle_check]:
            check.log()
        status = "PASS" if self.passed else "FAIL"
        getattr(logger, ("info" if self.passed else "warning").lower())(
            f"[统计显著性] 综合判定: {status} — {self.reason}"
        )


# ═══════════════════════════════════════════════════════════════
# 1. 最小样本量约束
# ═══════════════════════════════════════════════════════════════

def _record_time(t: dict[str, Any]) -> str:
    """统一取成交记录时间字段（引擎写 time，兼容历史 trade_date）。"""
    return str(t.get("time") or t.get("trade_date") or "")


def _count_complete_rounds(trade_log: list[dict[str, Any]]) -> int:
    """按持仓股数归零判定完整交易回合。

    引擎的部分卖出（sell_partial）与最终卖出会拆成多条记录，
    因此回合数不能按卖出记录条数统计，而应以某标的持仓
    从 >0 变为 0 的次数为准。
    """
    open_qty: dict[str, float] = defaultdict(float)
    rounds = 0
    for t in sorted(trade_log, key=_record_time):
        action = t.get("action", "")
        sym = t.get("symbol")
        if not sym:
            continue
        if action == "buy":
            open_qty[sym] += float(t.get("qty", 1))
        elif action.startswith("sell"):
            open_qty[sym] -= float(t.get("qty", 1))
            if open_qty[sym] <= 0:
                open_qty[sym] = 0.0
                rounds += 1
    return rounds


def check_min_trades(
    trade_log: list[dict[str, Any]],
) -> SignificanceReport:
    """检查总交易回合数是否 >= MIN_TOTAL_TRADES。

    交易回合定义为持仓从 0 到建仓再到清仓的完整过程；
    部分卖出（sell_partial）不单独计为回合。
    """
    report = SignificanceReport(check_name="最小样本量约束")

    total_rounds = _count_complete_rounds(trade_log)

    if total_rounds < MIN_TOTAL_TRADES:
        report.passed = False
        report.reason = (
            f"总交易回合 {total_rounds} < {MIN_TOTAL_TRADES}，"
            f"统计显著性不足，结果不予参考"
        )
        report.details.append(
            f"提示: 建议扩大回测时间窗口或降低买入阈值以增加样本量"
        )
    else:
        report.reason = f"总交易回合 {total_rounds} >= {MIN_TOTAL_TRADES}，样本量充足"

    return report


# ═══════════════════════════════════════════════════════════════
# 2. 平均持仓周期健康度
# ═══════════════════════════════════════════════════════════════

def _compute_avg_holding_days(
    trade_log: list[dict[str, Any]],
    trade_dates: list[str] | None = None,
) -> tuple[float, float]:
    """计算平均持仓周期和调仓胜率。

    按股数感知的 FIFO 匹配买入和卖出（支持部分卖出拆分多条记录），
    以成交股数为权重计算平均持仓天数。

    Args:
        trade_log: 引擎成交日志（字段: time/symbol/action/value/cost/qty）。
        trade_dates: 交易日历（从 K 线 trade_date 去重排序得到），
            传入时持仓天数按交易日计；缺省时退化为自然日近似。

    Returns:
        (avg_holding_days, win_rate)
    """
    calendar: dict[str, int] | None = None
    if trade_dates:
        try:
            calendar = {
                str(pd.Timestamp(d).date()): i
                for i, d in enumerate(sorted(trade_dates))
            }
        except (TypeError, ValueError):
            calendar = None

    def _td_diff(buy_date: str, sell_date: str) -> float | None:
        if not buy_date or not sell_date:
            return None
        try:
            d1 = pd.Timestamp(buy_date)
            d2 = pd.Timestamp(sell_date)
            if pd.isna(d1) or pd.isna(d2):
                return None
            if calendar is not None:
                k1 = str(d1.date())
                k2 = str(d2.date())
                if k1 in calendar and k2 in calendar:
                    days = calendar[k2] - calendar[k1]
                    if days >= 0:
                        return float(days)
            days = (d2 - d1).days
            return float(days) if days >= 0 else None
        except (TypeError, ValueError):
            return None

    # 按时间排序，逐条消费买入队列
    ordered = sorted(trade_log, key=_record_time)
    buy_queue: dict[str, list[dict[str, Any]]] = defaultdict(list)

    weighted_days: list[tuple[float, float]] = []
    pnl_list: list[float] = []

    for t in ordered:
        sym = t.get("symbol")
        if not sym:
            continue
        action = t.get("action", "")

        if action == "buy":
            buy_queue[sym].append(t)
            continue

        if not action.startswith("sell"):
            continue

        sell_time = _record_time(t)
        sell_value = float(t.get("value", 0))
        sell_qty = float(t.get("qty", 1))
        remaining = sell_qty
        allocated_cost = 0.0

        # 卖出按股数消费 FIFO 买入队列（部分卖出只消费对应股数）
        while remaining > 0 and buy_queue[sym]:
            b = buy_queue[sym][0]
            buy_qty = float(b.get("qty", 1)) or 1.0
            take = min(remaining, buy_qty)
            allocated_cost += (float(b.get("value", 0)) + float(b.get("cost", 0))) * (take / buy_qty)

            days = _td_diff(_record_time(b), sell_time)
            if days is not None:
                weighted_days.append((days, take))

            b["_rem"] = float(b.get("_rem", buy_qty)) - take
            if b["_rem"] <= 0:
                buy_queue[sym].pop(0)

            remaining -= take

        # 单腿 PnL：卖出净得 - 按股数摊分的买入成本
        pnl_list.append(sell_value - allocated_cost)

    if weighted_days:
        total_qty = sum(w for _, w in weighted_days)
        avg_days = sum(d * w for d, w in weighted_days) / total_qty if total_qty > 0 else 0.0
    else:
        avg_days = 0.0

    wins = sum(1 for p in pnl_list if p > 0)
    win_rate = wins / len(pnl_list) if pnl_list else 0.0

    return avg_days, win_rate


def check_holding_period_health(
    trade_log: list[dict[str, Any]],
    trade_dates: list[str] | None = None,
) -> SignificanceReport:
    """检查日频策略持仓周期健康度。

    若平均持仓 < 2 天 且 胜率 < 55%，判定策略在扣摩擦成本后无法盈利。

    Args:
        trade_log: 引擎成交日志。
        trade_dates: 交易日历（可选），传入时持仓天数按交易日精确计算。
    """
    report = SignificanceReport(check_name="平均持仓周期健康度")

    if not trade_log:
        report.passed = False
        report.reason = "无交易记录，无法计算持仓周期"
        return report

    avg_days, win_rate = _compute_avg_holding_days(trade_log, trade_dates)

    if avg_days == 0:
        report.passed = False
        report.reason = "无法匹配买卖对，持仓周期数据不足"
        return report

    report.details.append(f"平均持仓周期: {avg_days:.1f} 个交易日")
    report.details.append(f"调仓胜率: {win_rate:.1%}")

    if avg_days < MIN_AVG_HOLDING_DAYS and win_rate < MIN_DAILY_STRATEGY_WINRATE:
        report.passed = False
        report.reason = (
            f"平均持仓 {avg_days:.1f} 天 < {MIN_AVG_HOLDING_DAYS} 天 "
            f"且胜率 {win_rate:.1%} < {MIN_DAILY_STRATEGY_WINRATE:.0%}，"
            f"该日频策略在扣除双边高摩擦成本后无法实盘盈利，直接废弃"
        )
        report.details.append(
            "提示: 建议增加持仓天数或提高入场信号筛选标准以提升胜率"
        )
    elif avg_days < MIN_AVG_HOLDING_DAYS:
        report.details.append(
            f"平均持仓 {avg_days:.1f} 天较短，但胜率 {win_rate:.1%} 足够，暂不阻断"
        )
    elif win_rate < MIN_DAILY_STRATEGY_WINRATE:
        report.details.append(
            f"胜率 {win_rate:.1%} 偏低，但持仓 {avg_days:.1f} 天较长，暂不阻断"
        )
    else:
        report.reason = (
            f"平均持仓 {avg_days:.1f} 天、胜率 {win_rate:.1%}，"
            f"持仓周期健康"
        )

    return report


# ═══════════════════════════════════════════════════════════════
# 3. 多重牛熊覆盖
# ═══════════════════════════════════════════════════════════════

def check_market_cycle_coverage(
    kline_df: pd.DataFrame,
) -> SignificanceReport:
    """检查回测数据是否覆盖至少一个完整牛熊周期。

    判断标准：
    1. 回测跨度 ≥ 3 年（MIN_MARKET_CYCLE_YEARS）
    2. 包含至少 60 个大盘上涨交易日（牛市信号）
    3. 包含至少 30 个大盘下跌交易日（熊市信号）
    4. 不存在连续 120 天以上无回调的纯上涨行情（防止单一震荡上行）

    若 kline_df 含多只股票，取所有 close 的等权均值作为"大盘"代理。
    """
    report = SignificanceReport(check_name="多重牛熊覆盖")

    if kline_df.empty or "trade_date" not in kline_df.columns:
        report.passed = False
        report.reason = "K 线数据为空，无法判断牛熊覆盖"
        return report

    # ── 1. 时间跨度检查 ──
    dates = sorted(kline_df["trade_date"].unique())
    if len(dates) < 2:
        report.passed = False
        report.reason = "交易日不足 2 天，无法判断牛熊覆盖"
        return report

    try:
        first_date = pd.Timestamp(dates[0])
        last_date = pd.Timestamp(dates[-1])
    except (TypeError, ValueError):
        first_date = pd.to_datetime(dates[0], errors="coerce")
        last_date = pd.to_datetime(dates[-1], errors="coerce")

    if pd.isna(first_date) or pd.isna(last_date):
        report.passed = False
        report.reason = "日期解析失败，无法判断牛熊覆盖"
        return report

    span_years = (last_date - first_date).days / 365.25

    if span_years < MIN_MARKET_CYCLE_YEARS:
        report.passed = False
        report.reason = (
            f"回测时间跨度 {span_years:.1f} 年 < {MIN_MARKET_CYCLE_YEARS} 年，"
            f"无法覆盖完整牛熊周期，禁止仅在单一行情中局部回测"
        )
        report.details.append(f"回测区间: {first_date.strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")
        return report

    report.details.append(f"回测跨度: {span_years:.1f} 年（{len(dates)} 个交易日）")

    # ── 2. 构建大盘代理（等权平均日收益率） ──
    price_col = "close"
    if "close" in kline_df.columns:
        market_df = kline_df.groupby("trade_date")["close"].mean().to_frame()
    elif "close_adj" in kline_df.columns:
        price_col = "close_adj"
        market_df = kline_df.groupby("trade_date")["close_adj"].mean().to_frame()
    else:
        # fallback: 取含 price 的列
        price_col = next((c for c in kline_df.columns if "close" in c.lower() or "price" in c.lower()), None)
        if price_col:
            market_df = kline_df.groupby("trade_date")[price_col].mean().to_frame()
        else:
            report.passed = False
            report.reason = "无法找到收盘价列构建大盘代理"
            return report

    market_df = market_df.reset_index()
    if "trade_date" in market_df.columns:
        if pd.api.types.is_datetime64_any_dtype(market_df["trade_date"]):
            market_df["trade_date"] = market_df["trade_date"].dt.strftime("%Y-%m-%d")
        market_df = market_df.sort_values("trade_date")

    # 计算日收益率
    market_df["ret"] = market_df[price_col].pct_change()

    up_days = (market_df["ret"] > 0).sum()
    down_days = (market_df["ret"] < 0).sum()

    report.details.append(f"大盘上涨交易日: {up_days}")
    report.details.append(f"大盘下跌交易日: {down_days}")

    if up_days < MIN_BULL_MARKET_DAYS:
        report.passed = False
        report.reason = (
            f"上涨交易日仅 {up_days} < {MIN_BULL_MARKET_DAYS}，"
            f"缺乏牛市行情覆盖"
        )
        return report

    if down_days < MIN_BEAR_MARKET_DAYS:
        report.passed = False
        report.reason = (
            f"下跌交易日仅 {down_days} < {MIN_BEAR_MARKET_DAYS}，"
            f"缺乏熊市行情覆盖，禁止仅在单一震荡向上行情中回测"
        )
        return report

    # ── 4. 连续上涨检查（防止单一震荡上行） ──
    rets = market_df["ret"].values
    max_consec_up = 0
    current_consec = 0
    for r in rets:
        if r is not None and r > 0:
            current_consec += 1
            max_consec_up = max(max_consec_up, current_consec)
        else:
            current_consec = 0

    report.details.append(f"最大连续上涨天数: {max_consec_up}")

    if max_consec_up > MAX_CONTINUOUS_UP_DAYS:
        report.passed = False
        report.reason = (
            f"最大连续上涨 {max_consec_up} 天 > {MAX_CONTINUOUS_UP_DAYS} 天，"
            f"回测区间缺乏熊市回调，不能代表完整牛熊周期"
        )
        return report

    report.reason = (
        f"回测覆盖 {span_years:.1f} 年，含 {up_days} 天上涨 + {down_days} 天下跌，"
        f"最大连续上涨 {max_consec_up} 天，牛熊覆盖充分"
    )

    return report


# ═══════════════════════════════════════════════════════════════
# 一站式入口
# ═══════════════════════════════════════════════════════════════

def run_significance_check(
    trade_log: list[dict[str, Any]],
    kline_df: pd.DataFrame,
) -> SignificanceSummary:
    """执行全部三项统计显著性检查。

    Args:
        trade_log: 完整交易日志。
        kline_df: 全量 K 线数据（含 trade_date 和 close）。

    Returns:
        SignificanceSummary，.passed 为 False 时结果应废弃。
    """
    summary = SignificanceSummary()
    summary.min_sample_check = check_min_trades(trade_log)
    trade_dates = sorted(kline_df["trade_date"].astype(str).unique()) if not kline_df.empty else None
    summary.holding_period_check = check_holding_period_health(trade_log, trade_dates)
    summary.market_cycle_check = check_market_cycle_coverage(kline_df)
    summary.log()
    return summary
