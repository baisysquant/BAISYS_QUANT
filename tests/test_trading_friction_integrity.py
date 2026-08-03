"""1.8 交易摩擦合规（TradingFrictionCompliance）测试。

覆盖：双边显性成本（佣金/印花税/过户费/最低佣金）、固定滑点下限、
ADV 动态冲击成本（平方根模型）、卖出端完整性、一站式聚合与引擎集成。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BackTrading._engine_legacy import _run_single_backtest
from BackTrading.domain.models import CostModel
from BackTrading.engine import EngineConfig
from LogicAnalyzer.ml.trading_friction_integrity import (
    check_double_sided_explicit_costs,
    check_dynamic_impact,
    check_sell_side_completeness,
    check_slippage_floor,
    check_trading_friction_config,
    run_trading_friction_check,
)

# ── fixtures ───────────────────────────────────────────────

def _dates(n: int, start: str = "2024-01-01") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def _engine_bars(n_days: int = 60, n_syms: int = 6, seed: int = 11,
                 start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _dates(n_days, start=start)
    rows = []
    for i, sym in enumerate(["600000.SH", "600001.SH", "600002.SH",
                             "600003.SH", "600004.SH", "600005.SH"]):
        close_prev = 10.0 + i
        for d in dates:
            close = max(close_prev * (1 + rng.normal(0.0, 0.01)), 1.0)
            rows.append({
                "trade_date": d, "symbol": sym,
                "close": float(close), "close_adj": float(close),
                "volume": 1_000_000,
                "进场评分": float(rng.integers(0, 100)),
                "退出评分": float(rng.integers(0, 80)),
                "风险等级": rng.choice(["LOW", "MEDIUM"], size=1)[0],
            })
            close_prev = close
    return pd.DataFrame(rows)


def _cfg(**kw) -> EngineConfig:
    return EngineConfig(**kw)


def _cm(**kw) -> CostModel:
    return CostModel(**kw)


# ── 双边显性成本 ──────────────────────────────────────────

def test_explicit_costs_default_passes() -> None:
    assert check_double_sided_explicit_costs(_cm()).passed is True
    assert check_double_sided_explicit_costs(_cfg()).passed is True


def test_commission_below_floor_fails() -> None:
    report = check_double_sided_explicit_costs(_cm(commission_rate=0.0001))
    assert report.passed is False
    assert "佣金" in "；".join(report.details)


def test_stamp_tax_below_policy_fails() -> None:
    report = check_double_sided_explicit_costs(_cm(stamp_tax_rate=0.0001))
    assert report.passed is False
    assert "印花税" in "；".join(report.details)


def test_transfer_fee_zero_fails() -> None:
    report = check_double_sided_explicit_costs(_cm(transfer_fee_rate=0.0))
    assert report.passed is False
    assert "过户费" in "；".join(report.details)


def test_min_commission_below_five_fails() -> None:
    report = check_double_sided_explicit_costs(_cfg(min_commission_per_trade=1.0))
    assert report.passed is False
    assert "最低佣金" in "；".join(report.details)


# ── 固定滑点下限 ──────────────────────────────────────────

def test_slippage_floor_default_passes() -> None:
    assert check_slippage_floor(_cm()).passed is True
    assert check_slippage_floor(_cfg()).passed is True


def test_slippage_below_floor_fails() -> None:
    report = check_slippage_floor(_cm(market_slippage=0.0002))
    assert report.passed is False
    assert "市价单" in "；".join(report.details)


def test_engine_cfg_slippage_zero_fails() -> None:
    report = check_slippage_floor(_cfg(slippage=0.0))
    assert report.passed is False


def test_calc_slippage_enforces_floor() -> None:
    """引擎抬升行为：配置低于 0.05% 时 calc_slippage 输出仍 ≥ 下限。"""
    cm = _cm(market_slippage=0.0001, limit_slippage=0.0001)
    assert cm.calc_slippage(1_000, 1e9) >= 0.0005
    assert cm.calc_slippage(1_000, 1e9, order_type="limit") >= 0.0005


# ── 动态冲击成本 ──────────────────────────────────────────

def test_dynamic_impact_default_passes() -> None:
    assert check_dynamic_impact(_cm()).passed is True


def test_impact_disabled_fails() -> None:
    """全部档位冲击基数被关（0）→ 大单滑点不再上升 → 必须 FAIL。"""
    report = check_dynamic_impact(_cm(liquidity_tier_impact_base=(0.0, 0.0, 0.0, 0.0)))
    assert report.passed is False


def test_impact_scales_with_participation() -> None:
    cm = _cm()
    small = cm.calc_slippage(1_000, 100_000_000, amount_ma20=1e9)
    mid = cm.calc_slippage(5_000_000, 100_000_000, amount_ma20=1e9)
    big = cm.calc_slippage(10_000_000, 100_000_000, amount_ma20=1e9)
    assert small == pytest.approx(cm.market_slippage)  # 低参与率无冲击
    assert mid > small
    assert big > mid


def test_impact_capped_for_extreme_participation() -> None:
    cm = _cm()
    cap = min(cm.impact_cap, *cm.liquidity_tier_cap)
    slip = cm.calc_slippage(1e9, 100_000_000, amount_ma20=1e9)  # 参与率封顶 1.0
    assert slip <= cm.market_slippage + cap + 1e-9


# ── 卖出端完整性（热路径） ─────────────────────────────────

def _sell_trade(amount: float, cost: float, symbol: str = "600000.SH") -> dict:
    return {"time": "2024-05-10", "symbol": symbol, "action": "sell",
            "value": round(amount - cost, 2), "cost": round(cost, 2)}


def test_sell_completeness_passes() -> None:
    cm = _cm()
    amount = 1_000_000.0
    cost = amount * (cm.market_slippage + cm.stamp_tax_rate + cm.transfer_fee_rate) + cm.min_commission_per_trade
    trades = [_sell_trade(amount, cost), _sell_trade(amount, cost)]
    report = check_sell_side_completeness(trades, cm)
    assert report.passed is True
    assert report.n_checked == 2


def test_sell_completeness_catches_missing_explicit_costs() -> None:
    """回归锚点：卖出完全无显性成本（无滑点下限/佣金/过户费）→ 必须 FAIL。"""
    cm = _cm()
    amount = 1_000_000.0
    truncated = amount * cm.stamp_tax_rate
    trades = [_sell_trade(amount, truncated)]
    report = check_sell_side_completeness(trades, cm)
    assert report.passed is False
    assert report.n_violations == 1
    assert "印花税" in "；".join(report.details)


def test_sell_completeness_ignores_buy() -> None:
    cm = _cm()
    trades = [{"time": "2024-05-10", "symbol": "600000.SH", "action": "buy",
               "value": 100.0, "cost": 0.0}]
    report = check_sell_side_completeness(trades, cm)
    assert report.passed is True
    assert report.n_checked == 0


# ── 一站式入口 ────────────────────────────────────────────

def test_run_friction_check_default_passes() -> None:
    cm = _cm()
    amount = 1_000_000.0
    cost = amount * (cm.market_slippage + cm.stamp_tax_rate + cm.transfer_fee_rate) + cm.min_commission_per_trade
    result = run_trading_friction_check(cm, [_sell_trade(amount, cost)])
    assert result["passed"] is True
    assert result["summary"] == "PASS"
    assert len(result["reports"]) == 4


def test_run_friction_check_engine_cfg_without_cost_model_fails() -> None:
    result = run_trading_friction_check(_cfg())
    assert result["passed"] is False
    assert any("CostModel" in "；".join(r.details) for r in result["reports"])


def test_friction_config_hot_path() -> None:
    assert check_trading_friction_config(_cm()).passed is True
    assert check_trading_friction_config(_cfg()).passed is False
    assert check_trading_friction_config(_cm(commission_rate=0.0001)).passed is False


# ── 引擎集成 ───────────────────────────────────────────────

def _run_engine(cm: CostModel | None, seed: int = 11, start: str = "2024-01-01",
                **cfg_kw) -> tuple[list[dict], list[dict]]:
    data = _engine_bars(seed=seed, start=start)
    cfg = EngineConfig(
        initial_cash=1_000_000.0,
        buy_threshold=60,
        max_holdings=6,
        portfolio_method="score_weighted",
        max_position_pct=0.33,
        atr_stop_mult=0.0,
        cost_model=cm,
        **cfg_kw,
    )
    tl, ec = [], []
    _run_single_backtest(data, {}, cfg, tl, ec)
    return tl, ec


def test_engine_sell_costs_cover_explicit_costs() -> None:
    """引擎卖出成本必须覆盖 佣金+印花税+过户费（1.8 核心回归）。"""
    cm = _cm()
    tl, ec = _run_engine(cm)
    assert tl
    report = check_sell_side_completeness(tl, cm)
    assert report.passed is True, report.details
    assert report.n_checked >= 1


def test_engine_sell_cost_exceeds_stamp_and_slippage_only() -> None:
    """修复回归：卖出 cost/value 比例必须高于 滑点+印花税（旧缺陷漏佣金/过户费）。"""
    cm = _cm()
    tl, _ = _run_engine(cm)
    sell_rows = [t for t in tl if str(t["action"]).startswith("sell")]
    assert sell_rows
    for t in sell_rows:
        amount = float(t["value"]) + float(t["cost"])
        assert float(t["cost"]) > amount * (cm.market_slippage + cm.stamp_tax_rate) + 1e-6


def test_engine_runs_with_tiered_cost_model() -> None:
    """引擎挂载分档 CostModel 全流程不崩且产出成本为正。"""
    cm = _cm()
    tl, _ = _run_engine(cm)
    assert tl
    for t in tl:
        assert float(t["cost"]) > 0.0


def test_engine_friction_hot_path_passes_with_cost_model() -> None:
    data = _engine_bars()
    cfg = EngineConfig(initial_cash=1_000_000.0, buy_threshold=60, max_holdings=6,
                       portfolio_method="score_weighted", max_position_pct=0.33,
                       atr_stop_mult=0.0, cost_model=_cm())
    tl, ec = [], []
    _run_single_backtest(data, {}, cfg, tl, ec)
    assert tl


# ── 政策常量一致性 ─────────────────────────────────────────

def test_stamp_tax_policy_constants() -> None:
    from BackTrading._engine_legacy import _STAMP_TAX_OLD, _STAMP_TAX_RECENT
    assert _STAMP_TAX_RECENT == 0.0005  # 2023-08-28 减半后
    assert _STAMP_TAX_OLD == 0.001


def test_engine_stamp_tax_historical_split() -> None:
    """2023-08-28 前卖出须按 0.1% 印花税，其后 0.05%（分段生效）。"""
    old_tl, _ = _run_engine(_cm(), seed=21, start="2023-05-01")
    new_tl, _ = _run_engine(_cm(), seed=21, start="2024-01-01")
    old_ratios = [(float(t["cost"]) / (float(t["value"]) + float(t["cost"])))
                  for t in old_tl if str(t["action"]).startswith("sell")]
    new_ratios = [(float(t["cost"]) / (float(t["value"]) + float(t["cost"])))
                  for t in new_tl if str(t["action"]).startswith("sell")]
    assert old_ratios and new_ratios
    assert np.mean(old_ratios) >= np.mean(new_ratios) + 0.00025
