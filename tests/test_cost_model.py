from __future__ import annotations

import pandas as pd
import pytest

from BackTrading.domain.models import DEFAULT_TIER_EDGES, CostModel

# ── 流动性分档 ──────────────────────────────────────────────


def test_liquidity_tier_bucketing() -> None:
    cm = CostModel()
    # 边界：5M / 20M / 1亿 → 4 档
    assert cm.n_liquidity_tiers == 4
    assert cm.liquidity_tier(1_000_000) == 0  # 微盘
    assert cm.liquidity_tier(4_999_999) == 0
    assert cm.liquidity_tier(5_000_000) == 1  # 小盘（边界归入高档）
    assert cm.liquidity_tier(10_000_000) == 1
    assert cm.liquidity_tier(50_000_000) == 2  # 中盘
    assert cm.liquidity_tier(200_000_000) == 3  # 大盘
    assert cm.liquidity_tier(1e10) == 3


def test_liquidity_tier_missing_data_returns_minus_one() -> None:
    cm = CostModel()
    assert cm.liquidity_tier(None) == -1
    assert cm.liquidity_tier(float("nan")) == -1
    assert cm.liquidity_tier(0.0) == -1
    assert cm.liquidity_tier(-1.0) == -1
    assert cm.liquidity_tier("abc") == -1


def test_validate_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="分档数"):
        CostModel(
            liquidity_tier_edges=(1e6, 1e8),
            liquidity_tier_impact_base=(0.008, 0.003, 0.0015, 0.001),  # 4 个 ≠ 3 档
        )


def test_validate_raises_on_non_increasing_edges() -> None:
    with pytest.raises(ValueError, match="严格递增"):
        CostModel(
            liquidity_tier_edges=(1e8, 1e6),
            liquidity_tier_impact_base=(0.1, 0.1, 0.1),
            liquidity_tier_threshold=(0.1, 0.1, 0.1),
            liquidity_tier_cap=(0.1, 0.1, 0.1),
        )


def test_validate_raises_on_negative_params() -> None:
    with pytest.raises(ValueError, match="负值"):
        CostModel(liquidity_tier_impact_base=(0.008, 0.003, -0.001, 0.001))


# ── 分档冲击成本 ────────────────────────────────────────────


def test_tier_impact_small_cap_costlier_than_large_cap() -> None:
    cm = CostModel()
    volume, adv = 1_000_000, 50_000_000  # 2% 参与率，超过统一阈值 1%
    slip_micro = cm.calc_slippage(volume, adv, amount_ma20=1_000_000)
    slip_large = cm.calc_slippage(volume, adv, amount_ma20=1_000_000_000)
    # 微盘冲击显著高于大盘（同为 2% 参与率）
    assert slip_micro > slip_large
    # 大盘档阈值 2% > 参与率 2%，无冲击 → 只有基础滑点
    assert slip_large == pytest.approx(cm.market_slippage)


def test_tier_impact_below_threshold_no_impact() -> None:
    cm = CostModel()
    # 微盘档阈值 0.5%，0.2% 参与率不触发冲击
    slip = cm.calc_slippage(100_000, 50_000_000, amount_ma20=1_000_000)
    assert slip == pytest.approx(cm.market_slippage)


def test_tier_impact_capped_by_tier_cap() -> None:
    cm = CostModel()
    # 微盘档 cap=10%；极端参与率下冲击被封顶
    slip = cm.calc_slippage(1e10, 1_000_000, amount_ma20=1_000_000)
    assert slip <= cm.market_slippage + cm.liquidity_tier_cap[0]


def test_uniform_params_backward_compat() -> None:
    """不传 AMOUNT_MA20 时行为与旧版一致（统一冲击参数）。"""
    cm = CostModel()
    volume, adv = 1_000_000, 50_000_000  # 2% 参与率 > 统一阈值 1%
    slip = cm.calc_slippage(volume, adv)
    expected_impact = cm.impact_base * (0.02 / cm.impact_threshold) ** 1.5
    assert slip == pytest.approx(cm.market_slippage + min(expected_impact, cm.impact_cap))


def test_buy_cost_passes_tier_to_slippage() -> None:
    cm = CostModel()
    value, volume, adv = 100_000.0, 2_000_000, 100_000_000
    cost_small = cm.buy_cost(value, volume, adv, amount_ma20=2_000_000)
    cost_large = cm.buy_cost(value, volume, adv, amount_ma20=2_000_000_000)
    assert cost_small > cost_large


def test_sell_cost_passes_tier_to_slippage() -> None:
    cm = CostModel()
    value, volume, adv = 100_000.0, 2_000_000, 100_000_000
    cost_small = cm.sell_cost(value, volume, adv, amount_ma20=2_000_000)
    cost_large = cm.sell_cost(value, volume, adv, amount_ma20=2_000_000_000)
    assert cost_small > cost_large


# ── 引擎透传（AMOUNT_MA20 → 分档冲击成本） ─────────────────


def _make_engine_data(dates: list[str] | None = None) -> pd.DataFrame:
    """两只同市值股票：600001 微盘（AMOUNT_MA20=100万），600002 大盘（=10亿）。

    Day1 无信号（仅累计 ADV），Day2 双双买入，Day3 双双卖出（风险 HIGH）。
    dates 可覆盖日期（默认为 2024-01-01..05，用于印花税分段跨日测试）。
    """
    dates = dates or [f"2024-01-0{d}" for d in range(1, 6)]
    amounts = {"600001": 1_000_000.0, "600002": 1_000_000_000.0}
    rows = []
    for day, dt in enumerate(dates, start=1):
        for sym in ("600001", "600002"):
            buy = 100 if day == 2 else 0
            risk = "HIGH" if day == 3 else "LOW"
            rows.append(
                {
                    "trade_date": dt,
                    "symbol": sym,
                    "close": 10.0,
                    "close_adj": 10.0,
                    "volume": 200_000.0,
                    "AMOUNT_MA20": amounts[sym],
                    "进场评分": buy,
                    "退出评分": 0,
                    "风险等级": risk,
                    "止损价": 0.0,
                    "ATR": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_engine_applies_tiered_impact_costs() -> None:
    from BackTrading.engine import _run_single_backtest
    from BackTrading.engine import EngineConfig

    data = _make_engine_data()
    cm = CostModel()
    engine_cfg = EngineConfig(
        cost_model=cm,
        max_position_pct=0.5,
        max_holdings=2,
        point_in_time=False,
        initial_cash=100_000_000.0,
        execution_model="close",
    )
    tl: list[dict] = []
    ec: list[dict] = []
    _run_single_backtest(data, {}, engine_cfg, tl, ec)

    buys = {t["symbol"]: t for t in tl if t["action"] == "buy"}
    sells = {t["symbol"]: t for t in tl if t["action"] in ("sell", "sell_partial")}
    assert set(buys) == {"600001", "600002"}
    assert set(sells) == {"600001", "600002"}

    # 引擎将实际订单股数（受 ADV×10% 上限约束 = 20,000 股）作为 order volume
    # 买入参与率 10%：微盘档（threshold 0.5%）冲击打满 cap 10%；大盘档（threshold 2%）
    # 冲击 0.001×(0.10/0.02)^1.5 ≈ 1.12%。卖出为分批半仓（10,000 股）→ 参与率 5%
    # → 大盘档冲击 0.001×(0.05/0.02)^1.5 ≈ 0.40%。小票成本仍显著高于大票。
    assert buys["600001"]["cost"] > buys["600002"]["cost"]
    assert sells["600001"]["cost"] > sells["600002"]["cost"]

    sell_part = 0.05  # partial 卖半仓 → 参与率 5%
    impact_large = cm.liquidity_tier_impact_base[3] * (
        sell_part / cm.liquidity_tier_threshold[3]
    ) ** 1.5
    expected_large = (
        sells["600002"]["value"]
        * (cm.market_slippage + impact_large + cm.stamp_tax_rate
           + cm.transfer_fee_rate + cm.handling_fee_rate + cm.csrc_fee_rate)
        + max(sells["600002"]["value"] * cm.commission_rate, cm.min_commission_per_trade)
    )
    assert sells["600002"]["cost"] == pytest.approx(expected_large, rel=0.05)
    assert sells["600001"]["cost"] / sells["600001"]["value"] > 0.09
    assert sells["600002"]["cost"] / sells["600002"]["value"] < 0.05


# ── 印花税日期分段表（配置驱动，替代硬编码） ─────────────────


def test_stamp_tax_segments_date_driven() -> None:
    cm = CostModel()
    # 2023-08-28 财政部减半：其后 0.05%，其前 0.1%
    assert cm.stamp_tax_rate_for("2023-08-27") == pytest.approx(0.001)
    assert cm.stamp_tax_rate_for("2023-08-28") == pytest.approx(0.0005)
    assert cm.stamp_tax_rate_for("2024-01-03") == pytest.approx(0.0005)
    assert cm.stamp_tax_rate_for(None) == pytest.approx(cm.stamp_tax_rate)


def test_stamp_tax_custom_segments() -> None:
    cm = CostModel(stamp_tax_segments=(("2024-01-01", 0.0004), ("2022-01-01", 0.0015)))
    assert cm.stamp_tax_rate_for("2023-06-01") == pytest.approx(0.0015)
    assert cm.stamp_tax_rate_for("2024-06-01") == pytest.approx(0.0004)
    assert cm.stamp_tax_rate_for("2001-01-01") == pytest.approx(cm.stamp_tax_rate)  # 早于所有段 → 回落单值


def test_sell_cost_uses_date_segments() -> None:
    cm = CostModel()
    before = cm.sell_cost(100_000.0, 1_000, 1_000_000, dt="2023-08-27")
    after = cm.sell_cost(100_000.0, 1_000, 1_000_000, dt="2023-09-01")
    assert before - after == pytest.approx(100_000.0 * (0.001 - 0.0005))


# ── 成本拆解（各项占总成本百分比） ───────────────────────────


def test_breakdown_components_sum_to_total() -> None:
    cm = CostModel()
    parts = cm.buy_cost_breakdown(100_000.0, 2_000, 1_000_000)
    assert set(parts) == {"commission", "transfer", "handling", "csrc",
                          "slippage", "impact", "total"}
    # 经手费 + 证管费 确实收取（不再是"定义了但从未收取"）
    assert parts["handling"] == pytest.approx(100_000.0 * cm.handling_fee_rate)
    assert parts["csrc"] == pytest.approx(100_000.0 * cm.csrc_fee_rate)
    # 各分项之和 == total（留 1e-9 浮点误差）
    assert parts["total"] == pytest.approx(
        sum(v for k, v in parts.items() if k != "total"), abs=1e-6
    )
    # 拆解 sum 与 buy_cost 单值一致
    assert cm.buy_cost(100_000.0, 2_000, 1_000_000) == pytest.approx(parts["total"])

    sp = cm.sell_cost_breakdown(100_000.0, 2_000, 1_000_000, dt="2024-01-03")
    assert "stamp" in sp and sp["stamp"] == pytest.approx(100_000.0 * 0.0005)
    assert sp["total"] == pytest.approx(
        sum(v for k, v in sp.items() if k != "total"), abs=1e-6
    )
    assert cm.sell_cost(100_000.0, 2_000, 1_000_000, dt="2024-01-03") == pytest.approx(sp["total"])


def test_buy_cost_breakdown_has_no_stamp() -> None:
    cm = CostModel()
    parts = cm.buy_cost_breakdown(100_000.0, 2_000, 1_000_000)
    assert "stamp" not in parts  # 印花税仅卖出端


def test_from_backtest_config_parses_stamp_segments_and_fees(temp_config_ini: object) -> None:
    from UtilsManager.ConfigParser import Config

    cfg = Config(str(temp_config_ini))
    bt = cfg.app_config.backtest
    cm = CostModel.from_backtest_config(bt)
    assert cm.stamp_tax_rate_for("2023-08-27") == pytest.approx(0.001)
    assert cm.stamp_tax_rate_for("2024-01-03") == pytest.approx(0.0005)
    assert cm.handling_fee_rate == pytest.approx(0.0000341)
    assert cm.csrc_fee_rate == pytest.approx(0.00002)


def test_from_backtest_config_trading_cost_overrides(temp_config_ini: object) -> None:
    """[TRADING_COST] 节提供时覆盖 [BACKTEST] 对应费率（统一成本来源）。"""
    from UtilsManager.ConfigParser import Config

    ini = str(temp_config_ini)
    with open(ini, encoding="utf-8") as f:
        content = f.read()
    content += (
        "\n[TRADING_COST]\n"
        "commission_rate = 0.0005\n"
        "handling_fee_rate = 0.00005\n"
        "stamp_tax_segments = 2024-01-01:0.0004;2022-01-01:0.0015\n"
    )
    with open(ini, "w", encoding="utf-8") as f:
        f.write(content)
    cfg = Config(ini)
    cm = CostModel.from_backtest_config(
        cfg.app_config.backtest, trading_cost=cfg.app_config.trading_cost
    )
    assert cm.commission_rate == pytest.approx(0.0005)
    assert cm.handling_fee_rate == pytest.approx(0.00005)
    assert cm.stamp_tax_rate_for("2023-06-01") == pytest.approx(0.0015)
    assert cm.stamp_tax_rate_for("2024-06-01") == pytest.approx(0.0004)


# ── 配置构建 ────────────────────────────────────────────────


def test_from_backtest_config_parses_tiers(temp_config_ini: object) -> None:
    from UtilsManager.ConfigParser import Config

    cfg = Config(str(temp_config_ini))
    cm = CostModel.from_backtest_config(cfg.app_config.backtest)
    assert cm.liquidity_tier_edges == DEFAULT_TIER_EDGES
    assert cm.liquidity_tier(1_000_000) == 0
    # 统一参数同样被透传
    assert cm.commission_rate == pytest.approx(0.0003)
    assert cm.market_slippage == pytest.approx(0.001)
    # 分档成本生效：微盘 > 大盘
    assert cm.calc_slippage(
        1_000_000, 50_000_000, amount_ma20=1_000_000
    ) > cm.calc_slippage(1_000_000, 50_000_000, amount_ma20=1_000_000_000)


def test_from_backtest_config_invalid_tiers_falls_back(temp_config_ini: object) -> None:
    from UtilsManager.ConfigParser import Config

    ini = str(temp_config_ini)
    with open(ini, encoding="utf-8") as f:
        content = f.read()
    patched = content.replace(
        "[BACKTEST]\n",
        "[BACKTEST]\nLIQUIDITY_TIER_EDGES = 1e8,1e6\nLIQUIDITY_TIER_IMPACT_BASE = 0.1\n",
    )
    with open(ini, "w", encoding="utf-8") as f:
        f.write(patched)
    cfg = Config(ini)
    cm = CostModel.from_backtest_config(cfg.app_config.backtest)
    # 无效分档配置回落默认分档，不中断回测
    assert cm.liquidity_tier_edges == DEFAULT_TIER_EDGES
    assert cm.liquidity_tier(1_000_000) == 0


# ── 引擎级：印花税日期分段（卖出日驱动） ────────────────────


def _run_engine_on(dates: list[str]) -> tuple[list[dict], list[dict]]:
    from BackTrading.engine import _run_single_backtest
    from BackTrading.engine import EngineConfig

    data = _make_engine_data(dates=dates)
    engine_cfg = EngineConfig(
        cost_model=CostModel(),
        max_position_pct=0.5,
        max_holdings=2,
        point_in_time=False,
        initial_cash=100_000_000.0,
        execution_model="close",
    )
    tl: list[dict] = []
    ec: list[dict] = []
    _run_single_backtest(data, {}, engine_cfg, tl, ec)
    return tl, ec


def test_engine_stamp_tax_date_driven() -> None:
    """卖出日跨 2023-08-28 分界：卖出成本差 = 成交额 × (0.1% − 0.05%)。

    两段 K 线除日期外完全一致（价格/量/ADV 相同），仅卖出日的印花税档不同。
    买入 Day2 / 卖出 Day3 与旧测试一致。
    """
    before = ["2023-08-23", "2023-08-24", "2023-08-25", "2023-08-28", "2023-08-29"]
    after = ["2023-08-25", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31"]
    tl_a, _ = _run_engine_on(before)
    tl_b, _ = _run_engine_on(after)

    def _sell_cost(tl: list[dict], sym: str) -> float:
        return sum(t["cost"] for t in tl if t["symbol"] == sym and t["action"] in ("sell", "sell_partial"))

    def _sell_gross(tl: list[dict], sym: str) -> float:
        return sum(
            t["value"] + t["cost"]
            for t in tl
            if t["symbol"] == sym and t["action"] in ("sell", "sell_partial")
        )

    for sym in ("600001", "600002"):
        gross = _sell_gross(tl_a, sym)
        # 成交额相同 → 仅印花税差 (0.001 − 0.0005)
        assert _sell_gross(tl_b, sym) == pytest.approx(gross)
        assert _sell_cost(tl_a, sym) - _sell_cost(tl_b, sym) == pytest.approx(gross * (0.001 - 0.0005))


def test_engine_costs_accumulated_without_raise() -> None:
    """引擎买卖成本进入累计且成本拆解报告在 INFO 级正常输出。"""
    from loguru import logger

    records: list[str] = []
    sink_id = logger.add(records.append, format="{message}", level="INFO")
    try:
        tl, _ = _run_engine_on(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    finally:
        logger.remove(sink_id)
    text = "\n".join(records)
    assert "[成本拆解]" in text
    assert "总成本=" in text
    assert "%" in text
    assert any(t["action"] == "buy" for t in tl)
    assert any(t["action"] in ("sell", "sell_partial") for t in tl)
