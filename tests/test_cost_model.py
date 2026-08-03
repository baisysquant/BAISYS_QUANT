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


def _make_engine_data() -> pd.DataFrame:
    """两只同市值股票：600001 微盘（AMOUNT_MA20=100万），600002 大盘（=10亿）。

    Day1 无信号（仅累计 ADV），Day2 双双买入，Day3 双双卖出（风险 HIGH）。
    """
    amounts = {"600001": 1_000_000.0, "600002": 1_000_000_000.0}
    rows = []
    for day in range(1, 6):
        for sym in ("600001", "600002"):
            buy = 100 if day == 2 else 0
            risk = "HIGH" if day == 3 else "LOW"
            rows.append(
                {
                    "trade_date": f"2024-01-0{day}",
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
    from BackTrading._engine_legacy import _run_single_backtest
    from BackTrading.engine import EngineConfig

    data = _make_engine_data()
    cm = CostModel()
    engine_cfg = EngineConfig(
        cost_model=cm,
        max_position_pct=0.5,
        max_holdings=2,
        point_in_time=False,
    )
    tl: list[dict] = []
    ec: list[dict] = []
    _run_single_backtest(data, {}, engine_cfg, tl, ec)

    buys = {t["symbol"]: t for t in tl if t["action"] == "buy"}
    sells = {t["symbol"]: t for t in tl if t["action"] in ("sell", "sell_partial")}
    assert set(buys) == {"600001", "600002"}
    assert set(sells) == {"600001", "600002"}

    # 引擎将当日成交量作为 order volume → 参与率 ≈ 1.0，两档均触达各自冲击上限
    # 微盘档 cap=10% 显著高于大盘档 cap=3%，小票成本更高（纠正小票收益虚高）
    assert buys["600001"]["cost"] > buys["600002"]["cost"]
    assert sells["600001"]["cost"] > sells["600002"]["cost"]

    expected_large = (
        sells["600002"]["value"]
        * (
            cm.market_slippage
            + cm.liquidity_tier_cap[3]  # 大盘档冲击上限 3%
            + cm.stamp_tax_rate
            + cm.transfer_fee_rate
        )
        + cm.min_commission_per_trade
    )
    assert sells["600002"]["cost"] == pytest.approx(expected_large, rel=0.05)
    assert sells["600001"]["cost"] / sells["600001"]["value"] > 0.09
    assert sells["600002"]["cost"] / sells["600002"]["value"] < 0.05


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
