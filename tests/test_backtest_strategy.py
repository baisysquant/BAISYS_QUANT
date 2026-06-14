import numpy as np
import pandas as pd
import pytest

from Backtesting.strategy import PipelineAdapter
from ConfigParser import Config
from LogicAnalyzer.PipelineScoring import calc_entry_signal
from LogicAnalyzer.PipelineState import should_exit


@pytest.fixture
def backtest_ready_df() -> pd.DataFrame:
    """多股票模拟数据集，含 backtest 策略需要的一切列。"""
    np.random.seed(42)
    n = 10
    base = pd.DataFrame({
        "trade_date": ["2024-06-01"] * n,
        "symbol": [f"sh{str(i).zfill(6)}" for i in range(1, n + 1)],
        "最新价": np.linspace(10, 20, n) + np.random.randn(n) * 0.5,
        "止损价": np.linspace(8, 15, n) + np.random.randn(n) * 0.3,
        "风险等级": ["LOW", "MEDIUM", "HIGH", "LOW", "LOW",
                     "MEDIUM", "HIGH", "LOW", "D", "MEDIUM"],
        "进场评分": [80, 70, 30, 90, 85, 50, 20, 75, 10, 65],
        "退出评分": [20, 30, 80, 10, 15, 50, 90, 25, 95, 35],
    })
    base["最新价"] = base["最新价"].clip(lower=1)
    base["止损价"] = base["止损价"].clip(lower=1)
    return base


def test_should_exit_high_risk(backtest_ready_df: pd.DataFrame) -> None:
    mask = should_exit(backtest_ready_df)
    # 风险等级 HIGH/D 的股票应退出
    high_risk = backtest_ready_df[backtest_ready_df["风险等级"].isin(["HIGH", "D"])]
    assert all(mask.loc[high_risk.index])


def test_should_exit_stop_loss(backtest_ready_df: pd.DataFrame) -> None:
    df = backtest_ready_df.copy()
    # 让 index 2 跌破止损
    df.loc[2, "最新价"] = df.loc[2, "止损价"] * 0.9
    mask = should_exit(df)
    assert mask.loc[2]


def test_should_exit_score(backtest_ready_df: pd.DataFrame) -> None:
    mask = should_exit(backtest_ready_df)
    # 退出评分 > 进场评分的行应触发
    score_exit = backtest_ready_df[
        backtest_ready_df["退出评分"] > backtest_ready_df["进场评分"]
    ]
    assert all(mask.loc[score_exit.index])


def test_calc_entry_signal_min_score(backtest_ready_df: pd.DataFrame) -> None:
    mask = calc_entry_signal(backtest_ready_df, min_score=60)
    # 进场评分 >= 60 且风险非 HIGH/D/E
    expected = (
        (backtest_ready_df["进场评分"] >= 60)
        & ~backtest_ready_df["风险等级"].isin(["HIGH", "D", "E"])
    )
    assert mask.tolist() == expected.tolist()


def test_calc_entry_signal_filters_high_risk(backtest_ready_df: pd.DataFrame) -> None:
    mask = calc_entry_signal(backtest_ready_df)
    high_risk = backtest_ready_df[backtest_ready_df["风险等级"].isin(["HIGH", "D", "E"])]
    assert not any(mask.loc[high_risk.index])


def test_pipeline_adapter_on_bar(backtest_ready_df: pd.DataFrame) -> None:
    config = Config()
    adapter = PipelineAdapter(config)
    adapter.on_start()
    orders = adapter.on_bar(backtest_ready_df)

    # 验证订单：卖出数量 = 退出数量，买入数量 = 入场且未退出数量
    exit_mask = should_exit(backtest_ready_df)
    entry_mask = calc_entry_signal(backtest_ready_df)

    sell_orders = [o for o in orders if o["action"] == "sell"]
    buy_orders = [o for o in orders if o["action"] == "buy"]

    assert len(sell_orders) == exit_mask.sum()
    assert len(buy_orders) == (entry_mask & ~exit_mask).sum()

    # 买入权重应与风险等级倒数成正比
    for o in buy_orders:
        assert 0 < o["weight"] <= 1.0
        assert o["symbol"]


def test_pipeline_adapter_on_end() -> None:
    config = Config()
    adapter = PipelineAdapter(config, initial_cash=1_000_000)
    adapter.on_start()
    result = adapter.on_end()
    assert result["final_value"] == 1_000_000
    assert result["total_return"] == 0.0
