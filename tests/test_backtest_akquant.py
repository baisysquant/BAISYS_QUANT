from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from akquant import prepare_dataframe



@pytest.fixture
def backtest_akquant_df() -> pd.DataFrame:
    """模拟多股票 120 日 K 线 + 信号列，经 ``prepare_dataframe`` 标准化。"""
    np.random.seed(42)
    n_symbols = 5
    n_days = 120
    symbols = [f"sh{str(i).zfill(6)}" for i in range(1, n_symbols + 1)]
    trade_dates = pd.bdate_range("2024-01-01", periods=n_days)

    rows: list[dict] = []
    for sym in symbols:
        close = 10.0 * (1 + np.linspace(0, 0.2, n_days)) + np.random.randn(n_days) * 0.3
        close = np.maximum(close, 5.0)
        for i, dt in enumerate(trade_dates):
            rows.append({
                "symbol": sym,
                "trade_date": dt,
                "open": float(close[i] - 0.1),
                "high": float(close[i] + 0.2),
                "low": float(close[i] - 0.2),
                "close": float(close[i]),
                "volume": int(np.random.uniform(1e6, 1e7)),
                "amount": float(close[i] * 1e6),
                "进场评分": float(np.random.uniform(30, 95)),
                "退出评分": float(np.random.uniform(10, 80)),
                "风险等级": str(np.random.choice(["LOW", "MEDIUM", "HIGH"], p=[0.5, 0.3, 0.2])),
                "止损价": float(close[i] * 0.95),
            })

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return prepare_dataframe(df, date_col="trade_date", tz="Asia/Shanghai")


def test_quant_pipeline_strategy_import() -> None:
    from BackTrading.akquant_strategy import QuantPipelineStrategy, QuantPipelineParams
    assert hasattr(QuantPipelineStrategy, "on_bar")
    assert hasattr(QuantPipelineStrategy, "PARAM_MODEL")
    assert QuantPipelineStrategy.PARAM_MODEL is QuantPipelineParams


@pytest.mark.slow
def test_run_backtest(backtest_akquant_df: pd.DataFrame) -> None:
    from akquant import run_backtest
    from BackTrading.akquant_strategy import QuantPipelineStrategy

    result = run_backtest(
        data=backtest_akquant_df,
        strategy=QuantPipelineStrategy,
        initial_cash=1_000_000,
        t_plus_one=False,
        show_progress=False,
    )
    assert hasattr(result, "metrics")
    assert hasattr(result.metrics, "total_return")
    assert hasattr(result.metrics, "sharpe_ratio")


@pytest.mark.slow
def test_run_bayesian_walk_forward(backtest_akquant_df: pd.DataFrame) -> None:
    from BackTrading.calibration import run_bayesian_walk_forward

    result_df = run_bayesian_walk_forward(
        kline_df=backtest_akquant_df,
        train_period=60,
        test_period=10,
        num_paths=2,
        initial_cash=1_000_000,
    )
    assert not result_df.empty
    assert "params" in result_df.columns
    assert "sharpe_ratio" in result_df.columns or "oos_sharpe" in result_df.columns
