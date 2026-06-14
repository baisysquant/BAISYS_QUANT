"""回测示例 — 使用模拟数据，不依赖数据库。

运行:
    uv run python Backtesting/examples/run_example.py

流程:
    1. 生成 3 只股票 252 日模拟 K 线
    2. prepare_backtest_data 预计算信号列
    3. run_backtest 回测
    4. 打印绩效摘要
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from akquant import prepare_dataframe, run_backtest

from Backtesting.akquant_strategy import QuantPipelineStrategy
from Backtesting.prepare import prepare_backtest_data


def _generate_mock_data() -> pd.DataFrame:
    np.random.seed(1)
    n_symbols = 3
    n_days = 252
    symbols = [f"sh{str(i).zfill(6)}" for i in range(1, n_symbols + 1)]
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    rows: list[dict] = []
    for sym in symbols:
        trend = np.random.choice([-0.3, -0.1, 0.1, 0.3])
        close = 20.0 * (1 + np.linspace(0, trend, n_days)) + np.random.randn(n_days) * 0.5
        close = np.maximum(close, 5.0)
        for i, dt in enumerate(dates):
            rows.append({
                "symbol": sym,
                "trade_date": dt,
                "open": float(close[i] - 0.1),
                "high": float(close[i] + np.random.uniform(0.1, 0.5)),
                "low": float(close[i] - np.random.uniform(0.1, 0.5)),
                "close": float(close[i]),
                "volume": int(np.random.uniform(5e5, 5e6)),
                "amount": float(close[i] * np.random.uniform(5e5, 5e6)),
            })
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return prepare_dataframe(df, date_col="trade_date", tz="Asia/Shanghai")


def main() -> None:
    print("正在生成模拟数据...")
    raw = _generate_mock_data()
    print(f"  原始 K 线: {len(raw)} 行, {raw['symbol'].nunique()} 股票")

    print("预计算信号列...")
    prepared = prepare_backtest_data(raw)
    signal_cols = [c for c in prepared.columns if c not in raw.columns]
    print(f"  新增信号列: {signal_cols}")
    print(f"  总列数: {len(prepared.columns)}")

    print("运行回测...")
    result = run_backtest(
        data=prepared,
        strategy=QuantPipelineStrategy,
        initial_cash=1_000_000,
        t_plus_one=False,
        show_progress=False,
    )

    m = result.metrics
    print("\n=== 回测结果 ===")
    print(f"  总收益率: {m.total_return:.2%}")
    print(f"  年化收益率: {m.annual_return:.2%}" if hasattr(m, 'annual_return') else "")
    print(f"  Sharpe Ratio: {m.sharpe_ratio:.2f}" if hasattr(m, 'sharpe_ratio') else "")
    print(f"  最大回撤: {m.max_drawdown:.2%}")
    print(f"  交易次数: {m.total_trades if hasattr(m, 'total_trades') else 'N/A'}")
    print("完成!")


if __name__ == "__main__":
    main()
