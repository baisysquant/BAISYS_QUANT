"""1.7 信号执行滞后校验（Signal Execution Lag Integrity）测试。

覆盖：
  - check_execution_price            成交价时点合规
  - check_same_day_contribution     信号日收益贡献归零
  - check_return_multiplier_alignment 收益乘数对齐
  - run_execution_lag_check         一站式
  - _engine_legacy 集成（收盘成交模型）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from BackTrading._engine_legacy import _run_single_backtest
from LogicAnalyzer.ml.execution_lag_integrity import (
    check_execution_price,
    check_price_vs_close_adj,
    check_return_multiplier_alignment,
    check_same_day_contribution,
    run_execution_lag_check,
)

# ── fixtures ───────────────────────────────────────────────

def _dates(n: int, start: str = "2024-01-01") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def _make_bars(n_days: int = 60, n_syms: int = 3, seed: int = 7) -> pd.DataFrame:
    """构造 3 支股票 × n_days 交易日的面板（含 close_adj/open/high/low）。"""
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    rows = []
    for i, sym in enumerate(["600000.SH", "600001.SH", "600002.SH"]):
        base = 10.0 + i
        close_prev = base
        for d in dates:
            close = close_prev * (1 + rng.normal(0.0, 0.01))
            close = max(close, 1.0)
            open_ = close_prev * (1 + rng.normal(0.0008, 0.005))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.003)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.003)))
            rows.append({
                "trade_date": d, "symbol": sym,
                "open": float(open_), "high": float(high), "low": float(low),
                "close": float(close), "close_adj": float(close),
            })
            close_prev = close
    return pd.DataFrame(rows)


def _trade(time: str, symbol: str, action: str, price: float) -> dict:
    return {"time": time, "symbol": symbol, "action": action, "price": price}


# ── check_execution_price ─────────────────────────────────

def test_price_aligned_close_passes() -> None:
    bars = _make_bars()
    trades = [_trade(bars.iloc[i]["trade_date"], bars.iloc[i]["symbol"], "buy",
                     float(bars.iloc[i]["close_adj"]))
              for i in (0, 5, 40)]
    report = check_execution_price(trades, bars, exec_mode="close")
    assert report.passed is True
    assert report.n_violations == 0
    assert report.n_checked == 3


def test_price_aligned_all_rows_passes() -> None:
    bars = _make_bars()
    trades = [
        _trade(r["trade_date"], r["symbol"], "buy", float(r["close_adj"]))
        for r in bars.iloc[::7].to_dict("records")
    ]
    report = check_execution_price(trades, bars, exec_mode="close")
    assert report.passed is True


def test_same_day_open_execution_fails() -> None:
    """信号日 open 成交 = 当天信号吃当天收益，必须 FAIL。"""
    bars = _make_bars()
    open_price = float(bars.iloc[3]["open"])
    trades = [_trade("2024-01-04", "600000.SH", "buy", open_price)]
    report = check_execution_price(trades, bars, exec_mode="close")
    assert report.passed is False
    assert report.n_violations == 1
    assert "盘中" in "；".join(report.details) or "当日盘中" in "；".join(report.details)


def test_same_day_high_edge_execution_fails() -> None:
    bars = _make_bars()
    high_price = float(bars.iloc[6]["high"])
    trades = [_trade("2024-01-03", "600001.SH", "buy", high_price)]
    report = check_execution_price(trades, bars, exec_mode="close")
    assert report.passed is False


def test_next_open_mode_accepts_next_open_price() -> None:
    bars = _make_bars()
    # 信号日 2024-01-04 buy，次日(2024-01-05)开盘价成交 → next_open 模式合规
    next_open = float(bars[(bars["trade_date"] == "2024-01-05")
                           & (bars["symbol"] == "600000.SH")]["open"].iloc[0])
    trades = [_trade("2024-01-04", "600000.SH", "buy", next_open)]
    report = check_execution_price(trades, bars, exec_mode="next_open")
    assert report.passed is True
    # 同一成交在 close 模式下反而 FAIL（当日收盘价 != 次日开盘价）
    report_close = check_execution_price(trades, bars, exec_mode="close")
    assert report_close.passed is False


def test_missing_bar_skipped_not_failed() -> None:
    bars = _make_bars()
    trades = [_trade("2024-09-01", "600000.SH", "buy", 10.0)]  # 不在面板内
    trades.append(_trade("2024-01-02", "600003.SH", "buy", 10.0))  # 标的缺失
    report = check_execution_price(trades, bars, exec_mode="close")
    assert report.passed is True
    assert report.n_checked == 0


def test_invalid_mode_fails() -> None:
    bars = _make_bars()
    trades = [_trade("2024-01-02", "600000.SH", "buy", 10.0)]
    report = check_execution_price(trades, bars, exec_mode="midday")
    assert report.passed is False


# ── check_same_day_contribution ────────────────────────────

def test_contribution_zero_when_close_executed() -> None:
    bars = _make_bars()
    row = bars.iloc[4]
    trades = [_trade(row["trade_date"], row["symbol"], "buy", float(row["close_adj"]))]
    report = check_same_day_contribution(trades, bars, exec_mode="close")
    assert report.passed is True
    assert report.n_violations == 0


def test_contribution_nonzero_when_open_executed() -> None:
    bars = _make_bars()
    row = bars.iloc[2]
    trades = [_trade(row["trade_date"], row["symbol"], "buy", float(row["open"]))]
    report = check_same_day_contribution(trades, bars, exec_mode="close")
    assert report.passed is False
    assert report.n_violations == 1
    assert "乘数" in "；".join(report.details)


def test_contribution_next_open_mode() -> None:
    bars = _make_bars()
    next_open = float(bars[(bars["trade_date"] == "2024-01-05")
                           & (bars["symbol"] == "600001.SH")]["open"].iloc[0])
    trades = [_trade("2024-01-04", "600001.SH", "buy", next_open)]
    report = check_same_day_contribution(trades, bars, exec_mode="next_open")
    assert report.passed is True


def test_buy_within_day_open_fails_next_open_mode() -> None:
    """次日模型下仍以当日 open 成交 → 当日记收益，FAIL。"""
    bars = _make_bars()
    row = bars.iloc[0]
    trades = [_trade(row["trade_date"], row["symbol"], "buy", float(row["open"]))]
    report = check_same_day_contribution(trades, bars, exec_mode="next_open")
    assert report.passed is False


# ── check_return_multiplier_alignment ──────────────────────

def test_correct_shifted_multiplier_passes() -> None:
    n = 60
    idx = _dates(n, "2024-02-01")
    rng = np.random.default_rng(42)
    pos = pd.Series(rng.uniform(0.0, 1.0, n), index=idx)
    ret = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    compliant_pnl = pos.shift(1).fillna(0.0) * ret
    report = check_return_multiplier_alignment(pos, ret, compliant_pnl)
    assert report.passed is True


def test_contemporaneous_multiplier_fails() -> None:
    """position × ret（当天信号当天收益）必须被检出。"""
    n = 60
    idx = _dates(n, start="2024-02-01")
    rng = np.random.default_rng(123)
    pos = pd.Series(rng.uniform(0.0, 1.0, n), index=idx)
    ret = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    bad_pnl = pos * ret  # 超前 1 日
    report = check_return_multiplier_alignment(pos, ret, bad_pnl)
    assert report.passed is False
    assert report.n_violations > 0
    assert "超前" in "；".join(report.details)


def test_shifted_vs_contemporaneous_price_diff() -> None:
    """两种实现的重建差异：合规实现应零违规，违规实现违规数>0。"""
    n = 40
    idx = _dates(n, start="2024-03-01")
    rng = np.random.default_rng(7)
    pos = pd.Series(rng.uniform(0.2, 0.9, n), index=idx)
    ret = pd.Series(rng.normal(0.0002, 0.01, n), index=idx)
    ok = check_return_multiplier_alignment(pos, ret, pos.shift(1).fillna(0) * ret)
    bad = check_return_multiplier_alignment(pos, ret, pos * ret)
    assert ok.passed is True
    assert bad.passed is False
    assert bad.n_violations > ok.n_violations


def test_empty_pnl_skips() -> None:
    pos = pd.Series(dtype=float)
    ret = pd.Series(dtype=float)
    report = check_return_multiplier_alignment(pos, ret, pos)
    assert report.n_checked == 0


# ── run_execution_lag_check 一站式 ─────────────────────────

def test_run_aggregates_close_model() -> None:
    bars = _make_bars()
    trades = [
        _trade(bars.iloc[0]["trade_date"], bars.iloc[0]["symbol"], "buy",
               float(bars.iloc[0]["close_adj"])),
        _trade(bars.iloc[8]["trade_date"], bars.iloc[8]["symbol"], "buy",
               float(bars.iloc[8]["close_adj"])),
        _trade(bars.iloc[15]["trade_date"], bars.iloc[15]["symbol"], "sell",
               float(bars.iloc[15]["close_adj"])),
    ]
    result = run_execution_lag_check(trades, bars, exec_mode="close")
    assert result["passed"] is True
    assert len(result["reports"]) == 2


def test_run_aggregates_contamination() -> None:
    bars = _make_bars()
    open_price = float(bars.iloc[5]["open"])
    bad = [_trade("2024-01-03", "600000.SH", "buy", open_price)]
    result = run_execution_lag_check(bad, bars, exec_mode="close")
    assert result["passed"] is False
    assert any(not r.passed for r in result["reports"])


def test_run_with_multiplier_series() -> None:
    n = 40
    idx = _dates(40, start="2024-04-01")
    rng = np.random.default_rng(9)
    pos = pd.Series(rng.uniform(0.0, 1.0, n), index=idx)
    ret = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    pnl = pos * ret  # 违规
    bars = _make_bars(40)
    trades: list[dict] = []
    result = run_execution_lag_check(
        trades, bars, exec_mode="close",
        position_series=pos, asset_returns=ret, observed_pnl=pnl,
    )
    assert len(result["reports"]) == 3
    assert result["passed"] is False
    assert result["reports"][2].passed is False


# ── 热路径轻量校验（引擎运行期） ────────────────────────────

def test_hot_path_passes_when_price_equals_close_adj() -> None:
    trades = [
        {"time": "2024-01-02", "symbol": "600000.SH", "action": "buy",
         "price": 10.5, "close_adj": 10.5},
        {"time": "2024-01-03", "symbol": "600001.SH", "action": "sell",
         "price": 11.2, "close_adj": 11.2},
    ]
    report = check_price_vs_close_adj(trades)
    assert report.passed is True
    assert report.n_checked == 2
    assert report.n_violations == 0


def test_hot_path_catches_open_price_execution() -> None:
    trades = [
        {"time": "2024-01-02", "symbol": "600000.SH", "action": "buy",
         "price": 10.1, "close_adj": 10.5},  # 开盘价成交 ≠ 收盘价
    ]
    report = check_price_vs_close_adj(trades)
    assert report.passed is False
    assert report.n_violations == 1
    assert "收盘价" in "；".join(report.details)


def test_hot_path_skips_legacy_logs_without_field() -> None:
    trades = [
        {"time": "2024-01-02", "symbol": "600000.SH", "action": "buy", "price": 10.5},
    ]
    report = check_price_vs_close_adj(trades)
    assert report.passed is True
    assert report.n_checked == 0


def test_hot_path_catches_high_price_execution() -> None:
    trades = [
        {"time": "2024-01-02", "symbol": "600001.SH", "action": "buy",
         "price": 10.9, "close_adj": 10.5},  # 盘中冲高价成交
    ]
    report = check_price_vs_close_adj(trades)
    assert report.passed is False
    assert report.n_violations == 1


# ── 引擎集成 ───────────────────────────────────────────────

def _engine_bars(n_days: int = 60, n_syms: int = 6, seed: int = 11) -> pd.DataFrame:
    """构造引擎可直接消费的面板（close_adj + 进场/退出评分 + 风险等级）。"""
    rng = np.random.default_rng(seed)
    dates = _dates(n_days, start="2024-01-01")
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
    df = pd.DataFrame(rows)
    return df


def test_engine_trades_pass_execution_lag_check() -> None:
    data = _engine_bars()
    from BackTrading.engine import EngineConfig
    cfg = EngineConfig(
        initial_cash=1_000_000.0,
        max_holdings=6,
        buy_threshold=60,
        portfolio_method="score_weighted",
        max_position_pct=0.33,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage=0.0,
        atr_stop_mult=0.0,
    )
    tl, ec = [], []
    _run_single_backtest(data, {}, cfg, tl, ec)
    assert tl, "引擎应产生成交记录"
    result = run_execution_lag_check(tl, data, exec_mode="close")
    assert result["passed"] is True
    # 收益自次日起计：买入日组合当日收益贡献为 0 —— 由贡献归零检查保证


def test_engine_trade_prices_recorded_at_close() -> None:
    """回归：引擎成交价必须逐笔等于信号日收盘价（复权价语义）。"""
    data = _engine_bars()
    from BackTrading.engine import EngineConfig
    cfg = EngineConfig(initial_cash=1_000_000.0, buy_threshold=30,
                       portfolio_method="score_weighted", max_position_pct=0.5,
                       commission_rate=0.0, transfer_fee_rate=0.0,
                       stamp_tax_rate=0.0, slippage=0.0)
    tl, ec = [], []
    _run_single_backtest(data, {}, cfg, tl, ec)
    rows = []
    for t in tl:
        row = data[(data["trade_date"] == t["time"]) & (data["symbol"] == t["symbol"])]
        rows.append((t["action"], t["symbol"], t["time"], float(t["price"]),
                     float(row.iloc[0]["close_adj"])))
    for action, symbol, time_, px, close_ in rows:
        assert px == close_, f"{action} {symbol} {time_} price={px} != close={close_}"


def test_engine_trades_carry_close_adj_field() -> None:
    """回归：引擎每笔成交必须携带同日 close_adj 字段（热路径自检锚点）。"""
    data = _engine_bars()
    from BackTrading.engine import EngineConfig
    cfg = EngineConfig(initial_cash=1_000_000.0, buy_threshold=30,
                       portfolio_method="score_weighted", max_position_pct=0.5,
                       commission_rate=0.0, transfer_fee_rate=0.0,
                       stamp_tax_rate=0.0, slippage=0.0)
    tl, ec = [], []
    _run_single_backtest(data, {}, cfg, tl, ec)
    assert tl
    for t in tl:
        assert t.get("close_adj") is not None, f"缺少 close_adj: {t}"
        assert abs(float(t["price"]) - float(t["close_adj"])) < 1e-6
    report = check_price_vs_close_adj(tl)
    assert report.passed is True
    assert report.n_checked == len(tl)