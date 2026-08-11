"""涨跌停与撮合约束建模（Limit Matching）— 测试。

验收:
1. 涨跌停价计算：主板 10% / 创业板·科创板 20% / 北交所 30% / ST 5%，
   上市初期豁免（创业板/科创板前 5 日、主板注册制后前 5 日、核准制首日 44%/-36%）。
2. 撮合层：涨停/跌停日按可成交量比例部分成交或未成交（一字板比例更低），
   连续涨停逐板衰减；simulate_limit_up_down=false 回退简化撮合（触板一律禁止）。
3. 异常成交（无量/可成交量不足）在日志中以 [撮合约束] 前缀可追溯。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from BackTrading.engine import _run_single_backtest
from BackTrading.engine import EngineConfig
from BackTrading.limit_pricing import (
    MAIN_BOARD_FIRST_DAY_DOWN,
    MAIN_BOARD_FIRST_DAY_UP,
    Board,
    calc_limit_prices,
    classify_board,
    fill_ratio_for,
    limit_prices_for,
    listing_exempt_days,
    lot_size_for,
)


# ── 单元测试：涨跌停价计算 ─────────────────────────────────────────────

def test_classify_board():
    assert classify_board("600000.SH") == Board.MAIN
    assert classify_board("000001.SZ") == Board.MAIN
    assert classify_board("sh600000") == Board.MAIN
    assert classify_board("688001.SH") == Board.STAR
    assert classify_board("300001.SZ") == Board.GEM
    assert classify_board("830001.BJ") == Board.BSE
    assert classify_board("920001.BJ") == Board.BSE
    assert classify_board("430001.BJ") == Board.BSE


def test_calc_limit_prices_rounding():
    up, down = calc_limit_prices(10.0, 0.10, 0.10)
    assert (up, down) == (11.0, 9.0)
    up, down = calc_limit_prices(10.53, 0.10, 0.10)  # 11.583→11.58, 9.477→9.48
    assert (up, down) == (11.58, 9.48)


def test_board_ratios():
    assert limit_prices_for(10.0, "600000.SH").ratio_up == 0.10
    assert limit_prices_for(10.0, "688001.SH").ratio_up == 0.20
    assert limit_prices_for(10.0, "300001.SZ").ratio_up == 0.20
    assert limit_prices_for(10.0, "830001.BJ").ratio_up == 0.30


def test_st_ratio_5pct():
    info = limit_prices_for(10.0, "600000.SH", is_st=True)
    assert info.ratio_up == 0.05
    assert (info.limit_up, info.limit_down) == (10.5, 9.5)


def test_listing_exempt_days():
    assert listing_exempt_days(Board.GEM) == 5
    assert listing_exempt_days(Board.STAR) == 5
    assert listing_exempt_days(Board.BSE) == 1
    assert listing_exempt_days(Board.MAIN, "2023-04-10") == 5
    assert listing_exempt_days(Board.MAIN, "2023-04-09") == 0
    assert listing_exempt_days(Board.MAIN, None) == 0


def test_registered_board_exempt_5_days():
    info = limit_prices_for(10.0, "688001.SH", listing_days=5, trade_date="2024-01-05")
    assert info.exempt and info.limit_up == pytest.approx(20.0)
    info2 = limit_prices_for(10.0, "688001.SH", listing_days=6, trade_date="2024-01-05")
    assert not info2.exempt and info2.ratio_up == 0.20


def test_main_board_first_day_pre_reform_44pct():
    info = limit_prices_for(10.0, "600000.SH", listing_days=1, trade_date="2023-01-05")
    assert info.ratio_up == pytest.approx(MAIN_BOARD_FIRST_DAY_UP)
    assert info.ratio_down == pytest.approx(MAIN_BOARD_FIRST_DAY_DOWN)
    # 上市第 2 日恢复 10%
    info2 = limit_prices_for(10.0, "600000.SH", listing_days=2, trade_date="2023-01-05")
    assert info2.ratio_up == 0.10


def test_no_listing_info_is_conservative():
    info = limit_prices_for(10.0, "600000.SH", listing_days=None)
    assert not info.exempt and info.ratio_up == 0.10


def test_lot_size_for():
    """0.5 每手申报单位：科创板 200 股/手，其余板块（含创业板）100 股/手。"""
    assert lot_size_for("688001.SH") == 200
    assert lot_size_for("689009.SH") == 200
    assert lot_size_for("300001.SZ") == 100
    assert lot_size_for("600000.SH") == 100
    assert lot_size_for("000001.SZ") == 100
    assert lot_size_for("830001.BJ") == 100


def test_fill_ratio_for():
    # 非触板日：high/low 均未触及限价 → 1.0（不限制）
    assert fill_ratio_for(10.5, 10.2, 10.6, 10.0, 11.0, 9.0, 1) == 1.0
    # 0.3 盘中触板（炸板回落）：high 触涨停价但 close 未封住 → tradable_ratio，而非完全无约束
    assert fill_ratio_for(10.5, 10.2, 11.0, 9.0, 11.0, 9.0, 1) == pytest.approx(0.30)
    # 一字涨停：开=收=涨停价 → seal_ratio
    assert fill_ratio_for(11.0, 11.0, 11.0, 9.0, 11.0, 9.0, 1) == pytest.approx(0.05)
    # 盘中涨停（收盘封板但开低于涨停价）→ tradable_ratio
    assert fill_ratio_for(11.0, 10.5, 11.0, 9.0, 11.0, 9.0, 1) == pytest.approx(0.30)
    # 连板衰减：2 板 = 0.30 × 0.5
    assert fill_ratio_for(11.0, 10.5, 11.0, 9.0, 11.0, 9.0, 2) == pytest.approx(0.15)
    # 一字 2 板 = 0.05 × 0.5
    assert fill_ratio_for(11.0, 11.0, 11.0, 9.0, 11.0, 9.0, 2) == pytest.approx(0.025)
    # 跌停侧对称
    assert fill_ratio_for(9.0, 9.0, 9.0, 9.0, 11.0, 9.0, 1) == pytest.approx(0.05)
    assert fill_ratio_for(9.0, 9.5, 9.5, 9.0, 11.0, 9.0, 1) == pytest.approx(0.30)
    # 盘中触跌停但收回（low 触及跌停价，收盘远离）→ tradable_ratio
    assert fill_ratio_for(10.2, 10.0, 10.3, 9.0, 11.0, 9.0, 1) == pytest.approx(0.30)


# ── 撮合引擎集成测试 ───────────────────────────────────────────────────

def _df(symbol: str, rows: list[tuple], start: str = "2024-01-01") -> pd.DataFrame:
    """rows: (open, close, volume, 进场评分, 退出评分, 风险等级)。"""
    n = len(rows)
    dates = pd.date_range(start, periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    recs = []
    for i, (op, cl, vol, en, ex, risk) in enumerate(rows):
        recs.append({
            "trade_date": dates[i],
            "symbol": symbol,
            "open": op,
            "high": max(op, cl) * 1.01,
            "low": min(op, cl) * 0.99,
            "close": cl,
            "close_adj": cl,
            "ATR": abs(cl - min(op, cl) * 0.99),
            "volume": vol,
            "进场评分": en,
            "退出评分": ex,
            "风险等级": risk,
        })
    return pd.DataFrame(recs)


def _run(data: pd.DataFrame, params: dict | None = None, **cfg_kw):
    cfg_kw.setdefault("execution_model", "close")
    cfg = EngineConfig(
        initial_cash=1_000_000.0,
        buy_threshold=15,
        max_holdings=5,
        portfolio_method="score_weighted",
        max_position_pct=0.33,
        atr_stop_mult=0.0,
        **cfg_kw,
    )
    tl, ec = [], []
    _run_single_backtest(data, params or {}, cfg, tl, ec)
    return tl, ec


def _capture_log(fn):
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), format="{message}")
    try:
        fn()
    finally:
        logger.remove(sink_id)
    return records


def _buys(tl, day: str) -> list[dict]:
    return [r for r in tl if r["action"] == "buy" and str(r["time"]) == day]


def _sells(tl, day: str) -> list[dict]:
    return [r for r in tl if r["action"].startswith("sell") and str(r["time"]) == day]


def test_engine_seal_limit_up_zero_volume_unfilled():
    """一字涨停 + 零成交 → 未成交（不下单），日志可追溯。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),     # 基准日（首日豁免）
        (11.0, 11.0, 0, 90, 0, "LOW"),            # 一字涨停 无量
    ])
    logs: list[str] = []

    def _go():
        nonlocal_logs = _capture_log(lambda: _run(data))
        logs.extend(nonlocal_logs)

    _go()
    assert _buys(_run(data)[0], "2024-01-02") == []
    assert any("撮合约束" in r and "涨停无量" in r for r in logs)


def test_engine_seal_limit_up_tiny_volume_unfilled():
    """一字涨停 + 极小量（5%×500 < 一手）→ 未成交。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (11.0, 11.0, 500, 90, 0, "LOW"),          # 可成交量 25 股 < 100
    ])
    tl, _ = _run(data)
    assert _buys(tl, "2024-01-02") == []


def test_engine_intraday_limit_up_partial_fill():
    """盘中涨停：开 < 涨停价，可成交量比例 0.30 → 部分成交。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (10.5, 11.0, 20_000, 90, 0, "LOW"),       # 涨停 11.0；avail=6000 < 请求 7500
    ])
    tl, _ = _run(data)
    buys = _buys(tl, "2024-01-02")
    assert len(buys) == 1
    assert buys[0]["qty"] == 6_000
    assert buys[0]["limit"] == "up"
    assert buys[0]["fill_ratio"] == pytest.approx(0.30)


def _row(d: str, op, cl, hi, lo, vol, en=0, ex=0, risk="LOW") -> dict:
    return {
        "trade_date": d,
        "symbol": "600000.SH",
        "open": op,
        "high": hi,
        "low": lo,
        "close": cl,
        "close_adj": cl,
        "ATR": cl - lo,
        "volume": vol,
        "进场评分": en,
        "退出评分": ex,
        "风险等级": risk,
    }


def test_engine_intraday_limit_up_break_pullback_buy_capped():
    """0.3 盘中触板（炸板回落）：high 触涨停价但 close 未封住 → 当日买入按 tradable_ratio 折算，
    而非完全无约束。"""
    data = pd.DataFrame([
        _row("2024-01-01", 10.0, 10.0, 10.1, 9.9, 1_000_000),   # 前收 10 → 涨停价 11.0
        _row("2024-01-02", 10.2, 10.6, 11.0, 10.1, 20_000, en=90),  # high 触 11.0，close 10.6 < 11
    ])
    tl, _ = _run(data)
    buys = _buys(tl, "2024-01-02")
    assert len(buys) == 1
    assert buys[0]["qty"] == 6_000                  # 0.30×20000
    assert buys[0]["limit"] == "up"
    assert buys[0]["fill_ratio"] == pytest.approx(0.30)


def test_engine_intraday_limit_down_break_pullback_sell_capped():
    """0.3 盘中触跌停（收回）：low 触及跌停价但 close 未封住 → 当日卖出按 tradable_ratio 折算。"""
    data = pd.DataFrame([
        _row("2024-01-01", 10.0, 10.0, 10.1, 9.9, 1_000_000, en=90),  # 买入日
        _row("2024-01-02", 10.0, 10.2, 10.3, 9.0, 10_000, ex=0, risk="HIGH"),  # low 触 9.0，收 10.2
    ])
    tl, _ = _run(data)
    sells = _sells(tl, "2024-01-02")
    assert len(sells) == 1
    assert sells[0]["action"] == "sell_partial"
    assert sells[0]["qty"] == 3_000                  # 0.30×10000
    assert sells[0]["limit"] == "down"
    assert sells[0]["fill_ratio"] == pytest.approx(0.30)


def test_engine_consecutive_limit_ups_decay():
    """连续涨停：第 1 板一字无量未成交；第 2 板可成交量 0.05×0.5=0.025。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (11.0, 11.0, 30_000, 90, 0, "LOW"),       # 1 板 一字涨停 → 0.05×30000=1500
        (12.1, 12.1, 30_000, 0, 0, "LOW"),        # 2 板（次日不再买入，验持仓）
    ])
    tl, _ = _run(data)
    buys = _buys(tl, "2024-01-02")
    assert len(buys) == 1 and buys[0]["qty"] == 1_500
    assert buys[0]["fill_ratio"] == pytest.approx(0.05)  # 1 板一字：0.05×0.5^0


def test_engine_fast_limit_up_two_boards_partial():
    """快速涨停连续 2 板（盘中）：第 1 板 0.30，第 2 板 0.30×0.5=0.15。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (10.5, 11.0, 20_000, 90, 0, "LOW"),       # 1 板盘中涨停 → 6000
        (11.5, 12.1, 20_000, 0, 0, "LOW"),
        (12.5, 13.3, 20_000, 0, 0, "LOW"),        # 3 板未买入日（仅验证前 2 板）
    ])
    tl, _ = _run(data)
    buys = _buys(tl, "2024-01-02")
    assert len(buys) == 1 and buys[0]["qty"] == 6_000

    # 换一只股票：第 2 板买入 → 0.15
    data2 = _df("600001.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (10.5, 11.0, 20_000, 0, 0, "LOW"),        # 第 1 板（无买入）
        (11.5, 12.1, 20_000, 90, 0, "LOW"),       # 第 2 板 → avail=0.15×20000=3000
    ])
    tl2, _ = _run(data2)
    buys2 = _buys(tl2, "2024-01-03")
    assert len(buys2) == 1 and buys2[0]["qty"] == 3_000
    assert buys2[0]["fill_ratio"] == pytest.approx(0.15)


def test_engine_limit_down_sell_partial_fill():
    """一字跌停卖出：可成交量比例 0.05 → 部分成交 1000 股。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 90, 0, "LOW"),    # 买入日
        (9.0, 9.0, 20_000, 0, 0, "HIGH"),         # 一字跌停 + 风险 HIGH → 强卖
    ])
    tl, _ = _run(data)
    sells = _sells(tl, "2024-01-02")
    assert len(sells) == 1
    assert sells[0]["action"] == "sell_partial"
    assert sells[0]["qty"] == 1_000              # 0.05×20000
    assert sells[0]["limit"] == "down"


def test_engine_limit_down_zero_volume_sell_unfilled():
    """跌停无量 → 卖出未成交（日志可追溯）。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 90, 0, "LOW"),
        (9.0, 9.0, 0, 0, 0, "HIGH"),
    ])
    logs: list[str] = []

    def _go():
        records = _capture_log(lambda: _run(data))
        logs.extend(records)

    _go()
    assert _sells(_run(data)[0], "2024-01-02") == []
    assert any("撮合约束" in r and "跌停无量" in r for r in logs)


def test_engine_flag_off_simplified_no_limit_trades():
    """simulate_limit_up_down=false：回退简化撮合，涨停禁买 / 跌停禁卖。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 90, 0, "LOW"),
        (11.0, 11.0, 1_000_000, 90, 0, "LOW"),   # 涨停日（简化模型：完全不买）
        (9.0, 9.0, 1_000_000, 0, 0, "HIGH"),      # 跌停日（简化模型：完全不卖）
    ])
    tl, _ = _run(data, simulate_limit_up_down=False)
    assert _buys(tl, "2024-01-02") == []
    assert _sells(tl, "2024-01-03") == []
    assert len(_buys(tl, "2024-01-01")) == 1      # 正常日买入不受影响


def test_engine_normal_day_full_fill():
    """非涨跌停日：可成交量比例 1.0，不受撮合约束影响。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (10.2, 10.5, 1_000_000, 90, 0, "LOW"),   # 涨幅 5%，非涨停
    ])
    tl, _ = _run(data)
    buys = _buys(tl, "2024-01-02")
    assert len(buys) == 1
    assert buys[0]["qty"] == 7_800              # 82500/10.5 → 7857→7800
    assert "limit" not in buys[0]


def test_engine_star_buy_lot_200():
    """0.5 科创板买入按 200 股/手申报：tv≈82500/9.9→原始 8333 股 → 200 手=8200，
    而非按 100 股/手取 8300。"""
    data = _df("688001.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (10.0, 9.9, 1_000_000, 90, 0, "LOW"),   # 科创板 20% 涨跌幅内，正常买入
    ])
    tl, _ = _run(data)
    buys = _buys(tl, "2024-01-02")
    assert len(buys) == 1
    assert buys[0]["qty"] == 8_200               # 41 手 × 200
    assert buys[0]["qty"] % 200 == 0


def test_engine_st_stock_5pct_limit():
    """ST 股涨跌幅 5%：收盘 =5% 即涨停（受可成交量约束），<5% 不受限。"""
    st_hist = {"600000.SH": {"2024-01-02": (True, False)}}
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (10.3, 10.5, 20_000, 90, 0, "LOW"),      # 10.5 = 5% 涨停价（盘中）→ cap
    ])
    tl, _ = _run(data, {"_st_history": st_hist, "_exclude_st": False})
    buys = _buys(tl, "2024-01-02")
    assert len(buys) == 1 and buys[0]["qty"] == 6_000  # 0.30×20000
    assert buys[0]["fill_ratio"] == pytest.approx(0.30)

    data2 = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 0, 0, "LOW"),
        (10.4, 10.4, 1_000_000, 90, 0, "LOW"),   # 4% < 5% → 不触板
    ])
    tl2, _ = _run(data2, {"_st_history": st_hist, "_exclude_st": False})
    buys2 = _buys(tl2, "2024-01-02")
    assert len(buys2) == 1 and "limit" not in buys2[0]


def test_engine_next_open_seal_limit_up_cancels_pending_buy():
    """执行模型 next_open：信号日收盘的买入挂单遇次日一字涨停 → 撤销（不可买入）。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 90, 0, "LOW"),     # 信号日（进场评分 90）
        (11.0, 11.0, 1_000_000, 0, 0, "LOW"),      # 一字涨停：前收 10 → 涨停价 11；开=收=11
        (10.9, 11.0, 1_000_000, 0, 0, "LOW"),      # 正常日
    ])
    logs: list[str] = []

    def _go():
        records = _capture_log(lambda: _run(data, execution_model="next_open"))
        logs.extend(records)

    _go()
    tl, _ = _run(data, execution_model="next_open")
    assert _buys(tl, "2024-01-02") == []
    assert all(t["action"] != "buy" for t in tl)      # 挂单撤销后全程无买入
    assert any("一字涨停 → 买入未成交（撤销）" in r for r in logs)


def test_engine_next_open_seal_limit_down_cancels_pending_sell():
    """执行模型 next_open：卖出挂单遇次日一字跌停 → 撤销（当日不可卖出）；
    后续出现新卖出信号时，下一正常日开盘成交。"""
    data = _df("600000.SH", [
        (10.0, 10.0, 1_000_000, 90, 0, "LOW"),     # 信号日买入
        (10.2, 10.5, 1_000_000, 0, 0, "LOW"),      # 正常日：开盘成交
        (10.3, 10.0, 1_000_000, 0, 90, "LOW"),     # 信号日卖出 → 挂单
        (9.0, 9.0, 1_000_000, 0, 0, "LOW"),        # 一字跌停：前收 10 → 跌停价 9；开=收=9
        (9.5, 9.8, 1_000_000, 0, 90, "LOW"),       # 新卖出信号 → 重新挂单
        (9.7, 10.0, 1_000_000, 0, 0, "LOW"),       # 正常日：开盘成交 9.7
    ])
    logs: list[str] = []

    def _go():
        records = _capture_log(lambda: _run(data, execution_model="next_open"))
        logs.extend(records)

    _go()
    tl, _ = _run(data, execution_model="next_open")
    buys = [t for t in tl if t["action"] == "buy" and t["symbol"] == "600000.SH"]
    assert len(buys) == 1
    assert str(buys[0]["time"]) == "2024-01-02" and buys[0]["price"] == pytest.approx(10.2)
    assert _sells(tl, "2024-01-04") == []          # 跌停日未成交
    sells = [t for t in tl if t["action"].startswith("sell")]
    assert len(sells) == 1
    assert str(sells[0]["time"]) == "2024-01-08" and sells[0]["price"] == pytest.approx(9.7)
    assert any("一字跌停 → 卖出未成交（撤销）" in r for r in logs)


# ── 0.4 停牌盯市（Suspension Mark-to-Market）──────────────────────────────

def _mk_df(rows: list[dict]) -> pd.DataFrame:
    """rows: 每行含 trade_date/symbol/open/high/low/close/close_adj/volume/进场评分/退出评分/风险等级。"""
    return pd.DataFrame(rows)


def _bar(d, s, op, cl, hi, lo, vol, en=0, ex=0, risk="LOW") -> dict:
    return {
        "trade_date": d,
        "symbol": s,
        "open": op,
        "high": hi,
        "low": lo,
        "close": cl,
        "close_adj": cl,
        "ATR": cl - lo,
        "volume": vol,
        "进场评分": en,
        "退出评分": ex,
        "风险等级": risk,
    }


def test_engine_suspension_mark_to_market_and_metric():
    """0.4 停牌盯市：持仓在停牌日按停牌前最后收盘价估值（不再冻结在买入成本价），
    且 equity_curve 记录 susp_value_ratio（停牌持仓市值占比）。"""
    data = _mk_df([
        # 2024-01-01（周一）：A 建仓（收盘 10.0）
        _bar("2024-01-01", "600000.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000, en=90),
        _bar("2024-01-01", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        # 2024-01-02：A 正常交易 → 收 11.0（+10%），持仓浮盈已计入
        _bar("2024-01-02", "600000.SH", 10.8, 11.0, 11.1, 10.7, 1_000_000),
        _bar("2024-01-02", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        # 2024-01-03：A 停牌（无 bar），B 正常交易
        _bar("2024-01-03", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        # 2024-01-04：A 复牌
        _bar("2024-01-04", "600000.SH", 10.4, 10.5, 10.6, 10.3, 1_000_000),
        _bar("2024-01-04", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
    ])
    tl, ec = _run(data)
    buys = [t for t in tl if t["action"] == "buy" and t["symbol"] == "600000.SH"]
    assert len(buys) == 1 and str(buys[0]["time"]) == "2024-01-01"
    shares = int(buys[0]["qty"])

    ec_by_day = {str(r["time"]): r for r in ec}
    v_d2 = ec_by_day["2024-01-02"]["portfolio_value"]     # A 已含 11.0 浮盈
    v_d3 = ec_by_day["2024-01-03"]["portfolio_value"]     # 停牌日：应继续按 11.0 盯市
    # 停牌日无任何成交 → 净值必须保持 01-02（不得回落到成本价 10.0 口径）
    assert v_d3 == pytest.approx(v_d2, abs=0.01)

    r3 = ec_by_day["2024-01-03"]
    assert "susp_value_ratio" in r3 and r3["susp_value_ratio"] > 0
    expect_ratio = (shares * 11.0) / v_d3
    assert r3["susp_value_ratio"] == pytest.approx(expect_ratio, rel=1e-4)
    assert "susp_value_ratio" not in ec_by_day["2024-01-02"]   # 当日有行情 → 无停牌占比


def test_engine_resume_after_suspension_executes_deferred_order():
    """0.4 复牌补执行：next_open 挂单遇停牌 → 顺延至复牌日开盘成交。"""
    data = _mk_df([
        # 2024-01-01：A 信号日（next_open → 收盘挂单）
        _bar("2024-01-01", "600000.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000, en=90),
        _bar("2024-01-01", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        # 2024-01-02：A 停牌（无 bar）
        _bar("2024-01-02", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        # 2024-01-03：A 复牌 → 开盘 10.0 成交
        _bar("2024-01-03", "600000.SH", 10.0, 10.6, 10.7, 9.9, 1_000_000),
        _bar("2024-01-03", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
    ])
    tl, _ = _run(data, execution_model="next_open")
    buys = [t for t in tl if t["action"] == "buy" and t["symbol"] == "600000.SH"]
    assert len(buys) == 1
    assert str(buys[0]["time"]) == "2024-01-03"          # 复牌日成交，而非停牌日
    assert buys[0]["price"] == pytest.approx(10.0)       # 开盘价


def test_engine_suspension_bad_close_adj_falls_back_to_last_close():
    """0.4 停牌盯市硬化：持仓当日有 bar 但 close_adj 为 NaN（行情缺失）→ 回退停牌前
    最后收盘价估值，不得用 NaN 污染总市值。"""
    data = _mk_df([
        _bar("2024-01-01", "600000.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000, en=90),
        _bar("2024-01-02", "600000.SH", 10.8, 11.0, 11.1, 10.7, 1_000_000),
        # 2024-01-03：有 bar 但 close_adj 缺失 → 视为停牌
        {"trade_date": "2024-01-03", "symbol": "600000.SH", "open": 10.9,
         "high": 11.1, "low": 10.8, "close": 11.0, "close_adj": np.nan,
         "ATR": 0.3, "volume": 0, "进场评分": 0, "退出评分": 0, "风险等级": "LOW"},
        _bar("2024-01-04", "600000.SH", 10.4, 10.5, 10.6, 10.3, 1_000_000),
    ])
    tl, ec = _run(data)
    ec_by_day = {str(r["time"]): r for r in ec}
    r3 = ec_by_day["2024-01-03"]
    assert np.isfinite(r3["portfolio_value"])
    assert "susp_value_ratio" in r3 and r3["susp_value_ratio"] > 0
    # NaN 不应导致净值回落到成本价口径：保持 01-02（停牌前最后收盘价 11.0）的盯市结果
    assert r3["portfolio_value"] == pytest.approx(ec_by_day["2024-01-02"]["portfolio_value"], abs=0.01)


# ── 0.6 复牌跳空（Suspension Gap）─────────────────────────────────────────

def test_engine_resume_gap_up_sell_at_open_and_no_rebuy():
    """0.6 复牌高开兑现：持仓停牌后复牌日开盘跳空高开（补涨）→ 开盘价全部卖出，
    且当日不再追高买入。"""
    data = _mk_df([
        _bar("2024-01-01", "600000.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000, en=90),  # 买入
        _bar("2024-01-01", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        _bar("2024-01-02", "600000.SH", 10.2, 10.5, 10.6, 10.1, 1_000_000),  # 正常持有
        _bar("2024-01-02", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        _bar("2024-01-03", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),   # A 停牌
        # 2024-01-04：A 复牌，开盘 11.5（相对停牌前收 10.5 → +9.52% ≥ 5%）→ 兑现卖出；带买入信号 → 禁买
        _bar("2024-01-04", "600000.SH", 11.5, 11.6, 11.7, 11.4, 1_000_000, en=90),
        _bar("2024-01-04", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
    ])
    records: list[str] = []

    def _go():
        sink = logger.add(lambda m: records.append(str(m)), format="{message}")
        try:
            _run(data)
        finally:
            logger.remove(sink)

    _go()
    tl, _ = _run(data)
    buys = [t for t in tl if t["action"] == "buy" and t["symbol"] == "600000.SH"]
    assert len(buys) == 1 and str(buys[0]["time"]) == "2024-01-01"   # 无复牌日买入（禁买）
    qty = int(buys[0]["qty"])
    sells = [t for t in tl if t["action"] == "sell" and t["symbol"] == "600000.SH"]
    assert len(sells) == 1
    assert str(sells[0]["time"]) == "2024-01-04"
    assert sells[0]["price"] == pytest.approx(11.5)     # 开盘价成交
    assert sells[0]["qty"] == qty                        # 全部卖出
    assert any("复牌" in r and "开盘兑现卖出" in r for r in records)
    assert any("复牌" in r and "当日禁买" in r for r in records)


def test_engine_resume_gap_up_pending_buy_cancelled():
    """0.6 复牌高开追高单撤销：next_open 挂单跨停牌顺延后，复牌日跳空高开 → 买入挂单撤销。"""
    data = _mk_df([
        _bar("2024-01-01", "600000.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000, en=90),  # 信号 → 挂单
        _bar("2024-01-01", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        _bar("2024-01-02", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),   # A 停牌
        _bar("2024-01-03", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),   # A 停牌
        # 2024-01-04：A 复牌开盘 11.5（相对停牌前收 10.0 → +15%）→ 挂单撤销
        _bar("2024-01-04", "600000.SH", 11.5, 11.6, 11.7, 11.4, 1_000_000),
        _bar("2024-01-04", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
    ])
    records: list[str] = []

    def _go():
        sink = logger.add(lambda m: records.append(str(m)), format="{message}")
        try:
            _run(data, execution_model="next_open")
        finally:
            logger.remove(sink)

    _go()
    tl, _ = _run(data, execution_model="next_open")
    assert all(t["symbol"] != "600000.SH" or t["action"] != "buy" for t in tl)
    assert any("复牌高开 → 买入挂单撤销" in r for r in records)


def test_engine_resume_small_gap_buy_normal():
    """0.6 复牌小幅跳空（<阈值）→ 不触发兑现/禁买，正常买入。"""
    data = _mk_df([
        _bar("2024-01-01", "600000.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),   # 无信号
        _bar("2024-01-01", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        _bar("2024-01-02", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),   # A 停牌
        # 2024-01-03：A 复牌开盘 10.2（+2% < 5%）→ 正常买入
        _bar("2024-01-03", "600000.SH", 10.2, 10.3, 10.4, 10.1, 1_000_000, en=90),
        _bar("2024-01-03", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
    ])
    tl, _ = _run(data)
    buys = [t for t in tl if t["action"] == "buy" and t["symbol"] == "600000.SH"]
    assert len(buys) == 1
    assert str(buys[0]["time"]) == "2024-01-03"


def test_engine_resume_gap_down_logged_no_forced_sell():
    """0.6 复牌低开（补跌）：仅日志标记，不强制卖出（风控卖出照常判定）。"""
    data = _mk_df([
        _bar("2024-01-01", "600000.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000, en=90),  # 买入
        _bar("2024-01-01", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
        _bar("2024-01-02", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),   # A 停牌
        # 2024-01-03：A 复牌开盘 9.0（-10% ≤ -5%），无卖出信号 → 继续持有
        _bar("2024-01-03", "600000.SH", 9.0, 9.2, 9.3, 8.9, 1_000_000),
        _bar("2024-01-03", "600001.SH", 10.0, 10.0, 10.1, 9.9, 1_000_000),
    ])
    records: list[str] = []

    def _go():
        sink = logger.add(lambda m: records.append(str(m)), format="{message}")
        try:
            _run(data)
        finally:
            logger.remove(sink)

    _go()
    tl, _ = _run(data)
    assert all(t["symbol"] != "600000.SH" or not t["action"].startswith("sell") for t in tl)
    assert any("复牌" in r and "低开" in r for r in records)
