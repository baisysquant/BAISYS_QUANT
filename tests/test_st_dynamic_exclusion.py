"""ST/退市逐日动态剔除（_engine_legacy 消费 stock_st_history）单元测试"""

from __future__ import annotations

import pandas as pd

from BackTrading.engine import EngineConfig, _run_single_backtest

SYM_A = "600001.SH"   # 被标记 ST / 退市的股票
SYM_B = "600002.SH"   # 正常对照股票
DATES = [f"2024-01-{d:02d}" for d in range(1, 11)]
ST_DAY = DATES[5]      # 第 6 个交易日标记 ST/退市


def _bars() -> pd.DataFrame:
    rows = []
    for d in DATES:
        for sym in (SYM_A, SYM_B):
            rows.append({
                "trade_date": d,
                "symbol": sym,
                "close": 10.0,
                "close_adj": 10.0,
                "volume": 1_000_000,
                "进场评分": 90.0,
                "退出评分": 0.0,
                "风险等级": "LOW",
            })
    return pd.DataFrame(rows)


def _run(data: pd.DataFrame, st_history: dict | None, exclude_st: bool = True) -> tuple[list[dict], list[dict]]:
    cfg = EngineConfig(
        initial_cash=1_000_000.0,
        buy_threshold=60,
        max_holdings=10,
        portfolio_method="score_weighted",
        max_position_pct=0.5,
        atr_stop_mult=0.0,
        execution_model="close",
    )
    params = {}
    if st_history is not None:
        params = {"_st_history": st_history, "_exclude_st": exclude_st}
    tl: list[dict] = []
    ec: list[dict] = []
    _run_single_backtest(data, params, cfg, tl, ec)
    return tl, ec


def _buys(tl: list[dict], sym: str, day: str | None = None) -> list[dict]:
    return [t for t in tl if t["action"] == "buy"
            and t["symbol"] == sym and (day is None or t["time"] == day)]


def _sells(tl: list[dict], sym: str, day: str | None = None) -> list[dict]:
    return [t for t in tl if str(t["action"]).startswith("sell")
            and t["symbol"] == sym and (day is None or t["time"] == day)]


class TestExcludeSTTrue:
    def test_stock_never_buyable_when_always_st(self):
        """exclude_st=True + 全程 ST：该股全程禁止买入。"""
        st = {SYM_A: {d: (True, False) for d in DATES}}
        tl, _ = _run(_bars(), st, exclude_st=True)
        assert _buys(tl, SYM_A) == []
        # 对照股不受影响
        assert _buys(tl, SYM_B, DATES[0])

    def test_st_day_blocks_buy_and_force_sells_held(self):
        """ST 标记日：禁止买入 + 已持仓强平（全仓）。"""
        st = {SYM_A: {ST_DAY: (True, False)}}
        tl, _ = _run(_bars(), st, exclude_st=True)

        assert _buys(tl, SYM_A, DATES[0])          # ST 日前正常买入
        assert not _buys(tl, SYM_A, ST_DAY)        # ST 日禁止买入
        # 强平：卖出数量 = ST 日前的全部累计持仓
        buy_qty = sum(int(t.get("qty", 0)) for t in _buys(tl, SYM_A) if t["time"] < ST_DAY)
        sell_qty = sum(int(t.get("qty", 0)) for t in _sells(tl, SYM_A, ST_DAY))
        assert sell_qty == buy_qty > 0
        # 对照股同日无强平（逐日掩码只影响被标记标的）
        assert _sells(tl, SYM_B, ST_DAY) == []

    def test_st_recovery_day_buyable_again(self):
        """ST 解除后恢复正常交易。"""
        st = {SYM_A: {ST_DAY: (True, False)}}
        tl, _ = _run(_bars(), st, exclude_st=True)
        assert _buys(tl, SYM_A, DATES[6])


class TestExcludeSTFalse:
    def test_always_st_still_fully_tradable(self):
        """exclude_st=False：全程 ST 也正常买入（用户诉求: 让 ST 股参与回测）。"""
        st = {SYM_A: {d: (True, False) for d in DATES}}
        tl, _ = _run(_bars(), st, exclude_st=False)
        assert _buys(tl, SYM_A, DATES[0])
        assert _sells(tl, SYM_A) == []  # 无强平

    def test_st_day_no_forced_exit(self):
        """exclude_st=False：ST 标记日不触发强平。"""
        st = {SYM_A: {ST_DAY: (True, False)}}
        tl, _ = _run(_bars(), st, exclude_st=False)
        assert _sells(tl, SYM_A, ST_DAY) == []

    def test_no_st_history_unchanged(self):
        """无 ST 历史时行为与旧版一致（回归保护）。"""
        tl_a, _ = _run(_bars(), None)
        tl_b, _ = _run(_bars(), {}, exclude_st=False)
        assert len(tl_a) == len(tl_b)
        assert _buys(tl_b, SYM_A, DATES[0])


class TestDelisting:
    def test_delisting_force_sell_regardless_of_exclude_st(self):
        """退市日无条件强平，不受 exclude_st 影响。"""
        for exclude_st in (True, False):
            st = {SYM_A: {ST_DAY: (True, True)}}
            tl, _ = _run(_bars(), st, exclude_st=exclude_st)
            buy_qty = sum(int(t.get("qty", 0)) for t in _buys(tl, SYM_A) if t["time"] < ST_DAY)
            sell_qty = sum(int(t.get("qty", 0)) for t in _sells(tl, SYM_A, ST_DAY))
            assert sell_qty == buy_qty > 0, f"exclude_st={exclude_st} 退市未强平"

    def test_delisting_day_blocks_buy(self):
        st = {SYM_A: {ST_DAY: (True, True)}}
        tl, _ = _run(_bars(), st, exclude_st=False)
        assert not _buys(tl, SYM_A, ST_DAY)


class TestMaskIsolation:
    def test_blocked_day_only_affects_marked_symbol(self):
        """逐日掩码只影响对应标的与日期。"""
        st = {SYM_A: {ST_DAY: (True, False)}}
        tl, _ = _run(_bars(), st, exclude_st=True)
        # B 全程正常（第 1 天买入后持仓至期末，无强平）
        assert _buys(tl, SYM_B, DATES[0])
        assert _sells(tl, SYM_B) == []
        # A 仅在 ST 日被强平一次
        assert len(_sells(tl, SYM_A, ST_DAY)) == 1
