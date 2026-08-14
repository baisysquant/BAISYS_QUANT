"""
P0-6 引擎审计修复测试：退市整理期状态机 / 挂单时效 / 复牌价口径 /
上市日显式注入 / 市场状态客观变量 / 集合竞价成交率分档。

覆盖：
  1. _build_day_limit_model 退市整理期涨跌幅优先级（首日无限制 / 期间 ±10%）
  2. _regime_multiplier_for / _vol_percentile 客观状态仓位倍率
  3. 引擎级：退市整理期状态机（期间可交易可买入、仅摘牌日强平、摘牌后禁买）
  4. 引擎级：ST 日仍禁买 + 强平（回归旧语义）
  5. 引擎级：挂单次日过期 + 停牌废单撤销（买入）；强平单停牌逐日重挂
  6. 引擎级：复牌跳空卖出用后复权 open_adj（不复权开盘价已弃用）
  7. 引擎级：开盘集合竞价成交率分档（auction_fill_ratio 封单量代理）
  8. 引擎级：上市日显式注入（_listing_days）运行正常；缺省不再数据推断
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import pytest

from BackTrading.engine import EngineConfig, run_full_backtest
from BackTrading.engine.core import (
    _build_day_limit_model,
    _regime_multiplier_for,
    _vol_percentile,
)

_DAYS = [str(d.date()) for d in pd.bdate_range("2026-01-05", periods=25)]


def _mk_panel(specs: dict[str, dict[int, dict]], n_days: int = 25) -> pd.DataFrame:
    """构建引擎输入面板。

    specs: {symbol: {day_idx: {col: value}}}，未指定的行用默认价 10.0 / 量 1e6；
    指定 day_idx 缺失 → 该标的当日无行（停牌/缺数据）。
    """
    rows = []
    for sym, overrides in specs.items():
        for d in range(n_days):
            ov = overrides.get(d)
            if ov is None:
                continue
            base = {
                "symbol": sym,
                "trade_date": _DAYS[d],
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
                "volume": 1_000_000,
                "open_adj": 10.0, "high_adj": 10.5, "low_adj": 9.5,
                "close_adj": 10.0, "adj_factor": 1.0,
                "AMOUNT_MA20": 1e7,
                "进场评分": 0.0, "退出评分": 0.0, "风险等级": "LOW",
                "止损价": 0.0, "ATR": 1.0,
            }
            base.update(ov)
            rows.append(base)
    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _score(day: int, v: float = 90.0) -> dict:
    return {"进场评分": v}


def _price(day: int, p: float, **kw) -> dict:
    ov = {
        "open": p, "high": p * 1.02, "low": p * 0.98, "close": p,
        "open_adj": p, "high_adj": p * 1.02, "low_adj": p * 0.98,
        "close_adj": p, "adj_factor": 1.0, "volume": 1_000_000,
    }
    ov.update(kw)
    return ov


@pytest.fixture
def flat_engine() -> EngineConfig:
    """全倍率 1.0（隔离市场状态乘数对仓位的干扰）。"""
    return EngineConfig(
        regime_full_multiplier=1.0,
        regime_half_multiplier=1.0,
        regime_min_multiplier=1.0,
    )


# ── 1. 退市整理期涨跌幅（_build_day_limit_model 单元测试） ──


class TestDelistLimitPricing:
    def _limit(self, *, delist_first=(), delist_period=(), st=(), listing_map=None,
               day_str="2026-06-10", day_idx=None):
        syms = np.array(["sh600001"])
        close_raw = np.array([10.0])
        prev_bar = {"sh600001": (10.0, 1.0)}
        lu, ld, *_ = _build_day_limit_model(
            syms, close_raw, None, None, None, prev_bar, set(st),
            day_str, day_idx or {}, listing_map, {}, True,
            0.05, 0.30, 0.10, 0.5,
            delist_first_syms=set(delist_first),
            delist_period_syms=set(delist_period),
        )
        return float(lu[0]), float(ld[0])

    @pytest.mark.unit
    def test_delist_first_day_no_limit(self):
        lu, ld = self._limit(delist_first={"sh600001"})
        assert lu == pytest.approx(20.0)  # ±100% 近似无限制
        assert ld == pytest.approx(0.0)

    @pytest.mark.unit
    def test_delist_period_10pct_overrides_st_5pct(self):
        # 退市整理期次日起 ±10%，即使同时命中 ST 5% 也以整理期为准
        lu, ld = self._limit(delist_period={"sh600001"}, st={"sh600001"})
        assert lu == pytest.approx(11.0)
        assert ld == pytest.approx(9.0)

    @pytest.mark.unit
    def test_st_day_5pct(self):
        lu, ld = self._limit(st={"sh600001"})
        assert lu == pytest.approx(10.5)
        assert ld == pytest.approx(9.5)

    @pytest.mark.unit
    def test_registered_exempt_listing_days(self):
        # 注册制后上市第 2 个交易日 → 前 5 日豁免（无涨跌幅）
        lu, ld = self._limit(
            listing_map={"sh600001": "2026-06-08"},
            day_str="2026-06-10",
            day_idx={"2026-06-08": 0, "2026-06-10": 1},
        )
        assert lu == pytest.approx(20.0)

    @pytest.mark.unit
    def test_registered_exempt_expires_after_5_days(self):
        # 上市第 6 个交易日（索引差 5）→ 豁免结束，回 ±10%
        lu, ld = self._limit(
            listing_map={"sh600001": "2026-06-01"},
            day_str="2026-06-10",
            day_idx={"2026-06-01": 0, "2026-06-10": 5},
        )
        assert lu == pytest.approx(11.0)
        assert ld == pytest.approx(9.0)

    @pytest.mark.unit
    def test_main_board_first_day_44pct_pre_reform(self):
        # 核准制（2023-04-10 前）上市首日 ±44%/-36%（上市日 = 当日）
        lu, ld = self._limit(
            listing_map={"sh600001": "2020-06-10"},
            day_str="2020-06-10",
            day_idx={"2020-06-10": 5},
        )
        assert lu == pytest.approx(14.4)
        assert ld == pytest.approx(6.4)


# ── 2. 客观状态仓位倍率（P0-6 ⑤ 单元测试） ──


class TestRegimeObjective:
    @pytest.mark.unit
    def test_full_when_strong_trend(self):
        assert _regime_multiplier_for(0.05, 0.5, EngineConfig()) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_half_when_mild(self):
        assert _regime_multiplier_for(0.0, 0.5, EngineConfig()) == pytest.approx(0.5)

    @pytest.mark.unit
    def test_min_when_downtrend(self):
        assert _regime_multiplier_for(-0.05, 0.5, EngineConfig()) == pytest.approx(0.25)

    @pytest.mark.unit
    def test_high_vol_suppresses_mild_trend(self):
        # 温和趋势 + 高波动分位(>0.8) → 最低倍率
        assert _regime_multiplier_for(0.0, 0.9, EngineConfig()) == pytest.approx(0.25)

    @pytest.mark.unit
    def test_high_vol_does_not_suppress_strong_trend(self):
        assert _regime_multiplier_for(0.05, 0.9, EngineConfig()) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_vol_percentile(self):
        h = deque([1.0, 2.0, 3.0, 4.0])
        # P0-11：样本不足 _REGIME_VOL_MIN 时返回中性分位 0.5（早期样本少易极端误判）
        assert _vol_percentile(h, 3.0) == pytest.approx(0.5)
        assert _vol_percentile(deque(), 1.0) == pytest.approx(0.5)
        # 足量样本恢复精确分位
        big = deque([1.0] * 61 + [2.0])
        assert _vol_percentile(big, 2.0) == pytest.approx(1.0)


# ── 引擎级场景 ──


def _run(data: pd.DataFrame, params: dict | None = None,
         engine_cfg: EngineConfig | None = None):
    return run_full_backtest(data, params or {}, engine_cfg or EngineConfig())


def _by_symbol(tl: list[dict], sym: str) -> list[dict]:
    return [r for r in tl if r.get("symbol") == sym]


class TestDelistingPeriodStateMachine:
    """P0-6 ①：整理期首日豁免、期间可交易可买入、仅摘牌日强平、摘牌后禁买。"""

    def _scenario(self, flat_engine):
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # B：退市整理期 days[14..19]，摘牌日 = days[19]
            "sh600002": {
                **{d: _price(d, 10.0) for d in range(25)},
                9: _score(9),
                21: _score(21),
            },
            # C：退市整理期 days[14..18]，摘牌日 = days[18]；期间 day14 高分验证可买入
            "sh600003": {
                **{d: _price(d, 10.0) for d in range(25)},
                14: _score(14),
            },
        }
        st_history = {
            "sh600002": {
                _DAYS[d]: (False, True) for d in range(14, 20)
            },
            "sh600003": {
                _DAYS[d]: (False, True) for d in range(14, 19)
            },
        }
        tl, _ = _run(
            _mk_panel(specs),
            {"_st_history": st_history, "_exclude_st": True},
            flat_engine,
        )
        return tl

    @pytest.mark.integration
    def test_period_days_tradable_no_force_exit(self, flat_engine):
        tl = self._scenario(flat_engine)
        b = _by_symbol(tl, "sh600002")
        # 买入执行日 days[10]
        assert any(r["action"] == "buy" and r["time"] == _DAYS[10] for r in b)
        # 整理期 days[15..18] 内无任何强平/卖出（期间可正常持有）
        assert not any(
            r["action"].startswith("sell") and r["time"] in _DAYS[11:19] for r in b
        )

    @pytest.mark.integration
    def test_last_period_day_liquidated_at_close(self, flat_engine):
        tl = self._scenario(flat_engine)
        b = _by_symbol(tl, "sh600002")
        # 摘牌日 days[19] 当日收盘价强平（非次日开盘）
        liq = [r for r in b if r.get("force_exit")]
        assert len(liq) == 1
        assert liq[0]["time"] == _DAYS[19]
        assert liq[0]["price"] == pytest.approx(10.0)
        assert liq[0]["action"] == "sell"

    @pytest.mark.integration
    def test_buy_allowed_during_period(self, flat_engine):
        tl = self._scenario(flat_engine)
        c = _by_symbol(tl, "sh600003")
        # days[14] 信号 → days[15]（整理期、非摘牌日）可买入
        assert any(r["action"] == "buy" and r["time"] == _DAYS[15] for r in c)
        # 摘牌日 days[18] 强平
        liq = [r for r in c if r.get("force_exit")]
        assert len(liq) == 1 and liq[0]["time"] == _DAYS[18]

    @pytest.mark.integration
    def test_post_delist_buy_blocked(self, flat_engine):
        tl = self._scenario(flat_engine)
        b = _by_symbol(tl, "sh600002")
        # 摘牌日之后 days[21] 高分信号 → 禁买（终态兜底）
        assert not any(r["action"] == "buy" and r["time"] in _DAYS[20:] for r in b)


class TestStBlockRegression:
    """P0-6 ① 回归：ST/*ST 日仍禁买 + 强平（非整理期）。"""

    @pytest.mark.integration
    def test_st_day_force_exit_and_buy_block(self, flat_engine):
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # D：days[5] 信号买入（执行 days[6]）；days[7] 进入 ST
            "sh600004": {
                **{d: _price(d, 10.0) for d in range(25)},
                5: _score(5),
            },
            # E：days[7] 为 ST 日且当日高分 → 买入应被拦截
            "sh600005": {
                **{d: _price(d, 10.0) for d in range(25)},
                7: _score(7),
            },
        }
        st_history = {
            "sh600004": {_DAYS[7]: (True, False)},
            "sh600005": {_DAYS[7]: (True, False)},
        }
        tl, _ = _run(
            _mk_panel(specs),
            {"_st_history": st_history, "_exclude_st": True},
            flat_engine,
        )
        d = _by_symbol(tl, "sh600004")
        # ST 日 days[7] 强平挂单 → days[8] 开盘成交
        force = [r for r in d if r.get("force_exit")]
        assert len(force) == 1 and force[0]["time"] == _DAYS[8]
        e = _by_symbol(tl, "sh600005")
        assert not any(r["action"] == "buy" for r in e)


class TestOrderLifecycle:
    """P0-6 ②：挂单次日过期 + 停牌废单撤销；强平单停牌逐日重挂。"""

    @pytest.mark.integration
    def test_buy_order_cancelled_on_suspension(self, flat_engine):
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # F：days[5] 高分信号；days[6] 停牌（无行情行）→ 废单撤销，不复挂
            "sh600006": {
                **{d: _price(d, 10.0) for d in range(25) if d != 6},
                5: _score(5),
            },
        }
        tl, _ = _run(_mk_panel(specs), {}, flat_engine)
        f = _by_symbol(tl, "sh600006")
        assert not any(r["action"] == "buy" for r in f)

    @pytest.mark.integration
    def test_force_sell_defers_through_suspension(self, flat_engine):
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # G：days[3] 信号买入（执行 days[4]）；days[6] 进入 ST；
            # days[7] 停牌（强平单逐日重挂）→ days[8] 复牌开盘成交
            "sh600007": {
                **{d: _price(d, 10.0) for d in range(25) if d != 7},
                3: _score(3),
            },
        }
        st_history = {"sh600007": {_DAYS[6]: (True, False)}}
        tl, _ = _run(
            _mk_panel(specs),
            {"_st_history": st_history, "_exclude_st": True},
            flat_engine,
        )
        g = _by_symbol(tl, "sh600007")
        force = [r for r in g if r.get("force_exit")]
        assert len(force) == 1 and force[0]["time"] == _DAYS[8]


class TestResumeSellUsesOpen:
    """P0-11 真实价格体系：复牌跳空卖出以真实开盘价 open 成交（后复权 open_adj 已弃用）。

    引擎成交/现金/市值/费用统一真实价（不复权原始价）；后复权价仅保留用于
    止损/信号比较。复牌日 open=10.6（真实），open_adj=21.2（af=2）仅作数据列，
    成交价必须取真实 open=10.6。
    """

    @pytest.mark.integration
    def test_resume_sell_price_is_open(self, flat_engine):
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # H：days[0..4] 正常；days[5..7] 停牌；days[8] 复牌高开 6%（真实价），
            # open_adj=21.2（af=2）仅数据列——成交价必须用真实 open=10.6
            "sh600008": {
                **{d: _price(d, 10.0) for d in range(5)},
                8: _price(
                    8, 10.6,
                    open_adj=21.2, high_adj=21.3, low_adj=21.0,
                    close_adj=21.2, adj_factor=2.0,
                    high=10.65, low=10.55, close=10.6,
                ),
                2: _score(2),
            },
        }
        tl, _ = _run(_mk_panel(specs), {}, flat_engine)
        h = _by_symbol(tl, "sh600008")
        resume_sell = [r for r in h if r["action"] == "sell" and r["time"] == _DAYS[8]]
        assert len(resume_sell) == 1
        assert resume_sell[0]["price"] == pytest.approx(10.6)


class TestAuctionFillRatio:
    """P0-6 ⑥：开盘集合竞价成交率分档（auction_fill_ratio 封单量/可成交量代理）。"""

    @pytest.mark.integration
    def test_open_at_limit_buy_capped_by_auction_ratio(self):
        ecfg = EngineConfig(
            initial_cash=10_000_000.0,
            max_position_pct=0.2,
            max_order_pct=0.5,
            auction_fill_ratio=0.12,
            limit_tradable_ratio=0.30,
            regime_full_multiplier=1.0,
            regime_half_multiplier=1.0,
            regime_min_multiplier=1.0,
        )
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # I：days[0] 高分信号 → days[1] 涨停开盘（open=limit_up=11.0）后炸板
            # （close=10.8 < 11.0 → 非一字）→ 集合竞价可成交量 = 量 × min(0.30, 0.12)
            "sh600009": {
                0: {**_price(0, 10.0), **_score(0)},
                1: {
                    "open": 11.0, "high": 11.0, "low": 10.5, "close": 10.8,
                    "open_adj": 11.0, "high_adj": 11.0, "low_adj": 10.5,
                    "close_adj": 10.8, "adj_factor": 1.0,
                    "volume": 1_000_000, "AMOUNT_MA20": 1e7,
                    "进场评分": 0.0, "退出评分": 0.0, "风险等级": "LOW",
                    "止损价": 0.0, "ATR": 1.0,
                },
            },
        }
        tl, _ = _run(_mk_panel(specs), {}, ecfg)
        buys = [r for r in _by_symbol(tl, "sh600009") if r["action"] == "buy"]
        assert len(buys) == 1
        # 请求 ~181,900 股 → 竞价档上限 = 1,000,000 × 0.12 = 120,000 股
        assert buys[0]["qty"] == 120_000
        assert buys[0]["price"] == pytest.approx(11.0)


class TestListingDaysInjection:
    """P0-6 ④：上市日期显式注入；缺省不再数据推断（引擎不崩溃）。"""

    @pytest.mark.integration
    def test_engine_runs_without_listing_days(self, flat_engine):
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # K 数据自 days[5] 起（中途加入）——旧实现会误判为新股
            "sh600010": {
                **{d: _price(d, 10.0) for d in range(5, 25)},
                5: _score(5),
            },
        }
        tl, ec = _run(_mk_panel(specs), {}, flat_engine)
        assert len(ec) > 0
        # 数据推断已禁止：未注入 IPO 日期 → 无新股豁免，K 仍按常规限价正常交易
        assert any(r["action"] == "buy" for r in _by_symbol(tl, "sh600010"))

    @pytest.mark.integration
    def test_engine_accepts_explicit_listing_days(self, flat_engine):
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            "sh600010": {
                **{d: _price(d, 10.0) for d in range(5, 25)},
                5: _score(5),
            },
        }
        tl, _ = _run(
            _mk_panel(specs),
            {"_listing_days": {"sh600010": _DAYS[5]}},
            flat_engine,
        )
        assert any(r["action"] == "buy" for r in _by_symbol(tl, "sh600010"))