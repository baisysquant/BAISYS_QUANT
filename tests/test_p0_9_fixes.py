"""P0-9 审计修复测试：
  ① 半仓卖出后 pos_value 按 剩余/原股数 比例递减（停牌无价回退估值不高估净值）
  ② signal_model：预测按 day_valid 掩码落盘（NaN 预测不覆写进场评分）
  ③ signal_model：必需特征列强校验（缺列不 KeyError 崩溃）

覆盖：
  1. 引擎级：买入→半仓卖出→停牌，停牌日净值精确等于 现金+剩余股数×停牌前收盘，
     且低于"pos_value 未递减回退估值"的高估上界（回归防御）
  2. ML：Ridge 病态 coef 传播 NaN 预测时，NaN 行保留原生评分、有效行按掩码落盘
  3. ML：缺 _FEATURE_CN 列 / 基础行情列 → 明确日志跳过，不崩溃
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BackTrading.engine import EngineConfig, run_full_backtest  # noqa: E402

_DAYS = [str(d.date()) for d in pd.bdate_range("2026-01-05", periods=25)]


def _mk_panel(specs: dict[str, dict[int, dict]], n_days: int = 25) -> pd.DataFrame:
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


def _price(day: int, p: float, **kw) -> dict:
    ov = {
        "open": p, "high": p * 1.02, "low": p * 0.98, "close": p,
        "open_adj": p, "high_adj": p * 1.02, "low_adj": p * 0.98,
        "close_adj": p, "adj_factor": 1.0, "volume": 1_000_000,
    }
    ov.update(kw)
    return ov


def _score(day: int, v: float = 90.0) -> dict:
    return {"进场评分": v}


@pytest.fixture
def flat_engine() -> EngineConfig:
    return EngineConfig(
        regime_full_multiplier=1.0,
        regime_half_multiplier=1.0,
        regime_min_multiplier=1.0,
    )


# ── ① pos_value 半仓卖出按比例递减 ──────────────────────────────

class TestPosValuePartialSell:
    @pytest.mark.integration
    def test_partial_sell_then_suspension_value_not_inflated(self, flat_engine):
        """买入 → 半仓卖出 → 停牌：停牌日净值 = 现金 + 剩余股数×停牌前收盘。

        若 pos_value 未按剩余股数比例递减，停牌无价回退估值（core.py 495/511）
        会用 原成本市值/剩余股数 高估单位成本（≈2 倍），净值虚增。
        """
        ecfg = EngineConfig(
            initial_cash=1_000_000.0,
            max_position_pct=1.0,
            max_order_pct=0.5,
            regime_full_multiplier=1.0,
            regime_half_multiplier=1.0,
            regime_min_multiplier=1.0,
        )
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            # A：day0 高分信号 → day1 开盘买入；day2 低分触发常规卖出
            # （exit_score_low）→ day3 开盘半仓成交；day4 起停牌（无 bar）
            "sh600009": {
                **{d: _price(d, 10.0) for d in range(4)},
                0: _score(0),
                2: {"进场评分": 3.0},
            },
        }
        tl, ec = run_full_backtest(_mk_panel(specs), {}, ecfg)

        f = [r for r in tl if r["symbol"] == "sh600009"]
        buys = [r for r in f if r["action"] == "buy"]
        sells = [r for r in f if r["action"] == "sell_partial"]
        assert len(buys) == 1
        assert len(sells) == 1
        assert sells[0]["time"] == _DAYS[3]

        # 剩余持仓 = 买入股数 - 半仓卖出股数（剩余仍 ≥ 1 手）
        remaining = buys[0]["qty"] - sells[0]["qty"]
        assert remaining >= 100

        # 停牌日（day4 起无 bar）净值 = 现金 + 剩余股数 × 停牌前收盘价 10.0
        cash_after = 1_000_000.0
        for r in tl:
            if r["action"].startswith("buy"):
                cash_after -= r["value"] + r["cost"]
            else:
                # P0-11：value 统一为毛额，现金入账 = 毛额 - 成本（引擎 cash += proc）
                cash_after += r["value"] - r["cost"]
        pv_day4 = [e["portfolio_value"] for e in ec if e["time"] == _DAYS[4]]
        assert pv_day4, "停牌日应有净值"
        pv = pv_day4[0]
        assert abs(pv - (cash_after + remaining * 10.0)) < 1.0
        # 防御上界：若回退估值路径被触发且 pos_value 未按比例递减 → 高估 2 倍成本
        assert pv <= cash_after + remaining * 15.0

    @pytest.mark.integration
    def test_suspension_after_partial_sell_keeps_valuation_stable(self, flat_engine):
        """半仓卖出后连续停牌多日，净值保持稳定（= 停牌前收盘估值，不漂移）。"""
        ecfg = EngineConfig(
            initial_cash=1_000_000.0,
            max_position_pct=1.0,
            max_order_pct=0.5,
            regime_full_multiplier=1.0,
            regime_half_multiplier=1.0,
            regime_min_multiplier=1.0,
        )
        specs = {
            "sh600001": {d: _price(d, 10.0) for d in range(25)},
            "sh600009": {
                **{d: _price(d, 10.0) for d in range(4)},
                0: _score(0),
                2: {"进场评分": 3.0},
            },
        }
        tl, ec = run_full_backtest(_mk_panel(specs), {}, ecfg)
        pv_susp = [e["portfolio_value"] for e in ec if e["time"] >= _DAYS[4]]
        assert len(pv_susp) >= 2
        assert max(pv_susp) - min(pv_susp) < 1.0  # 停牌期净值零波动


# ── ② signal_model：day_valid 掩码落盘 ─────────────────────────

def _mk_ml_panel(n_days: int = 260, n_syms: int = 6) -> pd.DataFrame:
    """构造 ML 面板：必需列 + _FEATURE_CN 全列，价格随机游走（固定种子）。"""
    dates = [str(d.date()) for d in pd.bdate_range("2025-01-01", periods=n_days)]
    rng = np.random.default_rng(42)
    rows = []
    cn_cols = [
        "MACD趋势分", "金叉信号分", "柱状动能分",
        "DIF斜评分", "背离信号分", "量价配合分", "K线形态分",
    ]
    for k in range(n_syms):
        sym = f"sh6000{k + 1:02d}"
        price = 20.0 + np.cumsum(rng.normal(0, 0.5, n_days))
        price = np.maximum(price, 1.0)
        vol = rng.integers(500_000, 2_000_000, n_days)
        amt = price * vol
        for i, d in enumerate(dates):
            c = float(price[i])
            row = {
                "symbol": sym,
                "trade_date": d,
                "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                "volume": float(vol[i]), "amount": float(amt[i]),
                "ATR": 0.5,
                "进场评分": 50.0, "退出评分": 0.0, "风险等级": "LOW",
                "止损价": 0.0,
            }
            for cn in cn_cols:
                row[cn] = float(rng.normal(0, 1))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


class _SemiNaNModel:
    """模拟 Ridge 病态 coef（部分行预测 NaN）：偶数行 NaN，奇数行有限。"""

    name = "FakeSemiNaN"

    def fit(self, X, y, X_val=None, y_val=None) -> bool:
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.arange(X.shape[0]) * 0.01 + 0.5
        out[np.arange(X.shape[0]) % 2 == 0] = np.nan
        return out


class TestSignalModelNanMask:
    @pytest.mark.integration
    def test_nan_prediction_rows_keep_native_score(self, monkeypatch):
        """部分行预测 NaN（Ridge X@coef 传播）时：
        NaN 行保留原生评分（不被 NaN 覆写），有效行按掩码落盘。"""
        import LogicAnalyzer.ml.signal_model as sm

        monkeypatch.setattr(sm, "_pick_model", lambda: _SemiNaNModel())
        # 绕过显著性门控：稳定触发模型生效（本测试聚焦掩码落盘，不测门控）
        monkeypatch.setattr(sm, "_label_shuffle_test", lambda model, X, y: (0.5, 0.001))

        df = _mk_ml_panel(n_days=260, n_syms=6)
        out = sm.apply_ml_signal(df)

        # 核心：没有任何行被 NaN 覆写
        assert not out["进场评分"].isna().any()

        # 存在被掩码落盘覆写的行（有效预测行）
        predicted = out["进场评分"] != 50.0
        assert predicted.any()

        # 抽查一个预测日（dates[160]）：偶数行（NaN 预测）保留 50，奇数行被覆写
        target_date = sorted(out["trade_date"].unique())[160]
        day = out[out["trade_date"] == target_date].sort_values("symbol").reset_index(drop=True)
        for i in range(len(day)):
            if i % 2 == 0:
                assert day.loc[i, "进场评分"] == 50.0
            else:
                assert 1.0 < day.loc[i, "进场评分"] <= 100.0

    @pytest.mark.integration
    def test_all_nan_prediction_leaves_native_score_untouched(self, monkeypatch):
        """预测全 NaN（极端病态）→ 整日不落盘，评分保持原生。"""
        import LogicAnalyzer.ml.signal_model as sm

        class _AllNaNModel:
            name = "FakeAllNaN"

            def fit(self, X, y, X_val=None, y_val=None) -> bool:
                return True

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.full(X.shape[0], np.nan)

        monkeypatch.setattr(sm, "_pick_model", lambda: _AllNaNModel())
        monkeypatch.setattr(sm, "_label_shuffle_test", lambda model, X, y: (0.5, 0.001))

        df = _mk_ml_panel(n_days=260, n_syms=6)
        out = sm.apply_ml_signal(df)

        assert not out["进场评分"].isna().any()
        assert (out["进场评分"] == 50.0).all()  # 全 NaN 预测 → 全部保留原生评分


# ── ③ signal_model：必需特征列强校验 ───────────────────────────

class TestSignalModelMissingColumns:
    def _panel(self) -> pd.DataFrame:
        return _mk_ml_panel(n_days=30, n_syms=6)

    @pytest.mark.unit
    def test_missing_cn_feature_column_skips_without_crash(self):
        """缺 _FEATURE_CN 列（如"金叉信号分"）→ 明确日志跳过，不 KeyError 崩溃。"""
        import LogicAnalyzer.ml.signal_model as sm

        df = self._panel().drop(columns=["金叉信号分"])
        out = sm.apply_ml_signal(df)
        assert out.equals(df)
        assert not out["进场评分"].isna().any()

    @pytest.mark.unit
    def test_missing_base_price_columns_skips_without_crash(self):
        """缺基础行情列（high/low/volume/amount）→ 跳过，不在特征计算时崩溃。"""
        import LogicAnalyzer.ml.signal_model as sm

        df = self._panel().drop(columns=["high", "amount"])
        out = sm.apply_ml_signal(df)
        assert out.equals(df)

    @pytest.mark.unit
    def test_complete_columns_do_not_crash(self):
        """全列齐全 → 正常走完（短面板早退路径），不崩溃。"""
        import LogicAnalyzer.ml.signal_model as sm

        df = self._panel()
        out = sm.apply_ml_signal(df)
        assert "进场评分" in out.columns
        assert not out["进场评分"].isna().any()