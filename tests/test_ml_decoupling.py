"""P1 ML/参数解耦回归测试：ML 预测列按数据版本冻结，参数变体只注入不重训。"""

from __future__ import annotations

import pandas as pd
import pytest

from BackTrading import prepare as prepare_mod


def _make_kline(n_sym: int = 3, n_days: int = 110) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B").strftime("%Y-%m-%d")
    rows = []
    for si, sym in enumerate(["000001.SZ", "000002.SZ", "600000.SH"][:n_sym]):
        for di, d in enumerate(dates):
            rows.append({
                "symbol": sym,
                "trade_date": d,
                "open": 10.0 + si + di * 0.01,
                "high": 11.0 + si + di * 0.01,
                "low": 9.0 + si + di * 0.01,
                "close": 10.5 + si + di * 0.01,
                "volume": 1_000_000 + si * 100_000,
                "amount": 1_000_000 * 10.5,
            })
    return pd.DataFrame(rows)


def _fake_worker(symbol: str, stock_dir: str, params: dict, compute_exit_strategy: bool = False, susp_stats: dict | None = None) -> list[dict]:
    stock_df = pd.read_parquet(f"{stock_dir}/{symbol}.parquet", engine="fastparquet")
    rows = []
    for i, (_, bar) in enumerate(stock_df.iterrows()):
        rows.append({
            "symbol": symbol,
            "trade_date": str(bar["trade_date"])[:10],
            "entry_score": float(50 + i % 7),
            "exit_score": 40.0,
            "risk_level": "LOW",
            "score": 55.0,
            "atr": 1.0,
            "macd_trend": 10.0,
            "golden_cross": 5.0,
            "hist_momentum": 8.0,
            "dif_slope": 6.0,
            "divergence": 0.0,
            "vol_price": 7.0,
            "kline": 9.0,
        })
    return rows


class TestMLDecoupling:
    @pytest.mark.unit
    def test_ml_trains_once_and_injects_frozen_predictions(self, monkeypatch, tmp_path):
        # 隔离：磁盘缓存进 tmp，ML 缓存清空，重训只发生在首帧
        monkeypatch.setattr(prepare_mod, "_cache_dir_for", lambda *a, **k: tmp_path / "sigcache")
        monkeypatch.setattr(prepare_mod, "_ML_PRED_CACHE", {})
        monkeypatch.setattr(prepare_mod, "precompute_all_indicators", lambda *a, **k: None)
        monkeypatch.setattr(prepare_mod, "_stock_worker_vectorized", _fake_worker)

        calls = {"n": 0}

        def fake_apply_ml(merged: pd.DataFrame) -> pd.DataFrame:
            calls["n"] += 1
            merged["进场评分"] = (merged.groupby("trade_date")["close"].rank(pct=True) * 99 + 1).values
            return merged

        monkeypatch.setattr(prepare_mod, "apply_ml_signal", fake_apply_ml)

        kline = _make_kline()
        p1 = {"golden_cross_bonus": 10, "divergence_penalty": 20}
        p2 = {"golden_cross_bonus": 20, "divergence_penalty": 30}

        df1 = prepare_mod.prepare_backtest_data(
            kline, params=p1, compute_exit_strategy=False, vectorized=True,
        )
        assert calls["n"] == 1, "首帧应重训一次"

        df2 = prepare_mod.prepare_backtest_data(
            kline, params=p2, compute_exit_strategy=False, vectorized=True,
        )
        assert calls["n"] == 1, "参数变体不得重训（应注入冻结预测）"

        # 等价性：两次调用的进场评分必须逐行一致（同数据版本）
        m = df1[["symbol", "trade_date", "进场评分"]].merge(
            df2[["symbol", "trade_date", "进场评分"]],
            on=["symbol", "trade_date"], suffixes=("_a", "_b"),
        )
        assert len(m) == len(df1) == len(df2)
        assert (m["进场评分_a"] == m["进场评分_b"]).all(), "冻结预测注入不一致"
        # 注入确实生效：ML 值为横截面排名缩放（3 只 → 3 档），非原生评分（50+i%7 有 7 档）
        assert m["进场评分_a"].nunique() == 3, "进场评分应体现 ML 横截面排名而非原生值"
