"""
1.6 动态重训时间戳锚定 测试

覆盖：
  - check_train_cutoff：训练集（X/Y 样本行）最大时间戳必须 ≤ 锚点 T
    （> T 的样本绝对禁止进入 Fit）
  - check_first_signal_after_train：重训当天信号废弃——首个信号必须位于
    训练锚点之后的下一个交易窗口
  - run_anchor_time_check 一站式入口
  - 集成回归：walk-forward 循环中「重训当日不产信号，次日（T+1）生效」
    （对照组证明 ML 覆写仅在模型有效时发生）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import LogicAnalyzer.ml.signal_model as sm
from LogicAnalyzer.ml.anchor_integrity import (
    check_first_signal_after_train,
    check_train_cutoff,
    run_anchor_time_check,
)
from LogicAnalyzer.ml.signal_model import apply_ml_signal


def _bdays(n: int, start: str = "2024-01-01") -> list[str]:
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()


# ── 动态训练集截止线 ──────────────────────────────────────


def test_train_cutoff_all_before_anchor_passes() -> None:
    dates = _bdays(100)
    report = check_train_cutoff(dates[:40], dates[50])
    assert report.passed is True
    assert report.n_violations == 0
    assert "≤ T" in "；".join(report.details)


def test_train_cutoff_max_equal_anchor_allowed() -> None:
    """规范：max(训练样本时间戳) ≤ T 即合规（样本恰在锚点当日允许）。"""
    dates = _bdays(100)
    report = check_train_cutoff(dates[:41], dates[40])
    assert report.passed is True


def test_train_cutoff_future_sample_fails() -> None:
    """任何时间戳 > T 的样本绝对禁止进入 Fit。"""
    dates = _bdays(100)
    train = dates[:41] + [dates[55]]
    report = check_train_cutoff(train, dates[40])
    assert report.passed is False
    assert report.n_violations == 1
    assert "越界" in "；".join(report.details)


def test_train_cutoff_counts_all_future_days() -> None:
    dates = _bdays(100)
    report = check_train_cutoff(dates[:60], dates[40])
    assert report.passed is False
    assert report.n_violations == 19


def test_train_cutoff_empty_fails() -> None:
    report = check_train_cutoff([], "2024-03-01")
    assert report.passed is False


# ── 重训当天信号废弃 ──────────────────────────────────────


def test_first_signal_same_day_fails() -> None:
    """T 日盘中完成训练的模型不得产出 T 日信号（当日废弃）。"""
    dates = _bdays(60)
    report = check_first_signal_after_train(dates[40], dates[40])
    assert report.passed is False
    assert report.n_violations == 1
    assert "废弃" in "；".join(report.details)


def test_first_signal_next_window_passes() -> None:
    dates = _bdays(60)
    report = check_first_signal_after_train(dates[40], dates[41])
    assert report.passed is True


def test_first_signal_before_anchor_fails() -> None:
    dates = _bdays(60)
    report = check_first_signal_after_train(dates[40], dates[30])
    assert report.passed is False


# ── 一站式入口 ────────────────────────────────────────────


def test_anchor_time_check_one_stop() -> None:
    dates = _bdays(100)
    result = run_anchor_time_check(dates[:40], dates[50], dates[51])
    assert result["passed"] is True
    assert len(result["reports"]) == 2
    assert isinstance(result["summary"], pd.DataFrame)
    result2 = run_anchor_time_check(dates[:41], dates[40])  # 仅截止线
    assert result2["passed"] is True
    assert len(result2["reports"]) == 1


# ── 集成回归：walk-forward 循环时间戳锚定 ──────────────────


class _StubModel:
    """恒通过 fit、predict 输出有方差的伪模型。"""

    name = "StubModel"
    _best_iteration = 0

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> bool:
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0]


def _make_merged(n_days: int = 140, n_stocks: int = 12, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _bdays(n_days)
    symbols = [f"600{100 + i:03d}" for i in range(n_stocks)]
    rows = []
    for sym in symbols:
        close = 10.0 + np.cumsum(rng.normal(0, 0.1, n_days))
        prev = np.concatenate([[close[0]], close[:-1]])
        opn = prev * (1 + rng.normal(0, 0.002, n_days))
        rows.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "trade_date": dates,
                    "close": close,
                    "open": opn,
                    "high": np.maximum(close, opn) * 1.005,
                    "low": np.minimum(close, opn) * 0.995,
                    "volume": rng.integers(1_000_000, 5_000_000, n_days),
                    "amount": rng.integers(1e8, 1e9, n_days),
                    "进场评分": rng.uniform(10, 90, n_days),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["ATR"] = df["high"] - df["low"]
    for c in ("MACD趋势分", "金叉信号分", "DIF斜评分", "背离信号分", "量价配合分", "K线形态分"):
        df[c] = rng.uniform(0, 100, len(df))
    return df


def test_no_ml_signal_on_anchor_day() -> None:
    """重训当天信号废弃回归：模型在锚点 T 完成训练，首个 ML 信号必须为
    T+1（下一个交易窗口）；重训当天绝不允许新模型覆写当日进场评分。

    对照组（shuffle 检验失败 → 模型无效）证明 ML 覆写仅在模型有效时发生，
    实验组中所有被覆写的日期 = 首个锚点之后的全部交易日（含后续锚点日，
    由上一代旧模型产出——合法），唯独首个锚点 T 不被任何模型覆写。
    """
    panel = _make_merged()
    orig_split = sm._split_train_val
    orig_pick = sm._pick_model
    orig_shuffle = sm._label_shuffle_test
    try:
        sm._split_train_val = lambda w: (list(w[:20]), list(w[20:40]))
        sm._pick_model = lambda: _StubModel()
        sm._label_shuffle_test = lambda *a, **k: (0.0, 0.99)  # 模型无效（对照）
        ctrl = apply_ml_signal(panel)
        sm._label_shuffle_test = lambda *a, **k: (0.9, 0.001)  # 模型有效（实验）
        ml = apply_ml_signal(panel)
    finally:
        sm._split_train_val = orig_split
        sm._pick_model = orig_pick
        sm._label_shuffle_test = orig_shuffle

    dates = sorted(panel["trade_date"].unique())
    idx_of = {d: k for k, d in enumerate(dates)}

    changed = (ml["进场评分"].to_numpy(float) - ctrl["进场评分"].to_numpy(float)).__abs__() > 1e-6
    tmp = pd.DataFrame({"trade_date": panel["trade_date"], "changed": changed})
    ml_days = set(tmp.loc[tmp["changed"], "trade_date"])
    ml_idx = sorted(idx_of[d] for d in ml_days)

    # 首个锚点（_RETRAIN_FREQ）当天不被覆写；其后续全部交易日均被模型信号覆盖
    assert ml_idx == list(range(sm._RETRAIN_FREQ + 1, len(dates)))
