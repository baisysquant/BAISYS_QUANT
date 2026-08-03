from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from LogicAnalyzer.scoring.orthogonalizer import (
    FactorOrthogonalizer,
    OrthogonalizationResult,
)


def _make_synthetic(
    n_stocks: int = 60, n_days: int = 150, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造合成面板：f1=5日动量(真实信号)，f2=f1+噪音(高相关冗余)，
    f3=纯噪音，f4=30日动量(慢信号)。"""
    rng = np.random.default_rng(seed)
    n_extra = 25
    all_dates = pd.bdate_range("2023-11-01", periods=n_days + n_extra).strftime(
        "%Y-%m-%d"
    )
    panel_dates = all_dates[:n_days]
    symbols = [f"600{100 + i:03d}" for i in range(n_stocks)]
    alpha = rng.normal(0, 1, n_stocks)

    closes = np.empty((len(all_dates), n_stocks))
    prices = np.full(n_stocks, 10.0)
    for j in range(len(all_dates)):
        prices = prices * (1 + 0.008 * alpha + rng.normal(0, 0.02, n_stocks))
        closes[j] = prices

    kline = pd.DataFrame(
        {
            "symbol": np.repeat(symbols, len(all_dates)),
            "trade_date": np.tile(all_dates, n_stocks),
            "close": closes.T.ravel(),
        }
    )

    panel_rows = []
    for j, d in enumerate(panel_dates):
        for i, sym in enumerate(symbols):
            mom5 = closes[j, i] / closes[j - 5, i] - 1 if j >= 5 else 0.0
            mom30 = closes[j, i] / closes[max(j - 30, 0), i] - 1
            panel_rows.append(
                {
                    "symbol": sym,
                    "trade_date": d,
                    "f1": mom5 + rng.normal(0, 2e-4),
                    "f2": mom5 + rng.normal(0, 0.01),
                    "f3": rng.normal(0, 1),
                    "f4": mom30 + rng.normal(0, 1e-3),
                }
            )
    return pd.DataFrame(panel_rows), kline


# ── 对称正交化 ─────────────────────────────────────────────


def test_orthogonalize_correlation_identity() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(0, 1, (120, 4)), columns=["f1", "f2", "f3", "f4"]
    )
    X["f3"] = X["f1"] * 0.9 + rng.normal(0, 0.01, 120)

    res = FactorOrthogonalizer().orthogonalize(X)
    assert isinstance(res, OrthogonalizationResult)
    assert res.X_orth.shape == (120, 4)

    # 新因子间相关系数矩阵为单位阵（近共线时受 jitter 影响，放宽到 1e-3）
    off = res.corr_after.to_numpy().copy()
    np.fill_diagonal(off, 0)
    assert np.abs(off).max() < 1e-3

    # ZᵀZ / N = I 且可逆还原 X = Z·Lᵀ
    gram = (res.X_orth.T @ res.X_orth / len(res.X_orth)).to_numpy()
    assert np.abs(gram - np.eye(4)).max() < 1e-3
    Xs = X.astype(float)
    Xs = (Xs - Xs.mean(axis=0)) / Xs.std(axis=0)
    recovered = res.X_orth.to_numpy() @ res.cholesky_L.T
    assert np.abs(recovered - Xs.to_numpy()).max() < 1e-6


def test_orthogonalize_insufficient_samples() -> None:
    X = pd.DataFrame(np.random.default_rng(1).normal(0, 1, (3, 5)))
    with pytest.raises(ValueError, match="正交化"):
        FactorOrthogonalizer().orthogonalize(X)


def test_orthogonalize_constant_factor_dropped() -> None:
    X = pd.DataFrame(
        {
            "a": np.random.default_rng(2).normal(0, 1, 50),
            "b": np.ones(50),  # 常量因子
            "c": np.random.default_rng(3).normal(0, 1, 50),
        }
    )
    res = FactorOrthogonalizer().orthogonalize(X)
    assert res.X_orth.shape == (50, 2)
    assert "b" not in res.X_orth.columns


# ── 冗余剔除 ──────────────────────────────────────────────


def test_prune_by_correlation_drops_lower_ic() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.95, 0.2], [0.95, 1.0, 0.1], [0.2, 0.1, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    ic = {"a": 0.08, "b": 0.05, "c": 0.03}
    dropped = FactorOrthogonalizer.prune_by_correlation(corr, ic, threshold=0.8)
    assert dropped == ["b"]


def test_prune_by_correlation_below_threshold_untouched() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"]
    )
    assert FactorOrthogonalizer.prune_by_correlation(corr, {"a": 0.1, "b": 0.2}) == []


# ── IC 衰减分析 ───────────────────────────────────────────


def test_estimate_half_life_known_decay() -> None:
    lags = [1, 5, 10, 20]
    ic = pd.Series({h: 0.1 * 0.5 ** (h / 5) for h in lags})
    hl = FactorOrthogonalizer.estimate_half_life(ic, lags)
    assert abs(hl - 5.0) < 0.1


def test_estimate_half_life_no_decay_is_inf() -> None:
    ic = pd.Series({1: 0.1, 5: 0.12, 10: 0.11, 20: 0.13})
    assert FactorOrthogonalizer.estimate_half_life(ic, [1, 5, 10, 20]) == math.inf


def test_estimate_half_life_no_signal_is_zero() -> None:
    ic = pd.Series({1: 1e-9, 5: 1e-9, 10: 1e-9, 20: 1e-9})
    assert FactorOrthogonalizer.estimate_half_life(ic, [1, 5, 10, 20]) == 0.0


def test_classify_horizon_boundaries() -> None:
    orth = FactorOrthogonalizer
    assert orth.classify_horizon(2.0) == "noise"
    assert orth.classify_horizon(4.0) == "short"
    assert orth.classify_horizon(10.0) == "mid"
    assert orth.classify_horizon(20.0) == "long"
    assert orth.classify_horizon(math.inf) == "long"


# ── 自动权重分配 ──────────────────────────────────────────


def test_ir_weights_normalized_and_negative_zeroed() -> None:
    w = FactorOrthogonalizer.ir_weights({"a": 0.05, "b": 0.03, "c": -0.02})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["c"] == 0.0
    assert w["a"] > w["b"]


def test_ir_weights_allow_negative_literal_formula() -> None:
    w = FactorOrthogonalizer.ir_weights(
        {"a": 0.05, "b": 0.03, "c": -0.02}, allow_negative=True
    )
    assert abs(w["a"] - 0.5) < 1e-9
    assert abs(w["c"] + 0.2) < 1e-9


# ── 端到端 Pipeline ───────────────────────────────────────


def test_pipeline_end_to_end() -> None:
    panel, kline = _make_synthetic()
    result = FactorOrthogonalizer().run(panel, kline, ["f1", "f2", "f3", "f4"])

    assert "error" not in result

    # 冗余剔除：f2 与 f1 高度相关且 IC 更低
    assert "f2" in result["pruned"]["high_corr"]
    assert "f1" not in result["pruned"]["high_corr"]
    assert "f1" in result["kept_factors"]

    # 正交后相关性 ≈ 单位阵
    corr_after = result["orthogonalized"]["corr_after"]
    off = corr_after.to_numpy().copy()
    np.fill_diagonal(off, 0)
    assert np.abs(off).max() < 0.01

    # 衰减曲线包含全部滞后
    curves = result["ic_analysis"]["decay_curves"]
    assert list(curves.columns) == ["lag1", "lag5", "lag10", "lag20"]

    # 噪音因子 f3 的平均 |IC| 显著低于真实信号 f1
    assert curves.loc["f3"].abs().mean() < curves.loc["f1"].abs().mean()

    # IR-Weighted 权重归一化
    w_all = result["weights"]["all"]
    assert w_all
    assert abs(sum(w_all.values()) - 1.0) < 1e-6
    assert {"short", "long", "all"} <= set(result["weights"])

    # 报告完整
    report = result["report"]
    assert "分类" in report.columns
    assert len(report) == len(result["classification"])


def test_pipeline_insufficient_data() -> None:
    result = FactorOrthogonalizer().run(
        pd.DataFrame(), pd.DataFrame(), ["f1", "f2"]
    )
    assert "error" in result


def test_pipeline_too_few_factor_columns() -> None:
    panel, kline = _make_synthetic(n_days=30)
    result = FactorOrthogonalizer().run(panel, kline, ["f1"])
    assert "error" in result
