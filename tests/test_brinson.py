from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from LogicAnalyzer.risk.brinson import BrinsonDecomposition

INDUSTRIES = ["银行", "食品", "医药"]


def _result() -> dict:
    w = pd.Series([0.5, 0.3, 0.2], index=INDUSTRIES)
    b = pd.Series([0.4, 0.4, 0.2], index=INDUSTRIES)
    rp = pd.Series([0.03, 0.01, -0.02], index=INDUSTRIES)
    rb = pd.Series([0.02, 0.02, -0.01], index=INDUSTRIES)
    return BrinsonDecomposition().decompose(w, b, rp, rb)


def test_decompose_total_consistency() -> None:
    res = _result()
    assert "error" not in res
    assert res["超额收益"] == pytest.approx(res["组合收益"] - res["基准收益"], rel=1e-9)
    assert res["归因表"]["总贡献"].sum() == pytest.approx(res["超额收益"], rel=1e-9)


def test_decompose_industry_rows() -> None:
    res = _result()
    assert set(res["归因表"]["行业"]) == set(INDUSTRIES)
    assert list(res["归因表"].columns) == ["行业", "配置效应", "选择效应", "交互效应", "总贡献"]


def test_alloc_only_when_industry_returns_equal() -> None:
    ind = ["A", "B"]
    w = pd.Series([0.6, 0.4], index=ind)
    b = pd.Series([0.5, 0.5], index=ind)
    rp = pd.Series([0.05, 0.05], index=ind)
    rb = pd.Series([0.05, 0.05], index=ind)
    res = BrinsonDecomposition().decompose(w, b, rp, rb)
    assert res["归因表"]["选择效应"].abs().max() < 1e-9
    assert res["归因表"]["交互效应"].abs().max() < 1e-9
    assert res["超额收益"] == pytest.approx(res["归因表"]["配置效应"].sum(), rel=1e-9)


def test_selection_only_when_weights_equal() -> None:
    ind = ["A", "B"]
    w = pd.Series([0.5, 0.5], index=ind)
    b = pd.Series([0.5, 0.5], index=ind)
    rp = pd.Series([0.06, 0.03], index=ind)
    rb = pd.Series([0.05, 0.05], index=ind)
    res = BrinsonDecomposition().decompose(w, b, rp, rb)
    assert res["归因表"]["配置效应"].abs().max() < 1e-9
    # 选择 = b·(rp-rb) = 0.5·0.01 + 0.5·(-0.02) = -0.005
    assert res["归因表"]["选择效应"].sum() == pytest.approx(-0.005, rel=1e-9)
    assert res["超额收益"] == pytest.approx(-0.005, rel=1e-9)


def test_aggregate_periods() -> None:
    ind = ["A", "B"]
    br = BrinsonDecomposition()
    r1 = br.decompose(
        pd.Series([0.6, 0.4], index=ind),
        pd.Series([0.5, 0.5], index=ind),
        pd.Series([0.05, 0.02], index=ind),
        pd.Series([0.03, 0.03], index=ind),
    )
    r2 = br.decompose(
        pd.Series([0.5, 0.5], index=ind),
        pd.Series([0.5, 0.5], index=ind),
        pd.Series([0.01, 0.04], index=ind),
        pd.Series([0.02, 0.02], index=ind),
    )
    agg = br.aggregate_periods([r1, r2])
    assert "error" not in agg
    # 效应跨期算术加总；组合/基准收益复利合成
    assert agg["归因表"]["配置效应"].sum() == pytest.approx(
        r1["归因表"]["配置效应"].sum() + r2["归因表"]["配置效应"].sum(), rel=1e-9
    )
    assert agg["组合收益"] == pytest.approx(1.038 * 1.025 - 1, rel=1e-9)
    assert agg["基准收益"] == pytest.approx(1.03 * 1.02 - 1, rel=1e-9)


def test_aggregate_periods_empty() -> None:
    res = BrinsonDecomposition().aggregate_periods([])
    assert "error" in res


def test_from_holdings() -> None:
    holdings = pd.DataFrame(
        {
            "股票代码": ["600001", "600002"],
            "目标权重": [0.7, 0.3],
            "所属行业": ["银行", "食品"],
        }
    )
    dates = pd.bdate_range("2024-01-01", periods=20)
    up = 10.0 * (1 + np.linspace(0, 0.10, len(dates)))
    flat = np.full(len(dates), 20.0)
    kline = pd.DataFrame(
        {
            "symbol": ["sh600001"] * len(dates) + ["sh600002"] * len(dates),
            "trade_date": list(dates) + list(dates),
            "close": np.concatenate([up, flat]),
        }
    )
    res = BrinsonDecomposition().from_holdings(holdings, kline)
    assert "error" not in res
    # 组合收益 = 0.7·10% + 0.3·0% = 7%（股票代码前缀已自动归一）
    assert res["组合收益"] == pytest.approx(0.07, rel=1e-6)
    assert res["基准收益"] == pytest.approx(0.05, rel=1e-6)


def test_from_holdings_explicit_benchmark() -> None:
    holdings = pd.DataFrame(
        {
            "股票代码": ["600001"],
            "目标权重": [1.0],
            "所属行业": ["银行"],
        }
    )
    dates = pd.bdate_range("2024-01-01", periods=10)
    kline = pd.DataFrame(
        {
            "symbol": ["600001"] * len(dates),
            "trade_date": list(dates),
            "close": 10.0 * (1 + np.linspace(0, 0.05, len(dates))),
        }
    )
    res = BrinsonDecomposition().from_holdings(
        holdings,
        kline,
        benchmark_weights=pd.Series([0.5], index=["银行"]),
        benchmark_returns=pd.Series([0.02], index=["银行"]),
    )
    assert "error" not in res
    assert res["组合收益"] == pytest.approx(0.05, rel=1e-6)
    # 基准权重 [0.5] 自动归一化 → r_b = 1.0·0.02
    assert res["基准收益"] == pytest.approx(0.02, rel=1e-6)


def test_from_holdings_with_industry_map() -> None:
    holdings = pd.DataFrame(
        {
            "股票代码": ["600001"],
            "目标权重": [1.0],
            "所属行业": ["银行"],
        }
    )
    dates = pd.bdate_range("2024-01-01", periods=10)
    kline = pd.DataFrame(
        {
            "symbol": ["sh600001"] * len(dates) + ["sh600003"] * len(dates),
            "trade_date": list(dates) + list(dates),
            "close": np.concatenate(
                [10.0 * (1 + np.linspace(0, 0.10, len(dates))), np.full(len(dates), 30.0)]
            ),
        }
    )
    # 基准宇宙 = 持仓 ∪ 行业映射（600003 → 食品，未持仓）
    res = BrinsonDecomposition().from_holdings(
        holdings,
        kline,
        industry_map=pd.Series({"600003": "食品"}),
    )
    assert "error" not in res
    # 基准：银行 50% 收益 10% + 食品 50% 收益 0% → 5%
    assert res["基准收益"] == pytest.approx(0.05, rel=1e-6)


def test_build_report_shape() -> None:
    res = _result()
    df = BrinsonDecomposition().build_report(res)
    assert {"行业", "配置效应", "选择效应", "交互效应", "总贡献"} <= set(df.columns)
    assert (df["行业"] == "合计").any()
    assert (df["行业"] == "超额收益").any()
    # 合计行 = 各效应加总
    total_row = df[df["行业"] == "合计"].iloc[0]
    assert float(total_row["总贡献"]) == pytest.approx(res["超额收益"], rel=1e-9)


def test_to_excel(tmp_path) -> None:
    res = _result()
    path = BrinsonDecomposition().to_excel(res, str(tmp_path / "brinson.xlsx"))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_incomplete_inputs_error() -> None:
    br = BrinsonDecomposition()
    assert "error" in br.decompose(None, pd.Series([1.0]), pd.Series([0.1]), pd.Series([0.1]))
    assert "error" in br.decompose(pd.Series([1.0]), pd.Series([1.0]), None, pd.Series([0.1]))
