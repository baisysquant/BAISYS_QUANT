from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from LogicAnalyzer.risk.factor_risk import FactorRiskModel


def _orth_panel(n: int = 60, k: int = 3, seed: int = 1) -> pd.DataFrame:
    """构造列间精确正交的因子暴露矩阵（XᵀX = 9·I）。"""
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, k))
    U, _, Vt = np.linalg.svd(Z, full_matrices=False)
    X = (U @ Vt) * 3.0
    return pd.DataFrame(
        X,
        index=[f"600{100 + i:03d}" for i in range(n)],
        columns=[f"f{i + 1}" for i in range(k)],
    )


def test_decompose_reconstruction() -> None:
    X = _orth_panel()
    n = len(X)
    w = pd.Series(np.full(n, 1.0 / n), index=X.index)
    res = FactorRiskModel().decompose(X, weights=w)
    assert "error" not in res

    # 正交因子 + 单位协方差 + 常数特质波动 0.02
    h = X.T @ w
    expected_var = float(h @ h) + n * ((1.0 / n) ** 2 * 0.02**2)
    assert res["组合日波动"] ** 2 == pytest.approx(expected_var, rel=1e-6)
    assert res["因子风险占比"] + res["特质风险占比"] == pytest.approx(1.0, rel=1e-9)


def test_weights_are_normalized() -> None:
    X = _orth_panel()
    w = pd.Series(np.full(len(X), 2.0), index=X.index)  # 未归一化
    res = FactorRiskModel().decompose(X, weights=w)
    assert "error" not in res
    assert res["因子风险占比"] + res["特质风险占比"] == pytest.approx(1.0, rel=1e-9)


def test_equal_weight_default() -> None:
    X = _orth_panel()
    res1 = FactorRiskModel().decompose(X)
    res2 = FactorRiskModel().decompose(
        X, weights=pd.Series(1.0 / len(X), index=X.index)
    )
    assert res1["组合日波动"] == pytest.approx(res2["组合日波动"], rel=1e-9)


def test_weights_restricted_to_subset() -> None:
    X = _orth_panel()
    w = pd.Series(0.25, index=X.index[:4])  # 权重只覆盖 4 只股票
    res = FactorRiskModel().decompose(X, weights=w)
    assert "error" not in res
    # 仅这 4 只股票获得权重（TopN 前 4 行权重 = 0.25）
    assert (res["个股特质风险TopN"]["权重"].head(4) == 0.25).all()
    assert (res["个股特质风险TopN"]["权重"].iloc[4:] == 0).all()


def test_custom_factor_cov_diagonal() -> None:
    X = _orth_panel()
    cov = pd.DataFrame(
        np.diag([1.0, 2.0, 3.0]), index=X.columns, columns=X.columns
    )
    w = pd.Series(1.0 / len(X), index=X.index)
    res = FactorRiskModel().decompose(X, weights=w, factor_cov=cov)
    assert "error" not in res
    h = X.T @ w
    factor_var = float(h @ cov @ h)
    assert res["因子风险占比"] * res["组合日波动"] ** 2 == pytest.approx(factor_var, rel=1e-6)


def test_factor_contribution_table_sums_to_factor_share() -> None:
    X = _orth_panel()
    w = pd.Series(1.0 / len(X), index=X.index)
    res = FactorRiskModel().decompose(X, weights=w)
    # 表格数值四舍五入到 6 位小数，放宽到 1e-4 相对误差
    assert res["因子风险贡献"]["风险占比"].sum() == pytest.approx(res["因子风险占比"], rel=1e-4)


def test_idio_vol_provided_flag() -> None:
    X = _orth_panel()
    res = FactorRiskModel().decompose(X, idio_vol=pd.Series(0.01, index=X.index))
    assert res["特质波动为估计"] is False


def test_idio_vol_estimated_flag() -> None:
    X = _orth_panel()
    res = FactorRiskModel().decompose(X)
    assert res["特质波动为估计"] is True


def test_from_orthogonalizer() -> None:
    X = _orth_panel()
    orth_result = {"orthogonalized": {"X_orth_latest": X}}
    res = FactorRiskModel().from_orthogonalizer(orth_result)
    assert "error" not in res
    assert res["组合日波动"] > 0


def test_from_orthogonalizer_bad_input() -> None:
    assert "error" in FactorRiskModel().from_orthogonalizer({})
    assert "error" in FactorRiskModel().from_orthogonalizer({"orthogonalized": {}})
    assert "error" in FactorRiskModel().from_orthogonalizer(None)


def test_empty_exposure_error() -> None:
    assert "error" in FactorRiskModel().decompose(pd.DataFrame())


def test_wrong_cov_dimension_error() -> None:
    X = _orth_panel()
    res = FactorRiskModel().decompose(X, factor_cov=np.eye(2))
    assert "error" in res
