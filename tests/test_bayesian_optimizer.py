from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bayesian_test_df() -> pd.DataFrame:
    """Synthetic data with signal columns for BO tests."""
    np.random.seed(42)
    n_stocks, n_days = 10, 150
    symbols = [f"S{i:04d}" for i in range(n_stocks)]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({
                "symbol": sym, "trade_date": d,
                "open": 10, "high": 11, "low": 9,
                "close": np.random.rand() * 10 + 10,
                "volume": 1_000_000,
                "进场评分": float(np.random.uniform(30, 95)),
                "退出评分": float(np.random.uniform(10, 80)),
                "风险等级": str(np.random.choice(["LOW", "MEDIUM", "HIGH"], p=[0.5, 0.3, 0.2])),
                "止损价": float(np.random.rand() * 5),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def basic_spaces() -> dict:
    from BackTrading.bayesian.space import ParamSpace
    return {
        "atr_stop_mult": ParamSpace("atr_stop_mult", 1.0, 3.0, 1.0),
        "kelly_fraction": ParamSpace("kelly_fraction", 0.1, 0.5, 0.2),
    }


@pytest.fixture
def spaces_with_signal() -> dict:
    from BackTrading.bayesian.space import ParamSpace
    return {
        "boll_narrow_ratio": ParamSpace("boll_narrow_ratio", 0.6, 1.2, 0.2, is_signal=True),
        "atr_stop_mult": ParamSpace("atr_stop_mult", 1.0, 3.0, 1.0),
    }


@pytest.fixture
def basic_engine_cfg() -> object:
    from BackTrading._engine_legacy import EngineConfig
    from BackTrading.domain.models import CostModel
    return EngineConfig(
        initial_cash=1_000_000,
        commission_rate=0.0003, stamp_tax_rate=0.0005, slippage=0.001,
        max_position_pct=0.1, portfolio_method="score_weighted", point_in_time=True,
        cost_model=CostModel(commission_rate=0.0003, stamp_tax_rate=0.0005, market_slippage=0.001),
    )


def test_space_building() -> None:
    from BackTrading.bayesian.space import build_spaces, split_by_cost, describe
    from UtilsManager.ConfigParser import BacktestConfig
    bc = BacktestConfig()
    spaces = build_spaces(bc)
    # 死参数 (kelly_fraction/position_a/liq_veto_ratio/risk_none_multiplier) 已移出寻优空间
    assert len(spaces) == 8
    sig, port = split_by_cost(spaces)
    assert len(sig) == 5, f"Expected 5 signal params, got {len(sig)}"
    assert len(port) == 3, f"Expected 3 portfolio params, got {len(port)}"
    desc = describe(spaces)
    assert "信号参数" in desc
    assert "组合参数" in desc


def test_normalization_roundtrip() -> None:
    from BackTrading.bayesian.space import ParamSpace
    from BackTrading.bayesian.optimizer import _to_normalized, _from_normalized
    spaces = {
        "atr_stop_mult": ParamSpace("atr_stop_mult", 1.0, 3.0, 0.5),
        "kelly_fraction": ParamSpace("kelly_fraction", 0.1, 0.5, 0.1),
        "position_a": ParamSpace("position_a", 0.2, 0.5, 0.05),
    }
    cases = [
        {"atr_stop_mult": 2.0, "kelly_fraction": 0.2, "position_a": 0.35},
        {"atr_stop_mult": 1.5, "kelly_fraction": 0.1, "position_a": 0.2},
        {"atr_stop_mult": 3.0, "kelly_fraction": 0.5, "position_a": 0.5},
        {"atr_stop_mult": 2.5, "kelly_fraction": 0.3, "position_a": 0.4},
    ]
    for params in cases:
        x = _to_normalized(params, spaces)
        params2 = _from_normalized(x, spaces)
        for k in params:
            assert abs(params[k] - params2[k]) < 1e-6, f"{k}: {params[k]} vs {params2[k]}"


def test_gp_build() -> None:
    from BackTrading.bayesian.kernel import build_gp, save_gp_state, restore_gp_state
    np.random.seed(42)
    X = np.random.uniform(0, 1, (10, 3))
    Y = np.sin(3 * X[:, 0]) + 0.1 * np.random.randn(10)
    gp = build_gp(X, Y, n_restarts=3)
    mu, sigma = gp.predict(X, return_std=True)
    assert mu.shape == (10,)
    assert sigma.shape == (10,)
    state = save_gp_state(gp, X, Y)
    assert "kernel_params" in state
    restored = restore_gp_state(state, 3)
    assert restored is not None
    wrong = restore_gp_state(state, 5)
    assert wrong is None


def test_gp_state_sub_states_by_dim() -> None:
    """嵌套子空间状态（信号/组合/全空间）按维度取用，缺维才告警返回 None。"""
    from BackTrading.bayesian.kernel import build_gp, restore_gp_state, save_gp_state

    np.random.seed(42)
    X5 = np.random.uniform(0, 1, (12, 5))
    Y = np.sin(3 * X5[:, 0]) + 0.1 * np.random.randn(12)
    X3 = X5[:, :3]
    X8 = np.hstack([X5, np.random.uniform(0, 1, (12, 3))])

    state = {
        "sub_states": {
            5: save_gp_state(build_gp(X5, Y, n_restarts=3), X5, Y),
            3: save_gp_state(build_gp(X3, Y, n_restarts=3), X3, Y),
            8: save_gp_state(build_gp(X8, Y, n_restarts=3), X8, Y),
        },
        "n_dims": 5,
    }
    for d in (3, 5, 8):
        restored = restore_gp_state(state, d)
        assert restored is not None
        assert restored["n_dims"] == d
    assert restore_gp_state(state, 6) is None
    assert restore_gp_state(None, 5) is None


def test_acquisition() -> None:
    from BackTrading.bayesian.acquisition import (
        expected_improvement, mixed_acquisition, optimize_acquisition,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

    np.random.seed(42)
    X = np.random.uniform(0, 1, (10, 2))
    Y = np.sin(3 * X[:, 0])
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(2), nu=1.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp.fit(X, Y)

    ei = expected_improvement(X, gp, best_f=Y.max(), xi=0.01)
    assert ei.shape == (10,)
    assert np.all(ei >= 0)

    mixed = mixed_acquisition(X, gp, best_f=Y.max(), xi=0.01, dsr_lambda=0.05)
    assert mixed.shape == (10,)

    bounds = np.array([[0.0, 1.0]] * 2)
    best_x, best_val = optimize_acquisition(gp, bounds, best_f=Y.max(), n_restarts=5)
    assert best_x.shape == (2,)
    assert np.all(best_x >= 0) and np.all(best_x <= 1)


def test_optimize_window(bayesian_test_df, basic_spaces, basic_engine_cfg) -> None:
    from BackTrading.bayesian.optimizer import optimize_window
    best_params, gp_state, _topk, is_sharpe, is_equity = optimize_window(
        kline_df=bayesian_test_df, engine_cfg=basic_engine_cfg,
        spaces=basic_spaces,
        n_init_signal=4, n_iter_signal=3,
        n_init_portfolio=2, n_iter_portfolio=2,
        n_refine_top=1, seed=42,
    )
    assert isinstance(best_params, dict)
    assert len(best_params) == 2
    # portfolio-only space: gp_state is None (no signal-level model to transfer)


def test_optimize_window_with_signal_params(bayesian_test_df, basic_engine_cfg, spaces_with_signal) -> None:
    from BackTrading.bayesian.optimizer import optimize_window
    best_params, gp_state, _topk, is_sharpe, is_equity = optimize_window(
        kline_df=bayesian_test_df, engine_cfg=basic_engine_cfg,
        spaces=spaces_with_signal,
        n_init_signal=4, n_iter_signal=3,
        n_init_portfolio=2, n_iter_portfolio=2,
        n_refine_top=1, seed=42,
    )
    assert isinstance(best_params, dict)
    assert len(best_params) == 2
    assert gp_state is not None, "With signal params, gp_state should be saved"


@pytest.mark.slow
def test_meta_optimizer(bayesian_test_df, basic_spaces) -> None:
    from BackTrading.bayesian.meta_optimizer import bayesian_walk_forward_multi
    result = bayesian_walk_forward_multi(
        kline_df=bayesian_test_df, train_period=80, test_period=15,
        num_paths=1, initial_cash=1_000_000, spaces=basic_spaces,
    )
    assert not result.empty
    assert "sharpe_ratio" in result.columns
    assert "params" in result.columns
    assert "window" in result.columns
    assert result.iloc[0]["sharpe_ratio"] is not None


class TestWindowDatesCpcv:
    def _dates(self, n: int):
        import pandas as pd
        return pd.bdate_range("2023-01-01", periods=n).strftime("%Y-%m-%d").tolist()

    def test_baseline_no_purge_no_embargo(self) -> None:
        from BackTrading.bayesian.meta_optimizer import _window_dates
        dates = self._dates(100)
        windows = _window_dates(dates, train_period=40, test_period=10, offset=0)
        # 步长 = test_period，训练/测试首尾相接
        tr_s, tr_e, te_s, te_e = windows[0]
        assert tr_e == te_s
        assert te_e - te_s == 10
        assert len(windows) == 6  # (100-40)//10 = 6

    def test_purge_shrinks_train_end(self) -> None:
        from BackTrading.bayesian.meta_optimizer import _window_dates
        dates = self._dates(100)
        windows = _window_dates(dates, train_period=40, test_period=10, offset=0, purge_days=5)
        tr_s, tr_e, te_s, te_e = windows[0]
        # 训练尾部剔除 5 天：tr_e = start+40-5，测试起点仍 = start+40
        assert tr_e == te_s - 5
        assert te_s == 40

    def test_embargo_gap_between_train_and_test(self) -> None:
        from BackTrading.bayesian.meta_optimizer import _window_dates
        dates = self._dates(120)
        windows = _window_dates(dates, train_period=40, test_period=10, offset=0, embargo_days=3)
        tr_s, tr_e, te_s, te_e = windows[0]
        assert te_s - tr_e == 3  # 禁运间隔
        assert len(windows) == 6  # 步长 = 10+3=13，start∈{0,13,26,39,52,65}

    def test_purge_and_embargo_combined(self) -> None:
        from BackTrading.bayesian.meta_optimizer import _window_dates
        dates = self._dates(120)
        windows = _window_dates(dates, train_period=40, test_period=10, offset=0,
                                purge_days=5, embargo_days=3)
        tr_s, tr_e, te_s, te_e = windows[0]
        assert tr_e == 35          # 40 - 5
        assert te_s - tr_e == 8    # purge(5) + embargo(3) 的总间隔
        assert windows[0] != windows[1]

    def test_purge_larger_than_train_skips_invalid(self) -> None:
        from BackTrading.bayesian.meta_optimizer import _window_dates
        dates = self._dates(100)
        # purge 吃掉整个训练期 → 无有效窗口
        windows = _window_dates(dates, train_period=10, test_period=5, offset=0, purge_days=10)
        assert windows == []


def test_fallback_frame_has_full_schema(bayesian_test_df, basic_spaces) -> None:
    """无有效窗口时返回的兜底帧必须含 runner 依赖的全部绩效列。"""
    from BackTrading.bayesian.meta_optimizer import bayesian_walk_forward_multi
    # 数据量满足最低门槛，但 purge 吃掉全部训练期 → 所有窗口无效 → 兜底帧
    result = bayesian_walk_forward_multi(
        kline_df=bayesian_test_df, train_period=120, test_period=15,
        num_paths=2, initial_cash=1_000_000, spaces=basic_spaces,
        purge_days=200,
    )
    assert not result.empty
    for col in ("window", "params", "sharpe_ratio", "total_return",
                "max_drawdown", "num_combos", "num_paths"):
        assert col in result.columns, f"兜底帧缺少列 {col}"
    assert result.iloc[0]["total_return"] == 0.0
    assert result.iloc[0]["max_drawdown"] == 0.0


def test_fidelity_controller_with_signals(bayesian_test_df, basic_engine_cfg) -> None:
    from BackTrading.bayesian.cost_model import FidelityController
    ctrl = FidelityController(bayesian_test_df, basic_engine_cfg)
    assert ctrl._has_signals is True
    result = ctrl.evaluate({"atr_stop_mult": 2.0, "kelly_fraction": 0.3}, fidelity=0)
    assert "sharpe" in result
    assert "total_return" in result
    assert result["sharpe"] != -1e10


@pytest.mark.slow
def test_calibration_entry_point(bayesian_test_df, basic_spaces) -> None:
    from BackTrading.calibration import run_bayesian_walk_forward
    result = run_bayesian_walk_forward(
        kline_df=bayesian_test_df, train_period=80, test_period=15,
        num_paths=1, spaces=basic_spaces,
    )
    assert not result.empty
    assert "sharpe_ratio" in result.columns
