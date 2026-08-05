"""邻近参数抖动自检（Parameter Robustness Check）单元测试"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pandas as pd

from BackTrading.parameter_robustness import (
    run_robustness_check,
    _should_perturb,
    _build_perturbed_params,
    _build_engine_cfg,
    _prepare_for_params,
    HIGH_SHARPE_THRESHOLD,
    PERTURBATION_FRAC,
    ROBUSTNESS_SHARPE_DROP,
    MAX_ALLOWED_LOSS,
    RETURN_CLIFF_FRACTION,
    DEAD_KEYS,
)
from BackTrading.engine import EngineConfig


def _kline(n: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame({
        "trade_date": dates.strftime("%Y-%m-%d"),
        "symbol": "sh600000",
        "close": 10.0,
    })


def _best_params() -> dict:
    return {
        "atr_stop_mult": 2.0,
        "boll_narrow_ratio": 0.8,
        "cross_decay_days": 30,
        "conclusion_full_bull": 80,
        "buy_threshold": 15,
        "max_holdings": 10,
    }


# ── 1. 触发逻辑 ──

class TestTrigger:
    def test_not_triggered_below_threshold(self):
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest") as m:
            r = run_robustness_check(_kline(), _best_params(), 1.5)
            assert r.triggered is False
            m.assert_not_called()

    def test_triggered_at_threshold(self):
        """Sharpe > 2.0 严格大于才触发（业务定义）。"""
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=[(2.0, 1.5, 0.40), (1.9, 1.4, 0.38), (1.9, 1.4, 0.38)]) as m:
            r = run_robustness_check(_kline(), {"atr_stop_mult": 2.0}, 2.1)
            assert r.triggered is True
            m.assert_called()

    def test_not_triggered_exactly_at_threshold(self):
        """恰好等于 2.0 不触发。"""
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest") as m:
            r = run_robustness_check(_kline(), _best_params(), 2.0)
            assert r.triggered is False
            m.assert_not_called()

    def test_triggered_above_threshold(self):
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=[(2.0, 1.5, 0.40), (1.9, 1.4, 0.38), (1.9, 1.4, 0.38)]):
            r = run_robustness_check(_kline(), {"atr_stop_mult": 2.0}, 2.5)
        assert r.triggered is True
        assert r.eval_window_days == 250

    def test_base_evaluation_exception_skips(self):
        """基线回测异常时不阻断主流程，标记跳过。"""
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=RuntimeError("boom")):
            r = run_robustness_check(_kline(), _best_params(), 2.5)
        assert r.triggered is True
        assert r.overall_robust is True  # 无法评估，跳过而非误判 FAIL


# ── 2. 稳健性判定（同窗口对比 + 收益判据） ──

class TestRobustnessVerdict:
    def test_robust_when_perturbations_stable(self):
        """扰动后 Sharpe/收益平稳 → PASS。"""
        best = _best_params()
        n_perturb = len(best) * 2  # 每参数 ±10% 两个方向
        side_effect = [(2.0, 1.5, 0.40)] + [(1.9, 1.4, 0.38)] * n_perturb
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=side_effect) as m:
            r = run_robustness_check(_kline(), best, 2.5)
        assert r.triggered is True
        assert m.call_count == 1 + n_perturb
        assert r.overall_robust is True
        assert r.failed_params == []
        assert r.base_window_sharpe == 2.0
        assert r.base_window_return == 0.40

    def test_fail_sharpe_cliff(self):
        """扰动后 Sharpe 断崖（下跌 > 50%）→ FAIL 且为 CRITICAL。"""
        best = {"atr_stop_mult": 2.0}
        side_effect = [
            (2.0, 1.5, 0.40),              # 基线
            (0.8, 0.6, 0.10),              # +10%：Sharpe 下跌 60%
            (1.9, 1.4, 0.38),              # -10%：稳健
        ]
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=side_effect):
            r = run_robustness_check(_kline(), best, 2.5)
        assert r.overall_robust is False
        assert r.failed_params == ["atr_stop_mult"]
        assert r.warning_level == "CRITICAL"  # 唯一参数失败 → fail_ratio 1.0 > 0.5

    def test_fail_return_crash_big_loss(self):
        """扰动后收益转大额亏损（< -10%）→ FAIL。"""
        best = {"atr_stop_mult": 2.0}
        side_effect = [
            (2.0, 1.5, 0.40),              # 基线
            (1.8, 1.3, -0.12),             # +10%：大额亏损
            (1.9, 1.4, 0.38),
        ]
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=side_effect):
            r = run_robustness_check(_kline(), best, 2.5)
        assert r.overall_robust is False
        assert "atr_stop_mult" in r.failed_params
        fail_result = r.perturbation_results[0]
        assert fail_result.is_robust is False
        assert "大额亏损" in fail_result.detail

    def test_fail_return_cliff(self):
        """扰动后收益相对基线腰斩（< 基线×50%）→ FAIL（断崖式下跌）。"""
        best = {"atr_stop_mult": 2.0}
        side_effect = [
            (2.0, 1.5, 0.40),              # 基线
            (1.7, 1.2, 0.10),              # +10%：0.10 < 0.40×50% 断崖
            (1.9, 1.4, 0.38),
        ]
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=side_effect):
            r = run_robustness_check(_kline(), best, 2.5)
        assert r.overall_robust is False
        fail_result = r.perturbation_results[0]
        assert fail_result.is_robust is False
        assert "断崖" in fail_result.detail

    def test_warning_when_minority_fails(self):
        """部分参数失败（≤50%）→ WARNING 而非 CRITICAL。"""
        best = {"atr_stop_mult": 2.0, "boll_narrow_ratio": 0.8}
        side_effect = [
            (2.0, 1.5, 0.40),              # 基线
            (0.5, 0.3, -0.05),             # atr +10%：Sharpe 断崖
            (1.9, 1.4, 0.38),              # atr -10%
            (1.9, 1.4, 0.38),              # boll +10%
            (1.9, 1.4, 0.38),              # boll -10%
        ]
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=side_effect):
            r = run_robustness_check(_kline(), best, 2.5)
        assert r.overall_robust is False
        assert r.warning_level == "WARNING"  # 1/2 = 50%，不超半数


# ── 3. 参数扰动集（消费路径修复的核心） ──

class TestPerturbationSet:
    def test_dead_params_not_perturbed(self):
        """死参数（kelly_fraction/position_a/risk_none_multiplier 等）不扰动。

        这些参数引擎不消费，扰动后回测结果不变 → 恒稳健，只会稀释判定。
        """
        best = {"atr_stop_mult": 2.0, **{k: 0.3 for k in DEAD_KEYS}}
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest") as m:
            m.side_effect = [(2.0, 1.5, 0.40), (1.9, 1.4, 0.38), (1.9, 1.4, 0.38)]
            r = run_robustness_check(_kline(), best, 2.5)
        # 仅 atr_stop_mult 被扰动：1 次基线 + 2 次方向
        assert m.call_count == 3
        assert len(r.perturbation_results) == 2
        assert r.overall_robust is True

    def test_unknown_params_not_perturbed(self):
        """不在消费白名单的参数不扰动。"""
        best = {"atr_stop_mult": 2.0, "mystery_param": 3.0}
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest") as m:
            m.side_effect = [(2.0, 1.5, 0.40), (1.9, 1.4, 0.38), (1.9, 1.4, 0.38)]
            r = run_robustness_check(_kline(), best, 2.5)
        assert m.call_count == 3
        assert len(r.perturbation_results) == 2

    def test_engine_params_perturbed(self):
        """引擎消费参数（buy_threshold/max_holdings）参与扰动。"""
        best = _best_params()
        n_perturb = len(best) * 2
        side_effect = [(2.0, 1.5, 0.40)] + [(1.9, 1.4, 0.38)] * n_perturb
        with patch("BackTrading.parameter_robustness._run_perturbation_backtest",
                   side_effect=side_effect):
            r = run_robustness_check(_kline(), best, 2.5)
        names = {res.param_name for res in r.perturbation_results}
        assert "buy_threshold" in names
        assert "max_holdings" in names
        assert "conclusion_full_bull" in names

    def test_should_perturb_rules(self):
        """扰动规则：浮点 ±10%、整数 ±1 档、零值/死参数/未知参数跳过。"""
        assert _should_perturb(2.0, "atr_stop_mult") == (True, 0.20)
        assert _should_perturb(30, "cross_decay_days") == (True, 1.0)
        assert _should_perturb(80, "conclusion_full_bull") == (True, 1.0)
        assert _should_perturb(0.0, "atr_stop_mult") == (False, 0.0)
        assert _should_perturb(0.25, "kelly_fraction") == (False, 0.0)
        assert _should_perturb(3.0, "mystery_param") == (False, 0.0)

    def test_build_perturbed_params_int_rounding(self):
        p = _build_perturbed_params({"cross_decay_days": 30}, "cross_decay_days", 1.0, "+10%")
        assert p["cross_decay_days"] == 31
        p2 = _build_perturbed_params({"cross_decay_days": 1}, "cross_decay_days", 1.0, "-10%")
        assert p2["cross_decay_days"] == 1  # 下限保护


# ── 4. 参数分派（扰动真实生效的保证） ──

class TestParamDispatch:
    def test_build_engine_cfg_applies_engine_params(self):
        base = EngineConfig(atr_stop_mult=1.5, buy_threshold=15, max_holdings=5)
        flat = {"atr_stop_mult": 2.5, "buy_threshold": 18, "max_holdings": 12}
        cfg = _build_engine_cfg(flat, base)
        assert cfg.atr_stop_mult == 2.5
        assert cfg.buy_threshold == 18
        assert cfg.max_holdings == 12
        assert isinstance(cfg.buy_threshold, int)
        # 未扰动的字段保持不变
        assert cfg.kelly_fraction == base.kelly_fraction

    def test_prepare_dispatch_flat_path(self):
        """无 config 时 flat 参数直通 prepare（其内置分派处理 thresholds/regime）。"""
        df = _kline()
        flat = {"atr_stop_mult": 2.0, "boll_narrow_ratio": 0.9, "conclusion_full_bull": 90}
        with patch("BackTrading.parameter_robustness.prepare_backtest_data") as m:
            m.return_value = df
            _prepare_for_params(df, flat, None)
        passed = m.call_args.kwargs["params"]
        assert passed is flat

    def test_prepare_dispatch_config_path(self):
        """有 config 时按消费路径分派：scoring / regime / thresholds。"""
        df = _kline()
        fake_cfg = types.SimpleNamespace(
            app_config=types.SimpleNamespace(
                regime_detection=MagicMock(model_dump=lambda: {"boll_narrow_ratio": 0.8}),
                divergence=MagicMock(model_dump=lambda: {}),
                scoring_params=MagicMock(model_dump=lambda: {}),
                position_sizing=MagicMock(RISK_NONE_MULTIPLIER=1.0),
                technical_constants=MagicMock(model_dump=lambda: {}),
                full_bull_scoring=MagicMock(CONCLUSION_FULL_BULL=80, CONCLUSION_BULLISH=75, CONCLUSION_OSCILLATE=60),
            )
        )
        flat = {"atr_stop_mult": 2.0, "boll_narrow_ratio": 0.9, "conclusion_full_bull": 90}
        with patch("BackTrading.parameter_robustness.prepare_backtest_data") as m:
            m.return_value = df
            _prepare_for_params(df, flat, fake_cfg)
        passed = m.call_args.kwargs["params"]
        assert passed["scoring"]["atr_stop_mult"] == 2.0
        assert passed["regime"]["boll_narrow_ratio"] == 0.9
        assert passed["thresholds"]["fully_bull"] == 90
