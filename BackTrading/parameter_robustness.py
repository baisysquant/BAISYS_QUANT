"""
邻近参数抖动自检（Parameter Robustness Check）

业务定义：当回测产生优异夏普比率（Sharpe > 2.0）时，自动对关键参数
进行 ±10% 微调并重新评估。若收益出现断崖式下跌或变成大额亏损，
判定该策略不具备统计稳健性，标记为"参数敏感型"风险。

自检内容：
  1. 高 Sharpe 触发：当 Sharpe > 2.0 时自动触发
  2. 关键参数抖动：对每个可消费参数施加 +10% / -10% 扰动
     （整数参数退化为 ±1 档；死参数与未消费参数不扰动）
  3. 同窗口重新评估：基线与扰动均在相同评估窗口上重算，
     保证 Sharpe/收益对比口径一致（避免"60 天尾部 vs 全周期"错位）
  4. 稳健性判定：满足任一条件即 FAIL
     - 扰动后 Sharpe 下跌 > 50% 或转负
     - 扰动后收益 < -10%（大额亏损）
     - 扰动后收益 < 基线收益 × 50%（断崖式下跌）

用法:
    from BackTrading.parameter_robustness import run_robustness_check
    report = run_robustness_check(
        kline_df, best_params, original_sharpe, config, engine_cfg,
    )
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig, _run_single_backtest
from BackTrading.prepare import _build_params, prepare_backtest_data
from LogicAnalyzer.backtest_metrics import compute_risk_metrics


# 高 Sharpe 触发阈值
HIGH_SHARPE_THRESHOLD = 2.0
# 扰动幅度
PERTURBATION_FRAC = 0.10  # ±10%
# 稳健性判定：Sharpe 下跌超过此比例即 FAIL
ROBUSTNESS_SHARPE_DROP = 0.50  # 50%
# 同窗口评估：基线与扰动在同一评估窗口上对比（交易日）
ROBUSTNESS_EVAL_WINDOW = 250
# warmup 天数（评估段前预留交易日 buffer，让引擎建立 ADV/仓位后再进入评估段）
_ROBUSTNESS_WARMUP_DAYS = 30
# 扰动后收益低于此值视为大额亏损
MAX_ALLOWED_LOSS = -0.10
# 扰动后收益 < 基线收益 × 此比例视为断崖式下跌
RETURN_CLIFF_FRACTION = 0.50

# 参数消费路径分派（与 prepare.py flat 分派 / engine/core.py / vectorized_signal.py 保持一致）
# 经 prepare_backtest_data 消费：scoring / regime / thresholds
PREPARE_CONSUMED = {
    "atr_stop_mult", "boll_narrow_ratio", "cross_decay_days",
    "golden_cross_bonus", "divergence_penalty", "conclusion_full_bull",
}
# 经 EngineConfig 消费（扰动时必须重建 engine_cfg，否则扰动不生效）
ENGINE_CONSUMED = {"atr_stop_mult", "buy_threshold", "max_holdings"}
# 引擎审计确认的死参数（引擎仓位恒等权，不消费这些字段），扰动无意义，跳过
DEAD_KEYS = {"kelly_fraction", "position_a", "liq_veto_ratio", "risk_none_multiplier"}
# 整数型参数：用 ±1 档而非百分比
INT_PARAMS = {"cross_decay_days", "conclusion_full_bull", "buy_threshold", "max_holdings"}
# 全部可扰动参数 = 实际被消费的参数 ∪ 引擎参数
PERTURBABLE_KEYS = PREPARE_CONSUMED | ENGINE_CONSUMED


@dataclass
class PerturbationResult:
    """单个参数扰动结果。"""

    param_name: str = ""
    original_value: float = 0.0
    perturbed_value: float = 0.0
    direction: str = ""            # "+10%" / "-10%"
    perturbed_sharpe: float = 0.0
    perturbed_sortino: float = 0.0
    perturbed_return: float = 0.0
    sharpe_drop_pct: float = 0.0
    return_drop_pct: float = 0.0
    is_robust: bool = True
    detail: str = ""


@dataclass
class RobustnessReport:
    """参数稳健性自检报告。"""

    triggered: bool = False
    original_sharpe: float = 0.0
    best_params: dict[str, float] = field(default_factory=dict)
    perturbation_results: list[PerturbationResult] = field(default_factory=list)
    overall_robust: bool = True
    warning_level: str = "INFO"     # INFO / WARNING / CRITICAL
    failed_params: list[str] = field(default_factory=list)
    eval_window_days: int = 0       # 同窗口评估窗口（交易日）
    base_window_sharpe: float = 0.0
    base_window_return: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "original_sharpe": round(self.original_sharpe, 4),
            "eval_window_days": self.eval_window_days,
            "base_window_sharpe": round(self.base_window_sharpe, 4),
            "base_window_return": round(self.base_window_return, 4),
            "overall_robust": self.overall_robust,
            "warning_level": self.warning_level,
            "tested_params": len(self.perturbation_results),
            "failed_params": self.failed_params,
            "details": [
                {
                    "param": r.param_name,
                    "original": round(r.original_value, 6),
                    "perturbed": round(r.perturbed_value, 6),
                    "direction": r.direction,
                    "perturbed_sharpe": round(r.perturbed_sharpe, 4),
                    "perturbed_return": round(r.perturbed_return, 4),
                    "sharpe_drop": f"{r.sharpe_drop_pct:.1%}",
                    "return_drop": f"{r.return_drop_pct:.1%}",
                    "is_robust": r.is_robust,
                }
                for r in self.perturbation_results
            ],
        }

    def log(self) -> None:
        if not self.triggered:
            logger.info(f"[参数稳健性] Sharpe={self.original_sharpe:.2f} ≤ {HIGH_SHARPE_THRESHOLD}，不触发")
            return

        if self.overall_robust:
            logger.info(
                f"[参数稳健性] PASS | 原始 Sharpe={self.original_sharpe:.2f} | "
                f"窗口基线 Sharpe={self.base_window_sharpe:.2f}/收益={self.base_window_return:.2%} | "
                f"测试 {len(self.perturbation_results)} 个参数扰动，全部稳健"
            )
        else:
            severity = "🔴 CRITICAL" if self.warning_level == "CRITICAL" else "⚠️ WARNING"
            logger.warning(
                f"[参数稳健性] {severity} | 原始 Sharpe={self.original_sharpe:.2f} | "
                f"测试 {len(self.perturbation_results)} 个参数扰动，"
                f"{len(self.failed_params)} 个不稳健: {self.failed_params}"
            )
            for r in self.perturbation_results:
                if not r.is_robust:
                    logger.warning(
                        f"  {r.param_name}: {r.original_value:.4f} → {r.perturbed_value:.4f} ({r.direction}) "
                        f"| Sharpe {self.base_window_sharpe:.4f} → {r.perturbed_sharpe:.4f} "
                        f"(下跌 {r.sharpe_drop_pct:.1%}) | 收益 → {r.perturbed_return:.2%}"
                    )


def _should_perturb(value: float, param_name: str) -> tuple[bool, float]:
    """判断参数是否适合扰动，返回 (是否扰动, 扰动幅度)。

    - 死参数与未被消费的参数不扰动（扰动无效果，只会稀释判定）
    - 整数型参数（如 cross_decay_days）使用 ±1 档而非百分比
    """
    if value == 0:
        return False, 0.0

    if param_name not in PERTURBABLE_KEYS or param_name in DEAD_KEYS:
        return False, 0.0

    if param_name in INT_PARAMS:
        return True, 1.0  # ±1 档

    return True, abs(value) * PERTURBATION_FRAC


def _build_perturbed_params(
    best_params: dict[str, float],
    param_name: str,
    perturbation: float,
    direction: str,
) -> dict[str, float]:
    """生成扰动后的参数副本。"""
    perturbed = dict(best_params)
    orig = perturbed[param_name]
    if direction == "+10%":
        new_val = orig + perturbation
    else:
        new_val = orig - perturbation

    # 整数型参数取整
    if param_name in INT_PARAMS:
        new_val = max(1, int(new_val))
    else:
        # 防止负值（如 atr_stop_mult 不能为负）
        new_val = max(new_val, 1e-6)

    perturbed[param_name] = new_val
    return perturbed


def _build_engine_cfg(
    flat_params: dict[str, float],
    engine_cfg: EngineConfig | None,
) -> EngineConfig:
    """从扰动参数重建 EngineConfig，保证引擎参数扰动真实生效。

    审计（成本外部化）：engine_cfg 缺失时显式携带默认 CostModel()（含逐笔最低
    佣金与分段表），不再构造无 cost_model 的 EngineConfig（引擎内会 fail-fast）。
    """
    if engine_cfg is None:
        from BackTrading.domain.models import CostModel

        engine_cfg = EngineConfig(cost_model=CostModel())
    cfg = engine_cfg
    updates = {
        k: flat_params[k] for k in ENGINE_CONSUMED if k in flat_params
    }
    if not updates:
        return cfg
    # int 参数保持整数类型（dataclass 字段类型一致性）
    for k in ("buy_threshold", "max_holdings"):
        if k in updates:
            updates[k] = int(updates[k])
    return dataclasses.replace(cfg, **updates)


def _prepare_for_params(
    kline_df: pd.DataFrame,
    flat_params: dict[str, float],
    config: Any | None,
) -> pd.DataFrame:
    """按 flat 参数准备回测数据。

    直接走 prepare_backtest_data 的 flat 分派（scoring/regime/thresholds），
    与主回测 runner 的分派逻辑保持一致，避免 whitelist 错配导致扰动不生效。
    """
    if config is not None:
        # 以配置为基底，仅覆盖被扰动的键（与 runner 主回测结构一致）
        structured = _build_params(config)
        structured["scoring"].update(
            {k: v for k, v in flat_params.items()
             if k in ("atr_stop_mult", "cross_decay_days", "golden_cross_bonus", "divergence_penalty")}
        )
        if "boll_narrow_ratio" in flat_params:
            structured["regime"]["boll_narrow_ratio"] = float(flat_params["boll_narrow_ratio"])
        if "conclusion_full_bull" in flat_params:
            structured["thresholds"]["fully_bull"] = int(flat_params["conclusion_full_bull"])
        return prepare_backtest_data(
            kline_df, params=structured, compute_exit_strategy=False, vectorized=True,
        )
    return prepare_backtest_data(
        kline_df, params=flat_params, compute_exit_strategy=False, vectorized=True,
    )


def _run_perturbation_backtest(
    kline_df: pd.DataFrame,
    perturbed_params: dict[str, float],
    engine_cfg: EngineConfig | None,
    config: Any | None,
    sim_days: int = ROBUSTNESS_EVAL_WINDOW,
) -> tuple[float, float, float]:
    """在评估窗口上运行单次回测，返回 (Sharpe, Sortino, 总收益)。

    基线评估与扰动评估必须使用相同窗口，保证对比口径一致。
    """
    dates = sorted(kline_df["trade_date"].unique())
    if len(dates) < sim_days + 20:
        return 0.0, 0.0, 0.0  # 数据不足，跳过

    # 准备信号
    prepared = _prepare_for_params(kline_df, perturbed_params, config)

    # ── P2-4：warmup buffer 避免 ADV 冷启动 ──
    # 与 simulated_trading.py P1-1 同类修复：评估窗口前 ~20 天 ADV 未满载，
    # 加 30 天 warmup 让引擎建立 ADV/仓位后再进入评估段，指标仅在评估段计算。
    if pd.api.types.is_datetime64_any_dtype(prepared["trade_date"]):
        _prep = prepared.copy()
        _prep["trade_date"] = _prep["trade_date"].dt.strftime("%Y-%m-%d")
    else:
        _prep = prepared

    _date_range = _prep["trade_date"].astype(str)
    _unique_dates = sorted(_date_range.unique())
    _sim_n = min(sim_days, len(_unique_dates))
    _sim_start_pos = len(_unique_dates) - _sim_n
    sim_dates_str = set(_unique_dates[_sim_start_pos:])
    # 扩展段 [warmup_start, sim_end]
    _warmup_pos = max(0, _sim_start_pos - _ROBUSTNESS_WARMUP_DAYS)
    _warmup_start = _unique_dates[_warmup_pos]
    _sim_end = _unique_dates[-1]
    mask_ext = (_date_range >= _warmup_start) & (_date_range <= _sim_end)
    ext_data = _prep[mask_ext].copy()
    if ext_data.empty:
        return 0.0, 0.0, 0.0

    # 止损价
    stop_mult = perturbed_params.get("atr_stop_mult")
    if stop_mult is not None and "ATR" in ext_data.columns:
        # P0-1：止损价与引擎比较基准统一到后复权空间（指标 ATR 亦为后复权）
        _stop_close = ext_data["close_normal"] if "close_normal" in ext_data.columns else ext_data["close"]
        ext_data["止损价"] = _stop_close - ext_data["ATR"] * stop_mult
    elif "止损价" not in ext_data.columns:
        ext_data["止损价"] = 0.0

    ecfg = _build_engine_cfg(perturbed_params, engine_cfg)

    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(ext_data, perturbed_params, ecfg, tl, ec)

    # 仅保留评估段权益曲线（warmup 段不入指标）
    ec_sim = [row for row in ec if str(row.get("time", ""))[:10] in sim_dates_str]
    if not ec_sim:
        return 0.0, 0.0, 0.0

    risk = compute_risk_metrics(ec_sim) or {}
    return (
        float(risk.get("sharpe_ratio", 0.0) or 0.0),
        float(risk.get("sortino_ratio", 0.0) or 0.0),
        float(risk.get("total_return", 0.0) or 0.0),
    )


def run_robustness_check(
    kline_df: pd.DataFrame,
    best_params: dict[str, float],
    original_sharpe: float,
    config: Any | None = None,
    engine_cfg: EngineConfig | None = None,
    sim_days: int = ROBUSTNESS_EVAL_WINDOW,
) -> RobustnessReport:
    """邻近参数抖动自检入口。

    Args:
        kline_df: 全量 K 线数据。
        best_params: WFO 选出的最佳参数。
        original_sharpe: 原始 Sharpe 比率（建议传多重测试惩罚后的值）。
        config: Config 实例（可选）。
        engine_cfg: EngineConfig（可选）。
        sim_days: 同窗口评估窗口（交易日）。

    Returns:
        RobustnessReport。
    """
    report = RobustnessReport(
        original_sharpe=original_sharpe,
        best_params=best_params,
        eval_window_days=sim_days,
    )

    # ── 触发检查 ──
    if original_sharpe <= HIGH_SHARPE_THRESHOLD:
        report.triggered = False
        report.log()
        return report

    report.triggered = True
    logger.info(
        f"[参数稳健性] 触发检查: Sharpe={original_sharpe:.2f} > {HIGH_SHARPE_THRESHOLD}"
    )

    # ── 同窗口基线回测（未扰动），保证对比口径一致 ──
    try:
        base_sharpe, _, base_return = _run_perturbation_backtest(
            kline_df, best_params, engine_cfg, config, sim_days=sim_days,
        )
    except Exception as e:
        logger.warning(f"[参数稳健性] 基线回测异常: {e}，自检跳过（建议人工复核）")
        report.log()
        return report
    report.base_window_sharpe = base_sharpe
    report.base_window_return = base_return

    # 窗口内数据不足或基线无收益 → 无法进行收益对比，仅用 Sharpe 判据
    base_has_return = base_sharpe > 0
    if not base_has_return:
        logger.warning(
            f"[参数稳健性] 评估窗口({sim_days} 交易日)内基线 Sharpe={base_sharpe:.2f}，"
            f"收益对比判据停用，仅按 Sharpe 判据评估"
        )

    # ── 筛选可消费参数进行扰动 ──
    float_params = {
        k: v for k, v in best_params.items()
        if isinstance(v, (int, float)) and not str(k).startswith("_")
    }
    if not float_params:
        report.overall_robust = True
        report.log()
        return report

    # ── 对每个参数做 +10% / -10% 扰动 ──
    for param_name, orig_value in float_params.items():
        should_pert, perturbation = _should_perturb(orig_value, param_name)
        if not should_pert:
            continue

        for direction in ["+10%", "-10%"]:
            perturbed_params = _build_perturbed_params(best_params, param_name, perturbation, direction)

            try:
                p_sharpe, p_sortino, p_return = _run_perturbation_backtest(
                    kline_df, perturbed_params,
                    engine_cfg, config, sim_days=sim_days,
                )
            except Exception as e:
                logger.warning(f"[参数稳健性] {param_name} {direction} 扰动回测异常: {e}")
                p_sharpe, p_sortino, p_return = 0.0, 0.0, 0.0

            sharpe_drop = 1.0 - (p_sharpe / base_sharpe) if base_sharpe > 0 else 0.0
            return_drop = (
                1.0 - (p_return / base_return) if base_has_return and base_return > 0 else 0.0
            )
            # 断崖式下跌：收益相对基线腰斩；大额亏损：绝对收益跌破下限
            return_cliff = base_has_return and base_return > 0 and p_return < base_return * RETURN_CLIFF_FRACTION
            big_loss = p_return < MAX_ALLOWED_LOSS
            is_robust = (
                sharpe_drop < ROBUSTNESS_SHARPE_DROP
                and p_sharpe > 0
                and not return_cliff
                and not big_loss
            )

            reasons = []
            if sharpe_drop >= ROBUSTNESS_SHARPE_DROP or p_sharpe <= 0:
                reasons.append(f"Sharpe 下跌 {sharpe_drop:.1%} 或转负")
            if return_cliff:
                reasons.append(f"收益断崖（{p_return:.2%} < 基线×{RETURN_CLIFF_FRACTION:.0%}）")
            if big_loss:
                reasons.append(f"大额亏损（{p_return:.2%} < {MAX_ALLOWED_LOSS:.0%}）")

            result = PerturbationResult(
                param_name=param_name,
                original_value=orig_value,
                perturbed_value=perturbed_params[param_name],
                direction=direction,
                perturbed_sharpe=p_sharpe,
                perturbed_sortino=p_sortino,
                perturbed_return=p_return,
                sharpe_drop_pct=sharpe_drop,
                return_drop_pct=return_drop,
                is_robust=is_robust,
                detail="；".join(reasons) if reasons else "",
            )
            report.perturbation_results.append(result)

            if not is_robust and param_name not in report.failed_params:
                report.failed_params.append(param_name)

    # ── 综合判定 ──
    report.overall_robust = len(report.failed_params) == 0
    if not report.overall_robust:
        fail_ratio = len(report.failed_params) / len(float_params) if float_params else 0
        if fail_ratio > 0.5:
            report.warning_level = "CRITICAL"
        else:
            report.warning_level = "WARNING"

    report.log()
    return report
