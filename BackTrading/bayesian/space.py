from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ParamSpace:
    """单一参数的定义空间，从 config.ini _RANGE 字段解析。

    Attributes:
        name: 参数名（如 atr_stop_mult）
        low: 下界
        high: 上界
        step: 步长（None = 连续空间）
        is_signal: True = 影响信号计算（昂贵），False = 仅影响回测引擎（廉价）
    """
    name: str
    low: float
    high: float
    step: float | None = None
    is_signal: bool = False

    @property
    def n_ticks(self) -> int:
        """离散化后的档位数（step = None 时返回 0 表示连续）。"""
        if self.step is None or self.step <= 0:
            return 0
        return int((self.high - self.low) / self.step) + 1

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def extend_to_cover(self, value: float, margin: float = 1.0) -> bool:
        """扩展边界以覆盖指定值（含 margin 余量）。返回是否发生扩展。

        FIX(D): 校准历史参数（如 buy_threshold=12、atr_stop_mult=5.5）可能超出
        当前 _RANGE 搜索空间（如 30-60 / 1-4）。BO 只在空间内采样，预检参数
        若不在空间内，预检 Sharpe 高但 BO 永远搜不到该区域 → "预检 1.08 vs
        BO best -0.19"的根源。此方法把边界放宽至覆盖历史好点，让 BO 有机会
        在预检点邻域继续搜索；同时保证 discrete step 语义不破坏（新边界仍对齐
        步长，供 _from_normalized 取整）。
        """
        step = self.step if (self.step is not None and self.step > 0) else 1.0
        new_low = min(self.low, value - margin)
        new_high = max(self.high, value + margin)
        # 步长对齐：low 向下对齐、high 向上对齐，保证空间覆盖值
        new_low = float(np.floor(new_low / step) * step)
        new_high = float(np.ceil(new_high / step) * step)
        changed = (new_low < self.low) or (new_high > self.high)
        if changed:
            logger.info(
                f"[空间扩展] {self.name}: [{self.low},{self.high}]{f' x{self.step}' if self.step else ''}"
                f" → [{new_low},{new_high}]（覆盖校准历史值 {value}）"
            )
            self.low = new_low
            self.high = new_high
        return changed


def extend_spaces_to_cover(
    spaces: dict[str, ParamSpace],
    params: dict[str, float],
    margin: float = 1.0,
) -> bool:
    """对 spaces 中存在的参数，将其边界扩展以覆盖 params 中的值。返回是否发生扩展。

    FIX(D): 在 BO 之前调用 — 把预检/上次校准参数纳入搜索空间，避免"预检好点
    在空间外、BO 永远搜不到"的口径断裂。仅扩展已有参数的空间；params 中不在
    空间里的键（如 golden_cross_bonus/divergence_penalty 固定参数）直接忽略。
    """
    changed = False
    for name, sp in spaces.items():
        if name in params:
            changed = sp.extend_to_cover(float(params[name]), margin=margin) or changed
    return changed


# config.ini _RANGE 字段名 → 参数名映射
# 注：kelly_fraction / position_a / position_b / position_c / risk_none_multiplier
# 已在引擎审计中确认为 DEAD_KEYS（回测引擎仓位模型=等权，不消费这些字段）；
# 保留在 config.ini [BACKTEST_CALIBRATED] 供复盘 PositionSizer 读取（受控静态参数），
# 见 calibration.py CALIB_PARAM_MAP 中的 "受控静态参数" 注释。
# 若将来回测引擎改为 Kelly 仓位模型，需同步：
#  1. 从 parameter_robustness.py DEAD_KEYS 移除
#  2. 在此 _RANGE_TO_PARAM 中声明搜索空间
#  3. 在 config.ini 中声明 *_RANGE 字段
_RANGE_TO_PARAM: dict[str, str] = {
    # ── 信号/策略参数 ──
    "ATR_STOP_MULT_RANGE": "atr_stop_mult",
    "BOLL_NARROW_RATIO_RANGE": "boll_narrow_ratio",
    "CROSS_DECAY_DAYS_RANGE": "cross_decay_days",
    "CONCLUSION_FULL_BULL_RANGE": "conclusion_full_bull",
    # 但显著膨胀搜索空间维度，固定为配置默认值，不参与WFO寻优
    "BUY_THRESHOLD_RANGE": "buy_threshold",
    "MAX_HOLDINGS_RANGE": "max_holdings",
    # ── P4 组合优化器超参数 ──
    "OPTIMIZER_RISK_AVERSION_RANGE": "optimizer_risk_aversion",
    "OPTIMIZER_TURNOVER_PENALTY_RANGE": "optimizer_turnover_penalty",
    "OPTIMIZER_MAX_WEIGHT_RANGE": "optimizer_max_weight",
    "OPTIMIZER_COV_LOOKBACK_RANGE": "optimizer_cov_lookback",
}

# 影响信号计算的参数（昂贵）—— 与 prepare._compute_param_hash 保持一致
# 注：conclusion_full_bull 直接决定风险等级/进出场阈值（vectorized_signal），
# 必须纳入信号哈希做缓存隔离，否则评估会复用旧阈值的信号。
_SIGNAL_PARAMS: set[str] = {
    "boll_narrow_ratio",
    "cross_decay_days",
    "golden_cross_bonus",
    "divergence_penalty",
    "conclusion_full_bull",
}


# 以下参数经敏感性分析对 OOS 绩效影响微小，但显著膨胀搜索空间维度（10→6 维）。
# GP 在 10 维空间需 ~1024 个观测点才能充分覆盖，实际只有 ~350 次评估。
_FIXED_PARAMS: dict[str, tuple[float, str]] = {
    # (固定值, fixed_reason)
    "optimizer_cov_lookback": (120.0, "协方差估计窗口，固定为 120 交易日（~6 个月），敏感性可忽略"),
    "optimizer_risk_aversion": (0.5, "风险厌恶系数 0.5 中等风险偏好，对 Sharpe 影响 <2%"),
    "optimizer_turnover_penalty": (0.01, "换手惩罚 0.01 已足够抑制频繁交易，继续优化收益递减"),
    "optimizer_max_weight": (0.15, "单只上限 15% 为合理分散水平，>15% 集中风险与 Sharpe 无显著关系"),
}

# _RANGE_TO_PARAM 去掉固定的参数后的搜索参数
_ACTIVE_RANGE_ATTRS = [
    "ATR_STOP_MULT_RANGE",
    "BOLL_NARROW_RATIO_RANGE",
    "CROSS_DECAY_DAYS_RANGE",
    "CONCLUSION_FULL_BULL_RANGE",
    "BUY_THRESHOLD_RANGE",
    "MAX_HOLDINGS_RANGE",
]


def _get_fixed_params() -> dict[str, float]:
    """返回被固定的参数及其合理默认值。"""
    return {k: v[0] for k, v in _FIXED_PARAMS.items()}


def build_spaces(
    backtest_config: Any,
    portfolio_optimizer_config: Any | None = None,
) -> dict[str, ParamSpace]:
    """从配置实例构建全参数搜索空间（已排除固定参数）。

    Args:
        backtest_config: ConfigParser.BacktestConfig 实例（含 parse_range 方法）。
        portfolio_optimizer_config: ConfigParser.PortfolioOptimizerConfig 实例（可选，
            提供优化器超参数 _RANGE 字段）。

    Returns:
        参数名 → ParamSpace 的 dict（仅含搜索参数，不含 _FIXED_PARAMS）。
    """
    spaces: dict[str, ParamSpace] = {}

    # 确定配置源字典: _RANGE 字段可能分布在 backtest 和 portfolio_optimizer 两个段
    config_sources: list[Any] = [backtest_config]
    if portfolio_optimizer_config is not None:
        config_sources.append(portfolio_optimizer_config)

    # 只解析活跃参数的 RANGE（跳过 _FIXED_PARAMS 中的参数）
    active_param_names = {
        "atr_stop_mult", "boll_narrow_ratio", "cross_decay_days",
        "conclusion_full_bull", "buy_threshold", "max_holdings",
    }
    range_attr_map = {
        k: v for k, v in _RANGE_TO_PARAM.items()
        if v in active_param_names
    }
    for range_attr, param_name in range_attr_map.items():
        found = False
        for cfg in config_sources:
            try:
                low, high, step = cfg.parse_range(range_attr)
                spaces[param_name] = ParamSpace(
                    name=param_name,
                    low=low,
                    high=high,
                    step=step,
                    is_signal=(param_name in _SIGNAL_PARAMS),
                )
                found = True
                break
            except (AttributeError, ValueError, RuntimeError):
                continue
        if not found:
            logger.warning(f"解析 {range_attr} 失败（所有配置源均未找到）")
    return spaces


def split_by_cost(spaces: dict[str, ParamSpace]) -> tuple[dict[str, ParamSpace], dict[str, ParamSpace]]:
    """按评估成本拆分参数空间。

    Returns:
        (signal_spaces, portfolio_spaces) — 信号参数 vs 组合参数。
    """
    signal = {}
    portfolio = {}
    for name, sp in spaces.items():
        if sp.is_signal:
            signal[name] = sp
        else:
            portfolio[name] = sp
    return signal, portfolio


def describe(spaces: dict[str, ParamSpace]) -> str:
    """可读的空间描述（用于日志）。"""
    signal, portfolio = split_by_cost(spaces)
    lines = [f"  信号参数({len(signal)}): " + ", ".join(
        f"{s.name}[{s.low},{s.high}]" for s in signal.values()
    )]
    lines.append(f"  组合参数({len(portfolio)}): " + ", ".join(
        f"{s.name}[{s.low},{s.high}]" for s in portfolio.values()
    ))
    return "\n".join(lines)