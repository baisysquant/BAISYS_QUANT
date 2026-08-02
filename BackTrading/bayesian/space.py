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


# config.ini _RANGE 字段名 → 参数名映射
# 注：kelly_fraction / position_a / liq_veto_ratio / risk_none_multiplier
# 已在引擎审计中确认为死参数（引擎仓位恒等权，不消费这些字段），
# 保留在寻优空间只会空转浪费预算并产出无意义的"最优值"，故不纳入。
_RANGE_TO_PARAM: dict[str, str] = {
    "ATR_STOP_MULT_RANGE": "atr_stop_mult",
    "BOLL_NARROW_RATIO_RANGE": "boll_narrow_ratio",
    "CROSS_DECAY_DAYS_RANGE": "cross_decay_days",
    "CONCLUSION_FULL_BULL_RANGE": "conclusion_full_bull",
    "GOLDEN_CROSS_BONUS_RANGE": "golden_cross_bonus",
    "DIVERGENCE_PENALTY_RANGE": "divergence_penalty",
    "BUY_THRESHOLD_RANGE": "buy_threshold",
    "MAX_HOLDINGS_RANGE": "max_holdings",
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


def build_spaces(backtest_config: Any) -> dict[str, ParamSpace]:
    """从 BacktestConfig 实例构建全参数空间。

    Args:
        backtest_config: ConfigParser.BacktestConfig 实例（含 parse_range 方法）。

    Returns:
        参数名 → ParamSpace 的 dict。
    """
    spaces: dict[str, ParamSpace] = {}
    for range_attr, param_name in _RANGE_TO_PARAM.items():
        try:
            low, high, step = backtest_config.parse_range(range_attr)
        except (AttributeError, ValueError, RuntimeError) as exc:
            logger.warning(f"解析 {range_attr} 失败: {exc}")
            continue
        spaces[param_name] = ParamSpace(
            name=param_name,
            low=low,
            high=high,
            step=step,
            is_signal=(param_name in _SIGNAL_PARAMS),
        )
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
