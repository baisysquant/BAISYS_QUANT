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
_RANGE_TO_PARAM: dict[str, str] = {
    "ATR_STOP_MULT_RANGE": "atr_stop_mult",
    "KELLY_FRACTION_RANGE": "kelly_fraction",
    "POSITION_A_RANGE": "position_a",
    "LIQ_VETO_RATIO_RANGE": "liq_veto_ratio",
    "BOLL_NARROW_RATIO_RANGE": "boll_narrow_ratio",
    "CROSS_DECAY_DAYS_RANGE": "cross_decay_days",
    "CONCLUSION_FULL_BULL_RANGE": "conclusion_full_bull",
    "GOLDEN_CROSS_BONUS_RANGE": "golden_cross_bonus",
    "DIVERGENCE_PENALTY_RANGE": "divergence_penalty",
    "RISK_NONE_MULTIPLIER_RANGE": "risk_none_multiplier",
}

# 影响信号计算的参数（昂贵）—— 与 prepare._compute_param_hash 保持一致
_SIGNAL_PARAMS: set[str] = {
    "boll_narrow_ratio",
    "cross_decay_days",
    "golden_cross_bonus",
    "divergence_penalty",
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


def fallback_midpoints(spaces: dict[str, ParamSpace]) -> dict[str, float]:
    """取每个参数范围的中位数作为兜底值。"""
    return {name: (sp.low + sp.high) / 2 for name, sp in spaces.items()}


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
