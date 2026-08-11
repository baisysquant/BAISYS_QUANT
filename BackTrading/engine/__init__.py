from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from UtilsManager.ConfigParser import Config
from BackTrading.domain.models import CostModel


@dataclass
class EngineConfig:
    """回测引擎配置 - 纯数据容器，无业务逻辑"""

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.001
    transfer_fee_rate: float = 0.00001  # 过户费 0.001% 双边
    max_position_pct: float = 0.1
    portfolio_method: str = "score_weighted"
    point_in_time: bool = True
    atr_stop_mult: float = 1.5
    kelly_fraction: float = 0.25
    position_a: float = 0.3
    boll_narrow_ratio: float = 0.8
    cross_decay_days: int = 30
    risk_none_multiplier: float = 1.0
    max_holdings: int = 0  # 0=不限制
    buy_threshold: int = 15  # 买入评分阈值
    min_commission_per_trade: float = 5.0  # A股每笔最低佣金 5 元
    cost_model: Any = None  # CostModel | None — forward ref to avoid circular import
    # ── 成交时点模型（0.1 执行时序合规） ──
    # close=信号日收盘价成交（老行为，偏乐观）/ next_open=信号次日开盘价成交（默认，符合A股T+1）
    # vwap=信号次日VWAP（典型价）成交。next_open/vwap 下单挂至次日开盘撮合，
    # 并与 simulate_limit_up_down 联动：次日一字涨停不可买入、一字跌停不可卖出。
    execution_model: str = "next_open"  # close / next_open / vwap
    # ── 涨跌停撮合约束（simulate_limit_up_down=true 开启可成交量模型） ──
    simulate_limit_up_down: bool = True  # false=回退简化撮合（触板一律禁买/禁卖）
    limit_seal_ratio: float = 0.05  # 一字板（开=收=限价）可成交量比例
    limit_tradable_ratio: float = 0.30  # 盘中触板可成交量比例
    limit_seal_decay: float = 0.5  # 连续板每板可成交量衰减系数
    # ── 0.6 复牌跳空（停牌后复牌日开盘大幅跳空：补涨兑现 / 补跌标记） ──
    resume_gap_up: float = 0.05  # 复牌高开≥该比例（相对停牌前收盘）→ 开盘兑现卖出 + 当日禁买
    resume_gap_down: float = 0.05  # 复牌低开≤-该比例 → 日志标记（风控卖出照常）

    @property
    def buy_fee_rate(self) -> float:
        """买入费率（不含滑点）：佣金 + 过户费"""
        return self.commission_rate + self.transfer_fee_rate

    @property
    def sell_fee_rate(self) -> float:
        """卖出费率（不含滑点）：佣金 + 过户费 + 印花税"""
        return self.commission_rate + self.transfer_fee_rate + self.stamp_tax_rate


# ── 引擎公共 API re-export ──
# core.py 依赖上方已定义的 EngineConfig；此处放在类定义之后以避免循环 import：
#   __init__.py 先定义 EngineConfig → 再 import core → core 反向 import EngineConfig（已就绪）
from BackTrading.engine.core import (  # noqa: E402
    _MIN_SLIPPAGE_FLOOR,
    _run_single_backtest,
    run_full_backtest,
)

__all__ = [
    "EngineConfig",
    "run_full_backtest",
    "_run_single_backtest",
    "_MIN_SLIPPAGE_FLOOR",
]
