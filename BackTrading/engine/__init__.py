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

    @property
    def buy_fee_rate(self) -> float:
        """买入费率（不含滑点）：佣金 + 过户费"""
        return self.commission_rate + self.transfer_fee_rate

    @property
    def sell_fee_rate(self) -> float:
        """卖出费率（不含滑点）：佣金 + 过户费 + 印花税"""
        return self.commission_rate + self.transfer_fee_rate + self.stamp_tax_rate
