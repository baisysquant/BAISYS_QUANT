from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal


@dataclass
class CostModel:
    """分层交易成本模型。

    Attributes:
        commission_rate: 佣金费率
        stamp_tax_rate: 印花税率（仅卖出）
        market_slippage: 市价单滑点
        limit_slippage: 限价单滑点
        impact_threshold: 大单冲击阈值（占 ADV 比例），超过后启用非线性冲击成本
        impact_base: 阈值处的冲击成本基数
        short_cost_rate: 融券做空年化费率（预留，当前未使用）
    """

    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    market_slippage: float = 0.001
    limit_slippage: float = 0.0005
    impact_threshold: float = 0.01
    impact_base: float = 0.002
    short_cost_rate: float = 0.0

    def calc_slippage(self, volume: float, adv: float, side: str = "buy", order_type: str = "market") -> float:
        """计算总滑点 = 基础滑点 + 大单冲击成本。"""
        base = self.market_slippage if order_type == "market" else self.limit_slippage
        participation = volume / adv if adv > 0 else 0.0
        impact: float
        if participation > self.impact_threshold:
            impact = float(self.impact_base * (participation / self.impact_threshold) ** 1.5)
        else:
            impact = 0.0
        return base + impact

    def total_cost(self, value: float, volume: float, adv: float, side: str = "buy", order_type: str = "market") -> float:
        """计算一笔交易的总成本（佣金 + 印花税 + 滑点）。"""
        slip = self.calc_slippage(volume, adv, side, order_type)
        stamp = self.stamp_tax_rate if side == "sell" else 0.0
        return value * (self.commission_rate + stamp + slip)

    def sell_proceeds(self, value: float, volume: float, adv: float, order_type: str = "market") -> float:
        """卖出净收入（扣除所有成本后）。"""
        slip = self.calc_slippage(volume, adv, side="sell", order_type=order_type)
        return value * (1 - slip - self.stamp_tax_rate)

    def buy_cost(self, value: float, volume: float, adv: float, order_type: str = "market") -> float:
        """买入所需额外成本（佣金 + 滑点）。"""
        slip = self.calc_slippage(volume, adv, side="buy", order_type=order_type)
        return value * (self.commission_rate + slip)


@dataclass
class Order:
    symbol: str
    action: Literal["buy", "sell"]
    target_weight: float = 0.0


@dataclass
class TradeRecord:
    time: str
    symbol: str
    action: str
    price: float
    shares: float
    value: float
    cost: float


@dataclass
class BarRow:
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    features: dict[str, Any] = field(default_factory=dict)
