from dataclasses import dataclass


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
    stamp_tax_rate: float = 0.0005
    market_slippage: float = 0.001
    limit_slippage: float = 0.0005
    impact_threshold: float = 0.01
    impact_base: float = 0.002
    impact_cap: float = 0.05
    short_cost_rate: float = 0.0
    min_commission_per_trade: float = 5.0
    transfer_fee_rate: float = 0.00001

    def calc_slippage(self, volume: float, adv: float, side: str = "buy", order_type: str = "market") -> float:
        """计算总滑点 = 基础滑点 + 大单冲击成本。

        参与率上限 1.0（单日成交量 100%），冲击成本上限 impact_cap（5%），
        防止极端流动性场景下滑点 > 100% 导致现金为负。
        """
        base = self.market_slippage if order_type == "market" else self.limit_slippage
        participation = volume / adv if adv > 0 else 0.0
        participation = min(max(participation, 0.0), 1.0)
        impact: float
        if participation > self.impact_threshold:
            impact = float(self.impact_base * (participation / self.impact_threshold) ** 1.5)
        else:
            impact = 0.0
        impact = min(impact, self.impact_cap)
        return base + impact

    def buy_cost(self, value: float, volume: float, adv: float, order_type: str = "market") -> float:
        """买入成本 = 佣金（含最低5元） + 过户费 + 滑点。"""
        slip = self.calc_slippage(volume, adv, side="buy", order_type=order_type)
        commission = max(value * self.commission_rate, self.min_commission_per_trade)
        return value * (slip + self.transfer_fee_rate) + commission

    def sell_cost(self, value: float, volume: float, adv: float, order_type: str = "market",
                  stamp_tax_rate: float | None = None) -> float:
        """卖出成本 = 佣金（含最低5元） + 印花税 + 过户费 + 滑点。"""
        slip = self.calc_slippage(volume, adv, side="sell", order_type=order_type)
        commission = max(value * self.commission_rate, self.min_commission_per_trade)
        stamp = self.stamp_tax_rate if stamp_tax_rate is None else stamp_tax_rate
        return value * (slip + stamp + self.transfer_fee_rate) + commission

