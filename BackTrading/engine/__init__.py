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
    liq_veto_ratio: float = 0.05
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

    @classmethod
    def from_config(cls, config: "Config") -> "EngineConfig":
        """从 Config 实例构建 EngineConfig，自动读取最新配置。"""
        bt = config.app_config.backtest
        scoring = config.app_config.scoring_params
        regime = config.app_config.regime_detection
        filter_rules = config.app_config.filter_rules
        position = config.app_config.position_sizing
        return cls(
            initial_cash=bt.INITIAL_CASH,
            commission_rate=bt.COMMISSION_RATE,
            stamp_tax_rate=bt.STAMP_TAX_RATE,
            slippage=bt.SLIPPAGE,
            max_position_pct=bt.MAX_POSITION_PCT,
            portfolio_method=bt.PORTFOLIO_METHOD,
            point_in_time=bt.POINT_IN_TIME,
            atr_stop_mult=scoring.ATR_STOP_MULT,
            kelly_fraction=position.KELLY_FRACTION,
            position_a=position.POSITION_A,
            liq_veto_ratio=filter_rules.LIQ_VETO_RATIO,
            boll_narrow_ratio=regime.BOLL_NARROW_RATIO,
            cross_decay_days=scoring.CROSS_DECAY_DAYS,
            risk_none_multiplier=position.RISK_NONE_MULTIPLIER,
        )



