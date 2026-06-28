from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ConfigParser import Config
from Backtesting.domain.models import CostModel


@dataclass
class EngineConfig:
    """回测引擎配置 - 纯数据容器，无业务逻辑"""

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage: float = 0.001
    max_position_pct: float = 0.1
    portfolio_method: str = "score_weighted"
    point_in_time: bool = True
    atr_stop_mult: float = 1.5
    kelly_fraction: float = 0.25
    position_a: float = 0.3
    liq_veto_ratio: float = 0.05
    boll_narrow_ratio: float = 0.8
    cross_decay_days: int = 30
    cost_model: Any = None  # CostModel | None — forward ref to avoid circular import

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
        )


class BacktestEngine:
    """纯计算核心：给定数据和配置，运行回测返回权益曲线和交易记录。

    无状态、可序列化、易于测试。
    """

    def __init__(self, config: EngineConfig):
        self.config = config

    def run(
        self,
        data: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        from Backtesting._engine_legacy import _run_single_backtest

        if params is None:
            params = {}
        tl: list[dict] = []
        ec: list[dict] = []
        _run_single_backtest(data, params, self.config, tl, ec)
        return tl, ec

    def run_from_config(
        self,
        data: pd.DataFrame,
        config: "Config",
        params: dict[str, Any] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        engine_cfg = EngineConfig.from_config(config)
        return self.run(data, params)



