from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# 流动性分档默认边界（AMOUNT_MA20，单位：元）。
# 档数 = 边界数 + 1：微盘(<500万) / 小盘(500万-2000万) / 中盘(2000万-1亿) / 大盘(>1亿)。
DEFAULT_TIER_EDGES: tuple[float, ...] = (5_000_000.0, 20_000_000.0, 100_000_000.0)
# 各档独立冲击参数：小票冲击大、阈值低、上限高；大票反之，避免小票收益虚高。
DEFAULT_TIER_IMPACT_BASE: tuple[float, ...] = (0.008, 0.003, 0.0015, 0.001)
DEFAULT_TIER_THRESHOLD: tuple[float, ...] = (0.005, 0.01, 0.01, 0.02)
DEFAULT_TIER_CAP: tuple[float, ...] = (0.10, 0.05, 0.05, 0.03)
# 固定滑点强制下限（1.8 交易摩擦合规）：A股隐性成本不低于单边 0.05%，
# 配置值低于此下限时一律抬升，防止策略 Alpha 未被真实市场摩擦覆盖。
MIN_SLIPPAGE_FLOOR: float = 0.0005


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
        impact_cap: 冲击成本上限
        short_cost_rate: 融券做空年化费率（预留，当前未使用）
        liquidity_tier_edges: 流动性分档边界（AMOUNT_MA20，元），档数 = len+1
        liquidity_tier_impact_base: 各档冲击基数（小票高、大票低）
        liquidity_tier_threshold: 各档冲击启用阈值（占 ADV 比例）
        liquidity_tier_cap: 各档冲击成本上限
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
    liquidity_tier_edges: tuple[float, ...] = DEFAULT_TIER_EDGES
    liquidity_tier_impact_base: tuple[float, ...] = DEFAULT_TIER_IMPACT_BASE
    liquidity_tier_threshold: tuple[float, ...] = DEFAULT_TIER_THRESHOLD
    liquidity_tier_cap: tuple[float, ...] = DEFAULT_TIER_CAP

    def __post_init__(self) -> None:
        """构造时校验分档配置，避免档位错位导致成本失真。"""
        self.validate_liquidity_tiers()

    # ── 流动性分档 ────────────────────────────────────────────

    @property
    def n_liquidity_tiers(self) -> int:
        """分档数（档数 = 边界数 + 1）。"""
        return len(self.liquidity_tier_edges) + 1

    def validate_liquidity_tiers(self) -> None:
        """校验分档配置一致性：长度匹配、边界递增、参数非负。"""
        edges = self.liquidity_tier_edges
        n = len(edges) + 1
        for name, seq in (
            ("liquidity_tier_impact_base", self.liquidity_tier_impact_base),
            ("liquidity_tier_threshold", self.liquidity_tier_threshold),
            ("liquidity_tier_cap", self.liquidity_tier_cap),
        ):
            if len(seq) != n:
                raise ValueError(
                    f"{name} 长度 {len(seq)} 必须等于分档数 {n}"
                    f"（liquidity_tier_edges 长度 {len(edges)} + 1）"
                )
            if any(v < 0 for v in seq):
                raise ValueError(f"{name} 不允许负值: {seq}")
        for a, b in zip(edges, edges[1:]):
            if b <= a:
                raise ValueError(f"liquidity_tier_edges 必须严格递增: {edges}")

    def liquidity_tier(self, amount_ma20: float | None) -> int:
        """按 AMOUNT_MA20 将标的归入流动性档位（0 = 最不流动）。

        Args:
            amount_ma20: 20 日均成交额（元）；None/NaN/<=0 视为无数据。

        Returns:
            档位索引（0 起），无数据时返回 -1（回落统一参数模式）。
        """
        if amount_ma20 is None:
            return -1
        try:
            v = float(amount_ma20)
        except (TypeError, ValueError):
            return -1
        if not math.isfinite(v) or v <= 0:
            return -1
        for tier, edge in enumerate(self.liquidity_tier_edges):
            if v < edge:
                return tier
        return len(self.liquidity_tier_edges)

    def _impact_params(
        self, amount_ma20: float | None
    ) -> tuple[float, float, float]:
        """取 (impact_base, impact_threshold, impact_cap)。

        提供 AMOUNT_MA20 时按分档取独立参数，否则回落统一参数（向后兼容）。
        """
        tier = self.liquidity_tier(amount_ma20)
        if tier < 0:
            return self.impact_base, self.impact_threshold, self.impact_cap
        return (
            self.liquidity_tier_impact_base[tier],
            self.liquidity_tier_threshold[tier],
            self.liquidity_tier_cap[tier],
        )

    # ── 成本计算 ──────────────────────────────────────────────

    def calc_slippage(
        self,
        volume: float,
        adv: float,
        side: str = "buy",
        order_type: str = "market",
        amount_ma20: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """计算总滑点 = 基础滑点×波动倍率 + 大单冲击成本。

        波动倍率（1.9 流动性拟真）：单日振幅>5% 的剧烈波动日，盘中价格跳动加剧，
        基础滑点必须翻倍（×2）。振幅可观测性由调用方（引擎）计算，无高/低价时默认 1.0。

        冲击参数按流动性分档独立取值：小票（低 AMOUNT_MA20）冲击大、阈值低，
        大票反之，纠正统一冲击参数下小票收益虚高的问题。

        参与率上限 1.0（单日成交量 100%），冲击成本上限为该档 impact_cap，
        防止极端流动性场景下滑点 > 100% 导致现金为负。
        """
        base = self.market_slippage if order_type == "market" else self.limit_slippage
        base = max(base, MIN_SLIPPAGE_FLOOR)
        base *= max(volatility_multiplier, 1.0)
        participation = volume / adv if adv > 0 else 0.0
        participation = min(max(participation, 0.0), 1.0)
        impact_base, impact_threshold, impact_cap = self._impact_params(amount_ma20)
        impact: float
        if participation > impact_threshold:
            impact = float(
                impact_base * (participation / impact_threshold) ** 1.5
            )
        else:
            impact = 0.0
        impact = min(impact, impact_cap)
        return base + impact

    def buy_cost(
        self,
        value: float,
        volume: float,
        adv: float,
        order_type: str = "market",
        amount_ma20: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """买入成本 = 佣金（含最低5元） + 过户费 + 滑点。"""
        slip = self.calc_slippage(
            volume, adv, side="buy", order_type=order_type, amount_ma20=amount_ma20,
            volatility_multiplier=volatility_multiplier,
        )
        commission = max(value * self.commission_rate, self.min_commission_per_trade)
        return value * (slip + self.transfer_fee_rate) + commission

    def sell_cost(
        self,
        value: float,
        volume: float,
        adv: float,
        order_type: str = "market",
        stamp_tax_rate: float | None = None,
        amount_ma20: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """卖出成本 = 佣金（含最低5元） + 印花税 + 过户费 + 滑点。"""
        slip = self.calc_slippage(
            volume, adv, side="sell", order_type=order_type, amount_ma20=amount_ma20,
            volatility_multiplier=volatility_multiplier,
        )
        commission = max(value * self.commission_rate, self.min_commission_per_trade)
        stamp = self.stamp_tax_rate if stamp_tax_rate is None else stamp_tax_rate
        return value * (slip + stamp + self.transfer_fee_rate) + commission

    # ── 配置构建 ──────────────────────────────────────────────

    @classmethod
    def from_backtest_config(cls, bt: Any) -> CostModel:
        """从 BacktestConfig 构建成本模型（含流动性分档冲击参数）。

        Args:
            bt: UtilsManager.ConfigParser.BacktestConfig 实例。

        Returns:
            CostModel: 配置校验失败时按默认值回落并告警，不中断回测。
        """
        from loguru import logger

        def _parse_csv(s: str) -> tuple[float, ...]:
            return tuple(float(x.strip()) for x in str(s).split(",") if x.strip())

        try:
            return cls(
                commission_rate=float(bt.COMMISSION_RATE),
                stamp_tax_rate=float(bt.STAMP_TAX_RATE),
                market_slippage=float(bt.SLIPPAGE),
                limit_slippage=float(bt.SLIPPAGE) * 0.5,
                min_commission_per_trade=float(bt.MIN_COMMISSION_PER_TRADE),
                transfer_fee_rate=float(bt.TRANSFER_FEE_RATE),
                impact_base=getattr(bt, "IMPACT_BASE", 0.002),
                impact_threshold=getattr(bt, "IMPACT_THRESHOLD", 0.01),
                impact_cap=getattr(bt, "IMPACT_CAP", 0.05),
                liquidity_tier_edges=_parse_csv(
                    getattr(bt, "LIQUIDITY_TIER_EDGES", "5e6,2e7,1e8")
                ),
                liquidity_tier_impact_base=_parse_csv(
                    getattr(bt, "LIQUIDITY_TIER_IMPACT_BASE", "0.008,0.003,0.0015,0.001")
                ),
                liquidity_tier_threshold=_parse_csv(
                    getattr(bt, "LIQUIDITY_TIER_THRESHOLD", "0.005,0.01,0.01,0.02")
                ),
                liquidity_tier_cap=_parse_csv(
                    getattr(bt, "LIQUIDITY_TIER_CAP", "0.10,0.05,0.05,0.03")
                ),
            )
        except ValueError as e:
            logger.warning(f"[CostModel] 流动性分档配置无效，回落默认分档: {e}")
            return cls(
                commission_rate=float(bt.COMMISSION_RATE),
                stamp_tax_rate=float(bt.STAMP_TAX_RATE),
                market_slippage=float(bt.SLIPPAGE),
                limit_slippage=float(bt.SLIPPAGE) * 0.5,
                min_commission_per_trade=float(bt.MIN_COMMISSION_PER_TRADE),
                transfer_fee_rate=float(bt.TRANSFER_FEE_RATE),
                impact_base=getattr(bt, "IMPACT_BASE", 0.002),
                impact_threshold=getattr(bt, "IMPACT_THRESHOLD", 0.01),
                impact_cap=getattr(bt, "IMPACT_CAP", 0.05),
            )