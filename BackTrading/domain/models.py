from __future__ import annotations

import dataclasses
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
# 印花税日期分段表（配置驱动，替代硬编码）：date:rate 升序，取最晚 ≤ 交易日的档。
# 2023-08-28 财政部减半：0.1% → 0.05%（卖出单向）。默认含历史兜底段 2000-01-01。
DEFAULT_STAMP_TAX_SEGMENTS: tuple[tuple[str, float], ...] = (
    ("2023-08-28", 0.0005),
    ("2000-01-01", 0.001),
)
# 过户费日期分段表（双边收取，date:rate 升序，取最晚 ≤ 交易日的档）。
# 2022-04-28 前 0.02‰（万分之零点零二），2022-04-29 起减半为 0.01‰。
DEFAULT_TRANSFER_FEE_SEGMENTS: tuple[tuple[str, float], ...] = (
    ("2022-04-29", 0.00001),
    ("2000-01-01", 0.00002),
)
# 经手费（双边，沪/深交易所）+ 证管费（双边，证监会），
# 行业惯例通常已并入佣金，此处单独建模（拆分报告可见），并入 CostModel 统一收取。
DEFAULT_HANDLING_FEE_RATE: float = 0.0000341
DEFAULT_CSRC_FEE_RATE: float = 0.00002
# 经手费日期分段表（双边，date:rate 升序，取最晚 ≤ 交易日的档）。
# 2023-08-28 起 0.00341%（万分之零点三四一），此前 0.00487%。
DEFAULT_HANDLING_FEE_SEGMENTS: tuple[tuple[str, float], ...] = (
    ("2023-08-28", 0.0000341),
    ("2000-01-01", 0.0000487),
)
# 证管费日期分段表（双边，date:rate 升序，取最晚 ≤ 交易日的档）。
# 2015-08-01 起 0.002%（万分之零点二），此前 0.004%。
DEFAULT_CSRC_FEE_SEGMENTS: tuple[tuple[str, float], ...] = (
    ("2015-08-01", 0.00002),
    ("2000-01-01", 0.00004),
)


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
    handling_fee_rate: float = DEFAULT_HANDLING_FEE_RATE
    csrc_fee_rate: float = DEFAULT_CSRC_FEE_RATE
    # ── 佣金是否已含经手费+证管费（#1 审计修复：行业惯例佣金为全包价）──
    # True（默认）：佣金已含经手费(3.41bp)+证管费(2bp)，不再单独收取
    # False：佣金为净佣金，需额外叠加经手费+证管费
    commission_includes_fees: bool = True
    stamp_tax_segments: tuple[tuple[str, float], ...] = DEFAULT_STAMP_TAX_SEGMENTS
    transfer_fee_segments: tuple[tuple[str, float], ...] = DEFAULT_TRANSFER_FEE_SEGMENTS
    handling_fee_segments: tuple[tuple[str, float], ...] = DEFAULT_HANDLING_FEE_SEGMENTS
    csrc_fee_segments: tuple[tuple[str, float], ...] = DEFAULT_CSRC_FEE_SEGMENTS
    liquidity_tier_edges: tuple[float, ...] = DEFAULT_TIER_EDGES
    liquidity_tier_impact_base: tuple[float, ...] = DEFAULT_TIER_IMPACT_BASE
    liquidity_tier_threshold: tuple[float, ...] = DEFAULT_TIER_THRESHOLD
    liquidity_tier_cap: tuple[float, ...] = DEFAULT_TIER_CAP
    # ── 分档基础滑点下限（修复"小盘股静态滑点偏低"问题）──
    # 大盘股0.05%，中盘0.08%，小盘0.12%，微盘0.18%；基础滑点=max(配置值, 档下限)
    liquidity_tier_slippage_floor: tuple[float, ...] = (0.0018, 0.0012, 0.0008, 0.0005)

    def __post_init__(self) -> None:
        """构造时校验分档配置，避免档位错位导致成本失真。"""
        self.validate_liquidity_tiers()
        # 印花税分段表按日期升序，保证 stamp_tax_rate_for 取"最晚 ≤ 交易日"档
        self.stamp_tax_segments = tuple(
            sorted(self.stamp_tax_segments, key=lambda x: str(x[0]))
        )
        # 过户费分段表同理，保证 transfer_fee_rate_for 取"最晚 ≤ 交易日"档
        self.transfer_fee_segments = tuple(
            sorted(self.transfer_fee_segments, key=lambda x: str(x[0]))
        )
        # 经手费/证管费分段表同理（2023-08-28 / 2015-08-01 前后费率不同）
        self.handling_fee_segments = tuple(
            sorted(self.handling_fee_segments, key=lambda x: str(x[0]))
        )
        self.csrc_fee_segments = tuple(
            sorted(self.csrc_fee_segments, key=lambda x: str(x[0]))
        )
        # P0-10 审计修复：与印花税一致，经手费/证管费单值回退必须与分段表兜底段一致，
        # 否则 commission_includes_fees=False 时历史成本被低估 ~30%（0.00487%→0.00341%）
        for _name, _rate, _segs in (
            ("handling_fee_rate", self.handling_fee_rate, self.handling_fee_segments),
            ("csrc_fee_rate", self.csrc_fee_rate, self.csrc_fee_segments),
        ):
            if _segs:
                _fallback_date, _fallback_rate = _segs[0]  # 最早日期 = 兜底
                if abs(_rate - _fallback_rate) > 1e-9:
                    import warnings
                    warnings.warn(
                        f"CostModel.{_name}={_rate} 与分段表兜底段 "
                        f"({_fallback_date}: {_fallback_rate}) 不一致！"
                        f"回退路径（无命中时）将使用 {_rate}，"
                        f"可能导致历史经手费/证管费被低估。建议将 {_name} 设置为 {_fallback_rate}。"
                    )
        # P1 审计修复：stamp_tax_rate 回退值必须与分段表兜底段（最早日期档）一致
        # 否则 2023-08-28 前交易按错误税率收取，历史成本被低估
        if self.stamp_tax_segments:
            _fallback_date, _fallback_rate = self.stamp_tax_segments[0]  # 最早日期 = 兜底
            if abs(self.stamp_tax_rate - _fallback_rate) > 1e-9:
                import warnings
                warnings.warn(
                    f"CostModel.stamp_tax_rate={self.stamp_tax_rate} 与分段表兜底段 "
                    f"({_fallback_date}: {_fallback_rate}) 不一致！"
                    f"回退路径（无命中时）将使用 stamp_tax_rate={self.stamp_tax_rate}，"
                    f"可能导致 2023-08-28 前印花税被低估。建议将 stamp_tax_rate 设置为 {_fallback_rate}。"
                )

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
            ("liquidity_tier_slippage_floor", self.liquidity_tier_slippage_floor),
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

    def stamp_tax_rate_for(self, dt: str | None) -> float:
        """按配置日期表取印花税率（替代硬编码分段）。

        Args:
            dt: 交易日（YYYY-MM-DD）；None 返回单值 stamp_tax_rate。

        Returns:
            float: 最晚 ≤ dt 的档位税率；无命中时回落 stamp_tax_rate。
        """
        if dt is None:
            return self.stamp_tax_rate
        rate = self.stamp_tax_rate
        for _date, _r in self.stamp_tax_segments:
            if str(dt) >= str(_date):
                rate = _r
            else:
                break  # 升序：后续日期更晚，不可能命中
        return rate

    def transfer_fee_rate_for(self, dt: str | None) -> float:
        """按配置日期表取过户费率（双边收取，替代固定费率）。

        2022-04-28 前为 0.02‰（万分之零点零二），2022-04-29 起减半为 0.01‰。

        Args:
            dt: 交易日（YYYY-MM-DD）；None 返回单值 transfer_fee_rate。

        Returns:
            float: 最晚 ≤ dt 的档位费率；无命中时回落 transfer_fee_rate。
        """
        if dt is None:
            return self.transfer_fee_rate
        rate = self.transfer_fee_rate
        for _date, _r in self.transfer_fee_segments:
            if str(dt) >= str(_date):
                rate = _r
            else:
                break
        return rate

    def handling_fee_rate_for(self, dt: str | None) -> float:
        """按配置日期表取经手费率（双边收取，替代固定费率）。

        2023-08-28 起 0.00341%，此前 0.00487%（沪/深交易所）。
        """
        if dt is None:
            return self.handling_fee_rate
        rate = self.handling_fee_rate
        for _date, _r in self.handling_fee_segments:
            if str(dt) >= str(_date):
                rate = _r
            else:
                break
        return rate

    def csrc_fee_rate_for(self, dt: str | None) -> float:
        """按配置日期表取证管费率（双边收取，替代固定费率）。

        2015-08-01 起 0.002%，此前 0.004%（证监会）。
        """
        if dt is None:
            return self.csrc_fee_rate
        rate = self.csrc_fee_rate
        for _date, _r in self.csrc_fee_segments:
            if str(dt) >= str(_date):
                rate = _r
            else:
                break
        return rate

    def _slippage_components(
        self,
        volume: float,
        adv: float,
        order_type: str = "market",
        amount_ma20: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        """返回 (基础滑点率, 冲击成本率)。

        波动倍率（1.9 流动性拟真）：单日振幅>5% 的剧烈波动日，盘中价格跳动加剧，
        基础滑点必须翻倍（×2）。冲击参数按流动性分档独立取值（小票冲击大、阈值低，
        大票反之）。参与率上限 1.0，冲击上限为该档 impact_cap，防止极端场景滑点>100%。
        """
        base = self.market_slippage if order_type == "market" else self.limit_slippage
        # 分档基础滑点下限：小票天然流动性不足，滑点下限随档位抬升
        tier = self.liquidity_tier(amount_ma20)
        if tier >= 0 and len(self.liquidity_tier_slippage_floor) > tier:
            _tier_floor = self.liquidity_tier_slippage_floor[tier]
            base = max(base, _tier_floor, MIN_SLIPPAGE_FLOOR)
        else:
            base = max(base, MIN_SLIPPAGE_FLOOR)
        base *= max(volatility_multiplier, 1.0)
        # 参与率 = 委托量 / ADV（股数口径，测单日成交占比；AMOUNT_MA20 仅用于分档参数选择）
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
        return base, impact

    def calc_slippage(
        self,
        volume: float,
        adv: float,
        side: str = "buy",
        order_type: str = "market",
        amount_ma20: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """计算总滑点 = 基础滑点×波动倍率 + 大单冲击成本。"""
        base, impact = self._slippage_components(
            volume, adv, order_type=order_type, amount_ma20=amount_ma20,
            volatility_multiplier=volatility_multiplier,
        )
        return base + impact

    def buy_cost_breakdown(
        self,
        value: float,
        volume: float,
        adv: float,
        order_type: str = "market",
        amount_ma20: float | None = None,
        dt: str | None = None,
        volatility_multiplier: float = 1.0,
    ) -> dict[str, float]:
        """买入成本拆解 = 佣金(含最低5元) + 过户费 + 经手费 + 证管费 + 滑点 + 冲击。

        过户费按 dt 查 transfer_fee_segments 日期分段表（2022-04-29 前后费率不同）。
        返回各分项金额（元）与 total，供成本拆解报告按占比汇总。

        #1 审计修复：当 commission_includes_fees=True（默认）时，经手费/证管费已含在佣金中，
        不再单独收取；仅记录拆分值供报告展示（金额为 0）。
        """
        base, impact = self._slippage_components(
            volume, adv, order_type=order_type, amount_ma20=amount_ma20,
            volatility_multiplier=volatility_multiplier,
        )
        commission = max(value * self.commission_rate, self.min_commission_per_trade)
        transfer = value * self.transfer_fee_rate_for(dt)
        # #1 修复：佣金为全包价时，经手费/证管费不单独收取
        # P0-10：经手费/证管费按日期分段表取历史费率（2023-08-28 / 2015-08-01 分界）
        handling = 0.0 if self.commission_includes_fees else value * self.handling_fee_rate_for(dt)
        csrc = 0.0 if self.commission_includes_fees else value * self.csrc_fee_rate_for(dt)
        slippage = value * base
        impact_v = value * impact
        return {
            "commission": commission,
            "transfer": transfer,
            "handling": handling,
            "csrc": csrc,
            "slippage": slippage,
            "impact": impact_v,
            "total": commission + transfer + handling + csrc + slippage + impact_v,
        }

    def sell_cost_breakdown(
        self,
        value: float,
        volume: float,
        adv: float,
        order_type: str = "market",
        stamp_tax_rate: float | None = None,
        dt: str | None = None,
        amount_ma20: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> dict[str, float]:
        """卖出成本拆解 = 买入成本项 + 印花税（按日期表，仅卖出）。

        stamp_tax_rate 显式传入优先；否则按 dt 查 stamp_tax_segments 日期表。
        过户费同样按 dt 查 transfer_fee_segments 日期分段表（2022-04-29 前后费率不同）。
        """
        parts = self.buy_cost_breakdown(
            value, volume, adv, order_type=order_type, amount_ma20=amount_ma20,
            dt=dt,
            volatility_multiplier=volatility_multiplier,
        )
        if stamp_tax_rate is None:
            stamp_tax_rate = self.stamp_tax_rate_for(dt)
        stamp = value * stamp_tax_rate
        parts["stamp"] = stamp
        parts["total"] += stamp
        return parts

    def buy_cost(
        self,
        value: float,
        volume: float,
        adv: float,
        order_type: str = "market",
        amount_ma20: float | None = None,
        dt: str | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """买入成本 = 佣金（含最低5元） + 过户费 + 经手费 + 证管费 + 滑点 + 冲击。"""
        return self.buy_cost_breakdown(
            value, volume, adv, order_type=order_type, amount_ma20=amount_ma20,
            dt=dt,
            volatility_multiplier=volatility_multiplier,
        )["total"]

    def sell_cost(
        self,
        value: float,
        volume: float,
        adv: float,
        order_type: str = "market",
        stamp_tax_rate: float | None = None,
        dt: str | None = None,
        amount_ma20: float | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """卖出成本 = 佣金（含最低5元） + 印花税 + 过户费 + 经手费 + 证管费 + 滑点 + 冲击。"""
        return self.sell_cost_breakdown(
            value, volume, adv, order_type=order_type,
            stamp_tax_rate=stamp_tax_rate, dt=dt,
            amount_ma20=amount_ma20,
            volatility_multiplier=volatility_multiplier,
        )["total"]

    # ── 配置构建 ──────────────────────────────────────────────

    @classmethod
    def _parse_stamp_segments(cls, s: str) -> tuple[tuple[str, float], ...]:
        """解析 "date:rate;date:rate" 印花税日期表，按日期升序。

        #2 审计修复：强制注入兜底段（date <= 2005-01-01），确保自定义配置
        不丢失早期历史数据的税率。若用户未提供兜底段，从 DEFAULT_STAMP_TAX_SEGMENTS
        取最早段注入。
        """
        segs: list[tuple[str, float]] = []
        for part in str(s).split(";"):
            part = part.strip()
            if not part:
                continue
            _date, _, _rate = part.partition(":")
            if not _date or not _rate:
                raise ValueError(f"STAMP_TAX_SEGMENTS 段格式应为 date:rate，收到 {part!r}")
            segs.append((_date.strip(), float(_rate.strip())))
        if not segs:
            return DEFAULT_STAMP_TAX_SEGMENTS
        segs = sorted(segs, key=lambda x: x[0])
        # #2 修复：若无兜底段（最早日期 > 2005），注入 DEFAULT 兜底
        if segs[0][0] > "2005-01-01":
            _fallback = DEFAULT_STAMP_TAX_SEGMENTS[0] if DEFAULT_STAMP_TAX_SEGMENTS else ("2000-01-01", 0.001)
            segs.insert(0, _fallback)
        return tuple(segs)

    @classmethod
    def _parse_transfer_segments(cls, s: str) -> tuple[tuple[str, float], ...]:
        """解析 "date:rate;date:rate" 过户费日期表，按日期升序。

        #2 审计修复：同上，强制注入兜底段。
        """
        segs: list[tuple[str, float]] = []
        for part in str(s).split(";"):
            part = part.strip()
            if not part:
                continue
            _date, _, _rate = part.partition(":")
            if not _date or not _rate:
                raise ValueError(f"TRANSFER_FEE_SEGMENTS 段格式应为 date:rate，收到 {part!r}")
            segs.append((_date.strip(), float(_rate.strip())))
        if not segs:
            return DEFAULT_TRANSFER_FEE_SEGMENTS
        segs = sorted(segs, key=lambda x: x[0])
        # #2 修复：若无兜底段（最早日期 > 2005），注入 DEFAULT 兜底
        if segs[0][0] > "2005-01-01":
            _fallback = DEFAULT_TRANSFER_FEE_SEGMENTS[0] if DEFAULT_TRANSFER_FEE_SEGMENTS else ("2000-01-01", 0.00002)
            segs.insert(0, _fallback)
        return tuple(segs)

    @classmethod
    def _parse_handling_segments(cls, s: str) -> tuple[tuple[str, float], ...]:
        """解析 "date:rate;date:rate" 经手费日期表，按日期升序（P0-10）。

        2023-08-28 起 0.00341%，此前 0.00487%；强制注入兜底段。
        """
        segs: list[tuple[str, float]] = []
        for part in str(s).split(";"):
            part = part.strip()
            if not part:
                continue
            _date, _, _rate = part.partition(":")
            if not _date or not _rate:
                raise ValueError(f"HANDLING_FEE_SEGMENTS 段格式应为 date:rate，收到 {part!r}")
            segs.append((_date.strip(), float(_rate.strip())))
        if not segs:
            return DEFAULT_HANDLING_FEE_SEGMENTS
        segs = sorted(segs, key=lambda x: x[0])
        if segs[0][0] > "2005-01-01":
            _fallback = DEFAULT_HANDLING_FEE_SEGMENTS[-1] if DEFAULT_HANDLING_FEE_SEGMENTS else ("2000-01-01", 0.0000487)
            segs.insert(0, _fallback)
        return tuple(segs)

    @classmethod
    def _parse_csrc_segments(cls, s: str) -> tuple[tuple[str, float], ...]:
        """解析 "date:rate;date:rate" 证管费日期表，按日期升序（P0-10）。

        2015-08-01 起 0.002%，此前 0.004%；强制注入兜底段。
        """
        segs: list[tuple[str, float]] = []
        for part in str(s).split(";"):
            part = part.strip()
            if not part:
                continue
            _date, _, _rate = part.partition(":")
            if not _date or not _rate:
                raise ValueError(f"CSRC_FEE_SEGMENTS 段格式应为 date:rate，收到 {part!r}")
            segs.append((_date.strip(), float(_rate.strip())))
        if not segs:
            return DEFAULT_CSRC_FEE_SEGMENTS
        segs = sorted(segs, key=lambda x: x[0])
        if segs[0][0] > "2005-01-01":
            _fallback = DEFAULT_CSRC_FEE_SEGMENTS[-1] if DEFAULT_CSRC_FEE_SEGMENTS else ("2000-01-01", 0.00004)
            segs.insert(0, _fallback)
        return tuple(segs)

    @classmethod
    def from_backtest_config(cls, bt: Any, trading_cost: Any | None = None) -> CostModel:
        """从 BacktestConfig 构建成本模型（含流动性分档冲击参数）。

        Args:
            bt: UtilsManager.ConfigParser.BacktestConfig 实例。
            trading_cost: 可选 UtilsManager.ConfigParser.TradingCostConfig 实例；
                提供时以其费率覆盖 [BACKTEST] 的佣金/印花税/过户费/经手费/证管费/分段表，
                实现 [TRADING_COST] 节与回测引擎统一由 CostModel 单一来源驱动。

        Returns:
            CostModel: 配置校验失败时按默认值回落并告警，不中断回测。
        """
        from loguru import logger

        def _parse_csv(s: str) -> tuple[float, ...]:
            return tuple(float(x.strip()) for x in str(s).split(",") if x.strip())

        def _seg(s: str) -> tuple[tuple[str, float], ...]:
            try:
                return cls._parse_stamp_segments(s)
            except ValueError:
                return DEFAULT_STAMP_TAX_SEGMENTS

        def _tseg(s: str) -> tuple[tuple[str, float], ...]:
            try:
                return cls._parse_transfer_segments(s)
            except ValueError:
                return DEFAULT_TRANSFER_FEE_SEGMENTS

        def _hseg(s: str) -> tuple[tuple[str, float], ...]:
            try:
                return cls._parse_handling_segments(s)
            except ValueError:
                return DEFAULT_HANDLING_FEE_SEGMENTS

        def _cseg(s: str) -> tuple[tuple[str, float], ...]:
            try:
                return cls._parse_csrc_segments(s)
            except ValueError:
                return DEFAULT_CSRC_FEE_SEGMENTS

        base_kw = dict(
            commission_rate=float(bt.COMMISSION_RATE),
            stamp_tax_rate=float(bt.STAMP_TAX_RATE),
            market_slippage=float(bt.SLIPPAGE),
            limit_slippage=float(bt.SLIPPAGE) * 0.5,
            min_commission_per_trade=float(bt.MIN_COMMISSION_PER_TRADE),
            transfer_fee_rate=float(bt.TRANSFER_FEE_RATE),
            handling_fee_rate=float(getattr(bt, "HANDLING_FEE_RATE", DEFAULT_HANDLING_FEE_RATE)),
            csrc_fee_rate=float(getattr(bt, "CSRC_FEE_RATE", DEFAULT_CSRC_FEE_RATE)),
            stamp_tax_segments=_seg(getattr(bt, "STAMP_TAX_SEGMENTS", "")),
            transfer_fee_segments=_tseg(getattr(bt, "TRANSFER_FEE_SEGMENTS", "")),
            handling_fee_segments=_hseg(getattr(bt, "HANDLING_FEE_SEGMENTS", "")),
            csrc_fee_segments=_cseg(getattr(bt, "CSRC_FEE_SEGMENTS", "")),
            impact_base=getattr(bt, "IMPACT_BASE", 0.002),
            impact_threshold=getattr(bt, "IMPACT_THRESHOLD", 0.01),
            impact_cap=getattr(bt, "IMPACT_CAP", 0.05),
        )
        try:
            model = cls(
                **base_kw,
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
            model = cls(**base_kw)

        if trading_cost is not None:
            # [TRADING_COST] 节为统一成本来源：提供时覆盖 [BACKTEST] 对应费率
            model = dataclasses.replace(
                model,
                commission_rate=float(getattr(trading_cost, "COMMISSION_RATE", model.commission_rate)),
                stamp_tax_rate=float(getattr(trading_cost, "STAMP_TAX_RATE", model.stamp_tax_rate)),
                transfer_fee_rate=float(getattr(trading_cost, "TRANSFER_FEE_RATE", model.transfer_fee_rate)),
                handling_fee_rate=float(getattr(trading_cost, "HANDLING_FEE_RATE", model.handling_fee_rate)),
                csrc_fee_rate=float(getattr(trading_cost, "CSRC_FEE_RATE", model.csrc_fee_rate)),
                stamp_tax_segments=_seg(getattr(trading_cost, "STAMP_TAX_SEGMENTS", "")),
                transfer_fee_segments=_tseg(getattr(trading_cost, "TRANSFER_FEE_SEGMENTS", "")),
                handling_fee_segments=_hseg(getattr(trading_cost, "HANDLING_FEE_SEGMENTS", "")),
                csrc_fee_segments=_cseg(getattr(trading_cost, "CSRC_FEE_SEGMENTS", "")),
            )
        return model