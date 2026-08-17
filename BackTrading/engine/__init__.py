from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from UtilsManager.ConfigParser import Config
from BackTrading.domain.models import CostModel, DEFAULT_TRANSFER_FEE_SEGMENTS


@dataclass
class EngineConfig:
    """回测引擎配置 - 纯数据容器，无业务逻辑"""

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.001
    transfer_fee_rate: float = 0.00001  # 过户费 0.001% 双边
    # 过户费日期分段表（2022-04-29 前后费率不同，双边收取）；
    # fallback 构造 CostModel 时未显式传入则用 CostModel 默认值。
    transfer_fee_segments: tuple[tuple[str, float], ...] = DEFAULT_TRANSFER_FEE_SEGMENTS
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
    # next_open=信号次日开盘价成交（默认，符合A股T+1）
    # vwap=信号次日VWAP（成交额/成交量，后复权）成交。next_open/vwap 下单挂至次日开盘撮合，
    # 并与 simulate_limit_up_down 联动：次日一字涨停不可买入、一字跌停不可卖出。
    # close 模式已移除（固有前视偏差：信号依赖当日收盘数据计算，以同日收盘价成交=先知交易）
    execution_model: str = "next_open"  # next_open / vwap
    # ── 涨跌停撮合约束（simulate_limit_up_down=true 开启可成交量模型） ──
    simulate_limit_up_down: bool = True  # false=回退简化撮合（触板一律禁买/禁卖）
    limit_seal_ratio: float = 0.05  # 一字板（开=收=限价）可成交量比例
    limit_tradable_ratio: float = 0.30  # 开盘触板后炸板（open≥限价, close<限价）可成交量比例
    limit_intraday_ratio: float = 0.10  # 盘中冲板（open<限价, high≥限价）可成交量比例
    limit_seal_decay: float = 0.5  # 连续板每板可成交量衰减系数
    # ── P0-6 ⑥ 开盘集合竞价成交率分档（封单量/可成交量代理） ──
    # 开盘价触板日（open≥涨停价/≤跌停价、未一字封死）集合竞价可成交量上限 =
    # 当日成交量 × min(触板档比例, auction_fill_ratio)。假设文档化：
    # 成交价=开盘价（集合竞价价），开盘后向限价收敛的盘中成交不单独建模。
    auction_fill_ratio: float = 0.12
    # ── 经验填充模型（limit_calibration.py）：用历史日线 V_t/V_prev 分位数
    # 替代固定比例常量（技术债：固定比例缺经验依据）。分钟/tick/盘口数据
    # 不在数据源内，经验分位是可行的统计替代：
    #   fixed            = 旧行为（固定比例常量）
    #   empirical_median = 经验中位数（中性口径，竞价触板日可成交量=前日量×p50）
    #   empirical_p10    = 经验 10% 分位（worst-case 保守口径，暴露流动性枯竭）
    # 单元格样本不足 limit_calib_min_samples → 回退 fixed 档；校准表全样本静态
    # 统计（静态参数选择，应用时不带前视——竞价可成交量 = 前日量 × 校准分位）。
    limit_ratio_mode: str = "fixed"
    limit_calib_min_samples: int = 20
    # ── 0.6 复牌跳空（停牌后复牌日开盘大幅跳空：补涨兑现 / 补跌标记） ──
    resume_gap_up: float = 0.05  # 复牌高开≥该比例（相对停牌前收盘）→ 开盘兑现卖出 + 当日禁买
    resume_gap_down: float = 0.05  # 复牌低开≤-该比例 → 日志标记（风控卖出照常）
    # ── 交易参数（P1-2：从 core.py 硬编码提升为配置驱动） ──
    max_order_pct: float = 0.1  # 单笔订单上限（占 ADV 比例），超过则缩减股数
    top_k: int = 20  # 每日最大候选买入数（集中资金，避免每只分到极小额度 < 1 手）
    # ── P3-3（审计）挂单过期天数（P0-6 ②：A股订单当日有效，信号次日未成交即撤销） ──
    # 从 core.py 硬编码提升为配置项，便于研究（如"连续重挂 N 日"实验）；默认 1 保持原语义。
    order_expiry_days: int = 1
    # ── P3-4（审计）上市日表严格模式：params._listing_days 缺失时 fail-fast ──
    # 默认 False 兼容旧行为（告警后停用新股豁免）；True 与数据质量门禁联动，
    # 杜绝"表缺失 → 静默停用豁免 → 结果口径改变"的低风险漂移。
    strict_listing_days: bool = False
    # ── 市场状态仓位调节（P0-6 ⑤：客观状态变量，替代评分中位数口径） ──
    # 指数 20 日收益（全市场后复权收盘 ret_20d 中位数代理）+ 市场波动率分位
    # （横截面日收益 std 在过去 250 交易日分位）。评分口径字段（regime_full_
    # threshold 等）已弃用，仅保留以兼容旧配置。
    regime_ret20_full: float = 0.02  # 指数20日收益 ≥ 此值 → 全仓倍率
    regime_ret20_half: float = -0.02  # 指数20日收益 ≥ 此值（且非高波）→ 半仓倍率
    regime_vol_pct_max: float = 0.8  # 波动率分位 > 此值 → 高波动，压制到最低倍率
    # ── 以下为旧评分口径（P0-6 ⑤ 弃用，仅兼容保留） ──
    regime_full_threshold: int = 30  # [已弃用] 中位数评分 ≥ 此值 → 全仓倍率
    regime_half_threshold: int = 15  # [已弃用] 中位数评分 ≥ 此值 → 半仓倍率
    regime_full_multiplier: float = 1.0  # 全仓倍率
    regime_half_multiplier: float = 0.5  # 半仓倍率
    regime_min_multiplier: float = 0.25  # 最低仓位倍率

    # ── 组合优化器配置（P4 数学规划驱动） ──
    # 优化方法: mean_variance / min_variance / risk_parity / topk_equal(兼容旧版 Top-K 等权)
    optimizer_method: str = "topk_equal"
    # 风险厌恶系数 (γ, 越大越保守; 2.0 = 平衡, 5.0 = 保守)
    optimizer_risk_aversion: float = 2.0
    # 换手率惩罚系数 (λ_TC, 建议取年化交易成本 0.5~1.5 倍; 0.001 = 千1 轻度惩罚)
    optimizer_turnover_penalty: float = 0.001
    # 单票权重上限 (w_max, 默认 10%)
    optimizer_max_weight: float = 0.10
    # 协方差估计窗口 (交易日, 默认 60)
    optimizer_cov_lookback: int = 60
    # 协方差收缩 (Ledoit-Wolf, True = 小样本稳健)
    optimizer_shrinkage: bool = True
    # 行业中性约束 (是否启用)
    optimizer_industry_neutral: bool = False
    # 行业暴露偏离上限 (绝对值)
    optimizer_industry_deviation: float = 0.05
    # 最大持仓数 (0 = 不限制)
    optimizer_max_holdings: int = 0
    # 目标现金比例
    optimizer_target_cash: float = 0.0
    # 求解超时 (秒)
    optimizer_solve_timeout: float = 5.0

    @property
    def buy_fee_rate(self) -> float:
        """买入费率（不含滑点）：佣金 + 过户费。

        ⚠️ 仅供展示/兼容：简单相加不含逐笔最低佣金（min_commission_per_trade）
        与历史费率分段表，不能用于费用核算。引擎实际费用一律经
        engine_cfg.cost_model（CostModel.buy_cost_breakdown，逐笔强制下限）。
        """
        return self.commission_rate + self.transfer_fee_rate

    @property
    def sell_fee_rate(self) -> float:
        """卖出费率（不含滑点）：佣金 + 过户费 + 印花税。

        ⚠️ 仅供展示/兼容：简单相加不含逐笔最低佣金（min_commission_per_trade）
        与历史费率分段表，不能用于费用核算。引擎实际费用一律经
        engine_cfg.cost_model（CostModel.sell_cost_breakdown，逐笔强制下限）。
        """
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
