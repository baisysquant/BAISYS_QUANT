from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig, _run_single_backtest
from BackTrading.prepare import _build_params, prepare_backtest_data
from LogicAnalyzer.backtest_metrics import compute_risk_metrics


@dataclass
class SimTradeVerdict:
    """模拟交易验证结果与决策。"""

    sim_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    sim_sortino: float = 0.0
    oos_sortino: float = 0.0
    sharpe_degradation: float = 0.0     # 1 - sim/oos，负值表示 sim 优于 oos
    sortino_degradation: float = 0.0
    promote: bool = False
    reason: str = ""

    # 兼容性：保留旧字段名（degradation = sharpe_degradation）
    @property
    def degradation(self) -> float:
        return self.sharpe_degradation


# 衰减容忍度：30%（审计要求，原 50% 过松）
_DECAY_THRESHOLD = 0.30
# warmup 天数（sim 段前预留交易日 buffer，让引擎建立 ADV/仓位后再进入 sim 段，
# 避免 20 天 sim 窗口下 ADV 永不满载导致流动性约束失效）
_SIM_WARMUP_DAYS = 30


def _cost_model_from_config(config: Any) -> Any:
    """从 Config 构建 CostModel（含流动性分档冲击成本），无配置时返回 None（回落统一成本）。"""
    if config is None:
        return None
    try:
        from BackTrading.domain.models import CostModel
        return CostModel.from_backtest_config(config.app_config.backtest)
    except Exception as e:
        logger.warning(f"[模拟验证] CostModel 构建失败，回落统一成本: {e}")
        return None


def validate_params(
    kline_df: pd.DataFrame,
    best_params: dict[str, float],
    oos_sharpe: float,
    sim_days: int = 20,
    config: Any | None = None,
    engine_cfg: EngineConfig | None = None,
    oos_sortino: float = 0.0,  # 审计新增：样本外 Sortino
    validation_dates: set[str] | None = None,  # P0-10 ④：独立验证集（与选参区间无交集）
) -> SimTradeVerdict:
    """用独立验证集验证 best_params 的稳定性。

    P0-10 ④：原实现取"最近 sim_days 个交易日"做模拟验证，该区间与 WFO 的
    holdout/OOS 选参区间重叠 → 自引用验证（sim 段正是选参评估段）。
    现优先使用调用方传入的独立验证集（如末段 holdout，WFO 全程禁触）；
    未提供时才回退最近 N 日并告警（自引用回退，与主流程口径一致）。

    Args:
        kline_df: 全量 K 线数据（含信号列或原始数据均可）。
        best_params: WFO 选出的最佳参数（flat dict，至少含 atr_stop_mult）。
        oos_sharpe: WFO 在样本外窗口上的 Sharpe。
        sim_days: 验证集交易日数。
        config: Config 实例（可选，用于构建结构化 params）。
        engine_cfg: EngineConfig 实例（可选，构建最终回测引擎）。
        oos_sortino: WFO 在样本外窗口上的 Sortino（审计新增，默认 0=跳过 Sortino 校验）。
        validation_dates: 独立验证集日期集合（与选参区间无交集）；
            None 时回退"最近 sim_days 日"（自引用，告警）。

    Returns:
        SimTradeVerdict 包含决策和原因。
    """
    if best_params is None or oos_sharpe is None:
        return SimTradeVerdict(promote=False, reason="WFO 结果为空，跳过模拟验证")

    # 取验证段交易日（独立验证集优先；否则最近 sim_days 日自引用回退）
    dates = sorted(kline_df["trade_date"].unique())
    if validation_dates:
        _v_dates = sorted({str(d) for d in dates} & {str(d) for d in validation_dates})
        if len(_v_dates) < max(10, sim_days // 2):
            return SimTradeVerdict(
                promote=True,
                reason=f"独立验证集交易日仅 {len(_v_dates)} 天 < {max(10, sim_days // 2)}，"
                       f"跳过模拟验证",
            )
        sim_dates_sorted = _v_dates[-sim_days:]
    else:
        logger.warning(
            "[模拟验证] 未提供独立验证集（holdout 未激活），"
            "回退最近 N 个交易日做模拟验证（自引用，结果仅供参考）"
        )
        if len(dates) < sim_days + 20:
            return SimTradeVerdict(
                promote=True,
                reason=f"数据不足（{len(dates)} 个交易日），无法做模拟验证，直接放行",
            )
        sim_dates_sorted = dates[-sim_days:]

    # 准备信号 + 止损价
    if config is not None:
        from UtilsManager.ConfigParser import Config as _Cfg
        cfg = config if isinstance(config, _Cfg) else _Cfg()
        structured = _build_params(cfg)
        structured["scoring"].update(
            {k: v for k, v in best_params.items() if k in (
                "atr_stop_mult",
            )}
        )
        prepared = prepare_backtest_data(kline_df, params=structured, compute_exit_strategy=False, vectorized=True)
    else:
        prepared = prepare_backtest_data(kline_df, params=best_params, compute_exit_strategy=False, vectorized=True)

    # ── P1-1：warmup buffer 避免 ADV 冷启动 ──
    # 引擎 ADV 20 日窗口从 0 开始填充，若仅传入 sim_days（20天）切片，
    # 窗口永不满载 → _adv_val > 100 gate 失效 → 流动性约束（_max_order_pct）不触发 → 偏乐观。
    # 回测从 sim 段前 30 个交易日开始，引擎建立 ADV/仓位后再进入 sim 段，
    # 指标仅在 sim 段上计算（warmup 段不入指标）。
    _prep = prepared
    if pd.api.types.is_datetime64_any_dtype(_prep["trade_date"]):
        _prep = _prep.copy()
        _prep["trade_date"] = _prep["trade_date"].dt.strftime("%Y-%m-%d")
    _date_range = _prep["trade_date"].astype(str)
    _unique_dates = sorted(_date_range.unique())
    # P0-10 ④：验证段按 sim_dates_sorted 定位（独立验证集可能不在数据末尾），
    # 回退模式下即末尾 sim_days 日
    _sim_n = min(len(sim_dates_sorted), len(_unique_dates))
    _sim_dates_str = {d for d in sim_dates_sorted if d in _unique_dates}
    sim_dates_str = _sim_dates_str
    if sim_dates_sorted and sim_dates_sorted[0] in _unique_dates:
        _sim_start_pos = _unique_dates.index(sim_dates_sorted[0])
    else:
        _sim_start_pos = len(_unique_dates) - _sim_n
    # 扩展段 [warmup_start, sim_end]
    _warmup_pos = max(0, _sim_start_pos - _SIM_WARMUP_DAYS)
    _warmup_start = _unique_dates[_warmup_pos]
    _sim_end = sim_dates_sorted[-1] if sim_dates_sorted else _unique_dates[-1]
    mask_ext = (_date_range >= _warmup_start) & (_date_range <= _sim_end)
    ext_data = _prep[mask_ext].copy()
    if ext_data.empty:
        return SimTradeVerdict(promote=True, reason="模拟期数据为空，直接放行")

    # 按 best_params 的 atr_stop_mult 计算止损价
    stop_mult = best_params.get("atr_stop_mult")
    if stop_mult is not None and "ATR" in ext_data.columns:
        # P0-1：止损价与引擎比较基准统一到后复权空间（指标 ATR 亦为后复权）
        _stop_close = ext_data["close_adj"] if "close_adj" in ext_data.columns else ext_data["close"]
        ext_data["止损价"] = _stop_close - ext_data["ATR"] * stop_mult
    elif "止损价" not in ext_data.columns:
        ext_data["止损价"] = 0.0

    if engine_cfg is None:
        engine_cfg = EngineConfig(
            atr_stop_mult=best_params.get("atr_stop_mult", 1.5),
            cost_model=_cost_model_from_config(config),
        )

    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(ext_data, best_params, engine_cfg, tl, ec)

    # 仅保留 sim 段权益曲线（warmup 段不入指标）
    ec_sim = [row for row in ec if str(row.get("time", ""))[:10] in sim_dates_str]
    if not ec_sim:
        return SimTradeVerdict(promote=True, reason="模拟期权益数据为空，直接放行")

    risk = compute_risk_metrics(ec_sim) or {}
    sim_sharpe = risk.get("sharpe_ratio", 0.0) or 0.0
    sim_sortino = risk.get("sortino_ratio")
    if sim_sortino is None or not math.isfinite(sim_sortino):
        sim_sortino = 0.0

    # ── Sharpe 衰减校验（审计要求：阈值 30%） ──
    sharpe_deg = 1.0 - (sim_sharpe / oos_sharpe) if oos_sharpe > 0.01 else 0.0

    # ── Sortino 衰减校验（若 oos_sortino 未传入则跳过） ──
    sortino_deg = 0.0
    if oos_sortino > 0.01:
        sortino_deg = 1.0 - (sim_sortino / oos_sortino)

    # ── 判定：任一指标衰减超 30% 则拒绝 ──
    promote = sim_sharpe > 0.0 and sharpe_deg < _DECAY_THRESHOLD and sortino_deg < _DECAY_THRESHOLD

    if promote:
        reason = (
            f"模拟验证通过: sim_Sharpe={sim_sharpe:.2f} / oos_Sharpe={oos_sharpe:.2f}"
            f" (衰减 {sharpe_deg:.0%}) | sim_Sortino={sim_sortino:.2f}"
            f" 退化率={sharpe_deg:.0%} < {_DECAY_THRESHOLD:.0%}"
        )
    else:
        reason = (
            f"模拟验证不通过: sim_Sharpe={sim_sharpe:.2f} / oos_Sharpe={oos_sharpe:.2f}"
            f" (衰减 {sharpe_deg:.0%}) | sim_Sortino={sim_sortino:.2f}"
            f" 退化率 >= {_DECAY_THRESHOLD:.0%}（或 sim Sharpe ≤ 0），参数组不予上线"
        )

    logger.info(f"  {reason}")
    return SimTradeVerdict(
        sim_sharpe=sim_sharpe,
        oos_sharpe=oos_sharpe,
        sim_sortino=sim_sortino,
        oos_sortino=oos_sortino,
        sharpe_degradation=sharpe_deg,
        sortino_degradation=sortino_deg,
        promote=promote,
        reason=reason,
    )
