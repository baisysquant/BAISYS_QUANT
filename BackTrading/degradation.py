"""指标计算降级策略（STRICT/RELAX/SKIP）+ min_periods 支持 + 置信度标签。

目标：对 rolling / ewm / talib 指标实现可配置降级：
    STRICT — 样本不足时保持原周期全窗计算（头部自然 NaN，等价原行为，可回退）
    RELAX  — 样本不足时缩窗计算（min_periods 下限），结果标 low_confidence
    SKIP   — 样本不足时仍按全窗计算，但整段标 low_confidence，由策略层跳过

safe_* 包装（MACD/ATR/EMA/MA/BBANDS/ADX/RSI/STOCH/CCI）:
    样本不足时绝不抛异常，返回 (值, Confidence, start_bar)。
    start_bar: 该指标有效值起始 bar（降级后更早），供 _compute_indicators
    生成 bar 级 _IND_CONF 置信度列。

策略层消费（compute_signals / _stock_worker）:
    低置信度 bar → skip（进场评分归零，不下单）或 low_weight（按系数降权）。

配置: [BACKTEST]
    indicator_degradation = RELAX            # STRICT / RELAX（SKIP 可通过 API 参数启用）
    indicator_degradation_low_action = skip  # skip | low_weight
    indicator_degradation_low_weight = 0.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from UtilsManager import TACompatibility as ta

_CONF_HIGH = "high"
_CONF_LOW = "low"


class DegradeMode(str, Enum):
    STRICT = "STRICT"
    RELAX = "RELAX"
    SKIP = "SKIP"


# ── 配置读取（带兜底） ────────────────────────────────────────────────

def degrade_mode(override: str | None = None) -> DegradeMode:
    """指标计算降级模式：override 优先，其次 config INDICATOR_DEGRADATION，默认 RELAX。"""
    if override:
        m = override.upper()
    else:
        try:
            from UtilsManager.ConfigParser import Config
            m = str(Config().app_config.backtest.INDICATOR_DEGRADATION).upper()
        except Exception:
            m = "RELAX"
    return DegradeMode(m if m in ("STRICT", "RELAX", "SKIP") else "RELAX")


def low_confidence_action() -> str:
    """低置信度信号处理动作：skip（不下单）/ low_weight（降权）。"""
    try:
        from UtilsManager.ConfigParser import Config
        a = str(Config().app_config.backtest.INDICATOR_DEGRADATION_LOW_ACTION).lower()
        return a if a in ("skip", "low_weight") else "skip"
    except Exception:
        return "skip"


def low_confidence_weight() -> float:
    try:
        from UtilsManager.ConfigParser import Config
        return float(Config().app_config.backtest.INDICATOR_DEGRADATION_LOW_WEIGHT)
    except Exception:
        return 0.5


# ── 置信度与周期决策 ──────────────────────────────────────────────────

@dataclass
class Confidence:
    """置信度标签：high / low + 原因。"""

    level: str = _CONF_HIGH
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def high(cls) -> Confidence:
        return cls(_CONF_HIGH, [])

    @classmethod
    def low(cls, reasons: list[str]) -> Confidence:
        return cls(_CONF_LOW, list(reasons))

    @property
    def is_low(self) -> bool:
        return self.level == _CONF_LOW

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "reasons": self.reasons}


@dataclass
class IndicatorResult:
    """safe wrapper 统一返回：值 + 置信度 + 有效值起始 bar。"""

    value: Any
    confidence: Confidence
    start_bar: int = 0


def resolve_period(
    indicator: str,
    period: int,
    n: int,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    """指标周期降级决策。

    Returns:
        IndicatorResult(period_used, confidence, start_bar)：
            - 样本充足 (n ≥ period): 原周期, high, start=period-1
            - STRICT / 样本不足:    原周期, high, start=period-1（原行为，头部 NaN）
            - SKIP / 样本不足:      原周期, low,  start=0（整段低置信，策略层跳过）
            - RELAX / 样本不足:     缩窗周期, low, start=reduced-1
    """
    mode = mode or degrade_mode()
    if n >= period:
        return IndicatorResult(period, Confidence.high(), period - 1)
    if mode == DegradeMode.STRICT:
        return IndicatorResult(period, Confidence.high(), period - 1)
    if mode == DegradeMode.SKIP:
        return IndicatorResult(
            period,
            Confidence.low([f"{indicator}: insufficient n={n}<{period}（SKIP，策略层跳过）"]),
            0,
        )
    floor = max(1, int(min_periods or 3))
    reduced = max(1, min(period, max(floor, n // 2)))
    return IndicatorResult(
        reduced,
        Confidence.low([f"{indicator}: degraded period {period}->{reduced} (n={n})"]),
        max(0, reduced - 1),
    )


# ── safe wrappers（样本不足不抛异常，返回带置信度结果） ────────────────

def safe_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    n = len(close)
    res = resolve_period("macd", slow, n, mode, min_periods)
    slow2 = int(res.value)
    fast2 = min(fast, max(2, slow2 // 2))
    signal2 = min(signal, max(1, slow2 // 3))
    return IndicatorResult(ta.macd(close, fast=fast2, slow=slow2, signal=signal2),
                           res.confidence, res.start_bar)


def safe_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period("atr", length, len(close), mode, min_periods)
    return IndicatorResult(ta.atr(high, low, close, length=int(res.value)),
                           res.confidence, res.start_bar)


def safe_ma(
    close: pd.Series,
    period: int,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period(f"ma_{period}", period, len(close), mode, min_periods)
    mp = min(int(res.value), period)
    return IndicatorResult(close.rolling(period, min_periods=mp).mean(),
                           res.confidence, res.start_bar)


def safe_ema(
    close: pd.Series,
    span: int,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period(f"ema_{span}", span, len(close), mode, min_periods)
    mp = min(int(res.value), span)
    return IndicatorResult(close.ewm(span=span, adjust=False, min_periods=mp).mean(),
                           res.confidence, res.start_bar)


def safe_bbands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period("bbands", length, len(close), mode, min_periods)
    return IndicatorResult(ta.bbands(close, length=int(res.value), std=std),
                           res.confidence, res.start_bar)


def safe_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period("adx", length, len(close), mode, min_periods)
    return IndicatorResult(ta.adx(high, low, close, length=int(res.value)),
                           res.confidence, res.start_bar)


def safe_rsi(
    close: pd.Series,
    length: int = 14,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period("rsi", length, len(close), mode, min_periods)
    return IndicatorResult(ta.rsi(close, length=int(res.value)),
                           res.confidence, res.start_bar)


def safe_stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 9,
    d: int = 3,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period("stoch", k, len(close), mode, min_periods)
    k2 = int(res.value)
    d2 = min(d, max(1, k2 // 2))
    return IndicatorResult(ta.stoch(high, low, close, k=k2, d=d2),
                           res.confidence, res.start_bar)


def safe_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mode: DegradeMode | None = None,
    min_periods: int | None = None,
) -> IndicatorResult:
    res = resolve_period("cci", length, len(close), mode, min_periods)
    return IndicatorResult(ta.cci(high, low, close, length=int(res.value)),
                           res.confidence, res.start_bar)


# ── 策略层消费辅助 ────────────────────────────────────────────────────

def low_confidence_mask(df: pd.DataFrame) -> np.ndarray:
    """从指标帧提取 bar 级低置信度掩码（无 _IND_CONF 列 → 全高置信度）。"""
    if "_IND_CONF" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return (df["_IND_CONF"].fillna(_CONF_HIGH).astype(str).to_numpy() == _CONF_LOW)


def apply_confidence_consumption(
    scores: pd.DataFrame,
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """策略层消费 confidence_flag：低置信度 bar 抑制信号（skip）或降权（low_weight）。

    原地修改 scores 的 entry_score/score（golden_cross 在 low_weight 时同步降权），
    返回低置信度掩码（供调用方日志/审计）。
    """
    low_mask = low_confidence_mask(df)
    n_low = int(low_mask.sum())
    if n_low == 0:
        return low_mask
    conf_p = (params or {}).get("indicator_degradation") or {}
    action = str(conf_p.get("low_confidence_action") or low_confidence_action()).lower()
    if action == "low_weight":
        w = float(conf_p.get("low_confidence_weight") or low_confidence_weight())
        for col in ("entry_score", "score", "golden_cross"):
            if col in scores.columns:
                scores.loc[low_mask, col] = scores.loc[low_mask, col] * w
        logger.debug(f"指标降级: {n_low} 根 bar 低置信度，按 {w:.2f} 权重降权")
    else:
        for col in ("entry_score", "score"):
            if col in scores.columns:
                scores.loc[low_mask, col] = 0.0
        logger.debug(f"指标降级: {n_low} 根 bar 低置信度，跳过信号（不下单）")
    return low_mask
