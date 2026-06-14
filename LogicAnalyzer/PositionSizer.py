"""
专业级仓位管理器 — 多因子混合仓位模型

方法论（按优先级）：
  1. 凯利准则（Kelly Criterion）— 以盈亏比为锚，计算最优押注比例
  2. 信号置信度折价 — 综合评分/级别转换为基础仓位
  3. 风险等级折价 — 风险越高的股票，仓位越低
  4. 市场状态乘数 — 强势趋势放大，弱势/震荡缩小
  5. 波动率上限（ATR 推导）— 波动越大，仓位上限越低（风险预算约束）
  6. 行业集中度限制 — 单一行业不超过配置上限（逐行标注，外部组合约束）

接口：
    calculate_positions(df, config) -> pd.DataFrame
      输入：merged DataFrame + config dict
      输出：DataFrame + ［建议仓位比例, 仓位依据］两列
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from DataManager.ColumnNames import ColumnNames


def _safe_float(val: Any, default: float = 0.0) -> float:  # noqa: ANN401
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError, RuntimeError):
        return default


def _safe_str(val: Any, default: str = "") -> str:  # noqa: ANN401
    if isinstance(val, str):
        return val
    try:
        return str(val)
    except (TypeError, ValueError):
        return default


def calculate_positions(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """对合并后的 DataFrame 逐行计算建议仓位比例。

    Args:
        df: 经过 filter_signal_stocks 之后的 DataFrame，需要包含
            COMPREHENSIVE_LEVEL, COMPREHENSIVE_SCORE, RISK_LEVEL, EXIT_RRR,
            STOP_LOSS, LATEST_PRICE, MACD_TREND_TYPE, 行业 等列。
        config: 仓位配置字典。必须由调用方从 ``Config.POSITION_SIZING`` 构造，
            可使用 ``.get(key, default)`` 保证容错。

    Returns:
        添加了 ``SUGGESTED_POSITION`` 和 ``POSITION_REASON`` 两列的 DataFrame。
    """
    if df.empty:
        df[ColumnNames.SUGGESTED_POSITION] = np.nan
        df[ColumnNames.POSITION_REASON] = ""
        return df

    cfg = config or {}

    result = df.copy()
    _max_single = cfg.get("max_single_position", 0.33)
    _kelly_frac = cfg.get("kelly_fraction", 0.25)
    _win_rate = cfg.get("default_win_rate", 0.50)
    _risk_budget = cfg.get("risk_budget", 0.02)
    _atr_stop_mult = cfg.get("atr_stop_mult", 1.5)
    _level_pos = {
        "A": cfg.get("position_a", 0.30),
        "B": cfg.get("position_b", 0.15),
        "C": cfg.get("position_c", 0.05),
        "D": cfg.get("position_d", 0.00),
    }

    positions = []
    reasons = []

    for _, row in result.iterrows():
        # ── 1. 基础仓位：级别 + 评分 ────────────────────────────────────
        level = _safe_str(row.get(ColumnNames.COMPREHENSIVE_LEVEL, "C")).strip().upper()
        score_raw = _safe_float(row.get(ColumnNames.COMPREHENSIVE_SCORE, 0), 0)
        score_factor = min(1.0, max(0.0, score_raw) / 100.0)

        level_base = _level_pos.get(level, 0.03)
        base = level_base * score_factor
        parts = [f"级别{level}({level_base:.0%})×评分{score_factor:.0%}→{base:.1%}"]

        # ── 2. 风险等级折价 ────────────────────────────────────────────
        risk = _safe_str(row.get(ColumnNames.RISK_LEVEL, "MEDIUM")).strip().upper()
        risk_map = {"NONE": 1.0, "LOW": 0.85, "MEDIUM": 0.50, "HIGH": 0.0}
        risk_mult = risk_map.get(risk, 0.50)

        if risk == "HIGH":
            positions.append(0.0)
            reasons.append("风险等级 HIGH，不持仓")
            continue
        if risk_mult < 1.0:
            parts.append(f"风险{risk}(×{risk_mult:.0%})")

        # ── 3. 市场状态乘数（MACD 趋势分类）─────────────────────────────
        trend_type = _safe_str(row.get(ColumnNames.MACD_TREND_TYPE, ""))
        regime_map = {
            "指标超强": 1.0,
            "指标强势": 0.85,
            "指标弱势": 0.40,
            "指标超弱": 0.0,
        }
        regime_mult = regime_map.get(trend_type, 0.50)
        if regime_mult < 1.0:
            parts.append(f"状态{trend_type}(×{regime_mult:.0%})")
        if regime_mult <= 0.0:
            positions.append(0.0)
            reasons.append(f"趋势极弱({trend_type})，不持仓")
            continue

        # ── 4. Kelly 调整（盈亏比驱动）───────────────────────────────────
        rrr = _safe_float(row.get(ColumnNames.EXIT_RRR, 0), 0)
        kelly_mod = 1.0
        if rrr > 1.0:
            # f* = (p * b - q) / b  ;  q = 1 - p
            # 默认 p=0.5（保守），半凯利
            kelly_full = (_win_rate * rrr - (1 - _win_rate)) / rrr
            kelly_full = max(0.0, kelly_full)
            kelly_used = kelly_full * _kelly_frac
            # 映射到乘数范围 [0.8, 1.5]
            kelly_mod = 0.8 + kelly_used * 0.7
            kelly_mod = min(max(kelly_mod, 0.5), 1.5)
            parts.append(f"Kelly(RRR={rrr:.1f},×{kelly_mod:.0%})")
        elif rrr > 0:
            kelly_mod = 0.7
            parts.append(f"RRR≤1({rrr:.1f},×{kelly_mod:.0%})")
        else:
            parts.append("无RRR(×1.0)")

        # ── 5. 波动率上限（ATR 推导）────────────────────────────────────
        close = _safe_float(row.get(ColumnNames.LATEST_PRICE, 0), 0)
        stop_loss = _safe_float(row.get(ColumnNames.STOP_LOSS, 0), 0)
        vol_cap = _max_single
        if close > 0 and stop_loss > 0 and stop_loss < close:
            atr = (close - stop_loss) / _atr_stop_mult
            atr_pct = atr / close
            if atr_pct > 0.001:
                vol_cap = _risk_budget / atr_pct
                vol_cap = min(vol_cap, _max_single)
                if vol_cap < _max_single:
                    parts.append(f"波动约束(ATR%={atr_pct:.1%},上限{vol_cap:.0%})")

        # ── 6. Gate 4 仓位调整（来自 ScoringRules 规则引擎）───────────────
        pos_adj = _safe_float(row.get("position_adjust", 0), 0)
        pos_adj = max(-1.0, min(1.0, pos_adj))
        if pos_adj != 0:
            parts.append(f"规则调整({pos_adj:+.0%})")

        # ── 7. 流动性折扣（基于截面+时序+规模三因子评分）──────────────────
        liq_score = _safe_float(row.get(ColumnNames.LIQUIDITY_SCORE, 1.0), 1.0)
        liq_min = cfg.get("liq_min_discount", 0.3)
        liq_discount = liq_min + (1.0 - liq_min) * liq_score
        if liq_discount < 1.0:
            parts.append(f"流动性(×{liq_discount:.0%})")

        # ── 8. 合成最终仓位 ─────────────────────────────────────────────
        position = base * risk_mult * regime_mult * kelly_mod * (1 + pos_adj) * liq_discount
        position = min(position, vol_cap, _max_single)
        position = max(position, 0.0)
        position = round(position, 4)

        positions.append(position)
        reasons.append(" | ".join(parts))

    result[ColumnNames.SUGGESTED_POSITION] = positions
    result[ColumnNames.POSITION_REASON] = reasons
    return result


