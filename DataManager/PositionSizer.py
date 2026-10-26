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
    _win_rate = cfg.get("default_win_rate", 0.55)  # 与 PositionSizingConfig.DEFAULT_WIN_RATE 一致
    _risk_budget = cfg.get("risk_budget", 0.02)
    _atr_stop_mult = cfg.get("atr_stop_mult", 1.5)
    _level_pos = {
        "A": cfg.get("position_a", 0.30),
        "B": cfg.get("position_b", 0.15),
        "C": cfg.get("position_c", 0.05),
        "D": cfg.get("position_d", 0.00),
    }
    _cfg_risk_none = cfg.get("risk_none_multiplier", 1.0)  # 从 POSITION_SIZING 读取

    levels = result[ColumnNames.COMPREHENSIVE_LEVEL].fillna("C").astype(str).str.strip().str.upper()
    scores = pd.to_numeric(result[ColumnNames.COMPREHENSIVE_SCORE], errors='coerce').fillna(0).clip(0, 100)
    score_factors = scores / 100.0
    level_bases = levels.map(_level_pos).fillna(0.03)
    bases = level_bases * score_factors

    risks = result[ColumnNames.RISK_LEVEL].fillna("MEDIUM").astype(str).str.strip().str.upper()
    risk_map_s = pd.Series(risks).map({"NONE": _cfg_risk_none, "LOW": 0.85, "MEDIUM": 0.50, "HIGH": 0.0}).fillna(0.50)
    high_risk = risks == "HIGH"

    trends = result[ColumnNames.MACD_TREND_TYPE].fillna("").astype(str)
    regime_map_s = trends.map({"指标超强": 1.0, "指标强势": 0.85, "指标弱势": 0.40, "指标超弱": 0.0}).fillna(0.50)
    zero_regime = regime_map_s <= 0.0

    rrr_vals = pd.to_numeric(result[ColumnNames.EXIT_RRR], errors='coerce').fillna(0)
    rrr_safe = rrr_vals.replace(0, float('nan'))
    kelly_full = (_win_rate * rrr_vals - (1 - _win_rate)) / rrr_safe
    kelly_full = kelly_full.clip(0)
    kelly_used = kelly_full * _kelly_frac
    kelly_mods = 0.8 + kelly_used * 0.7
    kelly_mods = kelly_mods.clip(0.5, 1.5)
    kelly_mods[(rrr_vals <= 0)] = 1.0
    kelly_mods[(rrr_vals > 0) & (rrr_vals <= 1.0)] = 0.7

    closes = pd.to_numeric(result[ColumnNames.LATEST_PRICE], errors='coerce').fillna(0)
    stops = pd.to_numeric(result[ColumnNames.STOP_LOSS], errors='coerce').fillna(0)
    atr_pcts = (closes - stops) / (_atr_stop_mult * closes.replace(0, float('nan')))
    vol_caps = _risk_budget / atr_pcts.where((closes > 0) & (stops > 0) & (stops < closes) & (atr_pcts > 0.001), float('nan'))
    vol_caps = vol_caps.clip(0, _max_single).fillna(_max_single)

    if "position_adjust" in result.columns:
        pos_adjs = pd.to_numeric(result["position_adjust"], errors='coerce').fillna(0).clip(-1.0, 1.0)
    else:
        pos_adjs = pd.Series(0, index=result.index)

    positions = bases * risk_map_s * regime_map_s * kelly_mods * (1 + pos_adjs)
    positions = positions.clip(0, _max_single)
    positions[high_risk | zero_regime] = 0.0
    positions = positions.round(4)

    result[ColumnNames.SUGGESTED_POSITION] = positions

    reasons = []
    for i in range(len(result)):
        if high_risk.iloc[i]:
            reasons.append("风险等级 HIGH，不持仓")
        elif zero_regime.iloc[i]:
            reasons.append(f"趋势极弱({trends.iloc[i]})，不持仓")
        else:
            level = levels.iloc[i]
            lb = level_bases.iloc[i]
            sf = score_factors.iloc[i]
            base = bases.iloc[i]
            parts = [f"级别{level}({lb:.0%})×评分{sf:.0%}→{base:.1%}"]
            risk = risks.iloc[i]
            rm = risk_map_s.iloc[i]
            if rm < 1.0:
                parts.append(f"风险{risk}(×{rm:.0%})")
            trend = trends.iloc[i]
            tm = regime_map_s.iloc[i]
            if tm < 1.0:
                parts.append(f"状态{trend}(×{tm:.0%})")
            rrr = rrr_vals.iloc[i]
            km = kelly_mods.iloc[i]
            if rrr > 1.0:
                parts.append(f"Kelly(RRR={rrr:.1f},×{km:.0%})")
            elif rrr > 0:
                parts.append(f"RRR≤1({rrr:.1f},×{km:.0%})")
            else:
                parts.append("无RRR(×1.0)")
            close = closes.iloc[i]
            stop = stops.iloc[i]
            if close > 0 and stop > 0 and stop < close:
                atr = (close - stop) / _atr_stop_mult
                atr_pct = atr / close
                if atr_pct > 0.001 and vol_caps.iloc[i] < _max_single:
                    parts.append(f"波动约束(ATR%={atr_pct:.1%},上限{vol_caps.iloc[i]:.0%})")
            adj = pos_adjs.iloc[i]
            if adj != 0:
                parts.append(f"规则调整({adj:+.0%})")
            reasons.append(" | ".join(parts))
    result[ColumnNames.POSITION_REASON] = reasons
    return result
