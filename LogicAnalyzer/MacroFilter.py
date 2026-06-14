"""
宏观三级过滤器（Gate 0 前置）。

Level 1 — 指数趋势（上证指数 MA60/MA120 空头排列判据）
Level 2 — 成交量验证（缩量下跌 / 放量下跌 / 缩量反弹）
Level 3 — 市场广度（全A上涨比例）

综合决策：
  HIGH_RISK 出现任意一次 → SKIP_ALL（跳过当日全部个股分析）
  MEDIUM_RISK 出现任意一次且无 HIGH_RISK → CAUTION（阈值上浮 15%）
  全部 NORMAL → NORMAL（正常分析）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import akshare as ak
import pandas as pd
from loguru import logger


@dataclass
class MacroFilterResult:
    risk_level: str = "NORMAL"       # HIGH_RISK | MEDIUM_RISK | LOW_RISK | NORMAL
    decision: str = "NORMAL"         # SKIP_ALL | CAUTION | NORMAL
    detail: str = ""                 # 综合描述
    l1_level: str = "NORMAL"
    l1_reason: str = ""
    l2_level: str = "NORMAL"
    l2_reason: str = ""
    l3_level: str = "NORMAL"
    l3_reason: str = ""
    score_adjust: float = 1.0        # CAUTION 时 = 0.85

    _cache: dict = field(default_factory=dict, repr=False)


class MacroFilter:

    INDEX_SYMBOL = "sh000001"

    @staticmethod
    def check(
        spot_df: pd.DataFrame | None = None,
        index_df: pd.DataFrame | None = None,
        start_date: str = "20250101",
        end_date: str | None = None,
    ) -> MacroFilterResult:
        """
        执行三级宏观过滤。
        spot_df: 全A实时行情 DataFrame（已拉取），若 None 自动拉取
        index_df: 上证指数日K线 DataFrame，若 None 自动拉取
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        result = MacroFilterResult()

        # ── Level 1 + 2：上证指数分析 ──
        if index_df is None:
            try:
                index_df = ak.stock_zh_index_daily_tx(
                    symbol=MacroFilter.INDEX_SYMBOL,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                logger.warning("宏观过滤：上证指数获取失败 %s，跳过 Level 1/2", e)
                index_df = pd.DataFrame()

        if not index_df.empty:
            index_df = index_df.sort_values("date").reset_index(drop=True)
            index_df["MA60"] = index_df["close"].rolling(60).mean()
            index_df["MA120"] = index_df["close"].rolling(120).mean()
            last = index_df.iloc[-1]
            recent_60_low = index_df["low"].iloc[-60:].min()
            close, ma60, ma120 = last["close"], last["MA60"], last["MA120"]

            # Level 1
            if pd.notna(close) and pd.notna(ma60) and pd.notna(ma120):
                if close < ma60 and ma60 < ma120:
                    result.l1_level = "HIGH_RISK"
                    result.l1_reason = f"空头排列 close({close:.0f})<MA60({ma60:.0f})<MA120({ma120:.0f})"
                elif close < ma60 and ma60 > ma120:
                    result.l1_level = "MEDIUM_RISK"
                    result.l1_reason = f"中期回调 close({close:.0f})<MA60({ma60:.0f})"
                else:
                    result.l1_level = "NORMAL"
                    result.l1_reason = f"趋势正常 close({close:.0f}) MA60({ma60:.0f}) MA120({ma120:.0f})"

                if close < recent_60_low * 1.02:
                    result.l1_level = "HIGH_RISK"
                    result.l1_reason += f" | 接近阶段新低({recent_60_low:.0f})"

            # Level 2
            if len(index_df) >= 20:
                change_pct = (last["close"] - index_df.iloc[-2]["close"]) / index_df.iloc[-2]["close"] * 100
                amount_ma5 = index_df["amount"].iloc[-5:].mean()
                amount_ma20 = index_df["amount"].iloc[-20:].mean()
                if change_pct < -1 and amount_ma5 <= amount_ma20 * 0.8:
                    result.l2_level = "HIGH_RISK"
                    result.l2_reason = f"缩量下跌 {change_pct:.1f}% | 均额不足"
                elif change_pct < -2 and amount_ma5 >= amount_ma20 * 1.5:
                    result.l2_level = "MEDIUM_RISK"
                    result.l2_reason = f"放量下跌 可能见底 {change_pct:.1f}%"
                elif change_pct > 1 and amount_ma5 <= amount_ma20 * 0.7:
                    result.l2_level = "HIGH_RISK"
                    result.l2_reason = f"缩量反弹 {change_pct:.1f}% | 无量"
                else:
                    result.l2_level = "NORMAL"
                    result.l2_reason = f"量价正常 {change_pct:.1f}%"

        # ── Level 3：市场广度 ──
        if spot_df is None:
            try:
                spot_df = ak.stock_zh_a_spot_em()
            except Exception as e:
                logger.warning("宏观过滤：全A行情获取失败 %s，跳过 Level 3", e)

        if spot_df is not None and "涨跌幅" in spot_df.columns and len(spot_df) > 100:
            total = len(spot_df)
            up = (spot_df["涨跌幅"] > 0).sum()
            ratio = up / total
            if ratio < 0.25:
                result.l3_level = "HIGH_RISK"
                result.l3_reason = f"情绪冰点 上涨比例{ratio*100:.1f}%"
            elif ratio < 0.35:
                result.l3_level = "MEDIUM_RISK"
                result.l3_reason = f"偏弱 上涨比例{ratio*100:.1f}%"
            elif ratio > 0.70:
                result.l3_level = "MEDIUM_RISK"
                result.l3_reason = f"过热 上涨比例{ratio*100:.1f}%"
            else:
                result.l3_level = "NORMAL"
                result.l3_reason = f"正常 上涨比例{ratio*100:.1f}%"

        # ── 综合决策 ──
        all_levels = [result.l1_level, result.l2_level, result.l3_level]
        if "HIGH_RISK" in all_levels:
            result.risk_level = "HIGH_RISK"
            result.decision = "SKIP_ALL"
        elif "MEDIUM_RISK" in all_levels:
            result.risk_level = "MEDIUM_RISK"
            result.decision = "CAUTION"
            result.score_adjust = 0.85
        else:
            result.risk_level = "NORMAL"
            result.decision = "NORMAL"
            result.score_adjust = 1.0

        result.detail = (
            f"L1={result.l1_level}({result.l1_reason}); "
            f"L2={result.l2_level}({result.l2_reason}); "
            f"L3={result.l3_level}({result.l3_reason})"
        )
        return result
