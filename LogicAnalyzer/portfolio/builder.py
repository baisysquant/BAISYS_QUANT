"""
组合构建器

在 filter_weak_stocks 后重新归一化仓位权重，施加：
  - 行业集中度上限（保留行业内评分最高者或等比缩）
  - 单票上限（33%）
  - 总仓位上限（等比缩仓至 100% 以内）
  - 换手率约束（对比上日持仓，限制双边换手）
  - 交易成本模型（佣金 + 印花税 + 过户费，扣减预期收益）
  - 输出"目标权重"列
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text as sql_text

from DataManager.ColumnNames import ColumnNames


class PortfolioBuilder:
    """将因子评分转化为目标仓位权重的组合构建器。

    输入：已过滤弱势股的 consolidated_report（含建议仓位比例）
    输出：添加"目标权重"列，更新"建议仓位比例"为组合约束后值


    """

    def __init__(
        self,
        config: Any,
        db_engine: Any = None,
        today_str: str | None = None,
    ) -> None:
        self.config = config
        self._engine = db_engine
        self._today = today_str

        sizing = getattr(config, "POSITION_SIZING", None) or {}
        self._max_single = sizing.get("max_single_position", 0.33)
        self._max_industry = sizing.get("max_industry_exposure", 0.30)
        self._max_total = sizing.get("max_total_exposure", 1.0)
        self._max_turnover = sizing.get("max_day_turnover", 0.20)
        self._risk_aversion = sizing.get("risk_aversion", 1.0)

        # 交易成本参数
        tc = getattr(config, "TRADING_COST_PARAMS", None) or {}
        self._commission_rate = tc.get("commission_rate", 0.0003)
        self._stamp_tax_rate = tc.get("stamp_tax_rate", 0.0005)
        self._transfer_fee_rate = tc.get("transfer_fee_rate", 0.00001)

    # ── 入口 ───────────────────────────────────────────────

    def build(self, df: pd.DataFrame, hist_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """执行组合构建全流程，使用启发式约束。"""
        if df.empty:
            df[ColumnNames.SUGGESTED_POSITION] = np.nan
            df["目标权重"] = np.nan
            return df

        result = df.copy()
        result = self._build_heuristic(result)
        return result

    # ── 启发式方法 ──────────────────────────────────────────

    def _build_heuristic(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用启发式约束构建组合（原线性方法）。"""
        if df.empty:
            df[ColumnNames.SUGGESTED_POSITION] = np.nan
            df["目标权重"] = np.nan
            return df

        result = df.copy()
        pos_col = ColumnNames.SUGGESTED_POSITION

        # Step 1: 清零负值/空值
        result[pos_col] = pd.to_numeric(result.get(pos_col, 0), errors="coerce").fillna(0).clip(0)

        # Step 2: 行业集中度约束
        if ColumnNames.INDUSTRY in result.columns:
            result = self._constrain_industry(result)

        # Step 3: 单票上限约束
        result[pos_col] = result[pos_col].clip(0, self._max_single)

        # Step 4: 总仓位上限约束
        result = self._cap_total(result, pos_col)

        # Step 5: 换手率约束 + 交易成本
        result = self._apply_turnover_and_cost(result, pos_col)

        # Step 6: 输出"目标权重"列
        result["目标权重"] = result[pos_col].clip(0)

        filled = (result["目标权重"] > 0).sum()
        logger.info(
            f"[PortfolioBuilder] 启发式组合构建完成：{filled}/{len(result)} 只持仓，"
            f"总仓位 {result['目标权重'].sum():.1%}"
        )
        return result

    # ── 约束方法 ───────────────────────────────────────────

    def _cap_total(self, df: pd.DataFrame, pos_col: str) -> pd.DataFrame:
        """总仓位上限约束。"""
        total = df[pos_col].sum()
        if total > self._max_total:
            scale = self._max_total / total
            df[pos_col] = df[pos_col] * scale
            logger.info(f"[PortfolioBuilder] 总仓位 {total:.1%} > {self._max_total:.0%}，等比缩仓 x{scale:.3f}")
        return df

    def _constrain_industry(self, df: pd.DataFrame) -> pd.DataFrame:
        """行业集中度约束：超限行业仅保留评分最高者。"""
        pos_col = ColumnNames.SUGGESTED_POSITION
        score_col = ColumnNames.COMPREHENSIVE_SCORE
        result = df.copy()

        while True:
            industry_positions = result.groupby(ColumnNames.INDUSTRY)[pos_col].sum()
            over_limit = industry_positions[industry_positions > self._max_industry]
            if over_limit.empty:
                break

            ind = over_limit.index[0]
            mask = result[ColumnNames.INDUSTRY] == ind
            if mask.sum() <= 1:
                idx = result.loc[mask].index[0]
                result.loc[idx, pos_col] = min(result.loc[idx, pos_col], self._max_industry)
                continue

            if score_col in result.columns:
                best_idx = result.loc[mask, score_col].idxmax()
            else:
                best_idx = result.loc[mask, pos_col].idxmax()
            zero_idx = mask & (result.index != best_idx)
            result.loc[zero_idx, pos_col] = 0.0
            best_pos = result.loc[best_idx, pos_col]
            result.loc[best_idx, pos_col] = min(best_pos, self._max_industry)
            logger.info(
                f"[PortfolioBuilder] 行业 {ind} 超限({industry_positions[ind]:.1%})，"
                f"保留 {result.loc[best_idx, ColumnNames.STOCK_CODE]}，其余清零"
            )
        return result

    # ── 换手率约束 + 交易成本 ─────────────────────────────

    def _load_previous_positions(self) -> pd.Series:
        """加载上一交易日持仓（从 app_stock_strategy_report 读取）。"""
        if self._engine is None or self._today is None:
            return pd.Series(dtype=float)

        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            cal = TradingCalendarAnalyzer()
            prev_date = cal.get_previous_trading_day(self._today)
            if prev_date is None:
                return pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)

        sql = sql_text("""
        SELECT stock_code, target_weight
        FROM app_stock_strategy_report
        WHERE archive_date = :d AND target_weight > 0
        """)
        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"d": prev_date})
            if df.empty:
                return pd.Series(dtype=float)
            return df.set_index("stock_code")["target_weight"].astype(float)
        except Exception:
            return pd.Series(dtype=float)

    def _calc_turnover(self, target: pd.Series, prev: pd.Series) -> float:
        """计算双边换手率。"""
        all_codes = target.index.union(prev.index)
        t = target.reindex(all_codes, fill_value=0.0)
        p = prev.reindex(all_codes, fill_value=0.0)
        return (t - p).abs().sum() / 2

    def _estimate_trading_cost(self, code: str, target_w: float,
                                prev_w: float) -> float:
        """估算单只股票的交易成本（占净值比例）。

        成本 = |Δw| × (佣金 × 2 + 过户费 × 2 + 印花税 × 仅卖出)
        简化：取 max(双边佣金, 印花税单向), 忽略过户费。
        """
        delta = abs(target_w - prev_w)
        if delta < 1e-8:
            return 0.0
        is_sell = target_w < prev_w
        stamp = self._stamp_tax_rate if is_sell else 0.0
        return delta * (self._commission_rate * 2 + stamp + self._transfer_fee_rate * 2)

    def _apply_turnover_and_cost(self, df: pd.DataFrame,
                                  pos_col: str) -> pd.DataFrame:
        """应用换手率约束和交易成本调整。"""
        result = df.copy()
        code_col = ColumnNames.STOCK_CODE
        if code_col not in result.columns:
            return result

        # 加载上日持仓
        prev_positions = self._load_previous_positions()
        if prev_positions.empty:
            return result  # 无上日数据，跳过

        current_target = result.set_index(code_col)[pos_col]
        turnover = self._calc_turnover(current_target, prev_positions)

        if turnover <= self._max_turnover:
            # 换手率未超限，仅计算交易成本
            total_cost = 0.0
            for code in current_target.index:
                prev_w = prev_positions.get(code, 0.0)
                cost = self._estimate_trading_cost(code, current_target[code], prev_w)
                total_cost += cost
            if total_cost > 0:
                logger.info(f"[PortfolioBuilder] 估算交易成本: {total_cost:.4f} (换手率 {turnover:.1%})")
            return result

        # 换手率超限：按比例压缩变动量
        scale = self._max_turnover / turnover
        raw_target = current_target.copy()
        all_codes = raw_target.index.union(prev_positions.index)

        compressed = {}
        for code in all_codes:
            tw = raw_target.get(code, 0.0)
            pw = prev_positions.get(code, 0.0)
            delta = tw - pw
            compressed[code] = pw + delta * scale

        # 重归一化到总仓位上限
        total = sum(compressed.values())
        if total > self._max_total:
            for code in compressed:
                compressed[code] *= self._max_total / total

        for code, w in compressed.items():
            if code in result.index:
                result.loc[code, pos_col] = w

        total_cost = sum(
            self._estimate_trading_cost(code, compressed.get(code, 0.0), prev_positions.get(code, 0.0))
            for code in compressed if compressed.get(code, 0.0) > 0
        )

        logger.info(
            f"[PortfolioBuilder] 换手率 {turnover:.1%} > {self._max_turnover:.0%}，"
            f"压缩 x{scale:.3f}，估算成本 {total_cost:.4f}"
        )
        return result
