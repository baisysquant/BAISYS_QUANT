"""
组合构建器

在 filter_weak_stocks 后重新归一化仓位权重，施加：
  - 行业集中度上限（保留行业内评分最高者或等比缩）
  - 总仓位上限（等比缩仓至 100% 以内）
  - 输出"目标权重"列
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from DataManager.ColumnNames import ColumnNames


class PortfolioBuilder:
    """将因子评分转化为目标仓位权重的组合构建器。

    输入：已过滤弱势股的 consolidated_report（含建议仓位比例）
    输出：添加"目标权重"列，更新"建议仓位比例"为组合约束后值
    """

    def __init__(self, config: Any) -> None:  # noqa: ANN401
        self.config = config
        sizing = getattr(config, "POSITION_SIZING", None) or {}
        self._max_single = sizing.get("max_single_position", 0.33)
        self._max_industry = sizing.get("max_industry_exposure", 0.30)
        self._max_total = sizing.get("max_total_exposure", 1.0)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行组合构建全流程。

        Args:
            df: 过滤弱势股后的 DataFrame（含 建议仓位比例 列）。

        Returns:
            更新了 建议仓位比例 + 新增 目标权重 列的 DataFrame。
        """
        if df.empty:
            df[ColumnNames.SUGGESTED_POSITION] = np.nan
            df["目标权重"] = np.nan
            return df

        result = df.copy()

        # Step 1: 清零负值/空值
        pos_col = ColumnNames.SUGGESTED_POSITION
        result[pos_col] = pd.to_numeric(result.get(pos_col, 0), errors="coerce").fillna(0).clip(0)

        # Step 2: 行业集中度约束
        if ColumnNames.INDUSTRY in result.columns:
            result = self._constrain_industry(result)

        # Step 3: 单票上限约束
        result[pos_col] = result[pos_col].clip(0, self._max_single)

        # Step 4: 总仓位上限约束
        total = result[pos_col].sum()
        if total > self._max_total:
            scale = self._max_total / total
            result[pos_col] = result[pos_col] * scale
            logger.info(f"[PortfolioBuilder] 总仓位 {total:.1%} > {self._max_total:.0%}，等比缩仓 x{scale:.3f}")

        # Step 5: 输出"目标权重"列（最终组合权重）
        result["目标权重"] = result[pos_col]

        filled = (result["目标权重"] > 0).sum()
        logger.info(f"[PortfolioBuilder] 组合构建完成：{filled}/{len(result)} 只持仓，总仓位 {result['目标权重'].sum():.1%}")
        return result

    def _constrain_industry(self, df: pd.DataFrame) -> pd.DataFrame:
        """行业集中度约束：超限行业仅保留评分最高者。"""
        pos_col = ColumnNames.SUGGESTED_POSITION
        score_col = ColumnNames.COMPREHENSIVE_SCORE

        industry_positions = df.groupby(ColumnNames.INDUSTRY)[pos_col].sum()
        over_limit = industry_positions[industry_positions > self._max_industry]

        for ind in over_limit.index:
            mask = df[ColumnNames.INDUSTRY] == ind
            if mask.sum() <= 1:
                continue
            # 保留评分最高的，其余清零
            if score_col in df.columns:
                best_idx = df.loc[mask, score_col].idxmax()
            else:
                best_idx = df.loc[mask, pos_col].idxmax()
            zero_idx = mask & (df.index != best_idx)
            df.loc[zero_idx, pos_col] = 0.0
            best_pos = df.loc[best_idx, pos_col]
            df.loc[best_idx, pos_col] = min(best_pos, self._max_industry)
            logger.info(
                f"[PortfolioBuilder] 行业 {ind} 超限({industry_positions[ind]:.1%})，"
                f"保留 {df.loc[best_idx, ColumnNames.STOCK_CODE]}，其余清零"
            )
        return df
