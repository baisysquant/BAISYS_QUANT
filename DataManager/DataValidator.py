"""
数据验证工具模块

提供统一的数据质量检查和验证功能，包括：
- DataFrame 结构验证
- 数据类型检查
- 异常值检测
- 必需字段验证
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


class DataValidator:
    """
    数据验证器类

    提供多种数据质量检查方法，确保数据的完整性和准确性。
    """

    def __init__(self, logger_instance: Any = None) -> None:  # noqa: ANN401
        self.logger = logger_instance if logger_instance is not None else logger

    def validate_dataframe_not_empty(self, df: pd.DataFrame, data_name: str = "数据") -> bool:
        """
        验证 DataFrame 非空

        Args:
            df: 待验证的 DataFrame
            data_name: 数据名称，用于错误提示

        Returns:
            bool: True 表示验证通过，False 表示验证失败

        Raises:
            ValueError: 当 DataFrame 为空时抛出
        """
        if df is None:
            error_msg = f"{data_name} 为 None"
            if self.logger:
                logger.warning(f"[数据验证] {error_msg}")
            raise ValueError(error_msg)

        if not isinstance(df, pd.DataFrame):
            error_msg = f"{data_name} 不是 DataFrame 类型，实际类型: {type(df)}"
            if self.logger:
                logger.error(f"[数据验证] {error_msg}")
            raise TypeError(error_msg)

        if df.empty:
            error_msg = f"{data_name} 为空 DataFrame（无数据行）"
            if self.logger:
                logger.warning(f"[数据验证] {error_msg}")
            return False

        return True

    def validate_required_columns(
        self, df: pd.DataFrame, required_cols: list[str], data_name: str = "数据"
    ) -> tuple[bool, list[str]]:
        """
        验证 DataFrame 包含必需的列

        Args:
            df: 待验证的 DataFrame
            required_cols: 必需的列名列表
            data_name: 数据名称，用于错误提示

        Returns:
            Tuple[bool, List[str]]: (验证是否通过, 缺失的列名列表)
        """
        if df.empty:
            return False, required_cols.copy()

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_msg = f"{data_name} 缺少必需列: {missing_cols}"
            if self.logger:
                logger.warning(f"[数据验证] {error_msg}")
            return False, missing_cols

        return True, []


    def validate_price_data(
        self, df: pd.DataFrame, price_cols: list[str] | None = None, data_name: str = "价格数据"
    ) -> tuple[bool, dict[str, int]]:
        """
        验证价格数据的合理性

        检查价格列是否存在负值、零值或极端异常值。

        Args:
            df: 包含价格数据的 DataFrame
            price_cols: 需要验证的价格列名列表，默认为 ['最新价', '收盘价']
            data_name: 数据名称，用于日志记录

        Returns:
            Tuple[bool, Dict[str, int]]: (验证是否通过, 各列异常值数量)
        """
        if df.empty:
            return True, {}

        if price_cols is None:
            price_cols = ["最新价", "收盘价", "open", "high", "low", "close"]

        # 只检查实际存在的列
        existing_cols = [col for col in price_cols if col in df.columns]

        if not existing_cols:
            return True, {}

        anomaly_counts = {}
        has_anomaly = False

        for col in existing_cols:
            # 转换为数值类型
            numeric_col = pd.to_numeric(df[col], errors="coerce")

            # 统计异常值
            negative_count = (numeric_col < 0).sum()
            zero_count = (numeric_col == 0).sum()

            # 计算合理范围（使用分位数）
            if len(numeric_col.dropna()) > 0:
                q99 = numeric_col.quantile(0.99)
                extreme_count = (numeric_col > q99 * 10).sum()  # 超过99分位10倍视为极端值
            else:
                extreme_count = 0

            total_anomaly = negative_count + zero_count + extreme_count

            if total_anomaly > 0:
                anomaly_counts[col] = int(total_anomaly)
                has_anomaly = True

                if self.logger:
                    logger.warning(
                        f"[数据验证] {data_name} - {col}: "
                        f"发现 {total_anomaly} 个异常值 "
                        f"(负值:{negative_count}, 零值:{zero_count}, 极端值:{extreme_count})"
                    )

        return not has_anomaly, anomaly_counts


    def validate_data_completeness(
        self,
        df: pd.DataFrame,
        critical_cols: list[str] | None = None,
        null_threshold: float = 0.5,
        data_name: str = "数据",
    ) -> tuple[bool, dict[str, float]]:
        """
        验证数据完整性（空值比例检查）

        Args:
            df: 待验证的 DataFrame
            critical_cols: 关键列名列表，如果未指定则检查所有列
            null_threshold: 空值比例阈值（0-1），超过此比例视为不完整
            data_name: 数据名称，用于日志记录

        Returns:
            Tuple[bool, Dict[str, float]]: (验证是否通过, 各列空值比例)
        """
        if df.empty:
            return True, {}

        if critical_cols is None:
            critical_cols = df.columns.tolist()

        # 只检查实际存在的列
        existing_cols = [col for col in critical_cols if col in df.columns]

        if not existing_cols:
            return True, {}

        null_ratios = {}
        has_excessive_nulls = False

        for col in existing_cols:
            null_count = df[col].isna().sum()
            null_ratio = null_count / len(df)
            null_ratios[col] = round(float(null_ratio), 4)

            if null_ratio > null_threshold:
                has_excessive_nulls = True

                if self.logger:
                    logger.warning(
                        f"[数据验证] {data_name} - {col}: "
                        f"空值比例 {null_ratio:.2%} "
                        f"({null_count}/{len(df)})，超过阈值 {null_threshold:.0%}"
                    )

        return not has_excessive_nulls, null_ratios
