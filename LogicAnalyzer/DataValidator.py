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

    def validate_stock_code_format(
        self, df: pd.DataFrame, code_col: str = "股票代码", data_name: str = "股票数据"
    ) -> tuple[pd.DataFrame, int]:
        """
        验证并标准化股票代码格式

        检查股票代码是否为6位数字格式，自动修复常见格式问题。

        Args:
            df: 包含股票代码的 DataFrame
            code_col: 股票代码列名
            data_name: 数据名称，用于日志记录

        Returns:
            Tuple[pd.DataFrame, int]: (标准化后的DataFrame, 修复的股票数量)
        """
        if df.empty or code_col not in df.columns:
            return df, 0

        from UtilsManager.CodeNormalizer import CodeNormalizer

        original_codes = df[code_col].copy()
        df[code_col] = df[code_col].apply(CodeNormalizer.normalize)

        fixed_count = (original_codes != df[code_col]).sum()

        if fixed_count > 0 and self.logger:
            logger.info(f"[数据验证] {data_name}: 修复了 {fixed_count} 个股票代码格式")

        return df, fixed_count

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

    def validate_date_range(
        self,
        df: pd.DataFrame,
        date_col: str = "日期",
        expected_date: str | None = None,
        tolerance_days: int = 3,
        data_name: str = "时间序列数据",
    ) -> tuple[bool, str]:
        """
        验证日期范围的合理性

        Args:
            df: 包含日期列的 DataFrame
            date_col: 日期列名
            expected_date: 期望的日期（YYYYMMDD格式），如果不提供则不检查
            tolerance_days: 允许的日期偏差天数
            data_name: 数据名称，用于日志记录

        Returns:
            Tuple[bool, str]: (验证是否通过, 错误信息)
        """
        if df.empty or date_col not in df.columns:
            return True, ""

        try:
            # 尝试解析日期列
            dates = pd.to_datetime(df[date_col], errors="coerce")

            if dates.isna().all():
                error_msg = f"{data_name}: 无法解析日期列 '{date_col}'"
                if self.logger:
                    logger.warning(f"[数据验证] {error_msg}")
                return False, error_msg

            # 检查日期范围
            max_date = dates.max()

            if expected_date:
                try:
                    expected_dt = pd.to_datetime(expected_date, format="%Y%m%d")
                    date_diff = abs((max_date - expected_dt).days)

                    if date_diff > tolerance_days:
                        error_msg = (
                            f"{data_name}: 最新日期 {max_date.strftime('%Y-%m-%d')} "
                            f"与期望日期 {expected_date} 相差 {date_diff} 天 "
                            f"(允许偏差: {tolerance_days} 天)"
                        )
                        if self.logger:
                            logger.warning(f"[数据验证] {error_msg}")
                        return False, error_msg
                except Exception as e:
                    if self.logger:
                        logger.warning(f"[数据验证] {data_name}: 日期解析失败 - {e}")

            return True, ""

        except Exception as e:
            error_msg = f"{data_name}: 日期验证失败 - {e}"
            if self.logger:
                logger.error(f"[数据验证] {error_msg}")
            return False, error_msg

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

    def comprehensive_validate(
        self,
        df: pd.DataFrame,
        data_name: str = "数据",
        required_cols: list[str] | None = None,
        price_cols: list[str] | None = None,
        check_completeness: bool = True,
    ) -> dict[str, Any]:
        """
        综合验证 DataFrame 的质量

        执行多项验证检查，返回详细的验证报告。

        Args:
            df: 待验证的 DataFrame
            data_name: 数据名称
            required_cols: 必需列名列表
            price_cols: 价格列名列表
            check_completeness: 是否检查数据完整性

        Returns:
            Dict: 验证报告，包含以下字段：
                - passed: bool, 整体验证是否通过
                - checks: Dict, 各项检查结果
                - warnings: List[str], 警告信息列表
        """
        report = {"passed": True, "checks": {}, "warnings": []}

        # 1. 非空验证
        try:
            not_empty = self.validate_dataframe_not_empty(df, data_name)
            report["checks"]["not_empty"] = not_empty
        except (ValueError, TypeError) as e:
            report["passed"] = False
            report["checks"]["not_empty"] = False
            report["warnings"].append(str(e))
            return report

        # 2. 必需列验证
        if required_cols:
            cols_valid, missing_cols = self.validate_required_columns(df, required_cols, data_name)
            report["checks"]["required_columns"] = cols_valid
            if missing_cols:
                report["warnings"].append(f"缺少列: {missing_cols}")
                report["passed"] = False

        # 3. 价格数据验证
        if price_cols:
            price_valid, anomaly_counts = self.validate_price_data(df, price_cols, data_name)
            report["checks"]["price_data"] = price_valid
            if anomaly_counts:
                report["warnings"].append(f"价格异常: {anomaly_counts}")

        # 4. 数据完整性验证
        if check_completeness:
            completeness_valid, null_ratios = self.validate_data_completeness(df, data_name=data_name)
            report["checks"]["completeness"] = completeness_valid
            if not completeness_valid:
                report["warnings"].append("存在过多空值")

        return report


# 全局验证器实例（可选）
_global_validator: DataValidator | None = None


def get_validator() -> DataValidator:
    """
    获取全局数据验证器实例

    Returns:
        DataValidator: 数据验证器实例
    """
    global _global_validator
    if _global_validator is None:
        _global_validator = DataValidator()
    return _global_validator
