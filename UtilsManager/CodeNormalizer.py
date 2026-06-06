"""
股票代码标准化工具类

提供统一的股票代码格式转换功能，处理各种格式的股票代码，
包括带市场前缀/后缀的格式，统一转换为6位纯数字格式。
"""

import re

import pandas as pd


class CodeNormalizer:
    """
    股票代码标准化工具类

    职责：
    - 统一标准化股票代码为6位纯数字格式
    - 处理各种格式的股票代码（SH600000、600000.SH、600000等）
    - 提供批量标准化功能

    Examples:
        >>> CodeNormalizer.normalize('SH600000')
        '600000'
        >>> CodeNormalizer.normalize('000001.SZ')
        '000001'
        >>> CodeNormalizer.normalize('600000')
        '600000'
    """

    @staticmethod
    def normalize(code: str) -> str:
        """
        统一标准化股票代码为6位纯数字格式

        处理各种格式的股票代码，包括：
        - SH600000, SZ000001 (带市场前缀)
        - 600000.SH, 000001.SZ (带市场后缀)
        - 600000, 000001 (纯数字)
        - 其他非标准格式

        Args:
            code: 原始股票代码字符串

        Returns:
            str: 标准化后的6位数字股票代码，失败时返回空字符串

        Examples:
            >>> CodeNormalizer.normalize('SH600000')
            '600000'
            >>> CodeNormalizer.normalize('000001.SZ')
            '000001'
            >>> CodeNormalizer.normalize('600000')
            '600000'
            >>> CodeNormalizer.normalize(None)
            ''
        """
        if pd.isna(code) or code is None:
            return ""

        code_str = str(code).strip()

        # 尝试提取6位数字
        match = re.search(r"(\d{6})", code_str)
        if match:
            return match.group(1)

        # 如果没有找到6位数字，尝试补零
        digits_only = re.sub(r"\D", "", code_str)
        if len(digits_only) <= 6:
            return digits_only.zfill(6)

        return code_str

    @staticmethod
    def normalize_series(series: pd.Series) -> pd.Series:
        """
        批量标准化Series中的股票代码

        Args:
            series: 包含股票代码的pandas Series

        Returns:
            pd.Series: 标准化后的Series
        """
        return series.apply(CodeNormalizer.normalize)

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame, column: str = "股票代码") -> pd.DataFrame:
        """
        标准化DataFrame中指定列的股票代码

        Args:
            df: 包含股票代码的DataFrame
            column: 股票代码列名，默认为"股票代码"

        Returns:
            pd.DataFrame: 标准化后的DataFrame（原地修改）
        """
        if column in df.columns:
            df[column] = df[column].apply(CodeNormalizer.normalize)
        return df

    @staticmethod
    def add_market_prefix(code: str) -> str:
        """
        添加市场前缀（反向操作，委托给 ShareCodeFormatMgr）。
        6位纯数字 → sh/sz/bj + 6位数字。

        Args:
            code: 6位纯数字股票代码

        Returns:
            str: 带市场前缀的股票代码（如 sh600000, sz000001）
        """
        from DataManager.ShareCodeFormatMgr import format_stock_code
        return format_stock_code(code)
