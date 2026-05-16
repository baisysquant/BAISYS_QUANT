"""
价格提取工具类

从K线数据中提取最新价格的工具，支持多种日期和价格列名。
"""

import pandas as pd
from typing import Optional, List


class PriceExtractor:
    """
    价格提取工具类
    
    职责：
    - 从K线数据中提取每个股票的最新收盘价
    - 支持多种日期和价格列名
    - 提供股票代码标准化功能
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'symbol': ['SH600000', 'SZ000001'],
        ...     'trade_date': ['2024-01-01', '2024-01-01'],
        ...     'close': [10.5, 20.3]
        ... })
        >>> result = PriceExtractor.extract_latest_prices(df)
        >>> print(result)
             股票代码   最新价
        0  600000  10.5
        1  000001  20.3
    """
    
    # 支持的日期列名候选
    DATE_COLUMN_CANDIDATES = [
        "trade_date", "date", "日期", "datetime", "Date", "TRADE_DATE"
    ]
    
    # 支持的代码列名候选
    CODE_COLUMN_CANDIDATES = [
        "symbol", "ts_code", "code", "股票代码"
    ]
    
    # 支持的价格列名候选
    PRICE_COLUMN_CANDIDATES = [
        "close", "收盘价", "price", "latest_price", "最新价"
    ]
    
    @staticmethod
    def extract_latest_prices(
        hist_df: pd.DataFrame,
        date_col: Optional[str] = None,
        price_col: Optional[str] = None,
        code_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从K线数据中提取每个股票的最新收盘价
        
        Args:
            hist_df: K线数据DataFrame
            date_col: 日期列名，如果为None则自动检测
            price_col: 价格列名，如果为None则自动检测
            code_col: 代码列名，如果为None则自动检测
            
        Returns:
            pd.DataFrame: 包含股票代码和最新价的DataFrame
                         列名为：['股票代码', '最新价']
            
        Raises:
            ValueError: 当无法找到必需的列时
        """
        if hist_df.empty:
            return pd.DataFrame(columns=["股票代码", "最新价"])
        
        # 自动检测列名
        if date_col is None:
            date_col = PriceExtractor._detect_column(hist_df.columns, PriceExtractor.DATE_COLUMN_CANDIDATES)
            if date_col is None:
                raise ValueError(f"无法在K线数据中找到日期列。支持的列名: {PriceExtractor.DATE_COLUMN_CANDIDATES}")
        
        if code_col is None:
            code_col = PriceExtractor._detect_column(hist_df.columns, PriceExtractor.CODE_COLUMN_CANDIDATES)
            if code_col is None:
                raise ValueError(f"无法在K线数据中找到股票代码列。支持的列名: {PriceExtractor.CODE_COLUMN_CANDIDATES}")
        
        if price_col is None:
            price_col = PriceExtractor._detect_column(hist_df.columns, PriceExtractor.PRICE_COLUMN_CANDIDATES)
            if price_col is None:
                raise ValueError(f"无法在K线数据中找到价格列。支持的列名: {PriceExtractor.PRICE_COLUMN_CANDIDATES}")
        
        # 获取每个股票的最新一条记录（按日期排序）
        latest_records = hist_df.sort_values(date_col).groupby(code_col).tail(1)
        
        # 提取股票代码和收盘价
        latest_prices = latest_records[[code_col, price_col]].copy()
        latest_prices.columns = ["股票代码", "最新价"]
        
        # 标准化股票代码为6位纯数字
        from UtilsManager.CodeNormalizer import CodeNormalizer
        latest_prices["股票代码"] = CodeNormalizer.normalize_series(latest_prices["股票代码"])
        
        return latest_prices
    
    @staticmethod
    def _detect_column(columns: pd.Index, candidates: List[str]) -> Optional[str]:
        """
        从候选列表中检测第一个存在的列名
        
        Args:
            columns: DataFrame的列索引
            candidates: 候选列名列表
            
        Returns:
            Optional[str]: 找到的列名，如果都没找到返回None
        """
        for col in candidates:
            if col in columns:
                return col
        return None
    
    @staticmethod
    def extract_with_fallback(
        hist_df: pd.DataFrame,
        fallback_price: float = 0.0
    ) -> pd.DataFrame:
        """
        提取最新价格，失败时返回默认值
        
        Args:
            hist_df: K线数据DataFrame
            fallback_price: 失败时的默认价格
            
        Returns:
            pd.DataFrame: 包含股票代码和最新价的DataFrame
        """
        try:
            return PriceExtractor.extract_latest_prices(hist_df)
        except (ValueError, KeyError, Exception):
            # 提取所有唯一的股票代码
            code_col = PriceExtractor._detect_column(
                hist_df.columns, 
                PriceExtractor.CODE_COLUMN_CANDIDATES
            )
            if code_col:
                from UtilsManager.CodeNormalizer import CodeNormalizer
                unique_codes = hist_df[code_col].unique()
                normalized_codes = [CodeNormalizer.normalize(code) for code in unique_codes]
                
                result_df = pd.DataFrame({
                    "股票代码": normalized_codes,
                    "最新价": fallback_price
                })
                return result_df
            else:
                return pd.DataFrame(columns=["股票代码", "最新价"])
