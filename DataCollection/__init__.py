"""
数据收集模块

负责从各种数据源（akshare、tushare、数据库等）获取原始数据。
包含历史数据引擎、交易日历管理、股票基本信息获取等功能。
"""

from DataCollection.HistDataEngine import StockSyncEngine
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from DataCollection.GetStockBasicinfo import StockBasicInfoService

__all__ = [
    'StockSyncEngine',
    'TradingCalendarAnalyzer',
    'StockBasicInfoService'
]
