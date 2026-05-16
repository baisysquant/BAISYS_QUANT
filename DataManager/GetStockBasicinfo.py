import tushare as ts
import pandas as pd
from typing import List, Dict, Optional
import time
import logging
from datetime import datetime
import requests
import pytz
import json
import os
from datetime import timedelta
import akshare as ak
from DataManager.DatabaseWriter import QuantDBManager
from ConfigParser import Config
from DataManager.CalendarManager import TradingCalendarAnalyzer


class StockBasicInfoService:
    """股票基本信息业务服务类"""

    def __init__(self, config_parser: Config):
        self.config_parser = config_parser
        self.tushare_token = config_parser.TUSHARE_TOKEN
        self.system_config = self._get_system_config_from_attributes(config_parser)
        self.logger = self._setup_logger()

        # 设置tushare token
        ts.set_token(self.tushare_token)
        self.pro = ts.pro_api()

        # 初始化数据库管理器
        self.db_manager = None

        # 初始化交易日历分析器
        self.trading_calendar = TradingCalendarAnalyzer()

    def _get_system_config_from_attributes(self, config_parser) -> dict:
        """从Config对象的属性构建系统配置字典"""
        return {
            'max_workers': config_parser.MAX_WORKERS,
            'data_fetch_retries': config_parser.DATA_FETCH_RETRIES,
            'data_fetch_delay': config_parser.DATA_FETCH_DELAY
        }

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_database(self):
        """初始化数据库连接"""
        db_config = {
            'user': self.config_parser.DB_USER,
            'password': self.config_parser.DB_PASSWORD,
            'host': self.config_parser.DB_HOST,
            'port': self.config_parser.DB_PORT,
            'database': self.config_parser.DB_NAME
        }
        self.db_manager = QuantDBManager(
            db_config['user'],
            db_config['password'],
            db_config['host'],
            db_config['port'],
            db_config['database']
        )

    def _get_latest_data_date(self) -> str:
        """获取表中最新的数据日期"""
        if not self.db_manager:
            self._initialize_database()

        try:
            # 使用QuantDBManager新增的get_latest_record_date方法
            return self.db_manager.get_latest_record_date('stock_basic_info', 'create_time')

        except Exception as e:
            self.logger.error(f"获取最新数据日期失败: {str(e)}")
            return ""

    def _is_current_trading_day_data_up_to_date(self) -> bool:
        """检查当前交易日的数据是否已经是最新的"""
        # 使用最新交易日而不是系统时间
        current_trading_day_str = self.trading_calendar.get_last_trading_day().replace('-', '')  # 转换为 YYYYMMDD 格式
        latest_date = self._get_latest_data_date()

        if not latest_date:
            self.logger.info("表中没有数据")
            return False

        if latest_date == current_trading_day_str:
            self.logger.info(f"数据已经是当前交易日({current_trading_day_str})，无需更新")
            return True
        else:
            self.logger.info(f"表中数据日期为{latest_date}，不是当前交易日({current_trading_day_str})，需要更新")
            return False

    def fetch_stock_basic_info(self, exchange: str = '',
                               list_status: str = 'L') -> pd.DataFrame:
        """
        获取股票基本信息（仅包含四个字段）

        Parameters:
        - exchange: 交易所 SSE上交所 SZSE深交所 BSE北交所
        - list_status: 上市状态 L上市 D退市 P暂停上市
        """
        max_retries = self.system_config['data_fetch_retries']
        delay = self.system_config['data_fetch_delay']

        # 只获取需要的字段
        fields = 'ts_code,symbol,name,industry,market'

        for attempt in range(max_retries):
            try:
                df = self.pro.stock_basic(
                    exchange=exchange,
                    list_status=list_status,
                    fields=fields
                )

                self.logger.info(f"成功获取 {len(df)} 条股票基本信息")

                # 添加create_time列，使用最新交易日作为数据归属日期
                latest_trading_day = self.trading_calendar.get_last_trading_day()
                df['create_time'] = latest_trading_day

                return df

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"第 {attempt + 1} 次请求失败: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    raise
            except Exception as e:
                self.logger.error(f"获取股票基本信息失败: {str(e)}")
                raise

    def sync_all_stock_basic_info(self) -> bool:
        """同步所有股票基本信息 - 使用QuantDBManager的safe_insert_data方法"""
        try:
            # 确保数据库连接
            if not self.db_manager:
                self._initialize_database()

            # 检查当前交易日数据是否已经是最新
            if self._is_current_trading_day_data_up_to_date():
                self.logger.info("当前交易日的数据已是最新，跳过更新")
                return True

            # 获取数据 - 如果获取失败会抛出异常，不会继续执行后续操作
            df = self.fetch_stock_basic_info()

            if df.empty:
                self.logger.warning("获取到的数据为空")
                return False

            # 获取最新交易日用于记录日志
            latest_trading_day = self.trading_calendar.get_last_trading_day()
            self.logger.info(f"使用最新交易日: {latest_trading_day} 作为数据归属日期")

            # 使用QuantDBManager的safe_insert_data方法，实现幂等写入
            # 此方法会先删除当天数据再插入新数据，利用了_fast_pg_copy的高性能
            self.db_manager.safe_insert_data(df, 'stock_basic_info', 'create_time', latest_trading_day)

            # 显示统计信息
            industries = df['industry'].value_counts().head(10)
            self.logger.info(f"主要行业分布:\n{industries}")

            markets = df['market'].value_counts()
            self.logger.info(f"市场分布:\n{markets}")

            return True

        except Exception as e:
            self.logger.error(f"同步股票基本信息失败: {str(e)}")
            return False

    def get_stock_count(self) -> int:
        """获取股票总数"""
        if not self.db_manager:
            self._initialize_database()

        try:
            return self.db_manager.get_table_count('stock_basic_info')
        except Exception as e:
            self.logger.error(f"获取股票总数失败: {str(e)}")
            return 0


# 使用示例
if __name__ == "__main__":
    # 初始化配置解析器
    config_parser = Config("config.ini")

    # 创建服务实例
    service = StockBasicInfoService(config_parser)

    # 同步所有股票基本信息（只有当数据不是今天的时候才执行）
    success = service.sync_all_stock_basic_info()

    if success:
        count = service.get_stock_count()
        print(f"股票基本信息同步成功！当前数据库中共有 {count} 条记录")
    else:
        print("股票基本信息同步失败或跳过（今日已有最新数据）！")