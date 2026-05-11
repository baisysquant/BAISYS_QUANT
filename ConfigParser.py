# DataManager/Config.py
"""
配置管理模块

负责读取和验证 config.ini 配置文件，提供全局配置访问接口。
支持以下配置分组：
- DATABASE: 数据库连接配置
- SYSTEM: 系统运行参数
- FUND_FLOW: 资金流分析配置
- TECHNICAL_INDICATORS: 技术指标参数
- DATA_SYNC: 数据同步配置
"""

import os
import configparser
from pathlib import Path
from typing import List, Tuple, Optional

class Config:
    """
    配置管理器类
    
    负责加载、解析和验证配置文件，提供统一的配置访问接口。
    所有配置项在初始化时读取并验证，后续通过属性访问。
    
    Attributes:
        DB_USER: 数据库用户名
        DB_PASSWORD: 数据库密码
        DB_HOST: 数据库主机地址
        DB_PORT: 数据库端口
        DB_NAME: 数据库名称
        HOME_DIRECTORY: 主目录路径
        TEMP_DATA_DIRECTORY: 临时数据目录
        MAX_WORKERS: 最大线程数
        FUND_FLOW_PERIODS: 资金流周期列表（必须为3个）
        MACD_STANDARD_PARAMS: MACD标准周期 (12,26,9)
        MACD_SECOND_PARAMS: MACD第二周期（必填）
        ENABLE_MACD_SECOND: 是否启用MACD第二周期（始终为True）
    """
    
    def __init__(self, config_file: str = "config.ini") -> None:
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为 'config.ini'
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置参数验证失败
        """
        self.config_file = config_file
        self._validate_config_file()
        self._load_config()
        self._ensure_directories()

    def _validate_config_file(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件未找到: {os.path.abspath(self.config_file)}")

    def _load_config(self):

        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')


            # 读取数据库配置
        db = config['DATABASE']
        self.DB_USER = db.get('user')
        self.DB_PASSWORD = db.get('password')
        self.DB_HOST = db.get('host')
        self.DB_PORT = db.get('port')
        self.DB_NAME = db.get('db_name')

            # 读取 SYSTEM 配置
        system = config['SYSTEM']
        home_dir = system.get('HOME_DIRECTORY', '~/Downloads/CoreNews_Reports')
        self.HOME_DIRECTORY = os.path.expanduser(home_dir)
        temp_dir = system.get('TEMP_DATA_DIR', 'ShareData')
        self.TEMP_DATA_DIRECTORY = os.path.join(self.HOME_DIRECTORY, temp_dir)


        self.MAX_WORKERS = system.getint('MAX_WORKERS', fallback=15)
        self.DATA_FETCH_RETRIES = system.getint('DATA_FETCH_RETRIES', fallback=3)
        self.DATA_FETCH_DELAY = system.getint('DATA_FETCH_DELAY', fallback=5)

            # 其他配置...
        self.CODE_ALIASES = {'代码': '股票代码', '证券代码': '股票代码', '股票代码': '股票代码'}
        self.NAME_ALIASES = {'名称': '股票简称', '股票名称': '股票简称', '股票简称': '股票简称', '简称': '股票简称'}
        self.PRICE_ALIASES = {'最新价': '最新价', '现价': '最新价', '当前价格': '最新价', '今收盘': '最新价',
                              '收盘': '最新价', '收盘价': '最新价'}


        self.TUSHARE_TOKEN = db.get('tushare_token')  # 如果没有配置，默认为 None
        if not self.TUSHARE_TOKEN:
            raise ValueError("配置文件中缺少 'tushare_token'，请在 [DATABASE] 节点下添加。")

        log = config['LOGGING']
        self.LOG_LEVEL = log.get('LOG_LEVEL', 'INFO')
        self.LOG_DIR = os.path.join(self.HOME_DIRECTORY, log.get('LOG_DIR', 'Logs'))

        # 读取多头排列评分系统配置
        mha = config['MULTI_HEAD_ARRANGEMENT']
        self.FULL_BULL_THRESHOLD = mha.getint('FULL_BULL_THRESHOLD', fallback=85)
        self.TREND_ACCELERATION_THRESHOLD = mha.getint('TREND_ACCELERATION_THRESHOLD', fallback=65)
        self.TREND_OSCILLATION_THRESHOLD = mha.getint('TREND_OSCILLATION_THRESHOLD', fallback=45)
        self.TREND_WATCH_THRESHOLD = mha.getint('TREND_WATCH_THRESHOLD', fallback=45)

        # 读取弱势股过滤规则配置
        fr = config['FILTER_RULES']
        self.ENABLE_WEAK_STOCK_FILTER = fr.getboolean('ENABLE_WEAK_STOCK_FILTER', fallback=True)
        exempt_str = fr.get('EXEMPT_LEVELS', fallback='完全主升,趋势加速')
        self.EXEMPT_LEVELS = [level.strip() for level in exempt_str.split(',')]

        # 读取资金流分析配置
        ff = config['FUND_FLOW']
        periods_str = ff.get('FUND_FLOW_PERIODS', fallback='5,10,20')
        
        # 解析并验证资金流周期配置
        raw_periods = [int(p.strip()) for p in periods_str.split(',')]
        
        # akshare接口支持的周期（严格限制）
        VALID_FUND_FLOW_PERIODS = {3, 5, 10, 20}
        
        # 验证：必须是三个周期
        if len(raw_periods) != 3:
            raise ValueError(
                f"错误：资金流周期必须设置为三个参数，当前设置了 {len(raw_periods)} 个。\n"
                f"允许的组合：\n"
                f"  - 3,5,10   （短中周期组合，推荐短线）\n"
                f"  - 3,5,20   （短长周期组合）\n"
                f"  - 5,10,20  （中长周期组合，默认，推荐中线）\n"
                f"  - 3,10,20  （分散周期组合）"
            )
        
        # 验证：每个值必须在白名单内
        invalid_periods = [p for p in raw_periods if p not in VALID_FUND_FLOW_PERIODS]
        if invalid_periods:
            raise ValueError(
                f"错误：资金流周期包含无效值 {invalid_periods}。\n"
                f"仅支持以下周期：{sorted(VALID_FUND_FLOW_PERIODS)}"
            )
        
        # 验证：必须是允许的组合之一
        ALLOWED_COMBINATIONS = [
            (3, 5, 10),
            (3, 5, 20),
            (5, 10, 20),
            (3, 10, 20),
        ]
        
        sorted_periods = tuple(sorted(raw_periods))
        if sorted_periods not in ALLOWED_COMBINATIONS:
            raise ValueError(
                f"错误：资金流周期组合 {raw_periods} 不被允许。\n"
                f"允许的组合（顺序不限）：\n"
                f"  - 3,5,10   （短中周期组合，推荐短线）\n"
                f"  - 3,5,20   （短长周期组合）\n"
                f"  - 5,10,20  （中长周期组合，默认，推荐中线）\n"
                f"  - 3,10,20  （分散周期组合）"
            )
        
        self.FUND_FLOW_PERIODS = raw_periods
        self.MOMENTUM_WINDOW = ff.getint('MOMENTUM_WINDOW', fallback=5)

        # 读取技术指标信号配置
        ti = config['TECHNICAL_INDICATORS']
        
        # MACD标准周期（强制保留，不可修改）
        macd_std_str = ti.get('MACD_STANDARD_PARAMS', fallback='12,26,9')
        self.MACD_STANDARD_PARAMS = tuple(int(p.strip()) for p in macd_std_str.split(','))
        
        # 验证标准周期必须是(12,26,9)
        if self.MACD_STANDARD_PARAMS != (12, 26, 9):
            import warnings
            warnings.warn(
                f"警告：MACD标准周期被修改为 {self.MACD_STANDARD_PARAMS}，"
                f"已强制恢复为标准值 (12, 26, 9)。这是业界公认的经典参数。",
                UserWarning
            )
            self.MACD_STANDARD_PARAMS = (12, 26, 9)
        
        # MACD第二周期（必填）
        macd_second_str = ti.get('MACD_SECOND_PARAMS', fallback='6,13,5')
        self.MACD_SECOND_PARAMS = tuple(int(p.strip()) for p in macd_second_str.split(','))
        
        # 验证第二周期参数有效性
        if self.MACD_SECOND_PARAMS == (0, 0, 0):
            raise ValueError(
                "错误：MACD第二周期参数不能设置为(0,0,0)。"
                "第二周期为必填项，请设置有效的MACD参数，如(6,13,5)或(24,52,18)。"
            )
        
        # 验证参数合理性（快线 < 慢线）
        fast, slow, signal = self.MACD_SECOND_PARAMS
        if fast >= slow:
            import warnings
            warnings.warn(
                f"警告：MACD第二周期参数不合理（快线{fast} >= 慢线{slow}），"
                f"可能导致技术指标计算异常。建议调整为快线 < 慢线。",
                UserWarning
            )
        
        # 第二周期始终启用
        self.ENABLE_MACD_SECOND = True
        kdj_str = ti.get('KDJ_PARAMS', fallback='9,3,3')
        self.KDJ_PARAMS = tuple(int(p.strip()) for p in kdj_str.split(','))
        self.RSI_PERIOD = ti.getint('RSI_PERIOD', fallback=14)
        self.CCI_PERIOD = ti.getint('CCI_PERIOD', fallback=14)
        boll_str = ti.get('BOLL_PARAMS', fallback='20,2')
        self.BOLL_PARAMS = tuple(int(p.strip()) for p in boll_str.split(','))

        # 读取数据同步配置
        ds = config['DATA_SYNC']
        self.SYNC_RETRIES = ds.getint('SYNC_RETRIES', fallback=3)
        self.SYNC_INTERVAL = ds.getint('SYNC_INTERVAL', fallback=2)
        self.CACHE_TTL_HOURS = ds.getint('CACHE_TTL_HOURS', fallback=24)
        self.ENABLE_CACHE = ds.getboolean('ENABLE_CACHE', fallback=True)

        for key, val in self.__dict__.items():
            if val is None:
                raise ValueError(f"配置项 '{key}' 未设置，请在 {self.config_file} 中检查。")

    def _ensure_directories(self):
        dirs = [self.HOME_DIRECTORY, self.TEMP_DATA_DIRECTORY, self.LOG_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def get_db_connection_string(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
