# DataManager/Config.py
"""
配置管理模块（使用 Pydantic 重构）

负责读取和验证 config.ini 配置文件，提供全局配置访问接口。
支持以下配置分组：
- DATABASE: 数据库连接配置
- SYSTEM: 系统运行参数
- FUND_FLOW: 资金流分析配置
- TECHNICAL_INDICATORS: 技术指标参数
- DATA_SYNC: 数据同步配置

使用 Pydantic 带来的优势：
- 开箱即用的类型安全
- 自动类型转换（字符串→int/list/tuple）
- 优雅的数据校验（使用 @field_validator）
- 支持环境变量覆盖
"""

import os
import warnings

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_aliases(alias_str: str) -> dict[str, str]:
    """解析别名字符串为字典，格式：'别名1=目标,别名2=目标'"""
    aliases = {}
    for pair in alias_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            aliases[key.strip()] = value.strip()
    return aliases


class DatabaseConfig(BaseModel):
    """数据库配置模型"""

    user: str
    password: str
    host: str
    port: str
    db_name: str
    main_board_only: bool = Field(default=False)


class SystemConfig(BaseModel):
    """系统配置模型"""

    HOME_DIRECTORY: str = Field(default="~/Downloads/CoreNews_Reports")
    TEMP_DATA_DIR: str = Field(default=".")
    MAX_WORKERS: int = Field(default=15, ge=1)
    DATA_FETCH_RETRIES: int = Field(default=3, ge=1)
    DATA_FETCH_DELAY: int = Field(default=5, ge=1)
    SIGNAL_PROCESSING_PROCESSES: int = Field(default=2, ge=1)

    @field_validator("HOME_DIRECTORY")
    @classmethod
    def expand_home(cls, v: str) -> str:
        return os.path.expanduser(v)


class LoggingConfig(BaseModel):
    """日志配置模型"""

    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default="Logs")


class MultiHeadArrangementConfig(BaseModel):
    """多头排列评分系统配置"""

    FULL_BULL_THRESHOLD: int = Field(default=85, ge=0, le=100)
    TREND_ACCELERATION_THRESHOLD: int = Field(default=65, ge=0, le=100)
    TREND_OSCILLATION_THRESHOLD: int = Field(default=45, ge=0, le=100)
    TREND_WATCH_THRESHOLD: int = Field(default=45, ge=0, le=100)
    MOVING_AVERAGE_PERIODS: list[int] = Field(default=[5, 10, 20, 30, 60])

    @field_validator("MOVING_AVERAGE_PERIODS", mode="before")
    @classmethod
    def parse_periods(cls, v):
        if isinstance(v, str):
            return [int(p.strip()) for p in v.split(",")]
        return v


class FilterRulesConfig(BaseModel):
    """弱势股过滤规则配置"""

    ENABLE_WEAK_STOCK_FILTER: bool = Field(default=True)
    EXEMPT_LEVELS: list[str] = Field(default=["完全主升", "趋势加速"])

    @field_validator("EXEMPT_LEVELS", mode="before")
    @classmethod
    def parse_exempt_levels(cls, v):
        if isinstance(v, str):
            return [level.strip() for level in v.split(",")]
        return v


class FundFlowConfig(BaseModel):
    """资金流分析配置"""

    FUND_FLOW_PERIODS: list[int] = Field(default=[5, 10, 20])

    @field_validator("FUND_FLOW_PERIODS", mode="before")
    @classmethod
    def parse_periods(cls, v):
        if isinstance(v, str):
            return [int(p.strip()) for p in v.split(",")]
        return v

    @field_validator("FUND_FLOW_PERIODS")
    @classmethod
    def validate_periods(cls, v):
        VALID_FUND_FLOW_PERIODS = {3, 5, 10, 20}
        ALLOWED_COMBINATIONS = [
            (3, 5, 10),
            (3, 5, 20),
            (5, 10, 20),
            (3, 10, 20),
        ]

        if len(v) != 3:
            raise ValueError(
                f"错误：资金流周期必须设置为三个参数，当前设置了 {len(v)} 个。\n"
                f"允许的组合：\n"
                f"  - 3,5,10   （短中周期组合，推荐短线）\n"
                f"  - 3,5,20   （短长周期组合）\n"
                f"  - 5,10,20  （中长周期组合，默认，推荐中线）\n"
                f"  - 3,10,20  （分散周期组合）"
            )

        invalid_periods = [p for p in v if p not in VALID_FUND_FLOW_PERIODS]
        if invalid_periods:
            raise ValueError(
                f"错误：资金流周期包含无效值 {invalid_periods}。\n仅支持以下周期：{sorted(VALID_FUND_FLOW_PERIODS)}"
            )

        sorted_periods = tuple(sorted(v))
        if sorted_periods not in ALLOWED_COMBINATIONS:
            raise ValueError(
                f"错误：资金流周期组合 {v} 不被允许。\n"
                f"允许的组合（顺序不限）：\n"
                f"  - 3,5,10   （短中周期组合，推荐短线）\n"
                f"  - 3,5,20   （短长周期组合）\n"
                f"  - 5,10,20  （中长周期组合，默认，推荐中线）\n"
                f"  - 3,10,20  （分散周期组合）"
            )

        return v


class TechnicalIndicatorsConfig(BaseModel):
    """技术指标信号配置"""

    MACD_STANDARD_PARAMS: tuple[int, int, int] = Field(default=(12, 26, 9))
    MACD_SECOND_PARAMS: tuple[int, int, int] = Field(default=(6, 13, 5))

    @field_validator("MACD_STANDARD_PARAMS", mode="before")
    @classmethod
    def parse_standard_params(cls, v):
        if isinstance(v, str):
            return tuple(int(p.strip()) for p in v.split(","))
        return v

    @field_validator("MACD_STANDARD_PARAMS")
    @classmethod
    def validate_standard_params(cls, v):
        if v != (12, 26, 9):
            warnings.warn(
                f"警告：MACD标准周期被修改为 {v}，已强制恢复为标准值 (12, 26, 9)。这是业界公认的经典参数。", UserWarning
            )
            return (12, 26, 9)
        return v

    @field_validator("MACD_SECOND_PARAMS", mode="before")
    @classmethod
    def parse_second_params(cls, v):
        if isinstance(v, str):
            return tuple(int(p.strip()) for p in v.split(","))
        return v

    @field_validator("MACD_SECOND_PARAMS")
    @classmethod
    def validate_second_params(cls, v):
        if v == (0, 0, 0):
            raise ValueError(
                "错误：MACD第二周期参数不能设置为(0,0,0)。"
                "第二周期为必填项，请设置有效的MACD参数，如(6,13,5)或(24,52,18)。"
            )

        fast, slow, signal = v
        if fast >= slow:
            warnings.warn(
                f"警告：MACD第二周期参数不合理（快线{fast} >= 慢线{slow}），"
                f"可能导致技术指标计算异常。建议调整为快线 < 慢线。",
                UserWarning,
            )
        return v


class ColumnAliasesConfig(BaseModel):
    """列名别名配置"""

    code_aliases: str = Field(default="代码=股票代码,证券代码=股票代码,股票代码=股票代码")
    name_aliases: str = Field(default="名称=股票简称,股票名称=股票简称,股票简称=股票简称,简称=股票简称")
    price_aliases: str = Field(
        default="最新价=最新价,现价=最新价,当前价格=最新价,今收盘=最新价,收盘=最新价,收盘价=最新价"
    )


class ResearchReportFilterConfig(BaseModel):
    """研报过滤配置"""

    ENABLE_RESEARCH_REPORT_FILTER: bool = Field(default=False)
    RESEARCH_REPORT_MIN_COUNT: int = Field(default=1, ge=1)


class FullBullScoringConfig(BaseModel):
    """MACD 完全多头评分维度权重"""

    WEIGHT_ZERO_AXIS: int = Field(default=20, ge=0, le=100)
    WEIGHT_STRATEGY_GOLDEN: int = Field(default=20, ge=0, le=100)
    WEIGHT_TACTICAL_GOLDEN: int = Field(default=15, ge=0, le=100)
    WEIGHT_MOMENTUM: int = Field(default=20, ge=0, le=100)
    WEIGHT_DIF_SLOPE: int = Field(default=15, ge=0, le=100)
    WEIGHT_DIVERGENCE: int = Field(default=10, ge=0, le=100)
    WEIGHT_VOLUME_PRICE: int = Field(default=10, ge=0, le=100)
    CONCLUSION_FULL_BULL: int = Field(default=80, ge=0, le=100)
    CONCLUSION_BULLISH: int = Field(default=60, ge=0, le=100)
    CONCLUSION_OSCILLATE: int = Field(default=40, ge=0, le=100)


class KlineDataConfig(BaseModel):
    """K线数据获取配置"""

    KLINE_HISTORY_DAYS: int = Field(default=200, ge=1)

    @field_validator("KLINE_HISTORY_DAYS")
    @classmethod
    def validate_days(cls, v):
        if v > 1000:
            warnings.warn(f"警告：KLINE_HISTORY_DAYS 设置为 {v} 天，数值较大可能导致获取数据时间过长。", UserWarning)
        return v


class UserFocusStocksConfig(BaseModel):
    """用户关注股池配置"""

    USER_FOCUS_STOCKS: str = Field(default="")


class AppConfig(BaseSettings):
    """应用配置主模型"""

    model_config = SettingsConfigDict(
        env_prefix="BAISYS_", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    database: DatabaseConfig
    system: SystemConfig
    logging: LoggingConfig
    multi_head_arrangement: MultiHeadArrangementConfig
    filter_rules: FilterRulesConfig
    fund_flow: FundFlowConfig
    technical_indicators: TechnicalIndicatorsConfig
    column_aliases: ColumnAliasesConfig
    research_report_filter: ResearchReportFilterConfig
    full_bull_scoring: FullBullScoringConfig
    user_focus_stocks: UserFocusStocksConfig
    kline_data: KlineDataConfig


class Config:
    """
    配置管理器类（保持向后兼容性）

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
        import configparser

        config = configparser.ConfigParser()
        config.read(self.config_file, encoding="utf-8")

        # 读取数据库配置（所有敏感字段支持 ENC: 前缀加密）
        from UtilsManager.ConfigCipher import ConfigCipher

        db = config["DATABASE"]

        # 密钥路径可配置，默认 ~/.baisys_quant_key
        key_path = db.get("encryption_key_path", fallback=None)
        if key_path:
            ConfigCipher.default_key_path = key_path
        database_config = DatabaseConfig(
            user=db.get("user"),
            password=ConfigCipher.maybe_decrypt(db.get("password")),
            host=ConfigCipher.maybe_decrypt(db.get("host")),
            port=ConfigCipher.maybe_decrypt(db.get("port")),
            db_name=ConfigCipher.maybe_decrypt(db.get("db_name")),
            main_board_only=db.getboolean("main_board_only", fallback=False),
        )

        # 读取 SYSTEM 配置
        system = config["SYSTEM"]
        system_config = SystemConfig(
            HOME_DIRECTORY=system.get("HOME_DIRECTORY", "~/Downloads/CoreNews_Reports"),
            TEMP_DATA_DIR=system.get("TEMP_DATA_DIR", "."),
            MAX_WORKERS=system.getint("MAX_WORKERS", fallback=15),
            DATA_FETCH_RETRIES=system.getint("DATA_FETCH_RETRIES", fallback=3),
            DATA_FETCH_DELAY=system.getint("DATA_FETCH_DELAY", fallback=5),
            SIGNAL_PROCESSING_PROCESSES=system.getint("signal_processing_processes", fallback=2),
        )

        # 读取 LOGGING 配置
        log = config["LOGGING"]
        logging_config = LoggingConfig(LOG_LEVEL=log.get("LOG_LEVEL", "INFO"), LOG_DIR=log.get("LOG_DIR", "Logs"))

        # 读取多头排列评分系统配置
        mha = config["MULTI_HEAD_ARRANGEMENT"]
        mha_config = MultiHeadArrangementConfig(
            FULL_BULL_THRESHOLD=mha.getint("FULL_BULL_THRESHOLD", fallback=85),
            TREND_ACCELERATION_THRESHOLD=mha.getint("TREND_ACCELERATION_THRESHOLD", fallback=65),
            TREND_OSCILLATION_THRESHOLD=mha.getint("TREND_OSCILLATION_THRESHOLD", fallback=45),
            TREND_WATCH_THRESHOLD=mha.getint("TREND_WATCH_THRESHOLD", fallback=45),
            MOVING_AVERAGE_PERIODS=mha.get("MOVING_AVERAGE_PERIODS", fallback="5,10,20,30,60"),
        )

        # 读取弱势股过滤规则配置
        fr = config["FILTER_RULES"]
        fr_config = FilterRulesConfig(
            ENABLE_WEAK_STOCK_FILTER=fr.getboolean("ENABLE_WEAK_STOCK_FILTER", fallback=True),
            EXEMPT_LEVELS=fr.get("EXEMPT_LEVELS", fallback="完全主升,趋势加速"),
        )

        # 读取资金流分析配置
        ff = config["FUND_FLOW"]
        ff_config = FundFlowConfig(FUND_FLOW_PERIODS=ff.get("FUND_FLOW_PERIODS", fallback="5,10,20"))

        # 读取技术指标信号配置
        ti = config["TECHNICAL_INDICATORS"]
        ti_config = TechnicalIndicatorsConfig(
            MACD_STANDARD_PARAMS=ti.get("MACD_STANDARD_PARAMS", fallback="12,26,9"),
            MACD_SECOND_PARAMS=ti.get("MACD_SECOND_PARAMS", fallback="6,13,5"),
        )

        # 读取列名别名配置
        col_aliases = config["COLUMN_ALIASES"]
        col_config = ColumnAliasesConfig(
            code_aliases=col_aliases.get("code_aliases", "代码=股票代码,证券代码=股票代码,股票代码=股票代码"),
            name_aliases=col_aliases.get(
                "name_aliases", "名称=股票简称,股票名称=股票简称,股票简称=股票简称,简称=股票简称"
            ),
            price_aliases=col_aliases.get(
                "price_aliases", "最新价=最新价,现价=最新价,当前价格=最新价,今收盘=最新价,收盘=最新价,收盘价=最新价"
            ),
        )

        # 读取研报过滤配置
        try:
            rrf = config["RESEARCH_REPORT_FILTER"]
            rrf_config = ResearchReportFilterConfig(
                ENABLE_RESEARCH_REPORT_FILTER=rrf.getboolean("ENABLE_RESEARCH_REPORT_FILTER", fallback=False),
                RESEARCH_REPORT_MIN_COUNT=rrf.getint("RESEARCH_REPORT_MIN_COUNT", fallback=1),
            )
        except KeyError:
            rrf_config = ResearchReportFilterConfig()

        # 读取 MACD 完全多头评分配置
        try:
            fbs = config["FULL_BULL_SCORING"]
            fbs_config = FullBullScoringConfig(
                WEIGHT_ZERO_AXIS=fbs.getint("WEIGHT_ZERO_AXIS", fallback=20),
                WEIGHT_STRATEGY_GOLDEN=fbs.getint("WEIGHT_STRATEGY_GOLDEN", fallback=20),
                WEIGHT_TACTICAL_GOLDEN=fbs.getint("WEIGHT_TACTICAL_GOLDEN", fallback=15),
                WEIGHT_MOMENTUM=fbs.getint("WEIGHT_MOMENTUM", fallback=20),
                WEIGHT_DIF_SLOPE=fbs.getint("WEIGHT_DIF_SLOPE", fallback=15),
                WEIGHT_DIVERGENCE=fbs.getint("WEIGHT_DIVERGENCE", fallback=10),
                WEIGHT_VOLUME_PRICE=fbs.getint("WEIGHT_VOLUME_PRICE", fallback=10),
                CONCLUSION_FULL_BULL=fbs.getint("CONCLUSION_FULL_BULL", fallback=80),
                CONCLUSION_BULLISH=fbs.getint("CONCLUSION_BULLISH", fallback=60),
                CONCLUSION_OSCILLATE=fbs.getint("CONCLUSION_OSCILLATE", fallback=40),
            )
        except KeyError:
            fbs_config = FullBullScoringConfig()

        # 读取K线数据获取配置
        try:
            kd = config["KLINE_DATA"]
            kd_config = KlineDataConfig(KLINE_HISTORY_DAYS=kd.getint("KLINE_HISTORY_DAYS", fallback=200))
        except KeyError:
            kd_config = KlineDataConfig()

        # 读取用户关注股池配置
        try:
            ufs = config["USER_FOCUS_STOCKS"]
            ufc = UserFocusStocksConfig(USER_FOCUS_STOCKS=ufs.get("user_focus_stocks", fallback=""))
        except KeyError:
            ufc = UserFocusStocksConfig()

        # 创建主配置对象
        self.app_config = AppConfig(
            database=database_config,
            system=system_config,
            logging=logging_config,
            multi_head_arrangement=mha_config,
            filter_rules=fr_config,
            fund_flow=ff_config,
            technical_indicators=ti_config,
            column_aliases=col_config,
            research_report_filter=rrf_config,
            full_bull_scoring=fbs_config,
            user_focus_stocks=ufc,
            kline_data=kd_config,
        )

        # 设置向后兼容的属性
        self.DB_USER = database_config.user
        self.DB_PASSWORD = database_config.password
        self.DB_HOST = database_config.host
        self.DB_PORT = database_config.port
        self.DB_NAME = database_config.db_name
        self.MAIN_BOARD_ONLY = database_config.main_board_only

        self.HOME_DIRECTORY = system_config.HOME_DIRECTORY
        self.TEMP_DATA_DIRECTORY = os.path.join(system_config.HOME_DIRECTORY, system_config.TEMP_DATA_DIR)
        self.MAX_WORKERS = system_config.MAX_WORKERS
        self.DATA_FETCH_RETRIES = system_config.DATA_FETCH_RETRIES
        self.DATA_FETCH_DELAY = system_config.DATA_FETCH_DELAY
        self.SIGNAL_PROCESSING_PROCESSES = system_config.SIGNAL_PROCESSING_PROCESSES

        self.LOG_LEVEL = logging_config.LOG_LEVEL
        self.LOG_DIR = os.path.join(system_config.HOME_DIRECTORY, logging_config.LOG_DIR)

        self.FULL_BULL_THRESHOLD = mha_config.FULL_BULL_THRESHOLD
        self.TREND_ACCELERATION_THRESHOLD = mha_config.TREND_ACCELERATION_THRESHOLD
        self.TREND_OSCILLATION_THRESHOLD = mha_config.TREND_OSCILLATION_THRESHOLD
        self.TREND_WATCH_THRESHOLD = mha_config.TREND_WATCH_THRESHOLD
        self.MOVING_AVERAGE_PERIODS = mha_config.MOVING_AVERAGE_PERIODS

        self.ENABLE_WEAK_STOCK_FILTER = fr_config.ENABLE_WEAK_STOCK_FILTER
        self.EXEMPT_LEVELS = fr_config.EXEMPT_LEVELS

        self.FUND_FLOW_PERIODS = ff_config.FUND_FLOW_PERIODS

        self.MACD_STANDARD_PARAMS = ti_config.MACD_STANDARD_PARAMS
        self.MACD_SECOND_PARAMS = ti_config.MACD_SECOND_PARAMS
        self.ENABLE_MACD_SECOND = True

        self.CODE_ALIASES = parse_aliases(col_config.code_aliases)
        self.NAME_ALIASES = parse_aliases(col_config.name_aliases)
        self.PRICE_ALIASES = parse_aliases(col_config.price_aliases)

        self.ENABLE_RESEARCH_REPORT_FILTER = rrf_config.ENABLE_RESEARCH_REPORT_FILTER
        self.RESEARCH_REPORT_MIN_COUNT = rrf_config.RESEARCH_REPORT_MIN_COUNT

        self.USER_FOCUS_STOCKS = ufc.USER_FOCUS_STOCKS

        self.FULL_BULL_WEIGHTS = {
            "零轴条件": fbs_config.WEIGHT_ZERO_AXIS,
            "战略金叉": fbs_config.WEIGHT_STRATEGY_GOLDEN,
            "战术金叉": fbs_config.WEIGHT_TACTICAL_GOLDEN,
            "动能": fbs_config.WEIGHT_MOMENTUM,
            "DIF斜率": fbs_config.WEIGHT_DIF_SLOPE,
            "背离信号": fbs_config.WEIGHT_DIVERGENCE,
            "量价配合": fbs_config.WEIGHT_VOLUME_PRICE,
        }
        self.FULL_BULL_THRESHOLDS = {
            "fully_bull": fbs_config.CONCLUSION_FULL_BULL,
            "bullish": fbs_config.CONCLUSION_BULLISH,
            "oscillate": fbs_config.CONCLUSION_OSCILLATE,
        }

        self.KLINE_HISTORY_DAYS = kd_config.KLINE_HISTORY_DAYS

    def _ensure_directories(self):
        dirs = [self.HOME_DIRECTORY, self.TEMP_DATA_DIRECTORY, self.LOG_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def get_db_connection_string(self) -> str:
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
