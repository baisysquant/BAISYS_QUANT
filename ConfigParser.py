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


def _default_signal_workers() -> int:
    return max(os.cpu_count() or 4, 2)
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
    SIGNAL_PROCESSING_PROCESSES: int = Field(default_factory=_default_signal_workers, ge=1)

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

    MACD_PARAMS: tuple[int, int, int] = Field(default=(12, 26, 9))

    @field_validator("MACD_PARAMS", mode="before")
    @classmethod
    def parse_macd_params(cls, v):
        if isinstance(v, str):
            return tuple(int(p.strip()) for p in v.split(","))
        return v

    @field_validator("MACD_PARAMS")
    @classmethod
    def validate_macd_params(cls, v):
        fast, slow, signal = v
        if fast >= slow:
            raise ValueError(
                f"错误：MACD参数不合理（快线{fast} >= 慢线{slow}），"
                f"请确保快线 < 慢线。"
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
    """MACD 完全多头评分维度权重 + 规则阈值"""

    WEIGHT_ZERO_AXIS: int = Field(default=20, ge=0, le=100)
    WEIGHT_STRATEGY_GOLDEN: int = Field(default=15, ge=0, le=100)
    WEIGHT_TACTICAL_GOLDEN: int = Field(default=10, ge=0, le=100)
    WEIGHT_MOMENTUM: int = Field(default=15, ge=0, le=100)
    WEIGHT_DIF_SLOPE: int = Field(default=10, ge=0, le=100)
    WEIGHT_DIVERGENCE: int = Field(default=10, ge=0, le=100)
    WEIGHT_VOLUME_PRICE: int = Field(default=10, ge=0, le=100)
    WEIGHT_KLINE_PATTERN: int = Field(default=10, ge=0, le=100)
    CONCLUSION_FULL_BULL: int = Field(default=80, ge=0, le=100)
    CONCLUSION_BULLISH: int = Field(default=60, ge=0, le=100)
    CONCLUSION_OSCILLATE: int = Field(default=40, ge=0, le=100)
    # 规则阈值
    RULE_DIVERGENCE_THRESHOLD: float = Field(default=0.3, ge=0, le=1.0)
    RULE_WINNER_RATE_HIGH: int = Field(default=80, ge=0, le=100)
    RULE_WINNER_RATE_LOW: int = Field(default=15, ge=0, le=100)
    RULE_COST_RESISTANCE_RATIO: float = Field(default=0.95, ge=0, le=1.0)
    RULE_CHIP_CONCENTRATED_RATIO: float = Field(default=0.15, ge=0, le=1.0)
    RULE_PRICE_NEW_HIGH_DAYS: int = Field(default=20, ge=5, le=120)


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


class AShareHubConfig(BaseModel):
    """AShareHub 筹码分布数据配置"""

    API_KEY: str = Field(default="")
    ENABLE_CHIP_DISTRIBUTION: bool = Field(default=False)
    CHIP_LIMIT: int = Field(default=1, ge=1, le=200)
    @field_validator("CHIP_LIMIT")
    @classmethod
    def validate_limit(cls, v):
        if v > 50:
            import warnings
            warnings.warn("CHIP_LIMIT>50 会拉取多日历史快照，通常用 limit=1（最新快照）即可。", UserWarning)
        return v


class MacroFilterConfig(BaseModel):
    """宏观过滤器配置"""

    ENABLE_MACRO_FILTER: bool = Field(default=True)
    INDEX_SYMBOL: str = Field(default="sh000001")
    TREND_LOOKBACK_DAYS: int = Field(default=250, ge=60, le=500)
    VOLUME_LOOKBACK_DAYS: int = Field(default=20, ge=5, le=120)
    ADVANCE_RATIO_ICE: float = Field(default=0.25, ge=0, le=1.0)
    ADVANCE_RATIO_WEAK: float = Field(default=0.35, ge=0, le=1.0)
    ADVANCE_RATIO_HOT: float = Field(default=0.70, ge=0, le=1.0)


class RegimeDetectionConfig(BaseModel):
    """市场状态分类参数"""

    BOLL_NARROW_RATIO: float = Field(default=0.8, ge=0.1, le=2.0,
                                      description="窄布林判定阈值：近期BOLL带宽/历史平均带宽 < 此值→震荡")
    OSCILLATION_HIST_STD_RATIO: float = Field(default=0.1, ge=0.01, le=1.0,
                                               description="震荡模式柱状图标准差比：abs(柱状图) < 此值×close.std()→震荡")
    TOP_RISK_MA20_DEVIATION: float = Field(default=0.15, ge=0.01, le=0.5,
                                            description="顶风险MA20偏离阈值：(close-MA20)/MA20 > 此值→顶部风险")
    OSCILLATION_MIN_BARS: int = Field(default=30, ge=10, le=120,
                                       description="震荡判定最小K线数")
    REVERSAL_LOOKBACK: int = Field(default=10, ge=5, le=60,
                                    description="反转检测回溯长度（根K线）")


class DivergenceConfig(BaseModel):
    """背离检测参数"""

    BASE_DISTANCE: int = Field(default=10, ge=5, le=60,
                                description="背离检测基础窗口（adaptive_distance的base_distance）")
    STRENGTH_THRESHOLD: float = Field(default=0.15, ge=0.01, le=1.0,
                                       description="背离有效强度门限，超过此值才生成信号")
    DECAY_HALF_LIFE: int = Field(default=8, ge=2, le=60,
                                  description="背离信号半衰期（天）")
    SLOPE_WINDOW: int = Field(default=5, ge=3, le=30,
                               description="DIF斜率线性回归窗口（根K线）")


class ScoringParamsConfig(BaseModel):
    """评分计算参数"""

    CROSS_DECAY_DAYS: int = Field(default=30, ge=5, le=120,
                                   description="金叉信号衰减半衰期（天）")
    CROSS_DECAY_MIN: float = Field(default=0.3, ge=0.1, le=1.0,
                                    description="金叉衰减下限（比例）")
    KLINE_DECAY_DAYS: int = Field(default=10, ge=2, le=60,
                                   description="K线形态衰减半衰期（天）")
    KLINE_DECAY_MIN: float = Field(default=0.2, ge=0.05, le=1.0,
                                    description="K线形态衰减下限（比例）")
    VOL_NORM_DENOMINATOR: float = Field(default=0.15, ge=0.01, le=1.0,
                                         description="金叉强度波动率归一化分母：(DIF-DEA)/ATR/此值→vol_factor")
    ATR_STOP_MULT: float = Field(default=1.5, ge=0.5, le=5.0,
                                  description="止损ATR倍数：止损价=close-ATR×此值")
    ATR_T1_MULT: float = Field(default=3.0, ge=1.0, le=10.0,
                                description="T1目标价ATR倍数")
    ATR_T2_MULT: float = Field(default=5.0, ge=2.0, le=20.0,
                                description="T2目标价ATR倍数")
    TRAILING_STOP_HIGH_RATIO: float = Field(default=0.98, ge=0.9, le=1.0,
                                              description="移动止损高位触发比：close≥近N日最高价×此值")
    TRAILING_STOP_LOOKBACK: int = Field(default=10, ge=5, le=60,
                                         description="移动止损回溯窗口（根K线）")
    TRAILING_STOP_HIGH_LOOKBACK: int = Field(default=20, ge=10, le=120,
                                              description="移动止损参考高点回溯窗口（根K线）")
    EXPECTED_RETURN_LOOKBACK: int = Field(default=20, ge=5, le=120,
                                           description="预期盈亏比计算回溯窗口（根K线）")


class TechnicalConstantsConfig(BaseModel):
    """标准技术指标参数"""

    ATR_LENGTH: int = Field(default=14, ge=5, le=60,
                             description="ATR计算周期（Wilder标准14）")
    ADX_LENGTH: int = Field(default=14, ge=5, le=60,
                             description="ADX计算周期（Wilder标准14）")
    RSI_LENGTH: int = Field(default=14, ge=5, le=60,
                             description="RSI计算周期（Wilder标准14）")
    BOLL_LENGTH: int = Field(default=20, ge=5, le=60,
                              description="BOLL计算周期（Bollinger标准20）")
    BOLL_STD: float = Field(default=2.0, ge=1.0, le=4.0,
                             description="BOLL标准差倍数（标准2）")
    STOCH_K: int = Field(default=9, ge=3, le=30,
                          description="Stoch %K周期（Lane标准9）")
    STOCH_D: int = Field(default=3, ge=2, le=15,
                          description="Stoch %D平滑周期（标准3）")
    KLINE_SCAN_WINDOW: int = Field(default=60, ge=20, le=200,
                                    description="K线形态扫描窗口（根K线）")


class PositionSizingConfig(BaseModel):
    """仓位管理配置模型"""

    MAX_SINGLE_POSITION: float = Field(default=0.33, ge=0.0, le=1.0,
                                       description="最大单票仓位")
    KELLY_FRACTION: float = Field(default=0.25, ge=0.0, le=1.0,
                                  description="半凯利系数")
    DEFAULT_WIN_RATE: float = Field(default=0.50, ge=0.0, le=1.0,
                                    description="默认胜率假设")
    POSITION_A: float = Field(default=0.30, ge=0.0, le=1.0,
                              description="A级基础仓位")
    POSITION_B: float = Field(default=0.15, ge=0.0, le=1.0,
                              description="B级基础仓位")
    POSITION_C: float = Field(default=0.05, ge=0.0, le=1.0,
                              description="C级基础仓位")
    POSITION_D: float = Field(default=0.00, ge=0.0, le=1.0,
                              description="D级基础仓位")
    MAX_INDUSTRY_EXPOSURE: float = Field(default=0.30, ge=0.0, le=1.0,
                                         description="最大行业集中度")
    RISK_BUDGET: float = Field(default=0.02, ge=0.001, le=0.10,
                               description="波动率风险预算")
    MAX_DRAWDOWN_REDUCTION: float = Field(default=0.50, ge=0.0, le=1.0,
                                          description="最大回撤缩减系数")


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
    asharehub: AShareHubConfig
    macro_filter: MacroFilterConfig
    regime_detection: RegimeDetectionConfig
    divergence: DivergenceConfig
    scoring_params: ScoringParamsConfig
    technical_constants: TechnicalConstantsConfig
    position_sizing: PositionSizingConfig


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
        MACD_PARAMS: MACD参数 (fast, slow, signal)，默认(12,26,9)
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
            SIGNAL_PROCESSING_PROCESSES=system.getint("signal_processing_processes", fallback=_default_signal_workers()),
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
            MACD_PARAMS=ti.get("MACD_PARAMS", fallback="12,26,9"),
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
                WEIGHT_STRATEGY_GOLDEN=fbs.getint("WEIGHT_STRATEGY_GOLDEN", fallback=15),
                WEIGHT_TACTICAL_GOLDEN=fbs.getint("WEIGHT_TACTICAL_GOLDEN", fallback=10),
                WEIGHT_MOMENTUM=fbs.getint("WEIGHT_MOMENTUM", fallback=15),
                WEIGHT_DIF_SLOPE=fbs.getint("WEIGHT_DIF_SLOPE", fallback=10),
                WEIGHT_DIVERGENCE=fbs.getint("WEIGHT_DIVERGENCE", fallback=10),
                WEIGHT_VOLUME_PRICE=fbs.getint("WEIGHT_VOLUME_PRICE", fallback=10),
                WEIGHT_KLINE_PATTERN=fbs.getint("WEIGHT_KLINE_PATTERN", fallback=10),
                CONCLUSION_FULL_BULL=fbs.getint("CONCLUSION_FULL_BULL", fallback=80),
                CONCLUSION_BULLISH=fbs.getint("CONCLUSION_BULLISH", fallback=60),
                CONCLUSION_OSCILLATE=fbs.getint("CONCLUSION_OSCILLATE", fallback=40),
                RULE_DIVERGENCE_THRESHOLD=fbs.getfloat("RULE_DIVERGENCE_THRESHOLD", fallback=0.3),
                RULE_WINNER_RATE_HIGH=fbs.getint("RULE_WINNER_RATE_HIGH", fallback=80),
                RULE_WINNER_RATE_LOW=fbs.getint("RULE_WINNER_RATE_LOW", fallback=15),
                RULE_COST_RESISTANCE_RATIO=fbs.getfloat("RULE_COST_RESISTANCE_RATIO", fallback=0.95),
                RULE_CHIP_CONCENTRATED_RATIO=fbs.getfloat("RULE_CHIP_CONCENTRATED_RATIO", fallback=0.15),
                RULE_PRICE_NEW_HIGH_DAYS=fbs.getint("RULE_PRICE_NEW_HIGH_DAYS", fallback=20),
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

        # 读取 AShareHub 筹码分布配置
        try:
            ah = config["ASHAREHUB"]
            ah_config = AShareHubConfig(
                API_KEY=ConfigCipher.maybe_decrypt(ah.get("api_key", "")),
                ENABLE_CHIP_DISTRIBUTION=ah.getboolean("enable_chip_distribution", fallback=False),
                CHIP_LIMIT=ah.getint("chip_limit", fallback=1),
            )
        except KeyError:
            ah_config = AShareHubConfig()

        # 读取宏观过滤器配置
        try:
            mf = config["MACRO_FILTER"]
            mf_config = MacroFilterConfig(
                ENABLE_MACRO_FILTER=mf.getboolean("enable_macro_filter", fallback=True),
                INDEX_SYMBOL=mf.get("index_symbol", fallback="sh000001"),
                TREND_LOOKBACK_DAYS=mf.getint("trend_lookback_days", fallback=250),
                VOLUME_LOOKBACK_DAYS=mf.getint("volume_lookback_days", fallback=20),
                ADVANCE_RATIO_ICE=mf.getfloat("advance_ratio_ice", fallback=0.25),
                ADVANCE_RATIO_WEAK=mf.getfloat("advance_ratio_weak", fallback=0.35),
                ADVANCE_RATIO_HOT=mf.getfloat("advance_ratio_hot", fallback=0.70),
            )
        except KeyError:
            mf_config = MacroFilterConfig()

        # 读取市场状态分类参数
        try:
            rd = config["REGIME_DETECTION"]
            rd_config = RegimeDetectionConfig(
                BOLL_NARROW_RATIO=rd.getfloat("boll_narrow_ratio", fallback=0.8),
                OSCILLATION_HIST_STD_RATIO=rd.getfloat("oscillation_hist_std_ratio", fallback=0.1),
                TOP_RISK_MA20_DEVIATION=rd.getfloat("top_risk_ma20_deviation", fallback=0.15),
                OSCILLATION_MIN_BARS=rd.getint("oscillation_min_bars", fallback=30),
                REVERSAL_LOOKBACK=rd.getint("reversal_lookback", fallback=10),
            )
        except KeyError:
            rd_config = RegimeDetectionConfig()

        # 读取背离检测参数
        try:
            dv = config["DIVERGENCE"]
            dv_config = DivergenceConfig(
                BASE_DISTANCE=dv.getint("base_distance", fallback=10),
                STRENGTH_THRESHOLD=dv.getfloat("strength_threshold", fallback=0.15),
                DECAY_HALF_LIFE=dv.getint("decay_half_life", fallback=8),
                SLOPE_WINDOW=dv.getint("slope_window", fallback=5),
            )
        except KeyError:
            dv_config = DivergenceConfig()

        # 读取评分计算参数
        try:
            sp = config["SCORING_PARAMS"]
            sp_config = ScoringParamsConfig(
                CROSS_DECAY_DAYS=sp.getint("cross_decay_days", fallback=30),
                CROSS_DECAY_MIN=sp.getfloat("cross_decay_min", fallback=0.3),
                KLINE_DECAY_DAYS=sp.getint("kline_decay_days", fallback=10),
                KLINE_DECAY_MIN=sp.getfloat("kline_decay_min", fallback=0.2),
                VOL_NORM_DENOMINATOR=sp.getfloat("vol_norm_denominator", fallback=0.15),
                ATR_STOP_MULT=sp.getfloat("atr_stop_mult", fallback=1.5),
                ATR_T1_MULT=sp.getfloat("atr_t1_mult", fallback=3.0),
                ATR_T2_MULT=sp.getfloat("atr_t2_mult", fallback=5.0),
                TRAILING_STOP_HIGH_RATIO=sp.getfloat("trailing_stop_high_ratio", fallback=0.98),
                TRAILING_STOP_LOOKBACK=sp.getint("trailing_stop_lookback", fallback=10),
                TRAILING_STOP_HIGH_LOOKBACK=sp.getint("trailing_stop_high_lookback", fallback=20),
                EXPECTED_RETURN_LOOKBACK=sp.getint("expected_return_lookback", fallback=20),
            )
        except KeyError:
            sp_config = ScoringParamsConfig()

        # 读取标准技术指标参数
        try:
            tc = config["TECHNICAL_CONSTANTS"]
            tc_config = TechnicalConstantsConfig(
                ATR_LENGTH=tc.getint("atr_length", fallback=14),
                ADX_LENGTH=tc.getint("adx_length", fallback=14),
                RSI_LENGTH=tc.getint("rsi_length", fallback=14),
                BOLL_LENGTH=tc.getint("boll_length", fallback=20),
                BOLL_STD=tc.getfloat("boll_std", fallback=2.0),
                STOCH_K=tc.getint("stoch_k", fallback=9),
                STOCH_D=tc.getint("stoch_d", fallback=3),
                KLINE_SCAN_WINDOW=tc.getint("kline_scan_window", fallback=60),
            )
        except KeyError:
            tc_config = TechnicalConstantsConfig()

        # 读取仓位管理配置
        try:
            ps = config["POSITION_SIZING"]
            ps_config = PositionSizingConfig(
                MAX_SINGLE_POSITION=ps.getfloat("max_single_position", fallback=0.33),
                KELLY_FRACTION=ps.getfloat("kelly_fraction", fallback=0.25),
                DEFAULT_WIN_RATE=ps.getfloat("default_win_rate", fallback=0.50),
                POSITION_A=ps.getfloat("position_a", fallback=0.30),
                POSITION_B=ps.getfloat("position_b", fallback=0.15),
                POSITION_C=ps.getfloat("position_c", fallback=0.05),
                POSITION_D=ps.getfloat("position_d", fallback=0.00),
                MAX_INDUSTRY_EXPOSURE=ps.getfloat("max_industry_exposure", fallback=0.30),
                RISK_BUDGET=ps.getfloat("risk_budget", fallback=0.02),
                MAX_DRAWDOWN_REDUCTION=ps.getfloat("max_drawdown_reduction", fallback=0.50),
            )
        except KeyError:
            ps_config = PositionSizingConfig()

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
            asharehub=ah_config,
            macro_filter=mf_config,
            regime_detection=rd_config,
            divergence=dv_config,
            scoring_params=sp_config,
            technical_constants=tc_config,
            position_sizing=ps_config,
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

        self.MACD_PARAMS = ti_config.MACD_PARAMS

        self.CODE_ALIASES = parse_aliases(col_config.code_aliases)
        self.NAME_ALIASES = parse_aliases(col_config.name_aliases)
        self.PRICE_ALIASES = parse_aliases(col_config.price_aliases)

        self.ENABLE_RESEARCH_REPORT_FILTER = rrf_config.ENABLE_RESEARCH_REPORT_FILTER
        self.RESEARCH_REPORT_MIN_COUNT = rrf_config.RESEARCH_REPORT_MIN_COUNT

        self.USER_FOCUS_STOCKS = ufc.USER_FOCUS_STOCKS

        self.ASHAREHUB_API_KEY = ah_config.API_KEY
        self.ENABLE_CHIP_DISTRIBUTION = ah_config.ENABLE_CHIP_DISTRIBUTION
        self.CHIP_LIMIT = ah_config.CHIP_LIMIT

        self.ENABLE_MACRO_FILTER = mf_config.ENABLE_MACRO_FILTER
        self.MACRO_FILTER_INDEX_SYMBOL = mf_config.INDEX_SYMBOL

        self.REGIME_DETECTION = {
            "boll_narrow_ratio": rd_config.BOLL_NARROW_RATIO,
            "oscillation_hist_std_ratio": rd_config.OSCILLATION_HIST_STD_RATIO,
            "top_risk_ma20_deviation": rd_config.TOP_RISK_MA20_DEVIATION,
            "oscillation_min_bars": rd_config.OSCILLATION_MIN_BARS,
            "reversal_lookback": rd_config.REVERSAL_LOOKBACK,
        }
        self.DIVERGENCE_PARAMS = {
            "base_distance": dv_config.BASE_DISTANCE,
            "strength_threshold": dv_config.STRENGTH_THRESHOLD,
            "decay_half_life": dv_config.DECAY_HALF_LIFE,
            "slope_window": dv_config.SLOPE_WINDOW,
        }
        self.SCORING_PARAMS = {
            "cross_decay_days": sp_config.CROSS_DECAY_DAYS,
            "cross_decay_min": sp_config.CROSS_DECAY_MIN,
            "kline_decay_days": sp_config.KLINE_DECAY_DAYS,
            "kline_decay_min": sp_config.KLINE_DECAY_MIN,
            "vol_norm_denominator": sp_config.VOL_NORM_DENOMINATOR,
            "atr_stop_mult": sp_config.ATR_STOP_MULT,
            "atr_t1_mult": sp_config.ATR_T1_MULT,
            "atr_t2_mult": sp_config.ATR_T2_MULT,
            "trailing_stop_high_ratio": sp_config.TRAILING_STOP_HIGH_RATIO,
            "trailing_stop_lookback": sp_config.TRAILING_STOP_LOOKBACK,
            "trailing_stop_high_lookback": sp_config.TRAILING_STOP_HIGH_LOOKBACK,
            "expected_return_lookback": sp_config.EXPECTED_RETURN_LOOKBACK,
        }
        self.TECHNICAL_CONSTANTS = {
            "atr_length": tc_config.ATR_LENGTH,
            "adx_length": tc_config.ADX_LENGTH,
            "rsi_length": tc_config.RSI_LENGTH,
            "boll_length": tc_config.BOLL_LENGTH,
            "boll_std": tc_config.BOLL_STD,
            "stoch_k": tc_config.STOCH_K,
            "stoch_d": tc_config.STOCH_D,
            "kline_scan_window": tc_config.KLINE_SCAN_WINDOW,
        }

        self.FULL_BULL_WEIGHTS = {
            "MACD趋势": fbs_config.WEIGHT_ZERO_AXIS,
            "金叉信号": fbs_config.WEIGHT_STRATEGY_GOLDEN,
            "柱状动能": fbs_config.WEIGHT_MOMENTUM,
            "DIF斜率": fbs_config.WEIGHT_DIF_SLOPE,
            "背离信号": fbs_config.WEIGHT_DIVERGENCE,
            "量价配合": fbs_config.WEIGHT_VOLUME_PRICE,
            "K线形态": fbs_config.WEIGHT_KLINE_PATTERN,
        }
        self.FULL_BULL_THRESHOLDS = {
            "fully_bull": fbs_config.CONCLUSION_FULL_BULL,
            "bullish": fbs_config.CONCLUSION_BULLISH,
            "oscillate": fbs_config.CONCLUSION_OSCILLATE,
        }

        self.RULE_THRESHOLDS = {
            "divergence": fbs_config.RULE_DIVERGENCE_THRESHOLD,
            "winner_rate_high": fbs_config.RULE_WINNER_RATE_HIGH,
            "winner_rate_low": fbs_config.RULE_WINNER_RATE_LOW,
            "cost_resistance_ratio": fbs_config.RULE_COST_RESISTANCE_RATIO,
            "chip_concentrated_ratio": fbs_config.RULE_CHIP_CONCENTRATED_RATIO,
            "price_new_high_days": fbs_config.RULE_PRICE_NEW_HIGH_DAYS,
        }

        self.POSITION_SIZING = {
            "max_single_position": ps_config.MAX_SINGLE_POSITION,
            "kelly_fraction": ps_config.KELLY_FRACTION,
            "default_win_rate": ps_config.DEFAULT_WIN_RATE,
            "position_a": ps_config.POSITION_A,
            "position_b": ps_config.POSITION_B,
            "position_c": ps_config.POSITION_C,
            "position_d": ps_config.POSITION_D,
            "max_industry_exposure": ps_config.MAX_INDUSTRY_EXPOSURE,
            "risk_budget": ps_config.RISK_BUDGET,
            "max_drawdown_reduction": ps_config.MAX_DRAWDOWN_REDUCTION,
        }

        self.KLINE_HISTORY_DAYS = kd_config.KLINE_HISTORY_DAYS

    def _ensure_directories(self):
        dirs = [self.HOME_DIRECTORY, self.TEMP_DATA_DIRECTORY, self.LOG_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def get_db_connection_string(self) -> str:
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
