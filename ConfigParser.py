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


def _default_signal_workers() -> int:
    return max(os.cpu_count() or 4, 2)


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
    STOCK_BASIC_INFO_EXPIRE_DAYS: int = Field(default=30, ge=1, le=365,
                                                description="股票基本信息缓存过期天数")
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
    def parse_periods(cls, v: str | list[int]) -> list[int]:
        if isinstance(v, str):
            return [int(p.strip()) for p in v.split(",")]
        return v


class FilterRulesConfig(BaseModel):
    """弱势股过滤规则配置 + 流动性参数"""

    ENABLE_WEAK_STOCK_FILTER: bool = Field(default=True)
    EXEMPT_LEVELS: list[str] = Field(default=["完全主升", "趋势加速"])

    @field_validator("EXEMPT_LEVELS", mode="before")
    @classmethod
    def parse_exempt_levels(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [level.strip() for level in v.split(",")]
        return v

    # ── 流动性参数 ────────────────────────────────────────────────────
    LIQ_VETO_RATIO: float = Field(default=0.05, ge=0.01, le=1.0)
    LIQ_W_SECTION: float = Field(default=0.4, ge=0.0, le=1.0)
    LIQ_W_TIMESERIES: float = Field(default=0.4, ge=0.0, le=1.0)
    LIQ_W_MARKETCAP: float = Field(default=0.2, ge=0.0, le=1.0)
    LIQ_MIN_DISCOUNT: float = Field(default=0.3, ge=0.0, le=1.0)


class FundFlowConfig(BaseModel):
    """资金流分析配置"""

    FUND_FLOW_PERIODS: list[int] = Field(default=[5, 10, 20])

    @field_validator("FUND_FLOW_PERIODS", mode="before")
    @classmethod
    def parse_periods(cls, v: str | list[int]) -> list[int]:
        if isinstance(v, str):
            return [int(p.strip()) for p in v.split(",")]
        return v

    @field_validator("FUND_FLOW_PERIODS")
    @classmethod
    def validate_periods(cls, v: list[int]) -> list[int]:
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
    def parse_macd_params(cls, v: str | tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(v, str):
            return tuple(int(p.strip()) for p in v.split(","))
        return v

    @field_validator("MACD_PARAMS")
    @classmethod
    def validate_macd_params(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
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
    def validate_days(cls, v: int) -> int:
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
    MONEYFLOW_RETRY: int = Field(default=3, ge=0, le=10,
                                   description="资金流向 API 429 重试次数")
    MONEYFLOW_PAGE_DELAY: float = Field(default=1.0, ge=0.0, le=30.0,
                                          description="资金流分页间隔秒数")
    @field_validator("CHIP_LIMIT")
    @classmethod
    def validate_limit(cls, v: int) -> int:
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


class BacktestConfig(BaseModel):
    """回测系统配置模型"""

    ENABLED: bool = True
    OPTIMIZE_FREQUENCY: str = "monthly"
    BACKTEST_START_DATE: str = Field(default="20200101", pattern=r"^\d{8}$")
    OUT_OF_SAMPLE_DAYS: int = Field(default=20, ge=5, le=120)
    INITIAL_CASH: float = Field(default=1_000_000, gt=0)
    FULL_A_SHARE_MODE: bool = Field(default=False)

    # 待寻优参数范围（逗号分隔：min,max,step）
    ATR_STOP_MULT_RANGE: str = "1.0,3.0,0.5"
    ATR_T1_MULT_RANGE: str = "2.0,6.0,1.0"
    KELLY_FRACTION_RANGE: str = "0.1,0.5,0.1"
    POSITION_A_RANGE: str = "0.2,0.5,0.05"
    LIQ_VETO_RATIO_RANGE: str = "0.03,0.10,0.01"
    BOLL_NARROW_RATIO_RANGE: str = "0.6,1.2,0.1"
    CROSS_DECAY_DAYS_RANGE: str = "15,60,5"

    @field_validator("OPTIMIZE_FREQUENCY")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        v_lower = v.lower().strip()
        if v_lower not in ("monthly", "quarterly", "initial"):
            msg = f"OPTIMIZE_FREQUENCY 必须为 monthly/quarterly/initial，收到 {v}"
            raise ValueError(msg)
        return v_lower

    def parse_range(self, key: str) -> tuple[float, float, float]:
        raw = getattr(self, key.upper(), "")
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != 3:
            msg = f"{key} 格式应为 min,max,step，收到 {raw!r}"
            raise ValueError(msg)
        return (parts[0], parts[1], parts[2])


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
    backtest: BacktestConfig


class Config:
    """
    配置管理器（INI→Pydantic，自动类型转换）

    读取 config.ini 并委托 AppConfig（Pydantic）做类型校验与转换。
    所有历史平铺属性改为 @property 委托，数据归一在 app_config。
    """

    def __init__(self, config_file: str = "config.ini") -> None:
        self.config_file = config_file
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件未找到: {os.path.abspath(self.config_file)}")

        self._load_config()
        self._ensure_directories()

    # ── 加载：INI → Pydantic（单次操作，零手动类型转换） ─────────────────

    def _section_upper(self, name: str) -> dict[str, str]:
        """读取 INI 节并转大写 key，适配 Pydantic UPPER_CASE 字段。"""
        return {k.upper(): v for k, v in dict(self._raw_section(name)).items()}

    def _raw_section(self, name: str) -> dict[str, str]:
        """安全读取 INI 节，不存在时返回空 dict。"""
        try:
            return dict(self._cp[name])
        except KeyError:
            return {}

    def _load_config(self) -> None:
        import configparser

        self._cp = configparser.ConfigParser()
        self._cp.read(self.config_file, encoding="utf-8")

        from UtilsManager.ConfigCipher import ConfigCipher

        # DATABASE（lowercase 字段名 + 敏感字段解密）
        key_path = self._cp["DATABASE"].get("encryption_key_path", fallback=None)
        if key_path:
            ConfigCipher.default_key_path = key_path
        db_raw = self._raw_section("DATABASE")
        for enc_key in ("password", "host", "port", "db_name"):
            db_raw[enc_key] = ConfigCipher.maybe_decrypt(db_raw.get(enc_key, ""))

        # COLUMN_ALIASES（lowercase 字段名）
        col_raw = self._raw_section("COLUMN_ALIASES")

        # ASHAREHUB（API_KEY 需解密）
        ah_raw = self._section_upper("ASHAREHUB")
        if ah_raw:
            ah_raw["API_KEY"] = ConfigCipher.maybe_decrypt(ah_raw.get("API_KEY", ""))

        # 装配 AppConfig（Pydantic field_validator 自动处理逗号/bool/int/float 转换）
        self.app_config = AppConfig(
            database=DatabaseConfig(**db_raw),
            system=SystemConfig(**self._section_upper("SYSTEM")),
            logging=LoggingConfig(**self._section_upper("LOGGING")),
            multi_head_arrangement=MultiHeadArrangementConfig(**self._section_upper("MULTI_HEAD_ARRANGEMENT")),
            filter_rules=FilterRulesConfig(**self._section_upper("FILTER_RULES")),
            fund_flow=FundFlowConfig(**self._section_upper("FUND_FLOW")),
            technical_indicators=TechnicalIndicatorsConfig(**self._section_upper("TECHNICAL_INDICATORS")),
            column_aliases=ColumnAliasesConfig(**col_raw),
            research_report_filter=ResearchReportFilterConfig(**self._section_upper("RESEARCH_REPORT_FILTER")),
            full_bull_scoring=FullBullScoringConfig(**self._section_upper("FULL_BULL_SCORING")),
            kline_data=KlineDataConfig(**self._section_upper("KLINE_DATA")),
            user_focus_stocks=UserFocusStocksConfig(**self._section_upper("USER_FOCUS_STOCKS")),
            asharehub=AShareHubConfig(**ah_raw),
            macro_filter=MacroFilterConfig(**self._section_upper("MACRO_FILTER")),
            regime_detection=RegimeDetectionConfig(**self._section_upper("REGIME_DETECTION")),
            divergence=DivergenceConfig(**self._section_upper("DIVERGENCE")),
            scoring_params=ScoringParamsConfig(**self._section_upper("SCORING_PARAMS")),
            technical_constants=TechnicalConstantsConfig(**self._section_upper("TECHNICAL_CONSTANTS")),
            position_sizing=PositionSizingConfig(**self._section_upper("POSITION_SIZING")),
            backtest=BacktestConfig(**self._section_upper("BACKTEST")),
        )

    # ── 向后兼容属性（只读委托至 app_config） ──────────────────────────

    # 数据库
    @property
    def DB_USER(self) -> str: return self.app_config.database.user

    @property
    def DB_PASSWORD(self) -> str: return self.app_config.database.password

    @property
    def DB_HOST(self) -> str: return self.app_config.database.host

    @property
    def DB_PORT(self) -> str: return self.app_config.database.port

    @property
    def DB_NAME(self) -> str: return self.app_config.database.db_name

    @property
    def MAIN_BOARD_ONLY(self) -> bool: return self.app_config.database.main_board_only

    # 系统
    @property
    def HOME_DIRECTORY(self) -> str: return self.app_config.system.HOME_DIRECTORY

    @property
    def TEMP_DATA_DIRECTORY(self) -> str:
        return os.path.join(self.app_config.system.HOME_DIRECTORY, self.app_config.system.TEMP_DATA_DIR)

    @property
    def MAX_WORKERS(self) -> int: return self.app_config.system.MAX_WORKERS

    @property
    def DATA_FETCH_RETRIES(self) -> int: return self.app_config.system.DATA_FETCH_RETRIES

    @property
    def DATA_FETCH_DELAY(self) -> int: return self.app_config.system.DATA_FETCH_DELAY

    @property
    def STOCK_BASIC_INFO_EXPIRE_DAYS(self) -> int: return self.app_config.system.STOCK_BASIC_INFO_EXPIRE_DAYS

    @property
    def SIGNAL_PROCESSING_PROCESSES(self) -> int: return self.app_config.system.SIGNAL_PROCESSING_PROCESSES

    # 日志
    @property
    def LOG_LEVEL(self) -> str: return self.app_config.logging.LOG_LEVEL

    @property
    def LOG_DIR(self) -> str:
        return os.path.join(self.app_config.system.HOME_DIRECTORY, self.app_config.logging.LOG_DIR)

    # 多头排列
    @property
    def FULL_BULL_THRESHOLD(self) -> int: return self.app_config.multi_head_arrangement.FULL_BULL_THRESHOLD

    @property
    def TREND_ACCELERATION_THRESHOLD(self) -> int: return self.app_config.multi_head_arrangement.TREND_ACCELERATION_THRESHOLD

    @property
    def TREND_OSCILLATION_THRESHOLD(self) -> int: return self.app_config.multi_head_arrangement.TREND_OSCILLATION_THRESHOLD

    @property
    def TREND_WATCH_THRESHOLD(self) -> int: return self.app_config.multi_head_arrangement.TREND_WATCH_THRESHOLD

    @property
    def MOVING_AVERAGE_PERIODS(self) -> list[int]: return self.app_config.multi_head_arrangement.MOVING_AVERAGE_PERIODS

    # 过滤规则
    @property
    def ENABLE_WEAK_STOCK_FILTER(self) -> bool: return self.app_config.filter_rules.ENABLE_WEAK_STOCK_FILTER

    @property
    def EXEMPT_LEVELS(self) -> list[str]: return self.app_config.filter_rules.EXEMPT_LEVELS

    # 资金流
    @property
    def FUND_FLOW_PERIODS(self) -> list[int]: return self.app_config.fund_flow.FUND_FLOW_PERIODS

    # 技术指标
    @property
    def MACD_PARAMS(self) -> tuple[int, int, int]: return self.app_config.technical_indicators.MACD_PARAMS

    # 列名别名（需 parse_aliases 解析）
    @property
    def CODE_ALIASES(self) -> dict[str, str]: return parse_aliases(self.app_config.column_aliases.code_aliases)

    @property
    def NAME_ALIASES(self) -> dict[str, str]: return parse_aliases(self.app_config.column_aliases.name_aliases)

    @property
    def PRICE_ALIASES(self) -> dict[str, str]: return parse_aliases(self.app_config.column_aliases.price_aliases)

    # 研报
    @property
    def ENABLE_RESEARCH_REPORT_FILTER(self) -> bool: return self.app_config.research_report_filter.ENABLE_RESEARCH_REPORT_FILTER

    @property
    def RESEARCH_REPORT_MIN_COUNT(self) -> int: return self.app_config.research_report_filter.RESEARCH_REPORT_MIN_COUNT

    # 自选股
    @property
    def USER_FOCUS_STOCKS(self) -> str: return self.app_config.user_focus_stocks.USER_FOCUS_STOCKS

    # AShareHub
    @property
    def ASHAREHUB_API_KEY(self) -> str: return self.app_config.asharehub.API_KEY

    @property
    def ENABLE_CHIP_DISTRIBUTION(self) -> bool: return self.app_config.asharehub.ENABLE_CHIP_DISTRIBUTION

    @property
    def CHIP_LIMIT(self) -> int: return self.app_config.asharehub.CHIP_LIMIT

    @property
    def MONEYFLOW_RETRY(self) -> int: return self.app_config.asharehub.MONEYFLOW_RETRY

    @property
    def MONEYFLOW_PAGE_DELAY(self) -> float: return self.app_config.asharehub.MONEYFLOW_PAGE_DELAY

    # 宏观过滤
    @property
    def ENABLE_MACRO_FILTER(self) -> bool: return self.app_config.macro_filter.ENABLE_MACRO_FILTER

    @property
    def MACRO_FILTER_INDEX_SYMBOL(self) -> str: return self.app_config.macro_filter.INDEX_SYMBOL

    # K线
    @property
    def KLINE_HISTORY_DAYS(self) -> int: return self.app_config.kline_data.KLINE_HISTORY_DAYS

    # ── Dict 聚合属性（供 SignalManager / DataProcessingService 等使用） ──

    @property
    def FULL_BULL_WEIGHTS(self) -> dict:
        f = self.app_config.full_bull_scoring
        return {
            "MACD趋势": f.WEIGHT_ZERO_AXIS,
            "金叉信号": f.WEIGHT_STRATEGY_GOLDEN,
            "柱状动能": f.WEIGHT_MOMENTUM,
            "DIF斜率": f.WEIGHT_DIF_SLOPE,
            "背离信号": f.WEIGHT_DIVERGENCE,
            "量价配合": f.WEIGHT_VOLUME_PRICE,
            "K线形态": f.WEIGHT_KLINE_PATTERN,
        }

    @property
    def FULL_BULL_THRESHOLDS(self) -> dict:
        f = self.app_config.full_bull_scoring
        return {
            "fully_bull": f.CONCLUSION_FULL_BULL,
            "bullish": f.CONCLUSION_BULLISH,
            "oscillate": f.CONCLUSION_OSCILLATE,
        }

    @property
    def RULE_THRESHOLDS(self) -> dict:
        f = self.app_config.full_bull_scoring
        return {
            "divergence": f.RULE_DIVERGENCE_THRESHOLD,
            "winner_rate_high": f.RULE_WINNER_RATE_HIGH,
            "winner_rate_low": f.RULE_WINNER_RATE_LOW,
            "cost_resistance_ratio": f.RULE_COST_RESISTANCE_RATIO,
            "chip_concentrated_ratio": f.RULE_CHIP_CONCENTRATED_RATIO,
            "price_new_high_days": f.RULE_PRICE_NEW_HIGH_DAYS,
            "liq_veto_ratio": self.app_config.filter_rules.LIQ_VETO_RATIO,
        }

    @property
    def REGIME_DETECTION(self) -> dict:
        r = self.app_config.regime_detection
        return {
            "boll_narrow_ratio": r.BOLL_NARROW_RATIO,
            "oscillation_hist_std_ratio": r.OSCILLATION_HIST_STD_RATIO,
            "top_risk_ma20_deviation": r.TOP_RISK_MA20_DEVIATION,
            "oscillation_min_bars": r.OSCILLATION_MIN_BARS,
            "reversal_lookback": r.REVERSAL_LOOKBACK,
        }

    @property
    def DIVERGENCE_PARAMS(self) -> dict:
        d = self.app_config.divergence
        return {
            "base_distance": d.BASE_DISTANCE,
            "strength_threshold": d.STRENGTH_THRESHOLD,
            "decay_half_life": d.DECAY_HALF_LIFE,
            "slope_window": d.SLOPE_WINDOW,
        }

    @property
    def SCORING_PARAMS(self) -> dict:
        s = self.app_config.scoring_params
        return {
            "cross_decay_days": s.CROSS_DECAY_DAYS,
            "cross_decay_min": s.CROSS_DECAY_MIN,
            "kline_decay_days": s.KLINE_DECAY_DAYS,
            "kline_decay_min": s.KLINE_DECAY_MIN,
            "vol_norm_denominator": s.VOL_NORM_DENOMINATOR,
            "atr_stop_mult": s.ATR_STOP_MULT,
            "atr_t1_mult": s.ATR_T1_MULT,
            "atr_t2_mult": s.ATR_T2_MULT,
            "trailing_stop_high_ratio": s.TRAILING_STOP_HIGH_RATIO,
            "trailing_stop_lookback": s.TRAILING_STOP_LOOKBACK,
            "trailing_stop_high_lookback": s.TRAILING_STOP_HIGH_LOOKBACK,
            "expected_return_lookback": s.EXPECTED_RETURN_LOOKBACK,
        }

    @property
    def TECHNICAL_CONSTANTS(self) -> dict:
        t = self.app_config.technical_constants
        return {
            "atr_length": t.ATR_LENGTH,
            "adx_length": t.ADX_LENGTH,
            "rsi_length": t.RSI_LENGTH,
            "boll_length": t.BOLL_LENGTH,
            "boll_std": t.BOLL_STD,
            "stoch_k": t.STOCH_K,
            "stoch_d": t.STOCH_D,
            "kline_scan_window": t.KLINE_SCAN_WINDOW,
        }

    @property
    def POSITION_SIZING(self) -> dict:
        p = self.app_config.position_sizing
        f = self.app_config.filter_rules
        return {
            "max_single_position": p.MAX_SINGLE_POSITION,
            "kelly_fraction": p.KELLY_FRACTION,
            "default_win_rate": p.DEFAULT_WIN_RATE,
            "position_a": p.POSITION_A,
            "position_b": p.POSITION_B,
            "position_c": p.POSITION_C,
            "position_d": p.POSITION_D,
            "max_industry_exposure": p.MAX_INDUSTRY_EXPOSURE,
            "risk_budget": p.RISK_BUDGET,
            "max_drawdown_reduction": p.MAX_DRAWDOWN_REDUCTION,
            "liq_w_section": f.LIQ_W_SECTION,
            "liq_w_timeseries": f.LIQ_W_TIMESERIES,
            "liq_w_marketcap": f.LIQ_W_MARKETCAP,
            "liq_min_discount": f.LIQ_MIN_DISCOUNT,
        }

    # ── 工具方法 ────────────────────────────────────────────────────────

    def _ensure_directories(self) -> None:
        for d in (self.HOME_DIRECTORY, self.TEMP_DATA_DIRECTORY, self.LOG_DIR):
            os.makedirs(d, exist_ok=True)

    def get_db_connection_string(self) -> str:
        return (f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}")
