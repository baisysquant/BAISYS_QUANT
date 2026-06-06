"""
信号字符串常量中心

统一管理所有技术指标信号、评级、趋势分类的中文字符串，
避免魔法字符串散落在 8+ 个文件中。
所有业务模块应引用此类而非直接书写字面量。
"""


class MACDSignals:
    """MACD 金叉/死叉信号"""
    GOLDEN_CROSS_ABOVE_ZERO = "零轴上金叉"
    GOLDEN_CROSS_BELOW_ZERO = "零轴下金叉"
    DEATH_CROSS_ABOVE_ZERO = "零轴上死叉"
    DEATH_CROSS_BELOW_ZERO = "零轴下死叉"

    # 多头持续 / 空头死叉
    BULL_CONTINUE = "多头持续"
    BEAR_DEATH_CROSS = "空头/死叉"

    @classmethod
    def golden_cross_label(cls, dif, dea):
        import numpy as np
        return np.where((dif > 0) & (dea > 0), cls.GOLDEN_CROSS_ABOVE_ZERO, cls.GOLDEN_CROSS_BELOW_ZERO)

    @classmethod
    def death_cross_label(cls, dead, dif, dea):
        import numpy as np
        return np.where(dead, np.where((dif < 0) & (dea < 0), cls.DEATH_CROSS_BELOW_ZERO, cls.DEATH_CROSS_ABOVE_ZERO), "")


class MACDMomentum:
    """MACD 动能状态"""
    ACCELERATE_UP = "加速上涨 (红柱加长)"
    DECELERATE_UP = "减速上涨 (红柱缩短)"
    ACCELERATE_DOWN = "加速下跌 (绿柱加长)"
    DECELERATE_DOWN = "减速下跌 (绿柱缩短)"


class KLineLevels:
    """K 线形态反转级别"""
    STRONG_REVERSAL = "强反转"
    MEDIUM_REVERSAL = "中反转"
    WEAK_SIGNAL = "弱信号"
    CONTINUOUS = "持续"

    LEVEL_ORDER = {STRONG_REVERSAL: 0, MEDIUM_REVERSAL: 1, WEAK_SIGNAL: 2, CONTINUOUS: 3}


class KLineDirection:
    """K 线方向"""
    BULLISH = "看涨"
    BEARISH = "看跌"


class TrendLevels:
    """多头排列趋势四档定级"""
    FULL_BULL = "完全主升"
    TREND_ACCELERATION = "趋势加速"
    TREND_OSCILLATION = "趋势震荡"
    TREND_WATCH = "趋势观望"

    @classmethod
    def all_levels(cls):
        return [cls.FULL_BULL, cls.TREND_ACCELERATION, cls.TREND_OSCILLATION, cls.TREND_WATCH]


class BullArrangement:
    """均线多头排列标记"""
    PERFECT_BULL = "完全多头排列"
    BULL_TREND = "多头排列趋势"
    YES = "是"
    NO = "否"


class Divergence:
    """背离信号"""
    TOP_DIVERGENCE = "顶背离"
    BOTTOM_DIVERGENCE = "底背离"


class InvestmentRating:
    """综合投资评级"""
    STRONG_BUY = "[强烈买入]"
    BUY_ON_DIP = "[逢低布局]"
    WATCH = "[观望为主]"
    AVOID = "[回避/做空]"

    STRONG_BUY_PLAIN = "强烈买入"
    BUY_PLAIN = "买入"
    CAUTION_BUY = "谨慎买入"
    POTENTIAL_BUY = "潜在买入"
    LIGHT_POSITION = "轻仓试探"
    WATCH_PLAIN = "观望"

    @classmethod
    def from_score(cls, score: int) -> str:
        if score >= 80:
            return cls.STRONG_BUY
        elif score >= 60:
            return cls.BUY_ON_DIP
        elif score >= 40:
            return cls.WATCH
        else:
            return cls.AVOID


class Conclusion:
    """综合结论"""
    TOP_RISK = "D: 顶部风险"
    BULLISH_CAUTION = "B: 偏多(注意顶部风险)"
    BULLISH = "B: 偏多"
    SIDEWAYS = "C: 多空拉锯"

    # 评分引擎结论
    TOP_RISK_DETAIL = "D: 顶部风险: 顶背离+见顶形态+缩量"
    TREND_PULLBACK = "(趋势中注意回调)"
    RESONANCE = "共振"
    VOLUME_CONFIRM = "量价确认"
    FAKE_BREAKOUT = "假突破预警"
    RANGE_BREAKOUT = "横盘突破趋势启动"
    MOMENTUM_DECAY = "上涨力度衰减"
    WAIT_CONFIRM = "等待K线确认"
    CHIP_RISK_PREFIX = "筹码风险:获利"
    CHIP_RESISTANCE = "筹码阻力位"


class FullBullScoring:
    """完全多头评分维度键名（MACDAnalyzer 内部）"""
    ZERO_AXIS = "零轴条件"
    STRATEGIC_GOLDEN = "战略金叉"
    TACTICAL_GOLDEN = "战术金叉"
    MOMENTUM = "动能"
    DIF_SLOPE = "DIF斜率"
    DIVERGENCE = "背离信号"
    VOLUME_PRICE = "量价配合"
    KLINE_PATTERN = "K线形态评分"


class CombinedSignal:
    """组合背离信号"""
    STRATEGIC_BOTTOM_DIV_CROSS = "战略底背离 + 战术金叉确认 (强烈买入信号)"
    STRATEGIC_TOP_DIV_CROSS = "战略顶背离 + 战术死叉确认 (强烈卖出信号)"
    DUAL_BOTTOM_DIV = "双重底背离 (强烈买入关注)"
    DUAL_TOP_DIV = "双重顶背离 (强烈卖出预警)"
    STRATEGY_BOTTOM_DIV = "12269 底背离 (战略买入预警)"
    STRATEGY_TOP_DIV = "12269 顶背离 (战略卖出预警)"
    TACTICAL_BOTTOM_DIV_BULL = "{} 底背离 (可考虑买入)"
    TACTICAL_BOTTOM_DIV_BEAR = "{} 底背离 (大趋势偏空，谨慎)"
    TACTICAL_TOP_DIV_NEUTRAL = "{} 顶背离 (需结合大趋势)"
    TACTICAL_TOP_DIV_BEAR = "{} 顶背离 (可考虑卖出)"


class CCILevels:
    """CCI 指标分类"""
    EXTREME_OVERBOUGHT = "极度超买"
    STRONG_OVERBOUGHT = "强势超买"
    NORMAL = "常态波动"
    WEAK_OVERSOLD = "弱势超卖"
    EXTREME_OVERSOLD = "极度超卖"


class VolumePrice:
    """量价关系"""
    VOLUME_PRICE_UP = "量价齐升"
    PRICE_UP_VOL_DOWN = "价涨量缩"
    VOLUME_DOWN_PRICE_DOWN = "放量下跌"
    SHRINK_DOWN = "缩量下跌"


class BOLLSignals:
    """布林带信号"""
    LOW_VOLATILITY = "低波/缩口"
    NORMAL = "常态/张口"


class RSISignals:
    """RSI 信号"""
    DIV_PREFIX = "RSI底背离!"
    DEFAULT_PREFIX = "RSI="


class KDJSignals:
    """KDJ 信号模式（KDJAnalyzer 输出）"""
    EXTREME_J_REVERSAL = "买入-极值J线反转"
    BOTTOM_DIV_CROSS = "买入-底背离金叉"
    TREND_CONFIRM_CROSS = "买入-趋势确认金叉"
    OVERSOLD_CROSS = "买入-低位超卖金叉"
    DEEP_OVERSOLD_BOUNCE = "买入-深度超卖反弹"
    J_HIGH_TURN = "卖出-J线高位拐头"
    FAST_RALLY = "观望-K线快速拉升"
    THREE_LINE_BREAKOUT = "买入-三线聚合向上突破"
    DEATH_CROSS_SUPPORT = "观望-死叉回踩支撑"
    J_OVERBOUGHT_RETURN = "卖出-J线超买回归"
    J_OVERSOLD_RETURN = "买入-J线超卖回归"
    BOTTOM_DIV_SIGNAL = "买入-底背离信号"
    OSCILLATION_BREAKOUT = "买入-振荡区间向上突破"
    THREE_LINE_UP = "观望-KDJ三线同向上"
    OVERSOLD_REPAIR = "买入-超卖修复启动"


class KLinePatternCN:
    """K 线形态中文名映射（ta-lib → 中文）"""
    MAPPING = {
        'CDL_ENGULFING': '吞没形态',
        'CDL_HIKKAKE': '对策(类岛形反转)',
        'CDL_MORNINGSTAR': '早晨之星',
        'CDL_MORNINGDOJISTAR': '早晨十字星',
        'CDL_EVENINGSTAR': '黄昏之星',
        'CDL_EVENINGDOJISTAR': '黄昏十字星',
        'CDL_ABANDONEDBABY': '弃婴(岛形反转)',
        'CDL_DARKCLOUDCOVER': '乌云盖顶',
        'CDL_PIERCING': '刺透形态',
        'CDL_3WHITESOLDIERS': '红三兵',
        'CDL_3BLACKCROWS': '三只乌鸦',
        'CDL_HARAMI': '孕线',
        'CDL_HARAMICROSS': '十字孕线',
        'CDL_BREAKAWAY': '脱离(类旗形)',
        'CDL_KICKING': '踢形态',
        'CDL_DOJI': '十字星',
        'CDL_HAMMER': '锤子线',
        'CDL_HANGINGMAN': '上吊线',
        'CDL_SHOOTINGSTAR': '射击之星',
        'CDL_INVERTEDHAMMER': '倒锤子线',
        'CDL_DRAGONFLYDOJI': '蜻蜓十字',
        'CDL_GRAVESTONEDOJI': '墓碑十字',
        'CDL_LONGLEGGEDDOJI': '长脚十字',
        'CDL_SPINNINGTOP': '纺锤线',
        'CDL_TAKURI': '托里(类锤子)',
        'CDL_RISEFALL3METHODS': '上升/下降三法',
        'CDL_GAPSIDESIDEWHITE': '跳空并列阳线',
        'CDL_XSIDEGAP3METHODS': '侧跳空三法',
        'CDL_MATHOLD': '待变(类旗形)',
    }


class MarketSentiment:
    """市场情绪判断"""
    OVERALL_BULLISH = "整体偏多"
    OVERALL_BEARISH = "整体偏空"
