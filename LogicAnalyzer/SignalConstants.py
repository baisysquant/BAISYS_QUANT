"""
信号字符串常量中心

统一管理所有技术指标信号、评级、趋势分类的中文字符串，
避免魔法字符串散落在 8+ 个文件中。
所有业务模块应引用此类而非直接书写字面量。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MACDSignals:
    """MACD 金叉/死叉信号"""
    GOLDEN_CROSS_ABOVE_ZERO = "零轴上金叉"
    GOLDEN_CROSS_BELOW_ZERO = "零轴下金叉"
    DEATH_CROSS_ABOVE_ZERO = "零轴上死叉"
    DEATH_CROSS_BELOW_ZERO = "零轴下死叉"


    @classmethod
    def golden_cross_label(cls, dif: Any, dea: Any) -> Any:  # noqa: ANN401
        return np.where((dif > 0) & (dea > 0), cls.GOLDEN_CROSS_ABOVE_ZERO, cls.GOLDEN_CROSS_BELOW_ZERO)

    @classmethod
    def death_cross_label(cls, dead: Any, dif: Any, dea: Any) -> Any:  # noqa: ANN401
        return np.where(dead, np.where((dif < 0) & (dea < 0), cls.DEATH_CROSS_BELOW_ZERO, cls.DEATH_CROSS_ABOVE_ZERO), "")


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


from DataManager.Indicators import TrendLevels  # noqa: F401


class Divergence:
    """背离信号"""
    TOP_DIVERGENCE = "顶背离"
    BOTTOM_DIVERGENCE = "底背离"


class MACDTrend:
    """MACD 趋势级别"""
    SUPER_STRONG = "指标超强"
    STRONG = "指标强势"
    WEAK = "指标弱势"
    SUPER_WEAK = "指标超弱"
