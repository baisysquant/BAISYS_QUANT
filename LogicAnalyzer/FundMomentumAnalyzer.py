from typing import Dict, Any

import numpy as np
import pandas as pd


class FundMomentumAnalyzer:


    def __init__(self,
                 trend_weight: float = 0.4,
                 speed_weight: float = 0.6,
                 z_score_threshold: float = 1.0):
        """
        初始化.

        Args:
            trend_weight: 趋势因子权重.
            speed_weight: 速度/爆发因子权重.
            z_score_threshold: 异常值截断阈值 (防止极端值影响).
        """
        self.trend_weight = trend_weight
        self.speed_weight = speed_weight
        self.z_score_threshold = z_score_threshold

    def analyze(self, row: pd.Series) -> Dict[str, Any]:
        """
        对单只股票/单行数据进行分析.

        Args:
            row: 包含资金数据的Series.

        Returns:
            Dict: 分析结果.
        """

        # 提取标准化后的数值 (单位: 万元)
        T5 = self._safe_float(row.get('5日资金流入万元', row.get('5日资金流入', 0)))
        T10 = self._safe_float(row.get('10日资金流入万元', row.get('10日资金流入', 0)))
        T20 = self._safe_float(row.get('20日资金流入万元', row.get('20日资金流入', 0)))

        # 趋势因子 如果5日占了10日的80%，说明近期资金主导了中期趋势.
        participation_10 = T5 / T10 if T10 > 0 else 0
        participation_20 = T5 / T20 if T20 > 0 else 0

        factor_trend = np.clip(participation_10 * 0.5 + participation_20 * 0.25, 0, 1)

        # 强度因子 消除时间长度偏差, 看单位时间内的资金密度.
        avg_5 = T5 / 5
        avg_10 = T10 / 10
        avg_20 = T20 / 20

        # 计算环比增速 (加速度)
        # 避免除以0, 设置微小值 epsilon
        epsilon = 1e-5

        # 短期相对于中期的爆发倍数
        growth_vs_10 = (avg_5 - avg_10) / (avg_10 + epsilon)
        # 短期相对于长期的爆发倍数
        growth_vs_20 = (avg_5 - avg_20) / (avg_20 + epsilon)

        # 强度评分 (加权增速, 0-1映射)

        score_vs_10 = np.clip(growth_vs_10 / 0.5, 0, 1)
        score_vs_20 = np.clip(growth_vs_20 / 0.5, 0, 1)
        factor_speed = (score_vs_10 * 0.4) + (score_vs_20 * 0.6)

        # 综合动能评分
        momentum_score = (factor_trend * self.trend_weight) + (factor_speed * self.speed_weight)

        # 信号分类 (基于规则矩阵)
        signal = self._classify_signal(factor_trend, factor_speed, T5)

        return {
            '输入_5日总额(万元)': T5,
            '输入_10日总额(万元)': T10,
            '输入_20日总额(万元)': T20,
            '因子_趋势因子': factor_trend,
            '因子_强度因子': factor_speed,
            '综合_动能评分': momentum_score,
            '综合_交易信号': signal
        }

    def _classify_signal(self, trend: float, speed: float, t5: float) -> str:
        """
        基于因子矩阵判定信号.
        """
        # 定义阈值
        HIGH_TREND = 0.6
        LOW_TREND = 0.3
        HIGH_SPEED = 0.6
        LOW_SPEED = 0.3

        # 逻辑判定
        if t5 <= 0:
            return "资金流出"

        # 主升浪
        if trend >= HIGH_TREND and speed >= HIGH_SPEED:
            return "强势主升"

        # 主力回流/反转: 趋势低(之前弱), 但速度高(突然快)
        if trend < LOW_TREND and speed >= HIGH_SPEED:
            return "主力回流"

        # 动能衰竭: 趋势高(还在流入), 但速度低(变慢了)
        if trend >= HIGH_TREND and speed < LOW_SPEED:
            return "动能衰竭"

        # 温和流入: 速度一般, 趋势一般
        if speed >= LOW_SPEED and speed < HIGH_SPEED:
            return " 温和流入"

        # 默认
        return "观望"

    @staticmethod
    def _safe_float(val) -> float:
        """安全转浮点"""
        try:
            return float(val) if pd.notna(val) else 0.0
        except:
            return 0.0
