import pandas as pd
import numpy as np


class AdvancedKDJAnalyzer:
    """
    KDJ多维度信号分析类
    提供全面的KDJ技术分析，包括多种经典及进阶信号
    """

    def __init__(self):
        self.signals_map = {
            "买入-极值J线反转": self._check_extreme_j_reversal,
            "买入-底背离金叉": self._check_bottom_divergence_cross,
            "买入-趋势确认金叉": self._check_trend_confirmation_cross,
            "买入-低位超卖金叉": self._check_oversold_cross,
            "买入-深度超卖反弹": self._check_deep_oversold_bounce,
            "卖出-J线高位拐头": self._check_j_top_turn,
            "观望-K线快速拉升": self._check_k_rapid_rise,
            "买入-三线聚合突破": self._check_three_line_convergence_breakout,
            "观望-死叉回踩支撑": self._check_death_cross_support,
            "卖出-J线极限值回归": self._check_j_limit_regression,
            "买入-背离信号": self._check_divergence_signal,
            "买入-振荡区间突破": self._check_oscillation_breakout,
            "观望-KDJ三线同步": self._check_kdj_synchronization,
            "买入-超卖修复启动": self._check_oversold_recovery
        }

    def calculate_stochastic(self, high, low, close, k_period=9, d_period=3):
        """
        计算KDJ指标
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        stoch_j = 3 * stoch_d - 2 * stoch_k

        return stoch_k, stoch_d, stoch_j

    def _calculate_kdj_with_derivatives(self, df):
        """
        计算KDJ及其衍生指标
        """
        df = df.copy()
        df['k_value'], df['d_value'], df['j_value'] = self.calculate_stochastic(
            df['high'], df['low'], df['close']
        )

        # 识别金叉和死叉
        df['golden_cross'] = (df['k_value'] > df['d_value']) & (df['k_value'].shift(1) <= df['d_value'].shift(1))
        df['death_cross'] = (df['k_value'] < df['d_value']) & (df['k_value'].shift(1) >= df['d_value'].shift(1))

        # 计算KDJ变化率（斜率）
        df['k_slope'] = df['k_value'].diff()
        df['d_slope'] = df['d_value'].diff()
        df['j_slope'] = df['j_value'].diff()

        df['k_change_rate'] = df['k_value'].pct_change() * 100
        df['d_change_rate'] = df['d_value'].pct_change() * 100
        df['j_change_rate'] = df['j_value'].pct_change() * 100

        return df

    def _check_extreme_j_reversal(self, df):
        """检查极值J线反转"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]
        prev_j = df['j_value'].iloc[-2]

        if prev_j < 0 and current_j > 5 and df['golden_cross'].iloc[-1]:
            return f"买入-极值J线反转 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_bottom_divergence_cross(self, df):
        """检查底背离金叉"""
        current_k = df['k_value'].iloc[-1]
        current_low = df['low'].iloc[-1]
        window = 10
        min_k_recent = df['k_value'].iloc[-window:-1].min()
        min_low_recent = df['low'].iloc[-window:-1].min()

        price_lower = current_low <= min_low_recent * 1.02
        k_higher = current_k > min_k_recent * 1.1

        if df['golden_cross'].iloc[-1] and price_lower and k_higher and current_k < 30:
            return f"买入-底背离金叉 (K={current_k:.1f}, J={df['j_value'].iloc[-1]:.1f})"
        return None

    def _check_trend_confirmation_cross(self, df):
        """检查趋势确认金叉"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        had_oversold = (df['k_value'].iloc[-5:-1] < 20).any() and (df['d_value'].iloc[-5:-1] < 20).any()
        above_ma5 = df['close'].iloc[-1] > df['close'].rolling(window=5).mean().iloc[-1]

        if had_oversold and df['golden_cross'].iloc[-1] and above_ma5:
            return f"买入-趋势确认金叉 (K={current_k:.1f}, J={df['j_value'].iloc[-1]:.1f})"
        return None

    def _check_oversold_cross(self, df):
        """检查低位超卖金叉"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        had_oversold = (df['k_value'].iloc[-5:-1] < 20).any() and (df['d_value'].iloc[-5:-1] < 20).any()

        if had_oversold and df['golden_cross'].iloc[-1]:
            return f"买入-低位超卖金叉 (K={current_k:.1f}, J={df['j_value'].iloc[-1]:.1f})"
        return None

    def _check_deep_oversold_bounce(self, df):
        """检查深度超卖反弹"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        if current_k < 10 and current_d < 10 and df['golden_cross'].iloc[-1]:
            return f"买入-深度超卖反弹 (K={current_k:.1f}, J={df['j_value'].iloc[-1]:.1f})"
        return None

    def _check_j_top_turn(self, df):
        """检查J线高位拐头"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        if current_j > 100 and df['j_slope'].iloc[-1] < 0 and df['j_slope'].iloc[-2] > 0:
            return f"卖出-J线高位拐头 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_k_rapid_rise(self, df):
        """检查K线快速拉升"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        if df['k_slope'].iloc[-1] > 8 and df['k_slope'].iloc[-2] < 3 and current_k < 80:
            return f"观望-K线快速拉升 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_three_line_convergence_breakout(self, df):
        """检查三线聚合突破"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        if abs(current_k - current_d) < 5 and abs(current_k - current_j) < 5:
            if df['k_slope'].iloc[-1] > 0 and df['d_slope'].iloc[-1] > 0:
                return f"买入-三线聚合向上突破 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_death_cross_support(self, df):
        """检查死叉回踩支撑"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        if df['death_cross'].iloc[-1] and current_k > 20 and current_k < 50:
            recent_support = df['k_value'].rolling(window=5).min().iloc[-3]
            if abs(current_k - recent_support) < 5:
                return f"观望-死叉回踩支撑 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_j_limit_regression(self, df):
        """检查J线极限值回归"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        if current_j > 120 or current_j < -20:
            if current_j > 120 and df['j_slope'].iloc[-1] < 0:
                return f"卖出-J线超买回归 (K={current_k:.1f}, J={current_j:.1f})"
            elif current_j < -20 and df['j_slope'].iloc[-1] > 0:
                return f"买入-J线超卖回归 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_divergence_signal(self, df):
        """检查背离信号"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        price_change = df['close'].iloc[-1] - df['close'].iloc[-2]  # 价格变化值
        k_change = df['k_value'].iloc[-1] - df['k_value'].iloc[-2]  # K值变化值

        # 底背离：价格下跌但K值上升
        if price_change < -1.0 and k_change > 5.0 and current_k < 30:
            return f"买入-底背离信号 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_oscillation_breakout(self, df):
        """检查振荡区间突破"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        lookback = 20
        k_max = df['k_value'].iloc[-lookback:].max()
        k_min = df['k_value'].iloc[-lookback:].min()

        if current_k > k_max * 0.95 and df['k_slope'].iloc[-1] > 0:
            return f"买入-振荡区间向上突破 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_kdj_synchronization(self, df):
        """检查KDJ三线同步"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        k_positive = df['k_slope'].iloc[-1] > 0
        d_positive = df['d_slope'].iloc[-1] > 0
        j_positive = df['j_slope'].iloc[-1] > 0

        if k_positive == d_positive == j_positive and k_positive:
            return f"观望-KDJ三线同向上 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def _check_oversold_recovery(self, df):
        """检查超卖修复启动"""
        current_k = df['k_value'].iloc[-1]
        current_d = df['d_value'].iloc[-1]
        current_j = df['j_value'].iloc[-1]

        oversold_period = sum(df['k_value'] < 20)

        if oversold_period > 5 and df['k_slope'].iloc[-1] > 0 and current_k > 30:
            return f"买入-超卖修复启动 (K={current_k:.1f}, J={current_j:.1f})"
        return None

    def calculate_kdj_signal_from_df(self, df: pd.DataFrame) -> str:
        """
        根据单只股票的DataFrame计算详细的KDJ信号。

        Args:
            df: 包含 'close', 'high', 'low' 列的股票历史数据DataFrame

        Returns:
            str: KDJ信号描述字符串，如 "买入-极值J线反转 (K=15.2, J=-2.1)"。若无信号则返回空字符串 ""。
        """
        if len(df) < 25:
            return ""

        df = self._calculate_kdj_with_derivatives(df)

        # 按优先级检查各种信号
        for signal_name, check_func in self.signals_map.items():
            signal = check_func(df)
            if signal:
                return signal

        return ""


# 示例用法
if __name__ == "__main__":
    # 创建一个模拟的数据集用于测试
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    # 模拟一个震荡后上涨的场景
    closes = [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(30)]
    highs = [c + abs(np.random.normal(0, 0.3)) for c in closes]
    lows = [c - abs(np.random.normal(0, 0.3)) for c in closes]

    test_df = pd.DataFrame({
        'date': dates,
        'close': closes,
        'high': highs,
        'low': lows
    })

    analyzer = AdvancedKDJAnalyzer()
    result = analyzer.calculate_kdj_signal_from_df(test_df)
    print(f"检测到的信号: {result}")

