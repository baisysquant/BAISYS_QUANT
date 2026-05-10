import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any


def calculate_full_bull_score(df: pd.DataFrame, thresholds: Dict[str, int] = None) -> Dict[str, Any]:
    """
    A股多头排列“共振-容错”量化评分

    Args:
        df: 包含OHLCV及均线数据的DataFrame
        thresholds: 可选的阈值配置字典，包含以下键：
            - full_bull: 完全主升阈值（默认85）
            - trend_acceleration: 趋势加速阈值（默认65）
            - trend_oscillation: 趋势震荡阈值（默认45）

    Returns:
        Dict: 包含评级、因子详情、状态的分析结果
    """

    # --- 1. 参数校准与数据预处理 ---
    # 自适应检测日期列（兼容 trade_date / date / 日期 / datetime 等命名）
    _date_col_candidates = ['trade_date', 'date', '日期', 'datetime', 'Date', 'TRADE_DATE']
    _date_col = next((c for c in _date_col_candidates if c in df.columns), None)
    if _date_col is None:
        return _generate_empty_result(f"缺少日期列，实际列: {list(df.columns)[:10]}")
    # 统一重命名为 trade_date
    if _date_col != 'trade_date':
        df = df.rename(columns={_date_col: 'trade_date'})

    # 确保 trade_date 为可排序的字符串格式 YYYY-MM-DD
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df = df.sort_values('trade_date').copy()

    if len(df) < 30:
        return _generate_empty_result("数据不足30个交易日")

    # 若均线列不存在（原始K线表未预计算），则按需计算
    for _period in [5, 10, 20, 30, 60, 90, 120]:
        _col = f'MA{_period}'
        if _col not in df.columns:
            df[_col] = df['close'].rolling(window=_period, min_periods=1).mean()
    if 'MA_Volume_5' not in df.columns:
        df['MA_Volume_5'] = df['volume'].rolling(window=5, min_periods=1).mean()

    # 提取当前最新一行数据
    latest = df.iloc[-1]
    close_price = latest['close']

    # --- 2. 定义因子计算函数 ---

    def _trend_skeleton_score() -> Tuple[int, str]:
        """因子一：趋势骨架 (40分) - 脊梁"""
        # 中期多头 (20分): MA30 > MA60 > MA90 (允许2%容错)
        ma30, ma60, ma90 = latest['MA30'], latest['MA60'], latest['MA90']
        base_mid = (ma30 > ma60 * 0.98) and (ma60 > ma90 * 0.98)
        # 斜率加分 (10分)
        slope_30 = (ma30 - df['MA30'].iloc[-6]) / df['MA30'].iloc[-6]
        slope_60 = (ma60 - df['MA60'].iloc[-11]) / df['MA60'].iloc[-11]
        slope_benefit = 10 if (slope_30 > 0 and slope_60 > 0) else 0

        # 长期向上 (10分): MA120斜率向上
        ma120 = latest['MA120']
        long_up = 10 if ma120 > df['MA120'].iloc[-21] else 0

        # 位置确认 (10分): 收盘价 > MA20
        price_pos = 10 if close_price > latest['MA20'] else 0

        total = (20 if base_mid else 0) + slope_benefit + long_up + price_pos
        desc = f"骨架得分: {total}/40 (中期:{'✓' if base_mid else '✗'}, 长期:{'↑' if long_up else '→'}, 位置:{'↑' if price_pos else '↓'})"
        return total, desc

    def _short_attack_score() -> Tuple[int, str]:
        """因子二：短期攻击力 (30分) - 肌肉"""
        # 标准排列 (15分): MA5 > MA10 > MA20
        ma5, ma10, ma20 = latest['MA5'], latest['MA10'], latest['MA20']
        standard = 15 if (ma5 > ma10) and (ma10 > ma20) else 0

        # 斜率向上 (15分): MA5不拐头向下
        # 计算3日斜率
        if len(df) >= 5:
            y = df['MA5'].iloc[-3:].values
            x = np.arange(len(y))
            if len(y) > 1:
                z = np.polyfit(x, y, 1)
                slope = z[0]
                momentum = 15 if slope > 0 else 0
            else:
                momentum = 0
        else:
            momentum = 0

        total = standard + momentum
        desc = f"攻击得分: {total}/30 (排列:{standard}, 动能:{momentum})"
        return total, desc

    def _perfect_bonus_score() -> Tuple[int, str]:
        """因子三：完美度加成 (10分) - 锦上添花"""
        # 不再要求全周期完美，改为梯度加分
        weights = [5, 3, 2]  # MA5>10, MA10>20, MA20>30 的权重
        conditions = [
            latest['MA5'] > latest['MA10'],
            latest['MA10'] > latest['MA20'],
            latest['MA20'] > latest['MA30']
        ]
        score = sum(w for w, c in zip(weights, conditions) if c)

        # 股价在所有均线上方 (额外2分)
        above_all = 2 if close_price > max(latest['MA5'], latest['MA10'], latest['MA20']) else 0

        total = min(10, score + above_all)
        desc = f"完美度: {total}/10 (梯度匹配)"
        return total, desc

    def _oscillation_forgive_score() -> Tuple[int, str]:
        """因子四：震荡容错 (20分) - 核心容错逻辑"""
        ma5, ma10 = latest['MA5'], latest['MA10']
        vol, vol_ma5 = latest['volume'], latest['MA_Volume_5']

        # 均线收敛 (10分): 短期均线粘合 (动态阈值 3%)
        convergence = 0
        if max(ma5, ma10) > 0:
            convergence_ratio = abs(ma5 - ma10) / max(ma5, ma10)
            # 动态打分：越粘合分数越高
            convergence = 10 if convergence_ratio < 0.03 else (5 if convergence_ratio < 0.05 else 0)

        # 缩量回踩 (10分): 跌不动了，洗盘
        # 条件：缩量(<MA5量的80%) 且 价格在MA20上方
        is_shrinking = (vol < vol_ma5 * 0.8) if vol_ma5 > 0 else False
        is_above_ma20 = close_price > latest['MA20']
        volume_check = 10 if (is_shrinking and is_above_ma20) else 0

        total = convergence + volume_check
        desc = f"容错分: {total}/20 (收敛:{'✓' if convergence else '✗'}, 缩量:{'✓' if volume_check else '✗'})"
        return total, desc

    def _risk_control_check() -> Tuple[bool, str]:
        """风控与过滤器 (黑名单) - 采用动态流动性评估"""
        # 1. 动态流动性枯竭判断 (基于个股历史均量)
        # 获取历史成交量数据 (例如最近120个交易日)
        historical_vol_window = 120
        if len(df) < historical_vol_window:
            # 如果数据不够，使用现有数据的最大窗口
            historical_vol_window = min(len(df), 60)

        historical_volumes = df['volume'].iloc[-historical_vol_window:]

        # 计算历史平均成交量
        mean_vol = historical_volumes.mean()

        # 计算当前成交量相对于历史平均水平的比例
        current_vol = latest['volume']
        vol_ratio_to_mean = current_vol / mean_vol if mean_vol > 0 else float('inf')

        # 设定一个比例阈值 (例如 20%，表示当前量低于历史均量的20%，认为是枯竭)
        # 这个阈值可以根据回测结果调整
        liquidity_ratio_threshold = 0.2

        if vol_ratio_to_mean < liquidity_ratio_threshold:
            return False, f"流动性枯竭 (当前量: {current_vol:.0f}, 历史均量: {mean_vol:.0f}, 比例: {vol_ratio_to_mean:.2%})"

        # 2. 骨架塌陷 (中期趋势走坏)
        if latest['MA30'] < latest['MA60']:
            return False, "中期骨架塌陷"

        # 3. ST风险 (通常在数据层已过滤，这里双重检查)
        stock_name = str(latest.get('name', ''))
        if 'ST' in stock_name or '*' in stock_name:
            return False, "ST风险"

        return True, "通过"

    # --- 3. 执行评分逻辑 ---

    # 风控检查 (一票否决)
    is_safe, risk_reason = _risk_control_check()
    if not is_safe:
        return _generate_empty_result(f"风控拦截: {risk_reason}")

    # 计算各因子得分
    score_trend, desc_trend = _trend_skeleton_score()
    score_attack, desc_attack = _short_attack_score()
    score_bonus, desc_bonus = _perfect_bonus_score()
    score_forgive, desc_forgive = _oscillation_forgive_score()

    total_score = score_trend + score_attack + score_bonus + score_forgive

    # --- 4. 信号定级 - 使用描述性词语 ---
    # 从配置中读取阈值，如果未提供则使用默认值
    if thresholds is None:
        thresholds = {}
    
    full_bull_threshold = thresholds.get('full_bull', 85)
    trend_acceleration_threshold = thresholds.get('trend_acceleration', 65)
    trend_oscillation_threshold = thresholds.get('trend_oscillation', 45)

    if total_score >= full_bull_threshold:
        level = "完全主升"
    elif total_score >= trend_acceleration_threshold:
        level = "趋势加速"
    elif total_score >= trend_oscillation_threshold:
        level = "趋势震荡"
    else:
        level = "趋势观望"

    return {
        "level": level,
        "factors": {
            "trend_skeleton": {"desc": desc_trend, "score": score_trend},
            "short_attack": {"desc": desc_attack, "score": score_attack},
            "perfect_bonus": {"desc": desc_bonus, "score": score_bonus},
            "oscillation_forgive": {"desc": desc_forgive, "score": score_forgive}
        },
        "status": "SUCCESS"
    }


def _generate_empty_result(reason: str) -> Dict[str, Any]:
    """生成空结果"""
    return {
        "level": "趋势观望",
        "factors": {},
        "status": "FAILED",
    }

