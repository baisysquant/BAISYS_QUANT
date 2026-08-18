"""BackTrading/replay_optimizer.py
波动率自适应退出优化 (Volatility-Adaptive Exit Optimization, VAEO)。

通过历史交易重放，寻找最大化利润因子的 T1/T2 组合。
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import List, Dict, Any, Tuple

from loguru import logger


def _reconstruct_completed_trades(trade_log: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将 flat trade_log (包含 buy/sell 动作) 重构成已完成的交易对 DataFrame。
    使用 FIFO (先进先出) 匹配逻辑。
    
    Returns:
        DataFrame with columns: symbol, buy_date, sell_date, buy_price, sell_price
    """
    # 分离买入和卖出
    buys = {sym: [] for sym in set(t['symbol'] for t in trade_log if t['action'] == 'buy')}
    
    completed_trades = []
    
    for t in trade_log:
        sym = t['symbol']
        dt = str(t['time'])[:10]
        price = t['price']
        
        if t['action'] == 'buy':
            buys.setdefault(sym, []).append({'date': dt, 'price': price})
        elif t['action'].startswith('sell'):
            queue = buys.get(sym, [])
            if queue:
                buy_entry = queue.pop(0) # FIFO
                completed_trades.append({
                    'symbol': sym,
                    'buy_date': buy_entry['date'],
                    'sell_date': dt,
                    'buy_price': buy_entry['price'],
                    'sell_price': price
                })
                
    return pd.DataFrame(completed_trades)


def simulate_single_trade(
    segment: pd.DataFrame,
    entry_price: float,
    entry_atr: float,
    t1_mult: float,
    t2_mult: float,
    sl_mult: float,
) -> float:
    """
    单笔交易重放逻辑：
    1. 设定 T1, T2, SL 绝对价位。
    2. 逐日检查 High/Low 触发情况。
    3. 触发 T1/T2 减仓并上移止损（保本金）。
    
    Args:
        segment: 从入场到出场的 K 线片段 (含 OHLC 列)。
        entry_price: 入场价。
        entry_atr: 入场时 ATR 值。
        t1_mult: T1 倍数。
        t2_mult: T2 倍数。
        sl_mult: 止损倍数。
        
    Returns:
        模拟实现盈亏。
    """
    t1_price = entry_price + entry_atr * t1_mult
    t2_price = entry_price + entry_atr * t2_mult
    sl_price = entry_price - entry_atr * sl_mult
    
    pos = 1.0 # 初始满仓
    realized_pnl = 0.0
    current_sl = sl_price
    
    # 忽略入场当天，从次日开始模拟
    for _, row in segment.iloc[1:].iterrows():
        # 1. 先检查止损 (优先级最高)
        if row['low'] <= current_sl:
            realized_pnl += pos * (current_sl - entry_price)
            return realized_pnl # 止损出局
            
        # 2. 检查 T2 (高位止盈)
        if row['high'] >= t2_price and pos >= 0.5:
            realized_pnl += 0.5 * (t2_price - entry_price)
            pos = 0.5
            current_sl = entry_price # 剩余仓位保本
            
        # 3. 检查 T1 (中位止盈)
        elif row['high'] >= t1_price and pos >= 0.5:
            realized_pnl += 0.5 * (t1_price - entry_price)
            pos = 0.5
            current_sl = entry_price # 剩余仓位保本
            
    # 模拟周期结束，若仍有持仓按收盘价结算
    if pos > 0:
        realized_pnl += pos * (segment.iloc[-1]['close'] - entry_price)
        
    return realized_pnl


def optimize_vol_exits(
    trade_log: List[Dict[str, Any]],
    kline_df: pd.DataFrame,
    sl_mult: float = 1.5,
) -> Tuple[float, float]:
    """
    通过历史交易重放，寻找最大化利润因子的 T1/T2 组合。
    
    Args:
        trade_log: 原始回测产生的所有交易记录 (List[dict])。
        kline_df: 所有股票的历史 K 线数据 (需包含 symbol, trade_date, high, low, close, ATR)。
        sl_mult: 止损倍数（固定，或从配置读取）。
        
    Returns:
        best_t1_mult, best_t2_mult
    """
    # 1. 重构已完成交易
    trades_df = _reconstruct_completed_trades(trade_log)
    if trades_df.empty:
        logger.warning("[VAEO] 无完整交易记录，返回默认值 (3.0, 5.0)")
        return 3.0, 5.0
        
    # 2. 构建 K 线字典，加速查找
    kline_dict = {}
    for sym, group in kline_df.groupby('symbol'):
        df = group.sort_values('trade_date')
        # 确保 trade_date 是字符串以便匹配
        df = df.copy()
        df['trade_date'] = df['trade_date'].astype(str).str[:10]
        kline_dict[sym] = df
        
    # 3. 定义搜索网格 (可根据性能调整步长)
    t1_candidates = np.arange(1.5, 4.5, 0.5)
    t2_candidates = np.arange(2.5, 7.0, 0.5)
    
    best_score = -np.inf
    best_params = (3.0, 5.0) # 初始默认值
    
    logger.info(f"[VAEO] 启动波动率自适应退出优化 (SL={sl_mult}, 交易数={len(trades_df)})...")
    
    for t1, t2 in product(t1_candidates, t2_candidates):
        if t2 <= t1: continue # T2 必须大于 T1
        
        total_pnl = 0.0
        wins = 0
        total_count = 0
        
        for _, trade in trades_df.iterrows():
            sym = trade.get('symbol')
            buy_date = str(trade.get('buy_date'))
            sell_date = str(trade.get('sell_date'))
            entry_price = trade.get('buy_price')
            
            if sym not in kline_dict or pd.isna(entry_price):
                continue
                
            # 获取该段 K 线
            df_sym = kline_dict[sym]
            segment = df_sym[(df_sym['trade_date'] >= buy_date) & (df_sym['trade_date'] <= sell_date)]
            
            if len(segment) < 2:
                continue
                
            # 从 K 线中获取入场日 ATR
            entry_row = segment.iloc[0]
            entry_atr = entry_row.get('ATR')
            if pd.isna(entry_atr) or entry_atr <= 0:
                continue
                
            sim_pnl = simulate_single_trade(segment, entry_price, entry_atr, t1, t2, sl_mult)
            total_pnl += sim_pnl
            if sim_pnl > 0: wins += 1
            total_count += 1
            
        if total_count == 0: continue
        
        # 计算该组参数的综合得分 (总盈亏 * 胜率加权)
        win_rate = wins / total_count
        score = total_pnl * (1 + win_rate)
        
        if score > best_score:
            best_score = score
            best_params = (t1, t2)
            
    logger.info(f"[VAEO] 优化完成: T1={best_params[0]:.1f}, T2={best_params[1]:.1f}, Score={best_score:.2f}")
    return best_params[0], best_params[1]
