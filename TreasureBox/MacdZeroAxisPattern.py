#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MACD 零轴完整反弹形态扫描工具

扫描逻辑：下穿零轴(①) → 零轴下金叉(②) → 上穿零轴(③)
找出最近一次上穿零轴(③)，且在此前存在完整的 下穿(①)→金叉(②)→上穿(③) 序列的股票

输出：股票代码、股票简称、零轴下穿日期、零轴下金叉日期、上穿零轴日期
"""

from __future__ import annotations

import sys
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import talib
from sqlalchemy import text

# 把项目根目录加入 sys.path，以便导入通用模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataManager.DbEngine import get_engine
from DataManager.ColumnNames import ColumnNames
from UtilsManager.CodeNormalizer import CodeNormalizer
from UtilsManager.LoggerManager import get_logger

logger = get_logger(__name__)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """使用 TA-Lib 计算 MACD，返回包含 DIF、DEA、MACD(柱) 的 DataFrame"""
    dif, dea, macd = talib.MACD(
        close.to_numpy(dtype=float),
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )
    return pd.DataFrame({
        "DIF": dif,
        "DEA": dea,
        "MACD": macd * 2,  # TA-Lib 返回的是 DIF-DEA，乘2还原为常见的 MACD 柱状图
    }, index=close.index)


def detect_cross(series1: pd.Series, series2: pd.Series, direction: str) -> pd.Series:
    """
    检测 series1 穿越 series2
    direction: 'up' 表示上穿(series1 从下往上穿 series2)，'down' 表示下穿
    返回布尔 Series，True 表示当根 K 线发生穿越
    """
    prev_diff = series1.shift(1) - series2.shift(1)
    curr_diff = series1 - series2

    if direction == "up":
        return (prev_diff <= 0) & (curr_diff > 0)
    else:  # down
        return (prev_diff >= 0) & (curr_diff < 0)


def find_macd_zero_axis_pattern(symbol: str, df: pd.DataFrame) -> Optional[dict]:
    """
    在单只股票的 K 线数据中寻找最近一次完整的 零轴反弹形态：
    下穿零轴(①) → 零轴下金叉(②) → 上穿零轴(③)

    返回最近一次完整形态的三个日期，若不存在则返回 None
    """
    if len(df) < 60:
        return None

    # 计算 MACD
    macd_df = compute_macd(df["close"])
    df = df.copy()
    df["DIF"] = macd_df["DIF"]
    df["DEA"] = macd_df["DEA"]
    df["MACD"] = macd_df["MACD"]

    # 1. 识别所有零轴穿越事件
    # DIF 相对 0 轴的穿越
    zero_cross_up = detect_cross(df["DIF"], pd.Series(0, index=df.index), "up")   # 上穿零轴
    zero_cross_down = detect_cross(df["DIF"], pd.Series(0, index=df.index), "down")  # 下穿零轴

    # 2. 识别所有金叉/死叉事件（DIF 穿 DEA）
    golden_cross = detect_cross(df["DIF"], df["DEA"], "up")   # 金叉
    death_cross = detect_cross(df["DIF"], df["DEA"], "down")  # 死叉

    # 3. 标记零轴下金叉：金叉发生时 DIF < 0 且 DEA < 0
    golden_below_zero = golden_cross & (df["DIF"] < 0) & (df["DEA"] < 0)

    # 4. 收集所有关键事件的日期索引
    zero_down_dates = df.index[zero_cross_down].tolist()      # 下穿零轴
    golden_below_dates = df.index[golden_below_zero].tolist() # 零轴下金叉
    zero_up_dates = df.index[zero_cross_up].tolist()          # 上穿零轴

    if not zero_up_dates:
        return None

    # 过滤：最近一次上穿零轴必须在近 1 个月内
    one_month_ago = df.index[-1] - timedelta(days=30)
    recent_zero_up = [d for d in zero_up_dates if d >= one_month_ago]
    if not recent_zero_up:
        return None

    # 5. 从最近一次上穿零轴向前回溯，寻找完整序列
    # 遍历每一次上穿零轴（从最近到最早）
    for zero_up_idx in reversed(zero_up_dates):
        # 找到该上穿之前最近的一次零轴下金叉
        golden_before = [d for d in golden_below_dates if d < zero_up_idx]
        if not golden_before:
            continue
        golden_idx = max(golden_before)  # 最近的一次零轴下金叉

        # 再找到该金叉之前最近的一次下穿零轴
        zero_down_before = [d for d in zero_down_dates if d < golden_idx]
        if not zero_down_before:
            continue
        zero_down_idx = max(zero_down_before)  # 最近的一次下穿零轴

        # 时间顺序校验：下穿 < 金叉 < 上穿
        if zero_down_idx < golden_idx < zero_up_idx:
            # 找到完整形态，返回最近一次的
            return {
                "symbol": zero_up_idx,  # 占位，外层会填入 symbol
                "zero_axis_down_date": zero_down_idx.strftime("%Y-%m-%d"),
                "zero_axis_golden_date": golden_idx.strftime("%Y-%m-%d"),
                "zero_axis_up_date": zero_up_idx.strftime("%Y-%m-%d"),
            }

    return None


def scan_all_symbols() -> list[dict]:
    """扫描全市场股票，返回符合条件的股票列表"""
    from ConfigParser import Config
    # config.ini 在项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.ini")
    config = Config(config_path)
    engine = get_engine(config)
    cn = ColumnNames()

    # 1. 获取所有股票代码
    with engine.connect() as conn:
        sql = text("SELECT DISTINCT symbol FROM stock_daily_kline ORDER BY symbol")
        symbols = pd.read_sql(sql, conn)["symbol"].tolist()

    logger.info(f"共获取到 {len(symbols)} 只股票，开始扫描...")

    results = []
    processed = 0

    for symbol in symbols:
        try:
            # 读取该股票的完整 K 线数据
            with engine.connect() as conn:
                sql = text("""
                SELECT trade_date, symbol, open, close, high, low, volume
                FROM stock_daily_kline
                WHERE symbol = :symbol
                ORDER BY trade_date
                """)
                df = pd.read_sql(sql, conn, params={"symbol": symbol}, parse_dates=["trade_date"])
                df.set_index("trade_date", inplace=True)

            if len(df) < 60:
                continue

            pattern = find_macd_zero_axis_pattern(symbol, df)
            if pattern:
                # 获取股票简称
                with engine.connect() as conn:
                    sql_name = text("""
                    SELECT stock_name FROM stock_basic_info_sw
                    WHERE stock_code = :code
                    LIMIT 1
                    """)
                    name_row = conn.execute(sql_name, {"code": symbol[2:]}).fetchone()
                    stock_name = name_row[0] if name_row else symbol

                results.append({
                    "symbol": symbol,
                    "stock_name": stock_name,
                    "zero_axis_down_date": pattern["zero_axis_down_date"],
                    "zero_axis_golden_date": pattern["zero_axis_golden_date"],
                    "zero_axis_up_date": pattern["zero_axis_up_date"],
                })
                logger.info(f"发现形态: {symbol} {stock_name} "
                            f"下穿={pattern['zero_axis_down_date']} "
                            f"金叉={pattern['zero_axis_golden_date']} "
                            f"上穿={pattern['zero_axis_up_date']}")

        except Exception as e:
            logger.warning(f"处理 {symbol} 时出错: {e}")
            continue

        processed += 1
        if processed % 500 == 0:
            logger.info(f"已处理 {processed}/{len(symbols)} 只股票...")

    return results


def export_to_excel(results: list[dict], output_path: str) -> None:
    """导出结果到 Excel"""
    if not results:
        logger.warning("无结果，跳过导出")
        return

    df = pd.DataFrame(results)
    df.columns = ["股票代码", "股票简称", "零轴下穿日期", "零轴下金叉日期", "上穿零轴日期"]
    df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info(f"结果已导出到: {output_path}")


def main():
    logger.info("开始扫描 MACD 零轴完整反弹形态 (下穿→零轴下金叉→上穿)")
    results = scan_all_symbols()

    if results:
        output_path = os.path.join(
            os.path.expanduser("~/Downloads"),
            f"MACD零轴完整反弹形态_{date.today().strftime('%Y%m%d')}.xlsx"
        )
        export_to_excel(results, output_path)
        print(f"\n扫描完成，共发现 {len(results)} 只符合条件的股票")
        print(f"结果已保存至: {output_path}")
    else:
        print("未发现符合条件的股票")


if __name__ == "__main__":
    main()