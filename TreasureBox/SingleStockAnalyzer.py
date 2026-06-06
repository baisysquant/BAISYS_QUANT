#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单只股票技术指标分析工具

让用户输入一个股票代码，从 akshare 下载数据并复用现有分析类，
输出 Excel 报告中所有的技术指标因子结论。

用法：
    python TreasureBox/SingleStockAnalyzer.py          # 交互模式
    python TreasureBox/SingleStockAnalyzer.py 000001   # 命令行参数模式

依赖：akshare, pandas, pandas-ta
"""

import sys
import os
import time
import random
from datetime import datetime, timedelta
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError):
        pass

import pandas as pd
import traceback
import akshare as ak


# ── 从项目现有模块导入 ─────────────────────────────────────────────────────
from DataManager.ShareCodeFormatMgr import format_stock_code
from LogicAnalyzer.SignalConstants import InvestmentRating
from ConfigParser import Config


def extract_pure_code(code: str) -> str:
    code_str = str(code).lower()
    for prefix in ("sh", "sz", "bj"):
        if code_str.startswith(prefix):
            return code_str[len(prefix) :]
    return code_str.zfill(6)


# ── 打印辅助函数 ───────────────────────────────────────────────────────────
WIDTH = 68


def print_header():
    print()
    print("=" * WIDTH)
    print("  单只股票技术指标分析工具")
    print("=" * WIDTH)


def print_field(label: str, value: Any):
    if value is not None and str(value).strip():
        print(f"    {label:<26} : {value}")


# ── 数据获取（复用 StockSyncEngine._fetch_kline_for_symbol 模式）─────────
def fetch_kline_data(symbol: str, days: int = 300) -> pd.DataFrame | None:
    """
    获取个股前复权 + 不复权数据，合并输出。
    完全复用 StockSyncEngine._fetch_kline_for_symbol 的 akshare API 调用模式。
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    for attempt in range(3):
        try:
            # 1) 前复权数据（价格）
            df_qfq = ak.stock_zh_a_hist_tx(
                symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq"
            )
            if df_qfq is None or df_qfq.empty:
                raise ValueError("空数据")

            expected = ["date", "open", "close", "high", "low", "amount"]
            if any(c not in df_qfq.columns for c in expected):
                raise ValueError(f"前复权数据缺失列: {df_qfq.columns.tolist()}")

            time.sleep(0.05)

            # 2) 不复权数据（成交量）
            df_norm = ak.stock_zh_a_hist_tx(
                symbol=symbol, start_date=start_date, end_date=end_date, adjust=""
            )
            if df_norm is None or df_norm.empty:
                raise ValueError("空数据")

            df_norm = df_norm[["date", "close", "amount"]].rename(
                columns={"close": "close_normal", "amount": "volume_normal"}
            )

            # 3) 合并
            df = pd.merge(df_qfq, df_norm, on="date", how="inner")
            if df.empty:
                raise ValueError("合并后无数据")

            # 4) 数值转换 & 计算调整后成交量
            for col in ["close", "close_normal", "amount", "volume_normal"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["adj_ratio"] = df["close"] / df["close_normal"].replace(0, pd.NA)
            df["volume"] = df["volume_normal"] * (df["amount"] / df["volume_normal"].replace(0, pd.NA))
            df.dropna(subset=["adj_ratio", "volume"], inplace=True)

            # 5) 标准化
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.rename(columns={"amount": "amount_adj"}, inplace=True)

            return df

        except (ValueError, KeyError, TypeError, ConnectionError, AttributeError) as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"  [RETRY] 第 {attempt + 1} 次失败，{wait} 秒后重试...")
                print(f"       {type(e).__name__}: {e}")
                time.sleep(wait)
            else:
                print(f"  [ERROR] 下载数据失败: {type(e).__name__}: {e}")
                traceback.print_exc()
                return None
    return None


# ── 分析单只股票（复用逻辑）─────────────────────────────────────────────
def analyze_stock(pure_code: str):
    symbol = format_stock_code(pure_code)

    # 1. 获取 K 线数据
    print(f"\n  >>> 正在从 akshare 下载数据 ({pure_code})...")
    df = fetch_kline_data(symbol, days=300)
    if df is None or df.empty or len(df) < 30:
        print("  [ERROR] 数据不足（至少需要 30 个交易日）")
        return

    print(f"  [OK] 获取到 {len(df)} 条日 K 线数据")
    print(f"      日期范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    # 2. 基础信息
    print_field("股票代码", pure_code)

    stock_name = pure_code
    try:
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df is not None and not spot_df.empty:
            match = spot_df[spot_df["代码"] == pure_code]
            if not match.empty:
                stock_name = match.iloc[0]["名称"]
    except (KeyError, ValueError, TypeError, IndexError, ConnectionError):
        try:
            info_df = ak.stock_individual_info_em(symbol=pure_code)
            if info_df is not None and not info_df.empty and len(info_df.columns) >= 2:
                first_col = info_df.columns[0]
                name_row = info_df[info_df[first_col].astype(str).str.contains("股票名称")]
                if not name_row.empty:
                    stock_name = name_row.iloc[0].iloc[1]
        except (KeyError, ValueError, TypeError, IndexError, ConnectionError):
            pass
    print_field("股票名称", stock_name)

    latest_price = df["close"].iloc[-1]
    print_field("最新价", f"{latest_price:.2f}")
    print_field("数据条数", len(df))

    # 3. 复用 TASignalProcessor 计算全部技术信号
    config = Config()
    second_params = getattr(config, "MACD_SECOND_PARAMS", (6, 13, 5))
    second_period_name = f"{second_params[0]}{second_params[1]}{second_params[2]}"

    hist_df = df.copy()
    hist_df["股票代码"] = pure_code

    from LogicAnalyzer.SignalManager import TASignalProcessor

    processor = TASignalProcessor(None, config=config)
    result = processor._process_single_stock(symbol, hist_df, second_params, second_period_name)

    if result is None:
        print("  [ERROR] 技术指标分析失败")
        return

    # 4. MACD 指标
    print_field("MACD_12269", result.get("macd_12269_signal", ""))
    print_field(f"MACD_{second_period_name}", result.get("macd_second_signal", ""))
    print_field("MACD_12269_DIF", f"{result.get('dif_12269', 0):.4f}")
    print_field(f"MACD_{second_period_name}_DIF", f"{result.get('dif_second', 0):.4f}")
    print_field("MACD_12269_动能", result.get("mom_12269", ""))
    print_field(f"MACD_{second_period_name}_动能", result.get("mom_second", ""))

    # 5. 完全多头评分
    bull_result = result.get("bull")
    if bull_result:
        details = bull_result.get("details", {})
        if details:
            for dim_key, dim_val in details.items():
                desc = dim_val.get("desc", "")
                score = dim_val.get("score", 0)
                print(f"    {dim_key:<20} : {score:>3}  ({desc})")

    # 6. KDJ / CCI / RSI / BOLL
    print_field("KDJ_Signal", result.get("kdj_signal", "无信号"))
    print_field("CCI_Signal", result.get("cci_signal", "无信号"))
    print_field("RSI_Signal", result.get("rsi_signal", "无信号"))
    print_field("BOLL_Signal", result.get("boll_signal", "无信号"))

    # 7. 汇总
    if bull_result:
        score = bull_result.get("score", 0)
        print(f"  综合评分: {score}  {InvestmentRating.from_score(score)}")
        print(f"  综合结论: {bull_result.get('conclusion', 'N/A')}")


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    print_header()

    # 用户输入（支持 | 分隔多个股票代码）
    if len(sys.argv) > 1:
        raw_input = sys.argv[1]
    else:
        raw_input = input(f"\n  请输入股票代码，多个请用 | 分隔 (如 000001|000002): ").strip()

    if not raw_input:
        print("  [ERROR] 未输入股票代码")
        return

    codes = [extract_pure_code(c.strip()) for c in raw_input.split("|") if c.strip()]
    print(f"  [-] 共 {len(codes)} 只股票: {', '.join(codes)}")

    for i, pure_code in enumerate(codes):
        print()
        print("=" * WIDTH)
        print(f"  >>> 第 {i + 1}/{len(codes)} 只: {pure_code}")
        analyze_stock(pure_code)

    print()
    print("=" * WIDTH)
    print("  全部查询完成")
    print("=" * WIDTH)
    print()


if __name__ == "__main__":
    main()
