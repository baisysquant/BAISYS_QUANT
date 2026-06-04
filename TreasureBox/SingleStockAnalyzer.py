#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单只股票技术指标分析工具

让用户输入一个股票代码，从 akshare 下载数据并复用现有分析类，
输出 Excel 报告中所有的技术指标因子结论。

用法：
    python TreasureBox/SingleStockAnalyzer.py
    或从项目根目录：
    python -m TreasureBox.SingleStockAnalyzer

依赖：
    pip install akshare pandas pandas-ta
"""

import sys
import os
from datetime import datetime, timedelta

# 确保能找到项目根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import akshare as ak
import pandas_ta as ta


def _classify_cci_level(cci_value: float) -> str:
    if pd.isna(cci_value):
        return "N/A"
    if cci_value > 200:
        return f"极度超买 ({cci_value:.2f})"
    elif cci_value >= 100:
        return f"强势超买 ({cci_value:.2f})"
    elif cci_value > -100:
        return ""
    elif cci_value >= -200:
        return f"弱势超卖 ({cci_value:.2f})"
    else:
        return f"极度超卖 ({cci_value:.2f})"


def format_stock_code(code: str) -> str:
    code_str = str(code)
    if code_str.startswith(("sh", "sz", "bj")):
        return code_str
    code_str = code_str.zfill(6)
    if code_str.startswith("6"):
        return "sh" + code_str
    elif code_str.startswith(("0", "3")):
        return "sz" + code_str
    elif code_str.startswith(("4", "8", "9")):
        return "bj" + code_str
    return code_str


def extract_pure_code(code: str) -> str:
    code_str = str(code).lower()
    if code_str.startswith(("sh", "sz", "bj")):
        code_str = code_str[2:]
    import re
    match = re.search(r"(\d{6})", code_str)
    return match.group(1) if match else code_str.zfill(6)


def fetch_kline_data(symbol: str, days: int = 200) -> pd.DataFrame | None:
    """从 akshare 获取个股日 K 线数据（复用 StockSyncEngine 的获取模式）"""
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    except Exception as e:
        print(f"  [ERROR] 下载数据失败: {e}")
        return None

    if df is None or df.empty:
        print("  [WARN] 未获取到数据")
        return None

    df = df.rename(
        columns={
            "trade_date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "vol": "volume",
        }
    )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def print_analysis_header():
    width = 68
    print()
    print("=" * width)
    print("  单只股票技术指标分析工具")
    print("=" * width)


def print_section(title: str, width: int = 68):
    print()
    print("-" * width)
    print(f"  {title}")
    print("-" * width)


def print_field(label: str, value, width: int = 68):
    if value is not None and str(value).strip():
        print(f"    {label:<26} : {value}")


def main():
    print_analysis_header()

    # ── 1. 用户输入（支持命令行参数和交互式输入）────────────────────────
    if len(sys.argv) > 1:
        raw_code = sys.argv[1]
    else:
        raw_code = input("\n  请输入股票代码 (6位数字，如 000001): ").strip()

    if not raw_code:
        print("  [ERROR] 未输入股票代码")
        return

    pure_code = extract_pure_code(raw_code)
    symbol = format_stock_code(raw_code)
    print(f"\n  [-] 股票代码: {pure_code}  ({symbol})")

    # ── 2. 获取 K 线数据 ────────────────────────────────────────────────
    print(f"\n  >>> 正在从 akshare 下载数据 ({pure_code})...")
    df = fetch_kline_data(symbol, days=200)
    if df is None or df.empty or len(df) < 30:
        print("  [ERROR] 数据不足（至少需要 30 个交易日）")
        return

    print(f"  [OK] 获取到 {len(df)} 条日 K 线数据")
    print(f"      日期范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    # ── 2a. 获取股票名称 ──────────────────────────────────────────────
    stock_name = pure_code
    try:
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        if stock_info is not None and not stock_info.empty:
            name_row = stock_info[stock_info.iloc[:, 0].astype(str).str.contains("股票名称")]
            if not name_row.empty:
                stock_name = name_row.iloc[0, 1]
    except Exception:
        pass

    latest_price = df["close"].iloc[-1] if "close" in df.columns else None

    # ── 3. 基础信息 ──────────────────────────────────────────────────────
    print_section("基础信息")
    print_field("股票代码", pure_code)
    print_field("股票名称", stock_name)
    print_field("最新价", f"{latest_price:.2f}" if latest_price else "N/A")
    print_field("数据条数", len(df))

    # ── 4. MACD 分析 ──────────────────────────────────────────────────
    from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
    from LogicAnalyzer.KDJAnalyzer import AdvancedKDJAnalyzer

    macd_analyzer = MACDAnalyzer()
    kdj_analyzer = AdvancedKDJAnalyzer()

    # 第二周期参数
    second_params = (6, 13, 5)
    second_period_name = f"{second_params[0]}{second_params[1]}{second_params[2]}"

    # 计算 MACD
    try:
        df = macd_analyzer._custom_macd(df, second_params=second_params)
    except Exception as e:
        print(f"  [ERROR] MACD 计算失败: {e}")
        return

    print_section("MACD 指标")

    # ① MACD 12269 信号
    detail_col_12269 = "MACD_12269_SIGNAL_DETAIL"
    if detail_col_12269 in df.columns:
        val = df[detail_col_12269].iloc[-1]
        if pd.notna(val) and str(val).strip():
            print_field("MACD_12269", val)

    # ② MACD 第二周期信号
    detail_col_second = f"MACD_{second_period_name}_SIGNAL_DETAIL"
    if detail_col_second in df.columns:
        val = df[detail_col_second].iloc[-1]
        if pd.notna(val) and str(val).strip():
            print_field(f"MACD_{second_period_name}", val)

    # ③ DIF 值
    if "DIF_12269" in df.columns:
        print_field("MACD_12269_DIF", f"{df['DIF_12269'].iloc[-1]:.4f}")
    if f"DIF_{second_period_name}" in df.columns:
        print_field(f"MACD_{second_period_name}_DIF", f"{df[f'DIF_{second_period_name}'].iloc[-1]:.4f}")

    # ④ MACD 动能
    try:
        mom_12269 = MACDAnalyzer._calculate_macd_momentum(df, "DIF_12269", "DEA_12269")
        print_field("MACD_12269_动能", mom_12269)
    except Exception:
        pass
    try:
        mom_second = MACDAnalyzer._calculate_macd_momentum(
            df, f"DIF_{second_period_name}", f"DEA_{second_period_name}"
        )
        print_field(f"MACD_{second_period_name}_动能", mom_second)
    except Exception:
        pass

    # ── 5. 完全多头综合评分 ──────────────────────────────────────────────
    print_section("MACD 完全多头评分")

    from ConfigParser import Config

    config = Config()
    weights = getattr(config, "FULL_BULL_WEIGHTS", None)
    thresholds = getattr(config, "FULL_BULL_THRESHOLDS", None)

    bull_result = None
    try:
        bull_result = macd_analyzer.analyze_full_bull(
            df,
            second_params=second_params,
            recalc_macd=False,
            weights=weights,
            thresholds=thresholds,
        )
    except Exception as e:
        print(f"  [ERROR] 完全多头评分失败: {e}")

    if bull_result:
        print_field("FullBull_Score", bull_result.get("score", "N/A"))
        print_field("FullBull_Score_Base", bull_result.get("score_base", "N/A"))
        print_field("MACD_FULL_BULL_Label", bull_result.get("conclusion", "N/A"))

        details = bull_result.get("details", {})
        if details:
            print()
            print("  ─ 各维度得分 ─")
            for dim_key, dim_val in details.items():
                desc = dim_val.get("desc", "")
                score = dim_val.get("score", 0)
                print(f"    {dim_key:<20} : {score:>3}  ({desc})")

    # ── 6. KDJ 信号 ────────────────────────────────────────────────────
    print_section("KDJ 指标")
    try:
        kdj_signal = kdj_analyzer.calculate_kdj_signal_from_df(df)
        if kdj_signal:
            print_field("KDJ_Signal", kdj_signal)
        else:
            print_field("KDJ_Signal", "无信号")
    except Exception as e:
        print_field("KDJ_Signal", f"计算失败: {e}")

    # ── 7. CCI 指标 ────────────────────────────────────────────────────
    print_section("CCI 指标")
    try:
        df.ta.cci(append=True, close="close", high="high", low="low")
        cci_cols = [col for col in df.columns if col.startswith("CCI_")]
        if cci_cols:
            current_cci = df[cci_cols[0]].iloc[-1]
            cci_signal = _classify_cci_level(current_cci)
            print_field("CCI_Signal", cci_signal or f"常态波动 ({current_cci:.2f})")
    except Exception:
        print_field("CCI_Signal", "计算失败")

    # ── 8. RSI 指标 ────────────────────────────────────────────────────
    print_section("RSI 指标")
    try:
        df.ta.rsi(append=True, close="close", length=14)
        rsi_cols = [col for col in df.columns if col.startswith("RSI_")]
        if rsi_cols:
            rsi_col = rsi_cols[0]
            curr_rsi = df[rsi_col].iloc[-1]
            window = 10
            if len(df) >= window + 1:
                curr_low = df["low"].iloc[-1]
                min_low_window = df["low"].iloc[-window:-1].min()
                min_rsi_window = df[rsi_col].iloc[-window:-1].min()
                is_price_low = curr_low <= (min_low_window * 1.02)
                is_divergence = is_price_low and (curr_rsi > min_rsi_window * 1.05) and (curr_rsi < 50)
                rsi_signal = (
                    f"RSI底背离! ({curr_rsi:.1f})"
                    if is_divergence
                    else f"RSI={curr_rsi:.1f}"
                )
                print_field("RSI_Signal", rsi_signal)
            else:
                print_field("RSI_Signal", f"RSI={curr_rsi:.1f}")
    except Exception:
        print_field("RSI_Signal", "计算失败")

    # ── 9. BOLL 指标 ──────────────────────────────────────────────────
    print_section("BOLL 指标")
    try:
        df.ta.bbands(append=True, length=20, std=2, close="close")
        boll_lower_cols = [col for col in df.columns if col.startswith("BBL_")]
        boll_upper_cols = [col for col in df.columns if col.startswith("BBU_")]
        if boll_lower_cols and boll_upper_cols:
            df["BOLL_BANDWIDTH"] = (
                (df[boll_upper_cols[0]] - df[boll_lower_cols[0]]) / df["close"]
            )
            is_narrow = (
                df["BOLL_BANDWIDTH"].iloc[-5:].mean() < df["BOLL_BANDWIDTH"].mean()
            )
            boll_signal = "低波/缩口" if is_narrow else "常态/张口"
            print_field("BOLL_Signal", boll_signal)
    except Exception:
        print_field("BOLL_Signal", "计算失败")

    # ── 10. 汇总 ──────────────────────────────────────────────────────
    print()
    print("=" * 68)
    if bull_result:
        score = bull_result.get("score", 0)
        conclusion = bull_result.get("conclusion", "N/A")
        if score >= 80:
            rating = "[强烈买入]"
        elif score >= 60:
            rating = "[逢低布局]"
        elif score >= 40:
            rating = "[观望为主]"
        else:
            rating = "[回避/做空]"
        print(f"  综合评分: {score}  {rating}")
        print(f"  综合结论: {conclusion}")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
