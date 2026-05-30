
from typing import Any
import pandas as pd
import pandas_ta as ta
from concurrent.futures import ProcessPoolExecutor, as_completed

from ConfigParser import Config
from LogicAnalyzer.KDJAnalyzer import AdvancedKDJAnalyzer
from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer

pd.set_option('mode.chained_assignment', None)


def _classify_cci_level(cci_value: float) -&gt; str:
    if pd.isna(cci_value):
        return "N/A"
    if cci_value &gt; 200:
        return f"极度超买 ({cci_value:.2f})"
    elif cci_value &gt;= 100:
        return f"强势超买 ({cci_value:.2f})"
    elif cci_value &gt; -100:
        return ""
    elif cci_value &gt;= -200:
        return f"弱势超卖 ({cci_value:.2f})"
    else:
        return f"极度超卖 ({cci_value:.2f})"


def _process_single_stock(
    code: str,
    hist_df_subset: pd.DataFrame,
    second_period_name: str,
    config: Config
) -&gt; dict:
    """处理单个股票的技术指标（独立函数，用于多进程）"""
    result = {
        "code": code,
        "macd_12269": None,
        f"macd_{second_period_name}": None,
        "macd_divergence": None,
        "kdj": None,
        "cci": None,
        "rsi": None,
        "boll": None,
        "macd_momentum": None,
        "macd_full_bull": None,
    }

    df = hist_df_subset.copy()
    if df.empty or len(df) &lt; 30:
        return result

    for col in ["close", "open", "high", "low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)

    if df.empty:
        return result
    if "close" not in df.columns or "open" not in df.columns:
        return result

    try:
        df = MACDAnalyzer._custom_macd(df)
        dist_slow = MACDAnalyzer._adaptive_distance(df, base=10)
        dist_fast = MACDAnalyzer._adaptive_distance(df, base=5)

        combined_div = MACDAnalyzer.detect_combined_divergence(
            df,
            distance_slow=dist_slow,
            distance_fast=dist_fast,
            recent_window=5,
            decay_half_life=8,
            second_period_name=second_period_name,
        )
        divergence_signal = combined_div.get("combined_signal", "")
        if divergence_signal:
            result["macd_divergence"] = {
                "Combined_Divergence_Signal": divergence_signal,
                "Div_12269_Type": combined_div.get("div_12269", ""),
                "Div_12269_Strength": combined_div.get("strength_12269", 0.0),
                "Div_12269_Decay": combined_div.get("decay_12269", 0.0),
                f"Div_{second_period_name}_Type": combined_div.get(f"div_{second_period_name}", ""),
                f"Div_{second_period_name}_Strength": combined_div.get(f"strength_{second_period_name}", 0.0),
                f"Div_{second_period_name}_Decay": combined_div.get(f"decay_{second_period_name}", 0.0),
            }
    except Exception:
        pass

    try:
        bull_result = MACDAnalyzer.analyze_full_bull(df, second_params=config.MACD_SECOND_PARAMS)
        detail = bull_result.get("details", {})
        result["macd_full_bull"] = {
            "FullBull_Score": bull_result.get("score", 0),
            "FullBull_Conclusion": bull_result.get("conclusion", ""),
            "零轴条件": detail.get("零轴条件", {}).get("desc", ""),
            "战略金叉": detail.get("战略金叉", {}).get("desc", ""),
            "战术金叉": detail.get("战术金叉", {}).get("desc", ""),
            "动能": detail.get("动能", {}).get("desc", ""),
            "DIF斜率": detail.get("DIF斜率", {}).get("desc", ""),
            "背离信号": detail.get("背离信号", {}).get("desc", ""),
            "量价配合": detail.get("量价配合", {}).get("desc", ""),
        }
    except Exception:
        pass

    try:
        latest_row = df.iloc[-1]
        mom_12269 = MACDAnalyzer._calculate_macd_momentum(df, "DIF_12269", "DEA_12269")
        mom_second = MACDAnalyzer._calculate_macd_momentum(
            df, f"DIF_{second_period_name}", f"DEA_{second_period_name}"
        )
        result["macd_momentum"] = {
            "MACD_12269_DIF": latest_row.get("DIF_12269", 0),
            "MACD_12269_动能": mom_12269,
            f"MACD_{second_period_name}_DIF": latest_row.get(f"DIF_{second_period_name}", 0),
            f"MACD_{second_period_name}_动能": mom_second,
        }
    except Exception:
        pass

    detail_col_12269 = "MACD_12269_SIGNAL_DETAIL"
    if detail_col_12269 in df.columns and df[detail_col_12269].iloc[-1] != "":
        result["macd_12269"] = {"MACD_12269_Signal": df[detail_col_12269].iloc[-1]}

    detail_col_second = f"MACD_{second_period_name}_SIGNAL_DETAIL"
    if detail_col_second in df.columns and df[detail_col_second].iloc[-1] != "":
        result[f"macd_{second_period_name}"] = {
            f"MACD_{second_period_name}_Signal": df[detail_col_second].iloc[-1]
        }

    try:
        kdj_analyzer = AdvancedKDJAnalyzer()
        kdj_signal = kdj_analyzer.calculate_kdj_signal_from_df(df)
        if kdj_signal:
            result["kdj"] = {"KDJ_Signal": kdj_signal}
    except Exception:
        pass

    try:
        cci_result = ta.cci(high=df["high"], low=df["low"], close=df["close"], append=False)
        if cci_result is not None:
            df["CCI_14_0.015"] = cci_result
            cci_cols = ["CCI_14_0.015"]
            if cci_cols:
                current_cci = df[cci_cols[0]].iloc[-1]
                cci_signal = _classify_cci_level(current_cci) or f"常态波动 ({current_cci:.2f})"
                result["cci"] = {"CCI_Signal": cci_signal}
    except Exception:
        pass

    try:
        rsi_result = ta.rsi(close=df["close"], length=14, append=False)
        if rsi_result is not None:
            df["RSI_14"] = rsi_result
            rsi_cols = ["RSI_14"]
            if rsi_cols:
                rsi_col = rsi_cols[0]
                curr_rsi = df[rsi_col].iloc[-1]
                window = 10
                curr_low = df["low"].iloc[-1]
                min_low_window = df["low"].iloc[-window:-1].min()
                min_rsi_window = df[rsi_col].iloc[-window:-1].min()
                is_price_low = curr_low &lt;= (min_low_window * 1.02)
                is_divergence = is_price_low and (curr_rsi &gt; min_rsi_window * 1.05) and (curr_rsi &lt; 50)
                rsi_msg = f"RSI底背离! ({curr_rsi:.1f})" if is_divergence else f"RSI={curr_rsi:.1f}"
                result["rsi"] = {"RSI_Signal": rsi_msg}
    except Exception:
        pass

    try:
        boll_result = ta.bbands(close=df["close"], length=20, std=2, append=False)
        if boll_result is not None:
            for col in boll_result.columns:
                df[col] = boll_result[col]
            boll_lower_cols = [col for col in df.columns if col.startswith("BBL_")]
            boll_upper_cols = [col for col in df.columns if col.startswith("BBU_")]
            if boll_lower_cols and boll_upper_cols:
                df["BOLL_BANDWIDTH"] = (df[boll_upper_cols[0]] - df[boll_lower_cols[0]]) / df["close"]
                is_narrow = df["BOLL_BANDWIDTH"].iloc[-5:].mean() &lt; df["BOLL_BANDWIDTH"].mean()
                result["boll"] = {"BOLL_Signal": "低波/缩口" if is_narrow else "常态/张口"}
    except Exception:
        pass

    return result


class TASignalProcessor:
    def __init__(self, analyzer_instance: Any, config: Config | None = None) -&gt; None:
        self.analyzer = analyzer_instance
        self.kdj_analyzer = AdvancedKDJAnalyzer()
        self.macd_analyzer = MACDAnalyzer()
        self.config = config

    def process_signals(
        self,
        all_codes: list[str],
        hist_df_all: pd.DataFrame,
        spot_df: pd.DataFrame,
    ) -&gt; dict[str, pd.DataFrame]:
        print(f"\n正在对 {len(all_codes)} 只股票进行技术分析...")

        if self.config and hasattr(self.config, "MACD_SECOND_PARAMS"):
            fast, slow, signal = self.config.MACD_SECOND_PARAMS
            second_period_name = f"{fast}{slow}{signal}"
        else:
            second_period_name = "9186"

        num_processes = 2
        if self.config and hasattr(self.config, "SIGNAL_PROCESSING_PROCESSES"):
            num_processes = self.config.SIGNAL_PROCESSING_PROCESSES
        print(f"使用 {num_processes} 个进程并行处理技术指标...")

        ta_signals = {
            "MACD_12269": pd.DataFrame(columns=["股票代码", "MACD_12269_Signal"]),
            f"MACD_{second_period_name}": pd.DataFrame(columns=["股票代码", f"MACD_{second_period_name}_Signal"]),
            "MACD_COMBINED_DIVERGENCE": pd.DataFrame(
                columns=[
                    "股票代码",
                    "Combined_Divergence_Signal",
                    "Div_12269_Type",
                    "Div_12269_Strength",
                    "Div_12269_Decay",
                    f"Div_{second_period_name}_Type",
                    f"Div_{second_period_name}_Strength",
                    f"Div_{second_period_name}_Decay",
                ]
            ),
            "KDJ": pd.DataFrame(columns=["股票代码", "KDJ_Signal"]),
            "CCI": pd.DataFrame(columns=["股票代码", "CCI_Signal"]),
            "RSI": pd.DataFrame(columns=["股票代码", "RSI_Signal"]),
            "BOLL": pd.DataFrame(columns=["股票代码", "BOLL_Signal"]),
            "MACD_DIF_MOMENTUM": pd.DataFrame(
                columns=[
                    "股票代码",
                    "MACD_12269_DIF",
                    "MACD_12269_动能",
                    f"MACD_{second_period_name}_DIF",
                    f"MACD_{second_period_name}_动能",
                ]
            ),
            "MACD_FULL_BULL": pd.DataFrame(
                columns=[
                    "股票代码",
                    "FullBull_Score",
                    "FullBull_Conclusion",
                    "零轴条件",
                    "战略金叉",
                    "战术金叉",
                    "动能",
                    "DIF斜率",
                    "背离信号",
                    "量价配合",
                ]
            ),
        }

        if hist_df_all.empty:
            print("[WARN] 历史数据为空，跳过技术分析。")
            return ta_signals

        if "symbol" not in hist_df_all.columns:
            print("[ERROR] K 线数据中缺少 'symbol' 列！")
            return ta_signals

        symbol_str = hist_df_all["symbol"].astype(str)
        extracted_digits = symbol_str.str.extract(r"(\d{6})", expand=False).fillna("N/A")
        hist_df_all["股票代码"] = extracted_digits.str.zfill(6)

        if "date" not in hist_df_all.columns and "trade_date" in hist_df_all.columns:
            hist_df_all.rename(columns={"trade_date": "date"}, inplace=True)

        hist_df_all.sort_values(["股票代码", "date"], inplace=True)

        pure_codes_list = [c[2:] if str(c).startswith(("sh", "sz", "bj")) else c for c in all_codes]
        code_set = set(pure_codes_list)
        hist_df_filtered = hist_df_all[hist_df_all["股票代码"].isin(code_set)].copy()

        code_to_hist = {}
        for pure_code, group in hist_df_filtered.groupby("股票代码"):
            code_to_hist[pure_code] = group.reset_index(drop=True)

        tasks = []
        for code in all_codes:
            pure_code = code[2:] if str(code).startswith(("sh", "sz", "bj")) else code
            if pure_code in code_to_hist:
                tasks.append((code, pure_code))

        if num_processes &gt; 1 and len(tasks) &gt; 10:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for code, pure_code in tasks:
                    future = executor.submit(
                        _process_single_stock,
                        code,
                        code_to_hist[pure_code],
                        second_period_name,
                        self.config,
                    )
                    futures.append(future)

                completed = 0
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        self._merge_result_to_signals(result, second_period_name, ta_signals)
                        completed += 1
                        if completed % 50 == 0:
                            print(f"已处理 {completed}/{len(tasks)} 只股票...")
                    except Exception:
                        pass
        else:
            completed = 0
            for code, pure_code in tasks:
                result = _process_single_stock(
                    code, code_to_hist[pure_code], second_period_name, self.config
                )
                self._merge_result_to_signals(result, second_period_name, ta_signals)
                completed += 1
                if completed % 50 == 0:
                    print(f"已处理 {completed}/{len(tasks)} 只股票...")

        for key in ta_signals:
            df_sig = ta_signals[key]
            if not df_sig.empty and "股票代码" in df_sig.columns:
                ta_signals[key]["股票代码"] = df_sig["股票代码"].astype(str).str.extract(r"(\d{6})")

        print(f"\n技术指标处理完成，共处理 {len(tasks)} 只股票。")
        return ta_signals

    def _merge_result_to_signals(self, result: dict, second_period_name: str, ta_signals: dict):
        code = result["code"]

        if result["macd_12269"]:
            data = result["macd_12269"].copy()
            data["股票代码"] = code
            ta_signals["MACD_12269"] = pd.concat(
                [ta_signals["MACD_12269"], pd.DataFrame([data])], ignore_index=True
            )

        if result[f"macd_{second_period_name}"]:
            data = result[f"macd_{second_period_name}"].copy()
            data["股票代码"] = code
            ta_signals[f"MACD_{second_period_name}"] = pd.concat(
                [ta_signals[f"MACD_{second_period_name}"], pd.DataFrame([data])], ignore_index=True
            )

        if result["macd_divergence"]:
            data = result["macd_divergence"].copy()
            data["股票代码"] = code
            ta_signals["MACD_COMBINED_DIVERGENCE"] = pd.concat(
                [ta_signals["MACD_COMBINED_DIVERGENCE"], pd.DataFrame([data])], ignore_index=True
            )

        if result["kdj"]:
            data = result["kdj"].copy()
            data["股票代码"] = code
            ta_signals["KDJ"] = pd.concat(
                [ta_signals["KDJ"], pd.DataFrame([data])], ignore_index=True
            )

        if result["cci"]:
            data = result["cci"].copy()
            data["股票代码"] = code
            ta_signals["CCI"] = pd.concat(
                [ta_signals["CCI"], pd.DataFrame([data])], ignore_index=True
            )

        if result["rsi"]:
            data = result["rsi"].copy()
            data["股票代码"] = code
            ta_signals["RSI"] = pd.concat(
                [ta_signals["RSI"], pd.DataFrame([data])], ignore_index=True
            )

        if result["boll"]:
            data = result["boll"].copy()
            data["股票代码"] = code
            ta_signals["BOLL"] = pd.concat(
                [ta_signals["BOLL"], pd.DataFrame([data])], ignore_index=True
            )

        if result["macd_momentum"]:
            data = result["macd_momentum"].copy()
            data["股票代码"] = code
            ta_signals["MACD_DIF_MOMENTUM"] = pd.concat(
                [ta_signals["MACD_DIF_MOMENTUM"], pd.DataFrame([data])], ignore_index=True
            )

        if result["macd_full_bull"]:
            data = result["macd_full_bull"].copy()
            data["股票代码"] = code
            ta_signals["MACD_FULL_BULL"] = pd.concat(
                [ta_signals["MACD_FULL_BULL"], pd.DataFrame([data])], ignore_index=True
            )

