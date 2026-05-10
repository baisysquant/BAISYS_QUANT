import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, List, Optional
import akshare as ak
import pandas as pd
import pandas_ta as ta  # 勿删
from sqlalchemy import text, create_engine
import Industrytrending as industry
from DataManager import DatabaseWriter
from DataManager import ParallelUtils as utils
from DataManager import QuantDataPerformer
from FormatManager import Parse_Currency
from SignalManager import TASignalProcessor
from HistDataEngine import StockSyncEngine
from LoggerManager import LoggerManager
from pathlib import Path
from ConfigParser import Config
from FormatManager.ShareCodeFormatMgr import format_stock_code
from Distribution import MainCostDataManager
from DataManager.CalendarManager import TradingCalendarAnalyzer
from LogicAnalyzer.FundMomentumAnalyzer import FundMomentumAnalyzer
from LogicAnalyzer.Indicators import calculate_full_bull_score
from DataManager.DataFetcher import DataFetcher
from HistDataEngine import StockSyncEngine


class StockAnalyzer:

    def __init__(self, config_file: str = "config.ini"):

        self.stock_sync_engine = StockSyncEngine()
        self.config_file = config_file
        self.momentum_analyzer = FundMomentumAnalyzer()
        self.config = Config(config_file=config_file)
        self.calendar_mgr = TradingCalendarAnalyzer()
        self.today_str = self.calendar_mgr.get_last_trading_day()
        self.temp_dir = self.config.TEMP_DATA_DIRECTORY
        os.makedirs(self.temp_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self.start_time = time.time()
        self.logger = LoggerManager(
            log_dir=self.config.LOG_DIR,
            log_filename=f"Corenews_Main_{self.today_str}.log",
            level=self.config.LOG_LEVEL,
        )

        try:
            self.sync_engine = StockSyncEngine()
            self.db_engine = self.sync_engine.db
        except Exception as e:
            self.logger.critical(
                f"[CRITICAL] Corenews_Main: Failed to initialize StockSyncEngine or its database engine. Error: {e}"
            )
            raise

        # 初始化数据获取器
        self.data_fetcher = DataFetcher(self.config, self.calendar_mgr, self.logger)

        # 初始化主力成本数据管理器
        self.cost_manager = MainCostDataManager(
            cache_enabled=True,
            cache_dir=os.path.join(self.config.TEMP_DATA_DIRECTORY, "cost_data_cache"),
        )

    def _get_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """集中获取所有数据源 (包括主力研报盈利预测)，并支持缓存机制"""
        print("\n>>> 正在初始化数据获取和缓存检查...")

        data = {
            "market_fund_flow_raw": self.data_fetcher.fetch(
                ak.stock_fund_flow_individual, "5日市场资金流向", symbol="5日排行"
            ),
            "market_fund_flow_raw_10": self.data_fetcher.fetch(
                ak.stock_fund_flow_individual, "10日市场资金流向", symbol="10日排行"
            ),
            "market_fund_flow_raw_20": self.data_fetcher.fetch(
                ak.stock_fund_flow_individual, "20日市场资金流向", symbol="20日排行"
            ),
            "strong_stocks_raw": self.data_fetcher.fetch(
                ak.stock_zt_pool_strong_em,
                "强势股池",
                date=self.today_str
            ),
            "consecutive_rise_raw": self.data_fetcher.fetch(
                ak.stock_rank_lxsz_ths, "连续上涨"
            ),
            "ljqs_raw": self.data_fetcher.fetch(ak.stock_rank_ljqs_ths, "量价齐升"),
            "cxfl_raw": self.data_fetcher.fetch(ak.stock_rank_cxfl_ths, "持续放量"),
        }

        # 均线突破数据 (Akshare接口参数不同，需分开获取)
        data["xstp_10_raw"] = self.data_fetcher.fetch(
            ak.stock_rank_xstp_ths, "向上突破10日均线", symbol="10日均线"
        )
        data["xstp_30_raw"] = self.data_fetcher.fetch(
            ak.stock_rank_xstp_ths, "向上突破30日均线", symbol="30日均线"
        )
        data["xstp_60_raw"] = self.data_fetcher.fetch(
            ak.stock_rank_xstp_ths, "向上突破60日均线", symbol="60日均线"
        )

        # 行业板块数据
        print("\n>>> 正在获取行业板块名称并保存至本地...")
        industry_info_filename = f"行业板块信息_{self.today_str}.txt"
        industry_info_path = os.path.join(self.temp_dir, industry_info_filename)
        industry_board_df = pd.DataFrame()

        if os.path.exists(industry_info_path):
            try:
                print(f"  - 发现本地缓存文件，正在读取: {industry_info_filename}")
                industry_board_df = pd.read_csv(
                    industry_info_path, sep="|", encoding="utf-8-sig"
                )
            except Exception as e:
                self.logger.warning(
                    f"  - [WARN] 读取本地缓存失败: {e}，将尝试重新获取..."
                )
        else:
            print(f"本地无缓存，正在通过接口获取")
            try:
                industry_board_df = ak.stock_board_industry_name_em()
                if not industry_board_df.empty:
                    try:
                        industry_board_df.to_csv(
                            industry_info_path,
                            sep="|",
                            index=False,
                            encoding="utf-8-sig",
                        )
                        print(f"  - 获取成功并已保存至: {industry_info_filename}")
                    except Exception as e:
                        self.logger.error(f"  - [ERROR] 保存文件失败: {e}")
            except Exception as e:
                self.logger.error(f"  - [ERROR] 调用行业板块接口失败: {e}")

        data["top_industry_cons_df"] = self._get_top_industry_constituents(
            industry_board_df
        )
        data["industry_board_df"] = industry_board_df

        # 获取主力成本数据（使用新的管理类）
        print("\n>>> 正在获取主力成本数据...")
        main_cost_df = self.cost_manager.get_main_cost_data()
        main_cost_df = self.cost_manager.analyze_cost_data(main_cost_df)
        data["main_cost_data"] = main_cost_df

        # 打印主力成本数据摘要
        self.cost_manager.print_cost_summary(main_cost_df)

        return data

    def _safe_fetch_constituents(self, symbol: str) -> pd.DataFrame:
        """
        带重试机制获取单个行业板块的成分股。
        """
        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                df = ak.stock_board_industry_cons_em(symbol=symbol)
                if df is not None and not df.empty:
                    return df
                else:
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception:
                time.sleep(self.config.DATA_FETCH_DELAY)
        return pd.DataFrame()

    def _get_top_industry_constituents(
        self, industry_board_df: pd.DataFrame
    ) -> pd.DataFrame:

        if industry_board_df.empty or "板块名称" not in industry_board_df.columns:
            return pd.DataFrame()

        # 1. 缓存检查
        cache_name = "前十板块成分股"
        cleaned_file_path = self.data_fetcher._get_file_path(cache_name, cleaned=True)
        cached_df = self.data_fetcher._load_data_from_cache(cleaned_file_path)
        if not cached_df.empty:
            return cached_df

        top_industries = industry_board_df.sort_values(
            by="涨跌幅", ascending=False
        ).head(10)

        industry_list = []
        for _, row in top_industries.iterrows():
            pure_dict = {col: row[col] for col in top_industries.columns}
            industry_list.append(pure_dict)

        def fetch_worker(row):
            try:

                if isinstance(row, pd.Series):
                    industry_name = row["板块名称"]
                # 如果 row 是 dict
                elif isinstance(row, dict):
                    industry_name = row["板块名称"]
                else:
                    print(f"[ERROR] 无法识别的数据类型: {type(row)}")
                    return None

                print(f" - 正在获取板块成分股: {industry_name}")
                constituents_df = self._safe_fetch_constituents(symbol=industry_name)

                if constituents_df is not None and not constituents_df.empty:

                    if "代码" in constituents_df.columns:
                        constituents_df.rename(
                            columns={"代码": "股票代码"}, inplace=True
                        )

                    if "股票代码" in constituents_df.columns:
                        # 关键修复：使用 .astype(str).str.zfill 处理 Series
                        constituents_df["股票代码"] = (
                            constituents_df["股票代码"].astype(str).str.zfill(6)
                        )

                    constituents_df["所属板块"] = industry_name
                    return constituents_df[["股票代码", "所属板块"]].drop_duplicates()
                return None

            except Exception as e:
                self.logger.error(
                    f"[WORKER ERROR] 处理板块 {row.get('板块名称', 'Unknown')} 时出错: {e}"
                )
                return None

        results = utils.run_with_thread_pool(
            items=industry_list,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc="获取板块成分股",
        )

        if results:
            # 过滤掉 None 结果
            valid_results = [df for df in results if df is not None and not df.empty]
            if valid_results:
                final_df = pd.concat(valid_results, ignore_index=True).drop_duplicates(
                    subset=["股票代码"]
                )
                self.data_fetcher._save_data_to_cache(final_df, cleaned_file_path)
                return final_df

        return pd.DataFrame()

    def _save_ta_signals_to_txt(self, ta_signals: Dict[str, pd.DataFrame]):
        """
        将技术指标信号结果保存到独立的 TXT 文件。
        """
        print("\n>>> 正在保存技术指标信号到本地 TXT 文件...")

        save_dir = self.config.TEMP_DATA_DIRECTORY
        today_str = self.today_str

        for indicator_name, df in ta_signals.items():
            if df is None or df.empty:
                continue

            file_name = f"{indicator_name}_Signals_{today_str}.txt"
            file_path = os.path.join(save_dir, file_name)

            try:
                df.to_csv(file_path, sep="|", index=False, encoding="utf-8")
                print(f"  - 成功保存 {indicator_name} 信号文件: {file_name}")
            except Exception as e:
                self.logger.error(f"[ERROR] 保存 {indicator_name} 信号文件失败: {e}")

    def _process_xstp_and_filter(
        self, raw_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame
    ) -> pd.DataFrame:
        """处理并合并均线突破数据，并进行多头排列筛选。"""
        print("正在处理并合并均线突破数据...")

        # 1. 清洗均线数据
        processed_df10 = raw_data["xstp_10_raw"].rename(
            columns={"最新价": "10日均线价"}
        )
        processed_df30 = raw_data["xstp_30_raw"].rename(
            columns={"最新价": "30日均线价"}
        )
        processed_df60 = raw_data["xstp_60_raw"].rename(
            columns={"最新价": "60日均线价"}
        )

        # 2. 合并
        merged_df = pd.concat(
            [
                processed_df10[["股票代码", "股票简称"]].dropna(subset=["股票代码"]),
                processed_df30[["股票代码", "股票简称"]].dropna(subset=["股票代码"]),
                processed_df60[["股票代码", "股票简称"]].dropna(subset=["股票代码"]),
            ]
        ).drop_duplicates(subset=["股票代码"])

        # 3. 重新合并均线价格，确保同一行有所有数据
        xstp_base = merged_df[["股票代码", "股票简称"]].drop_duplicates()
        xstp_base = pd.merge(
            xstp_base,
            processed_df10[["股票代码", "10日均线价"]],
            on="股票代码",
            how="left",
        )
        xstp_base = pd.merge(
            xstp_base,
            processed_df30[["股票代码", "30日均线价"]],
            on="股票代码",
            how="left",
        )
        xstp_base = pd.merge(
            xstp_base,
            processed_df60[["股票代码", "60日均线价"]],
            on="股票代码",
            how="left",
        )

        # 4. 合并实时价格 (此处仍然按代码合并，以便于均线计算的准确性)
        xstp_base = pd.merge(
            xstp_base, spot_df[["股票代码", "最新价"]], on="股票代码", how="left"
        )

        # 5. 类型转换和过滤
        cols_to_convert = [
            col for col in xstp_base.columns if "最新价" in col or col == "最新价"
        ]
        for col in cols_to_convert:
            xstp_base[col] = pd.to_numeric(xstp_base[col], errors="coerce")

        # 过滤条件: 1. 最新价>10日均线 2. 多头排列 (10>30 或 30>60)
        filtered_df = xstp_base[
            (xstp_base["最新价"] > xstp_base["10日均线价"])
            & (
                (
                    xstp_base["10日均线价"]
                    > xstp_base["30日均线价"].fillna(float("-inf"))
                )
                | (
                    xstp_base["30日均线价"]
                    > xstp_base["60日均线价"].fillna(float("-inf"))
                )
            )
        ].copy()

        # 添加完全多头排列标记
        filtered_df["完全多头排列"] = filtered_df.apply(
            lambda row: (
                "是"
                if row["10日均线价"] > row["30日均线价"]
                and row["30日均线价"] > row["60日均线价"]
                else "否"
            ),
            axis=1,
        )

        filtered_df.rename(columns={"最新价": "当前价格"}, inplace=True)
        return filtered_df.fillna("N/A")

 

    def _get_stock_industry_mapping(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        从数据库获取股票的行业信息（通过调用StockSyncEngine的get_main_board_pool）
        """
        print("正在从数据库获取个股行业信息...")

        if not stock_codes:
            return pd.DataFrame(columns=["股票代码", "股票简称", "行业"])

        try:

            # 调用get_main_board_pool方法
            main_board_pool = self.stock_sync_engine.get_main_board_pool()

            # 筛选出需要的股票代码 - 注意保持与数据源的一致性
            # 如果stock_codes已经是不带前缀的6位数字，我们需要匹配数据库中的格式
            formatted_codes = [code.zfill(6) for code in stock_codes]
            filtered_pool = main_board_pool[
                main_board_pool["股票代码"].isin(formatted_codes)
            ]

            # 重命名列以匹配期望的格式
            industry_df = filtered_pool[["股票代码", "name", "industry"]].copy()
            industry_df.columns = ["股票代码", "股票简称", "行业"]

            print(f"从数据库成功获取 {len(industry_df)} 条行业信息")
            return industry_df

        except Exception as e:
            self.logger.warning(f"从数据库获取行业信息失败: {e}，返回空DataFrame")
            return pd.DataFrame(columns=["股票代码", "股票简称", "行业"])

    def _consolidate_data(
        self, processed_data: Dict[str, pd.DataFrame], base_stock_codes_pure: List[str]
    ) -> pd.DataFrame:
        """
        合并所有数据源和信号，生成最终汇总报告。
        参数 base_stock_codes_pure 是最终报告的基准股票代码列表（纯数字）。
        """
        print("\n>>> 正在汇总所有数据和信号 (技术指标作为独立列)...")

        # 初始化最终数据框架
        final_df = pd.DataFrame(base_stock_codes_pure, columns=["股票代码"])
        final_df["股票代码"] = final_df["股票代码"].astype(str).str.zfill(6)

        name_dfs = []
        for key, df in processed_data.items():
            if (
                isinstance(df, pd.DataFrame)
                and not df.empty
                and "股票代码" in df.columns
                and "股票简称" in df.columns
            ):
                temp = df[["股票代码", "股票简称"]].copy()
                temp["股票代码"] = temp["股票代码"].astype(str).str.zfill(6)
                name_dfs.append(temp)

        if name_dfs:
            combined_names = pd.concat(name_dfs, ignore_index=True)
            combined_names = combined_names.dropna(subset=["股票代码", "股票简称"])
            combined_names = combined_names[
                ~combined_names["股票简称"].isin(["N/A", "", "NaN", "nan"])
            ]
            name_mapping = combined_names.drop_duplicates(
                subset=["股票代码"], keep="first"
            )
            if not name_mapping.empty:
                final_df = pd.merge(final_df, name_mapping, on="股票代码", how="left")

        if "股票简称" not in final_df.columns:
            final_df["股票简称"] = "N/A"

        # 获取实时数据
        spot_df = processed_data.get("spot_data_all", pd.DataFrame())

        # 修复：先检查列是否存在
        if not spot_df.empty and "股票代码" in spot_df.columns:
            spot_df["股票代码"] = spot_df["股票代码"].astype(str).str.zfill(6)

            # 合并最新价
            if "最新价" in spot_df.columns:
                final_df = pd.merge(
                    final_df,
                    spot_df[["股票代码", "最新价"]].drop_duplicates(
                        subset=["股票代码"]
                    ),
                    on="股票代码",
                    how="left",
                )
            else:
                final_df["最新价"] = "N/A"
        else:
            final_df["最新价"] = "N/A"

        # 获取行业信息
        print("正在获取行业信息...")
        industry_df = self._get_stock_industry_mapping(base_stock_codes_pure)
        if not industry_df.empty:
            # 补全股票简称（如果原始数据中仍有缺失）
            if "股票简称" in industry_df.columns:
                ind_name_map = industry_df.set_index("股票代码")["股票简称"].to_dict()
                final_df["股票简称"] = final_df.apply(
                    lambda row: (
                        ind_name_map.get(row["股票代码"], "N/A")
                        if pd.isna(row["股票简称"]) or row["股票简称"] == "N/A"
                        else row["股票简称"]
                    ),
                    axis=1,
                )

            final_df = pd.merge(
                final_df, industry_df[["股票代码", "行业"]], on="股票代码", how="left"
            )
            final_df["行业"] = final_df["行业"].fillna("N/A")
        else:
            final_df["行业"] = "N/A"

        final_df["股票简称"] = final_df["股票简称"].fillna("N/A")

        # 添加所属行业信号列（暂时为空）
        final_df["所属行业信号"] = ""

        # 获取历史K线数据 (确保 hist_df_all 已从 processed_data 或数据库中正确获取)
        hist_df_all = processed_data.get("hist_data_all")
        if hist_df_all is None:
            hist_df_all = processed_data.get("kline_data", pd.DataFrame())

        if hist_df_all.empty:
            self.logger.warning(
                "[WARN] 历史K线数据为空，无法计算多头排列评分，将填充默认值。"
            )
            final_df["多头排列趋势"] = "趋势观望"
        else:
            date_col_candidates = [
                "trade_date",
                "date",
                "日期",
                "datetime",
                "Date",
                "TRADE_DATE",
            ]
            date_col_in_kline = next(
                (c for c in date_col_candidates if c in hist_df_all.columns), None
            )
            if date_col_in_kline is None:
                self.logger.warning(
                    f"[WARN] K线数据中未找到日期列（候选: {date_col_candidates}），"
                   
                )
                final_df["多头排列趋势"] = "趋势观望"
                date_col_in_kline = None

            if date_col_in_kline is not None:
                if date_col_in_kline != "trade_date":
                    hist_df_all = hist_df_all.rename(
                        columns={date_col_in_kline: "trade_date"}
                    )

                code_col_in_kline = None
                possible_cols = ["symbol", "ts_code", "code", "股票代码"]
                for col in possible_cols:
                    if col in hist_df_all.columns:
                        code_col_in_kline = col
                        break

                if not code_col_in_kline:
                    raise KeyError(
                        f"无法在K线数据中找到股票代码列。支持的列名: {possible_cols}, 实际列: {list(hist_df_all.columns)}"
                    )

                last_trade_day = self.today_str
                hist_df_all["trade_date"] = (
                    hist_df_all["trade_date"].astype(str).str[:10]
                )
                hist_df_all = hist_df_all[
                    hist_df_all["trade_date"] <= last_trade_day
                ].copy()
                self.logger.info(
                    f"[INFO] 评分用K线截止日期: {last_trade_day}，"
                    f"过滤后数据量: {len(hist_df_all)} 行"
                )

                def _compute_ma_if_missing(kline_df: pd.DataFrame) -> pd.DataFrame:
                    kline_df = kline_df.copy()
                    for period in [5, 10, 20, 30, 60, 90, 120]:
                        col = f"MA{period}"
                        if col not in kline_df.columns:
                            kline_df[col] = (
                                kline_df["close"]
                                .rolling(window=period, min_periods=1)
                                .mean()
                            )
                    if "MA_Volume_5" not in kline_df.columns:
                        kline_df["MA_Volume_5"] = (
                            kline_df["volume"].rolling(window=5, min_periods=1).mean()
                        )
                    return kline_df

                def calculate_bull_score_for_row(row):
                    stock_code = row["股票代码"]

                    if code_col_in_kline == "symbol":
                        stock_kline = hist_df_all[
                            hist_df_all[code_col_in_kline].str[-6:] == stock_code
                        ]
                    else:
                        stock_kline = hist_df_all[
                            hist_df_all[code_col_in_kline].str[:6] == stock_code
                        ]

                    if stock_kline.empty:
                        return pd.Series({"多头排列趋势": "趋势观望"})

                    try:
                        stock_kline = stock_kline.sort_values("trade_date").reset_index(
                            drop=True
                        )
                        stock_kline = _compute_ma_if_missing(stock_kline)
                        result = calculate_full_bull_score(stock_kline)
                        level = result.get("level", "趋势观望")
                        status = result.get("status", "FAILED")
                        if status != "SUCCESS":
                            level = "趋势观望"  # 如果计算失败，也视为观望
                        return pd.Series({"多头排列趋势": level})
                    except Exception as e:
                        print(f"计算评分失败 {stock_code}: {e}")
                        return pd.Series({"多头排列趋势": "趋势观望"})

                score_results = final_df.apply(calculate_bull_score_for_row, axis=1)
                final_df = pd.concat([final_df, score_results], axis=1)

        # 处理资金流数据
        for period, df_key, col_name in [
            (5, "market_fund_flow_raw", "5日资金流入万元"),
            (10, "market_fund_flow_raw_10", "10日资金流入万元"),
            (20, "market_fund_flow_raw_20", "20日资金流入万元"),
        ]:
            fund_flow_df = processed_data.get(df_key, pd.DataFrame())

            # 找到资金流入相关的列名
            flow_col = next(
                (
                    col
                    for col in ["净流入", "资金流入净额", "今日主力净流入-净额"]
                    if col in fund_flow_df.columns
                ),
                None,
            )

            if (
                not fund_flow_df.empty
                and "股票代码" in fund_flow_df.columns
                and flow_col
            ):
                fund_flow_df["股票代码"] = (
                    fund_flow_df["股票代码"].astype(str).str.zfill(6)
                )
                final_df = pd.merge(
                    final_df,
                    fund_flow_df[["股票代码", flow_col]].drop_duplicates(
                        subset=["股票代码"]
                    ),
                    on="股票代码",
                    how="left",
                )
                final_df = final_df.rename(columns={flow_col: col_name})
            elif (
                not fund_flow_df.empty
                and "股票简称" in fund_flow_df.columns
                and flow_col
            ):
                merge_df = fund_flow_df[["股票简称", flow_col]].drop_duplicates(
                    subset=["股票简称"]
                )
                final_df = pd.merge(final_df, merge_df, on="股票简称", how="left")
                final_df = final_df.rename(columns={flow_col: col_name})
            else:
                final_df[col_name] = 0.0

       

        # 资金流数据标准化处理
        f5_col, f10_col, f20_col = (
            "5日资金流入万元",
            "10日资金流入万元",
            "20日资金流入万元",
        )

        if any(col in final_df.columns for col in [f5_col, f10_col, f20_col]):
            final_df = utils._normalize_fund_data(final_df)

        fund_columns_to_normalize = [
            col for col in [f5_col, f10_col, f20_col] if col in final_df.columns
        ]
        
        if fund_columns_to_normalize:
            for col in fund_columns_to_normalize:

                def normalize_single_value(val):
                    if pd.isna(val) or val == "N/A" or val == "":
                        return 0.0
                    val_str = str(val).strip()
                    try:
                        if "亿" in val_str:
                            return float(val_str.replace("亿", "")) * 10000
                        elif "万" in val_str:
                            return float(val_str.replace("万", ""))
                        else:
                            return float(val_str)
                    except ValueError:
                        return 0.0

                final_df[col] = final_df[col].apply(normalize_single_value)

        # 如果三个资金流列都存在，则运行资金动能分析
        if all(col in final_df.columns for col in [f5_col, f10_col, f20_col]):
            try:
                result = final_df.apply(
                    lambda row: self.momentum_analyzer.analyze(row), axis=1
                )
                momentum_df = pd.json_normalize(result)
                if "综合_交易信号" in momentum_df.columns:
                    final_df["资金动能"] = momentum_df["综合_交易信号"]
                elif "资金动能状态" in momentum_df.columns:
                    final_df["资金动能"] = momentum_df["资金动能状态"]
                else:
                    final_df["资金动能"] = result.astype(str)
                if "综合_动能评分" in momentum_df.columns:
                    final_df["资金动能评分"] = momentum_df["综合_动能评分"]
                elif "资金动能评分" in momentum_df.columns:
                    final_df["资金动能评分"] = momentum_df["资金动能评分"]
                print(" - 资金动能新分析器运行成功。")
            except Exception as e:
                self.logger.error(f"运行 FundMomentumAnalyzer 失败: {e}")
                final_df["资金动能"] = "N/A"
        else:
            final_df["资金动能"] = "无数据"

 

        # 处理强势股数据
        if not processed_data["strong_stocks_raw"].empty:
            strong_df = processed_data["strong_stocks_raw"]
            if "股票代码" in strong_df.columns:
                strong_df["股票代码"] = strong_df["股票代码"].astype(str).str.zfill(6)
                strong_codes = set(strong_df["股票代码"].tolist())
                final_df["强势股"] = final_df["股票代码"].apply(
                    lambda x: "是" if x in strong_codes else "否"
                )
            else:
                final_df["强势股"] = "否"
        else:
            final_df["强势股"] = "否"

        # 处理连涨数据
        rise_df = processed_data["consecutive_rise_raw"]
        if not rise_df.empty and "股票代码" in rise_df.columns:
            rise_df["股票代码"] = rise_df["股票代码"].astype(str).str.zfill(6)
            rise_df = rise_df[["股票代码", "连涨天数"]].drop_duplicates(
                subset=["股票代码"]
            )
            final_df = pd.merge(final_df, rise_df, on="股票代码", how="left").fillna(
                {"连涨天数": 0}
            )
        else:
            final_df["连涨天数"] = 0

        final_df["连涨天数"] = final_df["连涨天数"].astype(int)

        # 处理量价齐升数据
        if not processed_data["ljqs_raw"].empty:
            ljqs_df = processed_data["ljqs_raw"]
            if "股票代码" in ljqs_df.columns:
                ljqs_df["股票代码"] = ljqs_df["股票代码"].astype(str).str.zfill(6)
                ljqs_codes = set(ljqs_df["股票代码"].tolist())
                final_df["量价齐升"] = final_df["股票代码"].apply(
                    lambda x: "是" if x in ljqs_codes else "否"
                )
            else:
                final_df["量价齐升"] = "否"
        else:
            final_df["量价齐升"] = "否"

        # 处理持续放量数据
        cxfl_df = processed_data["cxfl_raw"]
        if not cxfl_df.empty and "股票代码" in cxfl_df.columns:
            cxfl_df["股票代码"] = cxfl_df["股票代码"].astype(str).str.zfill(6)
            cxfl_df = cxfl_df[["股票代码", "放量天数"]].drop_duplicates(
                subset=["股票代码"]
            )
            final_df = pd.merge(final_df, cxfl_df, on="股票代码", how="left").fillna(
                {"放量天数": 0}
            )
        else:
            final_df["放量天数"] = 0

        final_df["放量天数"] = final_df["放量天数"].astype(int)

        # 合并技术指标数据
        ta_dfs_to_merge = []

        macd_df_standard = processed_data.get("MACD_12269", pd.DataFrame())
        if not macd_df_standard.empty:
            if "股票代码" in macd_df_standard.columns:
                ta_dfs_to_merge.append(
                    macd_df_standard[["股票代码", "MACD_12269_Signal"]].rename(
                        columns={"MACD_12269_Signal": "MACD_12269"}
                    )
                )

        macd_df_fast = processed_data.get("MACD_6135", pd.DataFrame())
        if not macd_df_fast.empty:
            if "股票代码" in macd_df_fast.columns:
                ta_dfs_to_merge.append(
                    macd_df_fast[["股票代码", "MACD_6135_Signal"]].rename(
                        columns={"MACD_6135_Signal": "MACD_6135"}
                    )
                )

        macd_div_df = processed_data.get("MACD_COMBINED_DIVERGENCE", pd.DataFrame())
        if not macd_div_df.empty:
            if "股票代码" in macd_div_df.columns:
                ta_dfs_to_merge.append(
                    macd_div_df[["股票代码", "Combined_Divergence_Signal"]].rename(
                        columns={"Combined_Divergence_Signal": "MACD_组合背离"}
                    )
                )

        kdj_df = processed_data.get("KDJ", pd.DataFrame())
        if not kdj_df.empty:
            if "股票代码" in kdj_df.columns:
                ta_dfs_to_merge.append(kdj_df[["股票代码", "KDJ_Signal"]])

        cci_df = processed_data.get("CCI", pd.DataFrame())
        if not cci_df.empty:
            if "股票代码" in cci_df.columns:
                ta_dfs_to_merge.append(
                    cci_df[["股票代码", "CCI_Signal"]].rename(
                        columns={"CCI_Signal": "CCI_Signal"}
                    )
                )

        rsi_df = processed_data.get("RSI", pd.DataFrame())
        if not rsi_df.empty:
            if "股票代码" in rsi_df.columns:
                rsi_df["RSI_Signal"] = (
                    rsi_df["RSI_Signal"].astype(str).str.split(" ").str[0]
                )
                ta_dfs_to_merge.append(
                    rsi_df[["股票代码", "RSI_Signal"]].rename(
                        columns={"RSI_Signal": "RSI_Signal"}
                    )
                )

        boll_df = processed_data.get("BOLL", pd.DataFrame())
        if not boll_df.empty:
            if "股票代码" in boll_df.columns:
                ta_dfs_to_merge.append(
                    boll_df[["股票代码", "BOLL_Signal"]].rename(
                        columns={"BOLL_Signal": "BOLL_Signal"}
                    )
                )

        for ta_df in ta_dfs_to_merge:
            if "股票代码" in ta_df.columns:
                final_df = pd.merge(
                    final_df,
                    ta_df.drop_duplicates(subset=["股票代码"]),
                    on="股票代码",
                    how="left",
                )

        momentum_df = processed_data.get("MACD_DIF_MOMENTUM", pd.DataFrame())
        if not momentum_df.empty and "股票代码" in momentum_df.columns:
            final_df = pd.merge(final_df, momentum_df, on="股票代码", how="left")
            for col in ["MACD_12269_动能", "MACD_6135_动能"]:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna("")

        for col in [
            "MACD_12269",
            "MACD_6135",
            "MACD_组合背离",
            "KDJ_Signal",
            "CCI_Signal",
            "RSI_Signal",
            "BOLL_Signal",
        ]:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna("")
            else:
                final_df[col] = ""

        # 处理行业数据
        top_ind_df = processed_data.get("top_industry_cons_df", pd.DataFrame())
        if not top_ind_df.empty and "股票代码" in top_ind_df.columns:
            top_ind_df["股票代码"] = top_ind_df["股票代码"].astype(str).str.zfill(6)
            top_codes = set(top_ind_df["股票代码"].astype(str).unique())
            final_df["TOP10行业"] = final_df["股票代码"].apply(
                lambda x: "是" if str(x) in top_codes else "否"
            )
        else:
            final_df["TOP10行业"] = "否"

        # 主力成本数据
        main_cost_df = processed_data.get("main_cost_data", pd.DataFrame())
        if not main_cost_df.empty:
            if "代码" in main_cost_df.columns:
                main_cost_df.rename(columns={"代码": "股票代码"}, inplace=True)
            if "股票代码" in main_cost_df.columns:
                main_cost_df["股票代码"] = (
                    main_cost_df["股票代码"].astype(str).str.zfill(6)
                )
                final_df = pd.merge(
                    final_df,
                    main_cost_df[
                        [
                            "股票代码",
                            "主力成本",
                            "机构参与度",
                            "主力成本差价",
                            "主力成本差价百分比",
                            "成本位置",
                            "机构参与度等级",
                            "主力控盘强度",
                        ]
                    ],
                    on="股票代码",
                    how="left",
                )
                final_df["主力成本"] = final_df["主力成本"].fillna("N/A")
                final_df["主力成本差价"] = final_df["主力成本差价"].fillna("N/A")
                final_df["成本位置"] = final_df["成本位置"].fillna("N/A")
                final_df["主力控盘强度"] = final_df["主力控盘强度"].fillna("N/A")
        else:
            final_df["主力成本"] = "N/A"
            final_df["主力成本差价"] = "N/A"
            final_df["成本位置"] = "N/A"
            final_df["主力控盘强度"] = "N/A"

        # 均线突破数据
        xstp_df = processed_data.get("processed_xstp_df", pd.DataFrame())
        xstp_cols = [
            "股票代码",
            "完全多头排列",
            "当前价格",
            "10日均线价",
            "30日均线价",
            "60日均线价",
        ]

        if not xstp_df.empty and "股票代码" in xstp_df.columns:
            xstp_df["股票代码"] = xstp_df["股票代码"].astype(str).str.zfill(6)
            cols_present = [col for col in xstp_cols if col in xstp_df.columns]
            merge_df = xstp_df[cols_present].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, merge_df, on="股票代码", how="left")

        if "完全多头排列" not in final_df.columns:
            final_df["完全多头排列"] = "否"
        else:
            final_df["完全多头排列"] = final_df["完全多头排列"].fillna("否")

        # 筛选有信号的股票
        def has_any_signal(row):
            return (
                row["完全多头排列"] == "是"
                or row["强势股"] == "是"
                or row["量价齐升"] == "是"
                or row.get("TOP10行业") == "是"
                or row["MACD_12269"] != ""
                or row["MACD_6135"] != ""
                or row["MACD_组合背离"] != ""
                or row["KDJ_Signal"] != ""
                or row["CCI_Signal"] != ""
                or row["RSI_Signal"] != ""
                or row["BOLL_Signal"] != ""
            )

        bool_cols = ["强势股", "量价齐升"]
        str_cols = [
            "MACD_12269",
            "MACD_6135",
            "MACD_组合背离",
            "KDJ_Signal",
            "CCI_Signal",
            "RSI_Signal",
            "BOLL_Signal",
        ]

        mask = (
            (final_df["完全多头排列"] == "是")
            | final_df["强势股"].eq("是")
            | final_df["量价齐升"].eq("是")
            | final_df.get("TOP10行业", "").eq("是")
            | final_df[str_cols].apply(lambda s: s.str.strip().ne("")).any(axis=1)
        )
        final_df = final_df[mask].copy()

        final_df.sort_values(
            by=["连涨天数", "放量天数"], ascending=[False, False], inplace=True
        )
        final_df.reset_index(drop=True, inplace=True)

        final_df["完整股票代码"] = final_df["股票代码"].apply(format_stock_code)
        final_df["股票链接"] = (
            "https://hybrid.gelonghui.com/stock-check/" + final_df["完整股票代码"]
        )

        final_df.drop(columns=["完整股票代码"], inplace=True, errors="ignore")

        if "当前价格" in final_df.columns and "最新价" in final_df.columns:
            final_df.drop(columns=["当前价格"], inplace=True, errors="ignore")

        # 重新排列列顺序
        base_cols = [
            "股票代码",
            "股票简称",
            "行业",
            "所属行业信号",
            "最新价",
            "主力成本",
            "主力成本差价",
            "成本位置",
            "主力控盘强度",
        ]
        signal_cols = [
            "强势股",
            "量价齐升",
            "连涨天数",
            "放量天数",
            "TOP10行业",
            "MACD_12269",
            "MACD_12269_动能",
            "MACD_12269_DIF",
            "MACD_6135",
            "MACD_6135_动能",
            "MACD_6135_DIF",
            "MACD_组合背离",
            "KDJ_Signal",
            "CCI_Signal",
            "RSI_Signal",
            "BOLL_Signal",
        ]

        report_cols = [
            "多头排列趋势",
            "资金动能",
            "5日资金流入万元",
            "10日资金流入万元",
            "20日资金流入万元",
        ]
        final_cols = base_cols + signal_cols + report_cols + ["股票链接"]
        final_df = final_df[[col for col in final_cols if col in final_df.columns]]

        return final_df

    def _merge_industry_signal_to_stocks(
        self, stock_df: pd.DataFrame, industry_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将行业分析的结论('行业信号'列)，精准匹配到每一只股票上。
        """
        if industry_df.empty or stock_df.empty or "行业" not in stock_df.columns:
            stock_df["所属行业信号"] = ""
            return stock_df

        print("  - 正在将行业信号映射至个股...")
        signal_map = industry_df.set_index("行业名称")["行业信号"].to_dict()
        stock_df["所属行业信号"] = stock_df["行业"].map(signal_map).fillna("")

        return stock_df

    def _generate_report(self, sheets_data: Dict[str, pd.DataFrame]):
        """生成 Excel 报告。"""
        print(f"\n>>> 正在生成 Excel 报告...")
        report_path = os.path.join(
            self.config.TEMP_DATA_DIRECTORY, f"审计报告_{self.today_str}.xlsx"
        )

        try:
            writer = pd.ExcelWriter(report_path, engine="xlsxwriter")
            workbook = writer.book

            header_format = workbook.add_format(
                {
                    "bold": True,
                    "text_wrap": True,
                    "valign": "top",
                    "fg_color": "#D7E4BC",
                    "border": 1,
                }
            )
            currency_format = workbook.add_format({"num_format": "#,##0.00"})
            code_format = workbook.add_format({"num_format": "@"})

            for sheet_name, df in sheets_data.items():

                if df is None or df.empty:
                    print(f"  - 警告：工作表 '{sheet_name}' 数据为空，跳过创建。")
                    continue

                df.to_excel(
                    writer, sheet_name=sheet_name, startrow=1, header=False, index=False
                )
                worksheet = writer.sheets[sheet_name]

                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col))
                    col_width = min(max_len + 2, 30)

                    if (
                        col == "最新价"
                        or "价格" in col
                        or "价" in col
                        or "线" in col
                        or "均线" in col
                    ):
                        worksheet.set_column(i, i, col_width, currency_format)
                    elif "代码" in col:
                        worksheet.set_column(i, i, 10, code_format)
                    elif col in [
                        "5日资金流入万元",
                        "10日资金流入万元",
                        "20日资金流入万元",
                    ]:
                        # 确保资金流入列使用货币格式
                        worksheet.set_column(i, i, col_width, currency_format)
                    else:
                        worksheet.set_column(i, i, col_width)

            writer.close()
            print(f"  - 报告已成功生成并保存到: {report_path}")

        except Exception as e:
            self.logger.critical(f"[FATAL] 致命错误：生成 Excel 报告失败。原因: {e}")
            raise

    def _get_latest_prices_from_kline(self, hist_df_all: pd.DataFrame) -> pd.DataFrame:
        """
        从K线数据中获取最新的收盘价作为"实时价格"
        """
        if hist_df_all.empty:
            return pd.DataFrame(columns=["股票代码", "最新价"])

        # 获取每个股票的最新一条记录（按日期排序）
        latest_records = hist_df_all.sort_values("trade_date").groupby("symbol").tail(1)

        # 提取股票代码和收盘价
        latest_prices = latest_records[["symbol", "close"]].copy()
        latest_prices.columns = ["股票代码", "最新价"]

        # 提取纯数字股票代码
        latest_prices["股票代码"] = (
            latest_prices["股票代码"].astype(str).str.extract(r"(\d{6})")[0]
        )

        return latest_prices

    def _load_industry_info_from_generated_file(
        self, stock_codes_pure: List[str]
    ) -> pd.DataFrame:
        """
        从生成的行业文件中加载行业信息
        """
        print("\n>>> 正在加载行业信息...")

        # 尝试从已有的行业数据中获取
        industry_df = pd.DataFrame()

        # 如果有行业板块数据，则使用它
        if hasattr(self, "industry_board_df") and self.industry_board_df is not None:
            industry_df = self.industry_board_df

        # 创建一个包含股票代码和行业信息的DataFrame
        if not industry_df.empty and "板块名称" in industry_df.columns:
            # 这里简化处理，实际上我们需要根据股票代码关联行业信息
            # 但由于行业数据是板块级别的，不是个股级别的，我们暂时返回空DataFrame
            industry_info_df = pd.DataFrame(
                {
                    "股票代码": stock_codes_pure,
                    "行业": "N/A",  # 暂时填充为N/A，实际应从个股行业数据获取
                }
            )
        else:
            industry_info_df = pd.DataFrame(
                {"股票代码": stock_codes_pure, "行业": "N/A"}
            )

        return industry_info_df

    def run(self):

        print(
            f"[INFO]  股票分析程序启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"[INFO] 识别的业务日期(最后一个交易日)为: {self.today_str}")  # 日志提示

        try:

            self.sync_engine.run_engine()
            self.sync_engine.run_engine(target_date=self.today_str)
            synced_codes_df_from_db = pd.DataFrame(
                columns=["symbol"]
            )  # 初始化为空，以防查询失败

            try:
                # 确保 self.db_engine 已经被成功初始化
                if self.db_engine is None:
                    raise RuntimeError("数据库引擎未成功初始化，无法从数据库获取数据。")

                with self.db_engine.connect() as conn:
                    # 查询数据库中最新的一个交易日期
                    latest_date_query = text(
                        "SELECT MAX(trade_date) FROM stock_daily_kline;"
                    )
                    latest_db_date_result = conn.execute(
                        latest_date_query
                    ).scalar_one_or_none()
                    if latest_db_date_result is None:
                        self.logger.critical(
                            "[FATAL] 数据库中 'stock_daily_kline' 表没有K线数据，无法获取股票代码列表，流程终止。"
                        )
                        return
                    # 查询在该最新交易日期有数据的股票代码
                    query_symbols = text(
                        f"""
                                    SELECT DISTINCT symbol
                                    FROM stock_daily_kline
                                    WHERE trade_date = :latest_date
                                """
                    )
                    synced_codes_df_from_db = pd.read_sql(
                        query_symbols,
                        conn,
                        params={"latest_date": latest_db_date_result},
                    )
                    print(
                        f">>> 已从数据库获取 {len(synced_codes_df_from_db)} 只股票代码，基于最新交易日  "
                    )
            except Exception as e:
                self.logger.critical(
                    f"[FATAL] 查询数据库获取股票代码失败: {e}，流程终止。"
                )
                return  # 异常时也终止流程

            if synced_codes_df_from_db.empty:
                self.logger.critical(
                    "[FATAL] 从数据库获取已同步股票代码列表失败，流程终止。"
                )
                return

            final_analysis_codes_prefixed = synced_codes_df_from_db["symbol"].tolist()

            final_analysis_codes_pure = [
                code[2:] for code in final_analysis_codes_prefixed
            ]

            print(
                f">>> HistDataWatchDog 成功同步 {len(final_analysis_codes_prefixed)} 只股票数据到数据库，并作为分析基础。"
            )

            # 预处理行业权重数据
            industry_analyzer = industry.IndustryFlowAnalyzer(self.config)
            industry_analysis_df = industry_analyzer.run_analysis()

            # 获取K线数据
            raw_data = self._get_all_raw_data()

            # 从K线数据获取最新价格替代实时行情
            print("\n>>> 从K线数据获取最新收盘价...")
            # 构造查询语句
            if not final_analysis_codes_prefixed:
                print("[WARN] 待分析股票代码列表为空，跳过历史数据查询。")
                hist_df_all = pd.DataFrame()
            else:

                symbols_str = ",".join(
                    [f"'{s}'" for s in final_analysis_codes_prefixed]
                )
                query = text(
                    f"""
                    SELECT *
                    FROM stock_daily_kline
                    WHERE symbol IN ({symbols_str})
                    ORDER BY trade_date
                """
                )

                hist_df_all = pd.DataFrame()  # 初始化为空
                try:
                    with self.db_engine.connect() as conn:

                        hist_df_all = pd.read_sql(query, conn)

                        if not hist_df_all.empty:
                            print(
                                f"[INFO] 数据日期范围: {hist_df_all['trade_date'].min()} 至 {hist_df_all['trade_date'].max()}"
                            )
                        else:
                            print(
                                "[ERROR] 查询结果为空！可能是股票代码不匹配或日期条件过滤了所有数据。"
                            )

                except Exception as e:
                    print(f"[ERROR] 数据库查询失败: {e}")
                    hist_df_all = pd.DataFrame()

            if hist_df_all.empty:
                print("[WARN] 由于历史数据为空，将跳过所有技术指标计算。")
            else:
                # 正常调用信号处理
                pass

            # 从K线数据获取最新价格
            latest_prices_df = self._get_latest_prices_from_kline(hist_df_all)
            print(f"[INFO] 从K线数据获取了 {len(latest_prices_df)} 只股票的最新收盘价")

            # 将最新价格数据加入到raw_data中，替代原来的spot_data_all
            raw_data["spot_data_all"] = latest_prices_df
            raw_data["hist_data_all"] = hist_df_all

            signal_processor = TASignalProcessor(self)
            ta_signals = signal_processor.process_signals(
                final_analysis_codes_prefixed, hist_df_all, raw_data["spot_data_all"]
            )
           
            self._save_ta_signals_to_txt(ta_signals)
            print(">>> 股票历史数据和技术指标分析完成。")

            # 行业信息获取，注意这里需要纯数字的代码
            industry_info_df = self._load_industry_info_from_generated_file(
                final_analysis_codes_pure
            )
            universe_codes_set_pure = set(final_analysis_codes_pure)

            def filter_df_by_universe(df, universe_set):
                if df is None or df.empty or "股票代码" not in df.columns:
                    return pd.DataFrame()
                df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
                return df[df["股票代码"].isin(universe_set)].copy()

            # 均线突破数据处理
            processed_xstp_df = self._process_xstp_and_filter(
                raw_data, raw_data["spot_data_all"]
            )
            processed_xstp_df = filter_df_by_universe(
                processed_xstp_df, universe_codes_set_pure
            )

            # 过滤其他每日排名数据
            raw_data["market_fund_flow_raw"] = filter_df_by_universe(
                raw_data["market_fund_flow_raw"], universe_codes_set_pure
            )
            raw_data["market_fund_flow_raw_10"] = filter_df_by_universe(
                raw_data["market_fund_flow_raw_10"], universe_codes_set_pure
            )
            raw_data["market_fund_flow_raw_20"] = filter_df_by_universe(
                raw_data["market_fund_flow_raw_20"], universe_codes_set_pure
            )
            raw_data["strong_stocks_raw"] = filter_df_by_universe(
                raw_data["strong_stocks_raw"], universe_codes_set_pure
            )
            raw_data["consecutive_rise_raw"] = filter_df_by_universe(
                raw_data["consecutive_rise_raw"], universe_codes_set_pure
            )
            raw_data["ljqs_raw"] = filter_df_by_universe(
                raw_data["ljqs_raw"], universe_codes_set_pure
            )
            raw_data["cxfl_raw"] = filter_df_by_universe(
                raw_data["cxfl_raw"], universe_codes_set_pure
            )

            processed_data = {
                **raw_data,
                **ta_signals,
                "processed_xstp_df": processed_xstp_df,
                "processed_main_report": pd.DataFrame(),  # 此时为空DataFrame
                "individual_industry": industry_info_df,
            }

            # 调用 _consolidate_data 时，传入基础的纯数字股票代码列表
            consolidated_report = self._consolidate_data(
                processed_data, final_analysis_codes_pure
            )
            consolidated_report = self._merge_industry_signal_to_stocks(
                consolidated_report, industry_analysis_df
            )

            cols = list(consolidated_report.columns)
            if "所属行业信号" in cols and "行业" in cols:
                cols.remove("所属行业信号")
                idx = cols.index("行业")
                cols.insert(idx + 1, "所属行业信号")
                consolidated_report = consolidated_report[cols]

            print(">>> 正在执行最终数据清洗：剔除弱势且加速下跌的个股...")

            if not consolidated_report.empty:
                # 为了安全比较，确保 DIF 列被正确解析为数字，非数字转为 NaN
                dif_12269 = pd.to_numeric(
                    consolidated_report.get("MACD_12269_DIF"), errors="coerce"
                )
                dif_6135 = pd.to_numeric(
                    consolidated_report.get("MACD_6135_DIF"), errors="coerce"
                )
                kdj_col = consolidated_report.get(
                    "KDJ_Signal",
                    pd.Series(
                        [""] * len(consolidated_report), index=consolidated_report.index
                    ),
                )
                kdj_is_empty = kdj_col.isna() | (
                    kdj_col.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["", "nan", "none"])
                )

                full_bull_score = pd.to_numeric(
                    consolidated_report.get("FullBull_Score", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0)

                full_bull_level = consolidated_report.get(
                    "多头排列趋势", pd.Series(dtype=str)
                )
                exempt_from_drop = (full_bull_level == "完全主升") | (
                    full_bull_level == "趋势加速"
                )

                drop_condition = (
                    (consolidated_report.get("强势股") == "否")
                    & (consolidated_report.get("量价齐升") == "否")
                    & (consolidated_report.get("连涨天数") == 0)
                    & (consolidated_report.get("放量天数") == 0)
                    & (
                        consolidated_report.get("MACD_12269_动能")
                        == "加速下跌 (绿柱加长)"
                    )
                    & (
                        consolidated_report.get("MACD_6135_动能")
                        == "加速下跌 (绿柱加长)"
                    )
                    & (dif_12269 < 0)
                    & (dif_6135 < 0)
                    & kdj_is_empty
                    & (
                        consolidated_report.get("5日资金流入万元", pd.Series(dtype=str))
                        .astype(str)
                        .str.contains("-", na=False)
                    )
                    & (~exempt_from_drop)  # 使用豁免条件
                )

                initial_count = len(consolidated_report)
                consolidated_report = consolidated_report[~drop_condition].copy()
                dropped_count = initial_count - len(consolidated_report)
                print(f" 排除极度弱势特征的股票。剩余 {len(consolidated_report)} 只。")

            # 准备报告数据
            sheets_data = {
                "数据汇总": consolidated_report,
                "行业深度分析": industry_analysis_df,
                "主力研报筛选": processed_data["processed_main_report"],
                "前十板块成分股": raw_data["top_industry_cons_df"],
                "主力成本分析": processed_data["main_cost_data"],
            }

            # 生成报告
            self._generate_report(sheets_data)

            try:
                db_manager = DatabaseWriter.QuantDBManager(
                    user=self.config.DB_USER,
                    password=self.config.DB_PASSWORD,
                    host=self.config.DB_HOST,
                    port=self.config.DB_PORT,
                    db_name=self.config.DB_NAME,
                )

                sync_task = QuantDataPerformer.QuantDBSyncTask(db_manager)

                sync_task.sync_all(
                    today_str=self.today_str,
                    consolidated_report=consolidated_report,
                    industry_df=industry_analysis_df,
                    raw_data=raw_data,
                )

                db_manager.close()
                print("数据库同步成功完成。")

            except Exception as e:
                self.logger.error(f"!!! [同步中断] 任务运行异常: {e}")

        except Exception as e:
            self.logger.critical(f"\n[FATAL] 致命错误：数据分析流程意外终止。原因: {e}")
            raise

        finally:
            end_time = time.time()
            print(
                f"\n>>> 流程结束。总耗时: {timedelta(seconds=end_time - self.start_time)}"
            )


if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run()
