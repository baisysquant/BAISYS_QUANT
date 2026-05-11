import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, List, Optional, Tuple, Set
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
from DataValidator import DataValidator


class StockAnalyzer:
    """
    股票分析器主类
    
    负责整合多个数据源（资金流、强势股、技术指标等），进行综合分析和筛选，
    生成最终的投资分析报告。
    
    Attributes:
        config: 配置管理器实例
        calendar_mgr: 交易日历管理器
        today_str: 当前交易日字符串 (YYYYMMDD格式)
        data_fetcher: 数据获取器，支持缓存机制
        cost_manager: 主力成本数据管理器
        logger: 日志管理器
        executor: 线程池执行器
    """

    def __init__(self, config_file: str = "config.ini") -> None:
        """
        初始化股票分析器
        
        Args:
            config_file: 配置文件路径，默认为 'config.ini'
        """

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
        
        # 初始化数据验证器
        self.data_validator = DataValidator(self.logger)

        # 初始化主力成本数据管理器
        self.cost_manager = MainCostDataManager(
            cache_enabled=True,
            cache_dir=os.path.join(self.config.TEMP_DATA_DIRECTORY, "cost_data_cache"),
        )

    @staticmethod
    def _normalize_stock_code(code: str) -> str:
        """
        统一标准化股票代码为6位纯数字格式
        
        处理各种格式的股票代码，包括：
        - SH600000, SZ000001 (带市场前缀)
        - 600000.SH, 000001.SZ (带市场后缀)
        - 600000, 000001 (纯数字)
        - 其他非标准格式
        
        Args:
            code: 原始股票代码字符串
            
        Returns:
            str: 标准化后的6位数字股票代码，失败时返回空字符串
            
        Examples:
            >>> StockAnalyzer._normalize_stock_code('SH600000')
            '600000'
            >>> StockAnalyzer._normalize_stock_code('000001.SZ')
            '000001'
            >>> StockAnalyzer._normalize_stock_code('600000')
            '600000'
        """
        if pd.isna(code) or code is None:
            return ""
        
        code_str = str(code).strip()
        
        # 尝试提取6位数字
        import re
        match = re.search(r'(\d{6})', code_str)
        if match:
            return match.group(1)
        
        # 如果没有找到6位数字，尝试补零
        digits_only = re.sub(r'\D', '', code_str)
        if len(digits_only) <= 6:
            return digits_only.zfill(6)
        
        return code_str

    def _get_first_fund_flow_col(self) -> str:
        """
        获取配置的第一个资金流列名
        
        Returns:
            str: 资金流列名，如 "5日资金流入万元"
        """
        # 与akshare接口严格对应
        period_map = {
            3: "3日资金流入万元",
            5: "5日资金流入万元",
            10: "10日资金流入万元",
            20: "20日资金流入万元",
        }
        
        if self.config.FUND_FLOW_PERIODS:
            first_period = self.config.FUND_FLOW_PERIODS[0]
            return period_map.get(first_period, "5日资金流入万元")
        
        return "5日资金流入万元"

    def _get_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        集中获取所有数据源，并支持缓存机制
        
        该方法负责从多个 akshare 接口获取原始数据，包括：
        - 资金流数据（根据配置的周期动态获取）
        - 强势股池数据
        - 连续上涨、量价齐升、持续放量等技术指标
        - 均线突破数据（10日、30日、60日）
        - 行业板块信息
        - 主力研报盈利预测
        
        所有数据都通过 DataFetcher 获取，支持自动缓存和重试机制。
        
        Returns:
            Dict[str, pd.DataFrame]: 键值对字典，key为数据名称，value为对应的DataFrame
            包含的键包括：
            - market_fund_flow_raw[_3/_10/_20]: 市场资金流向数据
            - strong_stocks_raw: 强势股池数据
            - consecutive_rise_raw: 连续上涨数据
            - ljqs_raw: 量价齐升数据
            - cxfl_raw: 持续放量数据
            - xstp_10_raw/xstp_30_raw/xstp_60_raw: 均线突破数据
            - industry_board_df: 行业板块成分股映射
            - profit_forecast_raw: 主力研报盈利预测数据
            
        Raises:
            Exception: 当数据获取失败时抛出异常
        """
        print("\n>>> 正在初始化数据获取和缓存检查...")

        # 根据配置动态获取资金流数据
        data = {}
        
        # akshare接口参数映射（严格对应 stock_fund_flow_individual 的 symbol 参数）
        # 支持：3日排行, 5日排行, 10日排行, 20日排行
        period_to_akshare = {
            3: ("3日市场资金流向", "3日排行", "market_fund_flow_raw_3"),
            5: ("5日市场资金流向", "5日排行", "market_fund_flow_raw"),
            10: ("10日市场资金流向", "10日排行", "market_fund_flow_raw_10"),
            20: ("20日市场资金流向", "20日排行", "market_fund_flow_raw_20"),
        }
        
        for period in self.config.FUND_FLOW_PERIODS:
            if period in period_to_akshare:
                desc, symbol, key = period_to_akshare[period]
                try:
                    fund_flow_df = self.data_fetcher.fetch(
                        ak.stock_fund_flow_individual, desc, symbol=symbol
                    )
                    
                    # 验证资金流数据
                    if not fund_flow_df.empty:
                        # 检查必需列
                        required_cols = ['股票代码', '最新价']
                        is_valid, missing = self.data_validator.validate_required_columns(
                            fund_flow_df, required_cols, f"{period}日资金流"
                        )
                        
                        if is_valid:
                            data[key] = fund_flow_df
                            print(f"  - 已获取 {period}日资金流数据 ({len(fund_flow_df)} 条记录)")
                        else:
                            self.logger.warning(
                                f"  - [WARN] {period}日资金流数据缺少列: {missing}，跳过"
                            )
                    else:
                        self.logger.warning(f"  - [WARN] {period}日资金流数据为空")
                        
                except Exception as e:
                    self.logger.error(f"  - [ERROR] 获取{period}日资金流数据失败: {e}")
            else:
                self.logger.warning(f"不支持的资金流周期: {period}日（仅支持3,5,10,20）")
        
        # 获取其他数据源（带验证）
        print("\n>>> 正在获取其他技术指标数据...")
        
        data_sources = {
            "strong_stocks_raw": (ak.stock_zt_pool_strong_em, "强势股池", {'date': self.today_str}),
            "consecutive_rise_raw": (ak.stock_rank_lxsz_ths, "连续上涨", {}),
            "ljqs_raw": (ak.stock_rank_ljqs_ths, "量价齐升", {}),
            "cxfl_raw": (ak.stock_rank_cxfl_ths, "持续放量", {}),
        }
        
        for key, (api_func, desc, params) in data_sources.items():
            try:
                df = self.data_fetcher.fetch(api_func, desc, **params)
                
                if not df.empty:
                    # 验证必需列
                    required_cols = ['股票代码']
                    is_valid, missing = self.data_validator.validate_required_columns(
                        df, required_cols, desc
                    )
                    
                    if is_valid:
                        data[key] = df
                        print(f"  - ✓ {desc}: {len(df)} 条记录")
                    else:
                        self.logger.warning(f"  - ⚠ {desc} 缺少列: {missing}")
                        data[key] = pd.DataFrame()  # 置空
                else:
                    self.logger.warning(f"  - ⚠ {desc} 数据为空")
                    data[key] = pd.DataFrame()
                    
            except Exception as e:
                self.logger.error(f"  - ✗ 获取{desc}失败: {e}")
                data[key] = pd.DataFrame()

        # 均线突破数据 (Akshare接口参数不同，需分开获取)
        print("\n>>> 正在获取均线突破数据...")
        
        xstp_configs = [
            ("xstp_10_raw", "向上突破10日均线", "10日均线"),
            ("xstp_30_raw", "向上突破30日均线", "30日均线"),
            ("xstp_60_raw", "向上突破60日均线", "60日均线"),
        ]
        
        for key, desc, symbol in xstp_configs:
            try:
                df = self.data_fetcher.fetch(
                    ak.stock_rank_xstp_ths, desc, symbol=symbol
                )
                
                if not df.empty:
                    required_cols = ['股票代码']
                    is_valid, missing = self.data_validator.validate_required_columns(
                        df, required_cols, desc
                    )
                    
                    if is_valid:
                        data[key] = df
                        print(f"  - ✓ {desc}: {len(df)} 条记录")
                    else:
                        self.logger.warning(f"  - ⚠ {desc} 缺少列: {missing}")
                        data[key] = pd.DataFrame()
                else:
                    self.logger.warning(f"  - ⚠ {desc} 数据为空")
                    data[key] = pd.DataFrame()
                    
            except Exception as e:
                self.logger.error(f"  - ✗ 获取{desc}失败: {e}")
                data[key] = pd.DataFrame()

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
                
                # 验证缓存数据
                if not industry_board_df.empty:
                    required_cols = ['板块名称', '板块代码']
                    is_valid, missing = self.data_validator.validate_required_columns(
                        industry_board_df, required_cols, "行业板块缓存"
                    )
                    
                    if is_valid:
                        print(f"  - ✓ 缓存数据有效: {len(industry_board_df)} 个板块")
                    else:
                        self.logger.warning(
                            f"  - ⚠ 缓存数据缺少列: {missing}，将重新获取"
                        )
                        industry_board_df = pd.DataFrame()
                else:
                    self.logger.warning("  - ⚠ 缓存数据为空，将重新获取")
                    industry_board_df = pd.DataFrame()
                    
            except Exception as e:
                self.logger.warning(
                    f"  - [WARN] 读取本地缓存失败: {e}，将尝试重新获取..."
                )
                industry_board_df = pd.DataFrame()
        else:
            print(f"本地无缓存，正在通过接口获取")
            try:
                industry_board_df = ak.stock_board_industry_name_em()
                
                if not industry_board_df.empty:
                    # 验证接口数据
                    required_cols = ['板块名称', '板块代码']
                    is_valid, missing = self.data_validator.validate_required_columns(
                        industry_board_df, required_cols, "行业板块接口"
                    )
                    
                    if is_valid:
                        try:
                            industry_board_df.to_csv(
                                industry_info_path,
                                sep="|",
                                index=False,
                                encoding="utf-8-sig",
                            )
                            print(f"  - ✓ 获取成功并已保存: {len(industry_board_df)} 个板块")
                        except Exception as e:
                            self.logger.error(f"  - ✗ 保存文件失败: {e}")
                    else:
                        self.logger.warning(f"  - ⚠ 接口数据缺少列: {missing}")
                        industry_board_df = pd.DataFrame()
                else:
                    self.logger.warning("  - ⚠ 行业板块接口返回空数据")
                    
            except Exception as e:
                self.logger.error(f"  - ✗ 调用行业板块接口失败: {e}")

        data["top_industry_cons_df"] = self._get_top_industry_constituents(
            industry_board_df
        )
        data["industry_board_df"] = industry_board_df

        # 获取主力成本数据（使用新的管理类）
        print("\n>>> 正在获取主力成本数据...")
        try:
            main_cost_df = self.cost_manager.get_main_cost_data()
            
            if not main_cost_df.empty:
                # 验证必需列
                required_cols = ['股票代码', '主力成本']
                is_valid, missing = self.data_validator.validate_required_columns(
                    main_cost_df, required_cols, "主力成本数据"
                )
                
                if is_valid:
                    main_cost_df = self.cost_manager.analyze_cost_data(main_cost_df)
                    data["main_cost_data"] = main_cost_df
                    print(f"  - ✓ 主力成本数据: {len(main_cost_df)} 条记录")
                    
                    # 打印主力成本数据摘要
                    self.cost_manager.print_cost_summary(main_cost_df)
                else:
                    self.logger.warning(f"  - ⚠ 主力成本数据缺少列: {missing}")
                    data["main_cost_data"] = pd.DataFrame()
            else:
                self.logger.warning("  - ⚠ 主力成本数据为空")
                data["main_cost_data"] = pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"  - ✗ 获取主力成本数据失败: {e}")
            data["main_cost_data"] = pd.DataFrame()

        return data

    def _safe_fetch_constituents(self, symbol: str) -> pd.DataFrame:
        """
        安全地获取行业板块成分股数据
        
        使用异常处理包裹 akshare 接口调用，避免因单个板块数据获取失败
        而影响整个分析流程。
        
        Args:
            symbol: 行业板块代码或名称
            
        Returns:
            pd.DataFrame: 成分股数据DataFrame，包含股票代码、名称等字段
                         如果获取失败，返回空的DataFrame
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
                        # 使用统一方法标准化股票代码
                        constituents_df["股票代码"] = constituents_df[
                            "股票代码"
                        ].apply(self._normalize_stock_code)

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
        processed_df10 = raw_data.get("xstp_10_raw", pd.DataFrame()).rename(
            columns={"最新价": "10日均线价"}
        )
        processed_df30 = raw_data.get("xstp_30_raw", pd.DataFrame()).rename(
            columns={"最新价": "30日均线价"}
        )
        processed_df60 = raw_data.get("xstp_60_raw", pd.DataFrame()).rename(
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
            formatted_codes = [self._normalize_stock_code(code) for code in stock_codes]
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

    def _merge_basic_info(
        self,
        final_df: pd.DataFrame,
        processed_data: Dict[str, pd.DataFrame],
        base_stock_codes_pure: List[str]
    ) -> pd.DataFrame:
        """
        合并基础信息：股票名称、实时价格、行业信息
        """
        # 从各数据源提取股票名称
        name_dfs = []
        for key, df in processed_data.items():
            if (
                isinstance(df, pd.DataFrame)
                and not df.empty
                and "股票代码" in df.columns
                and "股票简称" in df.columns
            ):
                temp = df[["股票代码", "股票简称"]].copy()
                temp["股票代码"] = temp["股票代码"].apply(self._normalize_stock_code)
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
        if not spot_df.empty and "股票代码" in spot_df.columns:
            spot_df["股票代码"] = spot_df["股票代码"].apply(self._normalize_stock_code)
            if "最新价" in spot_df.columns:
                final_df = pd.merge(
                    final_df,
                    spot_df[["股票代码", "最新价"]].drop_duplicates(subset=["股票代码"]),
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
        final_df["所属行业信号"] = ""
        
        return final_df

    def _calculate_bull_scores(
        self,
        final_df: pd.DataFrame,
        processed_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        计算多头排列评分（批量优化版）
        
        基于K线数据计算每只股票的多头排列得分，评估其趋势强度。
        使用 groupby 预分组和向量化计算，大幅提升性能。
        
        性能优化：
        - 预先将所有K线数据按股票代码分组
        - 批量计算每只股票的评分
        - 避免在循环中重复过滤DataFrame
        
        Args:
            final_df: 基础DataFrame，包含股票代码列
            processed_data: 已处理的原始数据字典，必须包含 'hist_data_all' 或 'kline_data'
            
        Returns:
            pd.DataFrame: 添加了 '多头排列评分' 列的DataFrame，评分范围0-100
        """
        hist_df_all = processed_data.get("hist_data_all")
        if hist_df_all is None:
            hist_df_all = processed_data.get("kline_data", pd.DataFrame())

        if hist_df_all.empty:
            self.logger.warning(
                "[WARN] 历史K线数据为空，无法计算多头排列评分，将填充默认值。"
            )
            final_df["多头排列趋势"] = "趋势观望"
            return final_df

        # 检测日期列
        date_col_candidates = [
            "trade_date", "date", "日期", "datetime", "Date", "TRADE_DATE"
        ]
        date_col_in_kline = next(
            (c for c in date_col_candidates if c in hist_df_all.columns), None
        )
        if date_col_in_kline is None:
            self.logger.warning(
                f"[WARN] K线数据中未找到日期列（候选: {date_col_candidates}）"
            )
            final_df["多头排列趋势"] = "趋势观望"
            return final_df

        # 标准化日期列名
        if date_col_in_kline != "trade_date":
            hist_df_all = hist_df_all.rename(columns={date_col_in_kline: "trade_date"})

        # 检测代码列
        code_col_in_kline = None
        possible_cols = ["symbol", "ts_code", "code", "股票代码"]
        for col in possible_cols:
            if col in hist_df_all.columns:
                code_col_in_kline = col
                break

        if not code_col_in_kline:
            raise KeyError(
                f"无法在K线数据中找到股票代码列。支持的列名: {possible_cols}, "
                f"实际列: {list(hist_df_all.columns)}"
            )

        # 过滤数据到业务日期
        last_trade_day = self.today_str
        hist_df_all["trade_date"] = hist_df_all["trade_date"].astype(str).str[:10]
        hist_df_all = hist_df_all[hist_df_all["trade_date"] <= last_trade_day].copy()
        self.logger.info(
            f"[INFO] 评分用K线截止日期: {last_trade_day}，"
            f"过滤后数据量: {len(hist_df_all)} 行"
        )

        # 标准化K线数据中的股票代码
        hist_df_all["normalized_code"] = hist_df_all[code_col_in_kline].apply(
            self._normalize_stock_code
        )

        # 预计算所有均线（向量化操作，比逐行计算快得多）
        for period in [5, 10, 20, 30, 60, 90, 120]:
            col = f"MA{period}"
            if col not in hist_df_all.columns:
                hist_df_all[col] = (
                    hist_df_all.groupby("normalized_code")["close"]
                    .transform(lambda x: x.rolling(window=period, min_periods=1).mean())
                )
        if "MA_Volume_5" not in hist_df_all.columns:
            hist_df_all["MA_Volume_5"] = (
                hist_df_all.groupby("normalized_code")["volume"]
                .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            )

        # 按股票代码分组
        grouped_klines = hist_df_all.groupby("normalized_code")
        print(f">>> 开始批量计算多头排列评分，共 {len(final_df)} 只股票...")

        # 使用配置中的阈值参数
        thresholds = {
            'full_bull': self.config.FULL_BULL_THRESHOLD,
            'trend_acceleration': self.config.TREND_ACCELERATION_THRESHOLD,
            'trend_oscillation': self.config.TREND_OSCILLATION_THRESHOLD
        }

        # 批量计算评分
        results = {}
        total_stocks = len(final_df)
        processed_count = 0

        for stock_code in final_df["股票代码"]:
            try:
                if stock_code not in grouped_klines.groups:
                    results[stock_code] = "趋势观望"
                    continue

                stock_kline = grouped_klines.get_group(stock_code)
                if stock_kline.empty or len(stock_kline) < 30:
                    results[stock_code] = "趋势观望"
                    continue

                # 按日期排序
                stock_kline = stock_kline.sort_values("trade_date").reset_index(drop=True)

                # 计算评分
                result = calculate_full_bull_score(stock_kline, thresholds=thresholds)
                level = result.get("level", "趋势观望")
                status = result.get("status", "FAILED")
                if status != "SUCCESS":
                    level = "趋势观望"

                results[stock_code] = level

            except Exception as e:
                self.logger.debug(f"计算评分失败 {stock_code}: {e}")
                results[stock_code] = "趋势观望"

            processed_count += 1
            if processed_count % 500 == 0:
                print(f"  已处理 {processed_count}/{total_stocks} 只股票...")

        print(f">>> 多头排列评分计算完成")

        # 将结果添加到 final_df
        final_df["多头排列趋势"] = final_df["股票代码"].map(results).fillna("趋势观望")

        return final_df

    def _merge_fund_flow_data(
        self,
        final_df: pd.DataFrame,
        processed_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合并资金流数据并进行动能分析
        
        使用配置中的 FUND_FLOW_PERIODS 决定处理哪些周期的数据。
        akshare接口限制：仅支持3、5、10、20日四个周期。
        
        功能包括：
        - 根据配置的周期动态获取资金流数据
        - 计算资金流入/流出动能（加速、减速等）
        - 合并到最终报告DataFrame
        
        Args:
            final_df: 基础DataFrame，包含股票代码列
            processed_data: 已处理的原始数据字典，包含各周期的资金流数据
            
        Returns:
            pd.DataFrame: 添加了资金流列和动能列的DataFrame
        """
        # 定义周期映射关系（与akshare接口严格对应）
        period_map = {
            3: ("market_fund_flow_raw_3", "3日资金流入万元"),
            5: ("market_fund_flow_raw", "5日资金流入万元"),
            10: ("market_fund_flow_raw_10", "10日资金流入万元"),
            20: ("market_fund_flow_raw_20", "20日资金流入万元"),
        }
        
        # 根据配置动态处理资金流数据
        for period in self.config.FUND_FLOW_PERIODS:
            if period not in period_map:
                self.logger.warning(f"不支持的资金流周期: {period}日")
                continue
            
            df_key, col_name = period_map[period]
            fund_flow_df = processed_data.get(df_key, pd.DataFrame())
            flow_col = next(
                (col for col in ["净流入", "资金流入净额", "今日主力净流入-净额"]
                 if col in fund_flow_df.columns),
                None,
            )

            if not fund_flow_df.empty and "股票代码" in fund_flow_df.columns and flow_col:
                fund_flow_df["股票代码"] = fund_flow_df["股票代码"].apply(
                    self._normalize_stock_code
                )
                final_df = pd.merge(
                    final_df,
                    fund_flow_df[["股票代码", flow_col]].drop_duplicates(subset=["股票代码"]),
                    on="股票代码",
                    how="left",
                )
                final_df = final_df.rename(columns={flow_col: col_name})
            elif not fund_flow_df.empty and "股票简称" in fund_flow_df.columns and flow_col:
                merge_df = fund_flow_df[["股票简称", flow_col]].drop_duplicates(
                    subset=["股票简称"]
                )
                final_df = pd.merge(final_df, merge_df, on="股票简称", how="left")
                final_df = final_df.rename(columns={flow_col: col_name})
            else:
                final_df[col_name] = 0.0

       

        # 资金流数据标准化处理
        fund_flow_cols = [
            period_map[p][1] for p in self.config.FUND_FLOW_PERIODS if p in period_map
        ]
        if any(col in final_df.columns for col in fund_flow_cols):
            final_df = utils._normalize_fund_data(final_df)

        fund_columns_to_normalize = [
            col for col in fund_flow_cols if col in final_df.columns
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

        # 检查是否所有配置的周期都存在，以计算资金动能
        if all(col in final_df.columns for col in fund_flow_cols):
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
        strong_df = processed_data.get("strong_stocks_raw", pd.DataFrame())
        if not strong_df.empty and "股票代码" in strong_df.columns:
            strong_df["股票代码"] = strong_df["股票代码"].apply(self._normalize_stock_code)
            strong_codes = set(strong_df["股票代码"].tolist())
            final_df["强势股"] = final_df["股票代码"].apply(
                lambda x: "是" if x in strong_codes else "否"
            )
        else:
            final_df["强势股"] = "否"

        # 处理连涨数据
        rise_df = processed_data.get("consecutive_rise_raw", pd.DataFrame())
        if not rise_df.empty and "股票代码" in rise_df.columns:
            rise_df["股票代码"] = rise_df["股票代码"].apply(self._normalize_stock_code)
            rise_df = rise_df[["股票代码", "连涨天数"]].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, rise_df, on="股票代码", how="left").fillna(
                {"连涨天数": 0}
            )
        else:
            final_df["连涨天数"] = 0
        final_df["连涨天数"] = final_df["连涨天数"].astype(int)

        # 处理量价齐升数据
        ljqs_df = processed_data.get("ljqs_raw", pd.DataFrame())
        if not ljqs_df.empty and "股票代码" in ljqs_df.columns:
            ljqs_df["股票代码"] = ljqs_df["股票代码"].apply(self._normalize_stock_code)
            ljqs_codes = set(ljqs_df["股票代码"].tolist())
            final_df["量价齐升"] = final_df["股票代码"].apply(
                lambda x: "是" if x in ljqs_codes else "否"
            )
        else:
            final_df["量价齐升"] = "否"

        # 处理持续放量数据
        cxfl_df = processed_data.get("cxfl_raw", pd.DataFrame())
        if not cxfl_df.empty and "股票代码" in cxfl_df.columns:
            cxfl_df["股票代码"] = cxfl_df["股票代码"].apply(self._normalize_stock_code)
            cxfl_df = cxfl_df[["股票代码", "放量天数"]].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, cxfl_df, on="股票代码", how="left").fillna(
                {"放量天数": 0}
            )
        else:
            final_df["放量天数"] = 0
        final_df["放量天数"] = final_df["放量天数"].astype(int)
        
        return final_df

    def _merge_technical_indicators(
        self,
        final_df: pd.DataFrame,
        processed_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合并技术指标数据：MACD、KDJ、CCI、RSI、BOLL
        """
        ta_dfs_to_merge = []

        # MACD 标准参数（强制保留）
        macd_df_standard = processed_data.get("MACD_12269", pd.DataFrame())
        if not macd_df_standard.empty and "股票代码" in macd_df_standard.columns:
            ta_dfs_to_merge.append(
                macd_df_standard[["股票代码", "MACD_12269_Signal"]].rename(
                    columns={"MACD_12269_Signal": "MACD_12269"}
                )
            )

        # MACD 第二周期（必填）
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"
        macd_key = f"MACD_{second_period_name}"
        macd_df_second = processed_data.get(macd_key, pd.DataFrame())
        if not macd_df_second.empty and "股票代码" in macd_df_second.columns:
            signal_col = f"{macd_key}_Signal"
            ta_dfs_to_merge.append(
                macd_df_second[["股票代码", signal_col]].rename(
                    columns={signal_col: macd_key}
                )
            )

        # MACD 组合背离
        macd_div_df = processed_data.get("MACD_COMBINED_DIVERGENCE", pd.DataFrame())
        if not macd_div_df.empty and "股票代码" in macd_div_df.columns:
            ta_dfs_to_merge.append(
                macd_div_df[["股票代码", "Combined_Divergence_Signal"]].rename(
                    columns={"Combined_Divergence_Signal": "MACD_组合背离"}
                )
            )

        # KDJ
        kdj_df = processed_data.get("KDJ", pd.DataFrame())
        if not kdj_df.empty and "股票代码" in kdj_df.columns:
            ta_dfs_to_merge.append(kdj_df[["股票代码", "KDJ_Signal"]])

        # CCI
        cci_df = processed_data.get("CCI", pd.DataFrame())
        if not cci_df.empty and "股票代码" in cci_df.columns:
            ta_dfs_to_merge.append(cci_df[["股票代码", "CCI_Signal"]])

        # RSI
        rsi_df = processed_data.get("RSI", pd.DataFrame())
        if not rsi_df.empty and "股票代码" in rsi_df.columns:
            rsi_df["RSI_Signal"] = rsi_df["RSI_Signal"].astype(str).str.split(" ").str[0]
            ta_dfs_to_merge.append(rsi_df[["股票代码", "RSI_Signal"]])

        # BOLL
        boll_df = processed_data.get("BOLL", pd.DataFrame())
        if not boll_df.empty and "股票代码" in boll_df.columns:
            ta_dfs_to_merge.append(boll_df[["股票代码", "BOLL_Signal"]])

        # 合并所有技术指标
        for ta_df in ta_dfs_to_merge:
            if "股票代码" in ta_df.columns:
                final_df = pd.merge(
                    final_df,
                    ta_df.drop_duplicates(subset=["股票代码"]),
                    on="股票代码",
                    how="left",
                )

        # 合并 MACD 动能数据
        momentum_df = processed_data.get("MACD_DIF_MOMENTUM", pd.DataFrame())
        if not momentum_df.empty and "股票代码" in momentum_df.columns:
            final_df = pd.merge(final_df, momentum_df, on="股票代码", how="left")
            # 动态填充动能列
            for col in ["MACD_12269_动能"]:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna("")
            
            # 第二周期动能列（必填）
            fast, slow, signal = self.config.MACD_SECOND_PARAMS
            second_period_name = f"{fast}{slow}{signal}"
            mom_col = f"MACD_{second_period_name}_动能"
            if mom_col in final_df.columns:
                final_df[mom_col] = final_df[mom_col].fillna("")

        # 填充缺失的技术指标列
        macd_cols = ["MACD_12269", "MACD_组合背离"]
        # 添加第二周期列名（必填）
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"
        macd_cols.append(f"MACD_{second_period_name}")
        
        for col in macd_cols + ["KDJ_Signal", "CCI_Signal", "RSI_Signal", "BOLL_Signal"]:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna("")
            else:
                final_df[col] = ""
        
        return final_df

    def _merge_special_data(
        self,
        final_df: pd.DataFrame,
        processed_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合并特殊数据：TOP10行业、主力成本、均线突破
        """
        # 处理行业数据
        top_ind_df = processed_data.get("top_industry_cons_df", pd.DataFrame())
        if not top_ind_df.empty and "股票代码" in top_ind_df.columns:
            top_ind_df["股票代码"] = top_ind_df["股票代码"].apply(
                self._normalize_stock_code
            )
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
                main_cost_df["股票代码"] = main_cost_df["股票代码"].apply(
                    self._normalize_stock_code
                )
                final_df = pd.merge(
                    final_df,
                    main_cost_df[
                        [
                            "股票代码", "主力成本", "机构参与度", "主力成本差价",
                            "主力成本差价百分比", "成本位置", "机构参与度等级",
                            "主力控盘强度"
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
            "股票代码", "完全多头排列", "当前价格",
            "10日均线价", "30日均线价", "60日均线价"
        ]
        if not xstp_df.empty and "股票代码" in xstp_df.columns:
            xstp_df["股票代码"] = xstp_df["股票代码"].apply(self._normalize_stock_code)
            cols_present = [col for col in xstp_cols if col in xstp_df.columns]
            merge_df = xstp_df[cols_present].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, merge_df, on="股票代码", how="left")

        if "完全多头排列" not in final_df.columns:
            final_df["完全多头排列"] = "否"
        else:
            final_df["完全多头排列"] = final_df["完全多头排列"].fillna("否")
        
        return final_df

    def _consolidate_data(
        self, processed_data: Dict[str, pd.DataFrame], base_stock_codes_pure: List[str]
    ) -> pd.DataFrame:
        """
        合并所有数据源和信号，生成最终汇总报告。
        
        参数 base_stock_codes_pure 是最终报告的基准股票代码列表（纯数字）。
        
        Args:
            processed_data: 已处理的原始数据字典
            base_stock_codes_pure: 纯数字格式的股票代码列表
            
        Returns:
            pd.DataFrame: 最终汇总报告DataFrame
            
        Raises:
            ValueError: 当关键数据缺失时抛出
        """
        print("\n>>> 正在汇总所有数据和信号 (技术指标作为独立列)...")
        
        # 验证输入数据
        if not base_stock_codes_pure:
            self.logger.warning("[数据验证] 基准股票代码列表为空")
            return pd.DataFrame(columns=["股票代码"])
        
        if not isinstance(processed_data, dict):
            raise TypeError(f"processed_data 必须是字典类型，实际为 {type(processed_data)}")

        # 初始化最终数据框架
        final_df = pd.DataFrame(base_stock_codes_pure, columns=["股票代码"])
        final_df["股票代码"] = final_df["股票代码"].apply(self._normalize_stock_code)

        # 步骤1：合并基础信息（股票名称、实时价格、行业）
        final_df = self._merge_basic_info(final_df, processed_data, base_stock_codes_pure)

        # 步骤2：计算多头排列评分
        final_df = self._calculate_bull_scores(final_df, processed_data)

        # 步骤3：合并资金流数据
        final_df = self._merge_fund_flow_data(final_df, processed_data)

        # 步骤4：合并信号数据（强势股、连涨、量价齐升、持续放量）
        final_df = self._merge_signal_data(final_df, processed_data)

        # 步骤5：合并技术指标（MACD、KDJ、CCI、RSI、BOLL）
        final_df = self._merge_technical_indicators(final_df, processed_data)

        # 步骤6：合并特殊数据（TOP10行业、主力成本、均线突破）
        final_df = self._merge_special_data(final_df, processed_data)

        # 筛选有信号的股票
        # 动态获取第二周期MACD列名
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"
        
        str_cols = [
            "MACD_12269", f"MACD_{second_period_name}", "MACD_组合背离",
            "KDJ_Signal", "CCI_Signal", "RSI_Signal", "BOLL_Signal"
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
        ]
        
        # 动态添加第二周期MACD列（必填）
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"
        signal_cols.extend([
            f"MACD_{second_period_name}",
            f"MACD_{second_period_name}_动能",
            f"MACD_{second_period_name}_DIF",
        ])
        
        signal_cols.extend([
            "MACD_组合背离",
            "KDJ_Signal",
            "CCI_Signal",
            "RSI_Signal",
            "BOLL_Signal",
        ])

        report_cols = [
            "多头排列趋势",
            "资金动能",
        ]
        # 动态添加配置的资金流列（与akshare接口严格对应）
        period_map = {
            3: "3日资金流入万元",
            5: "5日资金流入万元",
            10: "10日资金流入万元",
            20: "20日资金流入万元",
        }
        for period in self.config.FUND_FLOW_PERIODS:
            if period in period_map:
                report_cols.append(period_map[period])
        final_cols = base_cols + signal_cols + report_cols + ["股票链接"]
        final_df = final_df[[col for col in final_cols if col in final_df.columns]]
        
        # 最终数据验证
        if not final_df.empty:
            # 检查必需列
            required_report_cols = ["股票代码", "股票简称", "最新价"]
            is_valid, missing = self.data_validator.validate_required_columns(
                final_df, required_report_cols, "最终报告"
            )
            
            if not is_valid:
                self.logger.error(f"[数据验证] 最终报告缺少关键列: {missing}")
            else:
                # 验证价格数据
                price_valid, anomalies = self.data_validator.validate_price_data(
                    final_df, ['最新价'], "最终报告价格"
                )
                
                if not price_valid:
                    self.logger.warning(f"[数据验证] 最终报告价格异常: {anomalies}")
                
                self.logger.info(
                    f"[数据验证] 最终报告生成成功: {len(final_df)} 条记录, "
                    f"{len(final_df.columns)} 个字段"
                )
        else:
            self.logger.warning("[数据验证] 最终报告为空")

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
                    print(f"工作表 '{sheet_name}' 数据为空，跳过创建。")
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
                        "3日资金流入万元",
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
        latest_prices["股票代码"] = latest_prices["股票代码"].apply(
            self._normalize_stock_code
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
            spot_data = latest_prices_df
            raw_data["spot_data_all"] = spot_data
            raw_data["hist_data_all"] = hist_df_all

            signal_processor = TASignalProcessor(self, config=self.config)
            ta_signals = signal_processor.process_signals(
                final_analysis_codes_prefixed, hist_df_all, spot_data
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
                df["股票代码"] = df["股票代码"].apply(self._normalize_stock_code)
                return df[df["股票代码"].isin(universe_set)].copy()

            # 均线突破数据处理
            processed_xstp_df = self._process_xstp_and_filter(
                raw_data, spot_data
            )
            processed_xstp_df = filter_df_by_universe(
                processed_xstp_df, universe_codes_set_pure
            )

            # 过滤其他每日排名数据
            raw_data["market_fund_flow_raw"] = filter_df_by_universe(
                raw_data.get("market_fund_flow_raw", pd.DataFrame()), universe_codes_set_pure
            )
            raw_data["market_fund_flow_raw_10"] = filter_df_by_universe(
                raw_data.get("market_fund_flow_raw_10", pd.DataFrame()), universe_codes_set_pure
            )
            raw_data["market_fund_flow_raw_20"] = filter_df_by_universe(
                raw_data.get("market_fund_flow_raw_20", pd.DataFrame()), universe_codes_set_pure
            )
            raw_data["strong_stocks_raw"] = filter_df_by_universe(
                raw_data.get("strong_stocks_raw", pd.DataFrame()), universe_codes_set_pure
            )
            raw_data["consecutive_rise_raw"] = filter_df_by_universe(
                raw_data.get("consecutive_rise_raw", pd.DataFrame()), universe_codes_set_pure
            )
            raw_data["ljqs_raw"] = filter_df_by_universe(
                raw_data.get("ljqs_raw", pd.DataFrame()), universe_codes_set_pure
            )
            raw_data["cxfl_raw"] = filter_df_by_universe(
                raw_data.get("cxfl_raw", pd.DataFrame()), universe_codes_set_pure
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
                
                # 动态获取第二周期DIF列名（必填）
                fast, slow, signal = self.config.MACD_SECOND_PARAMS
                second_period_name = f"{fast}{slow}{signal}"
                dif_second_col = f"MACD_{second_period_name}_DIF"
                
                dif_second = pd.to_numeric(
                    consolidated_report.get(dif_second_col), errors="coerce"
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
                # 使用配置中的豁免条件
                exempt_from_drop = full_bull_level.isin(self.config.EXEMPT_LEVELS)

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
                        consolidated_report.get(f"MACD_{second_period_name}_动能")
                        == "加速下跌 (绿柱加长)"
                    )
                    & (dif_12269 < 0)
                    & (dif_second < 0)
                    & kdj_is_empty
                    & (
                        # 使用配置的第一个资金流周期进行检查
                        consolidated_report.get(
                            self._get_first_fund_flow_col(),
                            pd.Series(dtype=str)
                        )
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
                "主力研报筛选": processed_data.get("processed_main_report", pd.DataFrame()),
                "前十板块成分股": raw_data.get("top_industry_cons_df", pd.DataFrame()),
                "主力成本分析": processed_data.get("main_cost_data", pd.DataFrame()),
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
                
                # 获取第二周期名称
                fast, slow, signal = self.config.MACD_SECOND_PARAMS
                second_period_name = f"{fast}{slow}{signal}"

                sync_task.sync_all(
                    today_str=self.today_str,
                    consolidated_report=consolidated_report,
                    industry_df=industry_analysis_df,
                    raw_data=raw_data,
                    second_period_name=second_period_name,
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
