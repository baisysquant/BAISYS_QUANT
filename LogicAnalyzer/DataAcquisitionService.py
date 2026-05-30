"""
数据获取服务类

负责从akshare等外部数据源获取原始数据，并管理数据缓存。
"""

import os
import time
from concurrent.futures import as_completed

import akshare as ak
import pandas as pd
from loguru import logger

from DataManager.ColumnNames import ColumnNames
from DataManager.DataFetcher import DataFetcher
from DataManager.DataSchemas import SchemaValidator
from LogicAnalyzer.DataValidator import DataValidator
from UtilsManager.Exceptions import DataFetchError, handle_exception_with_recovery


class DataAcquisitionService:
    """
    数据获取服务

    职责：
    - 从akshare接口获取原始数据
    - 管理数据缓存
    - 数据源验证
    - 并行获取优化

    Attributes:
        config: 配置管理器实例
        calendar_mgr: 交易日历管理器
        logger: 日志管理器
        cache_manager: 统一缓存管理器
        data_fetcher: 数据获取器（带缓存）
        data_validator: 数据验证器
    """

    def __init__(self, config, calendar_mgr, logger, cache_manager):
        """
        初始化数据获取服务

        Args:
            config: 配置管理器
            calendar_mgr: 交易日历管理器
            logger: 日志管理器
            cache_manager: 统一缓存管理器
        """
        self.config = config
        self.calendar_mgr = calendar_mgr
        self.logger = logger
        self.cache_manager = cache_manager
        self.data_fetcher = DataFetcher(config, calendar_mgr)
        self.data_validator = DataValidator()

    def get_all_raw_data(self, today_str: str) -> dict[str, pd.DataFrame]:
        """
        获取所有原始数据

        该方法负责从多个 akshare 接口获取原始数据，包括：
        - 资金流数据（根据配置的周期动态获取）
        - 强势股池数据
        - 连续上涨、量价齐升、持续放量等技术指标
        - 均线突破数据（10日、30日、60日）
        - 行业板块信息
        - 主力研报盈利预测

        所有数据都通过 DataFetcher 获取，支持自动缓存和重试机制。

        Args:
            today_str: 当前交易日字符串 (YYYYMMDD格式)

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
            - top_industry_cons_df: 前十板块成分股
            - main_cost_data: 主力成本数据

        Raises:
            Exception: 当数据获取失败时抛出异常
        """
        logger.info("\n>>> 正在初始化数据获取和缓存检查...")

        # 根据配置动态获取资金流数据（并行优化版）
        data = {}

        # akshare接口参数映射（严格对应 stock_fund_flow_individual 的 symbol 参数）
        period_to_akshare = {
            3: ("3日市场资金流向", "3日排行", "market_fund_flow_raw_3"),
            5: ("5日市场资金流向", "5日排行", "market_fund_flow_raw"),
            10: ("10日市场资金流向", "10日排行", "market_fund_flow_raw_10"),
            20: ("20日市场资金流向", "20日排行", "market_fund_flow_raw_20"),
        }

        # 构建待获取的任务列表
        fund_flow_tasks = []
        for period in self.config.FUND_FLOW_PERIODS:
            if period in period_to_akshare:
                desc, symbol, key = period_to_akshare[period]
                fund_flow_tasks.append((period, desc, symbol, key))
            else:
                logger.warning(f"不支持的资金流周期: {period}日（仅支持3,5,10,20）")

        # 串行获取所有资金流数据（避免 py_mini_racer 多线程内存冲突）
        logger.info("\n>>> 正在获取资金流数据...")

        for task in fund_flow_tasks:
            period, desc, symbol, key = task
            try:
                logger.info(f"  - 正在获取: {desc}...")
                fund_flow_df = self.data_fetcher.fetch(ak.stock_fund_flow_individual, desc, symbol=symbol)

                # 验证资金流数据
                if not fund_flow_df.empty:
                    required_cols = [ColumnNames.STOCK_CODE, ColumnNames.LATEST_PRICE]
                    is_valid, missing = self.data_validator.validate_required_columns(
                        fund_flow_df, required_cols, f"{period}日资金流"
                    )

                    if is_valid:
                        # Pandera 数据契约校验
                        is_pandera_valid, pandera_errors = SchemaValidator.validate_fund_flow(fund_flow_df)
                        if is_pandera_valid:
                            logger.info(f"  - ✓ 已获取 {period}日资金流数据 ({len(fund_flow_df)} 条记录)")
                            data[key] = fund_flow_df
                        else:
                            logger.warning(f"  - ⚠ {period}日资金流数据契约校验失败: {pandera_errors}")
                            logger.warning(f"  - ⚠ 继续使用该数据，但请注意可能存在数据质量问题")
                            data[key] = fund_flow_df
                    else:
                        logger.warning(f"  - ⚠ {period}日资金流数据缺少列: {missing}，跳过")
                else:
                    logger.warning(f"  - ⚠ {period}日资金流数据为空")

            except Exception as e:
                handle_exception_with_recovery(
                    DataFetchError(f"{period}日资金流", str(e)),
                    self.logger,
                    f"获取{period}日资金流数据",
                    default_value=None,
                    raise_on_critical=False,
                )

        # 获取其他数据源（带验证，并行优化版）
        logger.info("\n>>> 正在并行获取其他技术指标数据...")

        data_sources = {
            "strong_stocks_raw": (ak.stock_zt_pool_strong_em, "强势股池", {"date": today_str}),
            "consecutive_rise_raw": (ak.stock_rank_lxsz_ths, "连续上涨", {}),
            "ljqs_raw": (ak.stock_rank_ljqs_ths, "量价齐升", {}),
            "cxfl_raw": (ak.stock_rank_cxfl_ths, "持续放量", {}),
        }

        # 定义单个数据源获取的worker函数
        def fetch_data_source_worker(item):
            key, (api_func, desc, params) = item
            try:
                df = self.data_fetcher.fetch(api_func, desc, **params)

                if not df.empty:
                    required_cols = [ColumnNames.STOCK_CODE]
                    is_valid, missing = self.data_validator.validate_required_columns(df, required_cols, desc)

                    if is_valid:
                        logger.info(f"  - ✓ {desc}: {len(df)} 条记录")
                        return key, df
                    else:
                        logger.warning(f"  - ⚠ {desc} 缺少列: {missing}")
                        return key, pd.DataFrame()
                else:
                    logger.warning(f"  - ⚠ {desc} 数据为空")
                    return key, pd.DataFrame()

            except Exception as e:
                logger.error(f"  - ✗ 获取{desc}失败: {e}")
                return key, pd.DataFrame()

        # 并行获取所有数据源
        from concurrent.futures import ThreadPoolExecutor

        executor2 = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        futures = {executor2.submit(fetch_data_source_worker, item): item for item in data_sources.items()}

        for future in as_completed(futures):
            try:
                key, df = future.result()
                if not df.empty:
                    data[key] = df
            except Exception as e:
                logger.error(f"获取数据源时发生异常: {e}")

        # 均线突破数据 (Akshare接口参数不同，需分开获取，并行优化版)
        logger.info("\n>>> 正在并行获取均线突破数据...")

        xstp_configs = [
            ("xstp_10_raw", "向上突破10日均线", "10日均线"),
            ("xstp_30_raw", "向上突破30日均线", "30日均线"),
            ("xstp_60_raw", "向上突破60日均线", "60日均线"),
        ]

        # 定义单个均线突破数据获取的worker函数
        def fetch_xstp_worker(config):
            key, desc, symbol = config
            try:
                df = self.data_fetcher.fetch(ak.stock_rank_xstp_ths, desc, symbol=symbol)

                if not df.empty:
                    required_cols = [ColumnNames.STOCK_CODE]
                    is_valid, missing = self.data_validator.validate_required_columns(df, required_cols, desc)

                    if is_valid:
                        logger.info(f"  - ✓ {desc}: {len(df)} 条记录")
                        return key, df
                    else:
                        logger.warning(f"  - ⚠ {desc} 缺少列: {missing}")
                        return key, pd.DataFrame()
                else:
                    logger.warning(f"  - ⚠ {desc} 数据为空")
                    return key, pd.DataFrame()

            except Exception as e:
                logger.error(f"  - ✗ 获取{desc}失败: {e}")
                return key, pd.DataFrame()

        # 并行获取所有均线突破数据
        from concurrent.futures import ThreadPoolExecutor

        executor3 = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)

        futures = {executor3.submit(fetch_xstp_worker, config): config for config in xstp_configs}

        for future in as_completed(futures):
            try:
                key, df = future.result()
                if not df.empty:
                    data[key] = df
            except Exception as e:
                logger.error(f"获取均线突破数据时发生异常: {e}")

        # 行业板块数据
        industry_board_df = self._fetch_industry_data(today_str)
        data["top_industry_cons_df"] = self._get_top_industry_constituents(industry_board_df)
        data["industry_board_df"] = industry_board_df

        # 获取主力成本数据
        logger.info("\n>>> 正在获取主力成本数据...")
        try:
            from LogicAnalyzer.Distribution import MainCostDataManager

            cost_manager = MainCostDataManager(
                cache_enabled=True,
                cache_dir=os.path.join(self.config.TEMP_DATA_DIRECTORY, "cost_data_cache"),
            )
            main_cost_df = cost_manager.get_main_cost_data()

            if not main_cost_df.empty:
                # 标准化列名：将 "代码" 重命名为 "股票代码"
                if (
                    ColumnNames.AKSHARE_CODE_RAW in main_cost_df.columns
                    and ColumnNames.STOCK_CODE not in main_cost_df.columns
                ):
                    main_cost_df.rename(columns={ColumnNames.AKSHARE_CODE_RAW: ColumnNames.STOCK_CODE}, inplace=True)

                # 验证必需列
                required_cols = [ColumnNames.STOCK_CODE, ColumnNames.MAIN_COST]
                is_valid, missing = self.data_validator.validate_required_columns(
                    main_cost_df, required_cols, "主力成本数据"
                )

                if is_valid:
                    # 先分析数据，生成所需的计算列
                    main_cost_df = cost_manager.analyze_cost_data(main_cost_df)
                    
                    # 再进行 Pandera 数据契约校验
                    is_pandera_valid, pandera_errors = SchemaValidator.validate_main_cost(main_cost_df)
                    if is_pandera_valid:
                        data["main_cost_data"] = main_cost_df
                        logger.info(f"  - ✓ 主力成本数据: {len(main_cost_df)} 条记录")
                    else:
                        logger.warning(f"  - ⚠ 主力成本数据契约校验失败: {pandera_errors}")
                        logger.warning(f"  - ⚠ 继续使用该数据，但请注意可能存在数据质量问题")
                        data["main_cost_data"] = main_cost_df

                    # 打印主力成本数据摘要
                    cost_manager.print_cost_summary(main_cost_df)
                else:
                    logger.warning(f"  - ⚠ 主力成本数据缺少列: {missing}")
                    data["main_cost_data"] = pd.DataFrame()
            else:
                logger.warning("  - ⚠ 主力成本数据为空")
                data["main_cost_data"] = pd.DataFrame()

        except Exception as e:
            logger.error(f"  - ✗ 获取主力成本数据失败: {e}")
            data["main_cost_data"] = pd.DataFrame()

        return data

    def _fetch_industry_data(self, today_str: str) -> pd.DataFrame:
        """
        获取行业板块数据

        Args:
            today_str: 当前交易日字符串

        Returns:
            pd.DataFrame: 行业板块数据
        """
        logger.info("\n>>> 正在获取行业板块名称并保存至本地...")
        industry_info_filename = f"行业板块信息_{today_str}.txt"
        industry_info_path = os.path.join(self.config.TEMP_DATA_DIRECTORY, industry_info_filename)
        industry_board_df = pd.DataFrame()

        if os.path.exists(industry_info_path):
            try:
                logger.info(f"  - 发现本地缓存文件，正在读取: {industry_info_filename}")
                industry_board_df = pd.read_csv(industry_info_path, sep="|", encoding="utf-8-sig")

                # 验证缓存数据
                if not industry_board_df.empty:
                    required_cols = [ColumnNames.AKSHARE_INDUSTRY_BOARD_NAME, ColumnNames.AKSHARE_INDUSTRY_BOARD_CODE]
                    is_valid, missing = self.data_validator.validate_required_columns(
                        industry_board_df, required_cols, "行业板块缓存"
                    )

                    if is_valid:
                        # Pandera 数据契约校验
                        is_pandera_valid, pandera_errors = SchemaValidator.validate_industry_board(industry_board_df)
                        if is_pandera_valid:
                            logger.info(f"  - ✓ 缓存数据有效: {len(industry_board_df)} 个板块")
                        else:
                            logger.warning(f"  - ⚠ 行业板块数据契约校验失败: {pandera_errors}")
                            logger.warning(f"  - ⚠ 继续使用该数据，但请注意可能存在数据质量问题")
                    else:
                        logger.warning(f"  - ⚠ 缓存数据缺少列: {missing}，将重新获取")
                        industry_board_df = pd.DataFrame()
                else:
                    logger.warning("  - ⚠ 缓存数据为空，将重新获取")
                    industry_board_df = pd.DataFrame()

            except Exception as e:
                logger.warning(f"  - [WARN] 读取本地缓存失败: {e}，将尝试重新获取...")
                industry_board_df = pd.DataFrame()
        else:
            logger.info("本地无缓存，正在通过接口获取")
            try:
                industry_board_df = ak.stock_board_industry_name_em()

                if not industry_board_df.empty:
                    # 验证接口数据
                    required_cols = [ColumnNames.AKSHARE_INDUSTRY_BOARD_NAME, ColumnNames.AKSHARE_INDUSTRY_BOARD_CODE]
                    is_valid, missing = self.data_validator.validate_required_columns(
                        industry_board_df, required_cols, "行业板块接口"
                    )

                    if is_valid:
                        # Pandera 数据契约校验
                        is_pandera_valid, pandera_errors = SchemaValidator.validate_industry_board(industry_board_df)
                        if is_pandera_valid:
                            try:
                                industry_board_df.to_csv(
                                    industry_info_path,
                                    sep="|",
                                    index=False,
                                    encoding="utf-8-sig",
                                )
                                logger.info(f"  - ✓ 获取成功并已保存: {len(industry_board_df)} 个板块")
                            except Exception as e:
                                logger.error(f"  - ✗ 保存文件失败: {e}")
                        else:
                            logger.warning(f"  - ⚠ 行业板块数据契约校验失败: {pandera_errors}")
                            logger.warning(f"  - ⚠ 继续使用该数据，但请注意可能存在数据质量问题")
                            try:
                                industry_board_df.to_csv(
                                    industry_info_path,
                                    sep="|",
                                    index=False,
                                    encoding="utf-8-sig",
                                )
                                logger.info(f"  - ✓ 获取成功并已保存: {len(industry_board_df)} 个板块")
                            except Exception as e:
                                logger.error(f"  - ✗ 保存文件失败: {e}")
                    else:
                        logger.warning(f"  - ⚠ 接口数据缺少列: {missing}")
                        industry_board_df = pd.DataFrame()
                else:
                    logger.warning("  - ⚠ 行业板块接口返回空数据")

            except Exception as e:
                logger.error(f"  - ✗ 调用行业板块接口失败: {e}")

        return industry_board_df

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

    def _get_top_industry_constituents(self, industry_board_df: pd.DataFrame) -> pd.DataFrame:
        """
        获取前十板块的成分股数据

        Args:
            industry_board_df: 行业板块数据DataFrame

        Returns:
            pd.DataFrame: 前十板块的成分股数据
        """
        if industry_board_df.empty or ColumnNames.AKSHARE_INDUSTRY_BOARD_NAME not in industry_board_df.columns:
            return pd.DataFrame()

        # 1. 使用统一缓存管理器检查缓存
        cache_name = "前十板块成分股"
        cached_df = self.cache_manager.load_dataframe(cache_name)
        if cached_df is not None:
            logger.debug(f"命中板块成分股缓存: {len(cached_df)}条记录")
            return cached_df

        top_industries = industry_board_df.sort_values(by="涨跌幅", ascending=False).head(10)

        industry_list = []
        for _, row in top_industries.iterrows():
            pure_dict = {col: row[col] for col in top_industries.columns}
            industry_list.append(pure_dict)

        def fetch_worker(row):
            try:
                if isinstance(row, pd.Series) or isinstance(row, dict):
                    industry_name = row[ColumnNames.AKSHARE_INDUSTRY_BOARD_NAME]
                else:
                    logger.error(f"[ERROR] 无法识别的数据类型: {type(row)}")
                    return None

                logger.info(f" - 正在获取板块成分股: {industry_name}")
                constituents_df = self._safe_fetch_constituents(symbol=industry_name)

                if constituents_df is not None and not constituents_df.empty:
                    if ColumnNames.AKSHARE_CODE_RAW in constituents_df.columns:
                        constituents_df.rename(
                            columns={ColumnNames.AKSHARE_CODE_RAW: ColumnNames.STOCK_CODE}, inplace=True
                        )

                    if "股票代码" in constituents_df.columns:
                        # 使用统一方法标准化股票代码
                        from UtilsManager.CodeNormalizer import CodeNormalizer

                        constituents_df["股票代码"] = constituents_df["股票代码"].apply(CodeNormalizer.normalize)

                    constituents_df["所属板块"] = industry_name
                    return constituents_df[["股票代码", "所属板块"]].drop_duplicates()
                return None

            except Exception as e:
                logger.error(
                    f"[WORKER ERROR] 处理板块 {row.get(ColumnNames.AKSHARE_INDUSTRY_BOARD_NAME, 'Unknown')} 时出错: {e}"
                )
                return None

        from DataManager import ParallelUtils as utils

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
                final_df = pd.concat(valid_results, ignore_index=True).drop_duplicates(subset=["股票代码"])
                # 使用统一缓存管理器保存
                self.cache_manager.save_dataframe(final_df, cache_name)
                logger.info(f"板块成分股数据已缓存: {len(final_df)}条记录")
                return final_df

        return pd.DataFrame()
