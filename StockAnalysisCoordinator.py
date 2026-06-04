"""
股票分析协调器

负责编排整个股票分析流程，协调各个服务类的工作。
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from DataCollection.HistDataEngine import StockSyncEngine
from DataManager.DataProcessingService import DataProcessingService
from DataManager.ReportService import ReportService
from LogicAnalyzer.AnalysisService import AnalysisService
from LogicAnalyzer.DataAcquisitionService import DataAcquisitionService
from UtilsManager.Exceptions import DatabaseConnectionError
from UtilsManager.UnifiedCacheManager import UnifiedCacheManager


class StockAnalysisCoordinator:
    """
    股票分析协调器

    职责：
    - 编排分析流程
    - 异常处理和日志记录
    - 性能监控

    Attributes:
        config: 配置管理器实例
        calendar_mgr: 交易日历管理器
        today_str: 当前交易日字符串
        logger: 日志管理器
        cache_manager: 统一缓存管理器
        stock_sync_engine: 股票同步引擎
        db_engine: 数据库引擎
        executor: 线程池执行器
        data_acquisition: 数据获取服务
        data_processing: 数据处理服务
        analysis_service: 业务分析服务
        report_service: 报告生成服务
    """

    def __init__(
        self,
        config: Config,
        calendar_mgr: TradingCalendarAnalyzer,
        logger: Any,
        cache_manager: UnifiedCacheManager,
        stock_sync_engine: StockSyncEngine,
        db_engine: Engine,
        executor: ThreadPoolExecutor,
        data_acquisition: DataAcquisitionService,
        data_processing: DataProcessingService,
        analysis_service: AnalysisService,
        report_service: ReportService,
        today_str: str | None = None,
    ):
        """
        初始化股票分析协调器（完全依赖注入）

        Args:
            config: 配置管理器
            calendar_mgr: 交易日历管理器
            logger: 日志管理器
            cache_manager: 统一缓存管理器
            stock_sync_engine: 股票同步引擎
            db_engine: 数据库引擎
            executor: 线程池执行器
            data_acquisition: 数据获取服务
            data_processing: 数据处理服务
            analysis_service: 业务分析服务
            report_service: 报告生成服务
            today_str: 当前交易日（可选，默认从calendar_mgr获取）
        """
        self.config = config
        self.calendar_mgr = calendar_mgr
        self.logger = logger
        self.cache_manager = cache_manager
        self.stock_sync_engine = stock_sync_engine
        self.db_engine = db_engine
        self.executor = executor
        self.data_acquisition = data_acquisition
        self.data_processing = data_processing
        self.analysis_service = analysis_service
        self.report_service = report_service

        self.today_str = today_str or self.calendar_mgr.get_last_trading_day()
        self.start_time = time.time()

 
    def run(self) -> None:
        """
        执行完整的股票分析流程

        流程步骤：
        1. 同步历史数据到数据库
        2. 获取待分析股票代码列表
        3. 获取所有原始数据
        4. 获取K线数据并提取最新价格
        5. 处理技术指标信号
        6. 运行行业分析
        7. 处理均线突破数据
        8. 合并和处理数据
        9. 映射行业信号
        10. 剔除弱势股
        11. 生成报告
        12. 同步到数据库
        """
        self.logger.info(f"[INFO] 股票分析程序启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"[INFO] 最后一个交易日为: {self.today_str}")

        try:
            # 步骤1：同步历史数据
            self._sync_historical_data()

            # 步骤2：获取股票代码列表
            stock_codes_prefixed, stock_codes_pure = self._get_stock_codes_from_db()
            if not stock_codes_prefixed:
                self.logger.critical("未获取到股票代码，流程终止")
                return

            self.logger.info(
                f">>> HistDataWatchDog 成功同步 {len(stock_codes_prefixed)} 只股票数据到数据库，并作为分析基础。"
            )

            # 步骤3：获取原始数据
            raw_data = self.data_acquisition.get_all_raw_data(self.today_str)

            # 步骤4：获取K线数据并提取最新价格
            hist_df, spot_data = self._get_kline_and_prices(stock_codes_prefixed)

            # 步骤5：处理技术指标信号
       
            ta_signals = self.analysis_service.process_technical_signals(stock_codes_prefixed, hist_df, spot_data)
            self.report_service.save_ta_signals_to_txt(ta_signals, self.today_str)

 
            print("\n=== 技术指标数据检查 ===")
            for key, df in ta_signals.items():
                  if isinstance(df, pd.DataFrame) and not df.empty:
                     print(f"{key}: {len(df)} 条数据，列名: {list(df.columns)}")
                     print(f"  样本数据:\n{df.head(2)}")
                  else:
                      print(f"{key}: 空DataFrame")


            # 步骤6：运行行业分析
            industry_df = self.analysis_service.run_industry_analysis()

            # 步骤7：处理均线突破数据
            processed_xstp_df = self.analysis_service.process_xstp_and_filter(raw_data, spot_data)
            processed_xstp_df = self._filter_by_universe(processed_xstp_df, set(stock_codes_pure))

            # 过滤其他每日排名数据
            raw_data = self._filter_raw_data(raw_data, set(stock_codes_pure))

            # 步骤8：准备处理后的数据字典
            processed_data = self._prepare_processed_data(
                raw_data, ta_signals, hist_df, spot_data, industry_df, processed_xstp_df
            )

            # 步骤9：合并和处理数据
            consolidated_report = self.data_processing.consolidate_data(processed_data, stock_codes_pure)

            # 步骤10：映射行业信号
            consolidated_report = self.analysis_service.merge_industry_signal_to_stocks(
                consolidated_report, industry_df
            )

            # 调整列顺序：将"所属行业信号"放在"行业"后面
            cols = list(consolidated_report.columns)
            if "所属行业信号" in cols and "行业" in cols:
                cols.remove("所属行业信号")
                idx = cols.index("行业")
                cols.insert(idx + 1, "所属行业信号")
                consolidated_report = consolidated_report[cols]

            # 步骤11：剔除弱势股
            consolidated_report = self.analysis_service.filter_weak_stocks(consolidated_report)

            # 步骤12：生成报告
            sheets_data = self._prepare_sheets_data(consolidated_report, industry_df, processed_data)
            self.report_service.generate_excel_report(sheets_data, self.today_str)

            # 步骤13：同步到数据库
            self._sync_results_to_database(consolidated_report, industry_df, raw_data)

            end_time = time.time()
            self.logger.info(f"\n>>> 流程结束。总耗时: {timedelta(seconds=end_time - self.start_time)}")

        except Exception as e:
            # 顶层异常处理：记录致命错误并重新抛出
            if isinstance(e, DatabaseConnectionError):
                self.logger.critical(f"\n[致命错误] 数据库连接失败，流程终止: {e}")
            else:
                self.logger.critical(f"\n[致命错误] 数据分析流程意外终止: {type(e).__name__}: {e}")
            raise

        finally:
            self.executor.shutdown(wait=True)

    def _sync_historical_data(self) -> None:
        """同步历史数据到数据库"""
        self.logger.info(">>> 正在同步历史数据到数据库...")
        self.stock_sync_engine.run_engine(target_date=self.today_str)

    def _load_research_report_data(self) -> pd.DataFrame:
        """
        加载研报数据（从缓存文件）

        Returns:
            pd.DataFrame: 包含股票代码和研报买入次数的DataFrame
        """
        try:
            report_cache_path = self.stock_sync_engine.cache_manager.get_cache_path("研报买入次数", cleaned=True)

            if os.path.exists(report_cache_path):
                report_df = pd.read_csv(report_cache_path, sep="|", encoding="utf-8-sig", dtype={"股票代码": str})
                self.logger.info(f"  - 已加载研报数据: {len(report_df)} 条记录")
                return report_df
            else:
                self.logger.warning("  - 研报数据缓存文件不存在")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"  - 加载研报数据失败: {e}")
            return pd.DataFrame()

    def _get_stock_codes_from_db(self) -> tuple[list[str], list[str]]:
        """
        从数据库获取待分析股票代码列表

        Returns:
            tuple: (带前缀的代码列表, 纯数字代码列表)
        """
        synced_codes_df_from_db = pd.DataFrame(columns=["symbol"])

        try:
            if self.db_engine is None:
                raise RuntimeError("数据库引擎未成功初始化，无法从数据库获取数据。")

            with self.db_engine.connect() as conn:
                # 查询数据库中最新的一个交易日期
                latest_date_query = text("SELECT MAX(trade_date) FROM stock_daily_kline;")
                latest_db_date_result = conn.execute(latest_date_query).scalar_one_or_none()

                if latest_db_date_result is None:
                    self.logger.critical(
                        "[FATAL] 数据库中 'stock_daily_kline' 表没有K线数据，无法获取股票代码列表，流程终止。"
                    )
                    return [], []

                # 查询在该最新交易日期有数据的股票代码
                query_symbols = text(
                    """
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
                self.logger.info(f"已从数据库获取 {len(synced_codes_df_from_db)} 只股票代码，基于最新交易日")

        except Exception as e:
            self.logger.critical(f"[FATAL] 查询数据库获取股票代码失败: {e}，流程终止。")
            return [], []

        if synced_codes_df_from_db.empty:
            self.logger.critical("[FATAL] 从数据库获取已同步股票代码列表失败，流程终止。")
            return [], []

        final_analysis_codes_prefixed = synced_codes_df_from_db["symbol"].tolist()
        final_analysis_codes_pure = [code[2:] for code in final_analysis_codes_prefixed]

        return final_analysis_codes_prefixed, final_analysis_codes_pure

    def _get_kline_and_prices(self, stock_codes_prefixed: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        从数据库获取K线数据并提取最新价格

        Args:
            stock_codes_prefixed: 带前缀的股票代码列表

        Returns:
            tuple: (K线数据DataFrame, 最新价格DataFrame)
        """
        self.logger.info("\n>>> 从K线数据获取最新收盘价...")

        if not stock_codes_prefixed:
            self.logger.warning("[WARN] 待分析股票代码列表为空，跳过历史数据查询。")
            return pd.DataFrame(), pd.DataFrame()

        query = text(
            """
            SELECT *
            FROM stock_daily_kline
            WHERE symbol = ANY(:symbols)
            ORDER BY trade_date
            """
        )

        hist_df_all = pd.DataFrame()
        try:
            with self.db_engine.connect() as conn:
                hist_df_all = pd.read_sql(query, conn, params={"symbols": list(stock_codes_prefixed)})

                if not hist_df_all.empty:
                    self.logger.info(
                        f"[INFO] 数据日期范围: {hist_df_all['trade_date'].min()} 至 {hist_df_all['trade_date'].max()}"
                    )
                else:
                    self.logger.error("[ERROR] 查询结果为空！可能是股票代码不匹配或日期条件过滤了所有数据。")

        except Exception as e:
            self.logger.error(f"[ERROR] 数据库查询失败: {e}")
            # 尝试回滚事务
            try:
                with self.db_engine.connect() as conn:
                    conn.rollback()
            except Exception:
                pass
            hist_df_all = pd.DataFrame()

        if hist_df_all.empty:
            self.logger.warning("[WARN] 由于历史数据为空，将跳过所有技术指标计算。")

        # 从K线数据获取最新价格
        from UtilsManager.PriceExtractor import PriceExtractor

        latest_prices_df = PriceExtractor.extract_latest_prices(hist_df_all)
        self.logger.info(f"[INFO] 从K线数据获取了 {len(latest_prices_df)} 只股票的最新收盘价")

        return hist_df_all, latest_prices_df

    def _filter_by_universe(self, df: pd.DataFrame, universe_set: set) -> pd.DataFrame:
        """
        根据股票池过滤DataFrame

        Args:
            df: 待过滤的DataFrame
            universe_set: 股票代码集合

        Returns:
            pd.DataFrame: 过滤后的DataFrame
        """
        from UtilsManager.CodeNormalizer import CodeNormalizer

        if df is None or df.empty or "股票代码" not in df.columns:
            return pd.DataFrame()

        df["股票代码"] = CodeNormalizer.normalize_series(df["股票代码"])
        return df[df["股票代码"].isin(universe_set)].copy()

    def _filter_raw_data(self, raw_data: dict[str, pd.DataFrame], universe_set: set) -> dict[str, pd.DataFrame]:
        """
        过滤原始数据中的各个DataFrame

        Args:
            raw_data: 原始数据字典
            universe_set: 股票代码集合

        Returns:
            Dict[str, pd.DataFrame]: 过滤后的数据字典
        """
        keys_to_filter = [
            "market_fund_flow_raw",
            "market_fund_flow_raw_10",
            "market_fund_flow_raw_20",
            "strong_stocks_raw",
            "consecutive_rise_raw",
            "ljqs_raw",
            "cxfl_raw",
        ]

        for key in keys_to_filter:
            if key in raw_data:
                raw_data[key] = self._filter_by_universe(raw_data.get(key, pd.DataFrame()), universe_set)

        return raw_data

    def _prepare_processed_data(
        self,
        raw_data: dict[str, pd.DataFrame],
        ta_signals: dict[str, pd.DataFrame],
        hist_df: pd.DataFrame,
        spot_data: pd.DataFrame,
        industry_df: pd.DataFrame,
        processed_xstp_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        准备处理后的数据字典

        Args:
            raw_data: 原始数据字典
            ta_signals: 技术指标信号字典
            hist_df: K线数据
            spot_data: 最新价格数据
            industry_df: 行业分析结果
            processed_xstp_df: 处理后的均线突破数据

        Returns:
            Dict[str, pd.DataFrame]: 处理后的数据字典
        """
        processed_data = {
            **raw_data,
            **ta_signals,
            "processed_xstp_df": processed_xstp_df,
            "processed_main_report": pd.DataFrame(),
            "individual_industry": industry_df,
            "hist_data_all": hist_df,
            "spot_data_all": spot_data,
        }

        # 加载研报数据（作为加分因子）
        report_df = self._load_research_report_data()
        if not report_df.empty:
            processed_data["research_report_data"] = report_df

        return processed_data

    def _prepare_sheets_data(
        self, consolidated_report: pd.DataFrame, industry_df: pd.DataFrame, processed_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        准备Excel报告的sheets数据

        Args:
            consolidated_report: 汇总报告
            industry_df: 行业分析结果
            processed_data: 处理后的数据字典

        Returns:
            Dict[str, pd.DataFrame]: sheets数据字典
        """
        sheets_data = {
            "数据汇总": consolidated_report,
            "行业深度分析": industry_df,
            "主力研报筛选": processed_data.get("processed_main_report", pd.DataFrame()),
            "前十板块成分股": processed_data.get("top_industry_cons_df", pd.DataFrame()),
            "主力成本分析": processed_data.get("main_cost_data", pd.DataFrame()),
        }

        return sheets_data

    def _sync_results_to_database(
        self, consolidated_report: pd.DataFrame, industry_df: pd.DataFrame, raw_data: dict[str, pd.DataFrame]
    ) -> None:
        """
        同步分析结果到数据库

        Args:
            consolidated_report: 汇总报告
            industry_df: 行业分析结果
            raw_data: 原始数据字典
        """
        try:
            # 获取第二周期名称
            fast, slow, signal = self.config.MACD_SECOND_PARAMS
            second_period_name = f"{fast}{slow}{signal}"

            success = self.report_service.sync_to_database(
                today_str=self.today_str,
                consolidated_report=consolidated_report,
                industry_df=industry_df,
                raw_data=raw_data,
                second_period_name=second_period_name,
            )

            if not success:
                self.logger.warning("数据库同步失败，但流程继续")

        except Exception as e:
            self.logger.error(f"!!! [同步中断] 任务运行异常: {e}")


class StockAnalysisCoordinatorFactory:
    """
    股票分析协调器工厂类

    负责组装和初始化所有依赖项，然后创建 StockAnalysisCoordinator 实例。
    这样可以将对象创建逻辑与业务逻辑分离。
    """

    @classmethod
    def create(
        cls,
        config_file: str = "config.ini",
    ) -> StockAnalysisCoordinator:
        """
        创建并组装完整的 StockAnalysisCoordinator

        Args:
            config_file: 配置文件路径

        Returns:
            完全初始化的 StockAnalysisCoordinator 实例

        Raises:
            DatabaseConnectionError: 如果数据库连接失败
        """
        # 导入在函数内部，避免循环依赖
        from LogicAnalyzer.FundMomentumAnalyzer import FundMomentumAnalyzer
        from UtilsManager.LoggerManager import get_logger
        from UtilsManager.UnifiedCacheManager import CacheStrategy

        # 1. 初始化配置和基础设施
        config = Config(config_file=config_file)
        calendar_mgr = TradingCalendarAnalyzer()
        today_str = calendar_mgr.get_last_trading_day()

        # 2. 初始化日志
        logger = get_logger(
            log_dir=config.LOG_DIR,
            log_filename=f"Corenews_Main_{today_str}.log",
            level=config.LOG_LEVEL,
        )

        # 3. 初始化缓存管理器
        cache_dir = os.path.join(config.TEMP_DATA_DIRECTORY, "cache")
        cache_manager = UnifiedCacheManager(
            cache_dir=cache_dir, default_strategy=CacheStrategy.DAILY, auto_cleanup=True
        )

        # 4. 初始化数据库引擎
        try:
            stock_sync_engine = StockSyncEngine()
            db_engine = stock_sync_engine.db

            # 启动时执行股票基本信息同步（物理日期锁机制）
            from DataCollection.GetStockBasicinfo import StockBasicInfoService

            basic_info_service = StockBasicInfoService(config)
            basic_info_service.sync_all_stock_basic_info()

        except Exception as e:
            raise DatabaseConnectionError(f"初始化数据库引擎失败: {e}")

        # 5. 初始化线程池
        executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

        # 将executor注入到config中，供服务类使用
        config.executor = executor

        # 6. 初始化所有业务服务
        data_acquisition = DataAcquisitionService(config, calendar_mgr, logger, cache_manager)

        fund_momentum_analyzer = FundMomentumAnalyzer()
        data_processing = DataProcessingService(config, logger, fund_momentum_analyzer, calendar_mgr)

        analysis_service = AnalysisService(config, logger, db_engine)

        report_service = ReportService(config, logger)

        # 7. 创建并返回协调器
        return StockAnalysisCoordinator(
            config=config,
            calendar_mgr=calendar_mgr,
            logger=logger,
            cache_manager=cache_manager,
            stock_sync_engine=stock_sync_engine,
            db_engine=db_engine,
            executor=executor,
            data_acquisition=data_acquisition,
            data_processing=data_processing,
            analysis_service=analysis_service,
            report_service=report_service,
            today_str=today_str,
        )
