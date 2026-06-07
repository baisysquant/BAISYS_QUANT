"""
股票分析协调器

负责编排整个股票分析流程，协调各个服务类的工作。

Pipeline 设计：
  - 每个分析步骤封装为独立的 _step_N_xxx 方法
  - 步骤间通过 PipelineContext 传递数据，互不直接依赖
  - 每步独立 try/except，单步失败不会导致整个流程崩溃
  - 可单独构造 PipelineContext 调用任一步骤进行单元测试
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError

from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from DataCollection.HistDataEngine import StockSyncEngine
from DataManager.DataProcessingService import DataProcessingService
from DataManager.ReportService import ReportService
from LogicAnalyzer.AnalysisService import AnalysisService
from LogicAnalyzer.DataAcquisitionService import DataAcquisitionService
from UtilsManager.Exceptions import DatabaseConnectionError
from UtilsManager.UnifiedCacheManager import UnifiedCacheManager


@dataclass
class PipelineContext:
    """流水线上下文：步骤之间通过此对象交换数据，解除顺序耦合"""

    data: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def has(self, *keys: str) -> bool:
        return all(k in self.data for k in keys)

    def record_error(self, step_name: str, message: str) -> None:
        self.errors[step_name] = message


class StockAnalysisCoordinator:
    """
    股票分析协调器

    职责：
    - 编排分析流程（Pipeline 模式）
    - 每步独立异常处理，单步失败不影响后续无关步骤
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

    # ──────────────────────────────────────────────
    # Pipeline 定义
    # ──────────────────────────────────────────────

    def run(self) -> None:
        """
        按序执行流水线各步骤。每步独立 try/except，
        致命步骤（如无股票代码）会终止流程，非致命步骤失败仅记录。
        """
        self.logger.info(f"[INFO] 股票分析程序启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"[INFO] 最后一个交易日为: {self.today_str}")

        ctx = PipelineContext()

        pipeline = [
            ("同步历史数据", self._step_1_sync_data, True),
            ("格式化股票代码", self._step_2_format_codes, True),
            ("获取原始数据", self._step_3_get_raw_data, False),
            ("获取K线数据及最新价", self._step_4_get_kline_and_prices, True),
            ("处理技术指标信号", self._step_5_technical_signals, False),
            ("运行行业分析", self._step_6_industry_analysis, False),
            ("处理均线突破数据", self._step_7_xstp_and_filter, False),
            ("准备处理数据字典", self._step_8_prepare_processed_data, False),
            ("合并处理数据", self._step_9_consolidate_data, True),
            ("映射行业信号", self._step_10_merge_industry_signal, False),
            ("剔除弱势股", self._step_11_filter_weak_stocks, False),
            ("生成Excel报告", self._step_12_generate_report, False),
            ("同步结果到数据库", self._step_13_sync_to_database, False),
        ]

        for step_name, step_fn, fatal in pipeline:
            ok = self._run_single_step(step_name, step_fn, ctx)
            if not ok and fatal:
                self.logger.critical(f"[流水线终止] 致命步骤 '{step_name}' 失败，结束流程")
                self._shutdown()
                return

        self.logger.info(
            f"\n>>> 流程结束。总耗时: {timedelta(seconds=time.time() - self.start_time)}"
        )
        self._shutdown()

    def _run_single_step(self, name: str, fn, ctx: PipelineContext) -> bool:
        try:
            self.logger.info(f">>> 步骤: {name}")
            return fn(ctx)
        except Exception as e:
            ctx.record_error(name, str(e))
            self.logger.error(f"[步骤失败] {name}: {type(e).__name__}: {e}")
            return False

    def _shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    # ──────────────────────────────────────────────
    # 各步骤实现（可独立测试）
    # ──────────────────────────────────────────────

    def _step_1_sync_data(self, ctx: PipelineContext) -> bool:
        self.logger.info(">>> 正在同步历史数据到数据库...")
        filtered_pure_codes = self.stock_sync_engine.run_engine(target_date=self.today_str)
        if not filtered_pure_codes:
            self.logger.critical("同步历史数据后无有效股票代码，流程终止")
            return False
        ctx.set("filtered_pure_codes", filtered_pure_codes)
        return True

    def _step_2_format_codes(self, ctx: PipelineContext) -> bool:
        from DataManager.ShareCodeFormatMgr import format_stock_code

        filtered_pure_codes: set = ctx.get("filtered_pure_codes")
        stock_codes_prefixed = [format_stock_code(code) for code in sorted(filtered_pure_codes)]
        stock_codes_pure = sorted(filtered_pure_codes)
        ctx.set("stock_codes_prefixed", stock_codes_prefixed)
        ctx.set("stock_codes_pure", stock_codes_pure)
        self.logger.info(
            f">>> HistDataWatchDog 成功同步 {len(stock_codes_prefixed)} 只股票数据到数据库，并作为分析基础。"
        )
        return True

    def _step_3_get_raw_data(self, ctx: PipelineContext) -> bool:
        raw_data = self.data_acquisition.get_all_raw_data(self.today_str)
        ctx.set("raw_data", raw_data)
        return True

    def _step_4_get_kline_and_prices(self, ctx: PipelineContext) -> bool:
        stock_codes_prefixed: list[str] = ctx.get("stock_codes_prefixed", [])
        self.logger.info("\n>>> 从K线数据获取最新收盘价...")

        if not stock_codes_prefixed:
            self.logger.warning("[WARN] 待分析股票代码列表为空，跳过历史数据查询。")
            ctx.set("hist_df", pd.DataFrame())
            ctx.set("spot_data", pd.DataFrame())
            return True

        query = text("""
            SELECT * FROM stock_daily_kline
            WHERE symbol = ANY(:symbols)
            ORDER BY trade_date
        """)

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
        except (DBAPIError, OperationalError) as e:
            self.logger.error(f"[ERROR] 数据库查询失败: {e}")
            try:
                with self.db_engine.connect() as conn:
                    conn.rollback()
            except (DBAPIError, OperationalError):
                pass
            hist_df_all = pd.DataFrame()

        if hist_df_all.empty:
            self.logger.warning("[WARN] 由于历史数据为空，将跳过所有技术指标计算。")

        from UtilsManager.PriceExtractor import PriceExtractor
        latest_prices_df = PriceExtractor.extract_latest_prices(hist_df_all)
        self.logger.info(f"[INFO] 从K线数据获取了 {len(latest_prices_df)} 只股票的最新收盘价")

        ctx.set("hist_df", hist_df_all)
        ctx.set("spot_data", latest_prices_df)
        return True

    def _step_5_technical_signals(self, ctx: PipelineContext) -> bool:
        if not ctx.has("stock_codes_prefixed", "hist_df", "spot_data"):
            self.logger.warning("[SKIP] 技术指标信号缺少前置依赖")
            return False

        stock_codes_prefixed: list[str] = ctx.get("stock_codes_prefixed")
        hist_df: pd.DataFrame = ctx.get("hist_df")
        spot_data: pd.DataFrame = ctx.get("spot_data")

        if hist_df.empty:
            self.logger.warning("[SKIP] K线数据为空，跳过技术指标计算")
            return False

        ta_signals = self.analysis_service.process_technical_signals(stock_codes_prefixed, hist_df, spot_data)
        self.report_service.save_ta_signals_to_txt(ta_signals, self.today_str)

        print("\n=== 技术指标数据检查 ===")
        for key, df in ta_signals.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"{key}: {len(df)} 条数据，列名: {list(df.columns)}")
                print(f"  样本数据:\n{df.head(2)}")
            else:
                print(f"{key}: 空DataFrame")

        ctx.set("ta_signals", ta_signals)
        return True

    def _step_6_industry_analysis(self, ctx: PipelineContext) -> bool:
        industry_df = self.analysis_service.run_industry_analysis()
        ctx.set("industry_df", industry_df)
        return True

    def _step_7_xstp_and_filter(self, ctx: PipelineContext) -> bool:
        raw_data: dict = ctx.get("raw_data", {})
        spot_data: pd.DataFrame = ctx.get("spot_data", pd.DataFrame())
        stock_codes_pure: list[str] = ctx.get("stock_codes_pure", [])

        if not raw_data or spot_data.empty:
            self.logger.warning("[SKIP] 均线突破处理缺少原始数据或价格数据")
            return False

        processed_xstp_df = self.analysis_service.process_xstp_and_filter(raw_data, spot_data)
        universe = set(stock_codes_pure)
        processed_xstp_df = self._filter_by_universe(processed_xstp_df, universe)
        raw_data = self._filter_raw_data(raw_data, universe)

        ctx.set("processed_xstp_df", processed_xstp_df)
        ctx.set("raw_data", raw_data)
        return True

    def _step_8_prepare_processed_data(self, ctx: PipelineContext) -> bool:
        processed_data = {
            **(ctx.get("raw_data", {})),
            **(ctx.get("ta_signals", {})),
            "processed_xstp_df": ctx.get("processed_xstp_df", pd.DataFrame()),
            "processed_main_report": pd.DataFrame(),
            "individual_industry": ctx.get("industry_df", pd.DataFrame()),
            "hist_data_all": ctx.get("hist_df", pd.DataFrame()),
            "spot_data_all": ctx.get("spot_data", pd.DataFrame()),
        }

        report_df = self._load_research_report_data()
        if not report_df.empty:
            processed_data["research_report_data"] = report_df

        ctx.set("processed_data", processed_data)
        return True

    def _step_9_consolidate_data(self, ctx: PipelineContext) -> bool:
        processed_data: dict = ctx.get("processed_data", {})
        stock_codes_pure: list[str] = ctx.get("stock_codes_pure", [])

        if not processed_data or not stock_codes_pure:
            self.logger.warning("[SKIP] 合并数据缺少依赖")
            return False

        consolidated_report = self.data_processing.consolidate_data(processed_data, stock_codes_pure)
        ctx.set("consolidated_report", consolidated_report)
        return True

    def _step_10_merge_industry_signal(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        industry_df: pd.DataFrame = ctx.get("industry_df", pd.DataFrame())

        if consolidated_report.empty or industry_df.empty:
            self.logger.warning("[SKIP] 行业信号映射缺少数据")
            return False

        consolidated_report = self.analysis_service.merge_industry_signal_to_stocks(
            consolidated_report, industry_df
        )

        cols = list(consolidated_report.columns)
        if "所属行业信号" in cols and "行业" in cols:
            cols.remove("所属行业信号")
            idx = cols.index("行业")
            cols.insert(idx + 1, "所属行业信号")
            consolidated_report = consolidated_report[cols]

        # ── 行业内百分位排名 + 背离检测 ──────────────────────────────────
        consolidated_report = self._apply_industry_neutralization(consolidated_report)

        ctx.set("consolidated_report", consolidated_report)
        return True

    @staticmethod
    def _apply_industry_neutralization(df: pd.DataFrame) -> pd.DataFrame:
        """行业内百分位排名 & 个股-行业背离折扣。

        机构做法：用行业内 percentile rank 消除行业间系统性偏差，
        再与行业信号强度对比，发现背离时扣分。
        """
        from DataManager.ColumnNames import ColumnNames as CN

        SCORE_COL = CN.COMPREHENSIVE_SCORE
        IND_COL = CN.INDUSTRY
        SIG_COL = CN.INDUSTRY_SIGNAL

        if SCORE_COL not in df.columns or IND_COL not in df.columns:
            return df

        has_signal = SIG_COL in df.columns

        # 行业信号 → 数值映射 (0-100)
        SIGNAL_SCORE_MAP = {
            "核心配置 (低估值+强趋势)": 80,
            "动量追击 (高景气+资金涌入)": 70,
            "左侧潜伏 (极度低估+等待拐点)": 50,
            "均衡/观望": 40,
            "情绪过热 (高估+趋势透支)": 30,
        }

        # 1) 行业内百分位 (cross-sectional, 消除行业偏差)
        df[CN.INDUSTRY_PERCENTILE] = (
            df.groupby(IND_COL)[SCORE_COL].rank(pct=True) * 100
        ).fillna(50.0)

        # 2) 所属行业信号 → 行业信号评分
        if has_signal:
            df[CN.INDUSTRY_SIGNAL_SCORE] = (
                df[SIG_COL].map(SIGNAL_SCORE_MAP).fillna(50)
            )
        else:
            df[CN.INDUSTRY_SIGNAL_SCORE] = 50

        # 3) 背离检测
        ind_score = df[CN.INDUSTRY_SIGNAL_SCORE]
        pct = df[CN.INDUSTRY_PERCENTILE]

        cond_low = (pct <= 25) & (ind_score >= 70)
        cond_high = (pct >= 75) & (ind_score <= 40)

        df[CN.INDUSTRY_DEVIATION] = 0
        df.loc[cond_low, CN.INDUSTRY_DEVIATION] = -10
        df.loc[cond_high, CN.INDUSTRY_DEVIATION] = -5

        # 4) 扣分
        discount = df[CN.INDUSTRY_DEVIATION]
        df[SCORE_COL] = (df[SCORE_COL] + discount).clip(lower=0)

        return df

    def _step_11_filter_weak_stocks(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        if consolidated_report.empty:
            return False
        consolidated_report = self.analysis_service.filter_weak_stocks(consolidated_report)
        ctx.set("consolidated_report", consolidated_report)
        return True

    def _step_12_generate_report(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        industry_df: pd.DataFrame = ctx.get("industry_df", pd.DataFrame())
        processed_data: dict = ctx.get("processed_data", {})

        # 裁剪仅保留 final column order 中的列（step 10 可能加入了计算用列）
        from DataManager.ReportService import ReportService
        from DataManager.ColumnNames import ColumnNames as CN
        final_cols = ReportService.get_final_column_order(
            fund_flow_periods=self.config.FUND_FLOW_PERIODS
        )
        existing_cols = [c for c in final_cols if c in consolidated_report.columns]
        # 明确剔除计算用列（即使因命名不一致混入）
        drop_cols = {CN.INDUSTRY_PERCENTILE, CN.INDUSTRY_SIGNAL_SCORE, CN.INDUSTRY_DEVIATION}
        consolidated_report = consolidated_report[[c for c in existing_cols if c not in drop_cols]]

        sheets_data = self._prepare_sheets_data(consolidated_report, industry_df, processed_data)
        self.report_service.generate_excel_report(sheets_data, self.today_str)
        self._validate_report_integrity(consolidated_report)
        return True

    def _validate_report_integrity(self, df: pd.DataFrame):
        if df.empty:
            self.logger.warning("[完整性断言] 报告为空，跳过校验")
            return
        total = len(df)
        warnings = []
        dim_cols = ["MACD趋势", "金叉信号", "柱状动能", "DIF斜率", "背离信号", "量价配合", "K线形态"]
        for col in dim_cols:
            if col in df.columns:
                empty = df[col].astype(str).str.strip().eq("").sum()
                ratio = empty / total * 100
                if ratio > 50:
                    warnings.append(f"  '{col}' 空值率 {ratio:.0f}%")
        level_col = "综合级别"
        if level_col in df.columns:
            dist = df[level_col].value_counts()
            for level in ["A", "B", "C", "D"]:
                if level not in dist.index:
                    warnings.append(f"  '{level}' 级别无股票")
        score_col = "综合分析评分"
        if score_col in df.columns:
            scores = pd.to_numeric(df[score_col], errors="coerce")
            if scores.nunique() <= 1:
                warnings.append(f"  '{score_col}' 所有值相同 (均分={scores.mean():.1f})")
        if warnings:
            self.logger.warning(f"[完整性断言] 发现 {len(warnings)} 个异常:\n" + "\n".join(warnings))
        else:
            self.logger.info("[完整性断言] 数据完整性检查通过")

    def _step_13_sync_to_database(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        industry_df: pd.DataFrame = ctx.get("industry_df", pd.DataFrame())
        raw_data: dict = ctx.get("raw_data", {})
        self._sync_results_to_database(consolidated_report, industry_df, raw_data)
        return True

    # ──────────────────────────────────────────────
    # 辅助方法（可复用）
    # ──────────────────────────────────────────────

    def _load_research_report_data(self) -> pd.DataFrame:
        try:
            report_cache_path = self.stock_sync_engine.cache_manager.get_cache_path(
                "研报买入次数", cleaned=True
            )
            if os.path.exists(report_cache_path):
                report_df = pd.read_csv(
                    report_cache_path, sep="|", encoding="utf-8-sig", dtype={"股票代码": str}
                )
                self.logger.info(f"  - 已加载研报数据: {len(report_df)} 条记录")
                return report_df
            else:
                self.logger.warning("  - 研报数据缓存文件不存在")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"  - 加载研报数据失败: {e}")
            return pd.DataFrame()

    def _filter_by_universe(self, df: pd.DataFrame, universe_set: set) -> pd.DataFrame:
        from UtilsManager.CodeNormalizer import CodeNormalizer

        if df is None or df.empty or "股票代码" not in df.columns:
            return pd.DataFrame()

        df["股票代码"] = CodeNormalizer.normalize_series(df["股票代码"])
        return df[df["股票代码"].isin(universe_set)].copy()

    def _filter_raw_data(
        self, raw_data: dict[str, pd.DataFrame], universe_set: set
    ) -> dict[str, pd.DataFrame]:
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

    def _prepare_sheets_data(
        self,
        consolidated_report: pd.DataFrame,
        industry_df: pd.DataFrame,
        processed_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        return {
            "数据汇总": consolidated_report,
            "行业深度分析": industry_df,
            "主力研报筛选": processed_data.get("processed_main_report", pd.DataFrame()),
            "主力成本分析": processed_data.get("main_cost_data", pd.DataFrame()),
        }

    def _sync_results_to_database(
        self,
        consolidated_report: pd.DataFrame,
        industry_df: pd.DataFrame,
        raw_data: dict[str, pd.DataFrame],
    ) -> bool:
        try:
            success = self.report_service.sync_to_database(
                today_str=self.today_str,
                consolidated_report=consolidated_report,
                industry_df=industry_df,
                raw_data=raw_data,
            )

            if not success:
                self.logger.warning("数据库同步失败，但流程继续")
            return success

        except (DBAPIError, OperationalError, Exception) as e:
            self.logger.warning(f"!!! [同步中断] 数据库异常: {e}，跳过同步")
            return False


class StockAnalysisCoordinatorFactory:
    """
    股票分析协调器工厂类
    负责组装和初始化所有依赖项，创建 StockAnalysisCoordinator 实例。
    """

    @classmethod
    def create(
        cls,
        config_file: str = "config.ini",
    ) -> StockAnalysisCoordinator:
        from LogicAnalyzer.FundMomentumAnalyzer import FundMomentumAnalyzer
        from UtilsManager.LoggerManager import get_logger
        from UtilsManager.UnifiedCacheManager import CacheStrategy

        config = Config(config_file=config_file)

        try:
            from UtilsManager.ConfigValidator import validate_and_repair
            validate_and_repair(config_file)
        except (FileNotFoundError, PermissionError):
            pass

        calendar_mgr = TradingCalendarAnalyzer()
        today_str = calendar_mgr.get_last_trading_day()

        logger = get_logger(
            log_dir=config.LOG_DIR,
            log_filename=f"Corenews_Main_{today_str}.log",
            level=config.LOG_LEVEL,
        )

        cache_dir = os.path.join(config.TEMP_DATA_DIRECTORY, "cache")
        cache_manager = UnifiedCacheManager(
            cache_dir=cache_dir, default_strategy=CacheStrategy.DAILY, auto_cleanup=True
        )

        executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

        try:
            stock_sync_engine = StockSyncEngine(executor=executor)
            db_engine = stock_sync_engine.db

            from DataCollection.GetStockBasicinfo import StockBasicInfoService
            basic_info_service = StockBasicInfoService(config)
            basic_info_service.sync_all_stock_basic_info()

        except (DBAPIError, OperationalError) as e:
            raise DatabaseConnectionError(f"初始化数据库引擎失败: {e}") from e

        data_acquisition = DataAcquisitionService(config, calendar_mgr, logger, cache_manager, executor=executor)
        fund_momentum_analyzer = FundMomentumAnalyzer()
        data_processing = DataProcessingService(config, logger, fund_momentum_analyzer, calendar_mgr)
        analysis_service = AnalysisService(config, logger, db_engine, executor=executor)
        report_service = ReportService(config, logger)

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
