"""
股票分析协调器

负责编排整个股票分析流程，协调各个服务类的工作。

Pipeline 设计：
  - 每个分析步骤封装为独立的 _step_N_xxx 方法
  - 步骤间通过 PipelineContext 传递数据，互不直接依赖
  - 每步独立 try/except，单步失败不会导致整个流程崩溃
  - 可单独构造 PipelineContext 调用任一步骤进行单元测试
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, TypedDict

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError

from UtilsManager.ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from DataManager.DataProcessingService import DataProcessingService
from DataManager.IncrementalSyncEngine import IncrementalSyncEngine
from DataManager.DataQualityChecker import DataQualityChecker
from Review.report import ReportService
from LogicAnalyzer.AnalysisService import AnalysisService
from LogicAnalyzer.portfolio.benchmark import BenchmarkEvaluator
from DataManager.DataAcquisitionService import DataAcquisitionService
from LogicAnalyzer.scoring.calculator import FactorCalculator
from LogicAnalyzer.scoring.decay import FactorDecayMonitor
from LogicAnalyzer.portfolio.builder import PortfolioBuilder
from LogicAnalyzer.portfolio.tracking import PositionTrackingService, SHEET_NAME as POSITION_BT_SHEET
from LogicAnalyzer.pipeline.dag import DagPipeline, PipelineStep
from UtilsManager.CodeNormalizer import CodeNormalizer
from UtilsManager.Exceptions import DatabaseConnectionError
from UtilsManager.IDataProvider import IDataProvider
from UtilsManager.UnifiedCacheManager import UnifiedCacheManager


class PipelineData(TypedDict, total=False):
    filtered_pure_codes: set[str]
    stock_codes_prefixed: list[str]
    stock_codes_pure: list[str]
    raw_data: dict[str, Any]
    hist_df: Any
    spot_data: Any
    ta_signals: dict[str, Any]
    industry_df: Any
    processed_xstp_df: Any
    processed_data: dict[str, Any]
    consolidated_report: Any


class PipelineContext:
    """流水线上下文：步骤之间通过此对象交换数据，解除顺序耦合"""

    def __init__(self) -> None:
        self.data: PipelineData = {}
        self.errors: dict[str, str] = {}

    def set(self, key: str, value: Any) -> None:  # noqa: ANN401
        self.data[key] = value  # type: ignore[literal-required]

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        return self.data.get(key, default)  # type: ignore[return-value]

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
        data_provider: K 线数据提供者（实时/回测可切换）
        data_acquisition: 实时数据获取服务（akshare）
        data_processing: 数据处理服务
        analysis_service: 业务分析服务
        report_service: 报告生成服务
    """

    def __init__(
        self,
        config: Config,
        calendar_mgr: TradingCalendarAnalyzer,
        logger: Any,  # noqa: ANN401
        cache_manager: UnifiedCacheManager,
        incremental_sync_engine: IncrementalSyncEngine,
        db_engine: Engine,
        executor: ThreadPoolExecutor,
        data_provider: IDataProvider,
        data_acquisition: DataAcquisitionService,
        data_processing: DataProcessingService,
        analysis_service: AnalysisService,
        report_service: ReportService,
        today_str: str | None = None,
    ) -> None:
        self.config = config
        self.calendar_mgr = calendar_mgr
        self.logger = logger
        self.cache_manager = cache_manager
        self.incremental_sync_engine = incremental_sync_engine
        self.db_engine = db_engine
        self.executor = executor
        self.data_provider = data_provider
        self.data_acquisition = data_acquisition
        self.data_processing = data_processing
        self.analysis_service = analysis_service
        self.report_service = report_service
        self.position_tracking_service = PositionTrackingService(
            config=config,
            logger=logger,
            data_provider=data_provider,
            calendar_mgr=calendar_mgr,
            db_engine=db_engine,
        )

        self.today_str = today_str or self.calendar_mgr.get_last_trading_day()

        self.factor_calculator = FactorCalculator(config=config, db_engine=db_engine)
        self.portfolio_builder = PortfolioBuilder(
            config=config, db_engine=db_engine, today_str=self.today_str
        )
        self.benchmark_evaluator = BenchmarkEvaluator(config=config)
        self.factor_decay_monitor = FactorDecayMonitor(config=config, db_engine=db_engine)
        self.quality_checker = DataQualityChecker(db_engine=db_engine)
        self.force_rerun = False
        self.start_time = time.time()

    # ──────────────────────────────────────────────
    # Pipeline 定义
    # ──────────────────────────────────────────────

    def _build_dag(self) -> DagPipeline:
        """构建 DAG 流水线，定义步骤及其依赖关系。"""
        dag = DagPipeline(
            name="stock_analysis",
            db_engine=self.db_engine,
            cache_dir=self.config.CACHE_DIRECTORY,
            config_path="config.ini",
        )

        # (步骤名, 方法, 依赖列表, 是否致命)
        steps: list[tuple[str, Callable, list[str], bool]] = [
            ("同步历史数据", self._step_1_sync_data, [], True),
            ("格式化股票代码", self._step_2_format_codes, ["同步历史数据"], True),
            ("获取原始数据", self._step_3_get_raw_data, ["格式化股票代码"], False),
            ("获取K线数据及最新价", self._step_4_get_kline_and_prices, ["格式化股票代码"], True),
            ("处理技术指标信号", self._step_5_technical_signals, ["获取K线数据及最新价"], False),
            ("运行行业分析", self._step_6_industry_analysis, [], False),
            ("处理均线突破数据", self._step_7_xstp_and_filter, ["获取原始数据", "获取K线数据及最新价"], False),
            ("准备处理数据字典", self._step_8_prepare_processed_data,
             ["处理技术指标信号", "运行行业分析", "处理均线突破数据"], False),
            ("合并处理数据", self._step_9_consolidate_data,
             ["准备处理数据字典", "格式化股票代码"], True),
            ("数据质量检查", self._step_9a_data_quality,
             ["合并处理数据"], False),
            ("映射行业信号", self._step_10_merge_industry_signal,
             ["合并处理数据", "运行行业分析"], False),
            ("多因子Alpha评分", self._step_11_multi_factor_alpha,
             ["映射行业信号", "获取K线数据及最新价"], False),
            ("剔除弱势股", self._step_12_filter_weak_stocks, ["多因子Alpha评分"], False),
            ("组合构建", self._step_13_build_portfolio, ["剔除弱势股"], False),
            ("基准对比", self._step_14_benchmark_compare,
             ["组合构建", "获取K线数据及最新价"], False),
            ("因子衰减监控", self._step_15_factor_decay,
             ["组合构建", "获取K线数据及最新价"], False),
            ("跟仓回测分析", self._step_16_position_backtest, [], False),
            ("生成Excel报告", self._step_17_generate_report,
             ["组合构建", "运行行业分析", "准备处理数据字典",
              "跟仓回测分析", "基准对比"], False),
            ("同步结果到数据库", self._step_18_sync_to_database,
             ["组合构建", "运行行业分析", "获取原始数据"], False),
        ]

        for name, fn, deps, fatal in steps:
            dag.add_step(PipelineStep(name=name, fn=fn, depends_on=deps, is_fatal=fatal))

        dag.register_intermediate("获取K线数据及最新价", "hist_df")
        dag.register_intermediate("处理均线突破数据", "processed_xstp_df")
        dag.register_intermediate("运行行业分析", "industry_df")

        return dag

    def run(self) -> None:
        """
        基于 DAG 执行流水线，支持断点续跑。
        每步独立 try/except，致命步骤失败终止流程。
        """
        self.logger.info(f"[INFO] 股票分析程序启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"[INFO] 最后一个交易日为: {self.today_str}")

        print(f"\n{'='*50}")
        print(f"  股票分析流水线 | 交易日: {self.today_str}")
        print(f"{'='*50}")

        ctx = PipelineContext()
        dag = self._build_dag()

        success = dag.run(ctx, trade_date=self.today_str, force_rerun=self.force_rerun)

        total_elapsed = timedelta(seconds=time.time() - self.start_time)
        if success:
            print(f"\n{'='*50}")
            print(f"  流水线完成 | 总耗时: {total_elapsed}")
            print(f"{'='*50}\n")
            self.logger.info(f"\n>>> 流程结束。总耗时: {total_elapsed}")
        else:
            print(f"\n{'='*50}")
            print(f"  流水线异常终止 | 总耗时: {total_elapsed}")
            print(f"{'='*50}\n")
            self.logger.warning(f"\n>>> 流程异常终止。总耗时: {total_elapsed}")

        self._shutdown()

    def _shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    # ──────────────────────────────────────────────
    # 各步骤实现（可独立测试）
    # ──────────────────────────────────────────────

    def _step_1_sync_data(self, ctx: PipelineContext) -> bool:
        self.logger.info(">>> 正在同步历史数据到数据库...")
        try:
            filtered_pure_codes = self.incremental_sync_engine.sync_stock_pool_and_kline(
                target_date=self.today_str
            )
            if not filtered_pure_codes:
                self.logger.critical("同步历史数据后无有效股票代码，流程终止")
                return False
            ctx.set("filtered_pure_codes", filtered_pure_codes)

            # 同步因子数据
            if self.config.MULTI_FACTOR_ALPHA_ENABLED:
                self._sync_factor_fetchers(filtered_pure_codes)

            return True
        except Exception as e:
            self.logger.error(f"同步失败: {e}")
            return False

    def _sync_factor_fetchers(self, stock_codes: set[str]) -> None:
        """同步估值/质量因子数据采集器。"""
        stock_list = sorted(stock_codes)

        # 估值因子（日频，AShareHub）
        try:
            from DataCollection.FinancialValuationFetcher import FinancialValuationFetcher
            val_fetcher = FinancialValuationFetcher(self.config)
            written = val_fetcher.sync_daily(trade_date=self.today_str)
            if written:
                self.logger.info(f"[因子数据] 估值因子同步完成，写入 {written} 条")
        except Exception as e:
            self.logger.warning(f"[因子数据] 估值因子同步失败: {e}")

        # 基准指数（日频，AShareHub）
        try:
            from DataCollection.BenchmarkFetcher import BenchmarkFetcher
            bm_fetcher = BenchmarkFetcher(self.config)
            bm_fetcher.sync_daily()
        except Exception as e:
            self.logger.warning(f"[因子数据] 基准指数同步失败: {e}")

        # 质量因子（季度，akShare，增量同步）
        try:
            from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher
            qual_fetcher = FinancialQualityFetcher(self.config)
            count = qual_fetcher.sync(stock_list, today_str=self.today_str)
            if count:
                self.logger.info(f"[因子数据] 质量因子同步完成，采集 {count} 只")
        except Exception as e:
            self.logger.warning(f"[因子数据] 质量因子同步失败: {e}")

    def _step_2_format_codes(self, ctx: PipelineContext) -> bool:
        filtered_pure_codes: set = ctx.get("filtered_pure_codes")
        stock_codes_prefixed = [CodeNormalizer.add_market_prefix(code) for code in sorted(filtered_pure_codes)]
        stock_codes_pure = sorted(filtered_pure_codes)
        ctx.set("stock_codes_prefixed", stock_codes_prefixed)
        ctx.set("stock_codes_pure", stock_codes_pure)
        print(f"  待分析股票: {len(stock_codes_pure)} 只", flush=True)
        self.logger.info(
            f">>> HistDataWatchDog 成功同步 {len(stock_codes_prefixed)} 只股票数据到数据库，并作为分析基础。"
        )
        return True

    def _step_3_get_raw_data(self, ctx: PipelineContext) -> bool:
        raw_data = self.data_acquisition.get_all_raw_data(self.today_str)
        if not raw_data:
            self.logger.warning("[获取原始数据] akshare 原始数据为空")
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

        try:
            hist_df_all = self.data_provider.get_kline(stock_codes_prefixed)
            if not hist_df_all.empty:
                self.logger.info(
                    f"[INFO] 数据日期范围: {hist_df_all['trade_date'].min()} 至 {hist_df_all['trade_date'].max()}"
                )
            else:
                self.logger.error("[ERROR] 查询结果为空！可能是股票代码不匹配或日期条件过滤了所有数据。")
        except Exception as e:
            self.logger.error(f"[ERROR] 数据库查询失败: {e}")
            hist_df_all = pd.DataFrame()

        if hist_df_all.empty:
            self.logger.warning("[WARN] 由于历史数据为空，将跳过所有技术指标计算。")

        # 从 hist_df_all 获取每只股票最新的 close_normal 作为 spot_data
        try:
            cn = hist_df_all[hist_df_all["close_normal"].notna()]
            if not cn.empty:
                last_cn = cn.sort_values("trade_date").groupby("symbol").last().reset_index()
                last_cn["股票代码"] = CodeNormalizer.normalize_series(last_cn["symbol"])
                latest_prices_df = last_cn[["股票代码", "close_normal"]].rename(
                    columns={"close_normal": "最新价"}
                )
                self.logger.info(f"[INFO] 从 DB 获取 {len(latest_prices_df)} 只股票的最新收盘价")
            else:
                latest_prices_df = pd.DataFrame(columns=["股票代码", "最新价"])
                self.logger.warning("[WARN] 数据库 close_normal 为空，最新价留空")
        except Exception as e:
            self.logger.warning(f"[WARN] 构建最新价失败: {e}")
            latest_prices_df = pd.DataFrame(columns=["股票代码", "最新价"])

        ctx.set("hist_df", hist_df_all)
        ctx.set("spot_data", latest_prices_df)
        return True

    def _step_5_technical_signals(self, ctx: PipelineContext) -> bool:
        if not ctx.has("stock_codes_prefixed"):
            self.logger.warning("[SKIP] 技术指标信号缺少前置依赖")
            return False

        stock_codes_prefixed: list[str] = ctx.get("stock_codes_prefixed")
        hist_df = self._ensure_hist_df(ctx)
        spot_data: pd.DataFrame = ctx.get("spot_data", pd.DataFrame())

        if hist_df.empty:
            self.logger.warning("[SKIP] K线数据为空，跳过技术指标计算")
            return False

        ta_signals = self.analysis_service.process_technical_signals(stock_codes_prefixed, hist_df, spot_data)
        self.report_service.save_ta_signals_to_txt(ta_signals, self.today_str)

        self.logger.info("=== 技术指标数据检查 ===")
        for key, df in ta_signals.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                self.logger.info(f"{key}: {len(df)} 条数据，列名: {list(df.columns)}")
                self.logger.info(f"  样本数据:\n{df.head(2)}")
            else:
                self.logger.info(f"{key}: 空DataFrame")

        ctx.set("ta_signals", ta_signals)
        return True

    def _step_6_industry_analysis(self, ctx: PipelineContext) -> bool:
        industry_df = self.analysis_service.run_industry_analysis()
        ctx.set("industry_df", industry_df)
        return True

    def _step_7_xstp_and_filter(self, ctx: PipelineContext) -> bool:
        raw_data: dict = ctx.get("raw_data", {})
        _ = self._ensure_hist_df(ctx)
        spot_data: pd.DataFrame = ctx.get("spot_data", pd.DataFrame())
        if spot_data.empty:
            hist = ctx.get("hist_df", pd.DataFrame())
            if not hist.empty:
                try:
                    cn = hist[hist["close_normal"].notna()]
                    if not cn.empty:
                        last_cn = cn.sort_values("trade_date").groupby("symbol").last().reset_index()
                        last_cn["股票代码"] = CodeNormalizer.normalize_series(last_cn["symbol"])
                        spot_data = last_cn[["股票代码", "close_normal"]].rename(
                            columns={"close_normal": "最新价"}
                        )
                        ctx.set("spot_data", spot_data)
                        self.logger.info(f"[spot_data] 从 hist_df 重建，{len(spot_data)} 条")
                except Exception:
                    pass
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

    def _step_9a_data_quality(self, ctx: PipelineContext) -> bool:
        """数据质量检查：在合并数据后执行，提前发现异常。"""
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        if consolidated_report.empty:
            self.logger.info("[数据质量] consolidated_report 为空，跳过检查")
            return True
        try:
            result = self.quality_checker.run_all(consolidated_report, self.today_str)
            if not result["all_pass"]:
                self.logger.warning("[数据质量] 部分检查未通过，但流程继续")
            else:
                self.logger.info("[数据质量] 全部检查通过")
        except Exception as e:
            self.logger.warning(f"[数据质量] 执行异常: {e}")
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

    def _step_11_multi_factor_alpha(self, ctx: PipelineContext) -> bool:
        if not self.config.MULTI_FACTOR_ALPHA_ENABLED:
            self.logger.info("[多因子Alpha] 未启用，跳过。")
            return True

        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        hist_df = self._ensure_hist_df(ctx)

        if consolidated_report.empty:
            self.logger.warning("[多因子Alpha] consolidated_report 为空，跳过。")
            return False

        symbols = list(consolidated_report["股票代码"].unique())

        quality_df = self.factor_calculator.load_quality_from_db(symbols)
        valuation_df = self.factor_calculator.load_valuation_from_db(symbols)

        if quality_df.empty and valuation_df.empty:
            self.logger.info("[多因子Alpha] 无外部因子数据，跳过。")
            return True

        try:
            updated = self.factor_calculator.fuse_scores(
                report=consolidated_report,
                macd_score_col="综合分析评分",
                industry_col="行业",
                hist_df=hist_df,
                quality_df=quality_df,
                valuation_df=valuation_df,
                trade_date=self.today_str,
            )
            ctx.set("consolidated_report", updated)
            self.logger.info(f"[多因子Alpha] 评分融合完成，结果 {len(updated)} 行。")
            return True
        except Exception as e:
            self.logger.opt(exception=True).warning(f"[多因子Alpha] 评分融合失败: {e}")
            return False

    def _step_12_filter_weak_stocks(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        if consolidated_report.empty:
            return False
        consolidated_report = self.analysis_service.filter_weak_stocks(consolidated_report)
        ctx.set("consolidated_report", consolidated_report)
        return True

    def _step_13_build_portfolio(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        if consolidated_report.empty:
            self.logger.warning("[组合构建] consolidated_report 为空，跳过。")
            return False
        hist_df = self._ensure_hist_df(ctx)
        updated = self.portfolio_builder.build(consolidated_report, hist_df=hist_df)
        ctx.set("consolidated_report", updated)
        return True

    def _ensure_hist_df(self, ctx: PipelineContext) -> pd.DataFrame:
        """确保 hist_df 可用，尝试从缓存/DB 恢复。"""
        hist_df: pd.DataFrame = ctx.get("hist_df", pd.DataFrame())
        if not hist_df.empty:
            if "trade_date" not in hist_df.columns and "date" in hist_df.columns:
                hist_df = hist_df.rename(columns={"date": "trade_date"})
                ctx.set("hist_df", hist_df)
            return hist_df
        stock_codes = ctx.get("stock_codes_prefixed", [])
        if stock_codes:
            self.logger.info("[hist_df] hist_df 缺失，重新从 DB 加载")
            hist_df = self.data_provider.get_kline(stock_codes)
            if not hist_df.empty and "trade_date" not in hist_df.columns and "date" in hist_df.columns:
                hist_df = hist_df.rename(columns={"date": "trade_date"})
            ctx.set("hist_df", hist_df)
        return hist_df

    def _step_14_benchmark_compare(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        hist_df = self._ensure_hist_df(ctx)

        if consolidated_report.empty or hist_df.empty:
            self.logger.warning("[基准对比] 数据不足，跳过。")
            return True

        self.logger.info(f"[基准对比] hist_df columns: {list(hist_df.columns)}, shape={hist_df.shape}")
        self.logger.info(f"[基准对比] consolidated_report columns: {list(consolidated_report.columns)}, shape={consolidated_report.shape}")

        # 用 PortfolioBuilder 输出的目标权重估算组合历史收益率
        try:
            portfolio_rets = self.benchmark_evaluator.estimate_portfolio_returns(
                portfolio_df=consolidated_report,
                kline_df=hist_df,
            )
        except Exception as e:
            self.logger.opt(exception=True).warning(f"[基准对比] 估算组合收益率失败: {e}")
            return True
        if portfolio_rets.empty:
            self.logger.info("[基准对比] 无法估算组合收益率，跳过。")
            return True

        # 从 DB 加载基准指数数据
        try:
            from DataCollection.BenchmarkFetcher import BenchmarkFetcher
            bm_fetcher = BenchmarkFetcher(self.config)
            bm_df = bm_fetcher.load_index_data()
            if bm_df.empty:
                self.logger.info("[基准对比] 基准指数数据为空，跳过。")
                return True
        except Exception as e:
            self.logger.warning(f"[基准对比] 加载基准数据失败: {e}")
            return True

        # 基准日收益率
        bm_df = bm_df.sort_values("trade_date")
        bm_df["daily_return"] = bm_df["close"].pct_change()

        # 执行对比
        result = self.benchmark_evaluator.evaluate(
            portfolio_returns=portfolio_rets,
            benchmark_df=bm_df,
        )
        ctx.set("benchmark_result", result)

        if "error" not in result:
            summary = pd.DataFrame([{
                "指标": k, "值": v
            } for k, v in result.items() if k != "日收益率数据"])
            ctx.set("benchmark_report", summary)

        self.logger.info("[基准对比] 完成。")
        return True

    def _step_15_factor_decay(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        hist_df = self._ensure_hist_df(ctx)
        if consolidated_report.empty or hist_df.empty:
            self.logger.info("[因子衰减] 数据不足，跳过。")
            return True
        try:
            decay_result = self.factor_decay_monitor.run(consolidated_report, hist_df)
            ctx.set("factor_decay_result", decay_result)
            if decay_result.get("needs_rebalance"):
                self.logger.warning("[因子衰减] 检测到因子衰减，建议重新平衡权重")
        except Exception as e:
            self.logger.warning(f"[因子衰减] 执行异常: {e}")
        return True

    def _step_16_position_backtest(self, ctx: PipelineContext) -> bool:
        df = self.position_tracking_service.run()
        ctx.set("position_backtest_report", df)
        if df.empty:
            self.logger.info("[跟仓回测] 无输出数据，跳过")
            return True
        self.logger.info(f"[跟仓回测] 生成 {len(df)} 条记录，等待写入 Excel")
        return True

    def _step_17_generate_report(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        industry_df: pd.DataFrame = ctx.get("industry_df", pd.DataFrame())
        processed_data: dict = ctx.get("processed_data", {})

        # 裁剪仅保留 final column order 中的列（step 10 可能加入了计算用列）
        from DataManager.ColumnNames import ColumnNames as CN
        from Review.report import ReportService
        final_cols = ReportService.get_final_column_order(
            fund_flow_periods=self.config.FUND_FLOW_PERIODS
        )
        existing_cols = [c for c in final_cols if c in consolidated_report.columns]
        # 明确剔除计算用列（即使因命名不一致混入）
        drop_cols = {
            CN.INDUSTRY_PERCENTILE, CN.INDUSTRY_SIGNAL_SCORE, CN.INDUSTRY_DEVIATION,
            "roe", "gross_profit_margin", "net_profit_margin",
            "pe_ttm", "pb", "total_mv", "circ_mv",
        }
        consolidated_report = consolidated_report[[c for c in existing_cols if c not in drop_cols]]

        position_backtest_report: pd.DataFrame = ctx.get("position_backtest_report", pd.DataFrame())
        benchmark_report: pd.DataFrame = ctx.get("benchmark_report", pd.DataFrame())
        sheets_data = self._prepare_sheets_data(
            consolidated_report, industry_df, processed_data,
            position_backtest_report=position_backtest_report,
            benchmark_report=benchmark_report,
        )
        self.report_service.generate_excel_report(sheets_data, self.today_str)
        self._validate_report_integrity(consolidated_report)
        return True

    def _validate_report_integrity(self, df: pd.DataFrame) -> None:
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

    def _step_18_sync_to_database(self, ctx: PipelineContext) -> bool:
        consolidated_report: pd.DataFrame = ctx.get("consolidated_report", pd.DataFrame())
        industry_df: pd.DataFrame = ctx.get("industry_df", pd.DataFrame())
        raw_data: dict = ctx.get("raw_data", {})
        self._sync_results_to_database(consolidated_report, industry_df, raw_data)
        return True

    # ──────────────────────────────────────────────
    # 辅助方法（可复用）
    # ──────────────────────────────────────────────

    def _load_research_report_data(self) -> pd.DataFrame:
        cache_path = os.path.join(self.config.CACHE_DIRECTORY, f"研报买入次数_经清洗_{self.today_str.replace('-', '')}.csv")
        try:
            if os.path.exists(cache_path):
                report_df = pd.read_csv(
                    cache_path, sep="|", encoding="utf-8-sig", dtype={"股票代码": str}
                )
                self.logger.info(f"  - 已加载研报数据: {len(report_df)} 条记录")
                return report_df
        except Exception as e:
            self.logger.warning(f"  - 缓存文件读取失败({e})，将重新拉取")

        self.logger.info("  - 从 akshare 拉取研报数据...")
        try:
            import akshare as ak
            raw = ak.stock_profit_forecast_em()
            if raw is not None and not raw.empty:
                df = raw.copy()
                if "代码" in df.columns and "股票代码" not in df.columns:
                    df.rename(columns={"代码": "股票代码"}, inplace=True)
                    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
                report_col = next((c for c in df.columns if "买入" in c), None)
                if report_col:
                    df.rename(columns={report_col: "研报买入次数"}, inplace=True)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path, sep="|", index=False, encoding="utf-8-sig")
                self.logger.info(f"  - 研报数据已缓存: {len(df)} 条")
                return df
        except Exception as e:
            self.logger.error(f"  - 拉取研报数据失败: {e}")
        return pd.DataFrame()

    def _filter_by_universe(self, df: pd.DataFrame, universe_set: set) -> pd.DataFrame:
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
        position_backtest_report: pd.DataFrame | None = None,
        benchmark_report: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        sheets = {
            "数据汇总": consolidated_report,
            "行业深度分析": industry_df,
            "主力研报筛选": processed_data.get("processed_main_report", pd.DataFrame()),
            "主力成本分析": processed_data.get("main_cost_data", pd.DataFrame()),
        }
        if position_backtest_report is not None and not position_backtest_report.empty:
            sheets[POSITION_BT_SHEET] = position_backtest_report
        if benchmark_report is not None and not benchmark_report.empty:
            sheets["基准对比"] = benchmark_report
        return sheets

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
        force_rerun: bool = False,
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

        cache_dir = os.path.join(config.CACHE_DIRECTORY, "unified_cache")
        cache_manager = UnifiedCacheManager(
            cache_dir=cache_dir, default_strategy=CacheStrategy.DAILY, auto_cleanup=True
        )

        executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

        try:
            from DataManager.DbEngine import get_engine as _get_engine

            db_engine = _get_engine(config)
            incremental_sync_engine = IncrementalSyncEngine(
                db_engine,
                default_start=config.BACKTEST_START_DATE,
                main_board_only=config.MAIN_BOARD_ONLY,
                enable_research_report_filter=config.ENABLE_RESEARCH_REPORT_FILTER,
                research_report_min_count=config.RESEARCH_REPORT_MIN_COUNT,
            )

            from DataCollection.GetStockBasicinfo import StockBasicInfoService
            basic_info_service = StockBasicInfoService(config)
            basic_info_service.sync_all_stock_basic_info()

        except (DBAPIError, OperationalError) as e:
            raise DatabaseConnectionError(f"初始化数据库引擎失败: {e}") from e

        from UtilsManager.IDataProvider import LiveDataProvider

        # 确保 stock_daily_kline 有 adj_factor 列
        from DataManager.sync import ensure_table
        ensure_table(db_engine)

        data_provider = LiveDataProvider(db_engine=db_engine)
        data_acquisition = DataAcquisitionService(config, calendar_mgr, logger, cache_manager, executor=executor)
        fund_momentum_analyzer = FundMomentumAnalyzer()
        data_processing = DataProcessingService(config, logger, fund_momentum_analyzer, calendar_mgr)
        analysis_service = AnalysisService(config, logger, db_engine, executor=executor, today_str=today_str)
        report_service = ReportService(config, logger)

        coordinator = StockAnalysisCoordinator(
            config=config,
            calendar_mgr=calendar_mgr,
            logger=logger,
            cache_manager=cache_manager,
            incremental_sync_engine=incremental_sync_engine,
            db_engine=db_engine,
            executor=executor,
            data_provider=data_provider,
            data_acquisition=data_acquisition,
            data_processing=data_processing,
            analysis_service=analysis_service,
            report_service=report_service,
            today_str=today_str,
        )
        coordinator.force_rerun = force_rerun
        return coordinator
