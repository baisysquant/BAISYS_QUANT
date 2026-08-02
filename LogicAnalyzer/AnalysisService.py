"""
业务分析服务类

负责技术指标信号处理、行业趋势分析、弱势股剔除等业务逻辑。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from DataManager.ColumnNames import ColumnNames
from DataManager.DataMergeService import get_stock_industry_mapping as _get_mapping
from DataManager.SignalDataLoader import SignalDataLoader
from LogicAnalyzer.Industrytrending import IndustryFlowAnalyzer
from LogicAnalyzer.SignalManager import TASignalProcessor


class AnalysisService:
    """
    业务分析服务

    职责：
    - 技术指标信号处理
    - 行业趋势分析
    - 弱势股剔除
    - 行业信号映射

    Attributes:
        config: 配置管理器实例
        logger: 日志管理器
        db_engine: 数据库引擎
    """

    def __init__(self, config: Any, logger: Any, db_engine: Any, executor: ThreadPoolExecutor | None = None, today_str: str | None = None) -> None:  # noqa: ANN401
        self.config = config
        self.logger = logger
        self.db_engine = db_engine
        self.executor = executor
        self.today_str = today_str

    def process_technical_signals(
        self, stock_codes: list[str], hist_df: pd.DataFrame, spot_data: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        处理技术指标信号

        Args:
            stock_codes: 股票代码列表（带前缀格式）
            hist_df: K线历史数据
            spot_data: 实时价格数据

        Returns:
            Dict[str, pd.DataFrame]: 包含各种技术指标信号的字典
        """
        self.logger.info(">>> 正在处理技术指标信号...")

        chip_lookup, moneyflow_lookup, forecast_lookup = SignalDataLoader.load_all(self.config, today_str=self.today_str)
        signal_processor = TASignalProcessor(None, config=self.config, executor=self.executor)
        ta_signals = signal_processor.process_signals(
            stock_codes, hist_df, spot_data,
            chip_lookup=chip_lookup,
            moneyflow_lookup=moneyflow_lookup,
            forecast_lookup=forecast_lookup,
        )

        self.logger.info(">>> 股票历史数据和技术指标分析完成。")

        return ta_signals

    def run_industry_analysis(self) -> pd.DataFrame:
        """
        运行行业深度分析

        Returns:
            pd.DataFrame: 行业分析结果
        """
        self.logger.info(">>> 正在执行行业深度分析...")
        print("  行业深度分析: 加载行业数据...", end="", flush=True)

        industry_analyzer = IndustryFlowAnalyzer(self.config, today_str=self.today_str)
        industry_analysis_df = industry_analyzer.run_analysis()

        status = "✓" if not industry_analysis_df.empty else "✗ (空)"
        print(f" {status} {len(industry_analysis_df)} 个行业", flush=True)
        self.logger.info(f">>> 行业分析完成，共 {len(industry_analysis_df)} 个行业")

        return industry_analysis_df

    def filter_weak_stocks(self, consolidated_report: pd.DataFrame) -> pd.DataFrame:
        """
        多因子弱信号过滤 v2 — 行业截面百分位 + 三级过滤。

        流程:
          Stage 0: 确保行业百分位列存在（由 fuse_scores 预计算）
          Stage 1: 豁免通道（强趋势 / 单因子前 N%）→ 保留
          Stage 2: 多因子评分硬地板（行业内后 N%）→ 剔除
          Stage 3: MACD 结论辅助剔除（D/C 级 + 低评分）→ 剔除

        Args:
            consolidated_report: 汇总报告DataFrame

        Returns:
            pd.DataFrame: 过滤后的DataFrame
        """
        self.logger.info(">>> 正在执行最终数据清洗：剔除弱信号个股...")

        if consolidated_report.empty:
            return consolidated_report

        # ── 配置读取 ──
        pct_hard = self.config.FILTER_PCT_HARD           # 10%
        pct_d = self.config.FILTER_PCT_D                # 30%
        pct_exempt = self.config.FILTER_PCT_EXEMPT       # 80%
        exempt_levels = self.config.EXEMPT_LEVELS

        # ── Stage 0: 确保百分位列存在 ──
        score_pct = consolidated_report.get(ColumnNames.SCORE_PCT_INDUSTRY, pd.Series(50.0, index=consolidated_report.index))
        momentum_pct = consolidated_report.get(ColumnNames.MOMENTUM_PCT_INDUSTRY, pd.Series(50.0, index=consolidated_report.index))
        quality_pct = consolidated_report.get(ColumnNames.QUALITY_PCT_INDUSTRY, pd.Series(50.0, index=consolidated_report.index))
        valuation_pct = consolidated_report.get(ColumnNames.VALUATION_PCT_INDUSTRY, pd.Series(50.0, index=consolidated_report.index))
        conclusion = consolidated_report.get(ColumnNames.COMPREHENSIVE_ANALYSIS, pd.Series([""] * len(consolidated_report), index=consolidated_report.index)).astype(str)
        bull_trend = consolidated_report.get(ColumnNames.BULL_TREND, pd.Series(dtype=str)).astype(str)

        # ── Stage 1: 豁免通道（满足任一即保留） ──
        is_exempt = (
            bull_trend.isin(exempt_levels)
            | (momentum_pct > pct_exempt)
            | (quality_pct > pct_exempt)
            | (valuation_pct > pct_exempt)
        )

        # ── Stage 2: 多因子评分硬地板（行业内评分后 N%） ──
        is_score_weak = score_pct < pct_hard

        # ── Stage 3: MACD 结论辅助剔除 ──
        is_d_weak = conclusion.str.startswith("D:")  # 所有 D 级 → 无条件剔除（绕过豁免）
        is_c_weak = conclusion.str.strip() == "C: 无明确入场信号"

        # ── 综合决策 ──
        # D 级 / C:无明确入场信号 无条件剔除（绕过豁免）；评分硬地板受豁免保护
        drop_condition = is_d_weak | is_c_weak | ((~is_exempt) & is_score_weak)

        n_before = len(consolidated_report)
        consolidated_report = consolidated_report[~drop_condition].copy()
        n_removed = n_before - len(consolidated_report)
        self.logger.info(f"  剔除 {n_removed} 只弱信号个股，剩余 {len(consolidated_report)} 只。")

        return consolidated_report

    def merge_industry_signal_to_stocks(self, stock_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        将行业分析的结论('行业信号'列)，精准匹配到每一只股票上。

        Args:
            stock_df: 股票数据DataFrame
            industry_df: 行业分析结果DataFrame

        Returns:
            pd.DataFrame: 添加了行业信号的DataFrame
        """
        if industry_df.empty or stock_df.empty or ColumnNames.INDUSTRY not in stock_df.columns:
            stock_df[ColumnNames.INDUSTRY_SIGNAL] = ""
            return stock_df

        required_cols = {"行业名称", "行业信号"}
        if not required_cols.issubset(industry_df.columns):
            self.logger.warning(f"行业分析结果缺少必要列: {required_cols - set(industry_df.columns)}")
            stock_df[ColumnNames.INDUSTRY_SIGNAL] = ""
            return stock_df

        self.logger.info("  - 正在将行业信号映射至个股...")

        industry_signal_df = industry_df[["行业名称", "行业信号"]].copy()
        industry_signal_df["行业名称"] = industry_signal_df["行业名称"].fillna("").astype(str).str.strip()
        industry_signal_df["行业信号"] = industry_signal_df["行业信号"].fillna("").astype(str).str.strip()
        industry_signal_df = industry_signal_df.drop_duplicates(subset=["行业名称"], keep="first")

        signal_map = industry_signal_df.set_index("行业名称")["行业信号"].to_dict()
        stock_df[ColumnNames.INDUSTRY_SIGNAL] = (
            stock_df[ColumnNames.INDUSTRY].fillna("").astype(str).str.strip().map(signal_map).fillna("")
        )

        return stock_df

    def get_stock_industry_mapping(self, stock_codes: list[str]) -> pd.DataFrame:
        return _get_mapping(stock_codes, self.logger, engine=self.db_engine)

    def process_xstp_and_filter(self, raw_data: dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        处理并合并均线突破数据，并进行多头排列筛选。

        Args:
            raw_data: 原始数据字典
            spot_df: 实时价格数据

        Returns:
            pd.DataFrame: 处理后的均线突破数据
        """
        self.logger.info("正在处理并合并均线突破数据...")

        # 1. 清洗均线数据
        processed_df10 = raw_data.get("xstp_10_raw", pd.DataFrame()).rename(columns={ColumnNames.LATEST_PRICE: ColumnNames.MA10_PRICE})
        processed_df30 = raw_data.get("xstp_30_raw", pd.DataFrame()).rename(columns={ColumnNames.LATEST_PRICE: ColumnNames.MA30_PRICE})
        processed_df60 = raw_data.get("xstp_60_raw", pd.DataFrame()).rename(columns={ColumnNames.LATEST_PRICE: ColumnNames.MA60_PRICE})

        # 2. 合并
        merged_df = pd.concat(
            [
                processed_df10[[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME]].dropna(subset=[ColumnNames.STOCK_CODE]),
                processed_df30[[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME]].dropna(subset=[ColumnNames.STOCK_CODE]),
                processed_df60[[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME]].dropna(subset=[ColumnNames.STOCK_CODE]),
            ]
        ).drop_duplicates(subset=[ColumnNames.STOCK_CODE])

        # 3. 重新合并均线价格，确保同一行有所有数据
        xstp_base = merged_df[[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME]].drop_duplicates()
        xstp_base = pd.merge(
            xstp_base,
            processed_df10[[ColumnNames.STOCK_CODE, ColumnNames.MA10_PRICE]],
            on=ColumnNames.STOCK_CODE,
            how="left",
        )
        xstp_base = pd.merge(
            xstp_base,
            processed_df30[[ColumnNames.STOCK_CODE, ColumnNames.MA30_PRICE]],
            on=ColumnNames.STOCK_CODE,
            how="left",
        )
        xstp_base = pd.merge(
            xstp_base,
            processed_df60[[ColumnNames.STOCK_CODE, ColumnNames.MA60_PRICE]],
            on=ColumnNames.STOCK_CODE,
            how="left",
        )

        # 4. 合并实时价格
        xstp_base = pd.merge(xstp_base, spot_df[[ColumnNames.STOCK_CODE, ColumnNames.LATEST_PRICE]], on=ColumnNames.STOCK_CODE, how="left")

        # 5. 类型转换和过滤
        cols_to_convert = [col for col in xstp_base.columns if ColumnNames.LATEST_PRICE in col or col == ColumnNames.LATEST_PRICE]
        for col in cols_to_convert:
            xstp_base[col] = pd.to_numeric(xstp_base[col], errors="coerce")

        # 过滤条件: 1. 最新价>10日均线 2. 多头排列 (10>30 或 30>60)
        filtered_df = xstp_base[
            (xstp_base[ColumnNames.LATEST_PRICE] > xstp_base[ColumnNames.MA10_PRICE])
            & (
                (xstp_base[ColumnNames.MA10_PRICE] > xstp_base[ColumnNames.MA30_PRICE].fillna(float("-inf")))
                | (xstp_base[ColumnNames.MA30_PRICE] > xstp_base[ColumnNames.MA60_PRICE].fillna(float("-inf")))
            )
        ].copy()

        filtered_df.rename(columns={ColumnNames.LATEST_PRICE: ColumnNames.CURRENT_PRICE}, inplace=True)
        return filtered_df.fillna("N/A")
