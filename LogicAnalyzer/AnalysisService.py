"""
业务分析服务类

负责技术指标信号处理、行业趋势分析、弱势股剔除等业务逻辑。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from DataManager.ColumnNames import ColumnNames


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
        from DataManager.SignalDataLoader import SignalDataLoader
        from LogicAnalyzer.SignalManager import TASignalProcessor

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
        from LogicAnalyzer.Industrytrending import IndustryFlowAnalyzer

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
        剔除弱势且加速下跌的个股

        筛选条件：
        - 非强势股
        - 无量价齐升
        - 无连涨/放量
        - MACD双周期加速下跌
        - KDJ无信号
        - 资金流为负
        - 非豁免级别

        Args:
            consolidated_report: 汇总报告DataFrame

        Returns:
            pd.DataFrame: 过滤后的DataFrame
        """
        self.logger.info(">>> 正在执行最终数据清洗：剔除弱势且加速下跌的个股...")

        if consolidated_report.empty:
            return consolidated_report

        # 为了安全比较，确保 DIF 列被正确解析为数字，非数字转为 NaN
        dif_single = pd.to_numeric(consolidated_report.get("_current_dif"), errors="coerce")

        kdj_col = consolidated_report.get(
            ColumnNames.KDJ_SIGNAL,
            pd.Series([""] * len(consolidated_report), index=consolidated_report.index),
        )
        kdj_is_empty = kdj_col.isna() | (kdj_col.astype(str).str.strip().str.lower().isin(["", "nan", "none"]))

        full_bull_level = consolidated_report.get(ColumnNames.BULL_TREND, pd.Series(dtype=str))
        # 使用配置中的豁免条件
        exempt_from_drop = full_bull_level.isin(self.config.EXEMPT_LEVELS)

        drop_condition = (
            (consolidated_report.get(ColumnNames.STRONG_STOCK) == "否")
            & (consolidated_report.get(ColumnNames.PRICE_VOLUME_RISE) == "否")
            & (consolidated_report.get(ColumnNames.CONSECUTIVE_RISE_DAYS) == 0)
            & (consolidated_report.get(ColumnNames.VOLUME_INCREASE_DAYS) == 0)
            & (dif_single < 0)
            & kdj_is_empty
            & (
                # 使用配置的第一个资金流周期进行检查
                consolidated_report.get(self._get_first_fund_flow_col(), pd.Series(dtype=str))
                .astype(str)
                .str.contains("-", na=False)
            )
            & (~exempt_from_drop)  # 使用豁免条件
        )

        consolidated_report = consolidated_report[~drop_condition].copy()
        self.logger.info(f"  排除极度弱势特征的股票。剩余 {len(consolidated_report)} 只。")

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
        from DataManager.DataMergeService import get_stock_industry_mapping as _get_mapping
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

    def _get_first_fund_flow_col(self) -> str:
        """
        获取配置的第一个资金流列名

        Returns:
            str: 资金流列名
        """
        period_map = {
            3: ColumnNames.FUND_FLOW_3D,
            5: ColumnNames.FUND_FLOW_5D,
            10: ColumnNames.FUND_FLOW_10D,
            20: ColumnNames.FUND_FLOW_20D,
        }

        if self.config.FUND_FLOW_PERIODS:
            first_period = self.config.FUND_FLOW_PERIODS[0]
            return period_map.get(first_period, ColumnNames.FUND_FLOW_5D)

        return ColumnNames.FUND_FLOW_5D
