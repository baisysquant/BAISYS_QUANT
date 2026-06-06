"""
业务分析服务类

负责技术指标信号处理、行业趋势分析、弱势股剔除等业务逻辑。
"""

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

    def __init__(self, config, logger, db_engine, executor=None):
        self.config = config
        self.logger = logger
        self.db_engine = db_engine
        self.executor = executor

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
        from LogicAnalyzer.SignalManager import TASignalProcessor

        self.logger.info(">>> 正在处理技术指标信号...")

        signal_processor = TASignalProcessor(None, config=self.config, executor=self.executor)
        ta_signals = signal_processor.process_signals(stock_codes, hist_df, spot_data)

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

        industry_analyzer = IndustryFlowAnalyzer(self.config)
        industry_analysis_df = industry_analyzer.run_analysis()

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
            "KDJ_Signal",
            pd.Series([""] * len(consolidated_report), index=consolidated_report.index),
        )
        kdj_is_empty = kdj_col.isna() | (kdj_col.astype(str).str.strip().str.lower().isin(["", "nan", "none"]))

        full_bull_level = consolidated_report.get(ColumnNames.BULL_TREND, pd.Series(dtype=str))
        # 使用配置中的豁免条件
        exempt_from_drop = full_bull_level.isin(self.config.EXEMPT_LEVELS)

        drop_condition = (
            (consolidated_report.get("强势股") == "否")
            & (consolidated_report.get("量价齐升") == "否")
            & (consolidated_report.get("连涨天数") == 0)
            & (consolidated_report.get("放量天数") == 0)
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

        initial_count = len(consolidated_report)
        consolidated_report = consolidated_report[~drop_condition].copy()
        dropped_count = initial_count - len(consolidated_report)
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
        if industry_df.empty or stock_df.empty or "行业" not in stock_df.columns:
            stock_df["所属行业信号"] = ""
            return stock_df

        required_cols = {"行业名称", "行业信号"}
        if not required_cols.issubset(industry_df.columns):
            self.logger.warning(f"行业分析结果缺少必要列: {required_cols - set(industry_df.columns)}")
            stock_df["所属行业信号"] = ""
            return stock_df

        self.logger.info("  - 正在将行业信号映射至个股...")

        industry_signal_df = industry_df[["行业名称", "行业信号"]].copy()
        industry_signal_df["行业名称"] = industry_signal_df["行业名称"].fillna("").astype(str).str.strip()
        industry_signal_df["行业信号"] = industry_signal_df["行业信号"].fillna("").astype(str).str.strip()
        industry_signal_df = industry_signal_df.drop_duplicates(subset=["行业名称"], keep="first")

        signal_map = industry_signal_df.set_index("行业名称")["行业信号"].to_dict()
        stock_df["所属行业信号"] = (
            stock_df["行业"].fillna("").astype(str).str.strip().map(signal_map).fillna("")
        )

        return stock_df

    def get_stock_industry_mapping(self, stock_codes: list[str]) -> pd.DataFrame:
        from DataManager.DataProcessingService import get_stock_industry_mapping as _get_mapping
        return _get_mapping(stock_codes, self.logger)

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
        processed_df10 = raw_data.get("xstp_10_raw", pd.DataFrame()).rename(columns={"最新价": "10日均线价"})
        processed_df30 = raw_data.get("xstp_30_raw", pd.DataFrame()).rename(columns={"最新价": "30日均线价"})
        processed_df60 = raw_data.get("xstp_60_raw", pd.DataFrame()).rename(columns={"最新价": "60日均线价"})

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

        # 4. 合并实时价格
        xstp_base = pd.merge(xstp_base, spot_df[["股票代码", "最新价"]], on="股票代码", how="left")

        # 5. 类型转换和过滤
        cols_to_convert = [col for col in xstp_base.columns if "最新价" in col or col == "最新价"]
        for col in cols_to_convert:
            xstp_base[col] = pd.to_numeric(xstp_base[col], errors="coerce")

        # 过滤条件: 1. 最新价>10日均线 2. 多头排列 (10>30 或 30>60)
        filtered_df = xstp_base[
            (xstp_base["最新价"] > xstp_base["10日均线价"])
            & (
                (xstp_base["10日均线价"] > xstp_base["30日均线价"].fillna(float("-inf")))
                | (xstp_base["30日均线价"] > xstp_base["60日均线价"].fillna(float("-inf")))
            )
        ].copy()

        filtered_df.rename(columns={"最新价": "当前价格"}, inplace=True)
        return filtered_df.fillna("N/A")

    def _get_first_fund_flow_col(self) -> str:
        """
        获取配置的第一个资金流列名

        Returns:
            str: 资金流列名
        """
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
