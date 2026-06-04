"""
数据处理服务类

负责数据的清洗、合并、转换、筛选和格式化。
"""

import pandas as pd

from DataManager.ColumnNames import ColumnNames
from DataManager.ShareCodeFormatMgr import format_stock_code
from UtilsManager.CodeNormalizer import CodeNormalizer
from UtilsManager.Exceptions import CalculationError, handle_exception_with_recovery


def get_stock_industry_mapping(
    stock_codes: list[str],
    logger=None,
) -> pd.DataFrame:
    """Standalone: 从数据库获取股票的行业信息。

    Args:
        stock_codes: 股票代码列表
        logger: 可选的日志器实例

    Returns:
        pd.DataFrame: 包含股票代码、名称、行业的DataFrame
    """
    if not stock_codes:
        return pd.DataFrame(columns=["股票代码", "股票简称", "行业"])

    try:
        from DataCollection.HistDataEngine import StockSyncEngine

        engine = StockSyncEngine()
        pool = engine.get_stock_pool_from_db()

        formatted = [CodeNormalizer.normalize(c) for c in stock_codes]
        filtered = pool[pool["股票代码"].isin(formatted)]

        if filtered.empty:
            if logger:
                logger.warning("数据库中未找到匹配的股票信息")
            return pd.DataFrame(columns=["股票代码", "股票简称", "行业"])

        result = filtered[["股票代码", "name", "industry"]].copy()
        result.columns = ["股票代码", "股票简称", "行业"]

        if logger:
            logger.info(f"从数据库成功获取 {len(result)} 条行业信息")
        return result

    except Exception as e:
        if logger:
            logger.warning(f"从数据库获取行业信息失败: {e}，返回空DataFrame")
        return pd.DataFrame(columns=["股票代码", "股票简称", "行业"])


class DataProcessingService:
    """
    数据处理服务

    职责：
    - 数据清洗和标准化
    - 多数据源合并
    - 信号筛选
    - 排序和格式化

    Attributes:
        config: 配置管理器实例
        logger: 日志管理器
        momentum_analyzer: 资金流动能分析器
    """

    def __init__(self, config, logger, momentum_analyzer, calendar_mgr=None):
        """
        初始化数据处理服务

        Args:
            config: 配置管理器
            logger: 日志管理器
            momentum_analyzer: 资金流动能分析器
            calendar_mgr: 交易日历管理器（可选）
        """
        self.config = config
        self.logger = logger
        self.momentum_analyzer = momentum_analyzer
        self.calendar_mgr = calendar_mgr

    def _normalize_stock_code_in_df(self, df: pd.DataFrame, code_col: str = "股票代码") -> pd.DataFrame:
        """
        统一标准化DataFrame中的股票代码列

        Args:
            df: 需要标准化的DataFrame
            code_col: 股票代码列名，默认"股票代码"

        Returns:
            pd.DataFrame: 标准化后的DataFrame（原地修改）
        """
        if code_col in df.columns:
            df[code_col] = CodeNormalizer.normalize_series(df[code_col])
        return df

    def _fill_missing_columns(self, df: pd.DataFrame, columns: list, default_value="N/A") -> pd.DataFrame:
        """
        批量填充缺失列或缺失值

        Args:
            df: DataFrame
            columns: 需要填充的列名列表
            default_value: 默认值，默认"N/A"

        Returns:
            pd.DataFrame: 填充后的DataFrame（原地修改）
        """
        for col in columns:
            if col not in df.columns:
                df[col] = default_value
            else:
                df[col] = df[col].fillna(default_value)
        return df

    def consolidate_data(self, processed_data: dict[str, pd.DataFrame], base_stock_codes: list[str]) -> pd.DataFrame:
        """
        合并所有数据源，生成最终汇总报告

        Args:
            processed_data: 已处理的原始数据字典
            base_stock_codes: 基准股票代码列表

        Returns:
            pd.DataFrame: 最终汇总报告DataFrame

        Raises:
            ValueError: 当关键数据缺失时抛出
        """
        self.logger.info("\n>>> 正在汇总所有数据和信号 (技术指标作为独立列)...")

        # 验证输入数据
        if not base_stock_codes:
            self.logger.warning("[数据验证] 基准股票代码列表为空")
            return pd.DataFrame(columns=["股票代码"])

        if not isinstance(processed_data, dict):
            raise TypeError(f"processed_data 必须是字典类型，实际为 {type(processed_data)}")

        # 初始化最终数据框架
        final_df = pd.DataFrame(base_stock_codes, columns=["股票代码"])
        final_df["股票代码"] = self._normalize_stock_code_in_df(final_df)["股票代码"]

        # 步骤1：合并基础信息（股票名称、实时价格、行业）
        final_df = self.merge_basic_info(final_df, processed_data, base_stock_codes)

        # 步骤2：计算多头排列评分
        final_df = self.calculate_bull_scores(final_df, processed_data)

        # 步骤3：合并资金流数据（包含强势股、连涨、量价齐升、持续放量等信号）
        final_df = self.merge_fund_flow_data(final_df, processed_data)

        # 步骤4：合并技术指标（MACD、KDJ、CCI、RSI、BOLL）
        final_df = self.merge_technical_indicators(final_df, processed_data)

        # 步骤5：合并特殊数据（TOP10行业、主力成本、均线突破）
        final_df = self.merge_special_data(final_df, processed_data)

        # 步骤6：筛选有信号的股票
        final_df = self.filter_signal_stocks(final_df)

        # 步骤7：排序和格式化
        final_df = self.sort_and_format_report(final_df)

        # 步骤8：最终数据验证
        self.validate_final_report(final_df)

        return final_df

    def merge_basic_info(
        self, final_df: pd.DataFrame, processed_data: dict[str, pd.DataFrame], base_stock_codes: list[str]
    ) -> pd.DataFrame:
        """
        合并基础信息：股票名称、实时价格、行业信息

        Args:
            final_df: 基础DataFrame
            processed_data: 已处理的原始数据字典
            base_stock_codes: 基准股票代码列表

        Returns:
            pd.DataFrame: 添加了基础信息的DataFrame
        """

        # 从各数据源提取股票名称
        name_dfs = []
        for key, df in processed_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty and "股票代码" in df.columns and "股票简称" in df.columns:
                temp = df[["股票代码", "股票简称"]].copy()
                temp = self._normalize_stock_code_in_df(temp)
                name_dfs.append(temp)

        if name_dfs:
            combined_names = pd.concat(name_dfs, ignore_index=True)
            combined_names = combined_names.dropna(subset=["股票代码", "股票简称"])
            combined_names = combined_names[~combined_names["股票简称"].isin(["N/A", "", "NaN", "nan"])]
            name_mapping = combined_names.drop_duplicates(subset=["股票代码"], keep="first")
            if not name_mapping.empty:
                final_df = pd.merge(final_df, name_mapping, on="股票代码", how="left")

        if "股票简称" not in final_df.columns:
            final_df["股票简称"] = "N/A"

        # 获取实时数据
        spot_df = processed_data.get("spot_data_all", pd.DataFrame())
        if not spot_df.empty and "股票代码" in spot_df.columns:
            spot_df = self._normalize_stock_code_in_df(spot_df)
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
        self.logger.info("正在获取行业信息...")
        industry_df = self._get_stock_industry_mapping(base_stock_codes)
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
            final_df = pd.merge(final_df, industry_df[["股票代码", "行业"]], on="股票代码", how="left")
            final_df["行业"] = final_df["行业"].fillna("N/A")
        else:
            final_df["行业"] = "N/A"

        final_df["股票简称"] = final_df["股票简称"].fillna("N/A")
        final_df["所属行业信号"] = ""

        return final_df

    def calculate_bull_scores(self, final_df: pd.DataFrame, processed_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算多头排列评分（批量优化版）

        Args:
            final_df: 基础DataFrame，包含股票代码列
            processed_data: 已处理的原始数据字典，必须包含 'hist_data_all' 或 'kline_data'

        Returns:
            pd.DataFrame: 添加了 '多头排列趋势' 列的DataFrame
        """
        from LogicAnalyzer.Indicators import calculate_full_bull_score

        hist_df_all = processed_data.get("hist_data_all")
        if hist_df_all is None:
            hist_df_all = processed_data.get("kline_data", pd.DataFrame())

        if hist_df_all.empty:
            self.logger.warning("[WARN] 历史K线数据为空，无法计算多头排列评分，将填充默认值。")
            final_df["多头排列趋势"] = "趋势观望"
            return final_df

        # 检测日期列
        date_col_candidates = ["trade_date", "date", "日期", "datetime", "Date", "TRADE_DATE"]
        date_col_in_kline = next((c for c in date_col_candidates if c in hist_df_all.columns), None)
        if date_col_in_kline is None:
            self.logger.warning(f"[WARN] K线数据中未找到日期列（候选: {date_col_candidates}）")
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
                f"无法在K线数据中找到股票代码列。支持的列名: {possible_cols}, 实际列: {list(hist_df_all.columns)}"
            )

        # 过滤数据到业务日期
        last_trade_day = self.calendar_mgr.get_last_trading_day()
        hist_df_all["trade_date"] = hist_df_all["trade_date"].astype(str).str[:10]
        hist_df_all = hist_df_all[hist_df_all["trade_date"] <= last_trade_day].copy()
        self.logger.info(f"[INFO] 评分用K线截止日期: {last_trade_day}，过滤后数据量: {len(hist_df_all)} 行")

        # 标准化K线数据中的股票代码
        hist_df_all = self._normalize_stock_code_in_df(hist_df_all, code_col_in_kline)
        # 将标准化后的代码复制到 normalized_code 列
        hist_df_all["normalized_code"] = hist_df_all[code_col_in_kline]

        # 预计算所有均线（向量化操作，比逐行计算快得多）
        for period in self.config.MOVING_AVERAGE_PERIODS:
            col = f"MA{period}"
            if col not in hist_df_all.columns:
                hist_df_all[col] = hist_df_all.groupby("normalized_code")["close"].transform(
                    lambda x: x.rolling(window=period, min_periods=1).mean()
                )

        # 成交量均线（固定5日）
        if "MA_Volume_5" not in hist_df_all.columns:
            hist_df_all["MA_Volume_5"] = hist_df_all.groupby("normalized_code")["volume"].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

        # 按股票代码分组
        grouped_klines = hist_df_all.groupby("normalized_code")
        self.logger.info(f">>> 开始批量计算多头排列评分，共 {len(final_df)} 只股票...")

        # 使用配置中的阈值参数
        thresholds = {
            "full_bull": self.config.FULL_BULL_THRESHOLD,
            "trend_acceleration": self.config.TREND_ACCELERATION_THRESHOLD,
            "trend_oscillation": self.config.TREND_OSCILLATION_THRESHOLD,
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
                handle_exception_with_recovery(
                    CalculationError("多头排列评分", f"股票 {stock_code}: {e}"),
                    self.logger,
                    f"计算{stock_code}的多头排列评分",
                    default_value="趋势观望",
                    raise_on_critical=False,
                )
                results[stock_code] = "趋势观望"

            processed_count += 1
            if processed_count % 500 == 0:
                self.logger.info(f"  已处理 {processed_count}/{total_stocks} 只股票...")

        self.logger.info(">>> 多头排列评分计算完成")

        # 将结果添加到 final_df
        final_df["多头排列趋势"] = final_df["股票代码"].map(results).fillna("趋势观望")

        return final_df

    def merge_fund_flow_data(self, final_df: pd.DataFrame, processed_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并资金流数据并进行动能分析

        Args:
            final_df: 基础DataFrame
            processed_data: 已处理的原始数据字典

        Returns:
            pd.DataFrame: 添加了资金流列和动能列的DataFrame
        """
        from DataManager import ParallelUtils as utils

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
                (col for col in ["净流入", "资金流入净额", "今日主力净流入-净额"] if col in fund_flow_df.columns),
                None,
            )

            if not fund_flow_df.empty and "股票代码" in fund_flow_df.columns and flow_col:
                fund_flow_df = self._normalize_stock_code_in_df(fund_flow_df)
                final_df = pd.merge(
                    final_df,
                    fund_flow_df[["股票代码", flow_col]].drop_duplicates(subset=["股票代码"]),
                    on="股票代码",
                    how="left",
                )
                final_df = final_df.rename(columns={flow_col: col_name})
            elif not fund_flow_df.empty and "股票简称" in fund_flow_df.columns and flow_col:
                merge_df = fund_flow_df[["股票简称", flow_col]].drop_duplicates(subset=["股票简称"])
                final_df = pd.merge(final_df, merge_df, on="股票简称", how="left")
                final_df = final_df.rename(columns={flow_col: col_name})
            else:
                final_df[col_name] = 0.0

        # 资金流数据标准化处理（统一单位转换为"万元"）
        fund_flow_cols = [period_map[p][1] for p in self.config.FUND_FLOW_PERIODS if p in period_map]

        if any(col in final_df.columns for col in fund_flow_cols):
            final_df = utils._normalize_fund_data(final_df)

        # 检查是否所有配置的周期都存在，以计算资金动能
        if all(col in final_df.columns for col in fund_flow_cols):
            try:
                result = final_df.apply(lambda row: self.momentum_analyzer.analyze(row), axis=1)
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
                self.logger.info(" - 资金动能新分析器运行成功。")
            except Exception as e:
                self.logger.error(f"运行 FundMomentumAnalyzer 失败: {e}")
                final_df["资金动能"] = "N/A"
        else:
            final_df["资金动能"] = "无数据"

        # 处理强势股数据
        strong_df = processed_data.get("strong_stocks_raw", pd.DataFrame())
        if not strong_df.empty and "股票代码" in strong_df.columns:
            strong_df = self._normalize_stock_code_in_df(strong_df)
            strong_codes = set(strong_df["股票代码"].tolist())
            final_df["强势股"] = final_df["股票代码"].apply(lambda x: "是" if x in strong_codes else "否")
        else:
            final_df["强势股"] = "否"

        # 处理连涨数据
        rise_df = processed_data.get("consecutive_rise_raw", pd.DataFrame())
        if not rise_df.empty and "股票代码" in rise_df.columns:
            rise_df = self._normalize_stock_code_in_df(rise_df)
            rise_df = rise_df[["股票代码", "连涨天数"]].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, rise_df, on="股票代码", how="left").fillna({"连涨天数": 0})
        else:
            final_df["连涨天数"] = 0
        final_df["连涨天数"] = final_df["连涨天数"].astype(int)

        # 处理量价齐升数据
        ljqs_df = processed_data.get("ljqs_raw", pd.DataFrame())
        if not ljqs_df.empty and "股票代码" in ljqs_df.columns:
            ljqs_df = self._normalize_stock_code_in_df(ljqs_df)
            ljqs_codes = set(ljqs_df["股票代码"].tolist())
            final_df["量价齐升"] = final_df["股票代码"].apply(lambda x: "是" if x in ljqs_codes else "否")
        else:
            final_df["量价齐升"] = "否"

        # 处理持续放量数据
        cxfl_df = processed_data.get("cxfl_raw", pd.DataFrame())
        if not cxfl_df.empty and "股票代码" in cxfl_df.columns:
            cxfl_df = self._normalize_stock_code_in_df(cxfl_df)
            cxfl_df = cxfl_df[["股票代码", "放量天数"]].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, cxfl_df, on="股票代码", how="left").fillna({"放量天数": 0})
        else:
            final_df["放量天数"] = 0
        final_df["放量天数"] = final_df["放量天数"].astype(int)

        return final_df

    def merge_technical_indicators(
        self, final_df: pd.DataFrame, processed_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合并技术指标数据：MACD、KDJ、CCI、RSI、BOLL

        Args:
            final_df: 基础DataFrame
            processed_data: 已处理的原始数据字典

        Returns:
            pd.DataFrame: 添加了技术指标列的DataFrame
        """
        ta_dfs_to_merge = []

        # MACD 标准参数（强制保留）
        macd_df_standard = processed_data.get("MACD_12269", pd.DataFrame())
        if not macd_df_standard.empty and "股票代码" in macd_df_standard.columns:
            ta_dfs_to_merge.append(
                macd_df_standard[["股票代码", "MACD_12269_Signal"]].rename(columns={"MACD_12269_Signal": "MACD_12269"})
            )

        # MACD 第二周期（必填）
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"
        macd_key = f"MACD_{second_period_name}"
        macd_df_second = processed_data.get(macd_key, pd.DataFrame())
        if not macd_df_second.empty and "股票代码" in macd_df_second.columns:
            signal_col = f"{macd_key}_Signal"
            ta_dfs_to_merge.append(macd_df_second[["股票代码", signal_col]].rename(columns={signal_col: macd_key}))

        # MACD 完全多头综合评分
        macd_full_bull_df = processed_data.get("MACD_FULL_BULL", pd.DataFrame())
        if not macd_full_bull_df.empty and "股票代码" in macd_full_bull_df.columns:
            cols = ["股票代码", "FullBull_Score", "MACD_FULL_BULL_Label"]
            if "FullBull_Score_Base" in macd_full_bull_df.columns:
                cols.append("FullBull_Score_Base")
            ta_dfs_to_merge.append(macd_full_bull_df[cols])

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
            # 使用辅助方法批量填充动能列
            macd_momentum_cols = ["MACD_12269_动能"]

            # 第二周期动能列（必填）
            fast, slow, signal = self.config.MACD_SECOND_PARAMS
            second_period_name = f"{fast}{slow}{signal}"
            mom_col = f"MACD_{second_period_name}_动能"
            macd_momentum_cols.append(mom_col)

            for col in macd_momentum_cols:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna("")

        # 使用辅助方法批量填充缺失的技术指标列
        macd_cols = ["MACD_12269", "MACD_FULL_BULL_Label"]
        # 添加第二周期列名（必填）
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"
        macd_cols.append(f"MACD_{second_period_name}")

        ta_signal_cols = macd_cols + ["KDJ_Signal", "CCI_Signal", "RSI_Signal", "BOLL_Signal"]
        for col in ta_signal_cols:
            if col not in final_df.columns:
                final_df[col] = ""
            else:
                final_df[col] = final_df[col].fillna("")

        if "FullBull_Score" not in final_df.columns:
            final_df["FullBull_Score"] = 0
        else:
            final_df["FullBull_Score"] = pd.to_numeric(final_df["FullBull_Score"], errors="coerce").fillna(0)

        if "FullBull_Score_Base" not in final_df.columns:
            final_df["FullBull_Score_Base"] = 0
        else:
            final_df["FullBull_Score_Base"] = pd.to_numeric(final_df["FullBull_Score_Base"], errors="coerce").fillna(0)

        return final_df

    def merge_special_data(self, final_df: pd.DataFrame, processed_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并特殊数据：TOP10行业、主力成本、均线突破

        Args:
            final_df: 基础DataFrame
            processed_data: 已处理的原始数据字典

        Returns:
            pd.DataFrame: 添加了特殊数据列的DataFrame
        """

        # 处理行业数据
        top_ind_df = processed_data.get("top_industry_cons_df", pd.DataFrame())
        if not top_ind_df.empty and "股票代码" in top_ind_df.columns:
            top_ind_df = self._normalize_stock_code_in_df(top_ind_df)
            top_codes = set(top_ind_df["股票代码"].astype(str).unique())
            final_df["TOP10行业"] = final_df["股票代码"].apply(lambda x: "是" if str(x) in top_codes else "否")
        else:
            final_df["TOP10行业"] = "否"

        # 主力成本数据
        main_cost_df = processed_data.get("main_cost_data", pd.DataFrame())
        if not main_cost_df.empty:
            if ColumnNames.AKSHARE_CODE_RAW in main_cost_df.columns:
                main_cost_df.rename(columns={ColumnNames.AKSHARE_CODE_RAW: ColumnNames.STOCK_CODE}, inplace=True)
            if ColumnNames.STOCK_CODE in main_cost_df.columns:
                main_cost_df = self._normalize_stock_code_in_df(main_cost_df, ColumnNames.STOCK_CODE)
                final_df = pd.merge(
                    final_df,
                    main_cost_df[
                        [
                            ColumnNames.STOCK_CODE,
                            ColumnNames.MAIN_COST,
                            ColumnNames.INSTITUTION_PARTICIPATION,
                            ColumnNames.MAIN_COST_DIFF,
                            ColumnNames.MAIN_COST_DIFF_PERCENT,
                            ColumnNames.COST_POSITION,
                            ColumnNames.INSTITUTION_LEVEL,
                            ColumnNames.MAIN_CONTROL_STRENGTH,
                        ]
                    ],
                    on=ColumnNames.STOCK_CODE,
                    how="left",
                )
                # 使用辅助方法批量填充主力成本相关列
                self._fill_missing_columns(
                    final_df, ["主力成本", "主力成本差价", "成本位置", "主力控盘强度"], default_value="N/A"
                )
            else:
                # 使用辅助方法批量设置默认值
                self._fill_missing_columns(
                    final_df, ["主力成本", "主力成本差价", "成本位置", "主力控盘强度"], default_value="N/A"
                )

        # 均线突破数据
        xstp_df = processed_data.get("processed_xstp_df", pd.DataFrame())
        xstp_cols = ["股票代码", "完全多头排列", "当前价格", "10日均线价", "30日均线价", "60日均线价"]
        if not xstp_df.empty and "股票代码" in xstp_df.columns:
            xstp_df = self._normalize_stock_code_in_df(xstp_df)
            cols_present = [col for col in xstp_cols if col in xstp_df.columns]
            merge_df = xstp_df[cols_present].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, merge_df, on="股票代码", how="left")

        if "完全多头排列" not in final_df.columns:
            final_df["完全多头排列"] = "否"
        else:
            final_df["完全多头排列"] = final_df["完全多头排列"].fillna("否")

        # 合并研报数据（作为加分因子）
        report_df = processed_data.get("research_report_data", pd.DataFrame())
        if not report_df.empty and "股票代码" in report_df.columns:
            report_df = self._normalize_stock_code_in_df(report_df)
            # 重命名列
            if "机构投资评级(近六个月)-买入" in report_df.columns:
                report_df = report_df.rename(columns={"机构投资评级(近六个月)-买入": "研报买入次数"})
            if "研报买入次数" in report_df.columns:
                final_df = pd.merge(
                    final_df,
                    report_df[["股票代码", "研报买入次数"]].drop_duplicates(subset=["股票代码"]),
                    on="股票代码",
                    how="left",
                )
                final_df["研报买入次数"] = final_df["研报买入次数"].fillna(0).astype(int)
            else:
                final_df["研报买入次数"] = 0
        else:
            final_df["研报买入次数"] = 0

        return final_df

    def filter_signal_stocks(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        筛选有信号的股票

        筛选条件：满足以下任一条件
        - 完全多头排列
        - 强势股
        - 量价齐升
        - TOP10行业
        - 任意技术指标有信号

        Args:
            final_df: 包含所有数据的DataFrame

        Returns:
            pd.DataFrame: 筛选后的DataFrame
        """
        if final_df.empty:
            return final_df

        # 动态获取第二周期MACD列名
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"

        # 使用常量类获取所有技术指标信号列
        from DataManager.ReportService import ReportService
        str_cols = ReportService.get_all_technical_signal_columns(second_period_name)

        mask = (
            (final_df[ColumnNames.PERFECT_BULL_ARRANGEMENT] == "是")
            | final_df[ColumnNames.STRONG_STOCK].eq("是")
            | final_df[ColumnNames.PRICE_VOLUME_RISE].eq("是")
            | final_df.get(ColumnNames.TOP10_INDUSTRY, "").eq("是")
            | final_df[str_cols].apply(lambda s: s.str.strip().ne("")).any(axis=1)
        )

        filtered_count = len(final_df) - mask.sum()
        if filtered_count > 0:
            self.logger.info(f"  - 筛选掉 {filtered_count} 只无信号股票，剩余 {mask.sum()} 只")

        return final_df[mask].copy()

    def sort_and_format_report(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        对报告进行排序、格式化和列重排

        Args:
            final_df: 筛选后的DataFrame

        Returns:
            pd.DataFrame: 格式化后的DataFrame
        """
        if final_df.empty:
            return final_df

        # 排序：连涨天数和放量天数降序
        final_df.sort_values(
            by=[ColumnNames.CONSECUTIVE_RISE_DAYS, ColumnNames.VOLUME_INCREASE_DAYS],
            ascending=[False, False],
            inplace=True,
        )
        final_df.reset_index(drop=True, inplace=True)

        # 生成股票链接
        final_df["完整股票代码"] = final_df[ColumnNames.STOCK_CODE].apply(format_stock_code)
        final_df[ColumnNames.STOCK_LINK] = "https://hybrid.gelonghui.com/stock-check/" + final_df["完整股票代码"]
        final_df.drop(columns=["完整股票代码"], inplace=True, errors="ignore")

        # 删除冗余的价格列
        if ColumnNames.CURRENT_PRICE in final_df.columns and ColumnNames.LATEST_PRICE in final_df.columns:
            final_df.drop(columns=[ColumnNames.CURRENT_PRICE], inplace=True, errors="ignore")

        # 重新排列列顺序
        final_df = self.reorder_columns(final_df)

        return final_df

    def reorder_columns(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        重新排列报告列顺序

        Args:
            final_df: 格式化后的DataFrame

        Returns:
            pd.DataFrame: 列重排后的DataFrame
        """
        # 动态获取第二周期名称
        fast, slow, signal = self.config.MACD_SECOND_PARAMS
        second_period_name = f"{fast}{slow}{signal}"

        # 使用常量类获取最终列顺序
        from DataManager.ReportService import ReportService
        final_cols = ReportService.get_final_column_order(
            second_period_name=second_period_name, fund_flow_periods=self.config.FUND_FLOW_PERIODS
        )

        # 只保留存在的列
        existing_cols = [col for col in final_cols if col in final_df.columns]

        return final_df[existing_cols]

    def validate_final_report(self, final_df: pd.DataFrame) -> bool:
        """
        最终报告数据验证

        Args:
            final_df: 最终报告DataFrame

        Returns:
            bool: 验证是否通过
        """
        from LogicAnalyzer.DataValidator import DataValidator

        if final_df.empty:
            self.logger.warning("[数据验证] 最终报告为空")
            return False

        data_validator = DataValidator(self.logger)

        # 检查必需列
        required_report_cols = [ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME, ColumnNames.LATEST_PRICE]
        is_valid, missing = data_validator.validate_required_columns(final_df, required_report_cols, "最终报告")

        if not is_valid:
            self.logger.error(f"[数据验证] 最终报告缺少关键列: {missing}")
            return False

        # 验证价格数据
        price_valid, anomalies = data_validator.validate_price_data(
            final_df, [ColumnNames.LATEST_PRICE], "最终报告价格"
        )

        if not price_valid:
            self.logger.warning(f"[数据验证] 最终报告价格异常: {anomalies}")

        self.logger.info(f"[数据验证] 最终报告生成成功: {len(final_df)} 条记录, {len(final_df.columns)} 个字段")

        return True

    def _get_stock_industry_mapping(self, stock_codes: list[str]) -> pd.DataFrame:
        return get_stock_industry_mapping(stock_codes, self.logger)
