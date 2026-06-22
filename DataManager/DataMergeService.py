"""
数据合并服务类

负责数据的清洗、合并、转换（从DataProcessingService拆分）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from DataManager.ColumnNames import ColumnNames
from LogicAnalyzer.SignalConstants import TrendLevels
from UtilsManager.CodeNormalizer import CodeNormalizer
from UtilsManager.Exceptions import CalculationError, handle_exception_with_recovery


def get_stock_industry_mapping(
    stock_codes: list[str],
    logger: Any | None = None,  # noqa: ANN401
    engine: Any | None = None,  # noqa: ANN401
) -> pd.DataFrame:
    """Standalone: 从数据库获取股票的行业信息。

    Args:
        stock_codes: 股票代码列表
        logger: 可选的日志器实例
        engine: 可选的 SQLAlchemy Engine（不传则使用全局单例）

    Returns:
        pd.DataFrame: 包含股票代码、名称、行业的DataFrame
    """
    if not stock_codes:
        return pd.DataFrame(columns=[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME, ColumnNames.INDUSTRY])

    try:
        from DataCollection.HistDataEngine import StockSyncEngine
        from DataManager.DbEngine import get_engine as _get_engine

        if engine is None:
            from ConfigParser import Config
            engine = _get_engine(Config())

        sync = StockSyncEngine(db_engine=engine)
        pool = sync.get_stock_pool_from_db()

        formatted = [CodeNormalizer.normalize(c) for c in stock_codes]
        filtered = pool[pool[ColumnNames.STOCK_CODE].isin(formatted)]

        if filtered.empty:
            if logger:
                logger.warning("数据库中未找到匹配的股票信息")
            return pd.DataFrame(columns=[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME, ColumnNames.INDUSTRY])

        result = filtered[[ColumnNames.STOCK_CODE, "name", "industry"]].copy()
        result.columns = [ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME, ColumnNames.INDUSTRY]

        if logger:
            logger.info(f"从数据库成功获取 {len(result)} 条行业信息")
        return result

    except (ImportError, KeyError, ValueError, TypeError) as e:
        if logger:
            logger.warning(f"从数据库获取行业信息失败: {e}，返回空DataFrame")
        return pd.DataFrame(columns=[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME, ColumnNames.INDUSTRY])


class DataMergeService:
    """
    数据合并服务（从DataProcessingService拆分）

    职责：
    - 数据清洗和标准化
    - 多数据源合并（基础信息、多头排列评分、资金流、技术指标、特殊数据）

    Attributes:
        config: 配置管理器实例
        logger: 日志管理器
        momentum_analyzer: 资金流动能分析器
    """

    def __init__(self, config: Any, logger: Any, momentum_analyzer: Any, calendar_mgr: Any | None = None) -> None:  # noqa: ANN401
        self.config = config
        self.logger = logger
        self.momentum_analyzer = momentum_analyzer
        self.calendar_mgr = calendar_mgr
        self._industry_cache: pd.DataFrame | None = None

    # ── 工具方法 ─────────────────────────────────────────────

    def _normalize_stock_code_in_df(self, df: pd.DataFrame, code_col: str = ColumnNames.STOCK_CODE) -> pd.DataFrame:
        if code_col in df.columns:
            df[code_col] = CodeNormalizer.normalize_series(df[code_col])
        return df

    def _fill_missing_columns(self, df: pd.DataFrame, columns: list, default_value: str = "N/A") -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                df[col] = default_value
            else:
                df[col] = df[col].fillna(default_value)
        return df

    def _get_stock_industry_mapping(self, stock_codes: list[str]) -> pd.DataFrame:
        if self._industry_cache is None:
            from DataCollection.HistDataEngine import StockSyncEngine
            from DataManager.DbEngine import get_engine as _get_engine
            sync = StockSyncEngine(db_engine=_get_engine(self.config))
            pool = sync.get_stock_pool_from_db()
            industry = pool[["ts_code", "name", "industry"]].copy()
            industry.columns = [ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME, ColumnNames.INDUSTRY]
            self._industry_cache = industry
        formatted = [CodeNormalizer.normalize(c) for c in stock_codes]
        return self._industry_cache[self._industry_cache[ColumnNames.STOCK_CODE].isin(formatted)]

    # ── 合并方法 ─────────────────────────────────────────────

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
            if isinstance(df, pd.DataFrame) and not df.empty and ColumnNames.STOCK_CODE in df.columns and ColumnNames.STOCK_NAME in df.columns:
                temp = df[[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME]].copy()
                temp = self._normalize_stock_code_in_df(temp)
                name_dfs.append(temp)

        if name_dfs:
            combined_names = pd.concat(name_dfs, ignore_index=True)
            combined_names = combined_names.dropna(subset=[ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME])
            combined_names = combined_names[~combined_names[ColumnNames.STOCK_NAME].isin(["N/A", "", "NaN", "nan"])]
            name_mapping = combined_names.drop_duplicates(subset=[ColumnNames.STOCK_CODE], keep="first")
            if not name_mapping.empty:
                final_df = pd.merge(final_df, name_mapping, on=ColumnNames.STOCK_CODE, how="left")

        if ColumnNames.STOCK_NAME not in final_df.columns:
            final_df[ColumnNames.STOCK_NAME] = "N/A"

        # 获取实时数据
        spot_df = processed_data.get("spot_data_all", pd.DataFrame())
        if not spot_df.empty and ColumnNames.STOCK_CODE in spot_df.columns:
            spot_df = self._normalize_stock_code_in_df(spot_df)
            if ColumnNames.LATEST_PRICE in spot_df.columns:
                self.logger.info(f"[DEBUG] spot_data_all 行数={len(spot_df)} 价格列存在")
                self.logger.info(f"[DEBUG] spot 样本: {spot_df.head(2).to_string()}")
                final_df = pd.merge(
                    final_df,
                    spot_df[[ColumnNames.STOCK_CODE, ColumnNames.LATEST_PRICE]].drop_duplicates(subset=[ColumnNames.STOCK_CODE]),
                    on=ColumnNames.STOCK_CODE,
                    how="left",
                )
                self.logger.info(f"[DEBUG] merge 后 final_df 行数={len(final_df)} 价格样本: {final_df[[ColumnNames.STOCK_CODE, ColumnNames.LATEST_PRICE]].head(3).to_string()}")
            else:
                self.logger.warning(f"[DEBUG] spot_data_all 无最新价列，列名: {list(spot_df.columns)}")
                final_df[ColumnNames.LATEST_PRICE] = float('nan')
        else:
            self.logger.warning("[DEBUG] spot_data_all 为空或无股票代码列")
            final_df[ColumnNames.LATEST_PRICE] = float('nan')

        # 获取行业信息
        self.logger.info("正在获取行业信息...")
        industry_df = self._get_stock_industry_mapping(base_stock_codes)
        if not industry_df.empty:
            # 补全股票简称（如果原始数据中仍有缺失）
            if ColumnNames.STOCK_NAME in industry_df.columns:
                ind_name_map = industry_df.set_index(ColumnNames.STOCK_CODE)[ColumnNames.STOCK_NAME].to_dict()
                final_df[ColumnNames.STOCK_NAME] = final_df.apply(
                    lambda row: (
                        ind_name_map.get(row[ColumnNames.STOCK_CODE], "N/A")
                        if pd.isna(row[ColumnNames.STOCK_NAME]) or row[ColumnNames.STOCK_NAME] == "N/A"
                        else row[ColumnNames.STOCK_NAME]
                    ),
                    axis=1,
                )
            final_df = pd.merge(final_df, industry_df[[ColumnNames.STOCK_CODE, ColumnNames.INDUSTRY]], on=ColumnNames.STOCK_CODE, how="left")
            final_df[ColumnNames.INDUSTRY] = final_df[ColumnNames.INDUSTRY].fillna("N/A")
        else:
            final_df[ColumnNames.INDUSTRY] = "N/A"

        final_df[ColumnNames.STOCK_NAME] = final_df[ColumnNames.STOCK_NAME].fillna("N/A")
        final_df[ColumnNames.INDUSTRY_SIGNAL] = ""

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
            final_df[ColumnNames.BULL_TREND] = TrendLevels.TREND_WATCH
            return final_df

        # 检测日期列
        date_col_candidates = ["trade_date", "date", "日期", "datetime", "Date", "TRADE_DATE"]
        date_col_in_kline = next((c for c in date_col_candidates if c in hist_df_all.columns), None)
        if date_col_in_kline is None:
            self.logger.warning(f"[WARN] K线数据中未找到日期列（候选: {date_col_candidates}）")
            final_df[ColumnNames.BULL_TREND] = TrendLevels.TREND_WATCH
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
                    results[stock_code] = TrendLevels.TREND_WATCH
                    continue

                stock_kline = grouped_klines.get_group(stock_code)
                if stock_kline.empty or len(stock_kline) < 30:
                    results[stock_code] = TrendLevels.TREND_WATCH
                    continue

                # 按日期排序
                stock_kline = stock_kline.sort_values("trade_date").reset_index(drop=True)

                # 计算评分
                result = calculate_full_bull_score(stock_kline, thresholds=thresholds)
                level = result.get("level", TrendLevels.TREND_WATCH)
                status = result.get("status", "FAILED")
                if status != "SUCCESS":
                    level = TrendLevels.TREND_WATCH

                results[stock_code] = level

            except (KeyError, ValueError, TypeError, AttributeError) as e:
                handle_exception_with_recovery(
                    CalculationError("多头排列评分", f"股票 {stock_code}: {e}"),
                    self.logger,
                    f"计算{stock_code}的多头排列评分",
                    default_value=TrendLevels.TREND_WATCH,
                    raise_on_critical=False,
                )
                results[stock_code] = "趋势观望"

            processed_count += 1
            if processed_count % 500 == 0:
                self.logger.info(f"  已处理 {processed_count}/{total_stocks} 只股票...")

        self.logger.info(">>> 多头排列评分计算完成")

        # 将结果添加到 final_df
        final_df[ColumnNames.BULL_TREND] = final_df["股票代码"].map(results).fillna(TrendLevels.TREND_WATCH)

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
            3: ("market_fund_flow_raw_3", ColumnNames.FUND_FLOW_3D),
            5: ("market_fund_flow_raw", ColumnNames.FUND_FLOW_5D),
            10: ("market_fund_flow_raw_10", ColumnNames.FUND_FLOW_10D),
            20: ("market_fund_flow_raw_20", ColumnNames.FUND_FLOW_20D),
        }

        # 根据配置动态处理资金流数据
        for period in self.config.FUND_FLOW_PERIODS:
            if period not in period_map:
                self.logger.warning(f"不支持的资金流周期: {period}日")
                continue

            df_key, col_name = period_map[period]
            fund_flow_df = processed_data.get(df_key, pd.DataFrame())
            flow_col = next(
                (col for col in [ColumnNames.AKSHARE_NET_FLOW, ColumnNames.AKSHARE_FUND_FLOW_NET, ColumnNames.AKSHARE_MAIN_NET_FLOW] if col in fund_flow_df.columns),
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
                    final_df[ColumnNames.FUND_MOMENTUM] = momentum_df["综合_交易信号"]
                elif ColumnNames.FUND_MOMENTUM_STATUS in momentum_df.columns:
                    final_df[ColumnNames.FUND_MOMENTUM] = momentum_df[ColumnNames.FUND_MOMENTUM_STATUS]
                else:
                    final_df[ColumnNames.FUND_MOMENTUM] = result.astype(str)
                if "综合_动能评分" in momentum_df.columns:
                    final_df[ColumnNames.FUND_MOMENTUM_SCORE] = momentum_df["综合_动能评分"]
                elif ColumnNames.FUND_MOMENTUM_SCORE in momentum_df.columns:
                    final_df[ColumnNames.FUND_MOMENTUM_SCORE] = momentum_df[ColumnNames.FUND_MOMENTUM_SCORE]
                self.logger.info(" - 资金动能新分析器运行成功。")
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                self.logger.error(f"运行 FundMomentumAnalyzer 失败: {e}")
                final_df[ColumnNames.FUND_MOMENTUM] = "N/A"
        else:
            final_df[ColumnNames.FUND_MOMENTUM] = "无数据"

        # 处理强势股数据
        strong_df = processed_data.get("strong_stocks_raw", pd.DataFrame())
        if not strong_df.empty and "股票代码" in strong_df.columns:
            strong_df = self._normalize_stock_code_in_df(strong_df)
            strong_codes = set(strong_df["股票代码"].tolist())
            final_df[ColumnNames.STRONG_STOCK] = final_df["股票代码"].apply(lambda x: "是" if x in strong_codes else "否")
        else:
            final_df[ColumnNames.STRONG_STOCK] = "否"

        # 处理连涨数据
        rise_df = processed_data.get("consecutive_rise_raw", pd.DataFrame())
        if not rise_df.empty and "股票代码" in rise_df.columns:
            rise_df = self._normalize_stock_code_in_df(rise_df)
            rise_df = rise_df[["股票代码", ColumnNames.CONSECUTIVE_RISE_DAYS]].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, rise_df, on="股票代码", how="left").fillna({ColumnNames.CONSECUTIVE_RISE_DAYS: 0})
        else:
            final_df[ColumnNames.CONSECUTIVE_RISE_DAYS] = 0
        final_df[ColumnNames.CONSECUTIVE_RISE_DAYS] = final_df[ColumnNames.CONSECUTIVE_RISE_DAYS].astype(int)

        # 处理量价齐升数据
        ljqs_df = processed_data.get("ljqs_raw", pd.DataFrame())
        if not ljqs_df.empty and "股票代码" in ljqs_df.columns:
            ljqs_df = self._normalize_stock_code_in_df(ljqs_df)
            ljqs_codes = set(ljqs_df["股票代码"].tolist())
            final_df[ColumnNames.PRICE_VOLUME_RISE] = final_df["股票代码"].apply(lambda x: "是" if x in ljqs_codes else "否")
        else:
            final_df[ColumnNames.PRICE_VOLUME_RISE] = "否"

        # 处理持续放量数据
        cxfl_df = processed_data.get("cxfl_raw", pd.DataFrame())
        if not cxfl_df.empty and "股票代码" in cxfl_df.columns:
            cxfl_df = self._normalize_stock_code_in_df(cxfl_df)
            cxfl_df = cxfl_df[["股票代码", ColumnNames.VOLUME_INCREASE_DAYS]].drop_duplicates(subset=["股票代码"])
            final_df = pd.merge(final_df, cxfl_df, on="股票代码", how="left").fillna({ColumnNames.VOLUME_INCREASE_DAYS: 0})
        else:
            final_df[ColumnNames.VOLUME_INCREASE_DAYS] = 0
        final_df[ColumnNames.VOLUME_INCREASE_DAYS] = final_df[ColumnNames.VOLUME_INCREASE_DAYS].astype(int)

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

        # MACD趋势综合评分（单参数，7维度+趋势分类+管线结论）
        macd_full_bull_df = processed_data.get("MACD_FULL_BULL", pd.DataFrame())
        if not macd_full_bull_df.empty and "股票代码" in macd_full_bull_df.columns:
            macd_full_bull_df = macd_full_bull_df.rename(columns={"cost_95pct": ColumnNames.CHIP_95_PRICE})
            cols = ["股票代码"]
            for pipe_col in [ColumnNames.MACD_TREND, ColumnNames.MACD_CROSS, ColumnNames.MACD_HIST_MOMENTUM, ColumnNames.DIF_SLOPE, ColumnNames.DIVERGENCE_SIGNAL, ColumnNames.VOLUME_PRICE_CONFIRM, ColumnNames.KLINE_PATTERN,
                             ColumnNames.COMPREHENSIVE_ANALYSIS, ColumnNames.COMPREHENSIVE_SCORE, ColumnNames.COMPREHENSIVE_LEVEL, ColumnNames.RISK_LEVEL, ColumnNames.MACD_TREND_TYPE, "macd_trend",
                             ColumnNames.CHIP_95_PRICE, "资金流净额", "_current_dif",
                             ColumnNames.DIVERGENCE_DAYS, ColumnNames.DIVERGENCE_PRICE,
                             ColumnNames.STOP_LOSS, ColumnNames.T1_TARGET, ColumnNames.T2_TARGET, ColumnNames.TRAILING_STOP, ColumnNames.EXIT_RRR,
                             "position_adjust",
                             ColumnNames.AMOUNT, ColumnNames.AMOUNT_MA20,
                             "宏观风险"]:
                if pipe_col in macd_full_bull_df.columns:
                    cols.append(pipe_col)
            ta_dfs_to_merge.append(macd_full_bull_df[cols])

        # KDJ
        kdj_df = processed_data.get("KDJ", pd.DataFrame())
        if not kdj_df.empty and "股票代码" in kdj_df.columns:
            ta_dfs_to_merge.append(kdj_df[["股票代码", ColumnNames.KDJ_SIGNAL]])

        # CCI
        cci_df = processed_data.get("CCI", pd.DataFrame())
        if not cci_df.empty and "股票代码" in cci_df.columns:
            ta_dfs_to_merge.append(cci_df[["股票代码", ColumnNames.CCI_SIGNAL]])

        # RSI
        rsi_df = processed_data.get("RSI", pd.DataFrame())
        if not rsi_df.empty and "股票代码" in rsi_df.columns:
            rsi_df[ColumnNames.RSI_SIGNAL] = rsi_df[ColumnNames.RSI_SIGNAL].astype(str).str.split(" ").str[0]
            ta_dfs_to_merge.append(rsi_df[["股票代码", ColumnNames.RSI_SIGNAL]])

        # BOLL
        boll_df = processed_data.get("BOLL", pd.DataFrame())
        if not boll_df.empty and "股票代码" in boll_df.columns:
            ta_dfs_to_merge.append(boll_df[["股票代码", ColumnNames.BOLL_SIGNAL]])

        # 合并所有技术指标
        for ta_df in ta_dfs_to_merge:
            if "股票代码" in ta_df.columns:
                final_df = pd.merge(
                    final_df,
                    ta_df.drop_duplicates(subset=["股票代码"]),
                    on="股票代码",
                    how="left",
                )

        # 使用辅助方法批量填充缺失的技术指标列
        ta_signal_cols = [ColumnNames.KDJ_SIGNAL, ColumnNames.CCI_SIGNAL, ColumnNames.RSI_SIGNAL, ColumnNames.BOLL_SIGNAL]
        for col in ta_signal_cols:
            if col not in final_df.columns:
                final_df[col] = ""
            else:
                final_df[col] = final_df[col].fillna("")

        # MACD 管线列也补 NaN → 空串（部分股票可能因早期返回未进 MACD_FULL_BULL）
        macd_fill_cols = [ColumnNames.MACD_TREND, ColumnNames.MACD_CROSS, ColumnNames.MACD_HIST_MOMENTUM, ColumnNames.DIF_SLOPE,
                          ColumnNames.DIVERGENCE_SIGNAL, ColumnNames.VOLUME_PRICE_CONFIRM, ColumnNames.KLINE_PATTERN,
                          ColumnNames.DIVERGENCE_DAYS, ColumnNames.DIVERGENCE_PRICE, ColumnNames.MACD_TREND_TYPE, "macd_trend"]
        for col in macd_fill_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna("")

        return final_df

    def merge_special_data(self, final_df: pd.DataFrame, processed_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并特殊数据：主力成本、均线突破

        Args:
            final_df: 基础DataFrame
            processed_data: 已处理的原始数据字典

        Returns:
            pd.DataFrame: 添加了特殊数据列的DataFrame
        """

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
                            ColumnNames.MAIN_COST_DIFF_PERCENT,
                            ColumnNames.COST_POSITION,
                            ColumnNames.INSTITUTION_LEVEL,
                        ]
                    ].drop_duplicates(subset=[ColumnNames.STOCK_CODE]),
                    on=ColumnNames.STOCK_CODE,
                    how="left",
                )
                # 主力成本保留2位小数
                if ColumnNames.MAIN_COST in final_df.columns:
                    final_df[ColumnNames.MAIN_COST] = pd.to_numeric(final_df[ColumnNames.MAIN_COST], errors='coerce').round(2)
                # 使用辅助方法批量填充主力成本相关列
                self._fill_missing_columns(
                    final_df, [ColumnNames.MAIN_COST, ColumnNames.COST_POSITION], default_value="N/A"
                )
            else:
                # 使用辅助方法批量设置默认值
                self._fill_missing_columns(
                    final_df, [ColumnNames.MAIN_COST, ColumnNames.COST_POSITION], default_value="N/A"
                )

        # 均线突破数据
        xstp_df = processed_data.get("processed_xstp_df", pd.DataFrame())
        xstp_cols = [ColumnNames.STOCK_CODE, ColumnNames.CURRENT_PRICE, ColumnNames.MA10_PRICE, ColumnNames.MA30_PRICE, ColumnNames.MA60_PRICE]
        if not xstp_df.empty and ColumnNames.STOCK_CODE in xstp_df.columns:
            xstp_df = self._normalize_stock_code_in_df(xstp_df)
            cols_present = [col for col in xstp_cols if col in xstp_df.columns]
            merge_df = xstp_df[cols_present].drop_duplicates(subset=[ColumnNames.STOCK_CODE])
            final_df = pd.merge(final_df, merge_df, on=ColumnNames.STOCK_CODE, how="left")


        # 合并研报数据（作为加分因子）
        report_df = processed_data.get("research_report_data", pd.DataFrame())
        if not report_df.empty and ColumnNames.STOCK_CODE in report_df.columns:
            report_df = self._normalize_stock_code_in_df(report_df)
            # 重命名列
            if "机构投资评级(近六个月)-买入" in report_df.columns:
                report_df = report_df.rename(columns={"机构投资评级(近六个月)-买入": ColumnNames.RESEARCH_REPORT_COUNT})
            if ColumnNames.RESEARCH_REPORT_COUNT in report_df.columns:
                final_df = pd.merge(
                    final_df,
                    report_df[[ColumnNames.STOCK_CODE, ColumnNames.RESEARCH_REPORT_COUNT]].drop_duplicates(subset=[ColumnNames.STOCK_CODE]),
                    on=ColumnNames.STOCK_CODE,
                    how="left",
                )
                final_df[ColumnNames.RESEARCH_REPORT_COUNT] = final_df[ColumnNames.RESEARCH_REPORT_COUNT].fillna(0).astype(int)
            else:
                final_df[ColumnNames.RESEARCH_REPORT_COUNT] = 0
        else:
            final_df[ColumnNames.RESEARCH_REPORT_COUNT] = 0

        return final_df
