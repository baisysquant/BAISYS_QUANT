"""
报告生成服务类

负责Excel报告生成、TXT信号文件保存和数据库同步。
"""

from __future__ import annotations

import datetime
import os
from typing import Any

import pandas as pd
from sqlalchemy.exc import DBAPIError, OperationalError

from DataManager.ColumnNames import ColumnNames
from UtilsManager.Exceptions import DatabaseError


class ReportService:
    """
    报告生成服务

    职责：
    - Excel报告生成
    - TXT信号文件保存
    - 数据库同步

    Attributes:
        config: 配置管理器实例
        logger: 日志管理器
    """

    def __init__(self, config: Any, logger: Any) -> None:  # noqa: ANN401
        """
        初始化报告生成服务
        
        Args:
            config: 配置管理器
            logger: 日志管理器
        """
        self.config = config
        self.logger = logger

    def _get_user_focus_stocks(self) -> set[str]:
        """
        从配置中获取用户关注的股票列表
        
        Returns:
            set[str]: 用户关注的股票代码集合（不含SZ/SH前缀）
        """
        try:
            user_focus_str = self.config.USER_FOCUS_STOCKS
            if not user_focus_str or user_focus_str.strip() == "":
                return set()
            
            # Split by | and clean up whitespace
            stocks = {stock.strip() for stock in user_focus_str.split("|") if stock.strip()}
            return stocks
        except AttributeError:
            # If USER_FOCUS_STOCKS config doesn't exist, return empty set
            return set()

    @staticmethod
    def get_base_columns() -> list:
        return [
            ColumnNames.STOCK_CODE,
            ColumnNames.STOCK_NAME,
            ColumnNames.INDUSTRY,
            ColumnNames.INDUSTRY_SIGNAL,
            ColumnNames.LATEST_PRICE,
            ColumnNames.CHIP_95_PRICE,
            ColumnNames.MAIN_COST,
            ColumnNames.COST_POSITION,
        ]

    @staticmethod
    def get_signal_columns() -> list:
        cols = [
            ColumnNames.STRONG_STOCK,
            ColumnNames.PRICE_VOLUME_RISE,
            "量价配合",
            ColumnNames.CONSECUTIVE_RISE_DAYS,
            ColumnNames.VOLUME_INCREASE_DAYS,
            # MACD趋势评分列（前4维度）
            ColumnNames.MACD_TREND,
            ColumnNames.MACD_CROSS,
            "柱状动能",
            "DIF斜率",
            # 独立技术指标（水平多因子交叉验证）
            ColumnNames.KDJ_SIGNAL,
            ColumnNames.CCI_SIGNAL,
            ColumnNames.RSI_SIGNAL,
            ColumnNames.BOLL_SIGNAL,
            ColumnNames.KLINE_PATTERN_SIGNAL,
            # 均线参考
            "10日均线价",
            "30日均线价",
            "60日均线价",
            # 背离信号 + 位置
            "背离信号",
            ColumnNames.DIVERGENCE_DAYS,
            ColumnNames.DIVERGENCE_PRICE,
            ColumnNames.RISK_LEVEL,
            "宏观风险",
        ]
        return cols

    @staticmethod
    def get_report_columns(fund_flow_periods: list = None) -> list:
        cols = [
            ColumnNames.BULL_TREND,
            ColumnNames.COMPREHENSIVE_ANALYSIS,
            ColumnNames.COMPREHENSIVE_SCORE,
            ColumnNames.COMPREHENSIVE_LEVEL,
            ColumnNames.STOP_LOSS,
            ColumnNames.T1_TARGET,
            ColumnNames.T2_TARGET,
            ColumnNames.TRAILING_STOP,
            ColumnNames.EXIT_RRR,
            ColumnNames.RESEARCH_REPORT_COUNT,
            ColumnNames.FUND_MOMENTUM,
        ]
        if fund_flow_periods:
            period_map = {
                5: ColumnNames.FUND_FLOW_5D,
                10: ColumnNames.FUND_FLOW_10D,
                20: ColumnNames.FUND_FLOW_20D,
            }
            for period in fund_flow_periods:
                if period in period_map:
                    cols.append(period_map[period])
        return cols

    @staticmethod
    def get_all_technical_signal_columns() -> list:
        cols = [
            ColumnNames.MACD_TREND,
            ColumnNames.KDJ_SIGNAL,
            ColumnNames.CCI_SIGNAL,
            ColumnNames.RSI_SIGNAL,
            ColumnNames.BOLL_SIGNAL,
        ]
        return cols

    @staticmethod
    def get_final_column_order(fund_flow_periods: list = None) -> list:
        base_cols = ReportService.get_base_columns()
        signal_cols = ReportService.get_signal_columns()
        report_cols = ReportService.get_report_columns(fund_flow_periods)
        tail_cols = [ColumnNames.LIQUIDITY_SCORE, ColumnNames.LIQUIDITY_LEVEL, ColumnNames.SUGGESTED_POSITION, ColumnNames.STOCK_LINK]
        return base_cols + signal_cols + report_cols + tail_cols

    def generate_excel_report(self, sheets_data: dict[str, pd.DataFrame], today_str: str) -> str:
        """
        生成Excel审计报告

        Args:
            sheets_data: 包含多个sheet数据的字典
            today_str: 当前交易日字符串

        Returns:
            str: 报告文件路径

        Raises:
            Exception: 当报告生成失败时抛出异常
        """
        from UtilsManager.Exceptions import ReportGenerationError

        self.logger.info("\n>>> 正在生成 Excel 报告...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.config.TEMP_DATA_DIRECTORY, f"审计报告_{timestamp}.xlsx")

        # Get user focus stocks once for all sheets
        user_focus_stocks = self._get_user_focus_stocks()
        if user_focus_stocks:
            self.logger.info(f"  - 用户关注股池: {', '.join(sorted(user_focus_stocks))}")

        try:
            writer = pd.ExcelWriter(report_path, engine="xlsxwriter", engine_kwargs={'options': {'nan_inf_to_errors': True}})
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
            # Format for user focus stocks: light red background
            user_focus_format = workbook.add_format({"bg_color": "#FFC7CE"})  # Light red

            for sheet_name, df in sheets_data.items():
                if df is None or df.empty:
                    self.logger.debug(f"工作表 '{sheet_name}' 数据为空，跳过创建。")
                    continue

                # If we have user focus stocks and the stock code column exists, sort and prepare for highlighting
                if user_focus_stocks:
                    stock_code_col = ColumnNames.STOCK_CODE
                    if stock_code_col in df.columns:
                        # Create a temporary column for sorting: 1 if in user focus, 0 otherwise
                        df_tmp = df.copy()
                        mask = df_tmp[stock_code_col].isin(user_focus_stocks)
                        # Move user focus stocks to top while preserving original order within each group
                        df_user = df_tmp[mask]
                        df_normal = df_tmp[~mask]
                        df_user = df_user.reset_index(drop=True)
                        df_normal = df_normal.reset_index(drop=True)
                        df_sorted = pd.concat([df_user, df_normal], ignore_index=True)
                    else:
                        # If stock code column not found, use original df
                        df_sorted = df
                else:
                    df_sorted = df

                df_sorted.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)
                worksheet = writer.sheets[sheet_name]

                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Apply formatting for columns and user focus rows
                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col))
                    col_width = min(max_len + 2, 30)

                    if col == ColumnNames.SUGGESTED_POSITION:
                        worksheet.set_column(i, i, col_width, workbook.add_format({"num_format": "0.0%"}))
                    elif col == ColumnNames.LATEST_PRICE or "价格" in col or "价" in col or "线" in col or "均线" in col:
                        worksheet.set_column(i, i, col_width, currency_format)
                    elif "代码" in col:
                        worksheet.set_column(i, i, 10, code_format)
                    elif col in [
                        ColumnNames.FUND_FLOW_3D,
                        ColumnNames.FUND_FLOW_5D,
                        ColumnNames.FUND_FLOW_10D,
                        ColumnNames.FUND_FLOW_20D,
                    ]:
                        # 确保资金流入列使用货币格式
                        worksheet.set_column(i, i, col_width, currency_format)
                    else:
                        worksheet.set_column(i, i, col_width)

                # Apply user focus highlighting if applicable
                if user_focus_stocks and stock_code_col in df.columns:
                    # We need to apply the format to the rows where stock code is in user_focus_stocks
                    # We have already sorted the df_sorted, so we can iterate over the rows and apply the format
                    for row_idx, (_, row) in enumerate(df_sorted.iterrows(), start=2):  # start=2 because header is at row 1 (0-indexed in excel, but we start at row 2 in excel because of header)
                        stock_code = row[stock_code_col]
                        if stock_code in user_focus_stocks:
                            # Apply the user focus format to the entire row
                            for col_idx in range(len(df.columns)):
                                worksheet.write(row_idx, col_idx, row[df.columns[col_idx]], user_focus_format)

            writer.close()
            self.logger.info(f"  - 报告已成功生成并保存到: {report_path}")

            return report_path

        except Exception as e:
            # 报告生成失败是不可恢复的致命错误
            raise ReportGenerationError("Excel审计报告", str(e))

    def save_ta_signals_to_txt(self, ta_signals: dict[str, pd.DataFrame], today_str: str) -> None:
        """
        将技术指标信号结果保存到独立的 TXT 文件。

        Args:
            ta_signals: 技术指标信号字典
            today_str: 当前交易日字符串
        """
        self.logger.info("\n>>> 正在保存技术指标信号到本地 TXT 文件...")

        save_dir = self.config.TEMP_DATA_DIRECTORY

        for indicator_name, df in ta_signals.items():
            if df is None or df.empty:
                continue

            file_name = f"{indicator_name}_Signals_{today_str}.txt"
            file_path = os.path.join(save_dir, file_name)

            try:
                df.to_csv(file_path, sep="|", index=False, encoding="utf-8")
                self.logger.info(f"  - 成功保存 {indicator_name} 信号文件: {file_name}")
            except Exception as e:
                self.logger.error(f"[ERROR] 保存 {indicator_name} 信号文件失败: {e}")

    def sync_to_database(
        self,
        today_str: str,
        consolidated_report: pd.DataFrame,
        industry_df: pd.DataFrame,
        raw_data: dict[str, pd.DataFrame],
    ) -> bool:
        """
        同步数据到数据库

        Args:
            today_str: 当前交易日字符串
            consolidated_report: 汇总报告DataFrame
            industry_df: 行业分析结果DataFrame
            raw_data: 原始数据字典

        Returns:
            bool: 是否成功
        """
        try:
            from DataManager import DatabaseWriter, QuantDataPerformer
            from DataManager.DbEngine import get_engine as _get_engine

            db_manager = DatabaseWriter.QuantDBManager(engine=_get_engine(self.config))

            sync_task = QuantDataPerformer.QuantDBSyncTask(db_manager)

            sync_task.sync_all(
                today_str=today_str,
                consolidated_report=consolidated_report,
                industry_df=industry_df,
                raw_data=raw_data,
            )

            self.logger.info("数据库同步成功完成。")

            return True

        except (DBAPIError, OperationalError, DatabaseError) as e:
            self.logger.error(f"!!! [同步中断] 数据库异常: {e}")
            return False
