"""
报告生成服务类

负责Excel报告生成、TXT信号文件保存和数据库同步。
"""

import os

import pandas as pd


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

    def __init__(self, config, logger):
        """
        初始化报告生成服务

        Args:
            config: 配置管理器
            logger: 日志管理器
        """
        self.config = config
        self.logger = logger

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
        report_path = os.path.join(self.config.TEMP_DATA_DIRECTORY, f"审计报告_{today_str}.xlsx")

        try:
            writer = pd.ExcelWriter(report_path, engine="xlsxwriter")
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

            for sheet_name, df in sheets_data.items():
                if df is None or df.empty:
                    self.logger.debug(f"工作表 '{sheet_name}' 数据为空，跳过创建。")
                    continue

                df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)
                worksheet = writer.sheets[sheet_name]

                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col))
                    col_width = min(max_len + 2, 30)

                    if col == "最新价" or "价格" in col or "价" in col or "线" in col or "均线" in col:
                        worksheet.set_column(i, i, col_width, currency_format)
                    elif "代码" in col:
                        worksheet.set_column(i, i, 10, code_format)
                    elif col in [
                        "3日资金流入万元",
                        "5日资金流入万元",
                        "10日资金流入万元",
                        "20日资金流入万元",
                    ]:
                        # 确保资金流入列使用货币格式
                        worksheet.set_column(i, i, col_width, currency_format)
                    else:
                        worksheet.set_column(i, i, col_width)

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
        second_period_name: str,
    ) -> bool:
        """
        同步数据到数据库

        Args:
            today_str: 当前交易日字符串
            consolidated_report: 汇总报告DataFrame
            industry_df: 行业分析结果DataFrame
            raw_data: 原始数据字典
            second_period_name: MACD第二周期名称

        Returns:
            bool: 是否成功
        """
        try:
            from DataManager import DatabaseWriter, QuantDataPerformer

            db_manager = DatabaseWriter.QuantDBManager(
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                db_name=self.config.DB_NAME,
            )

            sync_task = QuantDataPerformer.QuantDBSyncTask(db_manager)

            sync_task.sync_all(
                today_str=today_str,
                consolidated_report=consolidated_report,
                industry_df=industry_df,
                raw_data=raw_data,
                second_period_name=second_period_name,
            )

            db_manager.close()
            self.logger.info("数据库同步成功完成。")

            return True

        except Exception as e:
            self.logger.error(f"!!! [同步中断] 任务运行异常: {e}")
            return False
