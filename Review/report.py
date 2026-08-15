"""
报告生成服务类

负责Excel报告生成、TXT信号文件保存和数据库同步。
"""

from __future__ import annotations

import datetime
import os
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError

from DataManager import DatabaseWriter, QuantDataPerformer
from DataManager.ColumnNames import ColumnNames
from DataManager.DbEngine import get_engine
from UtilsManager.Exceptions import DatabaseError, ReportGenerationError


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
        from DataManager.ColumnNames import get_base_columns as _f
        return _f()

    @staticmethod
    def get_signal_columns() -> list:
        from DataManager.ColumnNames import get_signal_columns as _f
        return _f()

    @staticmethod
    def get_report_columns(fund_flow_periods: list = None) -> list:
        from DataManager.ColumnNames import get_report_columns as _f
        return _f(fund_flow_periods)

    @staticmethod
    def get_all_technical_signal_columns() -> list:
        from DataManager.ColumnNames import get_all_technical_signal_columns as _f
        return _f()

    @staticmethod
    def get_final_column_order(fund_flow_periods: list = None) -> list:
        from DataManager.ColumnNames import get_final_column_order as _f
        return _f(fund_flow_periods)

    def generate_excel_report(
        self,
        sheets_data: dict[str, pd.DataFrame],
        today_str: str,
        risk_sheets: dict[str, pd.DataFrame] | None = None,
    ) -> str:
        """
        生成Excel审计报告

        Args:
            sheets_data: 包含多个sheet数据的字典
            today_str: 当前交易日字符串
            risk_sheets: 风险分析 sheet 字典（VaR/Brinson/因子风险），
                非空时并入 sheets_data；可用 build_risk_sheets() 生成。

        Returns:
            str: 报告文件路径

        Raises:
            Exception: 当报告生成失败时抛出异常
        """
        if risk_sheets:
            sheets_data = {
                **sheets_data,
                **{k: v for k, v in risk_sheets.items() if v is not None and not v.empty},
            }
        self.logger.info("\n>>> 正在生成 Excel 报告...")
        trade_date = today_str.replace("-", "") if today_str else datetime.datetime.now().strftime("%Y%m%d")
        report_path = os.path.join(self.config.HOME_DIRECTORY, f"审计报告_{trade_date}.xlsx")

        # Get user focus stocks once for all sheets
        user_focus_stocks = self._get_user_focus_stocks()
        if user_focus_stocks:
            self.logger.info(f"  - 用户关注股池: {', '.join(sorted(user_focus_stocks))}")

        # ── LiveDataProvider 已返回不复权 OHLC，无需再次转换 ──
        # sheets_data = self._convert_adjusted_to_normal_prices(sheets_data, trade_date)

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

                    if col == ColumnNames.LATEST_PRICE or "价格" in col or "价" in col or "线" in col or "均线" in col:
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

                # ── 跟仓回测 Sheet 条件格式 ──
                if sheet_name == "跟仓回测" and "综合收益率" in df.columns:
                    import xlsxwriter
                    pnl_col_idx = list(df.columns).index("综合收益率")
                    col_letter = xlsxwriter.utility.xl_col_to_name(pnl_col_idx)
                    num_rows = len(df_sorted) + 1
                    rng = f"{col_letter}2:{col_letter}{num_rows}"
                    green_fmt = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
                    red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
                    worksheet.conditional_format(rng, {
                        "type": "formula",
                        "criteria": f'LEFT({col_letter}2,1)<>"-"',
                        "format": green_fmt,
                    })
                    worksheet.conditional_format(rng, {
                        "type": "formula",
                        "criteria": f'LEFT({col_letter}2,1)="-"',
                        "format": red_fmt,
                    })

            writer.close()
            self.logger.info(f"  - 报告已成功生成并保存到: {report_path}")

            return report_path

        except Exception as e:
            # 报告生成失败是不可恢复的致命错误
            raise ReportGenerationError("Excel审计报告", str(e))

    def build_risk_sheets(
        self,
        portfolio_returns: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
        holdings_df: pd.DataFrame | None = None,
        kline_df: pd.DataFrame | None = None,
        industry_map: pd.Series | None = None,
        orth_result: dict | None = None,
    ) -> dict[str, pd.DataFrame]:
        """组合三个风险分析模块，生成可写入 Excel 的风险报告 sheet 字典。

        包含：
          - "风险VaR分析"  —— 历史模拟法 VaR(95%/99%) / ES（基于组合日收益率）
          - "Brinson归因"  —— 行业配置 vs 个股选择收益归因（基于持仓 + K线）
          - "因子风险归因" —— 正交化因子暴露风险归因（基于正交化结果）

        各模块独立容错：单个模块失败仅记日志，不影响其余模块与主报告。

        Args:
            portfolio_returns: 组合日收益率 Series（VaR 用）。
            benchmark_returns: 基准日收益率 Series（暂用于日志/扩展）。
            holdings_df: 持仓明细 DataFrame（Brinson 用，需含 股票代码/目标权重/行业）。
            kline_df: 全市场日 K 线（Brinson 基准构建用）。
            industry_map: 股票代码 → 行业 映射 Series（Brinson 基准构成用）。
            orth_result: FactorOrthogonalizer().run() 的返回 dict（因子风险归因用）。

        Returns:
            dict[str, pd.DataFrame]: sheet 名 → DataFrame；无可用数据时为空 dict。
        """
        from LogicAnalyzer.risk.brinson import BrinsonDecomposition
        from LogicAnalyzer.risk.factor_risk import FactorRiskModel
        from LogicAnalyzer.risk.historical_var import HistoricalVaR

        sheets: dict[str, pd.DataFrame] = {}

        # ── 1. 历史模拟 VaR / ES ──
        if portfolio_returns is not None and not portfolio_returns.empty:
            try:
                var_sheet = HistoricalVaR().build_report(portfolio_returns)
                if not var_sheet.empty:
                    sheets["风险VaR分析"] = var_sheet
                    self.logger.info("[风险分析] VaR/ES 报告已生成")
            except Exception as e:
                self.logger.warning(f"[风险分析] VaR/ES 计算失败: {e}")

        # ── 2. Brinson 归因 ──
        if holdings_df is not None and not holdings_df.empty and kline_df is not None:
            try:
                br = BrinsonDecomposition()
                result = br.from_holdings(
                    holdings_df=holdings_df,
                    kline_df=kline_df,
                    industry_map=industry_map,
                )
                if "error" not in result:
                    br_sheet = br.build_report(result)
                    if not br_sheet.empty:
                        sheets["Brinson归因"] = br_sheet
                        self.logger.info("[风险分析] Brinson 归因报告已生成")
                else:
                    self.logger.warning(f"[风险分析] Brinson 归因失败: {result['error']}")
            except Exception as e:
                self.logger.warning(f"[风险分析] Brinson 归因异常: {e}")

        # ── 3. 因子风险归因 ──
        if orth_result is not None:
            try:
                weights = None
                if holdings_df is not None and not holdings_df.empty:
                    wcol = "目标权重" if "目标权重" in holdings_df.columns else None
                    ccol = "股票代码" if "股票代码" in holdings_df.columns else None
                    if wcol and ccol:
                        weights = pd.Series(
                            pd.to_numeric(holdings_df[wcol], errors="coerce").to_numpy(),
                            index=holdings_df[ccol].astype(str),
                        )
                risk = FactorRiskModel().from_orthogonalizer(
                    orth_result, weights=weights
                )
                if "error" not in risk:
                    sheets["因子风险归因"] = self._factor_risk_sheet(risk)
                    self.logger.info("[风险分析] 因子风险归因报告已生成")
                else:
                    self.logger.warning(f"[风险分析] 因子风险归因失败: {risk['error']}")
            except Exception as e:
                self.logger.warning(f"[风险分析] 因子风险归因异常: {e}")

        return sheets

    @staticmethod
    def _factor_risk_sheet(risk: dict) -> pd.DataFrame:
        """将因子风险归因结果拼装为单一 sheet 表（含组合总览 + 因子 + 个股）。"""
        summary = pd.DataFrame(
            [
                {"模块": "组合总览", "指标": "组合日波动", "数值": f"{risk['组合日波动']:.4%}"},
                {"模块": "组合总览", "指标": "组合年化波动", "数值": f"{risk['组合年化波动']:.4%}"},
                {"模块": "组合总览", "指标": "因子风险占比", "数值": f"{risk['因子风险占比']:.2%}"},
                {"模块": "组合总览", "指标": "特质风险占比", "数值": f"{risk['特质风险占比']:.2%}"},
                {"模块": "组合总览", "指标": "特质波动来源", "数值": "常数估计" if risk["特质波动为估计"] else "实际数据"},
            ]
        )
        factor_df = risk["因子风险贡献"].copy()
        factor_df.insert(0, "模块", "因子风险贡献")
        factor_df = factor_df.rename(columns={"因子": "指标"})
        stock_df = risk["个股特质风险TopN"].copy()
        stock_df.insert(0, "模块", "个股特质风险TopN")
        stock_df = stock_df.rename(columns={"股票": "指标"})
        return pd.concat([summary, factor_df, stock_df], ignore_index=True).fillna("")

    def _convert_adjusted_to_normal_prices(
        self, sheets_data: dict[str, pd.DataFrame], trade_date: str
    ) -> dict[str, pd.DataFrame]:
        """
        将所有 sheet 中的 后复权 止损/目标价/移动止损 转换为 不复权 价格输出。

        P1-6 修复：后复权价 = 不复权价 × adj_factor(交易日当日因子)
        → 不复权价 = 后复权价 / adj_factor(trade_date)。
        此前用 MAX(adj_factor) OVER (PARTITION BY symbol)（全历史最大因子）
        折算，引用了截至报告日的未来信息（除权会推高后续因子），属前视偏差；
        且 P0-12 复权语义统一后，后复权↔不复权只需当日因子，无需历史窗口。

        Args:
            sheets_data: sheet名称 -> DataFrame 映射
            trade_date: 交易日字符串 (YYYYMMDD)

        Returns:
            转换后的 sheets_data
        """
        # 收集所有涉及的股票代码
        stock_codes = set()
        price_cols = [
            ColumnNames.STOP_LOSS,    # "止损价"
            ColumnNames.T1_TARGET,    # "T1目标价"
            ColumnNames.T2_TARGET,    # "T2目标价"
            ColumnNames.TRAILING_STOP,# "移动止损"
        ]

        for df in sheets_data.values():
            if df is None or df.empty:
                continue
            if ColumnNames.STOCK_CODE in df.columns:
                stock_codes.update(df[ColumnNames.STOCK_CODE].dropna().astype(str).tolist())

        if not stock_codes:
            return sheets_data

        # 查询每只股票当日的 adj_factor
        engine = get_engine(self.config)
        trade_date_iso = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

        # P1-6：仅取交易日当日的复权因子（无历史窗口，无未来信息）
        placeholders = ",".join([f":code{i}" for i in range(len(stock_codes))])
        params = {f"code{i}": code for i, code in enumerate(stock_codes)}
        params["trade_date"] = trade_date_iso

        sql = f"""
            SELECT symbol, adj_factor
            FROM stock_daily_kline
            WHERE symbol IN ({placeholders})
              AND trade_date = :trade_date
        """

        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql), params).fetchall()

            # 构建映射: symbol -> adj_factor（当日）
            adj_map = {}
            for row in result:
                symbol, adj = row
                if adj and adj > 0:
                    adj_map[symbol] = float(adj)
        except Exception as e:
            self.logger.warning(f"获取 adj_factor 失败，将保留原价格: {e}")
            return sheets_data

        # 转换每个 sheet 中的价格列
        price_cols = [
            ColumnNames.STOP_LOSS,
            ColumnNames.T1_TARGET,
            ColumnNames.T2_TARGET,
            ColumnNames.TRAILING_STOP,
        ]

        converted_sheets = {}
        for sheet_name, df in sheets_data.items():
            if df is None or df.empty:
                converted_sheets[sheet_name] = df
                continue

            df_copy = df.copy()
            if ColumnNames.STOCK_CODE not in df_copy.columns:
                converted_sheets[sheet_name] = df_copy
                continue

            for col in price_cols:
                if col not in df_copy.columns:
                    continue

                # P1-6：不复权价 = 后复权价 / 当日 adj_factor（无前视）
                def _convert_price(row):
                    code = str(row[ColumnNames.STOCK_CODE]) if ColumnNames.STOCK_CODE in row else None
                    if code in adj_map:
                        adj = adj_map[code]
                        val = row[col]
                        if pd.notna(val) and val > 0 and adj > 0:
                            return round(float(val) / adj, 2)
                    return row[col]

                df_copy[col] = df_copy.apply(_convert_price, axis=1)

            converted_sheets[sheet_name] = df_copy

        return converted_sheets

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
            db_manager = DatabaseWriter.QuantDBManager(engine=get_engine(self.config))

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
