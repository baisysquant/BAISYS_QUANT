"""
数据处理服务类

负责数据的清洗、合并、转换、筛选和格式化。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandera.errors import SchemaErrors

from DataManager.ColumnNames import ColumnNames
from DataManager.DataMergeService import DataMergeService
from DataManager.DataSchemas import create_final_report_schema
from LogicAnalyzer.PositionSizer import calculate_positions
from UtilsManager.CodeNormalizer import CodeNormalizer


class DataProcessingService:
    """
    数据处理服务

    职责：
    - 数据清洗和标准化（委托 DataMergeService）
    - 多数据源合并（委托 DataMergeService）
    - 信号筛选
    - 排序和格式化

    Attributes:
        config: 配置管理器实例
        logger: 日志管理器
        momentum_analyzer: 资金流动能分析器
    """

    def __init__(self, config: Any, logger: Any, momentum_analyzer: Any, calendar_mgr: Any | None = None) -> None:  # noqa: ANN401
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
        self._data_merge = DataMergeService(config, logger, momentum_analyzer, calendar_mgr)

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
            return pd.DataFrame(columns=[ColumnNames.STOCK_CODE])

        if not isinstance(processed_data, dict):
            raise TypeError(f"processed_data 必须是字典类型，实际为 {type(processed_data)}")

        # 初始化最终数据框架
        final_df = pd.DataFrame(base_stock_codes, columns=[ColumnNames.STOCK_CODE])
        final_df[ColumnNames.STOCK_CODE] = self._data_merge._normalize_stock_code_in_df(final_df)[ColumnNames.STOCK_CODE]

        # 步骤1：合并基础信息（股票名称、实时价格、行业）
        final_df = self._data_merge.merge_basic_info(final_df, processed_data, base_stock_codes)

        # 步骤2：计算多头排列评分
        final_df = self._data_merge.calculate_bull_scores(final_df, processed_data)

        # 步骤3：合并资金流数据（包含强势股、连涨、量价齐升、持续放量等信号）
        final_df = self._data_merge.merge_fund_flow_data(final_df, processed_data)

        # 步骤4：合并技术指标（MACD、KDJ、CCI、RSI、BOLL）
        final_df = self._data_merge.merge_technical_indicators(final_df, processed_data)

        # 步骤5：合并特殊数据（主力成本、均线突破）
        final_df = self._data_merge.merge_special_data(final_df, processed_data)

        # 步骤5.5：流动性评分（截面+时序+规模三因子）
        if not final_df.empty:
            self._apply_liquidity_scoring(final_df)

        # 步骤6：筛选有信号的股票
        final_df = self.filter_signal_stocks(final_df)

        # 步骤7：计算建议仓位比例（基于 Kelly + 多因子混合模型）
        if not final_df.empty:
            pos_config = getattr(self.config, 'POSITION_SIZING', None) or {}
            scoring_params = getattr(self.config, 'SCORING_PARAMS', None) or {}
            pos_config["atr_stop_mult"] = scoring_params.get("atr_stop_mult", 1.5)
            final_df = calculate_positions(final_df, config=pos_config)
            self.logger.info(f"  - 仓位计算完成，共 {len(final_df)} 只")
        else:
            final_df[ColumnNames.SUGGESTED_POSITION] = None
            final_df[ColumnNames.POSITION_REASON] = ""

        # 步骤7.5：Gate 5 - 运行时仓位约束（流动性冲击 + 行业集中度 + 总仓位上限）
        if not final_df.empty:
            self._apply_gate5_position_constraints(final_df)

        # 步骤8：排序和格式化
        final_df = self.sort_and_format_report(final_df)

        # 步骤9：最终数据验证
        self.validate_final_report(final_df)

        return final_df

    def filter_signal_stocks(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        筛选有信号的股票

        筛选条件：满足以下任一条件
        - 强势股
        - 量价齐升
        - 任意技术指标有信号

        Args:
            final_df: 包含所有数据的DataFrame

        Returns:
            pd.DataFrame: 筛选后的DataFrame
        """
        if final_df.empty:
            return final_df

        # 使用常量类获取所有技术指标信号列
        from DataManager.ReportService import ReportService
        str_cols = ReportService.get_all_technical_signal_columns()
        str_cols = [c for c in str_cols if c in final_df.columns]

        mask = (
            final_df[ColumnNames.STRONG_STOCK].eq("是")
            | final_df[ColumnNames.PRICE_VOLUME_RISE].eq("是")
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
        final_df["完整股票代码"] = final_df[ColumnNames.STOCK_CODE].apply(CodeNormalizer.add_market_prefix)
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
        # 使用常量类获取最终列顺序
        from DataManager.ReportService import ReportService
        final_cols = ReportService.get_final_column_order(
            fund_flow_periods=self.config.FUND_FLOW_PERIODS
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

        Raises:
            ValueError: Pandera 数据合约校验失败时抛出，阻断 pipeline
        """
        from LogicAnalyzer.DataValidator import DataValidator

        if final_df.empty:
            self.logger.warning("[数据验证] 最终报告为空")
            return False

        # ── Pandera 数据合约校验（阻塞式） ──
        try:
            schema = create_final_report_schema()
            schema.validate(final_df, lazy=True)
            self.logger.info("[数据合约] 最终报告通过 Pandera 校验")
        except SchemaErrors as e:
            msg = f"[数据合约] 最终报告校验失败: {e}"
            self.logger.error(msg)
            raise ValueError(msg) from e

        # ── 业务规则校验（非阻塞，仅告警） ──
        data_validator = DataValidator(self.logger)

        required_report_cols = [ColumnNames.STOCK_CODE, ColumnNames.STOCK_NAME, ColumnNames.LATEST_PRICE]
        is_valid, missing = data_validator.validate_required_columns(final_df, required_report_cols, "最终报告")

        if not is_valid:
            self.logger.error(f"[数据验证] 最终报告缺少关键列: {missing}")
            return False

        price_valid, anomalies = data_validator.validate_price_data(
            final_df, [ColumnNames.LATEST_PRICE], "最终报告价格"
        )

        if not price_valid:
            self.logger.warning(f"[数据验证] 最终报告价格异常: {anomalies}")

        self.logger.info(f"[数据验证] 最终报告生成成功: {len(final_df)} 条记录, {len(final_df.columns)} 个字段")

        return True

    def _apply_liquidity_scoring(self, df: pd.DataFrame) -> None:
        """
        流动性评分（Gate 4 后处理）：三因子模型 → 流动性等级 + 连续评分。
        修改 df 新增 LIQUIDITY_SCORE / LIQUIDITY_LEVEL 列。
        """
        if df.empty or ColumnNames.AMOUNT not in df.columns:
            return

        cfg = getattr(self.config, 'POSITION_SIZING', None) or {}
        w_section = cfg.get("liq_w_section", 0.5)
        w_timeseries = cfg.get("liq_w_timeseries", 0.5)
        w_marketcap = cfg.get("liq_w_marketcap", 0.0)

        # 行业中位数成交额
        if ColumnNames.INDUSTRY in df.columns:
            df[ColumnNames.INDUSTRY_MEDIAN_AMOUNT] = df.groupby(ColumnNames.INDUSTRY)[
                ColumnNames.AMOUNT
            ].transform('median')

        scores = []
        levels = []

        for _, row in df.iterrows():
            amount = row.get(ColumnNames.AMOUNT, 0) or 0
            amount_ma20 = row.get(ColumnNames.AMOUNT_MA20, 0) or 0
            industry_median = row.get(ColumnNames.INDUSTRY_MEDIAN_AMOUNT, 0) or 0

            section_score = min(amount / max(industry_median, 1), 1.0) if industry_median > 0 else 0.5
            timeseries_score = min(amount / max(amount_ma20, 1), 1.0) if amount_ma20 > 0 else 0.5

            if w_section + w_timeseries + w_marketcap <= 0:
                liq_score = 1.0
            else:
                norm = w_section + w_timeseries + w_marketcap
                liq_score = (w_section * section_score + w_timeseries * timeseries_score) / norm

            liq_score = max(0.0, min(1.0, liq_score))
            scores.append(round(liq_score, 4))

            if liq_score >= 0.8:
                levels.append('充足')
            elif liq_score >= 0.4:
                levels.append('正常')
            elif liq_score >= 0.1:
                levels.append('不足')
            else:
                levels.append('枯竭')

        df[ColumnNames.LIQUIDITY_SCORE] = scores
        df[ColumnNames.LIQUIDITY_LEVEL] = levels

        n_low = sum(1 for l in levels if l in ('不足', '枯竭'))
        if n_low:
            self.logger.info(f"  - [流动性] 低流动性 {n_low} 只（不足+枯竭），共 {len(df)} 只")

    def _apply_gate5_liquidity_impact(self, df: pd.DataFrame) -> None:
        """
        Gate 5: 流动性冲击成本检查。
        对每只持仓股票，估算买入冲击成本，超限则缩仓或清零。

        R62: 单股冲击成本估计 = impact_base * (participation / threshold)^1.5
             如果冲击成本 > max_impact_rate（默认 2%），仓位等比缩至安全线。
        """
        if df.empty or ColumnNames.SUGGESTED_POSITION not in df.columns:
            return
        if ColumnNames.AMOUNT not in df.columns:
            self.logger.warning("  - [Gate5/R62] 缺少 AMOUNT 列，跳过冲击成本检查")
            return

        cfg = getattr(self.config, 'POSITION_SIZING', None) or {}
        max_impact_rate = cfg.get("max_single_impact", 0.02)
        impact_threshold = cfg.get("impact_threshold", 0.01)
        impact_base = cfg.get("impact_base", 0.002)

        total_pos = df[ColumnNames.SUGGESTED_POSITION].sum()
        # 使用 INITIAL_CASH 作为组合参考净值（实际净值每日变动，此处取近似值）
        portfolio_value = getattr(self.config, 'INITIAL_CASH', None)
        if portfolio_value is None:
            bt_cfg = getattr(self.config, 'BACKTEST', None) or {}
            portfolio_value = bt_cfg.get("initial_cash", 1_000_000)

        n_adjusted = 0
        for idx, row in df.iterrows():
            pos = row.get(ColumnNames.SUGGESTED_POSITION, 0) or 0
            if pos <= 0:
                continue
            amount = row.get(ColumnNames.AMOUNT, 0) or 0
            if amount <= 0:
                continue

            est_investment = pos * portfolio_value
            participation = est_investment / amount

            if participation > impact_threshold:
                impact = impact_base * (participation / impact_threshold) ** 1.5
                if impact > max_impact_rate:
                    safe_participation = impact_threshold * (max_impact_rate / impact_base) ** (1 / 1.5)
                    safe_pos = safe_participation * amount / portfolio_value
                    safe_pos = min(safe_pos, pos)
                    df.at[idx, ColumnNames.SUGGESTED_POSITION] = safe_pos
                    n_adjusted += 1
                    self.logger.info(
                        f"  - [Gate5/R62] {row.get(ColumnNames.STOCK_CODE, '')} "
                        f"冲击成本 {impact:.1%} > {max_impact_rate:.0%}，"
                        f"仓位 {pos:.1%} → {safe_pos:.1%}"
                    )

        if n_adjusted:
            self.logger.info(f"  - [Gate5/R62] 冲击成本超限调整 {n_adjusted} 只")

    def _apply_gate5_turnover_check(self, df: pd.DataFrame) -> None:
        """
        Gate 5: 组合换手率检查。

        日换手率 = sum(|新仓位 - 旧仓位|) / 2。由于每日复盘未跟踪昨日持仓，
        此处估算"单日建仓"场景：当日建仓总仓位 = 总仓位 / 组合杠杆上限。
        如果总仓位 > 0.5（估算换手率超限），发出告警。

        注：精确换手率需接入昨日持仓快照，当前为保守估算。
        """
        if df.empty or ColumnNames.SUGGESTED_POSITION not in df.columns:
            return

        total_pos = df[ColumnNames.SUGGESTED_POSITION].sum()
        turnover = total_pos  # 单日建仓场景下，换手率 ≈ 总仓位
        max_turnover = 0.5

        if turnover > max_turnover:
            scale = max_turnover / turnover
            df[ColumnNames.SUGGESTED_POSITION] = df[ColumnNames.SUGGESTED_POSITION] * scale
            self.logger.info(
                f"  - [Gate5/R63] 估算日换手率 {turnover:.1%} > {max_turnover:.0%}，"
                f"等比缩仓至 {max_turnover:.0%}（系数={scale:.3f}）"
            )

    def _apply_gate5_position_constraints(self, df: pd.DataFrame) -> None:
        """
        Gate 5: 运行时仓位约束。
        执行顺序：流动性冲击 → 行业集中度 → 组合换手率 → 总仓位上限。
        直接修改 df[SUGGESTED_POSITION]。
        """
        if df.empty or ColumnNames.SUGGESTED_POSITION not in df.columns:
            return

        cfg = getattr(self.config, 'POSITION_SIZING', None) or {}
        max_industry_exposure = cfg.get("max_industry_exposure", 0.30)

        # R62: 单股冲击成本检查（先执行，用原始仓位估算）
        self._apply_gate5_liquidity_impact(df)

        # R60: 同一行业集中度 > 30%，仅保留该行业评分最高者
        if ColumnNames.INDUSTRY in df.columns:
            industry_positions = df.groupby(ColumnNames.INDUSTRY)[ColumnNames.SUGGESTED_POSITION].sum()
            over_limit = industry_positions[industry_positions > max_industry_exposure]
            for ind, _ in over_limit.items():
                mask = df[ColumnNames.INDUSTRY] == ind
                if mask.sum() > 0:
                    best_idx = df.loc[mask, ColumnNames.COMPREHENSIVE_SCORE].idxmax()
                    df.loc[mask & (df.index != best_idx), ColumnNames.SUGGESTED_POSITION] = 0.0
                    df.loc[best_idx, ColumnNames.SUGGESTED_POSITION] = min(
                        df.loc[best_idx, ColumnNames.SUGGESTED_POSITION], max_industry_exposure
                    )
                    self.logger.info(
                        f"  - [Gate5/R60] 行业 {ind} 超限，仅保留 {df.loc[best_idx, ColumnNames.STOCK_CODE]}"
                    )

        # R63: 组合换手率检查
        self._apply_gate5_turnover_check(df)

        # R61: 总仓位 > 100%，等比缩仓
        total_pos = df[ColumnNames.SUGGESTED_POSITION].sum()
        if total_pos > 1.0:
            scale = 0.95 / total_pos
            df[ColumnNames.SUGGESTED_POSITION] = df[ColumnNames.SUGGESTED_POSITION] * scale
            self.logger.info(f"  - [Gate5/R61] 总仓位 {total_pos:.1%} > 100%，等比缩仓至 95%（系数={scale:.3f}）")
