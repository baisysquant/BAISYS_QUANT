import pandas as pd


class DataCleaner:
    """金融数据清洗工具类：处理亿、万、百分比及非法字符"""

    @staticmethod
    def parse_money_str(val):
        if pd.isna(val) or val == "" or val in ["--", "-"]:
            return 0.0
        s = str(val).strip()
        try:
            multiplier = 1.0
            if "亿" in s:
                multiplier = 100000000.0
                s = s.replace("亿", "")
            elif "万" in s:
                multiplier = 10000.0
                s = s.replace("万", "")
            elif "%" in s:
                multiplier = 0.01
                s = s.replace("%", "")
            return float(s) * multiplier
        except (ValueError, TypeError):
            return 0.0


class QuantDBSyncTask:
    """
    量化数据同步任务类
    支持联合主键: (archive_date, strategy_type, stock_code)
    """

    def __init__(self, db_manager):
        self.db = db_manager

    def sync_all(self, today_str, consolidated_report, industry_df, raw_data):
        """执行全量同步主入口

        Args:
            today_str: 业务日期字符串
            consolidated_report: 最终报告DataFrame
            industry_df: 行业分析DataFrame
            raw_data: 原始数据字典
        """
        print("\n" + "=" * 50)
        print(f">>> 启动数据库资产化同步任务 业务日期: {today_str}")
        print("=" * 50)

        try:
            # 1. 同步策略触发归档 (支持一码多策略)
            self._sync_strategy_stocks(today_str, raw_data)
            self._sync_industry_data(today_str, industry_df)
            self._sync_final_report(today_str, consolidated_report)

        except Exception as e:
            print(f"!!! [同步中断] 任务运行异常: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print(">>> 数据库同步任务结束。\n")

    def _sync_strategy_stocks(self, today, raw_data):
        """1. 同步策略触发归档 (ODS层) - 适配联合主键并修复列名KeyError"""
        strategy_frames = []
        # 配置映射：raw_data的key -> 数据库中的 strategy_type
        mapping_cfg = {
            "strong_stocks_raw": "强势股池",
            "consecutive_rise_raw": "连续上涨",
            "ljqs_raw": "量价齐升",
            "cxfl_raw": "持续放量",
        }

        for key, strat_name in mapping_cfg.items():
            df = raw_data.get(key)
            if df is not None and not df.empty:
                temp_df = df.copy()

                # --- 统一寻找代码列并重命名，防止 drop_duplicates 报错 ---
                potential_code_cols = ["代码", "证券代码", "股票代码", "code", "stock_code"]
                found_col = next((c for c in potential_code_cols if c in temp_df.columns), None)

                if found_col:
                    temp_df.rename(columns={found_col: "stock_code"}, inplace=True)
                else:
                    continue

                # 尝试统一名称列
                potential_name_cols = ["名称", "股票名称", "股票简称", "stock_name"]
                found_name_col = next((c for c in potential_name_cols if c in temp_df.columns), None)
                if found_name_col:
                    temp_df.rename(columns={found_name_col: "stock_name"}, inplace=True)

                # --- 策略内去重 ---
                temp_df = temp_df.drop_duplicates(subset=["stock_code"])

                # 对应建表语句中的 strategy_type 字段
                temp_df["strategy_type"] = strat_name

                # 这里的字段必须对应建表语句（id自增不需要管，archive_date由DatabaseWriter补齐）
                # 确定的字段：stock_code, strategy_type, stock_name
                cols = ["stock_code", "strategy_type"]
                if "stock_name" in temp_df.columns:
                    cols.append("stock_name")

                strategy_frames.append(temp_df[cols])

        if strategy_frames:
            # 合并并重置索引，解决 Reindexing 错误
            all_strat_df = pd.concat(strategy_frames, ignore_index=True)

            final_cols = ["stock_code", "strategy_type"]
            if "stock_name" in all_strat_df.columns:
                final_cols.append("stock_name")

            self.db.safe_insert_data(all_strat_df[final_cols], "ods_ak_ranking_stocks", "archive_date", today)

    def _sync_industry_data(self, today, industry_df):
        """同步行业权重趋势深度数据"""
        if industry_df is None or industry_df.empty:
            print("  - [警告] industry_df 为空，跳过行业数据同步")
            return

        df_db = industry_df.copy()

        # 必须匹配 TXT 文件的中文表头
        mapping = {
            "行业名称": "industry_name",
            "行业指数": "industry_index",
            "涨幅_now": "change_pct_now",
            "净额_now": "net_inflow_now",
            "流入资金": "total_inflow_money",
            "领涨股": "leading_stock",
            "领涨股-涨跌幅": "leading_stock_pct",
            "净额_3d": "net_inflow_3d",
            "净额_5d": "net_inflow_5d",
            "净额_10d": "net_inflow_10d",
            "净额_20d": "net_inflow_20d",
            "换手率": "turnover_rate",
            "资金分": "score_fund",
            "价格分": "score_price",
            "换手分": "score_turnover",
            "趋势得分": "score_trend",
            "行业信号": "industry_signal",
        }

        df_db.rename(columns=mapping, inplace=True)

        # 只保留上面定义的英文列
        valid_cols = list(mapping.values())
        df_db = df_db[[c for c in valid_cols if c in df_db.columns]]

        # 数值清洗（使用你已有的 DataCleaner）
        for col in df_db.columns:
            if col not in ["industry_name", "leading_stock", "industry_signal"]:
                df_db[col] = df_db[col].apply(DataCleaner.parse_money_str)

        self.db.safe_insert_data(df_db, "ods_ak_industry_analysis", "archive_date", today)

    def _sync_final_report(self, today, df):
        """3. 同步决策报告 (APP层) - 完整字段支持版

        Args:
            today: 业务日期
            df: 最终报告DataFrame
        """
        if df is None or df.empty:
            return

        df_db = df.copy()
        if df_db.index.name == "股票代码":
            df_db.reset_index(inplace=True)

        drop_list = ["index", "level_0"]
        df_db.drop(columns=[c for c in drop_list if c in df_db.columns], inplace=True)

        mapping = {
            "股票代码": "stock_code",
            "股票简称": "stock_name",
            "行业": "industry",
            "最新价": "close_price",
            "强势股": "is_strong_stock",
            "量价齐升": "is_vol_price_rise",
            "连涨天数": "consecutive_up_days",
            "放量天数": "high_vol_days",

            # MACD趋势评分列（单参数）
            "金叉信号": "macd_cross",
            "柱状动能": "macd_momentum",
            "DIF斜率": "dif_slope",
            "背离信号": "divergence_signal",
            "量价配合": "volume_price_score",
            "综合分析结论": "comprehensive_conclusion",
            "综合分析评分": "comprehensive_score",
            "综合级别": "comprehensive_level",
            "风险等级": "risk_level",
            "MACD趋势分类": "macd_trend_type",
            # 独立技术指标（水平多因子交叉验证）
            "KDJ_Signal": "kdj_signal",
            "CCI_Signal": "cci_signal",
            "RSI_Signal": "rsi_signal",
            "BOLL_Signal": "boll_signal",
            "研报买入次数": "report_buy_count",
            "资金动能": "fund_flow_trend",
            "5日资金流入": "fund_inflow_5d",
            "10日资金流入": "fund_inflow_10d",
            "20日资金流入": "fund_inflow_20d",
            "股票链接": "stock_link",
            # 行业中性化
            "所属行业信号": "industry_signal_tag",
            "行业内百分位": "industry_pctile",
            "行业信号评分": "industry_signal_score",
            "行业背离扣分": "industry_deviation",
            # 背离位置
            "背离距今": "divergence_days",
            "背离位置": "divergence_price",
            # 退出策略
            "止损价": "stop_loss",
            "T1目标价": "t1_target",
            "T2目标价": "t2_target",
            "移动止损": "trailing_stop",
            "盈亏比": "exit_rrr",
        }

        df_db.rename(columns=mapping, inplace=True)

        text_columns = [
            "stock_code",
            "stock_name",
            "industry",
            "stock_link",
            "macd_trend",
            "macd_cross",
            "macd_momentum",
            "dif_slope",
            "divergence_signal",
            "volume_price_score",
            "kline_pattern",
            "comprehensive_conclusion",
            "comprehensive_level",
            "risk_level",
            "macd_trend_type",
            "kdj_signal",
            "cci_signal",
            "rsi_signal",
            "boll_signal",
            "is_strong_stock",
            "is_vol_price_rise",
            "is_full_bullish",
            "industry_signal_tag",
        ]

        for col in df_db.columns:
            if col not in text_columns:
                df_db[col] = df_db[col].apply(DataCleaner.parse_money_str)

        int_columns = ["consecutive_up_days", "high_vol_days", "report_buy_count"]
        for col in int_columns:
            if col in df_db.columns:
                df_db[col] = pd.to_numeric(df_db[col], errors="coerce").fillna(0).astype(int)

        final_valid_cols = [v for v in mapping.values() if v in df_db.columns]

        # 【关键修复】在写入前对 stock_code 去重，防止主键冲突
        if "stock_code" in df_db.columns:
            before_count = len(df_db)
            df_db = df_db.drop_duplicates(subset=["stock_code"], keep="last")
            after_count = len(df_db)
            if before_count != after_count:
                print(f"  - [数据清洗] 发现 {before_count - after_count} 条重复股票代码，已自动去重。")

        try:
            self.db.safe_insert_data(df_db[final_valid_cols], "app_stock_strategy_report", "archive_date", today)
        except Exception as e:
            if 'UndefinedColumn' in type(e).__name__:
                col_name = str(e).split('"')[1] if '"' in str(e) else 'unknown'
                print(f"  - [WARN] DB表缺少列 '{col_name}'，跳过该列重试")
                safe_cols = [c for c in final_valid_cols if c != col_name]
                self.db.safe_insert_data(df_db[safe_cols], "app_stock_strategy_report", "archive_date", today)
            else:
                raise
