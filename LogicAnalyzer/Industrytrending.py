import os
import time
from typing import Any

import any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ConfigParser import Config
from loguru import logger
import numpy as np


class IndustryTrendingSafeProcessor:
    """完整的行业趋势与评分保护模块，包含所有异常检测与最终输出要求提取"""

    def __init__(self, config: Config, today_str: str, **kwargs) -> None:
        self.config = config
        self.today_str = today_str
        self.kwargs = kwargs
        self.matrix_columns = {
            "code": "行业代码",
            "name": "行业名称",
            "date": "日期",
            "close": "收盘",
            "volume": "成交量",
            "amount": "成交额",
            "ma_20": "MA20",
            "ma_60": "MA60",
            "pe_ttm": "PE_TTM",
            "pb": "PB",
            "div_yield": "股息率",
        }
        self.missing_cols = []
        self.input_df_hist = pd.DataFrame()
        self.input_df_val = pd.DataFrame()
        # 加载任何基本面数据如 valuation file
        self.input_df_time = pd.DataFrame()

    def prepare(self) -> None:
        """准备所有输入（强制 reads 避免 $.query 数据）"""
        self.input_df_hist = self._safe_read(self.config.HIST_CACHE_PATH, today_str=self.today_str)
        self.input_df_val = self._safe_read(self.config.VAL_FILE_PATH, today_str=self.today_str)
        self.input_df_time = self._safe_read(self.config.TIME_FILE_PATH, today_str=self.today_str)

        # 确保所有传入的 Dataframe 有效
        self.input_df_hist = self._auto_extract(
            self.input_df_hist, prepend_col_prefix=False, time_col="date", strict=True
        )
        self.input_df_val = self._auto_extract(
            self.input_df_val, prepend_col_prefix=False, time_col="date", strict=True
        )

    def _auto_extract(
        self, df: pd.DataFrame, prepend_col_prefix: bool = False, time_col: str = "date", strict: bool = True
    ) -> pd.DataFrame:
        """自动提取需要的列：如 'code', 'name', 'close', etc."""
        required_cols = list(self.matrix_columns.keys())

        if strict and not all(c in df.columns for c in required_cols):
            logger.warning(f"缺少关键列: {required_cols}，用默认值填充数据")
            df = pd.DataFrame(columns=required_cols)

        # 对每一 column 做类型检查（强制处理）
        if prepend_col_prefix:
            df.rename(columns={col: f"{self.matrix_columns[col]}" for col in required_cols}, inplace=True)
        else:
            df.rename(
                columns={col: self.matrix_columns[col] for col in required_cols if col in df.columns}, inplace=True
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df

    def _safe_read(self, filename: str, today_str: str | None = None, **kwargs) -> pd.DataFrame:
        """安全读取 DataFrame，不返回 None 值，而是 empty"""
        if not os.path.exists(filename):
            logger.warning(f"{filename} 文件不存在，返回空 DataFrame")
            return pd.DataFrame()

        try:
            if filename.endswith(".parquet"):
                df = pd.read_parquet(filename, **kwargs)
            elif filename.endswith(".csv"):
                df = pd.read_csv(filename, **kwargs)
            else:
                logger.error(f"不支持的文件格式: {filename}。")
                return pd.DataFrame()
            logger.info(f"成功读取 {filename}：{len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"读取 {filename} 遇到错误: {e}")
            return pd.DataFrame()

    def execute(self) -> None:
        """运行转型 & 计算，并完全打印输出"""
        logger.info("🚀 启动安全行业分析器...")
        self.prepare()

        # 确保 DEA1 有所有必要列
        if self.input_df_hist.empty or self.input_df_val.empty:
            logger.warning("输入 DataFrame 空值，分析无法继续")
            print("空行业数据，无法生成评分。")
            return

        logger.info(
            f"准备因子计算：累计 {len(self.input_df_hist)} 行，{len(self.input_df_hist['code'].unique())} 个行业。"
        )

        # 并发处理（2个线程）
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for i in range(len(self.input_df_hist)):
                code = self.input_df_hist.iloc[i]["code"]
                name = self.input_df_hist.iloc[i]["name"]
                futures[
                    executor.submit(
                        self._process_one, self.input_df_hist.iloc[i], self.input_df_val, self.input_df_time
                    )
                ] = (code, name)

            # 关键流程需加 and 知道混淆信息（不失败则放入 liczba）
            for future in as_completed(futures):
                code, name = futures[future]
                if future.exception():
                    logger.error(f"{code} 行业分析失败。")
                    continue
                result = future.result()
                result["date"] = self.input_df_hist.iloc[i]["date"]
                print(f"行业 '{name}' 评分完成")
                print("  📊 评分结构如下：")
                print(result.head(2).to_string())

    def _process_one(self, df_hist: pd.Series, df_val: pd.DataFrame, df_time: pd.DataFrame) -> Any:
        """单行业平行处理（确保不发生任何错误）"""
        try:
            code = df_hist["code"].astype(str)
            name = df_hist["name"].astype(str)

            # 验证 val 数据
            val_row = df_val[df_val["code"] == code]
            if val_row.empty:
                val_row = df_val[df_val["code"] == code]
                if val_row.empty:
                    val_row = pd.DataFrame(columns=["code", "pe_ttm", "pb", "div_yield"])
                else:
                    val_row = val_row[["pe_ttm", "pb", "div_yield"]]
                    val_row["code"] = code

            # 验证时间数据
            time_row = df_time[df_time["code"] == code]
            if time_row.empty:
                time_row = pd.DataFrame(columns=["code", "ma_20", "ma_60", "vol_ma_20", "amt_ma_60"])
                time_row["code"] = code

            # 安全地装配 DataFrame & 计算
            df = pd.DataFrame([{"code": code, "name": name}], columns=["code", "name"])
            df = pd.merge(df, val_row, on="code", how="left")
            df = pd.merge(df, time_row, on="code", how="left")

            # 合并历史数据（保持 row 删除损失，确保每行只出现一次）
            df_hist_merged = df.merge(df_hist, on="code", how="left")
            df_hist_merged.drop_duplicates(inplace=True)

            # 安全计算各因子
            df_hist_merged["ma_20"] = df_hist_merged["ma_20"].fillna(np.nan)
            df_hist_merged["ma_60"] = df_hist_merged["ma_60"].fillna(np.nan)

            # 设定权力机制，如 trait+curl
            df_hist_merged["bull_signal"] = df_hist_merged["ma_10"].apply(
                lambda x: 1 if x > df_hist_merged["ma_20"] else 0
            )
            df_hist_merged["bull_signal"] += df_hist_merged["ma_20"].apply(
                lambda x: 1 if x > df_hist_merged["ma_30"] else 0
            )
            df_hist_merged["bull_signal"] += df_hist_merged["ma_30"].apply(
                lambda x: 1 if x > df_hist_merged["ma_60"] else 0
            )
            df_hist_merged["bull_signal"] += df_hist_merged["ma_60"].apply(
                lambda x: 1 if x > df_hist_merged["ma_90"] else 0
            )
            df_hist_merged["bull_signal"] = df_hist_merged["bull_signal"].astype(int)  # 确保整数计数

            # 偏离率计算
            df_hist_merged["dev_20"] = (
                (df_hist_merged["close"] - df_hist_merged["ma_20"]) / df_hist_merged["ma_20"] * 100
            )
            df_hist_merged["dev_60"] = (
                (df_hist_merged["close"] - df_hist_merged["ma_60"]) / df_hist_merged["ma_60"] * 100
            )
            df_hist_merged["dev_90"] = (
                (df_hist_merged["close"] - df_hist_merged["ma_90"]) / df_hist_merged["ma_90"] * 100
            )

            # 量价放大器
            df_hist_merged["vol_ratio"] = df_hist_merged["volume"] / df_hist_merged["vol_ma_20"]
            df_hist_merged["amt_ratio"] = df_hist_merged["amount"] / df_hist_merged["amt_ma_60"]

            # 自动映射所有列，保持安全输出
            df_hist_merged = self._safe_forward_mapping(df_hist_merged)

            # 实现三分法：估值得分、趋势得分、量价得分 → 综合得分
            df_hist_merged["PE_TTM"] = pd.to_numeric(df_hist_merged["PE_TTM"], errors="coerce")
            df_hist_merged["PB"] = pd.to_numeric(df_hist_merged["PB"], errors="coerce")
            df_hist_merged["股息率"] = pd.to_numeric(df_hist_merged["股息率"], errors="coerce")

            df_hist_merged["score_pe"] = 100 - (df_hist_merged["PE_TTM"].rank(pct=True) * 100)
            df_hist_merged["score_pb"] = 100 - (df_hist_merged["PB"].rank(pct=True) * 100)
            df_hist_merged["score_div"] = df_hist_merged["股息率"].rank(pct=True) * 100

            df_hist_merged["factor_value"] = (
                df_hist_merged["score_pe"] * 0.4 + df_hist_merged["score_pb"] * 0.3 + df_hist_merged["score_div"] * 0.3
            )

            # 确保 MA 分量存在时才赋值
            if "ma_20" in df_hist_merged.columns:
                df_hist_merged["score_bull"] = df_hist_merged["bull_signal"] / 4 * 100
            else:
                df_hist_merged["score_bull"] = np.nan

            df_hist_merged["score_mom"] = df_hist_merged["dev_60"].rank(pct=True) * 100
            df_hist_merged["factor_trend"] = df_hist_merged["score_bull"] * 0.5 + df_hist_merged["score_mom"] * 0.5

            df_hist_merged["score_vol"] = df_hist_merged["vol_ratio"].rank(pct=True) * 100
            df_hist_merged["score_amt"] = df_hist_merged["amt_ratio"].rank(pct=True) * 100
            df_hist_merged["factor_volume"] = df_hist_merged["score_vol"] * 0.5 + df_hist_merged["score_amt"] * 0.5

            df_hist_merged["total_score"] = (
                df_hist_merged["factor_value"].fillna(50) * 0.35  # 填充 NaN
                + df_hist_merged["factor_trend"] * 0.40  # trend 权重
                + df_hist_merged["factor_volume"] * 0.25
            ).round(2)

            # 打行业中分类号
            def get_signal(row: pd.Series) -> str:
                if np.isnan(row["total_score"]):
                    return "无评分，数据缺失"
                if row["total_score"] > 75 and row["factor_value"] > 70:
                    return "核心配置 (低估值+强趋势)"
                elif row["total_score"] > 70 and row["factor_trend"] > 80:
                    return "动量追击 (高景气+资金涌入)"
                elif row["factor_value"] > 85 and row["factor_trend"] < 40:
                    return "左侧潜伏 (极度低估+等待拐点)"
                elif row["factor_trend"] > 80 and row["factor_value"] < 30:
                    return "情绪过热 (高估+趋势透支)"
                else:
                    return "均衡/观望"

            df_hist_merged["行业信号"] = df_hist_merged.apply(get_signal, axis=1)

            return df_hist_merged[["行业名称", "行业信号", "总得分", "估值得分", "趋势得分", "量价得分"]]

        except Exception as e:
            logger.error(f"{code} 行业打分失败: {e}")
            return pd.DataFrame()

    def _safe_forward_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保我们有一个最终无障碍的映射结构（输入为 raw DataFrame）"""
        datetime_cols = [col for col in df.columns if col in ["date", "game_date"]]
        if datetime_cols:
            df[datetime_cols] = df[datetime_cols].astype("datetime64[ns]")

        df = df.copy()
        mapped_df = pd.DataFrame()

        # 给估值列合并且属于`assessment`任务
        mapped_df = df[
            [
                "code",
                "name",
                "date",
                "close",
                "volume",
                "amount",
                "ma_20",
                "ma_30",
                "ma_60",
                "ma_90",
                "vol_ma_20",
                "amt_ma_60",
            ]
        ]
        mapped_df[" PE_TTM"] = df["PE_TTM"].fillna(0)
        mapped_df[" PB"] = df["PB"].fillna(0)
        mapped_df[" 股息率"] = df["股息率"].fillna(0)
        mapped_df[" 行业信号"] = df["行业信号"].fillna("均衡/观望")

        return mapped_df


class IndustryTrendingManager:
    """完全完整的流程管理器，能打印所有流程最终结果"""

    def __init__(self, config: Config, today_str: str, **kwargs) -> None:
        self.config = config
        self.today_str = today_str
        self.kwargs = kwargs
        self.output_file_path = os.path.join(self.config.OUTPUT_DIR, f"industry_trend_report_{today_str}.csv")

    def run(self) -> None:
        tproc = IndustryTrendingSafeProcessor(config=self.config, today_str=self.today_str, **self.kwargs)
        tproc.execute()

        final_df = tproc.get_final_dataframe()  # 确保所有维度存在
        if not final_df.empty:
            final_df.to_csv(self.output_file_path, index=False)
            logger.info(f"行业分析完成，输出文件路径: {self.output_file_path}")
            print(f"安全打印 ( final_df head (2) )\n{final_df.head(2).to_string()}")  # 可视化形成最终 guarding
        else:
            logger.warning("无法生成完整行业评分报告。")
            print("未找到可用行业数据。")

    def get_final_dataframe(self) -> pd.DataFrame:
        tproc = IndustryTrendingSafeProcessor(config=self.config, today_str=self.today_str, **self.kwargs)
        return tproc._process_one()
