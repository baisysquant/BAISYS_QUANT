from __future__ import annotations

import os
from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger


class MainCostDataManager:
    """
    主力成本数据管理类
    提供主力成本、机构参与度等相关数据的获取、分析和管理功能
    """

    def __init__(self, cache_enabled: bool = True, cache_dir: str | None = None) -> None:
        if cache_dir is None:
            try:
                from ConfigParser import Config
                cache_dir = os.path.join(Config().CACHE_DIRECTORY, "cost_data_cache")
            except Exception:
                cache_dir = os.path.expanduser("~/Downloads/CoreNews_Reports/cache/cost_data_cache")
        """
        初始化主力成本数据管理器

        Args:
            cache_enabled: 是否启用缓存
            cache_dir: 缓存目录
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        if cache_enabled and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _today_str(self) -> str:
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            return TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            from datetime import datetime
            return datetime.now().strftime("%Y%m%d")

    def get_main_cost_data(self) -> pd.DataFrame:
        """
        获取主力成本数据

        Returns:
            DataFrame: 包含代码、主力成本、机构参与度等字段的数据
        """
        cache_file = (
            os.path.join(self.cache_dir, f"main_cost_data_{self._today_str()}.csv")
            if self.cache_enabled
            else None
        )

        # 检查缓存
        if self.cache_enabled and cache_file and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"从缓存加载主力成本数据: {cache_file}")
                return df
            except Exception as e:
                logger.error(f"读取缓存失败: {e}")

        # 获取数据
        logger.info("正在获取主力成本数据...")
        df = ak.stock_comment_em()

        # 缓存数据
        if self.cache_enabled and cache_file:
            try:
                df.to_csv(cache_file, index=False)
                logger.info(f"主力成本数据已缓存: {cache_file}")
            except Exception as e:
                logger.error(f"缓存数据失败: {e}")

        return df

    def analyze_cost_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对主力成本数据进行进一步分析

        Args:
            df: 包含主力成本数据的DataFrame

        Returns:
            DataFrame: 包含额外分析字段的数据
        """
        if df is None or df.empty:
            return df

        # 创建副本以避免修改原数据
        result_df = df.copy()

        # 检查必要的列是否存在
        if "主力成本" in result_df.columns and "最新价" in result_df.columns:
            # 计算主力成本与当前价格的差价
            result_df["主力成本差价"] = result_df["最新价"] - result_df["主力成本"]

            # 计算主力成本与当前价格的百分比差异
            result_df["主力成本差价百分比"] = (
                (result_df["最新价"] - result_df["主力成本"]) / result_df["主力成本"]
            ) * 100

            # 判断当前价格相对于主力成本的位置
            def cost_position(row: pd.Series) -> str:
                if row["最新价"] > row["主力成本"]:
                    if row["主力成本差价百分比"] > 10:
                        return "大幅高于成本"
                    else:
                        return "略高于成本"
                elif row["最新价"] < row["主力成本"]:
                    if row["主力成本差价百分比"] < -10:
                        return "大幅低于成本"
                    else:
                        return "略低于成本"
                else:
                    return "等于成本"

            result_df["成本位置"] = result_df.apply(cost_position, axis=1)

        # 分析机构参与度
        if "机构参与度" in result_df.columns:
            # 创建机构参与度等级
            result_df["机构参与度等级"] = pd.cut(
                result_df["机构参与度"],
                bins=[-1, 20, 50, 80, 101],
                labels=["低", "中低", "中高", "高"],
                include_lowest=True,
            ).astype(str)

        # 添加主力控盘强度评估
        if "主力成本" in result_df.columns and "机构参与度" in result_df.columns:

            def control_strength(row: pd.Series) -> str:
                if row["机构参与度"] >= 80 and abs(row["主力成本差价百分比"]) <= 10:
                    return "高度控盘"
                elif row["机构参与度"] >= 50 and abs(row["主力成本差价百分比"]) <= 15:
                    return "中度控盘"
                elif row["机构参与度"] >= 20:
                    return "轻度控盘"
                else:
                    return "低度控盘"

            result_df["主力控盘强度"] = result_df.apply(control_strength, axis=1)

        return result_df

    def get_stock_cost_info(self, stock_code: str) -> dict[str, Any] | None:
        """
        获取指定股票的主力成本信息

        Args:
            stock_code: 股票代码

        Returns:
            Dict: 股票的主力成本相关信息
        """
        df = self.get_main_cost_data()
        if df is None or df.empty:
            return None

        # 格式化股票代码进行匹配
        formatted_code = str(stock_code).zfill(6)
        stock_data = df[df["代码"].astype(str).str.zfill(6) == formatted_code]

        if stock_data.empty:
            return None

        # 取第一条记录
        record = stock_data.iloc[0].to_dict()

        return {
            "代码": record.get("代码"),
            "名称": record.get("名称"),
            "最新价": record.get("最新价"),
            "主力成本": record.get("主力成本"),
            "机构参与度": record.get("机构参与度"),
            "主力成本差价": record.get("主力成本差价") if "主力成本差价" in record else None,
            "主力成本差价百分比": record.get("主力成本差价百分比") if "主力成本差价百分比" in record else None,
            "成本位置": record.get("成本位置") if "成本位置" in record else None,
            "机构参与度等级": record.get("机构参与度等级") if "机构参与度等级" in record else None,
            "主力控盘强度": record.get("主力控盘强度") if "主力控盘强度" in record else None,
        }

    def filter_by_cost_criteria(
        self,
        df: pd.DataFrame,
        cost_diff_threshold: float = 0.0,
        participation_threshold: float = 0.0,
        cost_position: str | None = None,
    ) -> pd.DataFrame:
        """
        根据主力成本相关条件筛选股票

        Args:
            df: 包含主力成本数据的DataFrame
            cost_diff_threshold: 主力成本差价百分比阈值
            participation_threshold: 机构参与度阈值
            cost_position: 成本位置筛选条件

        Returns:
            DataFrame: 筛选后的数据
        """
        if df is None or df.empty:
            return df

        result_df = df.copy()

        # 按成本差价筛选
        if "主力成本差价百分比" in result_df.columns:
            result_df = result_df[result_df["主力成本差价百分比"] >= cost_diff_threshold]

        # 按机构参与度筛选
        if "机构参与度" in result_df.columns:
            result_df = result_df[result_df["机构参与度"] >= participation_threshold]

        # 按成本位置筛选
        if cost_position and "成本位置" in result_df.columns:
            result_df = result_df[result_df["成本位置"] == cost_position]

        return result_df

    def print_cost_summary(self, df: pd.DataFrame) -> None:
        """
        打印主力成本数据摘要信息

        Args:
            df: 包含主力成本数据的DataFrame
        """
        if df is None or df.empty:
            logger.warning("主力成本数据为空")
            return

        logger.info("主力成本数据分析摘要")

        # 基础统计
        logger.info(f"总股票数量: {len(df)}")

        if "主力成本" in df.columns:
            logger.info(f"主力成本有效数量: {df['主力成本'].notna().sum()}")
            logger.info(f"主力成本平均值: {df['主力成本'].mean():.2f}")
            logger.info(f"主力成本中位数: {df['主力成本'].median():.2f}")
            logger.info(f"主力成本最高值: {df['主力成本'].max():.2f}")
            logger.info(f"主力成本最低值: {df['主力成本'].min():.2f}")

        # 成本位置分布
        if "成本位置" in df.columns:
            logger.info("\n成本位置分布:")
            position_counts = df["成本位置"].value_counts()
            for pos, count in position_counts.items():
                logger.info(f"  {pos}: {count} 只 ({count / len(df) * 100:.1f}%)")

        # 机构参与度等级分布
        if "机构参与度等级" in df.columns:
            logger.info("\n机构参与度等级分布:")
            level_counts = df["机构参与度等级"].value_counts()
            for level, count in level_counts.items():
                logger.info(f"  {level}: {count} 只 ({count / len(df) * 100:.1f}%)")

        # 主力控盘强度分布
        if "主力控盘强度" in df.columns:
            logger.info("\n主力控盘强度分布:")
            strength_counts = df["主力控盘强度"].value_counts()
            for strength, count in strength_counts.items():
                logger.info(f"  {strength}: {count} 只 ({count / len(df) * 100:.1f}%)")
