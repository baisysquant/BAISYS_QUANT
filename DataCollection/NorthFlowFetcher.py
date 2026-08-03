from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config


class NorthFlowFetcher:
    """北向资金（沪深港通）资金流向汇总获取器。

    通过 akshare stock_hsgt_fund_flow_summary_em 获取每日北向资金汇总数据（沪股通/深股通）。
    缓存策略：当日结果缓存到 CSV；再次运行直接读缓存。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache_dir = getattr(config, 'TEMP_DATA_DIRECTORY',
                                   os.path.expanduser("~/Downloads/CoreNews_Reports/cache"))
        os.makedirs(self._cache_dir, exist_ok=True)

    def _today_str(self) -> str:
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            return TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            return datetime.now().strftime("%Y%m%d")

    def _cache_path(self, trade_date: str) -> str:
        return os.path.join(self._cache_dir, f"north_flow_{trade_date}.csv")

    def fetch(self, trade_date: str | None = None) -> pd.DataFrame:
        """获取指定日期的北向资金汇总数据（沪股通/深股通）。

        Returns:
            DataFrame with columns: 交易日, 板块, 交易状态, 成交净买额, 资金净流入, ...
        """
        if trade_date is None:
            trade_date = self._today_str()

        cache_path = self._cache_path(trade_date)
        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, dtype={"symbol": str})
            logger.info(f"  北向资金缓存命中 [{trade_date}] ({len(cached)} 只)")
            return cached

        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装，无法获取北向资金数据")
            return pd.DataFrame()

        try:
            dt = datetime.strptime(trade_date, "%Y%m%d")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = trade_date

        try:
            df: pd.DataFrame = ak.stock_hsgt_fund_flow_summary_em()
        except Exception:
            logger.warning(f"akshare 北向资金接口异常（可能非交易日或无数据）[{date_str}]")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # 过滤指定日期和北向资金
        df = df[df["交易日"] == date_str].copy()
        df = df[df["资金方向"] == "北向"]

        if df.empty:
            logger.info(f"  北向资金接口返回空数据（可能非交易日）[{date_str}]")
            return pd.DataFrame()

        # 保留关键字段
        result = df[["板块", "交易状态", "成交净买额", "资金净流入", "当日资金余额",
                      "上涨数", "持平数", "下跌数", "相关指数", "指数涨跌幅"]].copy()
        result["交易日"] = date_str
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"  北向资金汇总获取完成 [{trade_date}] ({len(result)} 条)")
        return result

    def fetch_multi_day(self, days: int = 20) -> pd.DataFrame:
        """获取最近 days 个交易日的北向资金汇总数据，按板块聚合。"""
        from DataCollection.CalendarManager import TradingCalendarAnalyzer
        cal = TradingCalendarAnalyzer()
        all_dfs = []
        seen = set()
        for i in range(days * 2):  # 最多尝试 days*2 天
            d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            if not cal.is_trading_day(d):
                continue
            if d in seen:
                continue
            seen.add(d)
            df = self.fetch(d)
            if not df.empty:
                all_dfs.append(df)
            if len(seen) >= days:
                break
        if not all_dfs:
            return pd.DataFrame()
        combined = pd.concat(all_dfs, ignore_index=True)

        # 按板块聚合多日资金流
        agg_cols = ["成交净买额", "资金净流入", "当日资金余额", "上涨数", "持平数", "下跌数"]
        agg_dict = {col: "sum" for col in agg_cols if col in combined.columns}
        if "指数涨跌幅" in combined.columns:
            agg_dict["指数涨跌幅"] = "mean"

        if agg_dict:
            agg = combined.groupby("板块").agg(agg_dict).reset_index()
            agg["交易日_最新"] = combined["交易日"].max()
            return agg
        return combined
