from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config


class TopTraderFetcher:
    """龙虎榜（Top Trader）每日上榜数据获取器。

    通过 akshare 获取沪深两市每日龙虎榜上榜个股及净买入额。
    缓存策略：当日结果缓存到 CSV；再次运行直接读缓存。
    """

    SSE_API = "stock_sse_summary"
    SZSE_API = "stock_szse_summary"

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
        return os.path.join(self._cache_dir, f"top_trader_{trade_date}.csv")

    def fetch(self, trade_date: str | None = None) -> pd.DataFrame:
        """获取指定日期的龙虎榜数据。

        Returns:
            DataFrame with columns: symbol, 龙虎榜净买入_万元, 上榜次数
        """
        if trade_date is None:
            trade_date = self._today_str()

        cache_path = self._cache_path(trade_date)
        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, dtype={"symbol": str})
            logger.info(f"  龙虎榜缓存命中 [{trade_date}] ({len(cached)} 只)")
            return cached

        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装，无法获取龙虎榜数据")
            return pd.DataFrame()

        try:
            dt = datetime.strptime(trade_date, "%Y%m%d")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = trade_date

        records = []

        # 尝试上交所龙虎榜
        try:
            sse = ak.stock_sse_summary(date=date_str)
            if sse is not None and not sse.empty:
                records.append(self._parse_sse(sse))
        except Exception:
            pass

        # 尝试深交所龙虎榜
        try:
            szse = ak.stock_szse_summary(date=date_str)
            if szse is not None and not szse.empty:
                records.append(self._parse_szse(szse))
        except Exception:
            pass

        if not records:
            logger.info(f"  龙虎榜无数据 [{date_str}]")
            return pd.DataFrame()

        result = pd.concat(records, ignore_index=True)
        result = result.groupby("symbol", as_index=False).agg(
            龙虎榜净买入_万元=("净买入_万元", "sum"),
            上榜次数=("净买入_万元", "count"),
        )
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"  龙虎榜获取完成 [{trade_date}] ({len(result)} 只)")
        return result

    def _parse_sse(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析上交所龙虎榜返回格式。"""
        rows = []
        for _, r in df.iterrows():
            code = str(r.get("股票代码", r.get("code", ""))).strip().zfill(6)
            if not code or code == "000000":
                continue
            name = str(r.get("股票名称", r.get("name", "")))
            net = pd.to_numeric(r.get("净买入", r.get("net_buy", 0)), errors="coerce")
            if pd.isna(net):
                net = 0.0
            rows.append({"code": code, "name": name, "net": net})
        if not rows:
            return pd.DataFrame(columns=["symbol", "净买入_万元", "股票简称"])
        parsed = pd.DataFrame(rows)
        from UtilsManager.CodeNormalizer import CodeNormalizer
        parsed["symbol"] = parsed["code"].apply(CodeNormalizer.add_market_prefix)
        parsed["净买入_万元"] = parsed["net"]
        parsed["股票简称"] = parsed["name"]
        return parsed[["symbol", "净买入_万元", "股票简称"]]

    def _parse_szse(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析深交所龙虎榜返回格式。"""
        rows = []
        for _, r in df.iterrows():
            code = str(r.get("证券代码", r.get("code", ""))).strip().zfill(6)
            if not code or code == "000000":
                continue
            name = str(r.get("证券简称", r.get("name", "")))
            net = pd.to_numeric(r.get("买入金额", r.get("buy_amt", 0)), errors="coerce") \
                - pd.to_numeric(r.get("卖出金额", r.get("sell_amt", 0)), errors="coerce")
            if pd.isna(net):
                net = 0.0
            rows.append({"code": code, "name": name, "net": net})
        if not rows:
            return pd.DataFrame(columns=["symbol", "净买入_万元", "股票简称"])
        parsed = pd.DataFrame(rows)
        from UtilsManager.CodeNormalizer import CodeNormalizer
        parsed["symbol"] = parsed["code"].apply(CodeNormalizer.add_market_prefix)
        parsed["净买入_万元"] = parsed["net"]
        parsed["股票简称"] = parsed["name"]
        return parsed[["symbol", "净买入_万元", "股票简称"]]

    def fetch_multi_day(self, days: int = 20) -> pd.DataFrame:
        """获取最近 days 个交易日的龙虎榜数据，返回按股票聚合的统计量。"""
        from DataCollection.CalendarManager import TradingCalendarAnalyzer
        cal = TradingCalendarAnalyzer()
        all_dfs = []
        seen = set()
        for i in range(days * 2):
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
        agg = combined.groupby("symbol").agg(
            龙虎榜净买入总额=("龙虎榜净买入_万元", "sum"),
            龙虎榜净买入均值=("龙虎榜净买入_万元", "mean"),
            上榜总次数=("上榜次数", "sum"),
        ).reset_index()
        return agg
