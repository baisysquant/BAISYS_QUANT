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
    """北向资金（沪深港通）个股持仓变动获取器。

    通过 akshare stock_hsgt_north_flow_em 获取每日北向资金个股净买入。
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
        """获取指定日期的北向资金个股净买入数据。

        Returns:
            DataFrame with columns: symbol, north_net_buy (万元)
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
            df: pd.DataFrame = ak.stock_hsgt_north_flow_em(symbol="北上", start_date=date_str, end_date=date_str)
        except Exception:
            logger.warning(f"akshare 北向资金接口异常（可能非交易日或无数据）[{date_str}]")
            return pd.DataFrame()

        if df.empty:
            return df

        # 解析 akshare 返回格式：列名通常含"名称""持股数量""持股市值""占流通股比例"等
        # 适配不同版本的 akshare 输出
        name_col = [c for c in df.columns if "名称" in c or "name" in c.lower()]
        buy_col = [c for c in df.columns if "净买入" in c or "net" in c.lower()]
        hold_col = [c for c in df.columns if "持股市值" in c or "market" in c.lower()]

        if not name_col:
            logger.warning("北向资金数据格式无法识别，列名: " + ", ".join(df.columns))
            return pd.DataFrame()

        result = pd.DataFrame()
        nc = name_col[0]
        result["股票简称"] = df[nc].astype(str)
        if buy_col:
            result["北向净买入_万元"] = pd.to_numeric(df[buy_col[0]], errors="coerce").fillna(0)
        else:
            result["北向净买入_万元"] = 0.0
        if hold_col:
            result["北向持股市值_万元"] = pd.to_numeric(df[hold_col[0]], errors="coerce").fillna(0)
        else:
            result["北向持股市值_万元"] = 0.0

        # 从股票简称匹配 symbol
        from UtilsManager.CodeNormalizer import CodeNormalizer
        code_map = {}
        try:
            from sqlalchemy import create_engine, text
            db_url = self.config.DATABASE_URL if hasattr(self.config, 'DATABASE_URL') else ""
            if db_url:
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    rows = conn.execute(text(
                        "SELECT stock_code, stock_name FROM stock_basic_info_sw"
                    )).fetchall()
                    for r in rows:
                        code_map[str(r[1]).strip()] = str(r[0]).strip().zfill(6)
        except Exception:
            pass

        def _to_symbol(name: str) -> str:
            name = name.strip()
            if name in code_map:
                return CodeNormalizer.add_market_prefix(code_map[name])
            # fallback: 上海主板 60xxxx / 深圳主板 00xxxx / 创业板 30xxxx
            if name.startswith("6"):
                return f"sh{name}"
            if name.startswith("0") or name.startswith("3"):
                return f"sz{name}"
            return name

        result["symbol"] = result["股票简称"].apply(_to_symbol)
        result.dropna(subset=["symbol"], inplace=True)
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"  北向资金获取完成 [{trade_date}] ({len(result)} 只)")
        return result

    def fetch_multi_day(self, days: int = 20) -> pd.DataFrame:
        """获取最近 days 个交易日的北向资金数据，返回按股票聚合的统计量。"""
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
        agg = combined.groupby("symbol").agg(
            北向净买入总额=("北向净买入_万元", "sum"),
            北向净买入均值=("北向净买入_万元", "mean"),
            北向净买入天数=("北向净买入_万元", "count"),
            北向持仓市值均值=("北向持股市值_万元", "mean"),
        ).reset_index()
        agg["北向净买入趋势"] = agg["北向净买入均值"] * agg["北向净买入天数"]
        return agg
