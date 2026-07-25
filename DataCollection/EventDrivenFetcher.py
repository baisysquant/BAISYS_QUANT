from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config


class EventDrivenFetcher:
    """事件驱动因子获取器 — 回购 / 大股东增减持 / 分红。

    用 akshare 获取：
      - stock_repurchase_em          → 回购明细（金额、进度）
      - stock_holder_trade_em        → 大股东增减持（方向、金额）
      - stock_dividends_em           → 分红实施（每股分红）
    """

    CACHE_DIR: str | None = None

    def __init__(self, config: Config) -> None:
        self.config = config
        if hasattr(config, "TEMP_DATA_DIRECTORY"):
            self.CACHE_DIR = config.TEMP_DATA_DIRECTORY
        else:
            self.CACHE_DIR = os.path.expanduser("~/Downloads/CoreNews_Reports/cache")
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def fetch_repurchases(self) -> pd.DataFrame:
        """获取全市场回购数据。

        Returns:
            DataFrame with columns: symbol, 回购金额_万元, 回购进度
        """
        cache_path = os.path.join(self.CACHE_DIR, "event_repurchase.csv")
        if os.path.exists(cache_path):
            modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - modified).days < 1:
                cached = pd.read_csv(cache_path, dtype={"symbol": str})
                logger.info(f"[事件驱动] 回购缓存命中 ({len(cached)} 条)")
                return cached

        try:
            import akshare as ak
            df = ak.stock_repurchase_em()
        except Exception as e:
            logger.warning(f"[事件驱动] akshare 回购获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, dtype={"symbol": str})
            return pd.DataFrame()

        if df.empty:
            return df

        code_col = [c for c in df.columns if "代码" in c or "code" in c.lower()]
        amt_col = [c for c in df.columns if "金额" in c or "amount" in c.lower()]
        prog_col = [c for c in df.columns if "进度" in c or "progress" in c.lower() or "状态" in c]

        result = pd.DataFrame()
        if code_col:
            result["symbol"] = df[code_col[0]].astype(str).str.strip().str.zfill(6)
        else:
            return pd.DataFrame()

        result["回购金额_万元"] = pd.to_numeric(df[amt_col[0]], errors="coerce").fillna(0) if amt_col else 0.0
        result["回购进度"] = df[prog_col[0]].astype(str) if prog_col else ""

        # 只想已完成/正在进行的有效回购
        result["回购有效"] = result["回购进度"].apply(
            lambda s: 1 if any(k in s for k in ["完成", "实施", "进行", "股东提议"]) else 0
        )
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"[事件驱动] 回购获取完成 ({len(result)} 条)")
        return result

    def fetch_holder_trades(self) -> pd.DataFrame:
        """获取全市场大股东增减持数据。

        Returns:
            DataFrame with columns: symbol, 净增减持_万元（正=增持，负=减持）
        """
        cache_path = os.path.join(self.CACHE_DIR, "event_holder_trade.csv")
        if os.path.exists(cache_path):
            modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - modified).days < 1:
                cached = pd.read_csv(cache_path, dtype={"symbol": str})
                logger.info(f"[事件驱动] 增减持缓存命中 ({len(cached)} 条)")
                return cached

        try:
            import akshare as ak
            df = ak.stock_holder_trade_em()
        except Exception as e:
            logger.warning(f"[事件驱动] akshare 增减持获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, dtype={"symbol": str})
            return pd.DataFrame()

        if df.empty:
            return df

        code_col = [c for c in df.columns if "代码" in c or "code" in c.lower()]
        direction_col = [c for c in df.columns if "方向" in c or "type" in c.lower() or "买卖" in c]
        amt_col = [c for c in df.columns if "金额" in c or "amount" in c.lower() or "市值" in c]

        result = pd.DataFrame()
        if code_col:
            result["symbol"] = df[code_col[0]].astype(str).str.strip().str.zfill(6)
        else:
            return pd.DataFrame()

        if direction_col and amt_col:
            _amt = pd.to_numeric(df[amt_col[0]], errors="coerce").fillna(0)
            _dir = df[direction_col[0]].astype(str)
            # 增持为正，减持为负
            _is_buy = _dir.apply(lambda s: "增" in s or "买" in s)
            result["净增减持_万元"] = (_amt * _is_buy.map({True: 1, False: -1})).fillna(0)
        else:
            result["净增减持_万元"] = 0.0

        result = result.groupby("symbol", as_index=False)["净增减持_万元"].sum()
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"[事件驱动] 增减持获取完成 ({len(result)} 条)")
        return result

    def fetch_dividends(self) -> pd.DataFrame:
        """获取全市场分红数据。

        Returns:
            DataFrame with columns: symbol, 每股分红_元
        """
        cache_path = os.path.join(self.CACHE_DIR, "event_dividends.csv")
        if os.path.exists(cache_path):
            modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - modified).days < 1:
                cached = pd.read_csv(cache_path, dtype={"symbol": str})
                logger.info(f"[事件驱动] 分红缓存命中 ({len(cached)} 条)")
                return cached

        try:
            import akshare as ak
            df = ak.stock_dividends_em()
        except Exception as e:
            logger.warning(f"[事件驱动] akshare 分红获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, dtype={"symbol": str})
            return pd.DataFrame()

        if df.empty:
            return df

        code_col = [c for c in df.columns if "代码" in c or "code" in c.lower()]
        dividend_col = [c for c in df.columns if "分红" in c or "dividend" in c.lower() or "送转" in c]

        result = pd.DataFrame()
        if code_col:
            result["symbol"] = df[code_col[0]].astype(str).str.strip().str.zfill(6)
        else:
            return pd.DataFrame()

        if dividend_col:
            result["每股分红_元"] = pd.to_numeric(df[dividend_col[0]], errors="coerce").fillna(0)
        else:
            result["每股分红_元"] = 0.0

        result = result.groupby("symbol", as_index=False)["每股分红_元"].sum()
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"[事件驱动] 分红获取完成 ({len(result)} 条)")
        return result

    def fetch_all(self) -> pd.DataFrame:
        """获取所有事件数据，合并为综合事件驱动分。

        Returns:
            DataFrame with columns: symbol, 事件驱动总分
        """
        rep = self.fetch_repurchases()
        hol = self.fetch_holder_trades()
        div = self.fetch_dividends()

        scores = pd.DataFrame()

        if not rep.empty:
            _r = rep[rep["回购有效"] > 0].copy()
            if not _r.empty:
                _r["事件分"] = _r["回购金额_万元"] / (_r["回购金额_万元"].max() + 1)
                scores = _r[["symbol", "事件分"]].copy()

        if not hol.empty:
            _h = hol.copy()
            _h["事件分"] = _h["净增减持_万元"] / (_h["净增减持_万元"].abs().max() + 1)
            if scores.empty:
                scores = _h[["symbol", "事件分"]].copy()
            else:
                scores = pd.concat([scores, _h[["symbol", "事件分"]]], ignore_index=True)

        if not div.empty:
            _d = div.copy()
            _d["事件分"] = _d["每股分红_元"] / (_d["每股分红_元"].max() + 1)
            if scores.empty:
                scores = _d[["symbol", "事件分"]].copy()
            else:
                scores = pd.concat([scores, _d[["symbol", "事件分"]]], ignore_index=True)

        if scores.empty:
            return pd.DataFrame()

        agg = scores.groupby("symbol", as_index=False)["事件分"].sum()
        # 归一化到 [0, 1]
        mx = agg["事件分"].abs().max()
        if mx > 0:
            agg["事件驱动总分"] = agg["事件分"] / mx
        else:
            agg["事件驱动总分"] = 0.0
        return agg[["symbol", "事件驱动总分"]]
