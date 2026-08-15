from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config


class EventDrivenFetcher:
    """事件驱动因子获取器 — 回购 / 大股东增减持 / 分红。

    数据源：
      - akshare stock_repurchase_em              → 回购明细（金额、进度）
      - asharehub client.holder_trade()          → 股东增减持明细（逐笔 IN/DE）
      - akshare stock_fhps_detail_em             → 分红派息实施
    """

    CACHE_DIR: str | None = None

    def __init__(self, config: Config) -> None:
        self.config = config
        if hasattr(config, "TEMP_DATA_DIRECTORY"):
            self.CACHE_DIR = config.TEMP_DATA_DIRECTORY
        else:
            self.CACHE_DIR = os.path.expanduser("~/Downloads/CoreNews_Reports/cache")
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        # asharehub API key 从 Config 读取（已自动解密）
        self._asharehub_key = getattr(config, "ASHAREHUB_API_KEY", "")

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
        """获取全市场大股东增减持数据（asharehub 接口）。

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

        if not self._asharehub_key:
            logger.warning("[事件驱动] asharehub API key 未配置，跳过增减持获取")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, dtype={"symbol": str})
            return pd.DataFrame()

        try:
            from UtilsManager.AShareHubClient import make_asharehub_client
            client = make_asharehub_client(api_key=self._asharehub_key)
            df = client.holder_trade()
        except Exception as e:
            logger.warning(f"[事件驱动] asharehub 增减持获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, dtype={"symbol": str})
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # asharehub 返回列: symbol, ann_date, holder_name, holder_type, in_de,
        #                   change_vol, change_ratio, after_share, after_ratio, avg_price, ...
        # 转换 symbol 格式：000001.SZ → sz000001, 600000.SH → sh600000
        def _convert_symbol(s):
            s = str(s).strip()
            if "." in s:
                code, market = s.split(".")
                code = code.zfill(6)
                return f"sh{code}" if market.upper() == "SH" else f"sz{code}"
            # 无前缀直接补全
            s = s.zfill(6)
            if s.startswith("6"):
                return f"sh{s}"
            return f"sz{s}"

        result = df.copy()
        result["symbol"] = result["symbol"].apply(_convert_symbol)

        # 计算增减持金额（万元）：变动股数 × 成交均价 / 10000
        result["变动金额_万元"] = (
            pd.to_numeric(result.get("change_vol", 0), errors="coerce") *
            pd.to_numeric(result.get("avg_price", 0), errors="coerce")
        ) / 10000.0
        result["变动金额_万元"] = result["变动金额_万元"].fillna(0)

        # 方向：IN=增持(正)，DE=减持(负)
        result["方向"] = result.get("in_de", "").str.upper()
        result["净增减持_万元"] = np.where(
            result["方向"] == "IN",
            result["变动金额_万元"],
            -result["变动金额_万元"]
        )

        # 按股票汇总
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
            df = ak.stock_fhps_detail_em()
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
