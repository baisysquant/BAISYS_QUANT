from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigParser import Config


def _akshare_to_ts_code(symbol: str) -> str:
    """Convert 'sh000001' / 'sz000001' to '000001.SH' / '000001.SZ'."""
    code = symbol
    suffix = ""
    if symbol.startswith("sh"):
        code = symbol[2:]
        suffix = ".SH"
    elif symbol.startswith("sz"):
        code = symbol[2:]
        suffix = ".SZ"
    elif symbol.startswith("bj"):
        code = symbol[2:]
        suffix = ".BJ"
    return code + suffix


def _ts_code_to_akshare_symbol(ts_code: str) -> str:
    """Convert '000001.SH' / '000001.SZ' to 'sh000001' / 'sz000001'."""
    code = ts_code.upper()
    for prefix, exch in [(".SH", "sh"), (".SZ", "sz"), (".BJ", "bj")]:
        if code.endswith(prefix):
            return exch + code[: -len(prefix)]
    return ts_code


class ChipDistributionFetcher:
    """全市场筹码分布数据获取器。

    通过一次 API 调用（trade_date 参数）获取全市场最新筹码快照。
    免费版每日 100 次调用充足。

    缓存策略：当天首次调用拉取 API → 保存 CSV；当天再次调用直接读取缓存。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.api_key = config.ASHAREHUB_API_KEY
        self.enabled = config.ENABLE_CHIP_DISTRIBUTION
        self._client = None
        self._cache_dir = getattr(config, 'TEMP_DATA_DIRECTORY', os.path.expanduser("~/Downloads/CoreNews_Reports/cache"))

    @staticmethod
    def _get_trading_day() -> str:
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            return TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            return datetime.now().strftime("%Y%m%d")

    @property
    def _today(self) -> str:
        if hasattr(self, '_override_today') and self._override_today:
            return self._override_today
        return self._get_trading_day()

    @property
    def _cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"chip_distribution_{self._today}.csv")

    @property
    def client(self) -> Any:  # noqa: ANN401
        if self._client is None and self.api_key:
            from asharehub import AShareHub
            self._client = AShareHub(api_key=self.api_key)
        return self._client

    def fetch_chip_data(self, symbols: list[str] | None = None, date: str | None = None) -> pd.DataFrame:
        """获取全市场筹码分布数据（最新快照），带日级缓存。

        Args:
            symbols: 保留参数，不再使用。API 全局返回所有股票。
            date: 日期字符串 YYYYMMDD，用于缓存键，默认当天。

        Returns:
            DataFrame with columns: symbol, trade_date, winner_rate, weight_avg,
                                     cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct
        """
        if date:
            self._override_today = str(date).replace("-", "")
        if not self.enabled or not self.api_key:
            logger.info("[ChipDist] 筹码分布获取未启用或 API 密钥未配置，跳过。")
            return pd.DataFrame()

        # ── 读缓存（当日已有文件） ──
        if os.path.exists(self._cache_path):
            try:
                cached = pd.read_csv(self._cache_path)
                logger.info(f"[ChipDist] 读取当日缓存: {os.path.basename(self._cache_path)} ({len(cached)} 条)")
                return cached
            except Exception as e:
                logger.info(f"[ChipDist] 缓存读取失败，将重新拉取: {e}")

        # ── 拉取 API ──
        if not self.client:
            logger.info("[ChipDist] 客户端初始化失败，跳过。")
            return pd.DataFrame()

        logger.info("[ChipDist] 正在从 AShareHub 获取全市场筹码分布数据...")

        import time as _time

        df: pd.DataFrame | None = None
        for attempt in range(3):
            try:
                df = self.client.chip_distribution(trade_date=self._today)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                if attempt < 2:
                    _time.sleep(2 ** attempt)
                    continue
                logger.warning(f"[ChipDist] 获取失败 (已重试3次): {e}")
                df = None
                break

        if df is None or df.empty:
            logger.info("[ChipDist] 未获取到任何筹码分布数据。")
            return pd.DataFrame()

        # ── 统一列名 ──
        # v2 API 返回 symbol (如 "000001.SZ")，v1 返回 ts_code；统一转为 akshare 格式 sh000001
        code_col = "symbol" if "symbol" in df.columns else "ts_code"
        df["symbol"] = df[code_col].astype(str).apply(_ts_code_to_akshare_symbol)

        logger.info(f"[ChipDist] 获取完成，共 {len(df)} 条记录")

        # ── 写缓存 ──
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            df.to_csv(self._cache_path, index=False, encoding="utf-8-sig")
            logger.info(f"[ChipDist] 已缓存至: {os.path.basename(self._cache_path)}")
        except Exception as e:
            logger.info(f"[ChipDist] 缓存写入失败: {e}")

        return df
