from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigParser import Config


class FinancialForecastFetcher:
    """全市场业绩预告获取器。

    通过 AShareHub /v1/financials/forecast 获取，不传 ts_code 即全市场。
    缓存策略：当日首次分页拉取 → 写 CSV 缓存；当日再次运行直接读缓存。
    """

    API_PAGE_SIZE = 1000

    def __init__(self, config: Config) -> None:
        self.config = config
        if hasattr(config, 'ASHAREHUB_API_KEY'):
            self.api_key = config.ASHAREHUB_API_KEY
        else:
            self.api_key = None
        if hasattr(config, 'TEMP_DATA_DIRECTORY'):
            self._cache_dir = config.TEMP_DATA_DIRECTORY
        else:
            self._cache_dir = os.path.expanduser("~/Downloads/CoreNews_Reports")
        self._client = None

    @property
    def _today(self) -> str:
        if hasattr(self, '_override_today') and self._override_today:
            return self._override_today
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            return TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            from datetime import datetime
            return datetime.now().strftime("%Y%m%d")

    @property
    def _cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"forecast_{self._today}.csv")

    @property
    def client(self) -> Any:  # noqa: ANN401
        if self._client is None and self.api_key:
            from asharehub import AShareHub
            self._client = AShareHub(api_key=self.api_key)
        return self._client

    def fetch_all(self, date: str | None = None) -> pd.DataFrame:
        """获取全市场业绩预告数据，带日级缓存。

        Args:
            date: 日期字符串 YYYYMMDD，默认当天（用于判断缓存）。

        Returns:
            DataFrame，包含 type, p_change_min, p_change_max,
            net_profit_min, net_profit_max, summary, change_reason 等字段。
        """
        if not self.api_key:
            logger.info("[Forecast] API 密钥未配置，跳过。")
            return pd.DataFrame()

        if date:
            target_date = str(date).replace("-", "")
            self._override_today = target_date
        else:
            target_date = self._today
        cache_path = os.path.join(self._cache_dir, f"forecast_{target_date}.csv")

        if target_date == self._today and os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path)
                logger.info(f"[Forecast] 读取当日缓存: {os.path.basename(cache_path)} ({len(cached)} 条)")
                return cached
            except Exception as e:
                logger.info(f"[Forecast] 缓存读取失败，将重新拉取: {e}")

        if not self.client:
            logger.info("[Forecast] 客户端初始化失败，跳过。")
            return pd.DataFrame()

        all_dfs = []
        offset = 0
        page = 1
        logger.info("[Forecast] 正在从 AShareHub 获取全市场业绩预告...")

        while True:
            try:
                df = self.client.forecast(limit=self.API_PAGE_SIZE, offset=offset)
                if df is None or df.empty:
                    break
                all_dfs.append(df)
                row_count = len(df)
                logger.info(f"  [业绩预告分页 {page}] offset={offset}, 返回 {row_count} 行")
                if row_count < self.API_PAGE_SIZE:
                    break
                offset += row_count
                page += 1
            except Exception as e:
                logger.info(f"[Forecast] 获取失败: {e}")
                break

        if not all_dfs:
            logger.info("[Forecast] 未获取到任何业绩预告数据。")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"[Forecast] 获取完成，共 {len(combined)} 条记录（{page-1} 页）")

        if target_date == self._today:
            try:
                os.makedirs(self._cache_dir, exist_ok=True)
                combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
                logger.info(f"[Forecast] 已缓存至: {os.path.basename(cache_path)}")
            except Exception as e:
                logger.info(f"[Forecast] 缓存写入失败: {e}")

        return combined
