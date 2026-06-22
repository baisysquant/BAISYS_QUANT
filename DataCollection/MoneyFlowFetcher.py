from __future__ import annotations

import os
import sys
import time
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigParser import Config


class MoneyFlowFetcher:
    """全市场资金流向获取器（按订单大小分类：小/中/大/特大单）。

    通过 AShareHub /v1/flows/moneyflow 获取，不传 ts_code 即全市场。
    缓存策略：当日首次分页拉取 → 写 CSV 缓存；当日再次运行直接读缓存。
    429 限流时自动重试（指数退避），分页间隔可配置。
    """

    API_PAGE_SIZE = 2000

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
        self._retry = getattr(config, 'MONEYFLOW_RETRY', 3)
        self._page_delay = getattr(config, 'MONEYFLOW_PAGE_DELAY', 1.0)

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
        return os.path.join(self._cache_dir, f"moneyflow_{self._today}.csv")

    @property
    def client(self) -> Any:  # noqa: ANN401
        if self._client is None and self.api_key:
            from asharehub import AShareHub
            self._client = AShareHub(api_key=self.api_key)
        return self._client

    def fetch_all(self, date: str | None = None) -> pd.DataFrame:
        """获取指定日期全市场资金流向数据，带日级缓存。

        Args:
            date: 日期字符串 YYYYMMDD 或 YYYY-MM-DD，默认当天。

        Returns:
            DataFrame，列与 moneyflow API 一致：
            ts_code, trade_date, buy_sm_vol/amount, sell_sm_vol/amount,
            buy_md_vol/amount, sell_md_vol/amount,
            buy_lg_vol/amount, sell_lg_vol/amount,
            buy_elg_vol/amount, sell_elg_vol/amount,
            net_mf_vol, net_mf_amount
        """
        if not self.api_key:
            logger.info("[MoneyFlow] API 密钥未配置，跳过。")
            return pd.DataFrame()

        if date is not None:
            target_date = str(date).replace("-", "")
            self._override_today = target_date
        else:
            target_date = self._today

        # 仅当日相同日期走缓存
        cache_path = os.path.join(self._cache_dir, f"moneyflow_{target_date}.csv")

        if target_date == self._today and os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path)
                logger.info(f"[MoneyFlow] 读取当日缓存: {os.path.basename(cache_path)} ({len(cached)} 条)")
                return cached
            except Exception as e:
                logger.info(f"[MoneyFlow] 缓存读取失败，将重新拉取: {e}")

        if not self.client:
            logger.info("[MoneyFlow] 客户端初始化失败，跳过。")
            return pd.DataFrame()

        fmt_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"
        all_dfs = []
        offset = 0
        page = 1
        logger.info(f"[MoneyFlow] 正在从 AShareHub 获取全市场资金流向 (date={fmt_date})...")

        done = False
        while not done:
            last_err = None
            for attempt in range(1, self._retry + 2):
                try:
                    if attempt > 1:
                        wait = min(2 ** attempt, 30)
                        logger.info(f"  [资金流 重试 {attempt-1}/{self._retry}] 等待 {wait}s...")
                        time.sleep(wait)
                    df = self.client.moneyflow(start_date=fmt_date, end_date=fmt_date,
                                               limit=self.API_PAGE_SIZE, offset=offset)
                    if df is None or df.empty:
                        last_err = "空响应"
                        break
                    all_dfs.append(df)
                    row_count = len(df)
                    logger.info(f"  [资金流分页 {page}] offset={offset}, 返回 {row_count} 行")
                    if row_count < self.API_PAGE_SIZE:
                        done = True
                        break
                    offset += row_count
                    page += 1
                    if self._page_delay > 0:
                        time.sleep(self._page_delay)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if '429' in err_str or 'Too Many Requests' in err_str:
                        if attempt <= self._retry:
                            continue
                    logger.info(f"[MoneyFlow] 获取失败: {e}")
                    break
            if last_err:
                break

        if not all_dfs:
            logger.info("[MoneyFlow] 未获取到任何资金流向数据。")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"[MoneyFlow] 获取完成，共 {len(combined)} 条记录（{page-1} 页）")

        # 写缓存（仅当天数据）
        if target_date == self._today:
            try:
                os.makedirs(self._cache_dir, exist_ok=True)
                combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
                logger.info(f"[MoneyFlow] 已缓存至: {os.path.basename(cache_path)}")
            except Exception as e:
                logger.info(f"[MoneyFlow] 缓存写入失败: {e}")

        return combined
