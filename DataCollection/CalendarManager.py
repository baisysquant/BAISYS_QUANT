from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import akshare as ak
import pandas as pd
import pytz
import requests
from loguru import logger

from UtilsManager.AkshareConfig import ensure_akshare_timeout

ensure_akshare_timeout()


class TradingCalendarAnalyzer:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> TradingCalendarAnalyzer:  # noqa: ANN401
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: str | None = None) -> None:
        if self._initialized:
            return
        if cache_dir is None:
            try:
                from ConfigParser import Config
                cache_dir = os.path.join(Config().CACHE_DIRECTORY, "calendar")
            except Exception:
                cache_dir = "./cache"
        self._initialized = True
        self.beijing_tz = pytz.timezone("Asia/Shanghai")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_filename = "official_trading_dates.json"
        self.cache_path = os.path.join(self.cache_dir, self.cache_filename)

        self.cache_ttl = 24 * 60 * 60
        
        self._cached_dates = None
        self._cache_load_time = None

    def _fetch_from_akshare(self) -> set[str] | None:
        try:
            logger.info("[Calendar] 正在从 Akshare 接口获取最新的官方交易日历...")
            df = ak.tool_trade_date_hist_sina()

            if df is None or df.empty:
                logger.warning("[Calendar WARN] Akshare 返回的数据为空。")
                return None

            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
            dates = set(df["trade_date"].dropna().tolist())
            logger.info(f"[Calendar] 成功获取 {len(dates)} 条交易日数据。")
            return dates

        except (ConnectionError, ValueError, KeyError, AttributeError, requests.exceptions.SSLError) as e:
            logger.error(f"[Calendar ERROR] Akshare 接口调用失败: {e}")
            return None

    def _load_from_cache(self) -> set[str] | None:
        if os.path.exists(self.cache_path):
            try:
                file_stat = os.stat(self.cache_path)
                file_age = datetime.now().timestamp() - file_stat.st_mtime
                if file_age < self.cache_ttl:
                    with open(self.cache_path, encoding="utf-8") as f:
                        data = json.load(f)
                        dates = set(data.get("dates", []))
                    logger.info("[Calendar] 交易日历已从本地缓存加载 (文件未过期)。")
                    return dates
                else:
                    logger.info("[Calendar] 本地缓存文件已过期，将尝试更新。")
            except (json.JSONDecodeError, OSError, ValueError, KeyError) as e:
                logger.error(f"[Calendar ERROR] 读取缓存文件失败: {e}")
        else:
            logger.info(f"[Calendar] 本地缓存文件不存在: {self.cache_path}")
        return None

    def _save_to_cache(self, dates: set[str]) -> None:
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "date_count": len(dates),
                "dates": sorted(list(dates)),
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("[Calendar] 新的交易日历已保存到本地缓存。")
        except (OSError, PermissionError, TypeError) as e:
            logger.error(f"[Calendar ERROR] 保存缓存失败: {e}")

    def get_official_trading_dates(self) -> set[str]:
        if self._cached_dates is not None and self._cache_load_time is not None:
            memory_age = datetime.now().timestamp() - self._cache_load_time
            if memory_age < self.cache_ttl:
                return self._cached_dates

        dates = self._load_from_cache()
        if dates:
            self._cached_dates = dates
            self._cache_load_time = datetime.now().timestamp()
            return dates

        fresh_dates = self._fetch_from_akshare()
        if fresh_dates:
            self._save_to_cache(fresh_dates)
            self._cached_dates = fresh_dates
            self._cache_load_time = datetime.now().timestamp()
            return fresh_dates

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    data = json.load(f)
                    dates = set(data.get("dates", []))
                    logger.warning("[Calendar WARN] 接口失效，正在使用过期的本地缓存数据。")
                    self._cached_dates = dates
                    self._cache_load_time = datetime.now().timestamp()
                    return dates
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        logger.critical("[Calendar CRITICAL] 缓存和接口均不可用，使用 chinesecalendar 节假日库回退。")
        base = datetime.now() - timedelta(days=30)
        fallback_dates = []
        try:
            from chinese_calendar import is_workday
            for x in range(-30, 365):
                d = base + timedelta(days=x)
                if is_workday(d.date() if hasattr(d, "date") else d):
                    fallback_dates.append(d.strftime("%Y-%m-%d"))
        except ImportError:
            logger.warning("[Calendar WARN] chinesecalendar 未安装，回退到仅周末逻辑（无法识别法定节假日）。")
            for x in range(-30, 365):
                d = base + timedelta(days=x)
                if d.weekday() < 5:
                    fallback_dates.append(d.strftime("%Y-%m-%d"))
        self._cached_dates = set(fallback_dates)
        self._cache_load_time = datetime.now().timestamp()
        return self._cached_dates

    def get_last_trading_day(self, input_date: datetime = None) -> str:
        official_dates = self.get_official_trading_dates()

        check_date = input_date or datetime.now()
        if check_date.tzinfo is None:
            check_date = self.beijing_tz.localize(check_date)
        else:
            check_date = check_date.astimezone(self.beijing_tz)

        current_str = check_date.strftime("%Y-%m-%d")

        if current_str in official_dates and check_date.hour >= 6:
            return current_str

        for i in range(1, 60):
            prev_date = check_date - timedelta(days=i)
            prev_str = prev_date.strftime("%Y-%m-%d")
            if prev_str in official_dates:
                return prev_str

        return current_str

    def get_trading_day_offset(self, offset_days: int, base_date: str = None) -> str:
        official_dates = self.get_official_trading_dates()

        if base_date is None:
            base_date = self.get_last_trading_day()

        sorted_dates = sorted(list(official_dates))

        try:
            base_index = sorted_dates.index(base_date)
        except ValueError:
            logger.warning(f"[Calendar WARN] 基准日期 {base_date} 不在交易日列表中，使用最后交易日")
            base_index = len(sorted_dates) - 1

        target_index = base_index + offset_days

        if target_index < 0:
            target_index = 0
        elif target_index >= len(sorted_dates):
            target_index = len(sorted_dates) - 1

        return sorted_dates[target_index]


# --- 实例化供外部调用 ---
# trading_calendar = TradingCalendarAnalyzer()

