from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz
import requests
from loguru import logger

from UtilsManager.AkshareConfig import ensure_akshare_timeout


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
                from UtilsManager.ConfigParser import Config
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
            ensure_akshare_timeout()
            import akshare as ak
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

        if input_date is not None:
            check_date = input_date
            if check_date.tzinfo is None:
                check_date = self.beijing_tz.localize(check_date)
            else:
                check_date = check_date.astimezone(self.beijing_tz)
        else:
            # 北京时间判定（不依赖本机时区）：本机挂钟 → 本机实际 UTC 偏移 →
            # UTC → Asia/Shanghai。旧实现把本地时间直接当北京时间，本机时区
            # 非北京时间时会全天判错（如 UTC-7 机器，本地 06:00 = 北京 21:00）。
            _local = datetime.now().astimezone()
            check_date = _local.astimezone(self.beijing_tz)

        current_str = check_date.strftime("%Y-%m-%d")

        # 当日日线行情收盘后（北京时间 15:30）才完整：15:30 前"今天"的数据
        # 尚未产生，只应返回前一个交易日（回测/复盘共表的新鲜度基准）。
        # 旧阈值 06:00 过早——白天任意运行都会误判"今天"，触发全市场重复下载。
        if current_str in official_dates and (
            check_date.hour > 15 or (check_date.hour == 15 and check_date.minute >= 30)
        ):
            return current_str

        for i in range(1, 60):
            prev_date = check_date - timedelta(days=i)
            prev_str = prev_date.strftime("%Y-%m-%d")
            if prev_str in official_dates:
                return prev_str

        return current_str

    def is_trading_day(self, date_str: str) -> bool:
        """判断给定日期是否为交易日。支持 %Y-%m-%d 和 %Y%m%d 格式。"""
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str in self.get_official_trading_dates()


# --- 实例化供外部调用 ---
# trading_calendar = TradingCalendarAnalyzer()
