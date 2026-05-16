
import akshare as ak
from datetime import datetime, timedelta
import pytz
import pandas as pd
import os
import json
from typing import Set, Optional


class TradingCalendarAnalyzer:
    def __init__(self, cache_dir: str = "./cache"):
        self.beijing_tz = pytz.timezone('Asia/Shanghai')
        self.cache_dir = cache_dir
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 缓存文件名：包含日期，每天一个文件或通用文件均可，这里采用通用文件+过期机制
        self.cache_filename = "official_trading_dates.json"
        self.cache_path = os.path.join(self.cache_dir, self.cache_filename)

        # 缓存失效时间（秒），设为 24 小时，强制定期更新
        self.cache_ttl = 24 * 60 * 60

    def _fetch_from_akshare(self) -> Optional[Set[str]]:
        """
        私有方法：从 Akshare 获取官方交易日历。
        返回：交易日集合 (YYYY-MM-DD) 或 None
        """
        try:
            print("[Calendar] 正在从 Akshare 接口获取最新的官方交易日历...")
            df = ak.tool_trade_date_hist_sina()

            if df is None or df.empty:
                print("[Calendar WARN] Akshare 返回的数据为空。")
                return None

            # 标准化日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
            # 返回集合
            dates = set(df['trade_date'].dropna().tolist())
            print(f"[Calendar] 成功获取 {len(dates)} 条交易日数据。")
            return dates

        except Exception as e:
            print(f"[Calendar ERROR] Akshare 接口调用失败: {e}")
            return None

    def _load_from_cache(self) -> Optional[Set[str]]:
        """
        私有方法：从本地文件加载缓存。
        """
        if os.path.exists(self.cache_path):
            try:
                file_stat = os.stat(self.cache_path)
                # 检查文件是否在 TTL 有效期内
                file_age = datetime.now().timestamp() - file_stat.st_mtime
                if file_age < self.cache_ttl:
                    with open(self.cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 将列表转为集合
                        dates = set(data.get('dates', []))
                        print(f"[Calendar] 成功从本地缓存加载数据 (文件未过期)。")
                        return dates
                else:
                    print(f"[Calendar] 本地缓存文件已过期，将尝试更新。")
            except Exception as e:
                print(f"[Calendar ERROR] 读取缓存文件失败: {e}")
        else:
            print(f"[Calendar] 本地缓存文件不存在: {self.cache_path}")
        return None

    def _save_to_cache(self, dates: Set[str]):
        """
        私有方法：保存数据到本地缓存。
        """
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "date_count": len(dates),
                "dates": sorted(list(dates))  # 集合不可JSON序列化，转为排序后的列表
            }
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[Calendar] 新的交易日历已保存到本地缓存。")
        except Exception as e:
            print(f"[Calendar ERROR] 保存缓存失败: {e}")

    def get_official_trading_dates(self) -> Set[str]:
        """
        公共方法：获取官方交易日历（优先缓存，其次接口）。
        逻辑：
        1. 尝试读取本地缓存（未过期）。
        2. 如果缓存不可用，尝试从 Akshare 接口获取。
        3. 如果接口失败，尝试读取本地缓存（不管过没过期，保底用）。
        4. 如果全失败，回退到仅周末逻辑。
        """
        # 1. 尝试加载有效缓存
        dates = self._load_from_cache()
        if dates:
            return dates

        # 2. 缓存失效或不存在，尝试从网络获取
        fresh_dates = self._fetch_from_akshare()
        if fresh_dates:
            # 获取成功，保存新缓存
            self._save_to_cache(fresh_dates)
            return fresh_dates

        # 3. 网络获取失败，尝试强制读取旧缓存（即使过期，也比没有强）
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    dates = set(data.get('dates', []))
                    print(f"[Calendar WARN] 接口失效，正在使用过期的本地缓存数据。")
                    return dates
            except:
                pass

        # 4. 所有手段失败，回退到仅排除周末逻辑（保底）
        print("[Calendar CRITICAL] 缓存和接口均不可用，回退到仅周末逻辑（无法识别法定节假日）。")
        # 生成未来一年的日期作为保底
        base = datetime.now() - timedelta(days=30)
        fallback_dates = []
        for x in range(-30, 365):
            d = base + timedelta(days=x)
            if d.weekday() < 5:  # 排除周六周日
                fallback_dates.append(d.strftime('%Y-%m-%d'))
        return set(fallback_dates)

    def get_last_trading_day(self, input_date: datetime = None) -> str:
        """
        核心方法：计算最后一个交易日。
        """
        # 获取官方日历数据
        official_dates = self.get_official_trading_dates()

        # 处理输入时间
        check_date = input_date or datetime.now()
        if check_date.tzinfo is None:
            check_date = self.beijing_tz.localize(check_date)
        else:
            check_date = check_date.astimezone(self.beijing_tz)

        current_str = check_date.strftime('%Y-%m-%d')

        # 逻辑：如果当天是交易日且过了早上6点，算当天；否则找上一个交易日
        if current_str in official_dates and check_date.hour >= 6:
            return current_str

        # 向前回溯寻找
        for i in range(1, 60):  # 最多回溯60天（覆盖超长假期）
            prev_date = check_date - timedelta(days=i)
            prev_str = prev_date.strftime('%Y-%m-%d')
            if prev_str in official_dates:
                return prev_str

        # 理论上不会走到这里，除非日历数据为空
        return current_str

# --- 实例化供外部调用 ---
# 假设你的缓存目录配置在 Config 中，或者直接用默认的
# trading_calendar = TradingCalendarAnalyzer()