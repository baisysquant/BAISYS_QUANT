import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigParser import Config
from UtilsManager.ConfigCipher import ConfigCipher


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

    通过一次 API 调用（不传 ts_code）获取全市场最新筹码快照。
    免费版每日 100 次调用充足；如果全市场超过 2000 只，自动分页。

    缓存策略：当天首次调用拉取 API → 保存 CSV；当天再次调用直接读取缓存。
    """

    API_PAGE_SIZE = 2000

    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.ASHAREHUB_API_KEY
        self.enabled = config.ENABLE_CHIP_DISTRIBUTION
        self._client = None
        self._cache_dir = getattr(config, 'TEMP_DATA_DIRECTORY', os.path.expanduser("~/Downloads/CoreNews_Reports"))

    @property
    def _today(self) -> str:
        return datetime.now().strftime("%Y%m%d")

    @property
    def _cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"chip_distribution_{self._today}.csv")

    @property
    def client(self):
        if self._client is None and self.api_key:
            from asharehub import AShareHub
            self._client = AShareHub(api_key=self.api_key)
        return self._client

    def fetch_chip_data(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """获取全市场筹码分布数据（最新快照），带日级缓存。

        Args:
            symbols: 保留参数，不再使用。API 全局返回所有股票。

        Returns:
            DataFrame with columns: ts_code, trade_date, winner_rate, weight_avg,
                                     cost_5pct, cost_25pct, cost_50pct, cost_75pct, cost_95pct
        """
        if not self.enabled or not self.api_key:
            print("[ChipDist] 筹码分布获取未启用或 API 密钥未配置，跳过。")
            return pd.DataFrame()

        # ── 读缓存（当日已有文件） ──
        if os.path.exists(self._cache_path):
            try:
                cached = pd.read_csv(self._cache_path)
                print(f"[ChipDist] 读取当日缓存: {os.path.basename(self._cache_path)} ({len(cached)} 条)")
                return cached
            except Exception as e:
                print(f"[ChipDist] 缓存读取失败，将重新拉取: {e}")

        # ── 拉取 API ──
        if not self.client:
            print("[ChipDist] 客户端初始化失败，跳过。")
            return pd.DataFrame()

        all_dfs = []
        offset = 0
        page = 1
        print("[ChipDist] 正在从 AShareHub 获取全市场筹码分布数据...")

        while True:
            try:
                df = self.client.chip_distribution(limit=self.API_PAGE_SIZE, offset=offset)
                if df is None or df.empty:
                    break
                df["symbol"] = df["ts_code"].apply(_ts_code_to_akshare_symbol)
                all_dfs.append(df)
                row_count = len(df)
                print(f"  [筹码分页 {page}] offset={offset}, 返回 {row_count} 行")
                if row_count < self.API_PAGE_SIZE:
                    break
                offset += row_count
                page += 1
            except Exception as e:
                print(f"[ChipDist] 获取失败: {e}")
                break

        if not all_dfs:
            print("[ChipDist] 未获取到任何筹码分布数据。")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"[ChipDist] 获取完成，共 {len(combined)} 条记录（{page-1} 页）")

        # ── 写缓存 ──
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            combined.to_csv(self._cache_path, index=False, encoding="utf-8-sig")
            print(f"[ChipDist] 已缓存至: {os.path.basename(self._cache_path)}")
        except Exception as e:
            print(f"[ChipDist] 缓存写入失败: {e}")

        return combined
