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


class ChipDistributionFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.ASHAREHUB_API_KEY
        self.enabled = config.ENABLE_CHIP_DISTRIBUTION
        self.history_days = config.CHIP_HISTORY_DAYS
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            from asharehub import AShareHub
            self._client = AShareHub(api_key=self.api_key)
        return self._client

    def fetch_chip_data(self, symbols: list[str], batch_size: int = 100, max_workers: int = 10) -> pd.DataFrame:
        if not self.enabled or not self.client:
            print("[ChipDist] 筹码分布获取未启用或 API 密钥未配置，跳过。")
            return pd.DataFrame()

        total = len(symbols)
        total_batches = (total + batch_size - 1) // batch_size
        all_dfs = []

        print(f"[ChipDist] 正在获取 {total} 只股票的筹码分布数据（免费版每日 100 次调用）...")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_symbols = symbols[start:end]
            batch_results = []

            batch_desc = f"筹码批次{batch_idx + 1}/{total_batches}"
            with tqdm(total=len(batch_symbols), desc=batch_desc, unit="只", ncols=80, leave=False) as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(self._fetch_single, s): s for s in batch_symbols}
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            result = future.result()
                            if result is not None and not result.empty:
                                batch_results.append(result)
                        except Exception:
                            pass
                        pbar.update(1)

            if batch_results:
                all_dfs.extend(batch_results)

            if batch_idx < total_batches - 1:
                time.sleep(5)

        if not all_dfs:
            print("[ChipDist] 未获取到任何筹码分布数据。")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        return combined

    def _fetch_single(self, symbol: str) -> pd.DataFrame | None:
        ts_code = _akshare_to_ts_code(symbol)
        try:
            df = self.client.chip_distribution(ts_code=ts_code, limit=self.history_days)
            if df is not None:
                df["symbol"] = symbol
            return df
        except Exception:
            return None


if __name__ == "__main__":
    fetcher = ChipDistributionFetcher(Config())
    result = fetcher.fetch_chip_data(["sh600519", "sz000001", "sz300750"])
    if not result.empty:
        print(result.head())
