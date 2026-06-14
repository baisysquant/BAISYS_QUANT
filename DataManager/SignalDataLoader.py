"""数据加载器：为 TASignalProcessor 提供筹码分布、资金流向、业绩预告数据。"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger


class SignalDataLoader:
    """加载 TASignalProcessor 所需的辅助数据（筹码分布、资金流向、业绩预告）。"""

    @staticmethod
    def load_chip_distribution(config: Any) -> dict[str, dict]:  # noqa: ANN401
        """加载筹码分布 CSV，返回 {纯6位代码: row_dict}。"""
        lookup: dict[str, dict] = {}
        today_str = datetime.now().strftime('%Y%m%d')
        chip_path = os.path.join(
            getattr(config, 'HOME_DIRECTORY', '~/Downloads/CoreNews_Reports'),
            f"chip_distribution_{today_str}.csv",
        )
        chip_path = os.path.expanduser(chip_path)
        if not os.path.exists(chip_path):
            return lookup
        try:
            chip_df = pd.read_csv(chip_path)
            for _, row in chip_df.iterrows():
                pure = str(row['symbol'])
                for prefix in ('sh', 'sz', 'bj'):
                    if pure.startswith(prefix):
                        pure = pure[len(prefix):]
                        break
                lookup[pure] = row.to_dict()
            logger.info(f"[SignalDataLoader] 已加载 {len(lookup)} 条筹码数据")
        except Exception as e:
            logger.info(f"[SignalDataLoader] 加载筹码数据失败: {e}")
        return lookup

    @staticmethod
    def load_moneyflow_data(config: Any) -> dict[str, dict]:  # noqa: ANN401
        """加载资金流向数据，返回 {纯6位代码: row_dict}。"""
        lookup: dict[str, dict] = {}
        try:
            from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher
            mf_fetcher = MoneyFlowFetcher(config)
            mf_df = mf_fetcher.fetch_all()
            if mf_df is not None and not mf_df.empty:
                for _, row in mf_df.iterrows():
                    ts_code = str(row.get('ts_code', ''))
                    pure = ts_code.split('.')[0]
                    lookup[pure] = row.to_dict()
                logger.info(f"[SignalDataLoader] 已加载 {len(lookup)} 条资金流向数据")
        except Exception as e:
            logger.info(f"[SignalDataLoader] 加载资金流向失败: {e}")
        return lookup

    @staticmethod
    def load_forecast_data(config: Any) -> dict[str, dict]:  # noqa: ANN401
        """加载业绩预告数据，返回 {纯6位代码: row_dict}。"""
        lookup: dict[str, dict] = {}
        try:
            from DataCollection.FinancialForecastFetcher import FinancialForecastFetcher
            fc_fetcher = FinancialForecastFetcher(config)
            fc_df = fc_fetcher.fetch_all()
            if fc_df is not None and not fc_df.empty:
                for _, row in fc_df.iterrows():
                    ts_code = str(row.get('ts_code', ''))
                    pure = ts_code.split('.')[0]
                    lookup[pure] = row.to_dict()
                logger.info(f"[SignalDataLoader] 已加载 {len(lookup)} 条业绩预告数据")
        except Exception as e:
            logger.info(f"[SignalDataLoader] 加载业绩预告失败: {e}")
        return lookup

    @staticmethod
    def load_all(config: Any) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:  # noqa: ANN401
        """一次性加载全部辅助数据，返回 (chip_lookup, moneyflow_lookup, forecast_lookup)。"""
        return (
            SignalDataLoader.load_chip_distribution(config),
            SignalDataLoader.load_moneyflow_data(config),
            SignalDataLoader.load_forecast_data(config),
        )
