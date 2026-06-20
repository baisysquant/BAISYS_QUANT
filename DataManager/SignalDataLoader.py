"""数据加载器：为 TASignalProcessor 提供筹码分布、资金流向、业绩预告数据。"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger


def _get_last_trading_day() -> str:
    """从交易日历获取最后一个交易日 YYYYMMDD。"""
    from DataCollection.CalendarManager import TradingCalendarAnalyzer
    try:
        cal = TradingCalendarAnalyzer()
        raw = cal.get_last_trading_day()
        return raw.replace("-", "")
    except Exception:
        return datetime.now().strftime("%Y%m%d")


class SignalDataLoader:
    """加载 TASignalProcessor 所需的辅助数据（筹码分布、资金流向、业绩预告）。"""

    @staticmethod
    def load_chip_distribution(config: Any, today_str: str | None = None) -> dict[str, dict]:  # noqa: ANN401
        """加载筹码分布数据，返回 {纯6位代码: row_dict}。"""
        lookup: dict[str, dict] = {}
        if today_str is None:
            today_str = _get_last_trading_day()
        try:
            from DataCollection.ChipDistributionFetcher import ChipDistributionFetcher
            fetcher = ChipDistributionFetcher(config)
            chip_df = fetcher.fetch_chip_data(date=today_str)
            if chip_df is not None and not chip_df.empty:
                for _, row in chip_df.iterrows():
                    pure = str(row.get('symbol', ''))
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
    def load_moneyflow_data(config: Any, today_str: str | None = None) -> dict[str, dict]:  # noqa: ANN401
        """加载资金流向数据，返回 {纯6位代码: row_dict}。"""
        lookup: dict[str, dict] = {}
        if today_str is None:
            today_str = _get_last_trading_day()
        try:
            from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher
            mf_fetcher = MoneyFlowFetcher(config)
            mf_df = mf_fetcher.fetch_all(date=today_str)
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
    def load_forecast_data(config: Any, today_str: str | None = None) -> dict[str, dict]:  # noqa: ANN401
        """加载业绩预告数据，返回 {纯6位代码: row_dict}。"""
        lookup: dict[str, dict] = {}
        if today_str is None:
            today_str = _get_last_trading_day()
        try:
            from DataCollection.FinancialForecastFetcher import FinancialForecastFetcher
            fc_fetcher = FinancialForecastFetcher(config)
            fc_df = fc_fetcher.fetch_all(date=today_str)
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
    def load_all(config: Any, today_str: str | None = None) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:  # noqa: ANN401
        """一次性加载全部辅助数据，返回 (chip_lookup, moneyflow_lookup, forecast_lookup)。"""
        if today_str is None:
            today_str = _get_last_trading_day()
        return (
            SignalDataLoader.load_chip_distribution(config, today_str=today_str),
            SignalDataLoader.load_moneyflow_data(config, today_str=today_str),
            SignalDataLoader.load_forecast_data(config, today_str=today_str),
        )
