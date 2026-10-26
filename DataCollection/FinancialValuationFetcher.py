from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine


class FinancialValuationFetcher:
    """估值/市值因子采集器 — AShareHub（日频）。

    通过 AShareHub ``/v2/market/fundamentals`` 每日全量获取全 A 估值横截面。
    """

    TABLE_NAME = "ods_financial_valuation"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine = get_engine(config)
        self._api_key = config.ASHAREHUB_API_KEY
        self._retry = config.FUNDAMENTALS_RETRY
        self._client = None

    @property
    def client(self) -> Any:  # noqa: ANN401
        if self._client is None and self._api_key:
            from UtilsManager.AShareHubClient import make_asharehub_client
            self._client = make_asharehub_client(api_key=self._api_key)
        return self._client

    def fetch_by_date(self, trade_date: str) -> pd.DataFrame:
        """获取指定交易日全市场估值数据。

        Args:
            trade_date: YYYYMMDD 格式交易日。

        Returns:
            DataFrame: symbol, pe, pe_ttm, pb, total_mv, circ_mv
        """
        if not self.client:
            logger.info("[FinancialValuation] AShareHub 客户端未初始化，跳过。")
            return pd.DataFrame()

        import time as _time

        fmt_date = trade_date.replace("-", "")
        df: pd.DataFrame | None = None
        for attempt in range(1, self._retry + 2):
            try:
                df = self.client.fundamentals(trade_date=fmt_date)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                if attempt <= self._retry:
                    _time.sleep(2 ** attempt)
                    continue
                logger.warning(f"[FinancialValuation] 获取失败 (已重试 {self._retry} 次): {e}")
                return pd.DataFrame()

        if df is None or df.empty:
            logger.info("[FinancialValuation] 未获取到估值数据。")
            return pd.DataFrame()

        # AShareHub fundamentals 返回含 ts_code / symbol 列
        code_col = "symbol" if "symbol" in df.columns else "ts_code"
        required = {code_col, "pe", "pe_ttm", "pb", "total_mv", "circ_mv"}
        available = required & set(df.columns)
        if not available:
            logger.warning(f"[FinancialValuation] 返回列 {list(df.columns)} 不包含所需列 {required}")
            return pd.DataFrame()

        rename_map = {code_col: "symbol"}
        result = df[list(available)].rename(columns=rename_map).copy()
        result["trade_date"] = trade_date
        return result

    def sync_daily(self, trade_date: str | None = None) -> int:
        """每日同步一次全 A 估值数据到数据库。

        Args:
            trade_date: YYYYMMDD 格式，默认取最近交易日。

        Returns:
            int: 写入行数
        """
        if not self.config.ENABLE_FUNDAMENTALS or not self._api_key:
            logger.info("[FinancialValuation] 未启用或 API 密钥未配置，跳过。")
            return 0

        if trade_date is None:
            try:
                from DataCollection.CalendarManager import TradingCalendarAnalyzer
                trade_date = TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
            except Exception:
                trade_date = datetime.now().strftime("%Y%m%d")

        df = self.fetch_by_date(trade_date)
        if df.empty:
            return 0

        from sqlalchemy import text

        rows_written = 0
        with self._engine.begin() as conn:
            for _, row in df.iterrows():
                try:
                    sql = text(
                        f"INSERT INTO {self.TABLE_NAME} "
                        "(symbol, trade_date, pe, pe_ttm, pb, total_mv, circ_mv) "
                        "VALUES (:symbol, :trade_date, :pe, :pe_ttm, :pb, :total_mv, :circ_mv) "
                        "ON CONFLICT (symbol, trade_date) DO UPDATE SET "
                        "pe = EXCLUDED.pe, "
                        "pe_ttm = EXCLUDED.pe_ttm, "
                        "pb = EXCLUDED.pb, "
                        "total_mv = EXCLUDED.total_mv, "
                        "circ_mv = EXCLUDED.circ_mv"
                    )
                    conn.execute(sql, {
                        "symbol": str(row["symbol"]),
                        "trade_date": trade_date,
                        "pe": self._safe_float(row.get("pe")),
                        "pe_ttm": self._safe_float(row.get("pe_ttm")),
                        "pb": self._safe_float(row.get("pb")),
                        "total_mv": self._safe_float(row.get("total_mv")),
                        "circ_mv": self._safe_float(row.get("circ_mv")),
                    })
                    rows_written += 1
                except Exception as e:
                    logger.warning(f"[FinancialValuation] 写入失败 {row.get('symbol', '?')}: {e}")

        logger.info(f"[FinancialValuation] 同步完成，写入 {rows_written} 条记录")
        return rows_written

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        if val is None:
            return None
        try:
            v = float(val)
            return None if v != v else v
        except (TypeError, ValueError):
            return None

    def load_latest_valuation(self, symbols: list[str] | None = None,
                              trade_date: str | None = None) -> pd.DataFrame:
        """从数据库加载最近一期估值因子数据。"""
        from sqlalchemy import text

        if trade_date is None:
            try:
                from DataCollection.CalendarManager import TradingCalendarAnalyzer
                trade_date = TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
            except Exception:
                trade_date = datetime.now().strftime("%Y%m%d")

        if symbols:
            placeholders = ", ".join(f":s{i}" for i in range(len(symbols)))
            params = {f"s{i}": s for i, s in enumerate(symbols)}
            params["trade_date"] = trade_date
            sql = text(
                f"SELECT symbol, trade_date, pe, pe_ttm, pb, total_mv, circ_mv "
                f"FROM {self.TABLE_NAME} "
                "WHERE trade_date = :trade_date "
                f"AND symbol IN ({placeholders})"
            )
        else:
            sql = text(
                f"SELECT symbol, trade_date, pe, pe_ttm, pb, total_mv, circ_mv "
                f"FROM {self.TABLE_NAME} "
                "WHERE trade_date = :trade_date"
            )
            params = {"trade_date": trade_date}

        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
