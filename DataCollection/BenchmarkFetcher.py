from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text as sql_text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine


class BenchmarkFetcher:
    """基准指数行情获取器 — AShareHub。

    获取上证综指（000001.SH）等指数日线数据，用于后续基准对比。
    """

    TABLE_NAME = "ods_index_daily"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine = get_engine(config)
        self._api_key = config.ASHAREHUB_API_KEY
        self._client = None

    @property
    def client(self) -> Any:  # noqa: ANN401
        if self._client is None and self._api_key:
            from UtilsManager.AShareHubClient import make_asharehub_client
            self._client = make_asharehub_client(api_key=self._api_key)
        return self._client

    def fetch_index(self, index_code: str = "000001.SH",
                    start_date: str | None = None,
                    end_date: str | None = None) -> pd.DataFrame:
        """获取指数日线行情。

        Args:
            index_code: 指数代码，默认 000001.SH（上证综指）。
            start_date: 起始日期 YYYYMMDD。
            end_date: 截止日期 YYYYMMDD。

        Returns:
            DataFrame: trade_date, open, high, low, close, volume
        """
        if not self.client:
            logger.info("[Benchmark] AShareHub 客户端未初始化，跳过。")
            return pd.DataFrame()

        try:
            df = self.client.index_daily(symbol=index_code)
        except Exception as e:
            logger.warning(f"[Benchmark] 获取指数 {index_code} 失败: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 统一列名
        code_col = "ts_code" if "ts_code" in df.columns else ("symbol" if "symbol" in df.columns else None)
        col_map = {}
        if code_col:
            col_map[code_col] = "index_code"

        for src, tgt in [("trade_date", "trade_date"), ("open", "open"),
                         ("high", "high"), ("low", "low"),
                         ("close", "close"), ("vol", "volume"),
                         ("amount", "amount")]:
            if src in df.columns:
                col_map[src] = tgt

        result = df.rename(columns=col_map)
        if "index_code" not in result.columns:
            result["index_code"] = index_code

        if "trade_date" in result.columns:
            result["trade_date"] = result["trade_date"].astype(str).str[:10]

        # 日期过滤
        if start_date:
            result = result[result["trade_date"] >= start_date]
        if end_date:
            result = result[result["trade_date"] <= end_date]

        return result

    def sync_daily(self, index_code: str = "000001.SH") -> int:
        """每日同步一次基准指数数据到数据库。

        Args:
            index_code: 指数代码，默认上证综指。

        Returns:
            int: 写入行数。
        """
        if not self._api_key:
            logger.info("[Benchmark] API 密钥未配置，跳过。")
            return 0

        df = self.fetch_index(index_code)
        if df.empty:
            return 0

        rows_written = 0
        with self._engine.begin() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute(sql_text(
                        f"INSERT INTO {self.TABLE_NAME} "
                        "(index_code, trade_date, open, high, low, close, volume, amount) "
                        "VALUES (:index_code, :trade_date, :open, :high, :low, :close, :volume, :amount) "
                        "ON CONFLICT (index_code, trade_date) DO UPDATE SET "
                        "open = EXCLUDED.open, high = EXCLUDED.high, "
                        "low = EXCLUDED.low, close = EXCLUDED.close, "
                        "volume = EXCLUDED.volume, amount = EXCLUDED.amount"
                    ), {
                        "index_code": str(row.get("index_code", index_code)),
                        "trade_date": str(row["trade_date"]),
                        "open": _safe_float(row.get("open")),
                        "high": _safe_float(row.get("high")),
                        "low": _safe_float(row.get("low")),
                        "close": _safe_float(row.get("close")),
                        "volume": _safe_float(row.get("volume")),
                        "amount": _safe_float(row.get("amount")),
                    })
                    rows_written += 1
                except Exception as e:
                    logger.warning(f"[Benchmark] 写入失败 {row.get('trade_date', '?')}: {e}")

        logger.info(f"[Benchmark] 指数 {index_code} 同步完成，写入 {rows_written} 条")
        return rows_written

    def load_index_data(self, index_code: str = "000001.SH",
                        start_date: str | None = None,
                        end_date: str | None = None) -> pd.DataFrame:
        """从数据库加载指数数据。"""
        from sqlalchemy import text

        sql = f"SELECT * FROM {self.TABLE_NAME} WHERE index_code = :code"
        params: dict[str, Any] = {"code": index_code}
        if start_date:
            sql += " AND trade_date >= :start"
            params["start"] = start_date
        if end_date:
            sql += " AND trade_date <= :end"
            params["end"] = end_date
        sql += " ORDER BY trade_date"

        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)


def _safe_float(val: Any) -> float | None:  # noqa: ANN401
    if val is None:
        return None
    try:
        v = float(val)
        return None if v != v else v
    except (TypeError, ValueError):
        return None
