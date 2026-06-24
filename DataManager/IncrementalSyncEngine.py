from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from typing import Any

import akshare as ak
import pandas as pd
from loguru import logger
from sqlalchemy import text

from DataCollection.CalendarManager import TradingCalendarAnalyzer


TABLE = "stock_daily_kline"
OVERLAP_DAYS = 15


class IncrementalSyncEngine:
    def __init__(
        self,
        db_engine: Any,
        default_start: str | None = None,
        asharehub_api_key: str | None = None,
        cache_dir: str | None = None,
        main_board_only: bool = False,
        enable_research_report_filter: bool = False,
        research_report_min_count: int = 1,
    ) -> None:
        self._engine = db_engine
        self._default_start = self.align_to_trading_day(default_start) if default_start else None
        self._asharehub_api_key = asharehub_api_key
        self._main_board_only = main_board_only
        self._enable_research_report_filter = enable_research_report_filter
        self._research_report_min_count = research_report_min_count
        self._cache_dir = cache_dir
        if not self._cache_dir:
            try:
                from ConfigParser import Config
                self._cache_dir = os.path.join(Config().CACHE_DIRECTORY, "kline_batches")
            except Exception:
                self._cache_dir = os.path.join(
                    os.environ.get("TEMP", "/tmp"), "opencode", "kline_batches"
                )
        os.makedirs(self._cache_dir, exist_ok=True)
        # 使用最新交易日而非 date.today()，避免周末/节假日误判
        try:
            _cal = TradingCalendarAnalyzer()
            _today_str = _cal.get_last_trading_day()
            self._trade_date = datetime.strptime(_today_str, "%Y-%m-%d").date()
        except Exception:
            self._trade_date = date.today()
        self._trade_date_str = self._trade_date.isoformat().replace("-", "")
        self._cleanup_old_cache()

    # ── public API ──────────────────────────────────────────────

    def _cleanup_old_cache(self) -> None:
        """清理超过 7 天的缓存文件，以及脏 close_normal_*.csv（日期不匹配当前交易日）。"""
        try:
            now = datetime.now()
            today_tag = f"close_normal_{self._trade_date_str}.csv"
            for fname in os.listdir(self._cache_dir):
                # 清理脏 close_normal 缓存（日期与当前交易日不一致）
                if fname.startswith("close_normal_") and fname != today_tag:
                    os.remove(os.path.join(self._cache_dir, fname))
                    continue
                # 清理超过 7 天的旧缓存
                fpath = os.path.join(self._cache_dir, fname)
                if os.path.isfile(fpath):
                    age = now - datetime.fromtimestamp(os.path.getmtime(fpath))
                    if age.days > 7:
                        os.remove(fpath)
        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")

    def sync_all(self, symbols_prefixed: list[str]) -> int:
        # 跳过已有最新交易日数据的股票
        remaining = self._get_stale_symbols(symbols_prefixed)
        if not remaining:
            logger.info("所有股票已有最新交易日数据，无需同步")
            return 0

        from asharehub import AShareHub
        client = AShareHub(api_key=self._asharehub_api_key)

        # 确定 date range：从最老的缺失日期开始（含 overlap，用于除权检测）
        min_latest = self._get_min_latest_date(remaining)
        if min_latest is None:
            daily_start = (
                datetime.strptime(self._default_start, "%Y%m%d").strftime("%Y-%m-%d")
                if self._default_start else "2000-01-01"
            )
        else:
            daily_start = (min_latest - timedelta(days=OVERLAP_DAYS)).isoformat()

        end_iso = self._trade_date.isoformat()
        adj_start = (
            datetime.strptime(self._default_start, "%Y%m%d").strftime("%Y-%m-%d")
            if self._default_start else "2000-01-01"
        )

        # 批量获取原始日线（仅需缺失日期范围 + overlap）
        logger.info("AShareHub 批量获取 market_daily...")
        raw_df = self._batch_fetch(client.market_daily, daily_start, end_iso)
        if raw_df.empty:
            logger.warning("market_daily 返回空")
            return 0

        # 只保留需要的股票
        prefixed_set = set(remaining)
        raw_df = raw_df[raw_df['ts_code'].apply(self._tscode_to_symbol).isin(prefixed_set)]
        if raw_df.empty:
            return 0

        # 批量获取复权因子（全历史，用于 max_factor）
        logger.info("AShareHub 批量获取 adj_factor...")
        adj_df = self._batch_fetch(client.adj_factor, adj_start, end_iso)
        if adj_df.empty:
            logger.warning("adj_factor 返回空")
            return 0

        # 合并 → 向量化计算 HFQ
        logger.info("合并数据并计算后复权...")
        merged = raw_df.merge(adj_df, on=['ts_code', 'trade_date'], how='left')
        merged = merged.dropna(subset=['adj_factor'])
        if merged.empty:
            return 0

        max_factors = adj_df.groupby('ts_code', group_keys=True)['adj_factor'].max().reset_index()
        max_factors.columns = ['ts_code', 'max_factor']
        merged = merged.merge(max_factors, on='ts_code', how='left')

        for col in ('open', 'high', 'low', 'close'):
            merged[col] = pd.to_numeric(merged[col], errors='coerce') * merged['max_factor'] / merged['adj_factor']

        merged['symbol'] = merged['ts_code'].apply(self._tscode_to_symbol)
        # vol: 手 → 股（×100）；amount: 千元 → 元（×1000）
        merged['volume'] = pd.to_numeric(merged['vol'], errors='coerce').fillna(0).astype('int64') * 100
        merged['amount'] = pd.to_numeric(merged['amount'], errors='coerce').fillna(0) * 1000
        merged['trade_date'] = merged['trade_date'].apply(
            lambda d: d.isoformat() if hasattr(d, 'isoformat') else str(d)
        )
        merged['close_normal'] = merged['close'] * merged['adj_factor'] / merged['max_factor']
        merged['adj_factor'] = merged['adj_factor'].astype(float)

        # 逐只股票：除权检测 + 增量写入
        total_inserted = 0
        to_write: list[pd.DataFrame] = []
        full_refresh_symbols: list[str] = []

        for sym, grp in merged.groupby('symbol'):
            grp = grp.sort_values('trade_date')
            latest = self._get_latest_date(sym)

            if latest is None or self._is_tencent_data(sym):
                full_refresh_symbols.append(sym)
                to_write.append(grp)
                total_inserted += len(grp)
            elif self._detect_split_from_adj(sym, grp, latest):
                logger.info(f"  {sym} 除权除息，全量替换")
                full_refresh_symbols.append(sym)
                to_write.append(grp)
                total_inserted += len(grp)
            else:
                new = grp[grp['trade_date'] > latest.isoformat()]
                if not new.empty:
                    to_write.append(new)
                    total_inserted += len(new)

        if not to_write:
            return 0

        final = pd.concat(to_write, ignore_index=True)

        if full_refresh_symbols:
            with self._engine.begin() as conn:
                conn.execute(
                    text(f"DELETE FROM {TABLE} WHERE symbol = ANY(:symbols)"),
                    {"symbols": full_refresh_symbols},
                )
            logger.info(f"全量替换 {len(full_refresh_symbols)} 只")

        self._write_batch(final)
        logger.info(f"同步完成，写入 {total_inserted} 行")
        return total_inserted

    def backfill_close_normal(self, symbols_prefixed: list[str] | None = None) -> pd.DataFrame:
        """获取全市场最新不复权收盘价，写入本地缓存文件。

        AShareHub 批量获取后写入 ``close_normal_{YYYYMMDD}.csv``（ts_code, close 两列），
        各模块从此缓存文件读取，不再写 DB。

        Args:
            symbols_prefixed: 忽略（为了兼容旧调用签名）。
        Returns:
            DataFrame 包含 ts_code, close 两列；失败返回空 DataFrame。
        """
        cache_file = self._close_normal_cache_path()
        today_iso = self._trade_date.isoformat()

        # 当天缓存已存在，直接读取
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"[close_normal] 从缓存加载 {len(df)} 只 → {os.path.basename(cache_file)}")
                return df
            except Exception as e:
                logger.warning(f"[close_normal] 缓存读取失败: {e}，重新拉取")

        if not self._asharehub_api_key:
            logger.warning("[close_normal] ASHAREHUB_API_KEY 未配置，跳过")
            return pd.DataFrame()

        logger.info("[close_normal] AShareHub 批量获取全市场最新收盘价...")
        try:
            from asharehub import AShareHub
            client = AShareHub(api_key=self._asharehub_api_key)

            offset = 0
            limit = 5000
            all_rows = []
            while True:
                batch = client.market_daily(start_date=today_iso, end_date=today_iso, limit=limit, offset=offset)
                if batch is None or batch.empty:
                    break
                all_rows.extend(batch.to_dict("records"))
                if len(batch) < limit:
                    break
                offset += limit

            if not all_rows:
                logger.warning("[close_normal] AShareHub 返回空数据")
                return pd.DataFrame()

            df = pd.DataFrame(all_rows)[["ts_code", "close"]].copy()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"])
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_csv(cache_file, index=False)
            logger.info(f"[close_normal] 缓存已保存: {len(df)} 只 → {os.path.basename(cache_file)}")
            return df
        except Exception as e:
            logger.warning(f"[close_normal] AShareHub 批量请求失败: {e}")
            return pd.DataFrame()

    # ── stock pool (merged from StockSyncEngine) ──────────────

    def get_stock_pool_from_db(self) -> pd.DataFrame:
        query = """
            SELECT stock_code AS ts_code, stock_code AS 股票代码,
                   stock_name AS name, industry_name AS industry
            FROM stock_basic_info_sw ORDER BY stock_code
        """
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if "股票代码" in df.columns:
            df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
        for col in ("ts_code", "name", "industry", "股票代码"):
            if col not in df.columns:
                df[col] = "N/A"
        return df[["ts_code", "name", "industry", "股票代码"]]

    @staticmethod
    def filter_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
        if "name" not in df.columns:
            return df
        st = r"(?:\s*(?:\*|★|※|•|·))?(?:[Ss][Tt])"
        return df[~df["name"].astype(str).str.contains(st, na=False)].copy()

    def filter_main_board(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._main_board_only:
            return df
        return df[df["股票代码"].astype(str).str.match(r"^(60|00)")].copy()

    def _filter_by_research_report(self, pure_codes: set[str]) -> set[str]:
        if not self._enable_research_report_filter:
            return pure_codes
        try:
            for attempt in range(3):
                try:
                    raw = ak.stock_profit_forecast_em()
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        return pure_codes
            if raw is None or raw.empty:
                return pure_codes
            df = raw.copy()
            if "代码" in df.columns and "股票代码" not in df.columns:
                df.rename(columns={"代码": "股票代码"}, inplace=True)
            df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
            for col in df.columns:
                if "买入" in col:
                    rating_col = col
                    break
            else:
                return pure_codes
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0)
            qualified = set(df.loc[df[rating_col] > self._research_report_min_count, "股票代码"].unique())
            before = len(pure_codes)
            pure_codes &= qualified
            logger.info(f"研报过滤: {before} → {len(pure_codes)}（买入>{self._research_report_min_count}次）")
        except Exception as e:
            logger.warning(f"研报过滤异常: {e}，跳过研报过滤")
        return pure_codes

    def sync_stock_pool_and_kline(self, target_date: str | None = None) -> set[str]:
        from UtilsManager.CodeNormalizer import CodeNormalizer

        if target_date is None:
            target_date = TradingCalendarAnalyzer().get_last_trading_day()
        today_tag = target_date.replace("-", "")

        pool = self.get_stock_pool_from_db()
        before = len(pool)
        pool = self.filter_st_stocks(pool)
        pool = self.filter_main_board(pool)
        logger.info(f"股票池: {before} → {len(pool)}（过滤ST/板块）")

        pure_codes = set(pool["股票代码"].unique())
        # K 线同步覆盖全 A 股（已过滤 ST/板块），保持数据完整供回测使用
        symbols = [CodeNormalizer.add_market_prefix(c) for c in sorted(pure_codes)]
        inserted = self.sync_all(symbols)
        logger.info(f"K线同步完成，新增 {inserted} 行")

        # 研报过滤仅影响分析池，不影响 K 线数据完整性
        analysis_pool = self._filter_by_research_report(pure_codes)

        save_dir = os.path.dirname(self._cache_dir) if os.path.isdir(self._cache_dir) else os.getcwd()
        out = os.path.join(save_dir, f"final_filtered_stocks_{today_tag}.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for c in sorted(analysis_pool):
                f.write(f"{c}\n")
        logger.info(f"最终股票列表已保存: {len(analysis_pool)} 只 → {out}")
        return analysis_pool

    # ── batch helpers (AShareHub) ─────────────────────────────

    @staticmethod
    def _tscode_to_symbol(ts_code: str) -> str:
        """将 AShareHub ts_code 格式转为内部 symbol 格式（000001.SZ → sz000001）。"""
        parts = ts_code.split(".")
        if len(parts) == 2:
            return f"{parts[1].lower()}{parts[0]}"
        return ts_code

    def _get_min_latest_date(self, symbols: list[str]) -> date | None:
        """返回一组股票中最老的缺失日期——即最小 latest_date。"""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT MIN(latest) FROM (
                        SELECT MAX(trade_date::date) AS latest FROM {TABLE}
                        WHERE symbol = ANY(:symbols) GROUP BY symbol
                    ) sub
                """),
                {"symbols": symbols},
            ).scalar()
        return rows

    def _batch_fetch(self, fetch_fn: Any, start_date: str, end_date: str) -> pd.DataFrame:  # noqa: ANN401
        """分页批量获取 AShareHub 数据（market_daily 或 adj_factor），自动翻页。"""
        limit = 5000
        offset = 0
        all_parts = []
        while True:
            batch = fetch_fn(start_date=start_date, end_date=end_date, limit=limit, offset=offset)
            if batch is None or batch.empty:
                break
            all_parts.append(batch)
            if len(batch) < limit:
                break
            offset += limit
        if not all_parts:
            return pd.DataFrame()
        return pd.concat(all_parts, ignore_index=True) if len(all_parts) > 1 else all_parts[0]

    def _detect_split_from_adj(self, symbol: str, new_df: pd.DataFrame, latest: date) -> bool:
        """对比 DB 中最新的 adj_factor 与 AShareHub 新数据是否一致。"""
        with self._engine.connect() as conn:
            old = conn.execute(
                text(f"SELECT adj_factor FROM {TABLE} WHERE symbol = :sym AND trade_date::date = :date"),
                {"sym": symbol, "date": latest},
            ).scalar()
        if old is None:
            return False
        new_adj = new_df.loc[new_df['trade_date'] == latest.isoformat(), 'adj_factor']
        if new_adj.empty:
            return False
        ratio = new_adj.iloc[0] / old
        return ratio > 1.01 or ratio < 0.99

    # ── stale filter (P0-1) ─────────────────────────────────────

    def _get_stale_symbols(self, symbols: list[str]) -> list[str]:
        if not symbols:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT symbol FROM {TABLE}
                    WHERE symbol = ANY(:symbols)
                      AND trade_date::date = :trade_date
                """),
                {"symbols": symbols, "trade_date": self._trade_date.isoformat()},
            ).fetchall()
        up_to_date = {row[0] for row in rows}
        stale = [s for s in symbols if s not in up_to_date]
        skipped = len(symbols) - len(stale)
        if skipped:
            logger.info(f"跳过 {skipped} 只（已有 {self._trade_date_str} 数据），需处理 {len(stale)} 只")
        return stale

    def _is_tencent_data(self, symbol: str) -> bool:
        """判断 DB 中股票是否为 Tencent 旧数据（close_normal IS NULL 表示从未写入原始价）。"""
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT close_normal FROM {TABLE} WHERE symbol = :sym ORDER BY trade_date DESC LIMIT 1"),
                {"sym": symbol},
            ).scalar()
        return row is None  # close_normal IS NULL → Tencent era data

    # ── trading day alignment ────────────────────────────────────

    @staticmethod
    def align_to_trading_day(date_str: str) -> str:
        """将 YYYYMMDD 对齐到当天或之后的首个交易日，返回 YYYYMMDD。"""
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            cal = TradingCalendarAnalyzer()
            dates = sorted(cal.get_official_trading_dates())
            dt = datetime.strptime(date_str, "%Y%m%d")
            formatted = dt.strftime("%Y-%m-%d")
            for d in dates:
                if d >= formatted:
                    return d.replace("-", "")
        except Exception:
            pass
        return date_str

    _align_to_trading_day = align_to_trading_day

    # ── database ────────────────────────────────────────────────

    def _get_latest_date(self, symbol: str) -> date | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT MAX(trade_date::date) FROM {TABLE} WHERE symbol = :sym"),
                {"sym": symbol},
            ).scalar()
        return row if row is not None else None

    def _write_batch(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        symbols = df["symbol"].unique().tolist()
        with self._engine.begin() as conn:
            for sym in symbols:
                sym_dates = df[df["symbol"] == sym]["trade_date"].tolist()
                conn.execute(
                    text(
                        f"DELETE FROM {TABLE} WHERE symbol = :sym AND trade_date IN :dates"
                    ),
                    {"sym": sym, "dates": tuple(sym_dates)},
                )
            conn.execute(
                text(f"""
                    INSERT INTO {TABLE}
                        (symbol, trade_date, open, close, high, low, volume, amount, adj_factor, close_normal)
                    VALUES (:symbol, :trade_date, :open, :close, :high, :low, :volume, :amount, :adj_factor, :close_normal)
                """),
                df.rename(columns={}).to_dict("records"),
            )

    # ── helpers ─────────────────────────────────────────────────

    def _close_normal_cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"close_normal_{self._trade_date_str}.csv")


