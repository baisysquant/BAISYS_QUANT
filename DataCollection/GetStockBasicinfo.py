from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pandas as pd
from asharehub import AShareHub
from loguru import logger
from sqlalchemy.exc import DBAPIError, OperationalError

from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from DataManager.DatabaseWriter import QuantDBManager
from UtilsManager.Exceptions import DatabaseConnectionError, DatabaseError


class StockBasicInfoService:
    """股票基本信息业务服务类 (申万二级行业 - 30天缓存优化版)

    数据源：AShareHub /v2/reference/industries 接口
    一次 API 调用获取全量 A 股申万 SW2021 一/二/三级行业分类
    """

    TABLE_NAME = "stock_basic_info_sw"

    def __init__(self, config_parser: Config) -> None:
        self.config_parser = config_parser
        self.system_config = self._get_system_config_from_attributes(config_parser)
        self.logger = self._setup_logger()
        self.REFRESH_INTERVAL_DAYS = getattr(config_parser, 'STOCK_BASIC_INFO_EXPIRE_DAYS', 30)
        self.ah_client = AShareHub(api_key=config_parser.ASHAREHUB_API_KEY)
        self.db_manager = None
        self.trading_calendar = TradingCalendarAnalyzer()

    def _get_system_config_from_attributes(self, config_parser: Config) -> dict:
        return {
            "max_workers": config_parser.MAX_WORKERS,
            "data_fetch_retries": config_parser.DATA_FETCH_RETRIES,
            "data_fetch_delay": config_parser.DATA_FETCH_DELAY,
        }

    def _setup_logger(self) -> Any:  # noqa: ANN401
        return logger

    def _initialize_database(self) -> None:
        try:
            from DataManager.DbEngine import get_engine as _get_engine
            self.db_manager = QuantDBManager(engine=_get_engine(self.config_parser))
            self.logger.info("数据库连接初始化成功")
        except (DBAPIError, OperationalError, DatabaseError) as e:
            self.logger.error(f"数据库连接初始化失败: {e!s}")
            raise DatabaseConnectionError(str(e)) from e

    def _get_latest_data_date(self) -> Any | None:  # noqa: ANN401
        if not self.db_manager:
            self._initialize_database()
        try:
            latest_date = self.db_manager.get_latest_record_date(self.TABLE_NAME, "record_date")
            self.logger.debug(f"从数据库获取到的最新日期: {latest_date}")
            return latest_date
        except AttributeError as e:
            self.logger.error(f"调用数据库方法失败，可能方法名不匹配: {e!s}")
            return None
        except (DatabaseError, DBAPIError, OperationalError) as e:
            self.logger.error(f"获取最新数据日期失败: {e!s}")
            return None

    def _get_industry_count_from_api(self) -> int:
        """从 AShareHub 获取二级行业数量"""
        df = self.ah_client.industry_list()
        return df["l2_code"].nunique()

    def _is_data_up_to_date(self) -> bool:
        """
        核心校验逻辑：
        1. 超过30天 -> 强制刷新
        2. 小于30天 -> 校验表是否为空，以及二级行业数量是否一致
        """
        latest_date = self._get_latest_data_date()
        today = datetime.now().date()

        if not latest_date:
            self.logger.info(f"[缓存校验] 表 {self.TABLE_NAME} 中无数据，准备执行全量同步")
            return False

        if isinstance(latest_date, str):
            latest_date = datetime.strptime(latest_date.replace("-", "")[:8], "%Y%m%d").date()
        elif hasattr(latest_date, 'date'):
            latest_date = latest_date.date()

        delta_days = (today - latest_date).days
        self.logger.info(f"[缓存校验] 数据库最新数据日期: {latest_date}，距今 {delta_days} 天")

        if delta_days >= self.REFRESH_INTERVAL_DAYS:
            self.logger.info(f"[缓存校验] 数据已过期 (>= {self.REFRESH_INTERVAL_DAYS}天)，触发强制全量刷新")
            return False

        self.logger.info("[缓存校验] 数据在30天有效期内，开始校验行业数量是否发生变化...")

        try:
            api_industry_count = self._get_industry_count_from_api()
            db_industry_count = self.db_manager.get_distinct_count(self.TABLE_NAME, "industry_code")

            self.logger.info(f"[缓存校验] 接口二级行业数量: {api_industry_count} | 数据库行业数量: {db_industry_count}")

            if api_industry_count == db_industry_count:
                self.logger.info("[缓存校验] [OK] 行业数量一致，数据无需更新，直接使用本地缓存！")
                return True
            else:
                self.logger.info("[缓存校验] [WARN] 行业数量发生变化，触发全量刷新")
                return False

        except AttributeError as e:
            self.logger.error(f"[缓存校验] 数据库管理器方法调用失败: {e!s}，触发全量刷新")
            return False
        except Exception as e:
            self.logger.error(f"[缓存校验] 校验过程发生其他异常: {e!s}，触发全量刷新")
            return False

    # ── 静态工具方法 ──

    @staticmethod
    def _stock_symbol_to_code(symbol: str) -> str:
        """转换 AShareHub 格式 (000001.SZ) 到内部格式 (sz000001)"""
        parts = symbol.strip().split(".")
        if len(parts) == 2:
            raw_code = parts[0].zfill(6)
            market = parts[1].lower()
            return f"sh{raw_code}" if market == "sh" else f"sz{raw_code}"
        return symbol.strip()

    def fetch_stock_basic_info(self) -> pd.DataFrame:
        """获取申万二级行业成分股（严格对齐 PG 表结构的 6 个字段）"""
        max_retries = self.system_config["data_fetch_retries"]
        base_delay = self.system_config["data_fetch_delay"]

        self.logger.info("开始从 AShareHub 获取申万二级行业及成分股信息...")

        # 1. 全量获取行业分类数据（一次 API 调用）
        raw_df = None
        for attempt in range(max_retries):
            try:
                raw_df = self.ah_client.industry_list()
                self.logger.info(f"成功获取 {len(raw_df)} 条行业分类记录")
                break
            except Exception as e:
                self.logger.warning(f"获取行业数据第 {attempt + 1} 次失败: {e!s}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                else:
                    raise

        # 2. 提取二级行业列表
        industries = raw_df[["l2_code", "l2_name"]].drop_duplicates().dropna()
        total = len(industries)
        self.logger.info(f"二级行业数量: {total}")

        # 3. 获取计入日期
        raw_record_date = str(self.trading_calendar.get_last_trading_day()).replace("-", "")
        record_date = f"{raw_record_date[:4]}-{raw_record_date[4:6]}-{raw_record_date[6:8]}"

        # 4. 构建成分股记录
        def _build_one(ind_row: tuple) -> list[dict[str, Any]]:
            code, name = ind_row
            group = raw_df[raw_df["l2_code"] == code]
            if group.empty:
                return []
            rows = []
            for _, stock_row in group.iterrows():
                rows.append({
                    "industry_code": code,
                    "industry_name": name,
                    "stock_code": StockBasicInfoService._stock_symbol_to_code(str(stock_row["symbol"])),
                    "stock_name": str(stock_row.get("name", "")).strip(),
                    "weight": 0.0,
                    "record_date": record_date,
                })
            return rows

        all_stocks: list[dict[str, Any]] = []
        ind_list = list(zip(industries["l2_code"], industries["l2_name"]))

        with ThreadPoolExecutor(max_workers=min(10, total)) as pool:
            futs = {pool.submit(_build_one, ind): ind for ind in ind_list}
            done = 0
            for f in as_completed(futs):
                done += 1
                rows = f.result()
                all_stocks.extend(rows)
                if done % 20 == 0 or done == total:
                    self.logger.info(f"  进度: {done}/{total} 个行业")

        # 5. 数据清洗
        df = pd.DataFrame(all_stocks)
        if df.empty:
            self.logger.warning("未获取到任何成分股数据")
            return df

        df = df.drop_duplicates(subset=["industry_code", "stock_code", "record_date"], keep="first").reset_index(drop=True)

        self.logger.info(f"成功构建 {len(df)} 条成分股记录，覆盖 {total} 个二级行业")
        cols = ["industry_code", "industry_name", "stock_code", "stock_name", "weight", "record_date"]
        return df[cols]

    def sync_all_stock_basic_info(self) -> bool:
        """同步成分股信息：根据缓存策略决定是刷新还是跳过"""
        try:
            if not self.db_manager:
                self._initialize_database()

            if self._is_data_up_to_date():
                self.logger.info("缓存校验通过，无需更新数据。")
                return True

            df = self.fetch_stock_basic_info()
            if df.empty:
                self.logger.warning("获取到的数据为空")
                return False

            self.logger.info(f"[DB Sync] 开始全量刷新到 PostgreSQL 表 {self.TABLE_NAME}")
            self.db_manager.truncate_and_insert(df, self.TABLE_NAME)
            self.logger.info(f"[DB Sync] 成功写入 {len(df)} 条记录。")

            return True

        except (DatabaseError, DBAPIError, OperationalError) as e:
            self.logger.error(f"同步成分股信息失败: {e!s}")
            return False

    def get_stock_count(self) -> int:
        """获取表中的记录总数"""
        if not self.db_manager:
            self._initialize_database()
        try:
            return self.db_manager.get_table_count(self.TABLE_NAME)
        except (DatabaseError, DBAPIError, OperationalError) as e:
            self.logger.error(f"获取记录总数失败: {e!s}")
            return 0
