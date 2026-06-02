import logging
import time
from datetime import datetime

import pandas as pd
import akshare as ak

from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
# 注意：这里假设你的模块路径是正确的
from DataManager.DatabaseWriter import QuantDBManager


class StockBasicInfoService:
    """股票基本信息业务服务类 (申万二级行业 - 30天缓存优化版)"""

    TABLE_NAME = "stock_basic_info_sw"
    REFRESH_INTERVAL_DAYS = 30  # 强制刷新周期（天）

    def __init__(self, config_parser: Config):
        self.config_parser = config_parser
        self.system_config = self._get_system_config_from_attributes(config_parser)
        self.logger = self._setup_logger()

        self.db_manager = None
        self.trading_calendar = TradingCalendarAnalyzer()

    def _get_system_config_from_attributes(self, config_parser) -> dict:
        return {
            "max_workers": config_parser.MAX_WORKERS,
            "data_fetch_retries": config_parser.DATA_FETCH_RETRIES,
            "data_fetch_delay": config_parser.DATA_FETCH_DELAY,
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _initialize_database(self):
        """初始化数据库连接"""
        db_config = {
            "user": self.config_parser.DB_USER,
            "password": self.config_parser.DB_PASSWORD,
            "host": self.config_parser.DB_HOST,
            "port": self.config_parser.DB_PORT,
            "database": self.config_parser.DB_NAME,
        }
        # 使用你现有的 QuantDBManager 初始化方式
        try:
            self.db_manager = QuantDBManager(
                db_config["user"],
                db_config["password"],
                db_config["host"],
                db_config["port"],
                db_config["database"]
            )
            self.logger.info("数据库连接初始化成功")
        except Exception as e:
            self.logger.error(f"数据库连接初始化失败: {e!s}")
            raise

    def _get_latest_data_date(self):
        """获取表中最新的 record_date"""
        if not self.db_manager:
            self._initialize_database()
        try:
            # 确认调用方法名与 DatabaseWriter.py 中的一致
            latest_date = self.db_manager.get_latest_record_date(self.TABLE_NAME, "record_date")
            self.logger.debug(f"从数据库获取到的最新日期: {latest_date}")
            return latest_date
        except AttributeError as e:
            self.logger.error(f"调用数据库方法失败，可能方法名不匹配: {e!s}")
            return None
        except Exception as e:
            self.logger.error(f"获取最新数据日期失败: {e!s}")
            return None

    def _is_data_up_to_date(self) -> bool:
        """
        核心校验逻辑：
        1. 超过30天 -> 强制刷新
        2. 小于30天 -> 校验表是否为空，以及接口行业数量与DB行业数量是否一致
        """
        latest_date = self._get_latest_data_date()
        today = datetime.now().date()

        # 1. 如果表为空（没有最新日期），直接返回 False 触发全量获取
        if not latest_date:
            self.logger.info(f"[缓存校验] 表 {self.TABLE_NAME} 中无数据，准备执行全量同步")
            return False

        # 兼容 PG 返回的 datetime.date 对象或字符串
        if isinstance(latest_date, str):
            latest_date = datetime.strptime(latest_date.replace("-", "")[:8], "%Y%m%d").date()
        elif hasattr(latest_date, 'date'): # 如果是 datetime.datetime 对象
             latest_date = latest_date.date()
             
        delta_days = (today - latest_date).days
        self.logger.info(f"[缓存校验] 数据库最新数据日期: {latest_date}，距今 {delta_days} 天")

        # 2. 判断是否超过 30 天
        if delta_days >= self.REFRESH_INTERVAL_DAYS:
            self.logger.info(f"[缓存校验] 数据已过期 (>= {self.REFRESH_INTERVAL_DAYS}天)，触发强制全量刷新")
            return False

        # 3. 小于 30 天，进入第二层校验：比对行业数量
        self.logger.info("[缓存校验] 数据在30天有效期内，开始校验行业数量是否发生变化...")
        
        try:
            # 获取 AkShare 接口当前的行业总数
            max_retries = self.system_config["data_fetch_retries"]
            delay = self.system_config["data_fetch_delay"]
            industry_df = None
            for attempt in range(max_retries):
                try:
                    industry_df = ak.sw_index_second_info()
                    self.logger.info(f"接口校验：成功获取 {len(industry_df)} 个申万二级行业")
                    break
                except Exception as e:
                    self.logger.warning(f"接口校验：获取行业列表第 {attempt + 1} 次失败: {e!s}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            
            if industry_df is None or industry_df.empty:
                self.logger.warning("[缓存校验] 无法从接口获取行业列表，为安全起见触发全量刷新")
                return False
                
            api_industry_count = len(industry_df)
            
            # 获取 PostgreSQL 表中去重后的行业总数
            # 确认调用方法名与 DatabaseWriter.py 中的一致
            db_industry_count = self.db_manager.get_distinct_count(self.TABLE_NAME, "industry_code")
            
            self.logger.info(f"[缓存校验] 接口行业数量: {api_industry_count} | 数据库行业数量: {db_industry_count}")
            
            if api_industry_count == db_industry_count:
                self.logger.info("[缓存校验] ✅ 行业数量一致，数据无需更新，直接使用本地缓存！")
                return True
            else:
                self.logger.info("[缓存校验] ⚠️ 行业数量发生变化，触发全量刷新")
                return False
                
        except AttributeError as e:
            self.logger.error(f"[缓存校验] 数据库管理器方法调用失败 (方法名可能不匹配): {e!s}，触发全量刷新")
            return False
        except Exception as e:
            self.logger.error(f"[缓存校验] 校验过程发生其他异常: {e!s}，触发全量刷新")
            return False

    def fetch_stock_basic_info(self) -> pd.DataFrame:
        """获取申万二级行业成分股（严格对齐 PG 表结构的 6 个字段）"""
        max_retries = self.system_config["data_fetch_retries"]
        base_delay = self.system_config["data_fetch_delay"]

        self.logger.info("开始从 AkShare 获取申万二级行业及成分股信息...")
        
        # 1. 获取申万二级行业列表
        industry_df = None
        for attempt in range(max_retries):
            try:
                industry_df = ak.sw_index_second_info()
                self.logger.info(f"成功获取 {len(industry_df)} 个申万二级行业")
                break
            except Exception as e:
                self.logger.warning(f"获取行业列表第 {attempt + 1} 次失败: {e!s}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                else:
                    raise

        all_stocks = []
        total = len(industry_df)
        
        # 获取计入日期，并强制转换为 PostgreSQL 认识的 'YYYY-MM-DD' 格式
        raw_record_date = str(self.trading_calendar.get_last_trading_day()).replace("-", "")
        record_date = f"{raw_record_date[:4]}-{raw_record_date[4:6]}-{raw_record_date[6:8]}"
        
        # 2. 遍历每个行业获取成分股
        for idx, row in industry_df.iterrows():
            raw_code = str(row["行业代码"]).strip()
            ind_name = str(row["行业名称"]).strip()
            
            symbol = raw_code.replace(".SI", "").replace(".si", "")
            self.logger.info(f"[{idx + 1}/{total}] 正在获取: {ind_name} ({symbol})")
            
            component_df = None
            # 为获取单个行业成分股增加更强的重试逻辑
            for attempt in range(max_retries):
                try:
                    # 在每次请求前增加一个基础延迟，避免请求过快
                    time.sleep(base_delay)
                    component_df = ak.index_component_sw(symbol=symbol)
                    
                    # 对获取到的DataFrame做初步校验，看是否有预期的列
                    if component_df is not None and not component_df.empty and '证券代码' in component_df.columns:
                        self.logger.debug(f"  成功获取 {ind_name} 的成分股数据，共 {len(component_df)} 条")
                        break # 如果成功获取且数据结构正确，跳出重试循环
                    else:
                        self.logger.warning(f"  第 {attempt + 1} 次获取到的成分股数据结构异常，尝试重试...")
                        
                except Exception as e:
                    self.logger.warning(f"  获取成分股第 {attempt + 1} 次失败: {e!s}")
                    # 每次失败后增加延迟时间 (指数退避)
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (attempt + 1))
            
            # 检查最终是否成功获取
            if component_df is None or component_df.empty or '证券代码' not in component_df.columns:
                self.logger.warning(f"  {ind_name} ({symbol}) 成分股数据获取失败或结构异常，跳过此行业。")
                # 即使失败也休眠一下，保持节奏
                time.sleep(10)
                continue # 跳过当前行业，继续下一个
                
            # 3. 映射字段
            for _, stock_row in component_df.iterrows():
                all_stocks.append({
                    "industry_code": raw_code,
                    "industry_name": ind_name,
                    "stock_code": str(stock_row.get("证券代码", "")).strip(),
                    "stock_name": str(stock_row.get("证券名称", "")).strip(),
                    "weight": stock_row.get("最新权重", 0.0),
                    "record_date": record_date
                })
            
            # 每次读取之间休息，防止请求过快
            time.sleep(3) 

        # 4. 数据清洗
        df = pd.DataFrame(all_stocks)
        if df.empty:
            self.logger.warning("所有行业均未能成功获取成分股数据")
            return df

        # 清洗 weight 列
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).astype(float)
        df = df.drop_duplicates(subset=["industry_code", "stock_code", "record_date"], keep="first").reset_index(drop=True)
        
        self.logger.info(f"成功获取并清洗完毕，共 {len(df)} 条成分股记录")
        cols = ["industry_code", "industry_name", "stock_code", "stock_name", "weight", "record_date"]
        return df[cols]

    def sync_all_stock_basic_info(self) -> bool:
        """同步成分股信息：根据缓存策略决定是刷新还是跳过"""
        try:
            if not self.db_manager:
                self._initialize_database()

            # 核心：执行双重校验
            if self._is_data_up_to_date():
                self.logger.info("缓存校验通过，无需更新数据。")
                return True  # 缓存有效，直接返回成功

            # 缓存失效，开始抓取
            df = self.fetch_stock_basic_info()
            if df.empty:
                self.logger.warning("获取到的数据为空")
                return False

            self.logger.info(f"[DB Sync] 开始全量刷新到 PostgreSQL 表 {self.TABLE_NAME}")

            # 清表并写入
            self.db_manager.truncate_and_insert(df, self.TABLE_NAME)
            self.logger.info(f"[DB Sync] 成功写入 {len(df)} 条记录。")

            return True

        except Exception as e:
            self.logger.error(f"同步成分股信息失败: {e!s}")
            return False

    def get_stock_count(self) -> int:
        """获取表中的记录总数"""
        if not self.db_manager:
            self._initialize_database()
        try:
            return self.db_manager.get_table_count(self.TABLE_NAME)
        except Exception as e:
            self.logger.error(f"获取记录总数失败: {e!s}")
            return 0

 