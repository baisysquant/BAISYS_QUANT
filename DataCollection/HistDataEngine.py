import os
import akshare as ak
import pandas as pd
import datetime
import time
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from UtilsManager.CacheManager import CacheManager
import urllib3.util
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import tushare as ts
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import List, Set, Optional
from DataManager.ShareCodeFormatMgr import format_stock_code
from ConfigParser import Config


class StockSyncEngine:
    AKSHARE_RETRIES = 3
    AKSHARE_DELAY = 5

    def __init__(self, config_file: str = "config.ini"):

        self.config_file = config_file
        self.config = Config(config_file=config_file)
        self.token = self.config.TUSHARE_TOKEN

        url_object = URL.create(
            "postgresql+psycopg2",
            username=self.config.DB_USER,
            password=self.config.DB_PASSWORD,
            host=self.config.DB_HOST,
            port=self.config.DB_PORT,
            database=self.config.DB_NAME
        )

        #  初始化数据库引擎
        self.db = create_engine(
            url_object,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            client_encoding='utf8'

        )

        #  测试数据库连接
        try:
            with self.db.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("[INFO]  数据库引擎初始化成功")
        except Exception as e:
            raise RuntimeError("数据库引擎初始化失败") from e

        self.global_start = "20250301"
        # 使用业务交易日而非物理时间
        calendar_mgr = TradingCalendarAnalyzer()
        self.today = calendar_mgr.get_last_trading_day()
        self.today_dt = pd.to_datetime(self.today).normalize()
        self.base_data_dir = self.config.TEMP_DATA_DIRECTORY
        os.makedirs(self.base_data_dir, exist_ok=True)

        # 初始化缓存管理器
        self.cache_manager = CacheManager(
            temp_dir=self.base_data_dir,
            today_str=self.today,
            logger=None  # HistDataEngine 不使用 LoggerManager，直接用 print
        )

        # 缓存文件路径（使用 CacheManager 生成）
        self.main_report_cache_path = self.cache_manager.get_cache_path(
            "主力研报盈利预测_完整数据", cleaned=True, suffix=".csv"
        )
        self.raw_report_cache_path = self.cache_manager.get_cache_path(
            "主力研报盈利预测", cleaned=True
        )
        self.kline_cache_path = self.cache_manager.get_cache_path(
            "股票K线数据_已处理", cleaned=True, suffix=".csv"
        )

    def get_main_board_pool(self) -> pd.DataFrame:
        """
        从数据库获取主板股票池
        返回包含 ts_code、name、industry
        """
        try:
            # 直接从数据库查询股票基本信息
            query = """
            SELECT 
                ts_code,
                symbol as 股票代码,
                name,
                industry
            FROM stock_basic_info
            ORDER BY symbol
            """

            with self.db.connect() as conn:
                stock_index_df = pd.read_sql(text(query), conn)

            print(f"[INFO] 从数据库获取 {len(stock_index_df)} 只股票。")

            # 标准化股票代码格式（确保6位数字）
            if '股票代码' in stock_index_df.columns:
                stock_index_df['股票代码'] = stock_index_df['股票代码'].astype(str).str.zfill(6)

            required_cols = ['ts_code', 'name', 'industry', '股票代码']
            for col in required_cols:
                if col not in stock_index_df.columns:
                    stock_index_df[col] = 'N/A'

            return stock_index_df[required_cols]

        except Exception as e:
            print(f"[ERROR] 从数据库获取股票池失败: {e}")
            # 如果数据库查询失败，返回空DataFrame
            return pd.DataFrame(columns=['ts_code', 'name', 'industry', '股票代码'])

    def _safe_ak_fetch(self, fetch_func: callable, description: str, 
                       cache_base_name: str = None, **kwargs) -> pd.DataFrame:
        """带重试、缓存、清洗的 Akshare 数据获取。"""

        def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df.columns = [col.strip() for col in df.columns]
            if '代码' in df.columns and '股票代码' not in df.columns:
                df.rename(columns={'代码': '股票代码'}, inplace=True)
            if '名称' in df.columns and '股票简称' not in df.columns:
                df.rename(columns={'名称': '股票简称'}, inplace=True)
            return df

        # 1. 尝试加载清洗后的缓存
        if cache_base_name:
            cached_df = self.cache_manager.load_cache(
                cache_base_name, 
                cleaned=True,
                sep='|',
                encoding='utf-8-sig',
                dtype_mapping={
                    '股票代码': str, '股票简称': str,
                    '机构投资评级(近六个月)-买入': float,
                    '2024预测每股收益': float, '2025预测每股收益': float
                }
            )
            if not cached_df.empty:
                cached_df = _standardize_columns(cached_df)
                print(f"  -  从缓存加载: {os.path.basename(self.cache_manager.get_cache_path(cache_base_name, cleaned=True))}")
                return cached_df

        df = pd.DataFrame()
        for i in range(self.AKSHARE_RETRIES):
            try:
                print(f"  - 正在尝试第 {i + 1}/{self.AKSHARE_RETRIES} 次: {description}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    df = _standardize_columns(df)
                    break
                time.sleep(self.AKSHARE_DELAY)
            except Exception as e:
                print(f"[ERROR] 获取 {description} 失败: {e}，将在 {self.AKSHARE_DELAY} 秒后重试")
                time.sleep(self.AKSHARE_DELAY)

        if df.empty:
            print(f"[CRITICAL] 所有重试失败，未能获取 {description}")
            return pd.DataFrame()

        # 清洗 + 保存到缓存
        if '股票代码' in df.columns:
            df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
            df = df.drop_duplicates(subset=['股票代码'])

        if cache_base_name and not df.empty:
            self.cache_manager.save_cache(
                df, 
                cache_base_name, 
                cleaned=True,
                sep='|',
                encoding='utf-8-sig'
            )

        return df

    def _get_research_report_filtered_symbols(self) -> Set[str]:
        """
        获取研报"机构投资评级(近六个月)-买入" > 1 的股票代码集合。
        返回纯数字代码（如 '000001'）。
        """
        print("\n>>> 正在获取主力研报盈利预测并进行过滤...")

        # 获取原始数据
        report_df = self._safe_ak_fetch(
            fetch_func=ak.stock_profit_forecast_em,
            description="主力研报盈利预测",
            cache_base_name="主力研报盈利预测"
        )

        if report_df.empty:
            print("[WARNING] 未获取到研报数据，返回空集合")
            return set()

        # 标准化列名
        if '股票代码' not in report_df.columns:
            report_df.rename(columns={'代码': '股票代码'}, inplace=True)
        if '机构投资评级(近六个月)-买入' not in report_df.columns:
            report_df.rename(columns={'买入 (次)': '机构投资评级(近六个月)-买入'}, inplace=True)

        # 清洗
        report_df = report_df.drop_duplicates(subset=['股票代码'])
        report_df['股票代码'] = report_df['股票代码'].astype(str).str.zfill(6)

        # 过滤
        report_df['机构投资评级(近六个月)-买入'] = pd.to_numeric(report_df['机构投资评级(近六个月)-买入'],
                                                                 errors='coerce').fillna(0)
        qualified = report_df[report_df['机构投资评级(近六个月)-买入'] > 0]['股票代码'].unique()

        result = set(qualified)
        print(f"[INFO] 过滤后剩余 {len(result)} 只符合条件的股票。")

        return result

    def _fetch_kline_for_symbol(self, symbol: str) -> pd.DataFrame:
        """获取单个股票的前复权 + 不复权数据，合并输出"""
        try:
            df_qfq = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=self.global_start, end_date=self.today,
                                           adjust="qfq")
            time.sleep(0.05)
            if df_qfq.empty:
                return None

            expected_cols = ['date', 'open', 'close', 'high', 'low', 'amount']
            missing = [c for c in expected_cols if c not in df_qfq.columns]
            if missing:
                print(f"[ERROR] QFQ 数据缺失列: {missing}")
                return None

            df_norm = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=self.global_start, end_date=self.today, adjust="")
            time.sleep(0.05)
            if df_norm.empty:
                return None

            df_norm = df_norm[['date', 'close', 'amount']].rename(
                columns={'close': 'close_normal', 'amount': 'volume_normal'})
            df = pd.merge(df_qfq, df_norm, on='date', how='inner')
            if df.empty:
                return None

            df['close'] = pd.to_numeric(df['close'])
            df['close_normal'] = pd.to_numeric(df['close_normal'])
            df['adj_ratio'] = df['close'] / df['close_normal'].replace(0, pd.NA)

            # 计算复权成交量
            df['amount'] = pd.to_numeric(df['amount'])
            df['volume_normal'] = pd.to_numeric(df['volume_normal'])
            df['volume_adj_ratio'] = df['amount'] / df['volume_normal'].replace(0, pd.NA)
            df['volume'] = df['volume_normal'] * df['volume_adj_ratio']

            df.dropna(subset=['adj_ratio'], inplace=True)

            df['symbol'] = format_stock_code(symbol)
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date': 'trade_date'}, inplace=True)

            final_cols = [
                'trade_date', 'symbol', 'open', 'close', 'high', 'low', 'amount',
                'close_normal', 'volume', 'adj_ratio'
            ]
            return df[final_cols]
        except Exception as e:
            print(f"[ERROR] 获取 {symbol} 数据失败: {e}")
            return None

    def _clear_stock_daily_kline_table(self):
        """清空 stock_daily_kline 表"""
        if self.db is None:
            print("[CRITICAL] 数据库未初始化")
            return
        try:
            with self.db.connect() as conn:
                conn.execute(text("DELETE FROM stock_daily_kline;"))
                conn.commit()
            print("[INFO] 'stock_daily_kline' 表已清空。")
        except Exception as e:
            print(f"[ERROR] 清空失败: {e}")
            raise

    def run_engine(self, target_date: str = None):
        """主运行函数：研报过滤 + K线数据同步"""

        if target_date is None:
            # 优先使用业务交易日，如果失败则使用初始化时的 today
            try:
                target_date = TradingCalendarAnalyzer().get_last_trading_day()
            except Exception:
                target_date = self.today  # 使用初始化时获取的业务交易日

        self.today_str = target_date
        self.today_dt = pd.to_datetime(target_date).normalize()

        print(f"[DEBUG] 数据引擎运行日期: {self.today_str}")

        # Step 1: 获取数据库中的股票池
        tushare_df = self.get_main_board_pool()
        if tushare_df.empty or '股票代码' not in tushare_df.columns:
            print("[CRITICAL] 基础股票池无效")
            return
        tushare_pure_codes = set(tushare_df['股票代码'].unique().tolist())
        print(f"[INFO] 从数据库获取 {len(tushare_pure_codes)} 只股票。")

        # Step 2: 获取研报符合条件的股票
        report_codes = self._get_research_report_filtered_symbols()
        if not report_codes:
            print("[WARNING] 研报过滤后无股票，数据库将清空")
            self._clear_stock_daily_kline_table()
            return

        # Step 3: 交集
        final_codes = tushare_pure_codes.intersection(report_codes)
        if not final_codes:
            print("[CRITICAL] 无共同股票，停止任务")
            return
        print(f"[INFO] 双重过滤后保留 {len(final_codes)} 只股票。")

        #  Step 4: 缓存检查 → 决定是否重取

        if os.path.exists(self.kline_cache_path):
            print(f" 缓存文件存在: {os.path.basename(self.kline_cache_path)}")
            try:
                df = pd.read_csv(self.kline_cache_path, sep='|', encoding='utf-8-sig')
                # 确保字段存在
                expected_cols = ['trade_date', 'symbol', 'open', 'close', 'high', 'low', 'amount', 'close_normal',
                                 'volume', 'adj_ratio']
                missing = [c for c in expected_cols if c not in df.columns]
                if missing:
                    print(f"[WARN] 缓存缺少列: {missing}，将重取")
                    df = pd.DataFrame()
                else:
                    # 将 symbol 明确转换为 sh/sz/bj 格式（可选）
                    df['symbol'] = df['symbol'].astype(str)
                    print(f" 成功加载缓存，共 {len(df)} 条记录。")

                    try:
                        df.to_sql(
                            name='stock_daily_kline',
                            con=self.db,
                            if_exists='replace',
                            index=False,
                            method='multi',
                            chunksize=5000
                        )
                        print(f"成功将 {len(df)} 条记录写入 'stock_daily_kline' 表。")
                    except Exception as e:
                        print(f"[ERROR] 写入数据库失败: {e}")
                        raise

                    final_output_path = os.path.join(self.base_data_dir, f"final_filtered_stocks_{self.today}.txt")
                    try:
                        with open(final_output_path, 'w', encoding='utf-8') as f:
                            for code in sorted(final_codes):
                                f.write(f"{code}\n")
                        print(f"[INFO] 最终筛选代码已保存至: {final_output_path}")
                    except Exception as e:
                        print(f"[ERROR] 保存最终代码列表失败: {e}")

                    print(f"  - 今日日期: {self.today}")
                    print(f"  - 筛选股票数: {len(final_codes)}")

                    return

            except Exception as e:
                print(f"[WARN] 缓存加载失败: {e}")

        filtered_codes = final_codes  # ←  这才是真正的"最终要处理的股票"
        print(f"[INFO]  将获取 {len(filtered_codes)} 只股票的 K 线（基于交集结果）。")

        #  Step 6: 获取最终 Akshare 格式代码
        akshare_symbols = []
        for code in filtered_codes:
            code_str = code.zfill(6)
            if code_str.startswith('6'):
                akshare_symbols.append('sh' + code_str)
            elif code_str.startswith(('0', '3')):
                akshare_symbols.append('sz' + code_str)
            elif code_str.startswith(('4', '8', '9')):
                akshare_symbols.append('bj' + code_str)
            else:
                print(f"[WARN] 无法识别代码: {code_str}，跳过。")

        #  Step 7: 并发获取 K 线数据（多线程）
        print(f"[INFO] 正在并发获取 {len(akshare_symbols)} 只股票的 K 线数据...")
        kline_dfs = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(self._fetch_kline_for_symbol, sym): sym for sym in akshare_symbols}
            for future in as_completed(futures):
                result = future.result()
                symbol = futures[future]
                if result is not None and not result.empty:
                    kline_dfs.append(result)
                else:
                    print(f"[WARN] 获取 {symbol} 的 K 线失败，跳过。")

        #  Step 8: 合并所有 K 线数据
        if not kline_dfs:
            print("[WARNING] 所有股票 K 线获取失败，数据库将清空。")
            self._clear_stock_daily_kline_table()
            return

        combined_kline_df = pd.concat(kline_dfs, ignore_index=True)
        print(f"[INFO] 成功合并 {len(combined_kline_df)} 条 K 线记录。")
        try:
            # 保存为 CSV，不包含 index
            combined_kline_df.to_csv(
                self.kline_cache_path,
                sep='|',
                index=False,
                encoding='utf-8-sig'
            )
            print(f"[INFO]  K线数据已保存至本地缓存: {os.path.basename(self.kline_cache_path)}")
        except Exception as e:
            print(f"[ERROR] 保存 K线数据缓存失败: {e}")

        #  Step 9: 写入数据库
        try:
            combined_kline_df.to_sql(
                name='stock_daily_kline',
                con=self.db,
                if_exists='replace',  # 替换已有表
                index=False,
                method='multi',
                chunksize=5000
            )
            print(f"[INFO]  成功将 {len(combined_kline_df)} 条记录写入 'stock_daily_kline' 表。")
        except Exception as e:
            print(f"[ERROR] 写入数据库失败: {e}")
            raise

        print(f"  - 今日日期: {self.today}")
        print(f"  - 筛选股票数: {len(filtered_codes)}")
        print(f"  - 成功获取 K 线股票: {len(kline_dfs)}")

        #  可选：保存最终过滤列表
        final_output_path = os.path.join(self.base_data_dir, f"final_filtered_stocks_{self.today}.txt")
        try:
            with open(final_output_path, 'w', encoding='utf-8') as f:
                for code in sorted(filtered_codes):
                    f.write(f"{code}\n")
            print(f"[INFO] 最终筛选代码已保存至: {final_output_path}")
        except Exception as e:
            print(f"[ERROR] 保存最终代码列表失败: {e}")
