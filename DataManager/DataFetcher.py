import os
import time
from typing import Callable, Dict, Any, List, Optional
import pandas as pd
from UtilsManager.LoggerManager import LoggerManager

from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from UtilsManager.CacheManager import CacheManager




class DataFetcher:
    """
    统一的数据获取器，负责：
    1. 重试机制
    2. 缓存读写
    3. 可配置的数据清洗策略
    """

    def __init__(self, config: Config, calendar_mgr: TradingCalendarAnalyzer, logger: LoggerManager):
        self.config = config
        self.calendar_mgr = calendar_mgr
        self.logger = logger
        self.today_str = calendar_mgr.get_last_trading_day()
        self.temp_dir = config.TEMP_DATA_DIRECTORY
        # 初始化缓存管理器
        self.cache_manager = CacheManager(self.temp_dir, self.today_str, self.logger)



    def _clean_and_standardize(self, df: pd.DataFrame, df_name: str,
                               clean_pipeline: Optional[Callable] = None) -> pd.DataFrame:
        """
        通用数据清洗和列名标准化
        如果提供了clean_pipeline函数，则使用该函数进行清洗，否则使用默认清洗逻辑
        """
        if df.empty:
            return df

        # 如果提供了自定义清洗管道，则使用它
        if clean_pipeline:
            return clean_pipeline(df)

        # 默认清洗逻辑
        def extract_pure_code(code_str):
            if pd.isna(code_str):
                return None
            code_str = str(code_str).strip().upper()
            # 去掉 SH/SZ/BJ 前缀
            for prefix in ['SH', 'SZ', 'BJ']:
                if code_str.startswith(prefix):
                    code_str = code_str[2:]
                    break
            return code_str.zfill(6)

        def find_column_by_candidates(df: pd.DataFrame, candidates: list) -> str:
            """
            从候选列名列表中查找第一个存在的列
            
            Args:
                df: DataFrame
                candidates: 候选列名列表（支持大小写不敏感匹配）
                
            Returns:
                str: 找到的列名，未找到返回 None
            """
            # 先尝试精确匹配
            for col in candidates:
                if col in df.columns:
                    return col
            
            # 再尝试大小写不敏感匹配
            df_cols_lower = {col.lower(): col for col in df.columns}
            for col in candidates:
                if col.lower() in df_cols_lower:
                    return df_cols_lower[col.lower()]
            
            return None

        alias_mappings = [
            (self.config.CODE_ALIASES, '股票代码'),
            (self.config.NAME_ALIASES, '股票简称'),
            (self.config.PRICE_ALIASES, '最新价'),
        ]

        for aliases, target_col in alias_mappings:
            for old, new in aliases.items():
                if old in df.columns and new == target_col:
                    df.rename(columns={old: new}, inplace=True)
                    break  # 找到匹配即跳出

        # --- 2. 处理股票代码 ---
        if '股票代码' in df.columns:
            df['股票代码'] = df['股票代码'].astype(str).apply(extract_pure_code)
        else:
            # 尝试从通用列生成
            code_col = find_column_by_candidates(df, ['code', 'ts_code', 'symbol'])
            if code_col:
                df['股票代码'] = df[code_col].astype(str).apply(extract_pure_code)

            else:
                print(f"[ERROR] {df_name} 无代码字段！列名：{df.columns.tolist()}")
                return pd.DataFrame()

        # --- 3. 处理股票简称 (ST过滤) ---
        if '股票简称' not in df.columns:
            # 尝试从常见列获取
            name_col = find_column_by_candidates(df, ['name', '简称', 'symbol'])
            if name_col:
                df['股票简称'] = df[name_col]
            else:
                df['股票简称'] = 'N/A'
                print(f"[WARN] {df_name} 无简称列，使用占位符。")

        # ST股过滤 (统一正则)
        st_pattern = r'(?:\s*(?:\*|★|※|•|·))?(?:[Ss][Tt])'
        if (df['股票简称'].dtype == 'object' and
                df['股票简称'].astype(str).str.contains(st_pattern, na=False).any()):
            st_count = df['股票简称'].astype(str).str.contains(st_pattern, na=False).sum()
            df = df[~df['股票简称'].astype(str).str.contains(st_pattern, na=False)].copy()
            print(f"[FILTER] 已过滤 {st_count} 只ST股票。")

        # --- 4. 处理最新价 ---
        if '最新价' not in df.columns:
            price_col = find_column_by_candidates(df, ['price', 'close'])
            if price_col:
                df['最新价'] = pd.to_numeric(df[price_col], errors='coerce')
                print(f"[INFO] 已从 '{price_col}' 生成 '最新价' 列。")
            else:
                df['最新价'] = 0.0
                print(f"[WARN] {df_name} 无价格列，设为默认值 0.0。")

        # --- 5. 最终通用清洗 ---
        df.dropna(subset=['股票代码'], inplace=True)
        df.drop_duplicates(subset=['股票代码'], keep='first', inplace=True)
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)

        return df

    def fetch(self,
              fetch_func: Callable,
              file_base_name: str,
              clean_pipeline: Optional[Callable] = None,
              **kwargs: Any) -> pd.DataFrame:
        """
        统一的数据获取方法

        Args:
            fetch_func: 获取数据的函数
            file_base_name: 文件基础名称
            clean_pipeline: 可选的自定义清洗函数
            **kwargs: 传递给fetch_func的参数

        Returns:
            清洗后的DataFrame
        """
        # 1. 尝试从【清洗后的缓存】加载数据
        cached_df = self.cache_manager.load_cache(file_base_name, cleaned=True)
        if not cached_df.empty:
            return cached_df

        # 2. 如果清洗后的缓存不存在，则尝试从原始获取
        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                print(f"  - 正在尝试第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次获取数据: {file_base_name}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    break
                else:
                    # 使用指数级退避递增重试延时
                    wait_time = self.config.DATA_FETCH_DELAY * (2 ** i)
                    self.logger.warning(f"[WARN] 数据返回为空或无效: {file_base_name}，将在 {wait_time} 秒后重试。")
                    time.sleep(wait_time)
            except Exception as e:
                # 使用指数级退避递增重试延时
                wait_time = self.config.DATA_FETCH_DELAY * (2 ** i)
                self.logger.error(
                    f"[ERROR] 获取 {file_base_name} 时出错: {e}，将在 {wait_time} 秒后重试。")
                time.sleep(wait_time)

        if df.empty:
            self.logger.critical(f"[FATAL] 所有重试均失败，返回空 DataFrame: {file_base_name}")
            return pd.DataFrame()

        # 3. 清洗数据并保存到带有 "_经清洗" 后缀的缓存文件
        cleaned_df = self._clean_and_standardize(df, file_base_name, clean_pipeline)
        if not cleaned_df.empty:
            self.cache_manager.save_cache(cleaned_df, file_base_name, cleaned=True)

        return cleaned_df