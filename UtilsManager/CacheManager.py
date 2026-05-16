"""
缓存管理器模块

提供统一的数据缓存读写接口，避免在多个类中重复实现缓存逻辑。
支持自动识别交易日、文件路径生成、数据标准化等功能。
"""

import os
from typing import Optional
import pandas as pd
from UtilsManager.LoggerManager import LoggerManager


class CacheManager:
    """
    统一缓存管理器
    
    职责：
    - 生成标准化的缓存文件路径
    - 加载缓存数据（支持多种编码和分隔符）
    - 保存数据到缓存
    - 检查缓存有效性
    
    Attributes:
        temp_dir: 临时数据目录
        today_str: 当前交易日字符串
        logger: 日志管理器
    """
    
    def __init__(self, temp_dir: str, today_str: str, logger: LoggerManager):
        """
        初始化缓存管理器
        
        Args:
            temp_dir: 临时数据目录路径
            today_str: 当前交易日字符串（格式：YYYY-MM-DD）
            logger: 日志管理器实例
        """
        self.temp_dir = temp_dir
        self.today_str = today_str
        self.logger = logger
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _get_file_path(self, base_name: str, cleaned: bool = False, 
                       suffix: str = ".txt") -> str:
        """
        生成缓存文件的完整路径
        
        Args:
            base_name: 文件基础名称
            cleaned: 是否为清洗后的数据（添加"_经清洗"后缀）
            suffix: 文件扩展名，默认 .txt
            
        Returns:
            str: 完整的文件路径
        """
        clean_suffix = "_经清洗" if cleaned else ""
        file_name = f"{base_name}{clean_suffix}_{self.today_str}{suffix}"
        return os.path.join(self.temp_dir, file_name)
    
    def load_cache(self, base_name: str, cleaned: bool = True,
                   sep: str = '|', encoding: str = 'utf-8',
                   dtype_mapping: Optional[dict] = None) -> pd.DataFrame:
        """
        从缓存加载数据
        
        Args:
            base_name: 文件基础名称
            cleaned: 是否加载清洗后的缓存（默认True）
            sep: CSV分隔符，默认 '|'
            encoding: 文件编码，默认 'utf-8'
            dtype_mapping: 列数据类型映射字典
            
        Returns:
            pd.DataFrame: 加载的数据，如果缓存不存在或加载失败则返回空DataFrame
        """
        file_path = self._get_file_path(base_name, cleaned=cleaned)
        
        if not os.path.exists(file_path):
            return pd.DataFrame()
        
        try:
            # 设置默认的类型映射
            if dtype_mapping is None:
                dtype_mapping = {'股票代码': str, 'symbol': str}
            
            df = pd.read_csv(file_path, sep=sep, encoding=encoding, dtype=dtype_mapping)
            
            # 统一列名为 '股票代码'
            if 'symbol' in df.columns and '股票代码' not in df.columns:
                df.rename(columns={'symbol': '股票代码'}, inplace=True)
            
            if self.logger:
                self.logger.info(f"  - 发现缓存，加载: {os.path.basename(file_path)}")
            else:
                print(f"  - 发现缓存，加载: {os.path.basename(file_path)}")
            return df
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[WARN] 加载缓存 {os.path.basename(file_path)} 失败: {e}，将重新获取。")
            else:
                print(f"[WARN] 加载缓存 {os.path.basename(file_path)} 失败: {e}，将重新获取。")
            return pd.DataFrame()
    
    def save_cache(self, df: pd.DataFrame, base_name: str,
                   cleaned: bool = True, sep: str = '|',
                   encoding: str = 'utf-8') -> bool:
        """
        保存数据到缓存
        
        Args:
            df: 要保存的DataFrame
            base_name: 文件基础名称
            cleaned: 是否保存为清洗后的数据（默认True）
            sep: CSV分隔符，默认 '|'
            encoding: 文件编码，默认 'utf-8'
            
        Returns:
            bool: 保存是否成功
        """
        if df is None or df.empty:
            return False
        
        file_path = self._get_file_path(base_name, cleaned=cleaned)
        
        try:
            df.to_csv(file_path, sep=sep, index=False, encoding=encoding)
            if self.logger:
                self.logger.info(f"  - 保存数据至缓存: {os.path.basename(file_path)}")
            else:
                print(f"  - 保存数据至缓存: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[ERROR] 保存数据到缓存 {os.path.basename(file_path)} 失败: {e}")
            else:
                print(f"[ERROR] 保存数据到缓存 {os.path.basename(file_path)} 失败: {e}")
            return False
    
    def cache_exists(self, base_name: str, cleaned: bool = True) -> bool:
        """
        检查缓存文件是否存在
        
        Args:
            base_name: 文件基础名称
            cleaned: 是否检查清洗后的缓存
            
        Returns:
            bool: 缓存是否存在
        """
        file_path = self._get_file_path(base_name, cleaned=cleaned)
        return os.path.exists(file_path)
    
    def get_cache_path(self, base_name: str, cleaned: bool = True,
                       suffix: str = ".txt") -> str:
        """
        获取缓存文件的完整路径（不检查是否存在）
        
        Args:
            base_name: 文件基础名称
            cleaned: 是否为清洗后的数据
            suffix: 文件扩展名
            
        Returns:
            str: 完整的文件路径
        """
        return self._get_file_path(base_name, cleaned=cleaned, suffix=suffix)
