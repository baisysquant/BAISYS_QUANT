from __future__ import annotations

import datetime
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

from ConfigParser import Config

warnings.filterwarnings('ignore')

class SWIndustryDataPipeline:
    """模块一：数据管道（负责拉取、清洗与本地缓存）"""
    
    def __init__(self, config: Config | None = None, today_str: str | None = None) -> None:
        self.config = config or Config()
        if today_str:
            self.today_str = today_str.replace("-", "")
        else:
            try:
                from DataCollection.CalendarManager import TradingCalendarAnalyzer
                cal = TradingCalendarAnalyzer()
                self.today_str = cal.get_last_trading_day().replace("-", "")
            except Exception:
                self.today_str = datetime.datetime.now().strftime("%Y%m%d")
        self.cache_dir = os.path.join(self.config.CACHE_DIRECTORY, "sw_data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"sw_hist_250d_{self.today_str}.parquet")
        self.cache_csv_file = os.path.join(self.cache_dir, f"sw_hist_250d_{self.today_str}.csv")
        self.valuation_file = os.path.join(self.cache_dir, f"sw_valuation_{self.today_str}.csv")

    def _map_hist_columns(self, df_hist: pd.DataFrame) -> pd.DataFrame:
        """
        【核心防御机制】为历史数据接口动态映射列名
        """
        if df_hist.empty:
            return df_hist

        mapping = {
            'code': ['代码'], # 新增映射，处理历史数据的代码列
            'date': ['日期', 'date', 'trade_date'],
            'open': ['开盘', 'open', 'O'],
            'high': ['最高', 'high', 'H'],
            'low': ['最低', 'low', 'L'],
            'close': ['收盘', 'close', 'C'],
            'volume': ['成交量', 'volume', 'vol', 'V'],
            'amount': ['成交额', 'amount', 'amt', 'A']
        }
        
        rename_dict = {}
        for col in df_hist.columns:
            for standard_name, candidates in mapping.items():
                if any(cand.lower() in str(col).lower() for cand in candidates):
                    rename_dict[col] = standard_name
                    break
        
        return df_hist.rename(columns=rename_dict)

    def fetch_and_cache_all(self, force_update: bool = False) -> pd.DataFrame | None:
        """遍历所有申万二级行业，拉取250天数据并缓存到本地"""
        hist_cache_exists = os.path.exists(self.cache_file) or os.path.exists(self.cache_csv_file)
        valuation_cache_exists = os.path.exists(self.valuation_file)
        
        if hist_cache_exists and valuation_cache_exists and not force_update:
            # 先读取缓存，检查其完整性
            try:
                cached_hist = pd.read_parquet(self.cache_file)
            except (OSError, ValueError, TypeError, ImportError):
                cached_hist = pd.read_csv(self.cache_csv_file, parse_dates=['date'])
            
            cached_val = pd.read_csv(self.valuation_file)
            cached_industry_count = len(cached_val)
            
            # [KEY] 关键：获取当前接口的行业总数
            df_info = ak.sw_index_second_info()
            current_total_industries = len(df_info)
            
            # [OK] 只有当缓存的行业数量 == 当前接口总数时，才使用缓存
            if cached_industry_count == current_total_industries:
                logger.info(f"缓存完整({cached_industry_count}个行业)，使用缓存数据")
                return cached_hist
            else:
                logger.warning(f"缓存不完整({cached_industry_count}个 vs {current_total_industries}个)，重新拉取...")


        logger.info("获取申万二级行业列表及估值数据...")
        try:
            df_info = ak.sw_index_second_info()
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            return None

        valuation_cols_map = {
            '行业代码': 'code',
            '行业名称': 'name',
            '静态市盈率': 'pe_static',
            'TTM(滚动)市盈率': 'pe_ttm',
            '市净率': 'pb',
            '静态股息率': 'div_yield'
        }
        
        missing_valuation_cols = [k for k in valuation_cols_map.keys() if k not in df_info.columns]
        if missing_valuation_cols:
            logger.warning(f"估值接口缺少预期列 {missing_valuation_cols}。")
        
        available_valuation_cols = {k: v for k, v in valuation_cols_map.items() if k in df_info.columns}
        df_val = df_info[list(available_valuation_cols.keys())].copy()
        df_val.columns = list(available_valuation_cols.values())
        # --- 智能提取纯数字代码 ---
        # 使用正则表达式，提取字符串开头的连续数字部分
        df_val['code'] = df_val['code'].astype(str).apply(lambda x: re.match(r'^(\d+)', x).group(1) if re.match(r'^(\d+)', x) else x)
        df_val.to_csv(self.valuation_file, index=False, encoding='utf-8-sig')

        codes = df_val['code'].astype(str).tolist()
        names = df_val['name'].astype(str).tolist()
        
        print(f"  ↓ 拉取 {len(codes)} 个行业K线数据...", flush=True)
        logger.info(f"开始并行拉取 {len(codes)} 个行业的250天历史量价数据 (2线程)...")
        all_hist_data = []

        def fetch_one(code: str, name: str) -> pd.DataFrame | None:
            try:
                df_hist = ak.index_hist_sw(symbol=code, period="day")
                if df_hist is not None and not df_hist.empty:
                    df_hist_mapped = self._map_hist_columns(df_hist)
                    required_core_cols = ['date', 'close', 'volume', 'amount']
                    if not all(c in df_hist_mapped.columns for c in required_core_cols):
                        logger.warning(f"{code} 历史数据映射后缺少核心字段，跳过。")
                        return None
                    core_cols = [c for c in ['date', 'close', 'open', 'high', 'low', 'volume', 'amount'] if c in df_hist_mapped.columns]
                    df_sub = df_hist_mapped[core_cols].copy()
                    df_sub['date'] = pd.to_datetime(df_sub['date'])
                    df_sub = df_sub.sort_values('date').tail(250).reset_index(drop=True)
                    df_sub['code'] = code
                    df_sub['name'] = name
                    return df_sub
            except Exception as e:
                logger.warning(f"获取 {code} ({name}) 失败 -> {e}")
            return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(fetch_one, codes[i], names[i]): i for i in range(len(codes))}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    all_hist_data.append(result)
                if (idx + 1) % 5 == 0 or idx == len(codes) - 1:
                    print(f"  \r  行业进度: {len(all_hist_data)}/{len(codes)}", end="", flush=True)
                    logger.info(f"进度: {len(all_hist_data)}/{len(codes)}")
                time.sleep(0.1)

        print(f"\r  行业进度: {len(all_hist_data)}/{len(codes)} (完成)", flush=True)
        if not all_hist_data:
            logger.error("未能获取任何历史数据。")
            return None

        logger.info("数据合并与本地缓存...")
        df_all = pd.concat(all_hist_data, ignore_index=True)
        
        # 尝试保存为Parquet，如果失败则保存为CSV
        try:
            df_all.to_parquet(self.cache_file, index=False)
            logger.info(f"成功缓存 {len(df_all)} 条数据至 {self.cache_file} (Parquet格式)")
        except Exception:
            # 如果pyarrow或fastparquet不可用，则保存为CSV
            df_all.to_csv(self.cache_csv_file, index=False, encoding='utf-8-sig')
            logger.info(f"成功缓存 {len(df_all)} 条数据至 {self.cache_csv_file} (CSV格式)")
            
        return df_all


class SWMultiFactorModel:
    """模块二：多因子计算引擎（纯本地向量化计算，极速）"""
    
    def __init__(self, pipeline: SWIndustryDataPipeline) -> None:
        self.pipeline = pipeline
        self.ma_periods = [10, 20, 30, 60, 90]

    def _calculate_vectorized_factors(self, df_hist: pd.DataFrame) -> pd.DataFrame:
        """利用 GroupBy 进行向量化计算"""
        df = df_hist.sort_values(['code', 'date']).copy()
        
        for p in self.ma_periods:
            df[f'ma_{p}'] = df.groupby('code')['close'].transform(lambda x: x.rolling(p).mean())
            
        df['vol_ma_20'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
        df['amt_ma_60'] = df.groupby('code')['amount'].transform(lambda x: x.rolling(60).mean())
        
        df_latest = df.groupby('code').tail(1).set_index('code')
        
        bull_score = pd.Series(0, index=df_latest.index)
        mas = [df_latest[f'ma_{p}'] for p in self.ma_periods]
        for i in range(len(mas)-1):
            bull_score += (mas[i] > mas[i+1]).astype(int)
        df_latest['bull_align_score'] = bull_score
        
        df_latest['dev_20'] = (df_latest['close'] - df_latest['ma_20']) / df_latest['ma_20'] * 100
        df_latest['dev_60'] = (df_latest['close'] - df_latest['ma_60']) / df_latest['ma_60'] * 100
        
        df_latest['vol_ratio'] = df_latest['volume'] / df_latest['vol_ma_20']
        df_latest['amt_ratio'] = df_latest['amount'] / df_latest['amt_ma_60']
        
        # 只保留因子计算相关的列，避免与估值数据中的name列冲突
        return df_latest[['name', 'close', 'bull_align_score', 'dev_20', 'dev_60', 'vol_ratio', 'amt_ratio']]

    def run_scoring(self) -> pd.DataFrame:
        """执行完整的打分流程"""
        # 尝试读取Parquet，如果失败则读取CSV
        try:
            df_hist = pd.read_parquet(self.pipeline.cache_file)
        except (ImportError, OSError, ValueError, TypeError):
            df_hist = pd.read_csv(self.pipeline.cache_csv_file, parse_dates=['date'])
        
        df_val = pd.read_csv(self.pipeline.valuation_file)
        
        logger.info("正在执行向量化因子计算...")
        df_factors = self._calculate_vectorized_factors(df_hist)
        
        # 解决列名冲突：重命名因子数据中的name列为factor_name
        df_factors_renamed = df_factors.copy()
        df_factors_renamed = df_factors_renamed.rename(columns={'name': 'factor_name'})
        
        # 合并数据时指定suffixes来避免重复列名
        df = df_factors_renamed.join(df_val.set_index('code'), how='left', rsuffix='_val')
        
        # 使用估值数据中的name列覆盖因子数据中的name列
        df['name'] = df['name']
        
        # 数据清洗
        for col in ['pe_ttm', 'pb']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] <= 0, col] = np.nan
        df['div_yield'] = pd.to_numeric(df['div_yield'], errors='coerce').fillna(0)
        
        # 截面标准化打分 (0-100)
        df['score_pe'] = 100 - (df['pe_ttm'].rank(pct=True) * 100)
        df['score_pb'] = 100 - (df['pb'].rank(pct=True) * 100)
        df['score_div'] = df['div_yield'].rank(pct=True) * 100
        df['factor_value'] = df['score_pe']*0.4 + df['score_pb']*0.3 + df['score_div']*0.3
        
        df['score_bull'] = (df['bull_align_score'] / 4) * 100
        df['score_mom'] = df['dev_60'].rank(pct=True) * 100 
        df['factor_trend'] = df['score_bull']*0.5 + df['score_mom']*0.5
        
        df['score_vol'] = df['vol_ratio'].rank(pct=True) * 100
        df['score_amt'] = df['amt_ratio'].rank(pct=True) * 100
        df['factor_volume'] = df['score_vol']*0.5 + df['score_amt']*0.5
        
        df['total_score'] = (
            df['factor_value'].fillna(50) * 0.35 +  # 估值缺失给中性分50
            df['factor_trend'] * 0.40 + 
            df['factor_volume'] * 0.25
        ).round(2)
        
        def get_signal(row: pd.Series) -> str:
            if row['total_score'] > 75 and row['factor_value'] > 70:
                return "核心配置 (低估值+强趋势)"
            elif row['total_score'] > 70 and row['factor_trend'] > 80:
                return "动量追击 (高景气+资金涌入)"
            elif row['factor_value'] > 85 and row['factor_trend'] < 40:
                return "左侧潜伏 (极度低估+等待拐点)"
            elif row['factor_trend'] > 80 and row['factor_value'] < 30:
                return "情绪过热 (高估+趋势透支)"
            else:
                return "均衡/观望"
                
        df['signal'] = df.apply(get_signal, axis=1)
        return df.sort_values('total_score', ascending=False)


class IndustryFlowAnalyzer:
    """兼容主程序调用链的行业分析适配器。"""

    def __init__(self, config: Config | None = None, today_str: str | None = None) -> None:
        self.config = config
        self.pipeline = SWIndustryDataPipeline(config=config, today_str=today_str)
        self.model = SWMultiFactorModel(self.pipeline)

    @staticmethod
    def _output_columns() -> list[str]:
        return [
            '行业代码', '行业名称', '行业信号', '综合得分', '趋势得分', '估值得分', '量能得分',
            'PE_TTM', 'PB', '股息率', '多头排列分', '20日偏离率', '60日偏离率', '量比', '额比'
        ]

    def _format_main_output(self, result_df: pd.DataFrame) -> pd.DataFrame:
        if result_df is None or result_df.empty:
            return pd.DataFrame(columns=self._output_columns())

        df = result_df.reset_index().copy()
        df = df.rename(columns={
            'code': '行业代码',
            'name': '行业名称',
            'signal': '行业信号',
            'total_score': '综合得分',
            'factor_trend': '趋势得分',
            'factor_value': '估值得分',
            'factor_volume': '量能得分',
            'pe_ttm': 'PE_TTM',
            'pb': 'PB',
            'div_yield': '股息率',
            'bull_align_score': '多头排列分',
            'dev_20': '20日偏离率',
            'dev_60': '60日偏离率',
            'vol_ratio': '量比',
            'amt_ratio': '额比',
        })

        for col in self._output_columns():
            if col not in df.columns:
                df[col] = '' if col in ['行业代码', '行业名称', '行业信号'] else np.nan

        df['行业名称'] = df['行业名称'].fillna('').astype(str).str.strip()
        df['行业信号'] = df['行业信号'].fillna('').astype(str).str.strip()
        return df[self._output_columns()]

    def run_analysis(self, force_update: bool = False) -> pd.DataFrame:
        df_hist = self.pipeline.fetch_and_cache_all(force_update=force_update)
        if df_hist is None or df_hist.empty:
            return pd.DataFrame(columns=self._output_columns())

        result_df = self.model.run_scoring()
        return self._format_main_output(result_df)
 
