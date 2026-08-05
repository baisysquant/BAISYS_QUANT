from __future__ import annotations

import datetime
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import requests
from asharehub import AShareHub
from loguru import logger

from UtilsManager.ConfigParser import Config

class SWIndustryDataPipeline:
    """模块一：数据管道（负责拉取、清洗与本地缓存）"""
    
    def __init__(self, config: Config | None = None, today_str: str | None = None) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.config = config or Config()
            self.ah_client = AShareHub(api_key=self.config.ASHAREHUB_API_KEY)
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
            'code': ['代码', '^^^^'], # 新增映射，处理历史数据的代码列
            'date': ['日期', 'date', 'trade_date', '^^^^'],
            'open': ['开盘', 'open', 'O', '^^^^'],
            'high': ['最高', 'high', 'H', '^^^^'],
            'low': ['最低', 'low', 'L', '^^^'],
            'close': ['收盘', 'close', 'C', '^^^'],
            'volume': ['成交量', 'volume', 'vol', 'V', '^成交量^'],
            'amount': ['成交额', 'amount', 'amt', 'A', '^成交额^']
        }
        
        rename_dict = {}
        for col in df_hist.columns:
            for standard_name, candidates in mapping.items():
                if any(cand.lower() in str(col).lower() for cand in candidates):
                    rename_dict[col] = standard_name
                    break
        
        return df_hist.rename(columns=rename_dict)

    @staticmethod
    def _aggregate_industry_valuation(
        ah_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        df_val: pd.DataFrame,
        agg_method: str = "aggregate_profitable",
    ) -> pd.DataFrame:
        """按申万二级行业市值加权聚合个股估值（AShareHub 无行业板块估值接口）。

        口径（与 agg_method 相关）：
          - aggregate_profitable（中证口径）：PE(静态/TTM)/PB 为市值加权调和平均
            （= 行业总市值 / 行业盈利股总盈利），剔除亏损股（pe/pb <= 0 或 NaN）；
            行业全亏损时 PE 为空
          - aggregate_full（申万口径）：整体法含负利润 —— 亏损股利润计入分母，
            PE = 行业总市值 / 全样本总盈利，行业整体亏损时 PE 为负；盈亏平衡时为空
          - 股息率：总市值加权平均（= 行业总股息 / 行业总市值），剔除缺失个股
          - 市值权重用 total_mv（万元，与接口口径一致）
        """
        l2_map = ah_df[["symbol", "l2_code"]].dropna().drop_duplicates("symbol")
        est = fundamentals_df.merge(l2_map, on="symbol", how="inner")
        if est.empty:
            return df_val

        est["_mv"] = pd.to_numeric(est["total_mv"], errors="coerce")
        mv_ok = est["_mv"].notna() & (est["_mv"] > 0)
        est["_pe"] = pd.to_numeric(est["pe"], errors="coerce")
        est["_pe_ttm"] = pd.to_numeric(est["pe_ttm"], errors="coerce")
        est["_pb"] = pd.to_numeric(est["pb"], errors="coerce")
        est["_dv"] = pd.to_numeric(est["dv_ratio"], errors="coerce")

        def _harmonic(sub: pd.DataFrame, col: str, include_loss: bool) -> float:
            has_val = sub[col].notna()
            if include_loss:
                # 整体法（申万口径）：分子=全行业总市值（含无估值/亏损股），
                # 分母=Σ(市值/PE) 仅计有 PE 的股票（亏损股负利润计入）
                m_denom = has_val & (sub[col] != 0)
                if not m_denom.any():
                    return np.nan
                _mv_sum = sub.loc[mv_ok[sub.index], "_mv"].sum()
                _e_sum = (sub.loc[m_denom, "_mv"] / sub.loc[m_denom, col]).sum()
            else:
                # 剔除亏损口径（中证口径）：分子分母同步剔除亏损股与无 PE 股
                m = mv_ok[sub.index] & has_val & (sub[col] > 0)
                if not m.any():
                    return np.nan
                _mv_sum = sub.loc[m, "_mv"].sum()
                _e_sum = (sub.loc[m, "_mv"] / sub.loc[m, col]).sum()
            if abs(_e_sum) < 1e-12:
                return np.nan
            return float(_mv_sum / _e_sum)

        def _wmean(sub: pd.DataFrame, col: str) -> float:
            m = mv_ok[sub.index] & sub[col].notna()
            if not m.any():
                return np.nan
            return float((sub.loc[m, "_mv"] * sub.loc[m, col]).sum() / sub.loc[m, "_mv"].sum())

        _include_loss = agg_method == "aggregate_full"
        agg = est.groupby("l2_code", group_keys=False).apply(
            lambda g: pd.Series({
                "pe_static": _harmonic(g, "_pe", _include_loss),
                "pe_ttm": _harmonic(g, "_pe_ttm", _include_loss),
                "pb": _harmonic(g, "_pb", _include_loss),
                "div_yield": _wmean(g, "_dv"),
            }),
            include_groups=False,
        )
        out = df_val.copy()
        out["_l2"] = out["code"].astype(str)
        for col in ("pe_static", "pe_ttm", "pb", "div_yield"):
            out[col] = out["_l2"].map(agg[col])
        return out.drop(columns=["_l2"])

    def fetch_and_cache_all(self, force_update: bool = False) -> pd.DataFrame | None:
        """遍历所有申万二级行业，拉取250天数据并缓存到本地"""
        hist_cache_exists = os.path.exists(self.cache_file) or os.path.exists(self.cache_csv_file)
        valuation_cache_exists = os.path.exists(self.valuation_file)
        
        if hist_cache_exists and valuation_cache_exists and not force_update:
            # 先读取缓存，检查其完整性
            try:
                cached_hist = pd.read_parquet(self.cache_file)
                cached_hist = _ensure_numpy_backend(cached_hist)
            except (OSError, ValueError, TypeError, ImportError):
                cached_hist = pd.read_csv(self.cache_csv_file, dtype={'date': str})
                cached_hist['date'] = pd.to_datetime(cached_hist['date'])
            cached_hist = cached_hist.loc[:, ~cached_hist.columns.duplicated()]

            cached_val = pd.read_csv(self.valuation_file)
            cached_industry_count = len(cached_val)
            
            # [KEY] 关键：从 AShareHub 获取二级行业总数
            ah_df = self.ah_client.industry_list()
            current_total_industries = ah_df["l2_code"].nunique()
            
            # [OK] 只有当缓存的行业数量 == 当前接口总数时，才使用缓存
            if cached_industry_count == current_total_industries:
                logger.info(f"缓存完整({cached_industry_count}个行业)，使用缓存数据")
                return cached_hist
            else:
                logger.warning(f"缓存不完整({cached_industry_count}个 vs {current_total_industries}个)，重新拉取...")


        logger.info("获取申万二级行业列表及估值数据...")
        try:
            ah_df = self.ah_client.industry_list()
            df_info = ah_df[["l2_code", "l2_name"]].drop_duplicates().dropna().copy()
            df_info.columns = ["行业代码", "行业名称"]
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            return None

        valuation_cols_map = {
            '行业代码': 'code',
            '行业名称': 'name',
        }

        df_val = df_info[list(valuation_cols_map.keys())].copy()
        df_val.columns = list(valuation_cols_map.values())
        df_val['pe_static'] = None
        df_val['pe_ttm'] = None
        df_val['pb'] = None
        df_val['div_yield'] = None
        # AShareHub 无行业板块估值接口（industry_list 仅返回分类映射）：
        # 改用个股每日估值 /v2/market/fundamentals（pe/pe_ttm/pb/dv_ratio/total_mv），
        # 按申万二级行业市值加权聚合出行业估值
        try:
            fundamentals_df = self.ah_client.fundamentals(trade_date=self.today_str)
            if fundamentals_df is None or fundamentals_df.empty:
                # 交易日可能已变化，回退尝试前 3 个自然日
                _today_dt = datetime.datetime.strptime(self.today_str, "%Y%m%d")
                for _back in range(1, 4):
                    _d = (_today_dt - datetime.timedelta(days=_back)).strftime("%Y%m%d")
                    fundamentals_df = self.ah_client.fundamentals(trade_date=_d)
                    if fundamentals_df is not None and not fundamentals_df.empty:
                        break
            if fundamentals_df is not None and not fundamentals_df.empty:
                _agg_method = getattr(
                    self.config.app_config.scoring_params, "INDUSTRY_VALUATION_AGG_METHOD",
                    "aggregate_profitable",
                )
                df_val = self._aggregate_industry_valuation(
                    ah_df, fundamentals_df, df_val, agg_method=_agg_method,
                )
                logger.info(f"行业估值聚合完成: {len(df_val)} 个行业（个股→市值加权，口径={_agg_method}）")
            else:
                logger.warning("估值数据不可用 (AShareHub fundamentals 返回空)，使用默认值。")
        except Exception as e:
            logger.warning(f"估值数据不可用 (AShareHub fundamentals 调用失败: {e})，使用默认值。")

        # --- 智能提取纯数字代码 ---
        # 使用正则表达式，提取字符串开头的连续数字部分
        df_val['code'] = df_val['code'].astype(str).apply(lambda x: re.match(r'^(\d+)', x).group(1) if re.match(r'^(\d+)', x) else x)
        df_val.to_csv(self.valuation_file, index=False, encoding='utf-8-sig')

        codes = df_val['code'].astype(str).tolist()
        names = df_val['name'].astype(str).tolist()
        
        print(f"  ↓ 拉取 {len(codes)} 个行业K线数据...", flush=True)
        logger.info(f"开始拉取 {len(codes)} 个行业的250天历史量价数据（单线程）...")
        all_hist_data = []

        for idx, (code, name) in enumerate(zip(codes, names)):
            try:
                url = "https://www.swsresearch.com/institute-sw/api/index_publish/trend/"
                params = {"swindexcode": code, "period": "DAY"}
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                r = requests.get(url, params=params, headers=headers, verify=False, timeout=15)
                r.raise_for_status()
                data_json = r.json()
                df_hist = pd.DataFrame(data_json["data"])
                df_hist.rename(
                    columns={
                        "swindexcode": "代码",
                        "bargaindate": "日期",
                        "openindex": "开盘",
                        "maxindex": "最高",
                        "minindex": "最低",
                        "closeindex": "收盘",
                        "bargainamount": "成交量",
                        "bargainsum": "成交额",
                    },
                    inplace=True,
                )
                if df_hist is not None and not df_hist.empty:
                    df_hist_mapped = self._map_hist_columns(df_hist)
                    required_core_cols = ['date', 'close', 'volume', 'amount']
                    if all(c in df_hist_mapped.columns for c in required_core_cols):
                        core_cols = [c for c in ['date', 'close', 'open', 'high', 'low', 'volume', 'amount'] if c in df_hist_mapped.columns]
                        df_sub = df_hist_mapped[core_cols].copy()
                        df_sub['date'] = pd.to_datetime(df_sub['date'])
                        df_sub = df_sub.sort_values('date').tail(250).reset_index(drop=True)
                        df_sub['code'] = code
                        df_sub['name'] = name
                        all_hist_data.append(df_sub)
                    else:
                        logger.warning(f"{code} 历史数据映射后缺少核心字段，跳过。")
            except Exception as e:
                logger.warning(f"获取 {code} ({name}) 失败 -> {e}")
            print(f"    [{idx+1}/{len(codes)}] {name} ({code})", flush=True)
            time.sleep(0.1)

        print(f"  完成: {len(all_hist_data)}/{len(codes)} 个行业", flush=True)
        if not all_hist_data:
            logger.error("未能获取任何历史数据。")
            return None

        logger.info("数据合并与本地缓存...")
        df_all = pd.concat(all_hist_data, ignore_index=True)
        # 去重列（SW接口可能返回同名英文字段）
        df_all = df_all.loc[:, ~df_all.columns.duplicated()]
        
        # 尝试保存为Parquet，如果失败则保存为CSV
        try:
            df_all.to_parquet(self.cache_file, index=False)
            logger.info(f"成功缓存 {len(df_all)} 条数据至 {self.cache_file} (Parquet格式)")
        except Exception as e:
            logger.warning(f"Parquet保存失败 ({e})，回退到CSV...")
            df_all.to_csv(self.cache_csv_file, index=False, encoding='utf-8-sig')
            logger.info(f"成功缓存 {len(df_all)} 条数据至 {self.cache_csv_file} (CSV格式)")

        return df_all


def _ensure_numpy_backend(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 所有列转为 numpy 原生类型，避免 pandas 3.0 C 层崩溃"""
    def _safe_date(v):
        try:
            return str(pd.Timestamp(v).strftime('%Y-%m-%d')) if pd.notna(v) else '1970-01-01'
        except Exception:
            return '1970-01-01'

    NUMERIC_COLS = {'close', 'open', 'high', 'low', 'volume', 'amount'}
    for col in df.columns:
        if col == 'date':
            df[col] = df[col].apply(_safe_date)
        elif col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        elif col in ('code', 'name'):
            df[col] = df[col].fillna('').astype(str)
        else:
            dtype = df[col].dtype
            if isinstance(dtype, pd.ArrowDtype):
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    df[col] = df[col].apply(_safe_date)
                elif pd.api.types.is_string_dtype(dtype):
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(dtype.numpy_dtype)
            elif dtype == object:
                df[col] = df[col].astype(str)
    return df


class SWMultiFactorModel:
    """模块二：多因子计算引擎（纯本地向量化计算，极速）"""
    
    def __init__(self, pipeline: SWIndustryDataPipeline) -> None:
        self.pipeline = pipeline
        self.ma_periods = [10, 20, 30, 60, 90]

    def _calculate_vectorized_factors(self, df_hist: pd.DataFrame) -> pd.DataFrame:
        """利用 GroupBy 进行向量化计算"""
        logger.info(f"向量化因子计算开始: {len(df_hist)} 行, {df_hist['code'].nunique()} 个行业")

        # ── 列级类型强制转换（全部走 Python 级操作，避免 pandas 3.0 C 扩展崩溃）──
        df = df_hist.copy()

        def _safe_date(v):
            try:
                return str(pd.Timestamp(v).strftime('%Y-%m-%d')) if pd.notna(v) else '1970-01-01'
            except Exception:
                return '1970-01-01'

        for col in df.columns:
            if col == 'date':
                df[col] = df[col].apply(_safe_date)
            elif col in {'close', 'open', 'high', 'low', 'volume', 'amount'}:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif col in ('code', 'name'):
                df[col] = df[col].fillna('').astype(str)
            elif isinstance(df[col].dtype, pd.ArrowDtype):
                if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                    df[col] = df[col].apply(_safe_date)
                elif pd.api.types.is_string_dtype(df[col].dtype):
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(df[col].dtype.numpy_dtype)
            elif df[col].dtype == object:
                df[col] = df[col].astype(str)

        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        logger.info("排序完成")
        
        for p in self.ma_periods:
            logger.info(f"计算 MA{p}...")
            df[f'ma_{p}'] = df.groupby('code')['close'].transform(lambda x: x.rolling(p).mean())
            logger.info(f"MA{p} 完成")
              
        logger.info("计算 vol_ma_20...")
        df['vol_ma_20'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
        logger.info("计算 amt_ma_60...")
        df['amt_ma_60'] = df.groupby('code')['amount'].transform(lambda x: x.rolling(60).mean())
        logger.info("均线计算完成")
        
        df_latest = df.groupby('code').tail(1).set_index('code')
        logger.info(f"取最新行: {len(df_latest)} 行")
        
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
            df_hist = pd.read_csv(self.pipeline.cache_csv_file, dtype={'date': str})
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            df_hist = df_hist.loc[:, ~df_hist.columns.duplicated()]
        df_hist = _ensure_numpy_backend(df_hist)
        df_hist['code'] = df_hist['code'].astype(str)

        df_val = pd.read_csv(self.pipeline.valuation_file)
        df_val['code'] = df_val['code'].astype(str)
        
        logger.info("正在执行向量化因子计算...")
        try:
            df_factors = self._calculate_vectorized_factors(df_hist)
            logger.info(f"因子计算完成，shape={df_factors.shape}")
        except Exception as e:
            import traceback
            logger.error(f"因子计算崩溃: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # 解决列名冲突：重命名因子数据中的name列为factor_name
        df_factors_renamed = df_factors.copy()
        df_factors_renamed = df_factors_renamed.rename(columns={'name': 'factor_name'})
        
        # 合并数据时指定suffixes来避免重复列名
        logger.info("合并因子与估值数据...")
        try:
            df = df_factors_renamed.join(df_val.set_index('code'), how='left', rsuffix='_val')
            logger.info(f"合并完成，shape={df.shape}")
        except Exception as e:
            import traceback
            logger.error(f"合并崩溃: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
        
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
