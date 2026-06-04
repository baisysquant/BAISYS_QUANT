import re
import pandas as pd
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas_ta as ta  # 勿删
from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
from LogicAnalyzer.KDJAnalyzer import AdvancedKDJAnalyzer
from ConfigParser import Config


class TASignalProcessor:
    """
    技术指标信号处理类
    
    负责计算和处理多种技术指标信号，包括：
    - MACD（标准周期 + 第二周期）
    - KDJ、CCI、RSI、BOLL
    - 背离检测
    - 完全多头综合评分
    
    Attributes:
        analyzer: 股票分析器实例
        kdj_analyzer: KDJ 分析器
        macd_analyzer: MACD 分析器
        config: 配置管理器实例
    """

    def __init__(self, analyzer_instance: Any, config: Optional[Config] = None) -> None:
        """
        初始化技术指标信号处理器
        
        Args:
            analyzer_instance: 股票分析器实例
            config: 配置管理器实例，用于读取MACD第二周期等配置
        """
        self.analyzer      = analyzer_instance
        self.kdj_analyzer  = AdvancedKDJAnalyzer()
        self.macd_analyzer = MACDAnalyzer()
        self.config        = config

    def _classify_cci_level(self, cci_value: float) -> str:
        """
        根据 CCI 值分类
        
        CCI (Commodity Channel Index) 商品通道指标分类标准：
        - > 200: 极度超买
        - 100 ~ 200: 强势超买
        - -100 ~ 100: 正常区间（无信号）
        - -200 ~ -100: 弱势超卖
        - < -200: 极度超卖
        
        Args:
            cci_value: CCI 指标值
            
        Returns:
            str: CCI 状态描述字符串，如 '极度超买 (250.35)' 或空字符串
        """
        if pd.isna(cci_value):
            return 'N/A'
        if   cci_value >  200: return f'极度超买 ({cci_value:.2f})'
        elif cci_value >= 100: return f'强势超买 ({cci_value:.2f})'
        elif cci_value >  -100: return ''
        elif cci_value >= -200: return f'弱势超卖 ({cci_value:.2f})'
        else:                   return f'极度超卖 ({cci_value:.2f})'

    def _process_single_stock(
        self,
        code: str,
        valid_hist_df: pd.DataFrame,
        second_params: tuple,
        second_period_name: str,
    ) -> dict[str, Any] | None:
        code_str = str(code).lower()
        if code_str.startswith(('sh', 'sz', 'bj')):
            code_str = code_str[2:]
        match = re.search(r'(\d{6})', code_str)
        if not match:
            return None
        pure_code = match.group(1)

        df = valid_hist_df[valid_hist_df['股票代码'] == pure_code].copy()
        if df.empty or len(df) < 30:
            return None

        for col in ['close', 'open', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close', 'open', 'high', 'low'], inplace=True)
        if df.empty:
            return None
        required_cols = ['close', 'open', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None
        if 'date' in df.columns:
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)

        try:
            df = self.macd_analyzer._custom_macd(df, second_params=second_params)
        except Exception:
            return None

        result: dict[str, Any] = {'code': code, 'pure_code': pure_code}

        try:
            weights = getattr(self.config, 'FULL_BULL_WEIGHTS', None)
            thresholds = getattr(self.config, 'FULL_BULL_THRESHOLDS', None)
            bull_result = self.macd_analyzer.analyze_full_bull(
                df, second_params=second_params, recalc_macd=False,
                weights=weights, thresholds=thresholds,
            )
            result['bull'] = bull_result
        except Exception:
            pass

        try:
            latest_row = df.iloc[-1]
            result['mom_12269'] = MACDAnalyzer._calculate_macd_momentum(df, 'DIF_12269', 'DEA_12269')
            result['mom_second'] = MACDAnalyzer._calculate_macd_momentum(
                df, f'DIF_{second_period_name}', f'DEA_{second_period_name}',
            )
            result['dif_12269'] = latest_row.get('DIF_12269', 0)
            result['dif_second'] = latest_row.get(f'DIF_{second_period_name}', 0)
        except Exception:
            pass

        detail_col_12269 = 'MACD_12269_SIGNAL_DETAIL'
        if detail_col_12269 in df.columns:
            val = df[detail_col_12269].iloc[-1]
            if pd.notna(val) and val != '':
                result['macd_12269_signal'] = val

        detail_col_second = f'MACD_{second_period_name}_SIGNAL_DETAIL'
        if detail_col_second in df.columns:
            val = df[detail_col_second].iloc[-1]
            if pd.notna(val) and val != '':
                result['macd_second_signal'] = val

        try:
            kdj_signal = self.kdj_analyzer.calculate_kdj_signal_from_df(df)
            if kdj_signal:
                result['kdj_signal'] = kdj_signal
        except Exception:
            pass

        try:
            df.ta.cci(append=True, close='close', high='high', low='low')
            cci_cols = [col for col in df.columns if col.startswith('CCI_')]
            if cci_cols:
                current_cci = df[cci_cols[0]].iloc[-1]
                result['cci_signal'] = self._classify_cci_level(current_cci) or f'常态波动 ({current_cci:.2f})'
        except Exception:
            pass

        try:
            df.ta.rsi(append=True, close='close', length=14)
            rsi_cols = [col for col in df.columns if col.startswith('RSI_')]
            if rsi_cols:
                rsi_col = rsi_cols[0]
                curr_rsi = df[rsi_col].iloc[-1]
                window = 10
                if len(df) >= window + 1:
                    curr_low = df['low'].iloc[-1]
                    min_low_window = df['low'].iloc[-window:-1].min()
                    min_rsi_window = df[rsi_col].iloc[-window:-1].min()
                    is_price_low = curr_low <= (min_low_window * 1.02)
                    is_divergence = is_price_low and (curr_rsi > min_rsi_window * 1.05) and (curr_rsi < 50)
                    result['rsi_signal'] = f'RSI底背离! ({curr_rsi:.1f})' if is_divergence else f'RSI={curr_rsi:.1f}'
        except Exception:
            pass

        try:
            df.ta.bbands(append=True, length=20, std=2, close='close')
            boll_lower_cols = [col for col in df.columns if col.startswith('BBL_')]
            boll_upper_cols = [col for col in df.columns if col.startswith('BBU_')]
            if boll_lower_cols and boll_upper_cols:
                df['BOLL_BANDWIDTH'] = (
                    (df[boll_upper_cols[0]] - df[boll_lower_cols[0]]) / df['close']
                )
                is_narrow = (
                    df['BOLL_BANDWIDTH'].iloc[-5:].mean() < df['BOLL_BANDWIDTH'].mean()
                )
                result['boll_signal'] = '低波/缩口' if is_narrow else '常态/张口'
        except Exception:
            pass

        return result

    def process_signals(
        self,
        all_codes: list[str],
        hist_df_all: pd.DataFrame,
        spot_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        处理所有股票的技术指标信号
        
        对给定的股票列表进行批量技术分析，计算多种技术指标并生成信号。
        
        Args:
            all_codes: 股票代码列表（6位纯数字格式）
            hist_df_all: 历史K线数据DataFrame，包含所有股票的OHLCV数据
            spot_df: 实时行情数据DataFrame，包含最新价格等信息
            
        Returns:
            dict[str, pd.DataFrame]: 技术指标信号字典，key为指标名称，value为对应的信号DataFrame
            包括：
            - MACD_12269: 标准MACD信号（固定）
            - MACD_{second_period_name}: 第二周期MACD信号（动态，如 MACD_9186）
            - MACD_FULL_BULL: MACD完全多头综合评分信号
            - KDJ: KDJ信号
            - CCI: CCI信号
            - RSI: RSI信号
            - BOLL: 布林带信号
            - MACD_DIF_MOMENTUM: MACD动能状态
        """

        # 动态获取第二周期MACD列名
        if self.config and hasattr(self.config, 'MACD_SECOND_PARAMS'):
            fast, slow, signal = self.config.MACD_SECOND_PARAMS
            second_period_name = f"{fast}{slow}{signal}"
        else:
            second_period_name = '9186'  # 默认值（与config.ini一致）
        
        ta_signals = {
            'MACD_12269': pd.DataFrame(columns=['股票代码', 'MACD_12269_Signal']),
            f'MACD_{second_period_name}': pd.DataFrame(columns=['股票代码', f'MACD_{second_period_name}_Signal']),
            'KDJ':  pd.DataFrame(columns=['股票代码', 'KDJ_Signal']),
            'CCI':  pd.DataFrame(columns=['股票代码', 'CCI_Signal']),
            'RSI':  pd.DataFrame(columns=['股票代码', 'RSI_Signal']),
            'BOLL': pd.DataFrame(columns=['股票代码', 'BOLL_Signal']),
            'MACD_DIF_MOMENTUM': pd.DataFrame(columns=[
                '股票代码',
                'MACD_12269_DIF', 'MACD_12269_动能',
                f'MACD_{second_period_name}_DIF', f'MACD_{second_period_name}_动能',
            ]),
            # ── 新增：完全多头综合评分 ─────────────────────────────────────
            'MACD_FULL_BULL': pd.DataFrame(columns=[
                '股票代码', 'FullBull_Score', 'FullBull_Score_Base', 'MACD_FULL_BULL_Label',
                '零轴条件', '战略金叉', '战术金叉', '动能', 'DIF斜率', '背离信号', '量价配合',
            ]),
        }

        if hist_df_all.empty:
            return ta_signals

        if 'symbol' not in hist_df_all.columns:
            return ta_signals

        # ── 规范化股票代码处理 ───────────────────────────────────────────
        # 确保symbol列是字符串格式
        hist_df_all['symbol'] = hist_df_all['symbol'].astype(str)
        
        # 提取股票代码，支持多种格式（sh600000, sz000001, 600000, 000001等）
        def extract_code(symbol):
            clean_symbol = str(symbol).lower()
            if clean_symbol.startswith(('sh', 'sz', 'bj')):
                clean_symbol = clean_symbol[2:]
            match = re.search(r'(\d{6})', clean_symbol)
            return match.group(1) if match else 'N/A'
        
        hist_df_all['股票代码'] = hist_df_all['symbol'].apply(extract_code)

        # 日期列标准化
        if 'date' not in hist_df_all.columns and 'trade_date' in hist_df_all.columns:
            hist_df_all.rename(columns={'trade_date': 'date'}, inplace=True)

        # 按股票代码和日期排序
        hist_df_all.sort_values(['股票代码', 'date'], inplace=True)

        # 规范化输入的股票代码列表
        pure_codes_list = []
        for c in all_codes:
            code_str = str(c).lower()
            if code_str.startswith(('sh', 'sz', 'bj')):
                code_str = code_str[2:]
            match = re.search(r'(\d{6})', code_str)
            if match:
                pure_codes_list.append(match.group(1).zfill(6))
            else:
                pure_codes_list.append(code_str.zfill(6))
        
        code_set = set(pure_codes_list)
        
        # 过滤出有效的股票数据
        valid_hist_df = hist_df_all[hist_df_all['股票代码'].isin(code_set) & (hist_df_all['股票代码'] != 'N/A')].copy()

        # ── 并行处理所有股票 ──────────────────────────────────────────────
        second_params = getattr(self.config, 'MACD_SECOND_PARAMS', (6, 13, 5))
        max_workers = getattr(self.config, 'SIGNAL_PROCESSING_PROCESSES', 2)

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_stock, code, valid_hist_df, second_params, second_period_name
                ): code
                for code in all_codes
            }
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                except Exception:
                    continue

        # ── 从结果列表构建信号 DataFrame（避免逐行 O(n²) concat）─────────
        macd_12269_rows: list[dict] = []
        macd_second_rows: list[dict] = []
        bull_rows: list[dict] = []
        mom_rows: list[dict] = []
        kdj_rows: list[dict] = []
        cci_rows: list[dict] = []
        rsi_rows: list[dict] = []
        boll_rows: list[dict] = []

        for r in results:
            code = r['code']

            if 'macd_12269_signal' in r:
                macd_12269_rows.append({'股票代码': code, 'MACD_12269_Signal': r['macd_12269_signal']})

            if 'macd_second_signal' in r:
                macd_second_rows.append({'股票代码': code, f'MACD_{second_period_name}_Signal': r['macd_second_signal']})

            if 'bull' in r:
                detail = r['bull'].get('details', {})
                bull_rows.append({
                    '股票代码': code,
                    'FullBull_Score': r['bull'].get('score', 0),
                    'FullBull_Score_Base': r['bull'].get('score_base', 0),
                    'MACD_FULL_BULL_Label': r['bull'].get('conclusion', ''),
                    '零轴条件': detail.get('零轴条件', {}).get('desc', ''),
                    '战略金叉': detail.get('战略金叉', {}).get('desc', ''),
                    '战术金叉': detail.get('战术金叉', {}).get('desc', ''),
                    '动能': detail.get('动能', {}).get('desc', ''),
                    'DIF斜率': detail.get('DIF斜率', {}).get('desc', ''),
                    '背离信号': detail.get('背离信号', {}).get('desc', ''),
                    '量价配合': detail.get('量价配合', {}).get('desc', ''),
                })

            if 'mom_12269' in r:
                mom_rows.append({
                    '股票代码': code,
                    'MACD_12269_DIF': r['dif_12269'],
                    'MACD_12269_动能': r['mom_12269'],
                    f'MACD_{second_period_name}_DIF': r.get('dif_second', 0),
                    f'MACD_{second_period_name}_动能': r.get('mom_second', ''),
                })

            if 'kdj_signal' in r:
                kdj_rows.append({'股票代码': code, 'KDJ_Signal': r['kdj_signal']})

            if 'cci_signal' in r:
                cci_rows.append({'股票代码': code, 'CCI_Signal': r['cci_signal']})

            if 'rsi_signal' in r:
                rsi_rows.append({'股票代码': code, 'RSI_Signal': r['rsi_signal']})

            if 'boll_signal' in r:
                boll_rows.append({'股票代码': code, 'BOLL_Signal': r['boll_signal']})

        ta_signals['MACD_12269'] = pd.DataFrame(macd_12269_rows)
        ta_signals[f'MACD_{second_period_name}'] = pd.DataFrame(macd_second_rows)
        ta_signals['MACD_FULL_BULL'] = pd.DataFrame(bull_rows)
        ta_signals['MACD_DIF_MOMENTUM'] = pd.DataFrame(mom_rows)
        ta_signals['KDJ'] = pd.DataFrame(kdj_rows)
        ta_signals['CCI'] = pd.DataFrame(cci_rows)
        ta_signals['RSI'] = pd.DataFrame(rsi_rows)
        ta_signals['BOLL'] = pd.DataFrame(boll_rows)

        # ── 统一清洗股票代码格式 ──────────────────────────────────────────
        for key in ta_signals:
            df_sig = ta_signals[key]
            if not df_sig.empty and '股票代码' in df_sig.columns:
                df_sig['股票代码'] = df_sig['股票代码'].astype(str).apply(
                    lambda x: re.search(r'(\d{6})', str(x)).group(1) if re.search(r'(\d{6})', str(x)) else x
                )
                ta_signals[key] = df_sig

        return ta_signals
