import pandas as pd
from typing import List, Dict, Optional, Any
import pandas_ta as ta  # 勿删
from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
from DataManager.ShareCodeFormatMgr import format_stock_code
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

    def process_signals(
        self,
        all_codes: List[str],
        hist_df_all: pd.DataFrame,
        spot_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        处理所有股票的技术指标信号
        
        对给定的股票列表进行批量技术分析，计算多种技术指标并生成信号。
        
        Args:
            all_codes: 股票代码列表（6位纯数字格式）
            hist_df_all: 历史K线数据DataFrame，包含所有股票的OHLCV数据
            spot_df: 实时行情数据DataFrame，包含最新价格等信息
            
        Returns:
            Dict[str, pd.DataFrame]: 技术指标信号字典，key为指标名称，value为对应的信号DataFrame
            包括：
            - MACD_12269: 标准MACD信号（固定）
            - MACD_{second_period_name}: 第二周期MACD信号（动态，如 MACD_9186）
            - MACD_COMBINED_DIVERGENCE: MACD背离信号
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
            # ── 背离：新增强度 / 衰减字段 ──────────────────────────────────
            'MACD_COMBINED_DIVERGENCE': pd.DataFrame(columns=[
                '股票代码',
                'Combined_Divergence_Signal',
                'Div_12269_Type', 'Div_12269_Strength', 'Div_12269_Decay',
                f'Div_{second_period_name}_Type', f'Div_{second_period_name}_Strength', f'Div_{second_period_name}_Decay',
            ]),
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
                '股票代码', 'FullBull_Score', 'FullBull_Conclusion',
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
            # 移除前缀并提取6位数字
            clean_symbol = str(symbol).lower()
            if clean_symbol.startswith(('sh', 'sz', 'bj')):
                clean_symbol = clean_symbol[2:]
            # 提取6位数字
            import re
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
            # 再次提取6位数字
            import re
            match = re.search(r'(\d{6})', code_str)
            if match:
                pure_codes_list.append(match.group(1).zfill(6))
            else:
                pure_codes_list.append(code_str.zfill(6))
        
        code_set = set(pure_codes_list)
        
        # 过滤出有效的股票数据
        valid_hist_df = hist_df_all[hist_df_all['股票代码'].isin(code_set) & (hist_df_all['股票代码'] != 'N/A')].copy()

        # ── 逐只股票处理 ─────────────────────────────────────────────────
        for code in all_codes:
            # 提取标准6位股票代码
            code_str = str(code).lower()
            if code_str.startswith(('sh', 'sz', 'bj')):
                code_str = code_str[2:]
            import re
            match = re.search(r'(\d{6})', code_str)
            if not match:
                continue
            pure_code = match.group(1)
            
            # 获取该股票的历史数据
            df = valid_hist_df[valid_hist_df['股票代码'] == pure_code].copy()

            if df.empty or len(df) < 30:
                continue

            # 数据类型转换和清理
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除关键列中存在NaN的行
            df.dropna(subset=['close', 'open', 'high', 'low'], inplace=True)

            if df.empty:
                continue
            
            # 检查必要列是否存在
            required_cols = ['close', 'open', 'high', 'low']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                continue

            # 确保数据按日期排序
            if 'date' in df.columns:
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)

            # ── MACD 计算（一次调用，后续共用结果）────────────────────────
            try:
                df = self.macd_analyzer._custom_macd(df, second_params=self.config.MACD_SECOND_PARAMS if self.config and hasattr(self.config, 'MACD_SECOND_PARAMS') else (6, 13, 5))
            except Exception:
                continue

            # ── 自适应 distance（利用 ATR，替代固定值）────────────────────
            try:
                dist_slow = MACDAnalyzer._adaptive_distance(df, base=10)
                dist_fast = MACDAnalyzer._adaptive_distance(df, base=5)
            except Exception:
                dist_slow = 10
                dist_fast = 5

            # ── 背离检测（修复：每套参数只调一次，含强度 / 衰减）──────────
            try:
                combined_div = MACDAnalyzer.detect_combined_divergence(
                    df,
                    distance_slow=dist_slow,
                    distance_fast=dist_fast,
                    recent_window=5,
                    decay_half_life=8,
                    second_period_name=second_period_name,  # 传递动态的第二周期名称
                )
                divergence_signal = combined_div.get('combined_signal', '')

                # 只有存在信号时才写入（保持原逻辑）
                if divergence_signal:
                    new_row = pd.DataFrame([{
                        '股票代码':                  code,
                        'Combined_Divergence_Signal': divergence_signal,
                        # ── 新增：完整背离元数据，方便下游过滤 ──
                        'Div_12269_Type':     combined_div.get('div_12269', ''),
                        'Div_12269_Strength': combined_div.get('strength_12269', 0.0),
                        'Div_12269_Decay':    combined_div.get('decay_12269', 0.0),
                        f'Div_{second_period_name}_Type':      combined_div.get(f'div_{second_period_name}', ''),
                        f'Div_{second_period_name}_Strength':  combined_div.get(f'strength_{second_period_name}', 0.0),
                        f'Div_{second_period_name}_Decay':     combined_div.get(f'decay_{second_period_name}', 0.0),
                    }])
                    ta_signals['MACD_COMBINED_DIVERGENCE'] = pd.concat([
                        ta_signals['MACD_COMBINED_DIVERGENCE'],
                        new_row
                    ], ignore_index=True)
            except Exception:
                pass

            # ── 完全多头综合评分（新增核心能力接入）──────────────────────
            try:
                # 从配置中读取第二周期参数（必填）
                second_params = getattr(self.config, 'MACD_SECOND_PARAMS', (6, 13, 5))
                
                bull_result = self.macd_analyzer.analyze_full_bull(df, second_params=second_params)
                # 不论分数高低都记录，让下游自行筛选
                detail = bull_result.get('details', {})
                new_row = pd.DataFrame([{
                    '股票代码':           code,
                    'FullBull_Score':      bull_result.get('score', 0),
                    'FullBull_Conclusion': bull_result.get('conclusion', ''),
                    '零轴条件': detail.get('零轴条件', {}).get('desc', ''),
                    '战略金叉': detail.get('战略金叉', {}).get('desc', ''),
                    '战术金叉': detail.get('战术金叉', {}).get('desc', ''),
                    '动能':     detail.get('动能',     {}).get('desc', ''),
                    'DIF斜率':  detail.get('DIF斜率',  {}).get('desc', ''),
                    '背离信号': detail.get('背离信号', {}).get('desc', ''),
                    '量价配合': detail.get('量价配合', {}).get('desc', ''),
                }])
                ta_signals['MACD_FULL_BULL'] = pd.concat([
                    ta_signals['MACD_FULL_BULL'],
                    new_row
                ], ignore_index=True)
            except Exception:
                pass

            # ── 动能状态 ──────────────────────────────────────────────────
            try:
                latest_row = df.iloc[-1]
                mom_12269  = MACDAnalyzer._calculate_macd_momentum(df, 'DIF_12269', 'DEA_12269')
                
                # 根据配置确定第二周期名称（必填）
                if self.config and hasattr(self.config, 'MACD_SECOND_PARAMS'):
                    fast, slow, signal = self.config.MACD_SECOND_PARAMS
                    second_period_name = f"{fast}{slow}{signal}"
                else:
                    second_period_name = '6135'
                
                mom_second = MACDAnalyzer._calculate_macd_momentum(df, f'DIF_{second_period_name}', f'DEA_{second_period_name}')
                
                new_row = pd.DataFrame([{
                    '股票代码':        code,
                    'MACD_12269_DIF':  latest_row.get('DIF_12269', 0),
                    'MACD_12269_动能': mom_12269,
                    f'MACD_{second_period_name}_DIF':   latest_row.get(f'DIF_{second_period_name}',  0),
                    f'MACD_{second_period_name}_动能':  mom_second,
                }])
                ta_signals['MACD_DIF_MOMENTUM'] = pd.concat([
                    ta_signals['MACD_DIF_MOMENTUM'],
                    new_row
                ], ignore_index=True)
            except Exception:
                pass

            # ── MACD 12269 金叉 / 死叉信号 ───────────────────────────────
            detail_col_12269 = 'MACD_12269_SIGNAL_DETAIL'
            if detail_col_12269 in df.columns:
                signal_val = df[detail_col_12269].iloc[-1]
                if pd.notna(signal_val) and signal_val != '':
                    new_row = pd.DataFrame([{
                        '股票代码':           code,
                        'MACD_12269_Signal':  signal_val,
                    }])
                    ta_signals['MACD_12269'] = pd.concat([
                        ta_signals['MACD_12269'],
                        new_row
                    ], ignore_index=True)

            # ── MACD 第二周期金叉 / 死叉信号 ────────────────────────────────
            detail_col_second = f'MACD_{second_period_name}_SIGNAL_DETAIL'
            if detail_col_second in df.columns:
                signal_val = df[detail_col_second].iloc[-1]
                if pd.notna(signal_val) and signal_val != '':
                    new_row = pd.DataFrame([{
                        '股票代码':          code,
                        f'MACD_{second_period_name}_Signal':  signal_val,
                    }])
                    ta_signals[f'MACD_{second_period_name}'] = pd.concat([
                        ta_signals[f'MACD_{second_period_name}'],
                        new_row
                    ], ignore_index=True)

            # ── KDJ ───────────────────────────────────────────────────────
            try:
                kdj_signal = self.kdj_analyzer.calculate_kdj_signal_from_df(df)
                if kdj_signal:
                    new_row = pd.DataFrame([{'股票代码': code, 'KDJ_Signal': kdj_signal}])
                    ta_signals['KDJ'] = pd.concat([
                        ta_signals['KDJ'],
                        new_row
                    ], ignore_index=True)
            except Exception:
                pass

            # ── CCI ───────────────────────────────────────────────────────
            try:
                df.ta.cci(append=True, close='close', high='high', low='low')
                cci_cols = [col for col in df.columns if col.startswith('CCI_')]
                if cci_cols:
                    current_cci = df[cci_cols[0]].iloc[-1]
                    cci_signal  = self._classify_cci_level(current_cci) or f'常态波动 ({current_cci:.2f})'
                    new_row = pd.DataFrame([{'股票代码': code, 'CCI_Signal': cci_signal}])
                    ta_signals['CCI'] = pd.concat([
                        ta_signals['CCI'],
                        new_row
                    ], ignore_index=True)
            except Exception:
                pass

            # ── RSI ───────────────────────────────────────────────────────
            try:
                df.ta.rsi(append=True, close='close', length=14)
                rsi_cols = [col for col in df.columns if col.startswith('RSI_')]
                if rsi_cols:
                    rsi_col        = rsi_cols[0]
                    curr_rsi       = df[rsi_col].iloc[-1]
                    window         = 10
                    if len(df) >= window + 1:  # 确保有足够的数据
                        curr_low       = df['low'].iloc[-1]
                        min_low_window = df['low'].iloc[-window:-1].min()
                        min_rsi_window = df[rsi_col].iloc[-window:-1].min()
                        is_price_low   = curr_low <= (min_low_window * 1.02)
                        is_divergence  = is_price_low and (curr_rsi > min_rsi_window * 1.05) and (curr_rsi < 50)
                        rsi_msg        = f'RSI底背离! ({curr_rsi:.1f})' if is_divergence else f'RSI={curr_rsi:.1f}'
                        new_row = pd.DataFrame([{'股票代码': code, 'RSI_Signal': rsi_msg}])
                        ta_signals['RSI'] = pd.concat([
                            ta_signals['RSI'],
                            new_row
                        ], ignore_index=True)
            except Exception:
                pass

            # ── BOLL ──────────────────────────────────────────────────────
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
                    new_row = pd.DataFrame([{
                        '股票代码':    code,
                        'BOLL_Signal': '低波/缩口' if is_narrow else '常态/张口',
                    }])
                    ta_signals['BOLL'] = pd.concat([
                        ta_signals['BOLL'],
                        new_row
                    ], ignore_index=True)
            except Exception:
                pass

        # ── 统一清洗股票代码格式 ──────────────────────────────────────────
        for key in ta_signals:
            df_sig = ta_signals[key]
            if not df_sig.empty and '股票代码' in df_sig.columns:
                # 提取6位股票代码
                df_sig['股票代码'] = df_sig['股票代码'].astype(str).apply(
                    lambda x: re.search(r'(\d{6})', str(x)).group(1) if re.search(r'(\d{6})', str(x)) else x
                )
                ta_signals[key] = df_sig

        return ta_signals
