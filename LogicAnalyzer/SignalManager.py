import logging
import os
import re
from datetime import datetime

import pandas as pd
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas_ta as ta  # 勿删

logger = logging.getLogger(__name__)
from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
from LogicAnalyzer.KDJAnalyzer import AdvancedKDJAnalyzer
from LogicAnalyzer.SignalConstants import (
    KLineLevels, KLineDirection, CCILevels, KLinePatternCN,
    MarketSentiment, RSISignals, BOLLSignals
)
from ConfigParser import Config


class TASignalProcessor:
    """
    技术指标信号处理类
    
    负责计算和处理多种技术指标信号，包括：
    - MACD（单参数，用户可配置）
    - KDJ、CCI、RSI、BOLL
    - 背离检测
    - MACD趋势综合评分
    - 水平多因子交叉验证（各指标独立输出至 Excel）
    
    Attributes:
        analyzer: 股票分析器实例
        kdj_analyzer: KDJ 分析器
        macd_analyzer: MACD 分析器
        config: 配置管理器实例
    """

    def __init__(self, analyzer_instance: Any, config: Optional[Config] = None, executor=None) -> None:
        self.analyzer      = analyzer_instance
        self.kdj_analyzer  = AdvancedKDJAnalyzer()
        self.macd_analyzer = MACDAnalyzer()
        self.config        = config
        self.executor      = executor

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
        macro_adjust: float = 1.0,
        moneyflow_lookup: dict | None = None,
        forecast_lookup: dict | None = None,
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
            logger.debug("股票 %s 跳过: 数据不足(%s行)", pure_code, len(df) if not df.empty else 0)
            return None

        for col in ['close', 'open', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close', 'open', 'high', 'low'], inplace=True)
        if df.empty:
            logger.debug("股票 %s 跳过: close/open/high/low 全部空值", pure_code)
            return None
        required_cols = ['close', 'open', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.debug("股票 %s 跳过: 缺少必要列 %s", pure_code, missing_cols)
            return None
        if 'date' in df.columns:
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)

        try:
            df = self.macd_analyzer._custom_macd(df)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("股票 %s MACD计算跳过: %s", pure_code, e)
            return None

        result: dict[str, Any] = {'code': code, 'pure_code': pure_code}

        # K线形态检测（在 FullBull 评分之前执行，评分结果存入 df.attrs 供后续使用）
        try:
            kp = self._detect_kline_patterns(df, scan_window=60)
            if kp:
                result['kline_pattern'] = kp['signal']
                result['kline_pattern_score'] = kp['score']
                df.attrs['kline_pattern_score'] = kp['score']
                df.attrs['_kline_pattern_details'] = {'details': kp['details']}
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug("股票 %s K线形态检测跳过: %s", pure_code, e)

        weights = getattr(self.config, 'FULL_BULL_WEIGHTS', None)
        thresholds = getattr(self.config, 'FULL_BULL_THRESHOLDS', None)

        # 附加筹码分布数据
        code_str = str(code).lower()
        if code_str.startswith(('sh', 'sz', 'bj')):
            code_str = code_str[:2] + code_str[2:].zfill(6)
        else:
            code_str = code_str.zfill(6)
        # chip_lookup key 为纯6位代码（已剥离前缀）
        pure_code = code_str[2:] if code_str[:2] in ('sh', 'sz', 'bj') else code_str
        if pure_code in getattr(self, 'chip_lookup', {}):
            cd = self.chip_lookup[pure_code]
            df.attrs['chip_data'] = cd
            result['cost_95pct'] = cd.get('cost_95pct', None)

        # 附加资金流向数据
        if moneyflow_lookup and pure_code in moneyflow_lookup:
            mf_data = moneyflow_lookup[pure_code]
            df.attrs['moneyflow_data'] = mf_data
            result['net_mf_amount'] = mf_data.get('net_mf_amount', 0)

        # 附加业绩预告数据
        if forecast_lookup and pure_code in forecast_lookup:
            fc_data = forecast_lookup[pure_code]
            df.attrs['forecast_data'] = fc_data
            result['forecast_type'] = fc_data.get('type', '')

        # 流水线分析（统一入口）
        try:
            adj_thresholds = None
            if thresholds and macro_adjust < 1.0:
                adj_thresholds = {k: int(v * macro_adjust) for k, v in thresholds.items()}
            pipeline_params = {
                "regime": getattr(self.config, 'REGIME_DETECTION', {}),
                "divergence": getattr(self.config, 'DIVERGENCE_PARAMS', {}),
                "scoring": getattr(self.config, 'SCORING_PARAMS', {}),
                "technical": getattr(self.config, 'TECHNICAL_CONSTANTS', {}),
            }
            pipeline_result = self.macd_analyzer.pipeline_analysis(
                df,
                weights=weights, thresholds=adj_thresholds or thresholds,
                rule_thresholds=getattr(self.config, 'RULE_THRESHOLDS', None),
                params=pipeline_params,
            )
            result['pipeline'] = pipeline_result
            result['details'] = pipeline_result.get('details', {})
            result['stop_loss'] = pipeline_result.get('stop_loss')
            result['divergence_days'] = pipeline_result.get('divergence_days')
            result['divergence_price'] = pipeline_result.get('divergence_price')
            result['_current_dif'] = pipeline_result.get('current_dif')
            result['t1_target'] = pipeline_result.get('t1_target')
            result['t2_target'] = pipeline_result.get('t2_target')
            result['trailing_stop'] = pipeline_result.get('trailing_stop')
            result['exit_rrr'] = pipeline_result.get('exit_rrr')
            result['position_adjust'] = pipeline_result.get('position_adjust', 0.0)
            result['macd_trend_raw'] = pipeline_result.get('macd_trend', '')
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("股票 %s 管线分析跳过: %s", pure_code, e)

        try:
            detail_col = 'MACD_SIGNAL_DETAIL'
            if detail_col in df.columns:
                val = df[detail_col].iloc[-1]
                if pd.notna(val) and val != '':
                    result['macd_signal'] = val
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("股票 %s MACD信号详情跳过: %s", pure_code, e)

        try:
            kdj_signal = self.kdj_analyzer.calculate_kdj_signal_from_df(df)
            if kdj_signal:
                result['kdj_signal'] = kdj_signal
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug("股票 %s KDJ分析跳过: %s", pure_code, e)

        try:
            df.ta.cci(append=True, close='close', high='high', low='low')
            cci_cols = [col for col in df.columns if col.startswith('CCI_')]
            if cci_cols:
                current_cci = df[cci_cols[0]].iloc[-1]
                result['cci_signal'] = self._classify_cci_level(current_cci) or f'常态波动 ({current_cci:.2f})'
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug("股票 %s CCI分析跳过: %s", pure_code, e)

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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug("股票 %s RSI分析跳过: %s", pure_code, e)

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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug("股票 %s BOLL分析跳过: %s", pure_code, e)

        return result

    # ── K线形态检测 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_kline_patterns(df: pd.DataFrame, scan_window: int = 60) -> dict | None:
        """
        检测最近 scan_window 根 K 线的经典组合形态。

        使用 TA-Lib CDL 模式识别 + pandas_ta.cdl_pattern()。
        返回综合信号文本和评分（-10 ~ +10），供 FullBull 评分使用。
        """
        import pandas_ta as ta

        required = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required):
            return None

        df_window = df.tail(scan_window).copy()
        if len(df_window) < 30:
            return None

        open_s = df_window['open'].astype(float)
        high_s = df_window['high'].astype(float)
        low_s = df_window['low'].astype(float)
        close_s = df_window['close'].astype(float)

        result = ta.cdl_pattern(open_s, high_s, low_s, close_s, name='all')
        if result is None or result.empty:
            return None

        # TA-Lib 列名格式: CDL_ENGULFING, CDL_DOJI_10_0.1 等
        def _norm_name(col: str) -> str:
            for suffix in ('_10_0.1', '_0.1', '_10'):
                if suffix in col:
                    return col.split(suffix)[0]
            return col

        # 形态级别映射
        STRONG = {'CDL_ENGULFING', 'CDL_HIKKAKE', 'CDL_MORNINGSTAR', 'CDL_MORNINGDOJISTAR',
                  'CDL_EVENINGSTAR', 'CDL_EVENINGDOJISTAR', 'CDL_ABANDONEDBABY'}
        MEDIUM = {'CDL_DARKCLOUDCOVER', 'CDL_PIERCING', 'CDL_3WHITESOLDIERS', 'CDL_3BLACKCROWS',
                  'CDL_HARAMI', 'CDL_HARAMICROSS', 'CDL_BREAKAWAY', 'CDL_KICKING'}
        WEAK = {'CDL_DOJI', 'CDL_HAMMER', 'CDL_HANGINGMAN', 'CDL_SHOOTINGSTAR',
                'CDL_INVERTEDHAMMER', 'CDL_DRAGONFLYDOJI', 'CDL_GRAVESTONEDOJI',
                'CDL_LONGLEGGEDDOJI', 'CDL_SPINNINGTOP', 'CDL_TAKURI'}
        CONTINUOUS = {'CDL_RISEFALL3METHODS', 'CDL_GAPSIDESIDEWHITE', 'CDL_XSIDEGAP3METHODS', 'CDL_MATHOLD'}

        LABELS = {
            'CDL_ENGULFING': '吞没形态', 'CDL_HIKKAKE': '对策(类岛形反转)',
            'CDL_MORNINGSTAR': '早晨之星', 'CDL_MORNINGDOJISTAR': '早晨十字星',
            'CDL_EVENINGSTAR': '黄昏之星', 'CDL_EVENINGDOJISTAR': '黄昏十字星',
            'CDL_ABANDONEDBABY': '弃婴(岛形反转)',
            'CDL_DARKCLOUDCOVER': '乌云盖顶', 'CDL_PIERCING': '刺透形态',
            'CDL_3WHITESOLDIERS': '红三兵', 'CDL_3BLACKCROWS': '三只乌鸦',
            'CDL_HARAMI': '孕线', 'CDL_HARAMICROSS': '十字孕线',
            'CDL_BREAKAWAY': '脱离(类旗形)', 'CDL_KICKING': '踢形态',
            'CDL_DOJI': '十字星', 'CDL_HAMMER': '锤子线',
            'CDL_HANGINGMAN': '上吊线', 'CDL_SHOOTINGSTAR': '射击之星',
            'CDL_INVERTEDHAMMER': '倒锤子线',
            'CDL_DRAGONFLYDOJI': '蜻蜓十字', 'CDL_GRAVESTONEDOJI': '墓碑十字',
            'CDL_LONGLEGGEDDOJI': '长脚十字', 'CDL_SPINNINGTOP': '纺锤线',
            'CDL_TAKURI': '托里(类锤子)',
            'CDL_RISEFALL3METHODS': '上升/下降三法',
            'CDL_GAPSIDESIDEWHITE': '跳空并列阳线',
            'CDL_XSIDEGAP3METHODS': '侧跳空三法', 'CDL_MATHOLD': '待变(类旗形)',
        }

        LEVEL_ORDER = KLineLevels.LEVEL_ORDER
        LEVEL_MAP = {}
        for p in STRONG: LEVEL_MAP[p] = KLineLevels.STRONG_REVERSAL
        for p in MEDIUM: LEVEL_MAP[p] = KLineLevels.MEDIUM_REVERSAL
        for p in WEAK: LEVEL_MAP[p] = KLineLevels.WEAK_SIGNAL
        for p in CONTINUOUS: LEVEL_MAP[p] = KLineLevels.CONTINUOUS

        # 扫描窗口内所有非零模式
        detected = []
        for col in result.columns:
            base = _norm_name(col)
            series = result[col]
            nonzero = series[series != 0]
            if nonzero.empty:
                continue
            last_row_idx = nonzero.index[-1]
            last_val = nonzero.iloc[-1]
            bars_ago = len(series) - 1 - series.index.get_loc(last_row_idx)
            is_bullish = last_val > 0

            level = LEVEL_MAP.get(base)
            if level is None:
                continue

            detected.append({
                'base': base,
                'label': LABELS.get(base, base),
                'level': level,
                'direction': KLineDirection.BULLISH if is_bullish else KLineDirection.BEARISH,
                'bars_ago': bars_ago,
                'confirmed': bars_ago > 0,
                'score_val': last_val,
            })

        if not detected:
            return None

        # 按级别 + 时间排序，优先展示强反转且最新的
        detected.sort(key=lambda x: (LEVEL_ORDER.get(x['level'], 9), x['bars_ago']))

        # 生成综合信号文本
        strongest = detected[0]
        parts = []
        status_mark = '' if strongest['confirmed'] else ' [待确认]'
        parts.append(f"[{strongest['level']}] {strongest['label']} ({strongest['direction']}){status_mark}")

        # 统计多空
        bull_count = sum(1 for d in detected if d['direction'] == '看涨')
        bear_count = sum(1 for d in detected if d['direction'] == '看跌')
        if bull_count > bear_count * 1.5:
            parts.append('整体偏多')
        elif bear_count > bull_count * 1.5:
            parts.append('整体偏空')

        # 形态数量
        parts.append(f'形态{len(detected)}种')
        signal_text = ' | '.join(parts)

        # 评分（-10 ~ +10）
        score_raw = 0
        for d in detected:
            weight = 1.0
            if d['level'] == '强反转':
                weight = 3.0
            elif d['level'] == '中反转':
                weight = 2.0
            elif d['level'] == '弱信号':
                weight = 0.5
            else:
                weight = 0.3
            if not d['confirmed']:
                weight *= 0.5
            score_raw += weight if d['direction'] == '看涨' else -weight

        score = max(-10, min(10, int(score_raw)))

        return {'signal': signal_text, 'score': score, 'details': detected}

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
            - MACD_FULL_BULL: MACD趋势综合评分信号（7维度+趋势分类+管线结论）
            - KDJ: KDJ信号
            - CCI: CCI信号
            - RSI: RSI信号
            - BOLL: 布林带信号
            - KLINE_PATTERN: K线形态信号
        """
        
        ta_signals = {
            'KDJ':  pd.DataFrame(columns=['股票代码', 'KDJ_Signal']),
            'CCI':  pd.DataFrame(columns=['股票代码', 'CCI_Signal']),
            'RSI':  pd.DataFrame(columns=['股票代码', 'RSI_Signal']),
            'BOLL': pd.DataFrame(columns=['股票代码', 'BOLL_Signal']),
            # ── MACD趋势综合评分（7维度+趋势分类+管线结论） ──────────────
            'MACD_FULL_BULL': pd.DataFrame(columns=[
                '股票代码',
                'MACD趋势', '金叉信号', '柱状动能', 'DIF斜率', '背离信号', '量价配合', 'K线形态',
                '综合分析结论', '综合分析评分', '综合级别', '风险等级',
                'MACD趋势分类', 'macd_trend',
                'cost_95pct',
                '资金流净额',
                '背离距今', '背离位置',
                '止损价', 'T1目标价', 'T2目标价', '移动止损', '盈亏比',
                'position_adjust', 'macd_trend_raw',
            ]),
        }

        # ── 加载筹码分布数据 ───────────────────────────────────────────────
        self.chip_lookup = {}
        today_str = datetime.now().strftime('%Y%m%d')
        chip_path = os.path.join(getattr(self.config, 'HOME_DIRECTORY', '~/Downloads/CoreNews_Reports'), f"chip_distribution_{today_str}.csv")
        chip_path = os.path.expanduser(chip_path)
        if os.path.exists(chip_path):
            try:
                chip_df = pd.read_csv(chip_path)
                for _, row in chip_df.iterrows():
                    pure = row['symbol']
                    if pure.startswith(('sh', 'sz', 'bj')):
                        pure = pure[2:]
                    self.chip_lookup[pure] = row.to_dict()
                print(f"[ChipDist] 已加载 {len(self.chip_lookup)} 条筹码数据")
            except Exception as e:
                print(f"[ChipDist] 加载筹码数据失败: {e}")

        # ── 加载资金流向数据 ───────────────────────────────────────────────
        self.moneyflow_lookup = {}
        try:
            from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher
            mf_fetcher = MoneyFlowFetcher(self.config)
            mf_df = mf_fetcher.fetch_all()
            if mf_df is not None and not mf_df.empty:
                for _, row in mf_df.iterrows():
                    ts_code = str(row.get('ts_code', ''))
                    pure = ts_code.split('.')[0]  # 000001.SH → 000001
                    self.moneyflow_lookup[pure] = row.to_dict()
                print(f"[MoneyFlow] 已加载 {len(self.moneyflow_lookup)} 条资金流向数据")
        except Exception as e:
            print(f"[MoneyFlow] 加载资金流向失败: {e}")

        # ── 加载业绩预告数据 ──────────────────────────────────────────────
        self.forecast_lookup = {}
        try:
            from DataCollection.FinancialForecastFetcher import FinancialForecastFetcher
            fc_fetcher = FinancialForecastFetcher(self.config)
            fc_df = fc_fetcher.fetch_all()
            if fc_df is not None and not fc_df.empty:
                for _, row in fc_df.iterrows():
                    ts_code = str(row.get('ts_code', ''))
                    pure = ts_code.split('.')[0]
                    self.forecast_lookup[pure] = row.to_dict()
                print(f"[Forecast] 已加载 {len(self.forecast_lookup)} 条业绩预告数据")
        except Exception as e:
            print(f"[Forecast] 加载业绩预告失败: {e}")

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

        # ── 宏观过滤（全局一次性判断） ──────────────────────────────────────
        macro_result = None
        macro_adjust = 1.0
        if getattr(self.config, 'ENABLE_MACRO_FILTER', True):
            try:
                from LogicAnalyzer.MacroFilter import MacroFilter
                macro_result = MacroFilter.check(spot_df=spot_df)
                if macro_result.decision == 'SKIP_ALL':
                    print(f"[MacroFilter] {macro_result.detail} → SKIP_ALL，跳过当日分析")
                    return ta_signals
                elif macro_result.decision == 'CAUTION':
                    macro_adjust = macro_result.score_adjust
                    print(f"[MacroFilter] {macro_result.detail} → CAUTION，阈值上浮{int((1-macro_adjust)*100)}%")
                else:
                    print(f"[MacroFilter] {macro_result.detail} → NORMAL")
            except Exception as e:
                print(f"[MacroFilter] 异常: {e}")

        # ── 并行处理所有股票 ──────────────────────────────────────────────
        max_workers = getattr(self.config, 'SIGNAL_PROCESSING_PROCESSES', 2)

        results: list[dict[str, Any]] = []
        exec_signal = self.executor or ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {
                exec_signal.submit(
                    self._process_single_stock, code, valid_hist_df, macro_adjust,
                    self.moneyflow_lookup, self.forecast_lookup,
                ): code
                for code in set(all_codes)
            }
            for future in as_completed(futures):
                code = futures[future]
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    logger.warning("股票 %s 管线线程异常: %s", code, e)
        finally:
            if self.executor is None:
                exec_signal.shutdown(wait=True)

        # ── 从结果列表构建信号 DataFrame（避免逐行 O(n²) concat）─────────
        bull_rows: list[dict] = []
        kdj_rows: list[dict] = []
        cci_rows: list[dict] = []
        rsi_rows: list[dict] = []
        boll_rows: list[dict] = []
        kline_rows: list[dict] = []

        for r in results:
            code = r['code']

            pipeline = r.get('pipeline', {})
            detail = r.get('details', {}) or pipeline.get('details', {})
            if detail:
                bull_rows.append({
                    '股票代码': code,
                    'MACD趋势': detail.get('MACD趋势', {}).get('desc', ''),
                    '金叉信号': detail.get('金叉信号', {}).get('desc', ''),
                    '柱状动能': detail.get('柱状动能', {}).get('desc', ''),
                    'DIF斜率': detail.get('DIF斜率', {}).get('desc', ''),
                    '背离信号': detail.get('背离信号', {}).get('desc', ''),
                    '量价配合': detail.get('量价配合', {}).get('desc', ''),
                    'K线形态': detail.get('K线形态', {}).get('desc', ''),
                    '综合分析结论': pipeline.get('conclusion', ''),
                    '综合分析评分': pipeline.get('score', 0),
                    '综合级别': pipeline.get('level', ''),
                    '风险等级': pipeline.get('risk_level', ''),
                    'MACD趋势分类': pipeline.get('macd_trend', detail.get('MACD趋势', {}).get('desc', '')),
                    'macd_trend': pipeline.get('macd_trend', ''),
                    'cost_95pct': r.get('cost_95pct', None),
                    '资金流净额': r.get('net_mf_amount', 0),
                    '_current_dif': r.get('_current_dif', 0),
                    '背离距今': r.get('divergence_days'),
                    '背离位置': r.get('divergence_price'),
                    '止损价': r.get('stop_loss'),
                    'T1目标价': r.get('t1_target'),
                    'T2目标价': r.get('t2_target'),
                    '移动止损': r.get('trailing_stop'),
                    '盈亏比': r.get('exit_rrr'),
                    'position_adjust': r.get('position_adjust', 0.0),
                    'macd_trend_raw': r.get('macd_trend_raw', ''),
                })

            if 'kdj_signal' in r:
                kdj_rows.append({'股票代码': code, 'KDJ_Signal': r['kdj_signal']})

            if 'cci_signal' in r:
                cci_rows.append({'股票代码': code, 'CCI_Signal': r['cci_signal']})

            if 'rsi_signal' in r:
                rsi_rows.append({'股票代码': code, 'RSI_Signal': r['rsi_signal']})

            if 'boll_signal' in r:
                boll_rows.append({'股票代码': code, 'BOLL_Signal': r['boll_signal']})

            if 'kline_pattern' in r:
                kline_rows.append({'股票代码': code, 'K线形态信号': r['kline_pattern']})

        ta_signals['MACD_FULL_BULL'] = pd.DataFrame(bull_rows)
        ta_signals['KDJ'] = pd.DataFrame(kdj_rows)
        ta_signals['CCI'] = pd.DataFrame(cci_rows)
        ta_signals['RSI'] = pd.DataFrame(rsi_rows)
        ta_signals['BOLL'] = pd.DataFrame(boll_rows)
        ta_signals['KLINE_PATTERN'] = pd.DataFrame(kline_rows)

        # ── 统一清洗股票代码格式 ──────────────────────────────────────────
        for key in ta_signals:
            df_sig = ta_signals[key]
            if not df_sig.empty and '股票代码' in df_sig.columns:
                df_sig['股票代码'] = df_sig['股票代码'].astype(str).apply(
                    lambda x: re.search(r'(\d{6})', str(x)).group(1) if re.search(r'(\d{6})', str(x)) else x
                )
                ta_signals[key] = df_sig

        return ta_signals
