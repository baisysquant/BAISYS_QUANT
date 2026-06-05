import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class MACDAnalyzer:
    """
    双参数 MACD 分析器（标准 12-26-9 + 可配置第二周期）。

    功能：
      - 计算两套 MACD，区分零轴上/下金叉死叉
      - 背离检测（顶/底），量化背离强度，支持信号衰减
      - 量价配合验证
      - DIF 斜率/趋势强度（线性回归）
      - ATR 自适应 find_peaks distance
      - 完全多头综合评分（0~100）
      - 历史信号胜率回测

    注意：
      - 标准周期 (12,26,9) 强制保留，不可修改
      - 第二周期可由用户自定义（默认 6,13,5）
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1. 基础工具
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_peaks_troughs(series: np.ndarray, distance: int = 10):
        """使用 scipy.find_peaks 寻找序列的波峰和波谷。"""
        peaks, _ = find_peaks(series, distance=distance)
        troughs, _ = find_peaks(-series, distance=distance)
        return peaks, troughs

    @staticmethod
    def _adaptive_distance(df: pd.DataFrame, atr_period: int = 14, base: int = 5) -> int:
        """
        根据近期 ATR / 价格波动比，动态计算 find_peaks 的 distance 参数。
        波动越大 → distance 越大（过滤噪声峰）；波动越小 → distance 越小（捕捉转折）。
        返回范围：[base, 30]
        """
        high = df["high"] if "high" in df.columns else df["close"]
        low = df["low"] if "low" in df.columns else df["close"]
        close = df["close"]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(atr_period).mean().iloc[-1]
        price_range = close.iloc[-atr_period:].max() - close.iloc[-atr_period:].min()
        vol_ratio = atr / (price_range + 1e-9)  # 0 ~ 1

        distance = int(base + vol_ratio * 20)
        return max(base, min(distance, 30))

    @staticmethod
    def _slope_analysis(series: pd.Series, window: int = 5) -> dict:
        """
        对最近 window 根数据做线性回归，返回斜率、R² 和趋势描述。
        R² > 0.7 且斜率 > 0  →  明确上行
        R² > 0.7 且斜率 < 0  →  明确下行
        否则归为"震荡"
        """
        y = series.iloc[-window:].values
        x = np.arange(len(y), dtype=float)

        if len(y) < 3:
            return {"slope": 0.0, "r2": 0.0, "trend": "N/A"}

        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])

        y_pred = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-9)
        r2 = max(0.0, min(1.0, r2))

        if slope > 0 and r2 > 0.7:
            trend = "明确上行"
        elif slope > 0:
            trend = "弱势上行"
        elif slope < 0 and r2 > 0.7:
            trend = "明确下行"
        elif slope < 0:
            trend = "弱势下行"
        else:
            trend = "震荡"

        return {"slope": round(slope, 6), "r2": round(r2, 3), "trend": trend}

    @staticmethod
    def _signal_with_decay(
        signal_type: str | None, signal_idx: int | None, current_idx: int, half_life: int = 8
    ) -> float:
        """
        信号强度指数衰减：half_life 根 K 线后衰减至 50%。
        返回衰减系数 0.0 ~ 1.0。
        """
        if signal_type is None or signal_idx is None:
            return 0.0
        bars_elapsed = max(0, current_idx - signal_idx)
        decay = 0.5 ** (bars_elapsed / half_life)
        return round(float(decay), 3)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 背离检测（单参数组）
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_divergence_single_param(
        df: pd.DataFrame,
        price_series: pd.Series,
        indicator_series: pd.Series,
        distance: int = 10,
    ) -> tuple[str | None, int | None, float]:
        """
        检测顶/底背离，返回 (类型, 信号位置索引, 背离强度 0~1)。
        每次调用只返回一个信号（顶背离优先，因其风险意义更迫切）。

        背离强度 = 1 - (指标变化幅度 / 价格变化幅度)
        越接近 1.0 → 背离越极端。
        """
        price_arr = price_series.values
        ind_arr = indicator_series.values

        peaks_price, troughs_price = MACDAnalyzer._find_peaks_troughs(price_arr, distance)
        peaks_ind, troughs_ind = MACDAnalyzer._find_peaks_troughs(ind_arr, distance)

        def _strength(p_val1, p_val2, i_val1, i_val2) -> float:
            p_chg = abs(p_val2 - p_val1) / (abs(p_val1) + 1e-9)
            i_chg = abs(i_val2 - i_val1) / (abs(i_val1) + 1e-9)
            if p_chg < 1e-9:
                return 0.0
            return round(max(0.0, min(1.0, 1.0 - i_chg / p_chg)), 3)

        # ── 顶背离：价格创新高，指标未创新高 ──────────────────────────────
        if len(peaks_price) >= 2 and len(peaks_ind) >= 2:
            last_p, prev_p = int(peaks_price[-1]), int(peaks_price[-2])

            mask_last = peaks_ind[peaks_ind <= last_p]
            mask_prev = peaks_ind[peaks_ind <= prev_p]

            if len(mask_last) > 0 and len(mask_prev) > 0:
                ci_last = int(mask_last[-1])
                ci_prev = int(mask_prev[-1])

                if price_arr[last_p] > price_arr[prev_p] and ind_arr[ci_last] < ind_arr[ci_prev]:
                    s = _strength(price_arr[prev_p], price_arr[last_p], ind_arr[ci_prev], ind_arr[ci_last])
                    return "顶背离", last_p, s

        # ── 底背离：价格创新低，指标未创新低 ──────────────────────────────
        if len(troughs_price) >= 2 and len(troughs_ind) >= 2:
            last_t, prev_t = int(troughs_price[-1]), int(troughs_price[-2])

            mask_last = troughs_ind[troughs_ind <= last_t]
            mask_prev = troughs_ind[troughs_ind <= prev_t]

            if len(mask_last) > 0 and len(mask_prev) > 0:
                ci_last = int(mask_last[-1])
                ci_prev = int(mask_prev[-1])

                if price_arr[last_t] < price_arr[prev_t] and ind_arr[ci_last] > ind_arr[ci_prev]:
                    s = _strength(price_arr[prev_t], price_arr[last_t], ind_arr[ci_prev], ind_arr[ci_last])
                    return "底背离", last_t, s

        return None, None, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 3. MACD 计算
    # ─────────────────────────────────────────────────────────────────────────

    def _custom_macd(self, df: pd.DataFrame, second_params: tuple[int, int, int] = (6, 13, 5)) -> pd.DataFrame:
        """
        同时计算标准 (12, 26, 9) 和第二周期两套 MACD。

        参数：
          df: 包含 close 列的 DataFrame
          second_params: 第二周期的 (快线, 慢线, 信号线)，默认 (6, 13, 5)
                        此参数为必填项，不可为 None 或 (0,0,0)

        新增：
          - MACD_HIST_{name}  柱状值（×2 标准显示）
          - MACD_{name}_CROSS  1=金叉 / -1=死叉 / 0=无
          - MACD_{name}_SIGNAL_DETAIL  零轴上/下金叉/死叉文字
        """
        if "close" not in df.columns:
            return df

        close = df["close"]

        # 标准周期（强制保留）
        macd_periods = {
            "12269": (12, 26, 9),
        }

        # 添加第二周期（必填）
        fast, slow, signal = second_params
        name = f"{fast}{slow}{signal}"
        macd_periods[name] = second_params

        for name, (fast, slow, signal) in macd_periods.items():
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=signal, adjust=False).mean()

            df[f"DIF_{name}"] = dif
            df[f"DEA_{name}"] = dea
            df[f"MACD_HIST_{name}"] = 2 * (dif - dea)

            prev_dif = dif.shift(1).fillna(dea.shift(1).fillna(0))
            prev_dea = dea.shift(1).fillna(0)

            golden = (dif > dea) & (prev_dif <= prev_dea)
            dead = (dif < dea) & (prev_dif >= prev_dea)

            df[f"MACD_{name}_SIGNAL_DETAIL"] = np.where(
                golden,
                np.where((dif > 0) & (dea > 0), "零轴上金叉", "零轴下金叉"),
                np.where(dead, np.where((dif < 0) & (dea < 0), "零轴下死叉", "零轴上死叉"), ""),
            )
            df[f"MACD_{name}_CROSS"] = np.where(golden, 1, np.where(dead, -1, 0))

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 4. 动能状态
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_macd_momentum(df: pd.DataFrame, dif_col: str, dea_col: str) -> str:
        """
        计算 MACD 动能状态：加速上涨 / 减速上涨 / 加速下跌 / 减速下跌。
        判断依据：DIF 是否在 DEA 上方，以及 DIF 的最新变化方向。
        """
        if len(df) < 2:
            return "N/A (数据不足)"

        latest_dif = df[dif_col].iloc[-1]
        latest_dea = df[dea_col].iloc[-1]
        dif_change = latest_dif - df[dif_col].iloc[-2]

        if latest_dif >= latest_dea:
            return "加速上涨 (红柱加长)" if dif_change > 0 else "减速上涨 (红柱缩短)"
        else:
            return "加速下跌 (绿柱加长)" if dif_change < 0 else "减速下跌 (绿柱缩短)"

    # ─────────────────────────────────────────────────────────────────────────
    # 4b. 动能连续评分（替代三档分类，支持连续标准化分数 0~20）
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame, dif_col: str, dea_col: str, window: int = 5, max_score: int = 20) -> tuple[str, int]:
        if len(df) < window + 2:
            return "数据不足", 0

        hist = df[dif_col] - df[dea_col]
        cur_hist = hist.iloc[-1]
        prev_hist = hist.iloc[-2]
        hist_change = cur_hist - prev_hist

        recent_hist = hist.iloc[-window:]
        hist_vol = max(recent_hist.std(), 1e-9)
        norm_change = hist_change / hist_vol  # z-score-like

        is_bull = cur_hist > 0

        if is_bull:
            ratio = norm_change / (norm_change + 1) if norm_change >= 0 else norm_change / (norm_change - 1)
            score = min(max_score, max(0, int(max_score * (0.5 + 0.5 * ratio))))
            desc = f"加速上涨 (红柱加长, z={norm_change:.2f})" if norm_change > 0 else f"减速上涨 (红柱缩短, z={norm_change:.2f})"
        else:
            max_bear = max(8, max_score * 2 // 5)
            ratio = abs(norm_change) / (abs(norm_change) + 1)
            score = min(max_bear, max(0, int(max_bear * ratio)))
            desc = f"加速下跌 (绿柱加长, z={norm_change:.2f})" if norm_change < 0 else f"减速下跌 (绿柱缩短, z={norm_change:.2f})"

        return desc, score

    # ─────────────────────────────────────────────────────────────────────────
    # 5. 量价配合验证
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _volume_confirmation(df: pd.DataFrame, signal_type: str | None, peak_idx: int, lookback: int = 5) -> dict:
        """
        在背离峰/谷附近检查成交量是否配合信号方向。
          底背离：量能放大（vol_ratio >= 1.2）→ 确认买入
          顶背离：量能萎缩（vol_ratio <= 0.8）→ 确认卖出
        返回 dict：{'confirmed': bool, 'vol_ratio': float, 'reason': str}
        """
        if "volume" not in df.columns or signal_type is None:
            return {"confirmed": False, "vol_ratio": None, "reason": "无量数据"}

        vol_at_signal = df["volume"].iloc[peak_idx]
        vol_avg_before = df["volume"].iloc[max(0, peak_idx - lookback) : peak_idx].mean()
        vol_ratio = float(vol_at_signal / (vol_avg_before + 1e-9))

        if signal_type == "底背离":
            confirmed = vol_ratio >= 1.2
            tag = "放大" if confirmed else "未放大"
        else:
            confirmed = vol_ratio <= 0.8
            tag = "萎缩" if confirmed else "未萎缩"

        reason = f"{'确认' if confirmed else '否定'}: 信号处量能{tag}，量比均值 {vol_ratio:.2f}x"
        return {"confirmed": confirmed, "vol_ratio": round(vol_ratio, 2), "reason": reason}

    # ─────────────────────────────────────────────────────────────────────────
    # 5b. 通用量价趋势评分（不依赖背离锚点，独立奖励项）
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_kline_pattern(df: pd.DataFrame, max_score: int = 10) -> tuple[str, int]:
        """
        根据 K 线形态检测结果给出评分（0~max_score）。

        优先从 df.attrs 读取 SignalManager 传入的复合评分，
        若无则基于最近 20 根 K 线做简易反转特征评分。
        """
        attrs_score = df.attrs.get('kline_pattern_score', None)
        if attrs_score is not None:
            raw = attrs_score
        else:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            open_ = df['open'].astype(float)
            recent = df.tail(20)
            raw = 0.0
            for i in range(len(recent) - 1, max(len(recent) - 6, -1), -1):
                body = abs(recent['close'].iloc[i] - recent['open'].iloc[i])
                lower = min(recent['open'].iloc[i], recent['close'].iloc[i]) - recent['low'].iloc[i]
                upper = recent['high'].iloc[i] - max(recent['open'].iloc[i], recent['close'].iloc[i])
                if body > 0 and lower > body * 2 and upper < body * 0.5:
                    raw += 1 if recent['close'].iloc[i] > recent['open'].iloc[i] else -1
                if body > 0 and upper > body * 2 and lower < body * 0.5:
                    raw += 1 if recent['close'].iloc[i] > recent['open'].iloc[i] else -1
            for i in range(len(recent) - 2):
                if all(recent['close'].iloc[i + j] > recent['open'].iloc[i + j] for j in range(3)):
                    raw += 1
                if all(recent['close'].iloc[i + j] < recent['open'].iloc[i + j] for j in range(3)):
                    raw -= 1

        raw_norm = max(-1.0, min(1.0, raw / 10.0))
        score = int((raw_norm + 1.0) / 2.0 * max_score)
        if raw_norm > 0.3:
            desc = f"偏多形态 (score={raw:.1f})"
        elif raw_norm < -0.3:
            desc = f"偏空形态 (score={raw:.1f})"
        else:
            desc = f"中性形态 (score={raw:.1f})"
        return desc, score

    @staticmethod
    def _volume_price_trend_score(df: pd.DataFrame, lookback: int = 5, max_bonus: int = 10) -> tuple[str, int]:
        if len(df) < lookback + 1 or "volume" not in df.columns:
            return "数据不足", 0

        closes = df["close"].iloc[-lookback:].values
        volumes = df["volume"].iloc[-lookback:].values

        pct_change = (closes[-1] - closes[0]) / (closes[0] + 1e-9)
        vol_early = volumes[:2].mean()
        vol_trend = (volumes[-1] - vol_early) / (vol_early + 1e-9)

        half = max_bonus // 2
        if pct_change > 0.02 and vol_trend > 0.1:
            return f"量价齐升 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", max_bonus
        elif pct_change > 0.02:
            return f"价涨量缩 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", half
        elif pct_change < -0.02 and vol_trend > 0.1:
            return f"放量下跌 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", -half
        elif pct_change < -0.02:
            return f"缩量下跌 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", 0
        else:
            return f"量价平淡 (价格{pct_change:+.2%}, 量{vol_trend:+.2%})", 0

    # ─────────────────────────────────────────────────────────────────────────
    # 6. 背离综合检测（修复重复调用 Bug，支持强度 + 衰减）
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def detect_combined_divergence(
        df: pd.DataFrame,
        distance_slow: int = 25,
        distance_fast: int = 12,
        recent_window: int = 5,
        decay_half_life: int = 8,
        second_period_name: str = "6135",  # 第二周期的名称（如 '6135'）
    ) -> dict:
        """
        检测两套 MACD 的顶/底背离并合并为交易信号。

        参数：
          df: 包含 DIF_12269, DEA_12269, DIF_{second_period_name}, DEA_{second_period_name} 的 DataFrame
          distance_slow: 标准周期的 find_peaks distance
          distance_fast: 第二周期的 find_peaks distance
          recent_window: 信号有效窗口
          decay_half_life: 衰减半衰期
          second_period_name: 第二周期的名称（默认 '6135'）

        修复说明：
          原代码每套参数调用两次（分别存为 top/bot），实际两次返回同一结果。
          现改为每套参数只调用一次，通过返回的信号类型区分顶/底。

        返回字段：
          combined_signal  最终综合信号文字
          div_12269        顶/底背离类型（或 ''）
          idx_12269        信号位置
          strength_12269   背离强度 0~1
          decay_12269      当前衰减系数 0~1
          div_{second_period_name} / idx_{second_period_name} / strength_{second_period_name} / decay_{second_period_name}  同上
        """
        current_idx = len(df) - 1

        def _is_recent(idx):
            return idx is not None and (current_idx - idx) <= recent_window

        # ── 每套参数只调一次 ──────────────────────────────────────────────
        div_12269, idx_12269, str_12269 = MACDAnalyzer._detect_divergence_single_param(
            df, df["close"], df["DIF_12269"], distance=distance_slow
        )

        dif_col_second = f"DIF_{second_period_name}"
        if dif_col_second in df.columns:
            div_second, idx_second, str_second = MACDAnalyzer._detect_divergence_single_param(
                df, df["close"], df[dif_col_second], distance=distance_fast
            )
        else:
            div_second, idx_second, str_second = None, None, 0.0

        # 衰减系数（替代硬截断 is_recent）
        decay_12269 = MACDAnalyzer._signal_with_decay(div_12269, idx_12269, current_idx, decay_half_life)
        decay_second = MACDAnalyzer._signal_with_decay(div_second, idx_second, current_idx, decay_half_life)

        # 有效性：衰减后强度 × 背离强度 > 阈值
        eff_12269 = decay_12269 * str_12269
        eff_second = decay_second * str_second
        THRESHOLD = 0.15  # 综合有效阈值，可调

        top_12269 = div_12269 == "顶背离" and eff_12269 >= THRESHOLD
        bot_12269 = div_12269 == "底背离" and eff_12269 >= THRESHOLD
        top_second = div_second == "顶背离" and eff_second >= THRESHOLD
        bot_second = div_second == "底背离" and eff_second >= THRESHOLD

        # 战术层当前状态
        if dif_col_second in df.columns:
            fast_golden = df[dif_col_second].iloc[-1] > df[f"DEA_{second_period_name}"].iloc[-1]
            fast_dead = df[dif_col_second].iloc[-1] < df[f"DEA_{second_period_name}"].iloc[-1]
        else:
            fast_golden = False
            fast_dead = False

        slow_above = df["DIF_12269"].iloc[-1] > 0

        # ── 信号优先级（从强到弱）────────────────────────────────────────
        if bot_12269 and bot_second and fast_golden:
            combined = "战略底背离 + 战术金叉确认 (强烈买入信号)"
        elif top_12269 and top_second and fast_dead:
            combined = "战略顶背离 + 战术死叉确认 (强烈卖出信号)"
        elif bot_12269 and bot_second:
            combined = "双重底背离 (强烈买入关注)"
        elif top_12269 and top_second:
            combined = "双重顶背离 (强烈卖出预警)"
        elif bot_12269:
            combined = "12269 底背离 (战略买入预警)"
        elif top_12269:
            combined = "12269 顶背离 (战略卖出预警)"
        elif bot_second:
            combined = (
                f"{second_period_name} 底背离 (可考虑买入)"
                if slow_above
                else f"{second_period_name} 底背离 (大趋势偏空，谨慎)"
            )
        elif top_second:
            combined = (
                f"{second_period_name} 顶背离 (需结合大趋势)"
                if slow_above
                else f"{second_period_name} 顶背离 (可考虑卖出)"
            )
        else:
            combined = ""

        return {
            "combined_signal": combined,
            # 12269
            "div_12269": div_12269 or "",
            "idx_12269": idx_12269,
            "strength_12269": str_12269,
            "decay_12269": decay_12269,
            # 第二周期
            f"div_{second_period_name}": div_second or "",
            f"idx_{second_period_name}": idx_second,
            f"strength_{second_period_name}": str_second,
            f"decay_{second_period_name}": decay_second,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 7. 历史信号胜率回测
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def backtest_signal_winrate(
        df: pd.DataFrame,
        signal_col: str,
        target_signal: str,
        forward_bars: int = 5,
    ) -> dict:
        """
        统计 signal_col 列中历史出现 target_signal 的位置，
        计算其后 forward_bars 根 K 线的收益分布和胜率。

        返回：
          sample_count  有效样本数
          win_rate      胜率（收益 > 0 的比例）
          avg_return    平均收益率
          max_gain      最大单次盈利
          max_loss      最大单次亏损
        """
        if signal_col not in df.columns:
            return {"sample_count": 0, "win_rate": None, "avg_return": None, "max_gain": None, "max_loss": None}

        hit_locs = [df.index.get_loc(i) for i in df.index[df[signal_col] == target_signal]]

        results = []
        for loc in hit_locs:
            end_loc = min(loc + forward_bars, len(df) - 1)
            if end_loc <= loc:
                continue
            entry = df["close"].iloc[loc]
            exit_ = df["close"].iloc[end_loc]
            results.append((exit_ - entry) / (entry + 1e-9))

        if not results:
            return {"sample_count": 0, "win_rate": None, "avg_return": None, "max_gain": None, "max_loss": None}

        arr = np.array(results)
        return {
            "sample_count": len(arr),
            "win_rate": round(float((arr > 0).mean()), 3),
            "avg_return": round(float(arr.mean()), 4),
            "max_gain": round(float(arr.max()), 4),
            "max_loss": round(float(arr.min()), 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 8. 市场情景识别（Market Regime Detection）
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def detect_market_regime(df: pd.DataFrame, boll_col: str | None = None) -> str:
        """
        识别人市场状态（Gate 0 前置判断）。

        返回情景标签:
          STRONG_TREND, WEAK_TREND, BOTTOM_REVERSAL, TOP_RISK, OSCILLATION, UNCLEAR
        """
        close = df['close'].astype(float)

        # ── 计算均线（若无则现场计算） ─────────────────────────────────
        ma5 = df['close'].rolling(5).mean().iloc[-1] if 'close' in df.columns else 0
        ma10 = df['close'].rolling(10).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma30 = df['close'].rolling(30).mean().iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]

        # 均线方向
        ma_bullish = ma5 > ma10 > ma20 > ma30 > ma60
        ma_bearish = ma5 < ma10 < ma20 < ma30 < ma60

        # ── MACD 状态 ─────────────────────────────────────────────────────
        dif = df['DIF_12269'].iloc[-1] if 'DIF_12269' in df.columns else 0
        dea = df['DEA_12269'].iloc[-1] if 'DEA_12269' in df.columns else 0
        hist = df['DIF_12269'] - df['DEA_12269'] if 'DIF_12269' in df.columns and 'DEA_12269' in df.columns else None
        momentum_positive = hist.iloc[-1] > 0 if hist is not None else False

        # DIF 斜率
        slope_info = MACDAnalyzer._slope_analysis(df['DIF_12269'] if 'DIF_12269' in df.columns else df['close'])
        slope_positive = slope_info['slope'] > 0

        # ── 布林带宽（震荡判断） ─────────────────────────────────────────
        is_narrow_boll = False
        if boll_col and boll_col in df.columns:
            recent_bw = df[boll_col].iloc[-5:].mean()
            hist_bw = df[boll_col].mean()
            is_narrow_boll = recent_bw < hist_bw * 0.8

        # ── 背离与 K 线（仅在调用方传入 attrs 时可用） ───────────────────
        # 依赖 df.attrs 中的 kline_pattern_score 和 divergence 数据

        # ── 判定 ─────────────────────────────────────────────────────────
        if ma_bullish and slope_positive and momentum_positive:
            return 'STRONG_TREND'
        if ma_bearish and dif < 0 and not momentum_positive:
            return 'WEAK_TREND'
        if is_narrow_boll and abs(hist.iloc[-1]) < 0.1 * close.std() if hist is not None and len(df) > 30 else False:
            return 'OSCILLATION'

        # 需要外部数据的特殊情景通过 attrs 传入
        regime_from_attrs = df.attrs.get('_regime_hint', None)
        if regime_from_attrs:
            return regime_from_attrs

        return 'UNCLEAR'

    # ─────────────────────────────────────────────────────────────────────────
    # 8b. 级联流水线分析（替代 analyze_full_bull）
    # ─────────────────────────────────────────────────────────────────────────

    def pipeline_analysis(
        self,
        df: pd.DataFrame,
        second_params: tuple[int, int, int] = (6, 13, 5),
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
    ) -> dict:
        """
        级联流水线综合分析。

        Gate 0: 情景识别
        Gate 1: 信号确认（规则引擎 R03/R04/R08）
        Gate 2: 风险评估（规则引擎 R01/R05）
        Gate 3: 综合裁定（规则引擎 R02/R06/R07 + 情景权重矩阵）

        返回:
          score, level, conclusion, regime, risk_level, triggered_rules, details, ...
        """
        from LogicAnalyzer.ScoringRules import execute_rules

        if weights is None:
            weights = {"零轴条件": 20, "战略金叉": 15, "战术金叉": 10, "动能": 15,
                       "DIF斜率": 10, "背离信号": 10, "量价配合": 10, "K线形态": 10}
        if thresholds is None:
            thresholds = {"fully_bull": 80, "bullish": 60, "oscillate": 40}

        fast, slow, signal = second_params
        second_period_name = f"{fast}{slow}{signal}"

        # ── 前置指标计算（若不足则补充） ──────────────────────────────────
        if 'MA_5' not in df.columns:
            for p in [5, 10, 20, 30, 60]:
                df[f'MA_{p}'] = df['close'].rolling(p).mean()

        # 布林带宽
        boll_bw_col = None
        boll_upper_cols = [c for c in df.columns if c.startswith('BBU_')]
        boll_lower_cols = [c for c in df.columns if c.startswith('BBL_')]
        if boll_upper_cols and boll_lower_cols:
            bw_col = 'BOLL_BANDWIDTH'
            if bw_col not in df.columns:
                df[bw_col] = (df[boll_upper_cols[0]] - df[boll_lower_cols[0]]) / df['close']
            boll_bw_col = bw_col

        # ── 计算各因子数据 ──────────────────────────────────────────────────
        # 背离检测
        dist_slow = self._adaptive_distance(df, base=10)
        dist_fast = self._adaptive_distance(df, base=5)
        div_signals = self.detect_combined_divergence(
            df, distance_slow=dist_slow, distance_fast=dist_fast,
            second_period_name=second_period_name,
        )

        # 动能
        mom_desc, mom_score = self._calculate_momentum_score(df, 'DIF_12269', 'DEA_12269', max_score=20)

        # DIF 斜率
        slope_info = self._slope_analysis(df['DIF_12269'], window=5)

        # 量价趋势
        vol_desc, vol_score = self._volume_price_trend_score(df, max_bonus=10)

        # K 线形态评分
        kp_result = self._score_kline_pattern(df, max_score=10)

        # ── 构建 state ──────────────────────────────────────────────────────
        state = {
            'df': df,
            'regime': None,
            'signal_list': [],
            'risk_level': 'NONE',
            'risk_desc': '',
            'conclusion': '',
            'level': 'C',
            'score': 0,
            'triggered_rules': [],
            '_notes': [],
            'divergence': div_signals,
            'kline_data': df.attrs.get('_kline_pattern_details', None),
            'volume_trend': (vol_desc, vol_score),
            'momentum': {'desc': mom_desc, 'score': mom_score},
            'slope': slope_info,
            'chip_data': df.attrs.get('chip_data', None),
        }

        # ═══════════════════════════════════════════════════════════════════
        # Gate 0: 情景识别
        # ═══════════════════════════════════════════════════════════════════
        boll_col_name = boll_bw_col if boll_bw_col else None
        regime = MACDAnalyzer.detect_market_regime(df, boll_col=boll_col_name)
        state['regime'] = regime

        # 弱势趋势 → 直接终止
        if regime == 'WEAK_TREND':
            state['level'] = 'C'
            state['conclusion'] = 'C: 弱势趋势，环境不配合'
            state['score'] = 0
            return _pipeline_output(state)

        # 顶部风险 → 进入 R01 检查
        # （R01 在 Gate 2 规则引擎中处理）

        # ═══════════════════════════════════════════════════════════════════
        # Gate 1: 信号确认
        # ═══════════════════════════════════════════════════════════════════
        signal_list = []

        # MACD 金叉信号
        detail_12269 = df['MACD_12269_SIGNAL_DETAIL'].iloc[-1] if 'MACD_12269_SIGNAL_DETAIL' in df.columns else ''
        if '金叉' in str(detail_12269):
            if '零轴上' in str(detail_12269):
                signal_list.append({'type': 'MACD_金叉', 'confidence': 'high', 'desc': '零轴上金叉'})
            else:
                signal_list.append({'type': 'MACD_金叉', 'confidence': 'medium', 'desc': '零轴下金叉'})

        # 底背离信号
        cs = div_signals.get('combined_signal', '')
        if '底背离' in cs:
            strength = div_signals.get('strength_12269', 0) or div_signals.get(f'strength_{second_period_name}', 0)
            confidence = 'high' if strength > 0.6 else 'medium'
            signal_list.append({'type': '底背离', 'confidence': confidence, 'desc': cs})

        # K 线反转信号
        kd = state.get('kline_data')
        if kd and kd.get('details'):
            for d in kd['details']:
                if d['direction'] == '看涨' and d['level'] in ('强反转', '中反转'):
                    confidence = 'high' if d['level'] == '强反转' else 'medium'
                    signal_list.append({'type': 'K线反转', 'confidence': confidence, 'desc': d['label']})
                    break

        # 持续多头信号：强趋势 + DIF>0 → 视为趋势延续信号
        if not signal_list:
            regime = state.get('regime', '')
            last_dif = df['MACD_12269_DIF'].iloc[-1] if 'MACD_12269_DIF' in df.columns else 0
            if regime == 'STRONG_TREND' and last_dif > 0:
                signal_list.append({
                    'type': '强势延续',
                    'confidence': 'high',
                    'desc': 'MACD多头趋势持续'
                })

        state['signal_list'] = signal_list

        # 规则引擎 Gate 1
        execute_rules(state, gate=1)

        # 无信号 → 终止
        if not signal_list:
            state['level'] = 'C'
            state['conclusion'] = 'C: 无明确入场信号'
            return _pipeline_output(state)

        # ═══════════════════════════════════════════════════════════════════
        # Gate 2: 风险评估
        # ═══════════════════════════════════════════════════════════════════
        execute_rules(state, gate=2)

        if state['risk_level'] == 'HIGH':
            return _pipeline_output(state)

        # 手动风险兜底
        has_top_div = '顶背离' in cs or '卖出' in cs
        if has_top_div and 'R01' not in state['triggered_rules']:
            state['risk_level'] = 'MEDIUM'
            state['risk_desc'] = '顶背离(未达R01阈值)'

        # 筹码分布风险评估
        chip = state.get('chip_data')
        if chip:
            winner_rate = chip.get('winner_rate', 0)
            close = df['close'].iloc[-1]
            cost_95pct = chip.get('cost_95pct', 0)
            cost_5pct = chip.get('cost_5pct', 0)

            # 高位获利盘风险
            if winner_rate > 80 and state['regime'] in ('WEAK_TREND', 'TOP_RISK', 'UNCLEAR'):
                if state['risk_level'] in ('NONE', 'LOW'):
                    state['risk_level'] = 'MEDIUM'
                    state['risk_desc'] += ' 筹码高位获利[>80%]'

            # 接近成本上沿阻力
            if cost_95pct > 0 and close >= cost_95pct * 0.95 and winner_rate > 70:
                if state['risk_level'] in ('NONE', 'LOW'):
                    state['risk_level'] = 'MEDIUM'
                    state['risk_desc'] += ' 筹码阻力位'

            # 筹码极度集中（集中度高 = 窄区间）
            if cost_95pct > 0 and cost_5pct > 0:
                chip_range = (cost_95pct - cost_5pct) / max(cost_5pct, 0.01)
                if chip_range < 0.15:
                    # 筹码高度集中，根据位置判断
                    cost_50pct = chip.get('cost_50pct', 0)
                    if close > cost_50pct and winner_rate > 70:
                        if state['risk_level'] not in ('HIGH',):
                            state['risk_level'] = 'HIGH' if state['regime'] in ('WEAK_TREND', 'TOP_RISK') else 'MEDIUM'
                            state['risk_desc'] += ' 筹码密集区高位'

        # ═══════════════════════════════════════════════════════════════════
        # Gate 3: 综合裁定
        # ═══════════════════════════════════════════════════════════════════
        execute_rules(state, gate=3)

        # ── 情景权重矩阵 ──────────────────────────────────────────────────
        regime_mult = {
            'STRONG_TREND':   {'zero': 1.2, 'strat': 1.0, 'tact': 1.0, 'mom': 1.5, 'slope': 1.5, 'div': 1.0, 'vol': 1.0, 'kp': 0.3},
            'BOTTOM_REVERSAL':{'zero': 0.5, 'strat': 1.0, 'tact': 1.0, 'mom': 0.8, 'slope': 0.8, 'div': 2.0, 'vol': 1.5, 'kp': 1.5},
            'TOP_RISK':       {'zero': 0.3, 'strat': 0.5, 'tact': 0.5, 'mom': 0.5, 'slope': 0.5, 'div': 2.0, 'vol': 1.5, 'kp': 2.0},
            'OSCILLATION':    {'zero': 0.5, 'strat': 0.5, 'tact': 0.5, 'mom': 0.3, 'slope': 0.3, 'div': 1.5, 'vol': 0.5, 'kp': 1.2},
            'UNCLEAR':        {'zero': 1.0, 'strat': 1.0, 'tact': 1.0, 'mom': 1.0, 'slope': 1.0, 'div': 1.0, 'vol': 1.0, 'kp': 1.0},
        }
        mult = regime_mult.get(regime, regime_mult['UNCLEAR'])

        # 各维度计算（使用权重矩阵修正）
        w_zero = int(weights['零轴条件'] * mult['zero'])
        w_strat = int(weights['战略金叉'] * mult['strat'])
        w_tact = int(weights['战术金叉'] * mult['tact'])
        w_mom = int(weights['动能'] * mult['mom'])
        w_slope = int(weights['DIF斜率'] * mult['slope'])
        w_div = int(weights['背离信号'] * mult['div'])
        w_vol = int(weights['量价配合'] * mult['vol'])
        w_kp = int(weights['K线形态'] * mult['kp'])

        scores = {}

        # 零轴条件
        dif_above = df['DIF_12269'].iloc[-1] > 0
        dea_above = df['DEA_12269'].iloc[-1] > 0
        if dif_above and dea_above:
            scores['零轴条件'] = ('DIF/DEA 均在零轴上', w_zero)
        elif dif_above:
            scores['零轴条件'] = ('DIF 在零轴上，DEA 仍在下方', w_zero // 2)
        else:
            scores['零轴条件'] = ('DIF 在零轴下', 0)

        # 战略金叉
        slow_detail = df['MACD_12269_SIGNAL_DETAIL'].iloc[-1] if 'MACD_12269_SIGNAL_DETAIL' in df.columns else ''
        slow_bull = df['DIF_12269'].iloc[-1] > df['DEA_12269'].iloc[-1]
        if '零轴上金叉' in str(slow_detail):
            scores['战略金叉'] = ('12269 零轴上金叉', w_strat)
        elif '零轴下金叉' in str(slow_detail):
            scores['战略金叉'] = ('12269 零轴下金叉', w_strat // 2)
        elif slow_bull:
            scores['战略金叉'] = ('12269 多头持续', int(w_strat * 0.75))
        else:
            scores['战略金叉'] = ('12269 空头排列', 0)

        # 战术金叉
        fast_detail_col = f'MACD_{second_period_name}_SIGNAL_DETAIL'
        fast_dif_col = f'DIF_{second_period_name}'
        fast_dea_col = f'DEA_{second_period_name}'
        if fast_detail_col in df.columns:
            fast_detail = df[fast_detail_col].iloc[-1]
            fast_bull = df[fast_dif_col].iloc[-1] > df[fast_dea_col].iloc[-1]
            if '零轴上金叉' in str(fast_detail):
                scores['战术金叉'] = (f'{second_period_name} 零轴上金叉', w_tact)
            elif fast_bull:
                scores['战术金叉'] = (f'{second_period_name} 多头持续', int(w_tact * 0.65))
            else:
                scores['战术金叉'] = (f'{second_period_name} 空头/死叉', 0)

        # 动能
        prefix = '强势' if mom_score >= w_mom // 2 else ('关注' if mom_score >= w_mom // 4 else '弱势')
        scores['动能'] = (f'{prefix}: {mom_desc}', min(mom_score, w_mom))

        # DIF 斜率
        if slope_info['trend'] == '明确上行':
            scores['DIF斜率'] = (f"确认 {slope_info['trend']} (R²={slope_info['r2']})", w_slope)
        elif '上行' in slope_info['trend']:
            scores['DIF斜率'] = (f"关注 {slope_info['trend']} (R²={slope_info['r2']})", int(w_slope * 0.55))
        else:
            scores['DIF斜率'] = (f"弱势 {slope_info['trend']} (R²={slope_info['r2']})", 0)

        # 背离信号
        div_type = div_signals.get('div_12269') or div_signals.get(f'div_{second_period_name}')
        div_str = div_signals.get('strength_12269') or div_signals.get(f'strength_{second_period_name}', 0.0)
        div_decay = div_signals.get('decay_12269') or div_signals.get(f'decay_{second_period_name}', 0.0)
        eff = round(div_str * div_decay, 3)
        if '底背离' in cs:
            div_score = int(w_div * (0.5 + 0.5 * eff))
            scores['背离信号'] = (f'确认 {cs} (强度={div_str}, 衰减={div_decay}, 有效={eff})', div_score)
        elif '顶背离' in cs or '卖出' in cs:
            scores['背离信号'] = (f'否定 {cs}（一票否决）', 0)
        else:
            scores['背离信号'] = ('中性: 无背离信号', 0)

        # 量价
        has_top_div = '顶背离' in cs or '卖出' in cs
        vol_bonus = 0 if has_top_div else vol_score
        scores['量价配合'] = (vol_desc, min(vol_bonus, w_vol))

        # K线形态
        scores['K线形态'] = kp_result

        # ── 汇总 ──────────────────────────────────────────────────────────
        base_keys = [k for k in scores if k != '量价配合']
        total_base = sum(scores[k][1] for k in base_keys)
        bonus = scores['量价配合'][1]

        total_max_base = w_zero + w_strat + w_tact + w_mom + w_slope + w_div + w_kp
        total_base = max(0, min(total_max_base, total_base))
        total = max(0, min(total_max_base + w_vol, total_base + bonus))

        # ── 级别判定 ──────────────────────────────────────────────────────
        notes = state.get('_notes', [])
        notes_str = '+'.join(notes) if notes else ''

        is_high_risk = state['risk_level'] == 'HIGH' or has_top_div

        if is_high_risk:
            state['level'] = 'D'
            conclusion_parts = ['D: 顶部风险']
        elif total_base >= thresholds['fully_bull'] and not notes_str:
            state['level'] = 'A'
            conclusion_parts = ['A: 综合多头']
        elif total_base >= thresholds['fully_bull']:
            state['level'] = 'A'
            conclusion_parts = ['A: 综合多头']
        elif total_base >= thresholds['bullish'] and has_top_div:
            state['level'] = 'B'
            conclusion_parts = ['B: 偏多(注意顶部风险)']
        elif total_base >= thresholds['bullish']:
            state['level'] = 'B'
            conclusion_parts = ['B: 偏多']
        elif total_base >= thresholds['oscillate']:
            state['level'] = 'C'
            conclusion_parts = ['C: 多空拉锯']
        else:
            state['level'] = 'C'
            conclusion_parts = ['C: 偏空']

        # 情景描述
        regime_labels = {
            'STRONG_TREND': '强势趋势', 'WEAK_TREND': '弱势趋势',
            'BOTTOM_REVERSAL': '底部反转', 'TOP_RISK': '顶部风险',
            'OSCILLATION': '震荡', 'UNCLEAR': '方向不明',
        }
        regime_label = regime_labels.get(regime, '方向不明')

        # 信号描述
        signal_desc = '+'.join(s['desc'] for s in signal_list[:2]) if signal_list else ''

        # 备注
        if notes_str:
            conclusion_parts.append(notes_str)
        if regime_label != '方向不明':
            conclusion_parts.insert(1, regime_label)
        if signal_desc:
            conclusion_parts.insert(2, signal_desc)

        state['conclusion'] = ' | '.join(filter(None, conclusion_parts))
        state['score'] = total
        state['risk_desc'] = state.get('risk_desc', '')

        # ── 写入 df.attrs 供外部使用 ─────────────────────────────────────
        df.attrs['pipeline_level'] = state['level']
        df.attrs['pipeline_conclusion'] = state['conclusion']
        df.attrs['pipeline_regime'] = state['regime']

        state['_scores'] = scores
        return _pipeline_output(state)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. 完全多头综合评分（核心入口，待弃用）
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_full_bull(
        self,
        df: pd.DataFrame,
        decay_half_life: int = 8,
        slope_window: int = 5,
        second_params: tuple[int, int, int] = (6, 13, 5),
        recalc_macd: bool = True,
        weights: dict[str, int] | None = None,
        thresholds: dict[str, int] | None = None,
    ) -> dict:
        """
        完全多头信号综合评估，输出 0~100/110 分和各子项状态。

        参数：
          df: 包含 OHLCV 数据的 DataFrame
          decay_half_life: 信号衰减半衰期
          slope_window: DIF斜率计算窗口
          second_params: 第二周期参数（必填），默认 (6, 13, 5)
          recalc_macd: 是否重新计算 MACD（若已在外部计算，可设为 False）
          weights: 各维度权重，默认 零轴20 / 战略金叉20 / 战术金叉15 / 动能20 / DIF斜率15 / 背离10 / 量价10
          thresholds: 结论阈值 {fully_bull, bullish, oscillate}
        """
        if weights is None:
            weights = {"零轴条件": 20, "战略金叉": 20, "战术金叉": 15, "动能": 20, "DIF斜率": 15, "背离信号": 10, "量价配合": 10}
        if thresholds is None:
            thresholds = {"fully_bull": 80, "bullish": 60, "oscillate": 40}
        # 计算MACD指标（若外部已计算可跳过，避免重复）
        if recalc_macd:
            df = self._custom_macd(df, second_params=second_params)
        else:
            required = {"DIF_12269", "DEA_12269", "MACD_12269", "MACD_12269_SIGNAL_DETAIL"}
            if not required.issubset(df.columns):
                df = self._custom_macd(df, second_params=second_params)

        # 确定第二周期名称
        fast, slow, signal = second_params
        second_period_name = f"{fast}{slow}{signal}"

        # 自适应 distance
        dist_slow = self._adaptive_distance(df, base=10)
        dist_fast = self._adaptive_distance(df, base=5)

        # 检测背离
        div_signals = self.detect_combined_divergence(
            df,
            distance_slow=dist_slow,
            distance_fast=dist_fast,
            decay_half_life=decay_half_life,
            second_period_name=second_period_name,
        )

        current_idx = len(df) - 1
        scores: dict[str, tuple[str, int]] = {}

        w_zero = weights["零轴条件"]
        w_strat = weights["战略金叉"]
        w_tact = weights["战术金叉"]
        w_mom = weights["动能"]
        w_slope = weights["DIF斜率"]
        w_div = weights["背离信号"]
        w_vol = weights["量价配合"]

        # ① 零轴条件 ──────────────────────────────────────────────────────
        dif_above = df["DIF_12269"].iloc[-1] > 0
        dea_above = df["DEA_12269"].iloc[-1] > 0
        if dif_above and dea_above:
            scores["零轴条件"] = ("DIF/DEA 均在零轴上", w_zero)
        elif dif_above:
            scores["零轴条件"] = ("DIF 在零轴上，DEA 仍在下方（金叉进行中）", w_zero // 2)
        else:
            scores["零轴条件"] = ("DIF 在零轴下", 0)

        # ② 战略线金叉状态 ─────────────────────────────────────────────────
        slow_detail = df["MACD_12269_SIGNAL_DETAIL"].iloc[-1]
        slow_bull = df["DIF_12269"].iloc[-1] > df["DEA_12269"].iloc[-1]
        if slow_detail == "零轴上金叉":
            scores["战略金叉"] = ("12269 零轴上金叉（最强信号）", w_strat)
        elif slow_detail == "零轴下金叉":
            scores["战略金叉"] = ("12269 零轴下金叉（注意假突破）", w_strat // 2)
        elif slow_bull:
            scores["战略金叉"] = ("12269 多头持续（DIF > DEA）", int(w_strat * 0.75))
        else:
            scores["战略金叉"] = ("12269 空头排列", 0)

        # ③ 战术线金叉状态 ─────────────────────────────────────────────────
        fast_detail_col = f"MACD_{second_period_name}_SIGNAL_DETAIL"
        fast_dif_col = f"DIF_{second_period_name}"
        fast_dea_col = f"DEA_{second_period_name}"

        fast_detail = df[fast_detail_col].iloc[-1]
        fast_bull = df[fast_dif_col].iloc[-1] > df[fast_dea_col].iloc[-1]
        if fast_detail == "零轴上金叉":
            scores["战术金叉"] = (f"{second_period_name} 零轴上金叉", w_tact)
        elif fast_bull:
            scores["战术金叉"] = (f"{second_period_name} 多头持续", int(w_tact * 0.65))
        else:
            scores["战术金叉"] = (f"{second_period_name} 空头/死叉", 0)

        # ④ 动能评分（连续标准化分数）─────────────────────────────────────
        mom_desc, mom_score = self._calculate_momentum_score(df, "DIF_12269", "DEA_12269", max_score=w_mom)
        prefix = "强势" if mom_score >= w_mom // 2 else ("关注" if mom_score >= w_mom // 4 else "弱势")
        scores["动能"] = (f"{prefix}: {mom_desc}", mom_score)

        # ⑤ DIF 斜率 ─────────────────────────────────────────────────────
        slope_info = self._slope_analysis(df["DIF_12269"], window=slope_window)
        if slope_info["trend"] == "明确上行":
            scores["DIF斜率"] = (f"确认 {slope_info['trend']} (R²={slope_info['r2']})", w_slope)
        elif "上行" in slope_info["trend"]:
            scores["DIF斜率"] = (f"关注 {slope_info['trend']} (R²={slope_info['r2']})", int(w_slope * 0.55))
        else:
            scores["DIF斜率"] = (f"弱势 {slope_info['trend']} (R²={slope_info['r2']})", 0)

        # ⑥ 背离信号 ─────────────────────────────────────────────────────
        cs = div_signals["combined_signal"]

        div_second_key = f"div_{second_period_name}"
        strength_second_key = f"strength_{second_period_name}"
        decay_second_key = f"decay_{second_period_name}"
        idx_second_key = f"idx_{second_period_name}"

        div_type = div_signals.get("div_12269") or div_signals.get(div_second_key)
        div_str = (
            div_signals.get("strength_12269")
            if div_signals.get("div_12269")
            else div_signals.get(strength_second_key, 0.0)
        )
        div_decay = (
            div_signals.get("decay_12269") if div_signals.get("div_12269") else div_signals.get(decay_second_key, 0.0)
        )
        eff = round(div_str * div_decay, 3)

        has_top_div = "顶背离" in cs or "卖出" in cs

        if "底背离" in cs:
            div_score = int(w_div * (0.5 + 0.5 * eff))
            scores["背离信号"] = (f"确认 {cs} (强度={div_str}, 衰减={div_decay}, 有效={eff})", div_score)
        elif has_top_div:
            scores["背离信号"] = (f"否定 {cs}（一票否决）", 0)
        else:
            scores["背离信号"] = ("中性: 无背离信号", 0)

        # ⑦ 量价配合（独立奖励）─────────────────────────────────────────────
        vol_desc, vol_bonus = self._volume_price_trend_score(df, max_bonus=w_vol)
        if has_top_div:
            vol_bonus = 0
            vol_desc = f"顶背离压制，跳过量价奖励"
        scores["量价配合"] = (vol_desc, vol_bonus)

        # ⑧ K线形态评分 ────────────────────────────────────────────────────
        w_kp = weights.get("K线形态", 10)
        kp_result = self._score_kline_pattern(df, max_score=w_kp)
        if has_top_div:
            kp_result = (kp_result[0], max(0, kp_result[1]))
        scores["K线形态"] = kp_result

        # ── 历史胜率（仅参考，不计入评分）────────────────────────────────
        winrate = self.backtest_signal_winrate(df, "MACD_12269_SIGNAL_DETAIL", "零轴上金叉", forward_bars=5)
        winrate_str = (
            f"样本 {winrate['sample_count']} 次，胜率 {winrate['win_rate']}，均收益 {winrate['avg_return']}"
            if winrate["sample_count"] > 0
            else "样本不足"
        )

        # ── 汇总：区分基础分与奖励分 ──────────────────────────────────────
        base_keys = [k for k in scores if k != "量价配合"]
        total_base = sum(scores[k][1] for k in base_keys)
        bonus = scores["量价配合"][1]

        # 顶背离一票否决：基础分最高为 背离权重 × 4
        if has_top_div:
            total_base = min(total_base, w_div * 4)

        total_max_base = w_zero + w_strat + w_tact + w_mom + w_slope + w_div + w_kp
        total_base = max(0, min(total_max_base, total_base))
        total = max(0, min(total_max_base + w_vol, total_base + bonus))

        full_bull = thresholds["fully_bull"]
        bullish = thresholds["bullish"]
        oscillate = thresholds["oscillate"]

        if total_base >= full_bull:
            conclusion = "完全多头 (强烈买入)"
        elif total_base >= bullish:
            conclusion = "偏多 (可逢低布局)"
        elif total_base >= oscillate:
            conclusion = "多空拉锯 (观望为主)"
        else:
            conclusion = "偏空 (回避或做空)"

        return {
            "score": total,
            "score_base": total_base,
            "conclusion": conclusion,
            "details": {k: {"desc": v[0], "score": v[1]} for k, v in scores.items()},
            "divergence": div_signals,
            "momentum": mom_desc,
            "slope": slope_info,
            "winrate_ref": winrate_str,
        }


# ── 级联流水线输出构建（在 pipeline_analysis 末尾调用） ──────────────────


def _pipeline_output(state: dict) -> dict:
    """
    从 pipeline state 构建统一返回 dict。

    保持与 analyze_full_bull 相同的顶层 key，以便 SignalManager 兼容处理。
    """
    notes = state.get('_notes', [])
    details = {}
    # 如果 state 中有 scores 则包含（取自 Gate 3 计算）
    if '_scores' in state:
        details = {k: {'desc': v[0], 'score': v[1]} for k, v in state['_scores'].items()}

    return {
        "score": state.get('score', 0),
        "score_base": state.get('score', 0),
        "conclusion": state.get('conclusion', ''),
        "level": state.get('level', 'C'),
        "regime": state.get('regime', 'UNCLEAR'),
        "risk_level": state.get('risk_level', 'NONE'),
        "risk_desc": state.get('risk_desc', ''),
        "triggered_rules": state.get('triggered_rules', []),
        "notes": notes,
        "signal_count": len(state.get('signal_list', [])),
        "details": details if details else {},
        "divergence": state.get('divergence', {}),
        "momentum": state.get('momentum', {}).get('desc', ''),
        "slope": state.get('slope', {}),
        "winrate_ref": '参考 pipeline',
    }
