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

        reason = f"{'✅' if confirmed else '❌'} 信号处量能{tag}，量比均值 {vol_ratio:.2f}x"
        return {"confirmed": confirmed, "vol_ratio": round(vol_ratio, 2), "reason": reason}

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
            combined = "战略底背离 + 战术金叉确认 ✅ (精准买入)"
        elif top_12269 and top_second and fast_dead:
            combined = "战略顶背离 + 战术死叉确认 ❌ (精准卖出)"
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
    # 8. 完全多头综合评分（核心入口）
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_full_bull(
        self,
        df: pd.DataFrame,
        decay_half_life: int = 8,
        slope_window: int = 5,
        second_params: tuple[int, int, int] = (6, 13, 5),
    ) -> dict:
        """
        完全多头信号综合评估，输出 0~100 分和各子项状态。

        参数：
          df: 包含 OHLCV 数据的 DataFrame
          decay_half_life: 信号衰减半衰期
          slope_window: DIF斜率计算窗口
          second_params: 第二周期参数（必填），默认 (6, 13, 5)

        评分维度（满分 100，背离项可扣分）：
          ① 零轴条件        25 分  (12269 DIF>0 且 DEA>0)
          ② 战略线金叉状态  20 分  (12269 MACD 排列)
          ③ 战术线金叉状态  15 分  (第二周期 MACD 排列)
          ④ 动能加速        15 分  (红柱加长)
          ⑤ DIF 斜率        10 分  (线性回归趋势)
          ⑥ 背离信号        25 分  (底背离加分，顶背离扣 -10 分)
          ⑦ 量价配合        10 分  (奖励项，不计入基准 100)

        综合结论：
          ≥ 80  → 完全多头（强烈买入）
          ≥ 60  → 偏多（逢低布局）
          ≥ 40  → 多空拉锯（观望）
          < 40  → 偏空（回避或做空）
        """
        # 计算MACD指标
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

        # ① 零轴条件 ──────────────────────────────────────────────────────
        dif_above = df["DIF_12269"].iloc[-1] > 0
        dea_above = df["DEA_12269"].iloc[-1] > 0
        if dif_above and dea_above:
            scores["零轴条件"] = ("✅ DIF / DEA 均在零轴上", 25)
        elif dif_above:
            scores["零轴条件"] = ("⚠️ DIF 在零轴上，DEA 仍在下方（金叉进行中）", 12)
        else:
            scores["零轴条件"] = ("❌ DIF 在零轴下", 0)

        # ② 战略线金叉状态 ─────────────────────────────────────────────────
        slow_detail = df["MACD_12269_SIGNAL_DETAIL"].iloc[-1]
        slow_bull = df["DIF_12269"].iloc[-1] > df["DEA_12269"].iloc[-1]
        if slow_detail == "零轴上金叉":
            scores["战略金叉"] = ("✅ 12269 零轴上金叉（最强信号）", 20)
        elif slow_detail == "零轴下金叉":
            scores["战略金叉"] = ("⚠️ 12269 零轴下金叉（注意假突破）", 10)
        elif slow_bull:
            scores["战略金叉"] = ("✅ 12269 多头持续（DIF > DEA）", 15)
        else:
            scores["战略金叉"] = ("❌ 12269 空头排列", 0)

        # ③ 战术线金叉状态 ─────────────────────────────────────────────────
        fast_detail_col = f"MACD_{second_period_name}_SIGNAL_DETAIL"
        fast_dif_col = f"DIF_{second_period_name}"
        fast_dea_col = f"DEA_{second_period_name}"

        fast_detail = df[fast_detail_col].iloc[-1]
        fast_bull = df[fast_dif_col].iloc[-1] > df[fast_dea_col].iloc[-1]
        if fast_detail == "零轴上金叉":
            scores["战术金叉"] = (f"✅ {second_period_name} 零轴上金叉", 15)
        elif fast_bull:
            scores["战术金叉"] = (f"✅ {second_period_name} 多头持续", 10)
        else:
            scores["战术金叉"] = (f"❌ {second_period_name} 空头 / 死叉", 0)

        # ④ 动能状态 ───────────────────────────────────────────────────────
        momentum = self._calculate_macd_momentum(df, "DIF_12269", "DEA_12269")
        if "加速上涨" in momentum:
            scores["动能"] = (f"✅ {momentum}", 15)
        elif "减速上涨" in momentum:
            scores["动能"] = (f"⚠️ {momentum}", 8)
        else:
            scores["动能"] = (f"❌ {momentum}", 0)

        # ⑤ DIF 斜率 ───────────────────────────────────────────────────────
        slope_info = self._slope_analysis(df["DIF_12269"], window=slope_window)
        if slope_info["trend"] == "明确上行":
            scores["DIF斜率"] = (f"✅ {slope_info['trend']} (R²={slope_info['r2']})", 10)
        elif "上行" in slope_info["trend"]:
            scores["DIF斜率"] = (f"⚠️ {slope_info['trend']} (R²={slope_info['r2']})", 5)
        else:
            scores["DIF斜率"] = (f"❌ {slope_info['trend']} (R²={slope_info['r2']})", 0)

        # ⑥ 背离信号（含强度 × 衰减加权）─────────────────────────────────
        cs = div_signals["combined_signal"]

        # 动态获取第二周期的键名
        div_second_key = f"div_{second_period_name}"
        strength_second_key = f"strength_{second_period_name}"
        decay_second_key = f"decay_{second_period_name}"
        idx_second_key = f"idx_{second_period_name}"

        div_type = div_signals.get("div_12269") or div_signals.get(div_second_key)
        div_idx = (
            div_signals.get("idx_12269")
            if div_signals.get("idx_12269") is not None
            else div_signals.get(idx_second_key)
        )
        div_str = (
            div_signals.get("strength_12269")
            if div_signals.get("div_12269")
            else div_signals.get(strength_second_key, 0.0)
        )
        div_decay = (
            div_signals.get("decay_12269") if div_signals.get("div_12269") else div_signals.get(decay_second_key, 0.0)
        )
        eff = round(div_str * div_decay, 3)

        if "底背离" in cs:
            raw_score = 25
            div_score = int(raw_score * (0.5 + 0.5 * eff))  # 强度加权：最少 50% 分
            scores["背离信号"] = (f"✅ {cs}  (强度={div_str}, 衰减={div_decay}, 有效={eff})", div_score)
        elif "顶背离" in cs or "卖出" in cs:
            scores["背离信号"] = (f"❌ {cs}（扣分）", -10)
        else:
            scores["背离信号"] = ("➖ 无背离信号", 0)

        # ⑦ 量价配合（奖励项）────────────────────────────────────────────
        if div_idx is not None and div_type:
            vol_check = self._volume_confirmation(df, div_type, div_idx)
            bonus = 10 if vol_check["confirmed"] else 0
            scores["量价配合"] = (vol_check["reason"], bonus)
        else:
            scores["量价配合"] = ("➖ 无背离锚点，跳过量价验证", 0)

        # ── 历史胜率（仅参考，不计入评分）────────────────────────────────
        winrate = self.backtest_signal_winrate(df, "MACD_12269_SIGNAL_DETAIL", "零轴上金叉", forward_bars=5)
        winrate_str = (
            f"样本 {winrate['sample_count']} 次，胜率 {winrate['win_rate']}，均收益 {winrate['avg_return']}"
            if winrate["sample_count"] > 0
            else "样本不足"
        )

        # ── 汇总 ──────────────────────────────────────────────────────────
        total = sum(v for _, v in scores.values())
        total = max(0, min(100, total))

        if total >= 80:
            conclusion = "🚀 完全多头 (强烈买入)"
        elif total >= 60:
            conclusion = "📈 偏多 (可逢低布局)"
        elif total >= 40:
            conclusion = "⚖️ 多空拉锯 (观望为主)"
        else:
            conclusion = "📉 偏空 (回避或做空)"

        return {
            "score": total,
            "conclusion": conclusion,
            "details": {k: {"desc": v[0], "score": v[1]} for k, v in scores.items()},
            "divergence": div_signals,
            "momentum": momentum,
            "slope": slope_info,
            "winrate_ref": winrate_str,
        }
