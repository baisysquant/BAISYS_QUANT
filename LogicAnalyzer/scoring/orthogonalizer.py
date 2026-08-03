"""
因子正交化 + IC 衰减分析 — 因子信号处理 Pipeline

将「固定权重加权」升级为完整因子信号处理流水线：
  1. 冗余剔除     —— 相关系数 > 0.8 的因子对中剔除 IC 较低者
  2. 对称正交化   —— Cholesky 分解，消除共线性，识别纯净 alpha
  3. IC 衰减分析  —— 滚动 12 个月 Rank IC，按滞后 1/5/10/20 天输出衰减曲线
  4. 半衰期分类   —— < 5 天 → 短线因子，> 15 天 → 长线因子，< 3 天 → 噪音剔除
  5. 自动权重分配 —— IR-Weighted（wᵢ = ICIRᵢ / Σ|ICIRⱼ|），短线/长线分别配权

数学（对称正交化）：
  因子暴露矩阵 X (N×K) 按列标准化得 Xs，协方差矩阵 C = XsᵀXs / N = L·Lᵀ（Cholesky），
  正交化因子 Z = Xs·L⁻ᵀ，满足 ZᵀZ = N·I —— 新因子间相关系数矩阵为单位阵。
  即 X = Z·Lᵀ，X 的冗余信息被旋转剥离，保留互不相关的纯净 alpha。

用法:
    orth = FactorOrthogonalizer()
    result = orth.run(panel, kline, factor_cols)
    # result["orthogonalized"]       → 正交因子矩阵、旋转矩阵、正交前后相关矩阵
    # result["ic_analysis"]          → 衰减曲线（滞后 1/5/10/20）、半衰期、ICIR
    # result["classification"]       → {因子: short/mid/long/noise}
    # result["weights"]              → {"short": ..., "long": ..., "all": ...}
    # result["report"]               → 逐因子分析报告 DataFrame
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

DEFAULT_LAGS: tuple[int, ...] = (1, 5, 10, 20)


@dataclass
class OrthogonalizationResult:
    """对称正交化结果。"""

    X_orth: pd.DataFrame  # N×K 正交化后的因子矩阵（列间相关系数矩阵 = 单位阵）
    rotation_matrix: np.ndarray  # L⁻ᵀ：原始标准化因子 → 正交因子 的旋转矩阵
    cholesky_L: np.ndarray  # 协方差矩阵的 Cholesky 下三角
    corr_before: pd.DataFrame  # 正交前的相关矩阵
    corr_after: pd.DataFrame  # 正交后的相关矩阵（≈ 单位阵）


class FactorOrthogonalizer:
    """因子正交化 + IC 衰减分析 + 自动权重分配 Pipeline。"""

    MIN_CROSS_SECTION = 10  # 截面最少样本数，低于此不计算 Rank IC

    # ── 对称正交化 ────────────────────────────────────────────

    def orthogonalize(self, X: pd.DataFrame) -> OrthogonalizationResult:
        """对单期因子暴露矩阵 X (N×K) 做对称正交化。

        X 按列标准化后，对协方差矩阵 C = XsᵀXs / N 做 Cholesky 分解 C = L·Lᵀ，
        正交因子 X_orth = Xs·L⁻ᵀ，其列间相关系数矩阵为单位阵（即 X = Z·Lᵀ）。

        Args:
            X: N×K 因子暴露矩阵，行 = 股票，列 = 因子。

        Returns:
            OrthogonalizationResult: 正交因子矩阵与旋转信息。

        Raises:
            ValueError: 有效因子不足或协方差矩阵不可逆。
        """
        Xs = X.astype(float)
        stds = Xs.std(axis=0)
        dead = stds[stds < 1e-12]
        if not dead.empty:
            logger.warning(f"[正交化] 剔除常量因子: {list(dead.index)}")
            Xs = Xs.drop(columns=dead.index.tolist())
        if Xs.shape[1] < 2:
            raise ValueError("有效因子不足（至少需要 2 个非常量因子）")

        Xs = (Xs - Xs.mean(axis=0)) / Xs.std(axis=0)
        L, LinvT = self._cholesky_rotation(Xs)
        Z = Xs.to_numpy() @ LinvT
        X_orth = pd.DataFrame(Z, index=Xs.index, columns=Xs.columns)

        return OrthogonalizationResult(
            X_orth=X_orth,
            rotation_matrix=LinvT,
            cholesky_L=L,
            corr_before=Xs.corr(),
            corr_after=pd.DataFrame(
                np.corrcoef(Z, rowvar=False),
                index=Xs.columns,
                columns=Xs.columns,
            ),
        )

    @staticmethod
    def _cholesky_rotation(
        Xs: pd.DataFrame, jitter: float = 1e-8
    ) -> tuple[np.ndarray, np.ndarray]:
        """对标准化因子矩阵 Xs 求 Cholesky 旋转。

        C = XsᵀXs / N = L·Lᵀ（Cholesky），返回 (L, L⁻ᵀ)，
        使 Z = Xs·L⁻ᵀ 满足 ZᵀZ = N·I（列间正交）。

        Raises:
            ValueError: 样本数小于因子数或协方差矩阵不可逆。
        """
        n, k = Xs.shape
        if n < k:
            raise ValueError(f"样本数 {n} < 因子数 {k}，协方差矩阵不满秩，无法正交化")
        C = Xs.to_numpy().T @ Xs.to_numpy() / n
        j = float(jitter)
        while True:
            try:
                L = np.linalg.cholesky(C + np.eye(k) * j)
                break
            except np.linalg.LinAlgError:
                j *= 10.0
                if j > 1e-2:
                    raise ValueError(
                        "因子协方差矩阵不可逆（因子高度共线或样本不足），正交化失败"
                    )
        LinvT = np.linalg.solve(L.T, np.eye(k))
        return L, LinvT

    # ── Rank IC ──────────────────────────────────────────────

    @staticmethod
    def _rank_ic(a: pd.Series, b: pd.Series, min_obs: int = MIN_CROSS_SECTION) -> float:
        """Spearman 秩相关系数（截面 Rank IC），样本不足或常量时返回 0。"""
        valid = a.notna() & b.notna()
        if valid.sum() < min_obs:
            return 0.0
        ra = a[valid]
        rb = b[valid]
        if ra.nunique() < 2 or rb.nunique() < 2:
            return 0.0
        rho, _ = spearmanr(ra, rb)
        if rho is None:
            return 0.0
        rho_f = float(rho)
        return 0.0 if math.isnan(rho_f) else rho_f

    # ── 前向收益与 IC 系列 ────────────────────────────────────

    @staticmethod
    def _date_key(s: pd.Series) -> pd.Series:
        """统一日期格式为 'YYYY-MM-DD' 字符串（非法日期 → NaN）。"""
        dt = pd.to_datetime(s, errors="coerce")
        return dt.dt.strftime("%Y-%m-%d").where(dt.notna())

    def _precompute_forward_returns(
        self,
        kline: pd.DataFrame,
        lags: Sequence[int],
        symbols_col: str,
        date_col: str,
    ) -> dict[int, pd.Series]:
        """预计算各滞后天数 h 的前向收益率，index 为 (symbol, date)。"""
        k = kline.copy()
        k["_date"] = self._date_key(k[date_col])
        k["_close"] = pd.to_numeric(k["close"], errors="coerce")
        k = k.sort_values([symbols_col, "_date"])
        idx = pd.MultiIndex.from_arrays(
            [k[symbols_col], k["_date"]], names=[symbols_col, date_col]
        )
        out: dict[int, pd.Series] = {}
        for h in lags:
            fwd = k.groupby(symbols_col)["_close"].shift(-int(h))
            ret = (fwd / k["_close"] - 1).replace([np.inf, -np.inf], np.nan)
            out[int(h)] = pd.Series(ret.to_numpy(), index=idx)
        return out

    @staticmethod
    def _forward_at_date(
        fwd_returns: dict[int, pd.Series],
        horizon: int,
        date_key: str,
        date_col: str,
    ) -> pd.Series | None:
        """取某日前向收益截面，日期缺失时返回 None。"""
        try:
            return fwd_returns[int(horizon)].xs(date_key, level=date_col)
        except KeyError:
            return None

    def _collect_ic_series(
        self,
        frames: dict[str, pd.DataFrame],
        fwd_returns: dict[int, pd.Series],
        factors: Sequence[str],
        lags: Sequence[int],
        date_col: str,
    ) -> dict[str, dict[int, list[float]]]:
        """逐日计算每个因子在每个滞后下的 Rank IC 序列。"""
        series: dict[str, dict[int, list[float]]] = {
            f: {int(h): [] for h in lags} for f in factors
        }
        for dkey, frame in frames.items():
            for h in lags:
                fwd_t = self._forward_at_date(fwd_returns, h, dkey, date_col)
                if fwd_t is None:
                    continue
                common = frame.index.intersection(fwd_t.index)
                if len(common) < self.MIN_CROSS_SECTION:
                    continue
                fwd_aligned = fwd_t.reindex(common)
                for f in factors:
                    ic = self._rank_ic(frame[f].reindex(common), fwd_aligned)
                    series[f][int(h)].append(ic)
        return series

    # ── IC 衰减分析 ───────────────────────────────────────────

    @staticmethod
    def estimate_half_life(
        ic_curve: pd.Series, lags: Sequence[int] = DEFAULT_LAGS
    ) -> float:
        """从 IC 衰减曲线估计信号半衰期（天）。

        以最短滞后（lag=1）的 |IC| 为峰值，拟合 log₂(|IC_lag| / |IC₁|) = -lag / half_life，
        回归斜率 slope = -1/half_life ⇒ half_life = -1/slope。

        Returns:
            半衰期（天）：不衰减（slope ≥ 0）返回 +inf；lag=1 处 |IC|≈0（无信号）返回 0。
        """
        lags_arr = np.asarray([int(lag) for lag in lags], dtype=float)
        vals = np.asarray(
            [ic_curve.get(int(lag), np.nan) for lag in lags], dtype=float
        )
        valid = np.isfinite(vals) & (lags_arr > 0)
        if valid.sum() < 2:
            return math.inf
        lag_v = lags_arr[valid]
        abs_v = np.abs(vals[valid])
        if abs_v[0] < 1e-6:
            return 0.0
        y = np.log2(np.clip(abs_v / abs_v[0], 1e-8, None))
        slope = float(np.polyfit(lag_v, y, 1)[0])
        if not math.isfinite(slope) or slope >= 0:
            return math.inf
        return float(-1.0 / slope)

    @staticmethod
    def classify_horizon(
        half_life: float,
        noise_half_life: float = 3.0,
        short_threshold: float = 5.0,
        long_threshold: float = 15.0,
    ) -> str:
        """按衰减半衰期分类因子：noise(<3 天) / short(<5 天) / mid / long(>15 天)。

        无法观测到衰减（inf/NaN）视为长线因子。
        """
        if half_life < noise_half_life:
            return "noise"
        if not math.isfinite(half_life) or half_life > long_threshold:
            return "long"
        if half_life < short_threshold:
            return "short"
        return "mid"

    @staticmethod
    def decay_curve_ascii(decay_curves: pd.DataFrame) -> str:
        """以文本条形图输出 IC 衰减曲线（无 matplotlib 环境也可用）。"""
        cols = list(decay_curves.columns)
        lines = [
            f"IC 衰减曲线（滞后 {', '.join(c.replace('lag', '') for c in cols)} 天）"
        ]
        lines.append("-" * 100)
        for f, row in decay_curves.iterrows():
            bars = []
            for c in cols:
                v = float(row[c])
                width = int(min(abs(v) * 200, 25))
                bars.append("#" * width + f"{v:+.4f}")
            lines.append(f"{str(f):<18} " + "  ".join(bars))
        return "\n".join(lines)

    @staticmethod
    def plot_decay_curves(
        decay_curves: pd.DataFrame, save_path: str | None = None
    ) -> Any:
        """绘制 IC 衰减曲线（滞后 1/5/10/20 天）。

        matplotlib 未安装时自动跳过并返回 None。
        """
        try:
            import matplotlib  # type: ignore[import-not-found]

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("[IC衰减] matplotlib 未安装，跳过绘图")
            return None

        x = [int(c.replace("lag", "")) for c in decay_curves.columns]
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, row in decay_curves.iterrows():
            ax.plot(x, row.values, marker="o", label=str(idx))
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("滞后天数")
        ax.set_ylabel("Rank IC")
        ax.set_title("正交因子 IC 衰减曲线")
        ax.legend()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── 冗余剔除 ──────────────────────────────────────────────

    @staticmethod
    def prune_by_correlation(
        corr: pd.DataFrame,
        ic_scores: dict[str, float],
        threshold: float = 0.8,
    ) -> list[str]:
        """剔除相关系数 > threshold 的因子对中 IC 较低的因子。

        贪心策略：按 |相关系数| 降序处理所有超标因子对，
        对仍未被剔除的对，剔除 IC 较低者（IC 相同时剔除前者）。
        """
        cols = list(corr.columns)
        pairs: list[tuple[float, str, str]] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iloc[i, j]
                if pd.notna(c) and abs(float(c)) > threshold:
                    pairs.append((abs(float(c)), str(cols[i]), str(cols[j])))
        pairs.sort(key=lambda p: p[0], reverse=True)

        dropped: set[str] = set()
        for _, a, b in pairs:
            if a in dropped or b in dropped:
                continue
            ic_a = float(ic_scores.get(a, 0.0))
            ic_b = float(ic_scores.get(b, 0.0))
            dropped.add(a if ic_a <= ic_b else b)
        return sorted(dropped)

    # ── 自动权重分配 ──────────────────────────────────────────

    @staticmethod
    def ir_weights(
        icir: dict[str, float], allow_negative: bool = False
    ) -> dict[str, float]:
        """IR-Weighted 权重分配：wᵢ = ICIRᵢ / Σ|ICIRⱼ|。

        默认将负 ICIR 因子权重置 0（反向信号不可靠）后重新归一化；
        allow_negative=True 时按原始公式返回（负权重 = 反向信号）。
        """
        if not icir:
            return {}
        total = sum(abs(v) for v in icir.values())
        if total <= 0:
            return {}
        raw = {k: v / total for k, v in icir.items()}
        if allow_negative:
            return raw
        positive = {k: max(0.0, v) for k, v in raw.items()}
        s = sum(positive.values())
        if s <= 0:
            return {}
        return {k: v / s for k, v in positive.items()}

    # ── 主流程 ────────────────────────────────────────────────

    def run(
        self,
        panel: pd.DataFrame,
        kline: pd.DataFrame,
        factor_cols: Sequence[str],
        symbols_col: str = "symbol",
        date_col: str = "trade_date",
        lags: Sequence[int] = DEFAULT_LAGS,
        window_months: int = 12,
        corr_threshold: float = 0.8,
        noise_half_life: float = 3.0,
        short_threshold: float = 5.0,
        long_threshold: float = 15.0,
        short_horizon: int = 5,
        long_horizon: int = 20,
    ) -> dict[str, Any]:
        """完整因子信号处理 Pipeline。

        Args:
            panel: 长表因子面板，含 symbols_col / date_col 与因子列（逐日逐股一行）。
            kline: K 线长表，需含 symbol, trade_date, close 列。
            factor_cols: 因子列名列表（≥2 个）。
            symbols_col: 股票代码列名。
            date_col: 交易日列名。
            lags: IC 衰减分析的滞后天数序列。
            window_months: 滚动分析窗口（月），默认 12 个月。
            corr_threshold: 冗余剔除的相关性阈值。
            noise_half_life: 噪音剔除的半衰期阈值（天）。
            short_threshold / long_threshold: 短/长线分类阈值（天）。
            short_horizon / long_horizon: 短线/长线配权所用滞后（须在 lags 内）。

        Returns:
            dict: 见模块 docstring；数据不足时返回 {"error": ...}。
        """
        result: dict[str, Any] = {"timestamp": datetime.now().isoformat()}

        if panel is None or panel.empty or kline is None or kline.empty:
            result["error"] = "输入数据为空"
            return result
        if int(short_horizon) not in [int(h) for h in lags] or int(long_horizon) not in [
            int(h) for h in lags
        ]:
            result["error"] = "short_horizon / long_horizon 必须在 lags 中"
            return result

        panel = panel.copy()
        factor_cols_list = [c for c in factor_cols if c in panel.columns]
        if len(factor_cols_list) < 2:
            result["error"] = "面板中可用因子列不足 2 个"
            return result

        # 1. 滚动窗口（默认 12 个月）
        panel["_date_key"] = self._date_key(panel[date_col])
        panel = panel[panel["_date_key"].notna()]
        if panel.empty:
            result["error"] = "面板无有效日期"
            return result
        all_dates = pd.to_datetime(panel["_date_key"].unique()).sort_values()
        if len(all_dates) > 1:
            start = all_dates[-1] - pd.DateOffset(months=window_months)
            all_dates = all_dates[all_dates >= start]
        date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]

        # 2. 逐日截面因子矩阵
        frames: dict[str, pd.DataFrame] = {}
        for dkey in date_strs:
            sub = panel[panel["_date_key"] == dkey]
            sub = sub.drop_duplicates(subset=[symbols_col])
            frame = sub.set_index(symbols_col)[factor_cols_list].apply(
                pd.to_numeric, errors="coerce"
            )
            if len(frame) >= self.MIN_CROSS_SECTION:
                frames[dkey] = frame
        if len(frames) < 2:
            result["error"] = "滚动窗口内有效截面不足"
            return result

        # 3. 前向收益（滞后 1/5/10/20 天）
        fwd_returns = self._precompute_forward_returns(
            kline, lags, symbols_col, date_col
        )

        # 4. 原始因子 IC 与相关矩阵（冗余剔除用）
        pool = pd.concat(list(frames.values()), axis=0)
        stds = pool.std()
        valid = [
            f
            for f in factor_cols_list
            if pd.notna(stds.get(f, np.nan)) and stds[f] > 1e-12
        ]
        if len(valid) < 2:
            result["error"] = "非常量因子不足 2 个"
            return result
        series_orig = self._collect_ic_series(
            frames, fwd_returns, valid, lags, date_col
        )
        ic_scores = {
            f: float(
                np.mean([abs(ic) for h in lags for ic in series_orig[f][int(h)]])
            )
            for f in valid
        }
        corr_orig = pool[valid].corr()
        dropped_high_corr = self.prune_by_correlation(
            corr_orig, ic_scores, corr_threshold
        )
        kept = [f for f in valid if f not in dropped_high_corr]
        if len(kept) < 2:
            result.update(
                {
                    "error": "冗余剔除后因子不足 2 个",
                    "pruned": {"high_corr": dropped_high_corr, "noise": []},
                }
            )
            return result

        # 5. 对称正交化（池化协方差定旋转，逐日应用）
        pool_kept = pool[kept]
        means = pool_kept.mean(axis=0)
        stds_kept = pool_kept.std(axis=0)
        pooled_std = (pool_kept - means) / stds_kept
        L, LinvT = self._cholesky_rotation(pooled_std)
        frames_orth: dict[str, pd.DataFrame] = {}
        for dkey, frame in frames.items():
            z = (frame[kept] - means) / stds_kept
            frames_orth[dkey] = pd.DataFrame(
                z.to_numpy() @ LinvT, index=z.index, columns=kept
            )
        orth_pool = pd.concat(list(frames_orth.values()), axis=0)
        corr_after = orth_pool.corr()

        # 6. 正交因子 IC 衰减分析
        series_orth = self._collect_ic_series(
            frames_orth, fwd_returns, kept, lags, date_col
        )
        decay_curves = pd.DataFrame(
            0.0, index=kept, columns=[f"lag{h}" for h in lags]
        )
        icir: dict[str, dict[int, float]] = {f: {} for f in kept}
        half_lives: dict[str, float] = {}
        for f in kept:
            for h in lags:
                ics = series_orth[f][int(h)]
                ic_mean = float(np.mean(ics)) if ics else 0.0
                ic_std = float(np.std(ics)) if len(ics) > 1 else 0.0
                decay_curves.loc[f, f"lag{h}"] = ic_mean
                icir[f][int(h)] = ic_mean / ic_std if ic_std > 1e-12 else 0.0
            half_lives[f] = self.estimate_half_life(decay_curves.loc[f], lags)

        # 7. 噪音剔除 + 短/长线分类
        classifications = {
            f: self.classify_horizon(
                half_lives[f], noise_half_life, short_threshold, long_threshold
            )
            for f in kept
        }
        noise_factors = [
            f for f, c in classifications.items() if c == "noise"
        ]
        final_factors = [f for f in kept if classifications[f] != "noise"]

        # 8. IR-Weighted 权重：短线因子 → 短周期预测，长线因子 → 长周期预测
        def _weight_for(factors: list[str], horizon: int) -> dict[str, float]:
            return self.ir_weights({f: icir[f].get(int(horizon), 0.0) for f in factors})

        short_used = [f for f in final_factors if classifications[f] in ("short", "mid")]
        long_used = [f for f in final_factors if classifications[f] in ("long", "mid")]
        w_short = _weight_for(short_used, short_horizon)
        w_long = _weight_for(long_used, long_horizon)
        w_all = self.ir_weights(
            {f: float(np.mean(list(icir[f].values()))) for f in final_factors}
        )

        # 9. 逐因子报告
        rows = [
            {
                "因子": f,
                "半衰期(天)": (
                    half_lives[f] if math.isfinite(half_lives[f]) else "∞"
                ),
                "分类": classifications[f],
                f"ICIR({short_horizon}日)": round(
                    icir[f].get(int(short_horizon), 0.0), 4
                ),
                f"ICIR({long_horizon}日)": round(
                    icir[f].get(int(long_horizon), 0.0), 4
                ),
                "平均|IC|": round(float(decay_curves.loc[f].abs().mean()), 4),
                "权重(短线)": round(w_short.get(f, 0.0), 4),
                "权重(长线)": round(w_long.get(f, 0.0), 4),
            }
            for f in kept
        ]
        report = pd.DataFrame(rows)

        result.update(
            {
                "window_dates": date_strs,
                "n_cross_sections": len(frames),
                "pruned": {"high_corr": dropped_high_corr, "noise": noise_factors},
                "kept_factors": final_factors,
                "orthogonalized": {
                    "rotation_matrix": LinvT,
                    "cholesky_L": L,
                    "corr_before": corr_orig,
                    "corr_after": corr_after,
                    "X_orth_latest": frames_orth[list(frames_orth)[-1]],
                },
                "ic_analysis": {
                    "decay_curves": decay_curves,
                    "half_lives": half_lives,
                    "icir": icir,
                    "curves_text": self.decay_curve_ascii(decay_curves),
                },
                "classification": classifications,
                "weights": {"short": w_short, "long": w_long, "all": w_all},
                "report": report,
            }
        )

        logger.info(f"[正交化] 冗余剔除(相关>{corr_threshold}): {dropped_high_corr or '无'}")
        logger.info(
            f"[IC衰减] 噪音剔除(半衰期<{noise_half_life}天): {noise_factors or '无'}"
        )
        logger.info(f"[因子分类] {classifications}")
        logger.info(f"[权重] 短线: {w_short} | 长线: {w_long}")
        return result
