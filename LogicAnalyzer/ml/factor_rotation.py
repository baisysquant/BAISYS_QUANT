from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr


class FactorRotationPlatform:
    """因子 IC 轮动平台 — IC regime 驱动的因子择时。

    对每个因子跟踪滚动 IC（20 日/60 日），检测 IC 动量方向，
    在 step 15 中生成权重 tilt，叠加到 FactorDecayMonitor 的衰减调整之上。

    用法（coordinator step 15）:
        rotation = FactorRotationPlatform()
        tilts = rotation.step(report, hist_df)   # 每日调用
        # tilts 返回 {因子名: 权重乘数}
        # 在 adjust_weight 之前或之后应用 tilts
    """

    _FACTOR_KEYS = [
        "macd", "momentum", "moneyflow", "quality", "valuation",
        "north_flow", "top_trader", "liquidity", "volatility",
        "macro", "financial_forward", "event_driven",
    ]

    # consolidated_report 中的中文列名 → 因子 key
    _COL_TO_KEY = {
        "MACD评分": "macd", "动量评分": "momentum", "资金流评分": "moneyflow",
        "基本面评分": "quality", "估值评分": "valuation",
        "北向资金评分": "north_flow", "龙虎榜评分": "top_trader",
        "流动性评分": "liquidity", "波动率评分": "volatility",
        "宏观评分": "macro", "财务前瞻评分": "financial_forward",
        "事件驱动评分": "event_driven",
    }

    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 60,
        top_pct: float = 0.25,
        bottom_pct: float = 0.25,
    ):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self._ic_history: dict[str, list[tuple[str, float]]] = {}

    # ── 公开接口 ─────────────────────────────────

    def step(
        self,
        report: pd.DataFrame,
        hist_df: pd.DataFrame,
        base_weights: dict[str, float],
        trade_date: str | None = None,
    ) -> dict[str, float]:
        """每日调用：计算 IC → 更新历史 → 生成权重 tilts → 返回调整后权重。

        Args:
            report: consolidated_report（含因子评分列）。
            hist_df: 全量历史 K 线。
            base_weights: 当前基础权重（FactorRegistry 权重）。
            trade_date: 当前交易日，None 时自动推断。

        Returns:
            调整后的权重 dict（归一化）。
        """
        if report.empty or hist_df.empty:
            return dict(base_weights)

        score_cols = [c for c in self._COL_TO_KEY if c in report.columns]
        if not score_cols:
            return dict(base_weights)

        fwd = self._compute_forward_ret(report, hist_df)
        if fwd is None:
            return dict(base_weights)

        if trade_date is None:
            trade_date = str(report["trade_date"].iloc[-1]) if "trade_date" in report.columns else ""

        for col in score_cols:
            key = self._COL_TO_KEY[col]
            scores = report[col]
            aligned = fwd.reindex_like(scores).fillna(0) if len(fwd) == len(scores) else fwd
            ic = self._daily_ic(scores, aligned)
            self._update_history(key, trade_date, ic)

        tilts = self._compute_tilts([k for k in self._FACTOR_KEYS if k in base_weights])
        adjusted = self._apply_tilts(base_weights, tilts)
        logger.info(f"[IC轮动] tilts: {dict(sorted(tilts.items()))}")
        return adjusted

    # ── IC 计算 ───────────────────────────────────

    def _daily_ic(self, factor_scores: pd.Series, forward_ret: pd.Series) -> float:
        valid = factor_scores.notna() & forward_ret.notna()
        if valid.sum() < 10:
            return 0.0
        rho, _ = spearmanr(factor_scores[valid], forward_ret[valid])
        return 0.0 if np.isnan(rho) else float(rho)

    def _compute_forward_ret(
        self, report: pd.DataFrame, hist_df: pd.DataFrame,
    ) -> pd.Series | None:
        if "股票代码" not in report.columns or "trade_date" not in hist_df.columns:
            return None
        symbols = report["股票代码"].unique().tolist()
        sub = hist_df[hist_df["symbol"].isin(symbols)].copy()
        if sub.empty:
            return None
        sub = sub.sort_values(["symbol", "trade_date"])
        fwd = sub.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1)
        last = sub.groupby("symbol").last().reset_index()
        fwd_last = last.merge(
            sub[["symbol", "trade_date"]].groupby("symbol").last().reset_index(),
            on="symbol", how="left",
        )
        _map = sub.drop_duplicates(subset="symbol").set_index("symbol")["close"].index
        result = report["股票代码"].map(
            sub.groupby("symbol").last()["close"].pipe(lambda s: s.shift(-5) / s - 1).to_dict()
        ).fillna(0)
        return result

    # ── 历史维护 ───────────────────────────────────

    def _update_history(self, factor: str, date: str, ic: float) -> None:
        self._ic_history.setdefault(factor, []).append((date, ic))
        self._ic_history[factor] = self._ic_history[factor][-120:]

    def _rolling_ic(self, factor: str, window: int) -> float:
        entries = self._ic_history.get(factor, [])
        if len(entries) < window:
            return 0.0
        return float(np.mean([ic for _, ic in entries[-window:]]))

    # ── 权重 tilt ──────────────────────────────────

    def _compute_tilts(self, factors: list[str]) -> dict[str, float]:
        short_ics = {f: self._rolling_ic(f, self.lookback_short) for f in factors}
        long_ics = {f: self._rolling_ic(f, self.lookback_long) for f in factors}

        ic_vals = np.array([v for v in short_ics.values() if not (np.isnan(v) or v == 0)])
        high_th = float(np.percentile(ic_vals, 75)) if len(ic_vals) > 1 else 0.03
        low_th = float(np.percentile(ic_vals, 25)) if len(ic_vals) > 1 else 0.0

        tilts: dict[str, float] = {}
        for f in factors:
            s = short_ics.get(f, 0)
            l = long_ics.get(f, 0)
            tilt = 1.0
            if s >= high_th and s > l:
                tilt = 1.4
            elif s >= high_th:
                tilt = 1.2
            elif s <= low_th and s < l:
                tilt = 0.6
            elif s <= low_th:
                tilt = 0.8
            tilts[f] = tilt
        return tilts

    @staticmethod
    def _apply_tilts(
        base_weights: dict[str, float], tilts: dict[str, float],
    ) -> dict[str, float]:
        adjusted = {k: v * tilts.get(k, 1.0) for k, v in base_weights.items()}
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted
