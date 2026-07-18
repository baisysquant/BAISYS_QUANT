"""
因子衰减监控器

计算各因子 IC（信息系数）的滚动统计量，判断因子是否失效，
并在衰减时建议自动降权。

指标：
  - Rank IC：Spearman 秩相关系数（因子值与未来 N 日收益率）
  - Normal IC：Pearson 相关系数
  - ICIR：IC 均值 / IC 标准差（衡量因子预测稳定性）
  - 分层回测：十档分组测试，检查单调性和多空收益差
  - 衰减判定：滚动 IC 均值 < 0 且持续 D 天 → 衰减
  - 降权公式：新权重 = 原权重 × max(0, 滚动 IC 均值 / 初始 IC)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text as sql_text



class FactorDecayMonitor:
    """多因子衰减监控与自动降权。

    用法:
        monitor = FactorDecayMonitor(config, db_engine)
        status = monitor.run(consolidated_report, hist_df)
        # status 包含各因子的 IC、ICIR、分层回测、衰减状态和建议权重
    """

    TABLE_NAME = "ods_factor_ic_history"

    FACTOR_COLUMNS = {
        "macd": "MACD评分",
        "momentum": "动量评分",
        "moneyflow": "资金流评分",
        "quality": "基本面评分",
        "valuation": "估值评分",
    }

    def __init__(self, config: Any, db_engine: Any) -> None:
        self.config = config
        self._engine = db_engine
        from LogicAnalyzer.scoring.factor_registry import FactorRegistry
        config_dir = getattr(config, "CONFIG_DIR", None) or "config"
        self._registry = FactorRegistry(os.path.join(config_dir, "factor_registry.yaml"))
        self._weights: dict[str, float] = self._registry.weights
        self._ensure_table()

    # ── 表结构 ────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        if self._engine is None:
            return
        ddl = f"""
        CREATE TABLE IF NOT EXISTS public.{self.TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            factor_name VARCHAR(20) NOT NULL,
            check_date DATE NOT NULL,
            rolling_ic_mean FLOAT,
            rolling_ic_std FLOAT,
            icir FLOAT,
            rank_ic FLOAT,
            normal_ic FLOAT,
            decile_spread FLOAT,
            decile_monotonicity FLOAT,
            is_decayed BOOLEAN DEFAULT FALSE,
            current_weight FLOAT,
            suggested_weight FLOAT
        );
        CREATE INDEX IF NOT EXISTS idx_ic_factor_date
            ON public.{self.TABLE_NAME} (factor_name, check_date);
        """
        with self._engine.begin() as conn:
            for stmt in ddl.split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(sql_text(s))

    # ── IC 计算 ────────────────────────────────────────────────

    @staticmethod
    def calc_rank_ic(factor: pd.Series, forward_return: pd.Series) -> float:
        """Spearman 秩相关系数。"""
        valid = factor.notna() & forward_return.notna()
        if valid.sum() < 5:
            return 0.0
        rho, _ = spearmanr(factor[valid], forward_return[valid])
        return rho if not np.isnan(rho) else 0.0

    @staticmethod
    def calc_icir(ic_series: list[float]) -> float:
        """ICIR = IC 均值 / IC 标准差，衡量预测稳定性。"""
        if len(ic_series) < 2:
            return 0.0
        arr = np.array(ic_series)
        std = arr.std()
        return float(arr.mean() / std) if std > 0 else 0.0

    # ── 向前收益率矩阵（向量化预计算） ──────────────────────────

    @staticmethod
    def _precompute_forward_returns(kline: pd.DataFrame,
                                    horizon: int = 5) -> pd.Series:
        sorted_df = kline.sort_values(["symbol", "trade_date"]).copy()
        sorted_df["close"] = pd.to_numeric(sorted_df["close"], errors="coerce")
        grouped = sorted_df.groupby("symbol")["close"]
        future_close = grouped.shift(-horizon)
        fwd_ret = (future_close / sorted_df["close"] - 1).replace(
            [float("inf"), -float("inf")], 0
        ).fillna(0)
        idx = pd.MultiIndex.from_arrays(
            [sorted_df["symbol"], sorted_df["trade_date"].astype(str)],
            names=["symbol", "trade_date"],
        )
        return pd.Series(fwd_ret.values, index=idx)

    # ── 滚动 IC 计算 ─────────────────────────────────────────

    def calc_rolling_ic(self, factor_name: str, factor_scores: pd.Series,
                        symbols: list[str], kline: pd.DataFrame,
                        trade_dates: list[str],
                        horizon: int = 5,
                        window: int = 20) -> list[float]:
        fwd_matrix = self._precompute_forward_returns(kline, horizon)

        ics = []
        for _ in range(min(window, len(trade_dates))):
            if not trade_dates:
                break
            t = trade_dates.pop()
            try:
                fwd_vals = fwd_matrix.xs(t, level="trade_date")
            except KeyError:
                continue
            fwd_vals = fwd_vals.reindex(symbols).dropna()
            if len(fwd_vals) < 5:
                continue
            aligned = factor_scores.reindex(fwd_vals.index).dropna()
            common = aligned.index.intersection(fwd_vals.index)
            if len(common) < 5:
                continue
            ic = self.calc_rank_ic(aligned[common], fwd_vals[common])
            ics.append(ic)
        return ics if ics else [0.0]

    # ── 分层回测 ──────────────────────────────────────────────

    def decile_backtest(self, factor_scores: pd.Series,
                        symbols: list[str], kline: pd.DataFrame,
                        trade_date: str,
                        horizon: int = 5) -> dict[str, Any]:
        """十档分层回测：按因子值分组，计算每组等权未来收益。

        Returns:
            dict: {
                "decile_returns": [d1_return, ..., d10_return],
                "top_minus_bottom": spread,
                "monotonicity": Spearman 相关系数（档位 × 收益）,
                "long_only_return": 多头组收益 (top 30%),
                "short_only_return": 空头组收益 (bottom 30%),
            }
        """
        result: dict[str, Any] = {
            "decile_returns": [],
            "top_minus_bottom": 0.0,
            "monotonicity": 0.0,
            "long_only_return": 0.0,
            "short_only_return": 0.0,
        }

        fwd_ret = self._precompute_forward_returns(kline, horizon)
        try:
            fwd_vals = fwd_ret.xs(trade_date, level="trade_date")
        except KeyError:
            return result

        # 对齐因子值和收益率
        common = factor_scores.dropna().index.intersection(fwd_vals.dropna().index)
        if len(common) < 20:
            return result

        scores = factor_scores[common]
        returns = fwd_vals[common]

        # 十档分组
        ranked = scores.rank(method="first")
        decile_labels = pd.qcut(ranked, 10, labels=False, duplicates="drop")
        n_deciles = decile_labels.nunique()

        decile_rets = []
        for d in range(n_deciles):
            mask = decile_labels == d
            ret = returns[mask].mean()
            decile_rets.append(float(ret))

        result["decile_returns"] = decile_rets
        result["top_minus_bottom"] = float(decile_rets[-1] - decile_rets[0]) if len(decile_rets) >= 10 else 0.0

        # 单调性检验
        if len(decile_rets) >= 5:
            mono_rho, _ = spearmanr(range(len(decile_rets)), decile_rets)
            result["monotonicity"] = float(mono_rho) if not np.isnan(mono_rho) else 0.0

        # 多头组 (top 30%) vs 空头组 (bottom 30%)
        threshold_top = scores.quantile(0.7)
        threshold_bot = scores.quantile(0.3)
        long_mask = scores >= threshold_top
        short_mask = scores <= threshold_bot
        result["long_only_return"] = float(returns[long_mask].mean()) if long_mask.any() else 0.0
        result["short_only_return"] = float(returns[short_mask].mean()) if short_mask.any() else 0.0

        return result

    # ── 衰减检测 ───────────────────────────────────────────────

    @staticmethod
    def detect_decay(rolling_ic: list[float],
                     ic_threshold: float = 0.02,
                     decay_days: int = 10) -> tuple[bool, float]:
        if not rolling_ic:
            return False, 0.0
        recent = rolling_ic[-decay_days:] if len(rolling_ic) >= decay_days else rolling_ic
        recent_mean = float(np.mean(recent))
        is_decayed = recent_mean < ic_threshold and recent_mean < 0
        return is_decayed, recent_mean

    # ── 权重建议 ───────────────────────────────────────────────

    def suggest_weight(self, factor_name: str, recent_mean_ic: float,
                       initial_ic: float = 0.05) -> float:
        current = self._weights.get(factor_name, 0.0)
        if current <= 0:
            return 0.0
        if initial_ic <= 0:
            initial_ic = 0.05
        ratio = max(0.0, recent_mean_ic / initial_ic)
        suggested = current * ratio
        return round(suggested, 4)

    # ── 主流程 ─────────────────────────────────────────────────

    def run(self, consolidated_report: pd.DataFrame,
            hist_df: pd.DataFrame) -> dict[str, Any]:
        if consolidated_report.empty or hist_df.empty:
            return {"error": "数据不足", "factors": {}, "needs_rebalance": False,
                    "timestamp": datetime.now().isoformat()}

        symbols = consolidated_report["股票代码"].unique().tolist()
        trade_dates = sorted(hist_df["trade_date"].astype(str).unique().tolist())
        latest_date = trade_dates[-1] if trade_dates else ""

        result: dict[str, Any] = {"factors": {}, "needs_rebalance": False,
                                   "timestamp": datetime.now().isoformat()}

        for fname, fcol in self.FACTOR_COLUMNS.items():
            if fcol not in consolidated_report.columns:
                continue

            factor_scores = pd.to_numeric(consolidated_report[fcol], errors="coerce")
            rolling_ic = self.calc_rolling_ic(
                fname, factor_scores, symbols, hist_df, list(trade_dates)
            )

            # IC 统计
            ic_mean = float(np.mean(rolling_ic))
            ic_std = float(np.std(rolling_ic)) if len(rolling_ic) > 1 else 0.0
            icir_val = self.calc_icir(rolling_ic)

            # 分层回测（最新一期）
            decile_result = self.decile_backtest(
                factor_scores, symbols, hist_df, latest_date
            )

            # 衰减检测
            is_decayed, recent_mean = self.detect_decay(rolling_ic)
            suggested = self.suggest_weight(fname, recent_mean)
            current = self._weights.get(fname, 0.0)

            status = {
                "滚动IC均值": round(ic_mean, 4),
                "滚动IC标准差": round(ic_std, 4),
                "ICIR": round(icir_val, 4),
                "RankIC": round(ic_mean, 4),
                "分层多空差": round(decile_result.get("top_minus_bottom", 0.0), 6),
                "分层单调性": round(decile_result.get("monotonicity", 0.0), 4),
                "多头收益": round(decile_result.get("long_only_return", 0.0), 6),
                "空头收益": round(decile_result.get("short_only_return", 0.0), 6),
                "已衰减": is_decayed,
                "当前权重": current,
                "建议权重": suggested,
            }
            result["factors"][fname] = status

            if is_decayed:
                result["needs_rebalance"] = True
                logger.warning(
                    f"[因子衰减] {fname} 衰减！IC={ic_mean:.4f}, ICIR={icir_val:.2f}, "
                    f"权重 {current:.2f} → 建议 {suggested:.2f}"
                )
            else:
                logger.info(
                    f"[因子监控] {fname} IC={ic_mean:.4f}, ICIR={icir_val:.2f}, "
                    f"多空差={status['分层多空差']:.4f}, 权重 {current:.2f} (正常)"
                )

        self._save_ic_history(result)

        return result

    # ── 数据库持久化 ───────────────────────────────────────────

    def _save_ic_history(self, result: dict[str, Any]) -> None:
        try:
            with self._engine.begin() as conn:
                for fname, status in result.get("factors", {}).items():
                    conn.execute(sql_text(f"""
                    INSERT INTO {self.TABLE_NAME}
                        (factor_name, check_date, rolling_ic_mean, rolling_ic_std,
                         icir, rank_ic, decile_spread, decile_monotonicity,
                         is_decayed, current_weight, suggested_weight)
                    VALUES
                        (:fn, :cd, :icm, :ics, :icir, :rank_ic, :spread, :mono,
                         :dec, :cw, :sw)
                    """), {
                        "fn": fname,
                        "cd": datetime.now().date(),
                        "icm": status.get("滚动IC均值"),
                        "ics": status.get("滚动IC标准差"),
                        "icir": status.get("ICIR"),
                        "rank_ic": status.get("RankIC"),
                        "spread": status.get("分层多空差"),
                        "mono": status.get("分层单调性"),
                        "dec": status.get("已衰减", False),
                        "cw": status.get("当前权重"),
                        "sw": status.get("建议权重"),
                    })
        except Exception as e:
            logger.warning(f"[因子衰减] 写入 IC 历史失败: {e}")

    def load_ic_history(self, days: int = 60) -> pd.DataFrame:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = sql_text(
            f"SELECT * FROM {self.TABLE_NAME} "
            "WHERE check_date >= :since "
            "ORDER BY check_date, factor_name"
        )
        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"since": since})
