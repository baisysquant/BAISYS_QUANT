"""
因子衰减监控器

计算各因子 IC（信息系数）的滚动统计量，判断因子是否失效，
并在衰减时建议自动降权。

方法：
  - IC：Spearman 秩相关系数（因子值与未来 N 日收益率）
  - 滚动 IC 均值 + 标准差
  - 衰减判定：滚动 IC 均值 < 0 且持续 D 天 → 衰减
  - 降权公式：新权重 = 原权重 × max(0, 滚动 IC 均值 / 初始 IC)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text as sql_text

from DataManager.ColumnNames import ColumnNames
from DataManager.DbEngine import get_engine


class FactorDecayMonitor:
    """多因子衰减监控与自动降权。

    用法:
        monitor = FactorDecayMonitor(config, db_engine)
        status = monitor.run(consolidated_report, hist_df)
        # status 包含各因子的衰减状态和建议权重
    """

    TABLE_NAME = "ods_factor_ic_history"

    # 因子名 → 报告中对应的列名
    FACTOR_COLUMNS = {
        "macd": "MACD评分",
        "momentum": "动量评分",
        "moneyflow": "资金流评分",
        "quality": "基本面评分",
        "valuation": "估值评分",
    }

    def __init__(self, config: Any, db_engine: Any) -> None:  # noqa: ANN401
        self.config = config
        self._engine = db_engine
        self._weights: dict[str, float] = getattr(config, "FACTOR_WEIGHTS", None) or {}

    # ── IC 计算 ────────────────────────────────────────────────

    @staticmethod
    def calc_ic(factor: pd.Series, forward_return: pd.Series) -> float:
        """计算单期 IC：因子值与未来收益的 Spearman 秩相关。"""
        valid = factor.notna() & forward_return.notna()
        if valid.sum() < 5:
            return 0.0
        rho, _ = spearmanr(factor[valid], forward_return[valid])
        return rho if not np.isnan(rho) else 0.0

    @staticmethod
    def calc_forward_returns(kline: pd.DataFrame, symbol: str,
                             today: str, horizon: int = 5) -> float | None:
        """计算指定股票从 today 起未来 N 日的收益率。"""
        stock = kline[kline["symbol"] == symbol].sort_values("trade_date")
        dates = stock["trade_date"].astype(str).tolist()
        if today not in dates:
            return None
        pos = dates.index(today)
        if pos + horizon >= len(dates):
            return None
        today_close = stock.iloc[pos]["close"]
        future_close = stock.iloc[pos + horizon]["close"]
        if today_close == 0:
            return None
        return future_close / today_close - 1

    # ── 滚动 IC ────────────────────────────────────────────────

    def calc_rolling_ic(self, factor_name: str, factor_scores: pd.Series,
                        symbols: list[str], kline: pd.DataFrame,
                        trade_dates: list[str],
                        horizon: int = 5,
                        window: int = 20) -> list[float]:
        """计算滚动窗口内的平均 IC。"""
        ics = []
        for _ in range(min(window, len(trade_dates))):
            if not trade_dates:
                break
            t = trade_dates.pop()
            fwd_rets = {}
            for sym in symbols:
                ret = self.calc_forward_returns(kline, sym, t, horizon)
                if ret is not None:
                    fwd_rets[sym] = ret
            fwd_series = pd.Series(fwd_rets)
            aligned = factor_scores.reindex(fwd_series.index).dropna()
            if len(aligned) >= 5:
                ic = self.calc_ic(aligned, fwd_series.reindex(aligned.index))
                ics.append(ic)
        return ics if ics else [0.0]

    # ── 衰减检测 ───────────────────────────────────────────────

    @staticmethod
    def detect_decay(rolling_ic: list[float],
                     ic_threshold: float = 0.02,
                     decay_days: int = 10) -> tuple[bool, float]:
        """检测因子是否衰减。

        Returns:
            (is_decayed, mean_ic)
        """
        if not rolling_ic:
            return False, 0.0
        mean_ic = float(np.mean(rolling_ic))
        recent = rolling_ic[-decay_days:] if len(rolling_ic) >= decay_days else rolling_ic
        recent_mean = float(np.mean(recent))
        is_decayed = recent_mean < ic_threshold and recent_mean < 0
        return is_decayed, recent_mean

    # ── 权重建议 ───────────────────────────────────────────────

    def suggest_weight(self, factor_name: str, recent_mean_ic: float,
                       initial_ic: float = 0.05) -> float:
        """根据 IC 衰减程度建议新权重。"""
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
        """执行全因子衰减扫描。

        Args:
            consolidated_report: 合并后的报告 DataFrame（含各因子评分列）。
            hist_df: K 线 DataFrame（含 symbol, trade_date, close）。

        Returns:
            dict: {
                "factors": {
                    "macd": {"ic": float, "decayed": bool, "suggested_weight": float, ...},
                    ...
                },
                "needs_rebalance": bool,
                "timestamp": str,
            }
        """
        if consolidated_report.empty or hist_df.empty:
            return {"error": "数据不足", "factors": {}, "needs_rebalance": False,
                    "timestamp": datetime.now().isoformat()}

        symbols = consolidated_report["股票代码"].unique().tolist()
        trade_dates = sorted(hist_df["trade_date"].astype(str).unique().tolist())

        result: dict[str, Any] = {"factors": {}, "needs_rebalance": False,
                                   "timestamp": datetime.now().isoformat()}

        for fname, fcol in self.FACTOR_COLUMNS.items():
            if fcol not in consolidated_report.columns:
                continue
            factor_scores = pd.to_numeric(consolidated_report[fcol], errors="coerce")
            rolling_ic = self.calc_rolling_ic(
                fname, factor_scores, symbols, hist_df, list(trade_dates)
            )
            is_decayed, mean_ic = self.detect_decay(rolling_ic)
            suggested = self.suggest_weight(fname, mean_ic)
            current = self._weights.get(fname, 0.0)

            status = {
                "滚动IC均值": round(mean_ic, 4),
                "已衰减": is_decayed,
                "当前权重": current,
                "建议权重": suggested,
            }
            result["factors"][fname] = status

            if is_decayed:
                result["needs_rebalance"] = True
                logger.warning(
                    f"[因子衰减] {fname} 衰减！IC={mean_ic:.4f}, "
                    f"权重 {current:.2f} → 建议 {suggested:.2f}"
                )
            else:
                logger.info(
                    f"[因子监控] {fname} IC={mean_ic:.4f}, 权重 {current:.2f} (正常)"
                )

        # 保存到数据库
        self._save_ic_history(result)

        return result

    # ── 数据库持久化 ───────────────────────────────────────────

    def _save_ic_history(self, result: dict[str, Any]) -> None:
        """将本次 IC 扫描结果写入数据库。"""
        try:
            with self._engine.begin() as conn:
                for fname, status in result.get("factors", {}).items():
                    conn.execute(sql_text(
                        f"INSERT INTO {self.TABLE_NAME} "
                        "(factor_name, check_date, rolling_ic_mean, is_decayed, "
                        "current_weight, suggested_weight) "
                        "VALUES (:fn, :cd, :ic, :dec, :cw, :sw)"
                    ), {
                        "fn": fname,
                        "cd": datetime.now().date(),
                        "ic": status["滚动IC均值"],
                        "dec": status["已衰减"],
                        "cw": status["当前权重"],
                        "sw": status["建议权重"],
                    })
        except Exception as e:
            logger.warning(f"[因子衰减] 写入 IC 历史失败: {e}")

    def load_ic_history(self, days: int = 60) -> pd.DataFrame:
        """加载最近 N 天的 IC 历史。"""
        from sqlalchemy import text

        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = text(
            f"SELECT * FROM {self.TABLE_NAME} "
            "WHERE check_date >= :since "
            "ORDER BY check_date, factor_name"
        )
        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"since": since})
