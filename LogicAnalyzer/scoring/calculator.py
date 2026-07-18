from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text as sql_text


class FactorCalculator:
    """多因子 Alpha 计算引擎。

    计算质量、估值、动量、资金流四类因子 Z-Score（行业内中性化），
    与现有 MACD 评分加权融合生成新的综合分析评分。

    因子定义由 FactorRegistry（YAML 配置驱动）统一管理，
    修改 config/factor_registry.yaml 即可调整权重和参数，无需改代码。
    """

    def __init__(self, config: Any, db_engine: Any) -> None:  # noqa: ANN401
        self.config = config
        self._engine = db_engine
        from LogicAnalyzer.scoring.factor_registry import FactorRegistry

        config_dir = getattr(config, "CONFIG_DIR", None) or "config"
        registry_path = os.path.join(config_dir, "factor_registry.yaml")
        self._registry = FactorRegistry(config_path=registry_path)
        self._weights: dict[str, float] = self._registry.weights

    # ── 质量因子 ─────────────────────────────────────────────────

    @staticmethod
    def calc_quality_scores(df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """计算质量因子综合评分。

        公式: ROE × 0.4 + 毛利率 × 0.3 + 净利率 × 0.3
        然后行业内 Z-Score 标准化。
        """
        if df.empty:
            return pd.Series(dtype=float)

        composite = (
            df.get("roe", 0).fillna(0) * 0.4
            + df.get("gross_profit_margin", 0).fillna(0) * 0.3
            + df.get("net_profit_margin", 0).fillna(0) * 0.3
        )
        return FactorCalculator._industry_zscore(
            composite, df.get(industry_col, pd.Series(dtype=str))
        )

    # ── 估值因子 ─────────────────────────────────────────────────

    @staticmethod
    def calc_valuation_scores(df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """计算估值因子 Z-Score（行业内）。

        PE_TTM < 0 或 > 200 视为缺失；PB 同理。
        估值分数 = -(PE_TTM_z + PB_z) / 2 （高估值得低分）
        """
        if df.empty:
            return pd.Series(dtype=float)

        pe = df.get("pe_ttm", pd.Series(dtype=float))
        pb = df.get("pb", pd.Series(dtype=float))
        industry = df.get(industry_col, pd.Series(dtype=str))

        # 剔除异常值
        pe_clean = pe.where((pe > 0) & (pe <= 200), np.nan)
        pb_clean = pb.where((pb > 0) & (pb <= 100), np.nan)

        pe_z = FactorCalculator._industry_zscore(pe_clean, industry).fillna(0)
        pb_z = FactorCalculator._industry_zscore(pb_clean, industry).fillna(0)

        return -(pe_z + pb_z) / 2

    # ── 动量因子 ─────────────────────────────────────────────────

    @staticmethod
    def calc_momentum_scores(symbols: list[str], hist_df: pd.DataFrame,
                             industry_map: dict[str, str] | None = None) -> pd.Series:
        """计算 21 交易日动量（行业内中性化），向量化版本。

        Args:
            symbols: 股票代码列表（纯代码，如 600519）。
            hist_df: K线 DataFrame，必须含 symbol, trade_date, close 列。
            industry_map: {symbol: industry_name} 映射。

        Returns:
            Series: 动量 Z-Score，index 为 symbol。
        """
        if hist_df.empty:
            return pd.Series(0.0, index=symbols)

        # 向量化：过滤目标股票 → 排序 → 每组取最近 21 根 → 计算收益率
        subset = hist_df[hist_df["symbol"].isin(symbols)]
        if subset.empty:
            return pd.Series(0.0, index=symbols)

        sorted_df = subset.sort_values(["symbol", "trade_date"])
        last_21 = sorted_df.groupby("symbol").tail(21)

        first_close = last_21.groupby("symbol")["close"].first()
        last_close = last_21.groupby("symbol")["close"].last()
        momentum = ((last_close - first_close) / first_close).replace([float("inf"), -float("inf")], 0).fillna(0)

        # 确保所有请求的 symbol 都有值
        momentum = momentum.reindex(symbols, fill_value=0.0)

        if industry_map:
            aligned_ind = momentum.index.to_series().map(industry_map)
            return FactorCalculator._industry_zscore(momentum, aligned_ind).fillna(0)
        else:
            std = momentum.std()
            if std == 0:
                return momentum
            return ((momentum - momentum.mean()) / std).fillna(0)

    # ── 资金流因子 ───────────────────────────────────────────────

    @staticmethod
    def calc_moneyflow_scores(df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """从现有资金流数据计算资金流因子 Z-Score。

        使用 5 日/10 日/20 日资金流入的加权平均，行业内中性化。
        """
        if df.empty:
            return pd.Series(dtype=float)

        candidates = ["5日资金流入万元", "10日资金流入万元", "20日资金流入万元",
                       "3日资金流入万元"]
        available = [c for c in candidates if c in df.columns]
        if not available:
            return pd.Series(0.0, index=df.index)

        weights = {"3日资金流入万元": 0.3, "5日资金流入万元": 0.4,
                   "10日资金流入万元": 0.2, "20日资金流入万元": 0.1}
        total = sum(weights[c] for c in available)
        composite = sum(df[c].fillna(0) * weights[c] for c in available) / total
        return FactorCalculator._industry_zscore(
            composite, df.get(industry_col, pd.Series(dtype=str))
        ).fillna(0)

    # ── 行业内 Z-Score ──────────────────────────────────────────

    @staticmethod
    def _industry_zscore(series: pd.Series, industry: pd.Series) -> pd.Series:
        """行业内 Z-Score，clip 到 [-3, 3]。"""
        if industry.isna().all() or industry.nunique() <= 1:
            std = series.std()
            result = (series - series.mean()) / std if std != 0 else pd.Series(0, index=series.index)
            return result.clip(-3, 3).fillna(0)

        def _zscore(x: pd.Series) -> pd.Series:
            s = x.std()
            return (x - x.mean()) / s if s != 0 else pd.Series(0, index=x.index)

        result = series.groupby(industry).transform(_zscore)
        return result.clip(-3, 3).fillna(0)

    # ── 融合评分 ─────────────────────────────────────────────────

    def fuse_scores(
        self,
        report: pd.DataFrame,
        macd_score_col: str = "综合分析评分",
        industry_col: str = "行业",
        hist_df: pd.DataFrame | None = None,
        quality_df: pd.DataFrame | None = None,
        valuation_df: pd.DataFrame | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        """将多维因子评分融合到报告中，更新综合分析评分。

        Args:
            report: 合并处理的 DataFrame（含 MACD 评分）。
            macd_score_col: MACD 评分列名。
            industry_col: 行业列名。
            hist_df: K 线 DataFrame（用于动量因子计算）。
            quality_df: 质量因子 DataFrame（含 symbol, roe, ...）。
            valuation_df: 估值因子 DataFrame（含 symbol, pe_ttm, pb, ...）。

        Returns:
            添加了各因子评分列并更新综合分析评分的 DataFrame。
        """
        if report.empty or not self._weights:
            return report

        result = report.copy()
        symbol_map = {s: i for i, s in result.get("股票代码", pd.Series(dtype=str)).items()}
        logger.info(f"[v] fuse_scores start: result.shape={result.shape}, cols={list(result.columns)}")

        # 1. 将外部因子数据 merge 到 report
        if quality_df is not None and not quality_df.empty:
            q_df = quality_df.set_index("symbol")
            for col in ["roe", "gross_profit_margin", "net_profit_margin"]:
                if col in q_df.columns:
                    result[col] = result["股票代码"].map(q_df[col]).fillna(0)

        if valuation_df is not None and not valuation_df.empty:
            v_df = valuation_df.set_index("symbol")
            for col in ["pe_ttm", "pb"]:
                if col in v_df.columns:
                    result[col] = result["股票代码"].map(v_df[col]).fillna(0)

        # 2. 计算各因子评分
        quality_score = self.calc_quality_scores(result, industry_col)
        valuation_score = self.calc_valuation_scores(result, industry_col)
        moneyflow_score = self.calc_moneyflow_scores(result, industry_col)

        # 动量需要 kline
        symbols = [s for s in result["股票代码"].unique() if s]
        industry_map = (
            result.set_index("股票代码")[industry_col].to_dict()
            if industry_col in result.columns else None
        )
        momentum_score = self.calc_momentum_scores(symbols, hist_df if not hist_df.empty else pd.DataFrame(), industry_map)

        # 对齐索引
        result["基本面评分"] = quality_score.reindex(result.index).fillna(0)
        result["估值评分"] = valuation_score.reindex(result.index).fillna(0)
        if not result.empty:
            code_idx = result.drop_duplicates(subset="股票代码").set_index("股票代码").index
            aligned = momentum_score.reindex(code_idx)
            result["动量评分"] = result["股票代码"].map(aligned.to_dict()).fillna(0)
        else:
            result["动量评分"] = 0
        result["资金流评分"] = moneyflow_score.reindex(result.index).fillna(0)

        # 3. MACD 原始评分归一化到 [-3, 3]
        raw_macd = pd.to_numeric(result.get(macd_score_col, 0), errors="coerce").fillna(0)
        macd_std = raw_macd.std()
        macd_z = ((raw_macd - raw_macd.mean()) / (macd_std if macd_std != 0 else 1)).clip(-3, 3).fillna(0)
        result["MACD评分"] = macd_z

        # 4. 加权融合
        w = self._weights
        total_w = sum(w.values())
        if total_w == 0:
            total_w = 1

        result["综合分析评分"] = (
            result["MACD评分"] * w.get("macd", 0) / total_w
            + result["动量评分"] * w.get("momentum", 0) / total_w
            + result["资金流评分"] * w.get("moneyflow", 0) / total_w
            + result["基本面评分"] * w.get("quality", 0) / total_w
            + result["估值评分"] * w.get("valuation", 0) / total_w
        )

        # 映射回 0-100 评分
        raw = result["综合分析评分"]
        result["综合分析评分"] = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-10) * 100).clip(0, 100)

        # 行业截面百分位（用于步骤 14 过滤）
        result = self._add_industry_percentiles(result, industry_col)

        logger.info(
            "[FactorCalculator] 多因子评分融合完成，因子权重: "
            f"MACD={w.get('macd',0):.2f} 动量={w.get('momentum',0):.2f} "
            f"资金流={w.get('moneyflow',0):.2f} 质量={w.get('quality',0):.2f} "
            f"估值={w.get('valuation',0):.2f}"
        )

        # 写入 DW 层宽表
        if trade_date:
            try:
                self._save_to_dwd(result, trade_date)
            except Exception:
                logger.opt(exception=True).warning("[DW层] dwd_factor_daily 写入失败")

        return result

    @staticmethod
    def _add_industry_percentiles(df: pd.DataFrame, industry_col: str = "行业") -> pd.DataFrame:
        """在 DataFrame 中添加行业截面百分位列（0-100），用于步骤 14 过滤。"""
        if industry_col not in df.columns:
            return df
        from DataManager.ColumnNames import ColumnNames as CN
        score_cols = [
            ("综合分析评分", CN.SCORE_PCT_INDUSTRY),
            ("动量评分", CN.MOMENTUM_PCT_INDUSTRY),
            ("基本面评分", CN.QUALITY_PCT_INDUSTRY),
            ("估值评分", CN.VALUATION_PCT_INDUSTRY),
        ]
        for src, dst in score_cols:
            if src in df.columns:
                df[dst] = df.groupby(industry_col, observed=True)[src].rank(pct=True) * 100
            else:
                df[dst] = 50.0
        return df

    def _save_to_dwd(self, df: pd.DataFrame, trade_date: str) -> None:
        """将因子评分写入 dwd_factor_daily 宽表。"""
        import json

        FACTOR_KEYS = ["momentum", "quality", "valuation", "moneyflow", "macd"]
        COL_MAP = {
            "momentum": "动量评分",
            "quality": "基本面评分",
            "valuation": "估值评分",
            "moneyflow": "资金流评分",
            "macd": "MACD评分",
        }

        if "股票代码" not in df.columns:
            return

        rows = []
        for _, r in df.iterrows():
            symbol = str(r.get("股票代码", ""))
            if not symbol:
                continue

            factors = {}
            factor_z = {}
            factor_raw = {}

            for k in FACTOR_KEYS:
                col = COL_MAP.get(k, "")
                if col in df.columns:
                    val = r.get(col)
                    if val is not None:
                        factors[k] = float(val)

            composite = r.get("综合分析评分")
            industry = r.get("行业", "")

            rows.append({
                "trade_date": trade_date,
                "symbol": symbol,
                "industry": str(industry) if pd.notna(industry) else "",
                "composite_score": float(composite) if composite is not None else None,
                "composite_rank": 0,
                "factors": json.dumps(factors),
                "factor_z": json.dumps(factor_z),
                "factor_raw": json.dumps(factor_raw),
            })

        if not rows:
            return

        from sqlalchemy import text as sql_text

        INSERT_SQL = sql_text("""
        INSERT INTO public.dwd_factor_daily
            (trade_date, symbol, industry, composite_score, composite_rank,
             factors, factor_z, factor_raw)
         VALUES
             (:trade_date, :symbol, :industry, :composite_score, :composite_rank,
              CAST(:factors AS jsonb), CAST(:factor_z AS jsonb), CAST(:factor_raw AS jsonb))
        ON CONFLICT (trade_date, symbol) DO UPDATE SET
            industry = EXCLUDED.industry,
            composite_score = EXCLUDED.composite_score,
            factors = EXCLUDED.factors,
            factor_z = EXCLUDED.factor_z,
            factor_raw = EXCLUDED.factor_raw
        """)

        with self._engine.begin() as conn:
            for row in rows:
                conn.execute(INSERT_SQL, row)
        logger.info(f"[DW层] dwd_factor_daily 写入 {len(rows)} 条")

    def load_quality_from_db(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """从数据库加载质量因子数据。"""
        try:
            from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher
            fetcher = FinancialQualityFetcher(self.config)
            return fetcher.load_quality(symbols)
        except Exception as e:
            logger.warning(f"[FactorCalculator] 质量因子加载失败: {e}")
            return pd.DataFrame()

    def load_valuation_from_db(self, symbols: list[str] | None = None,
                                trade_date: str | None = None) -> pd.DataFrame:
        """从数据库加载估值因子数据。"""
        try:
            from DataCollection.FinancialValuationFetcher import FinancialValuationFetcher
            fetcher = FinancialValuationFetcher(self.config)
            return fetcher.load_latest_valuation(symbols, trade_date)
        except Exception as e:
            logger.warning(f"[FactorCalculator] 估值因子加载失败: {e}")
            return pd.DataFrame()
