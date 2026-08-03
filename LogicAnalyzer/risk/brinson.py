"""
Brinson 业绩归因模块

将组合相对基准的超额收益拆解为三个来源（按行业分组）：

  - 配置效应（Allocation）  ：行业超配/低配带来的收益
  - 选择效应（Selection）   ：行业内个股选择（选股能力）带来的收益
  - 交互效应（Interaction） ：配置与选择的联合项（两因素非正交）

公式（Brinson–Fachler 变体，行业 i）：
    r_b        = Σᵢ bᵢ·r_bi                      组合基准收益
    配置效应ᵢ  = (wᵢ − bᵢ)·(r_bi − r_b)
    选择效应ᵢ  = bᵢ·(r_pi − r_bi)
    交互效应ᵢ  = (wᵢ − bᵢ)·(r_pi − r_bi)
    超额收益   = Σᵢ (配置效应ᵢ + 选择效应ᵢ + 交互效应ᵢ) = r_p − r_b

支持：
  - 单期归因（decompose）
  - 多期归因汇总（aggregate_periods）
  - 由持仓明细 + 全市场 K 线自动归因（from_holdings，基准默认全市场等权）
  - Excel 报告输出（build_report / to_excel）

用法:
    br = BrinsonDecomposition()
    result = br.decompose(pf_weights, bm_weights, pf_returns, bm_returns)
    sheet = br.build_report(result)          # DataFrame，可写入 Excel
    br.to_excel(result, "brinson.xlsx")      # 独立 Excel 报告
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class BrinsonDecomposition:
    """Brinson 行业配置 / 个股选择收益归因。"""

    # ── 单期归因 ────────────────────────────────────────────

    def decompose(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict[str, Any]:
        """单期 Brinson 归因。

        Args:
            portfolio_weights: 组合行业权重 Series（index=行业）。
            benchmark_weights: 基准行业权重 Series（index=行业）。
            portfolio_returns: 组合行业收益率 Series（index=行业）。
            benchmark_returns: 基准行业收益率 Series（index=行业）。

        Returns:
            dict: {
                "归因表": pd.DataFrame(行业|配置效应|选择效应|交互效应|总贡献),
                "组合收益": float, "基准收益": float, "超额收益": float,
            }
            数据不足或合计不平衡（>1e-6）时返回 {"error": "..."}。
        """
        if (
            portfolio_weights is None
            or benchmark_weights is None
            or portfolio_returns is None
            or benchmark_returns is None
        ):
            return {"error": "权重/收益输入不完整"}

        w = pd.to_numeric(portfolio_weights, errors="coerce").fillna(0.0)
        b = pd.to_numeric(benchmark_weights, errors="coerce").fillna(0.0)
        rp = pd.to_numeric(portfolio_returns, errors="coerce").fillna(0.0)
        rb = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0)

        industries = w.index.union(b.index).union(rp.index).union(rb.index)
        if len(industries) == 0:
            return {"error": "无有效行业"}

        w = w.reindex(industries).fillna(0.0)
        b = b.reindex(industries).fillna(0.0)
        rp = rp.reindex(industries).fillna(0.0)
        rb = rb.reindex(industries).fillna(0.0)

        # 权重归一化（避免输入未归一化的组合）
        if w.sum() > 1e-12:
            w = w / w.sum()
        if b.sum() > 1e-12:
            b = b / b.sum()

        r_bench_total = float(b.dot(rb))
        r_portfolio_total = float(w.dot(rp))

        allocation = (w - b) * (rb - r_bench_total)
        selection = b * (rp - rb)
        interaction = (w - b) * (rp - rb)

        attribution = pd.DataFrame(
            {
                "行业": industries,
                "配置效应": np.round(allocation.to_numpy(), 8),
                "选择效应": np.round(selection.to_numpy(), 8),
                "交互效应": np.round(interaction.to_numpy(), 8),
                "总贡献": np.round((allocation + selection + interaction).to_numpy(), 8),
            }
        )

        excess = r_portfolio_total - r_bench_total
        check = float(attribution["总贡献"].sum())
        if abs(check - excess) > 1e-6:
            logger.warning(f"[Brinson] 归因合计 {check:.8f} 与超额收益 {excess:.8f} 不闭合")
            return {"error": f"归因合计不闭合（{check:.8f} vs {excess:.8f}）"}

        logger.info(
            f"[Brinson] 组合 {r_portfolio_total:.4%} vs 基准 {r_bench_total:.4%} | "
            f"超额 {excess:.4%} = 配置 {float(allocation.sum()):.4%} + "
            f"选择 {float(selection.sum()):.4%} + 交互 {float(interaction.sum()):.4%}"
        )
        return {
            "归因表": attribution,
            "组合收益": r_portfolio_total,
            "基准收益": r_bench_total,
            "超额收益": excess,
        }

    # ── 多期汇总 ────────────────────────────────────────────

    @staticmethod
    def aggregate_periods(results: list[dict[str, Any]]) -> dict[str, Any]:
        """对多期归因结果做跨期汇总（效应逐期加总，组合/基准收益按复利合成）。

        Args:
            results: 多期 decompose() / from_holdings() 的返回 dict 列表。

        Returns:
            单期归因同结构结果（跨期加总版）。
        """
        valid = [r for r in results if isinstance(r, dict) and "归因表" in r]
        if not valid:
            return {"error": "无有效归因期数"}

        total = sum(
            r["归因表"].set_index("行业") for r in valid
        ).reset_index()

        # 组合/基准收益按几何合成（跨期）
        port = 1.0
        bench = 1.0
        for r in valid:
            port *= 1.0 + float(r["组合收益"])
            bench *= 1.0 + float(r["基准收益"])
        port -= 1.0
        bench -= 1.0

        excess = float(total["总贡献"].sum())
        logger.info(f"[Brinson] 汇总 {len(valid)} 期 | 组合 {port:.4%} vs 基准 {bench:.4%} | 超额 {excess:.4%}")
        return {
            "归因表": total,
            "组合收益": port,
            "基准收益": bench,
            "超额收益": excess,
        }

    # ── 持仓明细自动归因 ────────────────────────────────────

    def from_holdings(
        self,
        holdings_df: pd.DataFrame,
        kline_df: pd.DataFrame,
        industry_map: pd.Series | None = None,
        weight_col: str = "目标权重",
        industry_col: str = "所属行业",
        code_col: str = "股票代码",
        date_col: str = "trade_date",
        close_col: str = "close",
        benchmark_weights: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        """由持仓明细 + 全市场 K 线自动计算单期 Brinson 归因。

        Args:
            holdings_df: 持仓明细，需含 股票代码 / 权重列 / 行业列。
            kline_df: 全市场日 K 线（symbol, date_col, close）。
            industry_map: 股票代码 → 行业 的映射 Series；
                未提供时用 holdings 的行业列补充基准行业构成。
            weight_col / industry_col / code_col: 持仓列名。
            date_col / close_col: K 线列名。
            benchmark_weights: 基准行业权重 Series；None 时以
                holdings ∪ industry_map 覆盖的股票为宇宙等权构建。
            benchmark_returns: 基准行业收益率 Series；None 时按宇宙内
                个股收益率等权平均。

        Returns:
            同 decompose()；数据不足时返回 {"error": "..."}。
        """
        if holdings_df is None or holdings_df.empty:
            return {"error": "持仓明细为空"}
        if kline_df is None or kline_df.empty:
            return {"error": "K 线数据为空"}

        holdings = holdings_df.copy()
        if industry_col not in holdings.columns and "行业" in holdings.columns:
            industry_col = "行业"
        if industry_col not in holdings.columns:
            return {"error": f"持仓缺少行业列（{industry_col}）"}

        code_col = code_col if code_col in holdings.columns else "股票代码"
        w = pd.to_numeric(holdings[weight_col], errors="coerce").fillna(0.0)
        industries = holdings[industry_col].astype(str).fillna("未知")
        codes = self._norm_codes(holdings[code_col])

        # 组合：行业内权重归一化 × 收益率
        pf_weights = pd.Series(w.to_numpy(), index=industries).groupby(level=0).sum()
        stock_rets = self._stock_returns(kline_df, codes, date_col, close_col)
        stock_ret_map = stock_rets.reindex(codes).fillna(0.0)
        ind_pf_ret = (
            pd.DataFrame({"w": w.to_numpy(), "r": stock_ret_map.to_numpy(), "ind": industries.to_numpy()})
            .groupby("ind")
            .apply(lambda g: float((g["w"] * g["r"]).sum()) / float(g["w"].sum()) if g["w"].sum() > 0 else 0.0)
        )

        # 基准：优先使用外部基准；否则 holdings ∪ industry_map 等权宇宙
        if benchmark_weights is not None and benchmark_returns is not None:
            bm_weights = benchmark_weights
            bm_returns = benchmark_returns
        else:
            ind_of: dict[str, str] = dict(zip(codes, industries))
            if industry_map is not None:
                for code, ind in industry_map.items():
                    ind_of[str(self._norm_codes(pd.Series([code])).iloc[0])] = str(ind)
            universe = stock_rets.index
            rows = []
            for code in universe:
                ind = ind_of.get(str(code))
                if ind is not None:
                    rows.append({"code": str(code), "ind": ind, "r": float(stock_rets.get(code, 0.0))})
            if not rows:
                return {"error": "无法构建基准行业构成"}
            u_df = pd.DataFrame(rows)
            n_universe = len(u_df)
            bm_weights = u_df.groupby("ind")["code"].count() / n_universe
            bm_returns = u_df.groupby("ind")["r"].mean()

        return self.decompose(pf_weights, bm_weights, ind_pf_ret, bm_returns)

    @staticmethod
    def _norm_codes(series: pd.Series) -> pd.Series:
        """将股票代码统一为纯 6 位格式（sh600519 → 600519）。"""
        try:
            from UtilsManager.CodeNormalizer import CodeNormalizer

            return series.map(lambda c: str(CodeNormalizer.normalize(str(c).strip())))
        except Exception:
            return series.astype(str).str.strip()

    @staticmethod
    def _stock_returns(
        kline_df: pd.DataFrame,
        codes: pd.Series,
        date_col: str,
        close_col: str,
    ) -> pd.Series:
        """全市场 K 线内每只股票在数据区间的收益率（期末/期初 - 1）。"""
        k = kline_df.copy()
        k["_sym"] = BrinsonDecomposition._norm_codes(k["symbol"])
        k["_date"] = pd.to_datetime(k[date_col], errors="coerce")
        k[close_col] = pd.to_numeric(k[close_col], errors="coerce")
        k = k.dropna(subset=["_date", close_col]).sort_values(["_sym", "_date"])

        first = k.groupby("_sym")[close_col].first()
        last = k.groupby("_sym")[close_col].last()
        returns = (last / first - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        returns.index = returns.index.astype(str)
        return returns

    # ── Excel 输出 ──────────────────────────────────────────

    def build_report(self, result: dict[str, Any]) -> pd.DataFrame:
        """生成适合写入 Excel 的归因表（含 合计 行与 总览 行）。"""
        if "error" in result:
            return pd.DataFrame()

        attribution = result["归因表"].copy()
        summary_row = pd.DataFrame(
            [{
                "行业": "合计",
                "配置效应": round(float(attribution["配置效应"].sum()), 8),
                "选择效应": round(float(attribution["选择效应"].sum()), 8),
                "交互效应": round(float(attribution["交互效应"].sum()), 8),
                "总贡献": round(float(attribution["总贡献"].sum()), 8),
            }]
        )
        overview_rows = pd.DataFrame(
            [
                {"行业": "组合收益", "配置效应": "", "选择效应": "", "交互效应": "", "总贡献": round(result["组合收益"], 8)},
                {"行业": "基准收益", "配置效应": "", "选择效应": "", "交互效应": "", "总贡献": round(result["基准收益"], 8)},
                {"行业": "超额收益", "配置效应": "", "选择效应": "", "交互效应": "", "总贡献": round(result["超额收益"], 8)},
            ]
        )
        return pd.concat([attribution, summary_row, overview_rows], ignore_index=True)

    def to_excel(self, result: dict[str, Any], path: str) -> str:
        """将归因结果导出为独立 Excel 报告（xlsxwriter）。

        Args:
            result: decompose() / from_holdings() / aggregate_periods() 的返回。
            path: 输出文件路径。

        Returns:
            str: 文件路径（写入成功）。
        """
        if "error" in result:
            raise ValueError(f"Brinson 归因失败，无法导出: {result['error']}")

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            self.build_report(result).to_excel(
                writer, sheet_name="Brinson归因", index=False
            )
        logger.info(f"[Brinson] 归因报告已导出: {path}")
        return path
