"""
凸组合优化器 — 用 scipy.optimize 替代启发式约束。

将组合构建建模为凸优化问题：
  目标： 最大化 w^T μ - λ × w^T Σ w
  约束： sum(w) = 1.0（满仓）
         0 ≤ w_i ≤ 0.33（无做空 + 单票上限）
         sum(w_i for i∈industry_j) ≤ 0.30（行业集中度）
         sum(|w_i - w_prev_i|) / 2 ≤ 0.20（换手率约束）

用法:
    opt = ConvexPortfolioOptimizer()
    weights = opt.optimize(scores, industry, kline, prev_weights)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize


class ConvexPortfolioOptimizer:
    """基于凸优化的组合构建器，替代启发式约束规则。"""

    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_single: float = 0.33,
        max_industry: float = 0.30,
        max_turnover: float = 0.20,
    ) -> None:
        self._lambda = risk_aversion
        self._max_single = max_single
        self._max_industry = max_industry
        self._max_turnover = max_turnover

    # ── 主入口 ───────────────────────────────────────────

    def optimize(
        self,
        scores: pd.Series,
        industry: pd.Series,
        daily_returns: pd.DataFrame | None = None,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        """执行凸优化求解最优权重。

        Args:
            scores: index=股票代码, values=因子评分 [0-100]。
            industry: index=股票代码, values=行业名称。
            daily_returns: DataFrame index=日期, columns=股票代码, values=日收益率。
            prev_weights: index=股票代码, values=前日权重。

        Returns:
            Series: index=股票代码, values=优化后权重 [0-1]。
        """
        codes = scores.index.tolist()
        n = len(codes)
        if n == 0:
            return pd.Series(dtype=float)

        μ = self._expected_returns(scores)
        Σ = self._covariance_matrix(codes, daily_returns)

        # 初始值：等权
        x0 = np.full(n, 1.0 / n)

        # 约束
        constraints = self._build_constraints(codes, industry, prev_weights)

        # 边界
        bounds = [(0.0, self._max_single)] * n

        # 求解
        result = minimize(
            fun=self._objective,
            x0=x0,
            args=(μ, Σ),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12, "disp": False},
        )

        if not result.success:
            logger.warning(f"[ConvexOptimizer] 优化未收敛: {result.message}，使用启发式回退")
            return self._fallback(scores, industry, prev_weights)

        w = pd.Series(result.x, index=codes)
        w = w.clip(0)

        # 归一化
        total = w.sum()
        if total > 0:
            w /= total

        nonzero = (w > 0.001).sum()
        logger.info(
            f"[ConvexOptimizer] 优化完成: {nonzero}/{n} 只持仓, "
            f"目标值={result.fun:.4f}, 迭代={result.nit}"
        )
        return w

    # ── 目标函数 ─────────────────────────────────────────

    @staticmethod
    def _objective(w: np.ndarray, μ: np.ndarray, Σ: np.ndarray) -> float:
        """最小化 -w^T μ + w^T Σ w （等价于最大化收益-风险）。"""
        ret = w @ μ
        risk = w @ Σ @ w
        return float(-ret + risk)

    # ── 预期收益 ─────────────────────────────────────────

    @staticmethod
    def _expected_returns(scores: pd.Series) -> np.ndarray:
        """将因子评分映射为预期收益信号。"""
        s = scores.values.astype(float)
        # 中心化 + 缩放
        s = (s - s.mean()) / (s.std() + 1e-10)
        return s.clip(-3, 3)

    # ── 协方差矩阵 ───────────────────────────────────────

    def _covariance_matrix(self, codes: list[str],
                           daily_returns: pd.DataFrame | None) -> np.ndarray:
        """估算股票收益率的协方差矩阵。"""
        n = len(codes)
        if daily_returns is None or daily_returns.empty:
            return np.eye(n) * 0.01  # 对角占优回退

        # 对齐可用列
        avail = [c for c in codes if c in daily_returns.columns]
        if len(avail) < 2:
            return np.eye(n) * 0.01

        rets = daily_returns[avail].dropna(how="all")
        if rets.empty or rets.shape[0] < 10:
            return np.eye(n) * 0.01

        # 样本协方差
        cov = rets.cov().values

        # 如果缺失列，扩展回退
        if len(avail) < n:
            full = np.eye(n) * 0.01
            idx_map = {c: i for i, c in enumerate(codes)}
            for i, c1 in enumerate(avail):
                for j, c2 in enumerate(avail):
                    full[idx_map[c1], idx_map[c2]] = cov[i, j]
            cov = full

        # Ledoit-Wolf 收缩估计（简化版）
        return self._shrink_cov(cov)

    @staticmethod
    def _shrink_cov(cov: np.ndarray, shrinkage: float = 0.5) -> np.ndarray:
        """简单收缩估计：混合样本协方差和对角矩阵。"""
        n = cov.shape[0]
        target = np.eye(n) * np.trace(cov) / n
        return (1 - shrinkage) * cov + shrinkage * target

    # ── 约束 ─────────────────────────────────────────────

    def _build_constraints(
        self,
        codes: list[str],
        industry: pd.Series,
        prev_weights: pd.Series | None,
    ) -> list[dict[str, Any]]:
        """构建优化约束列表。"""
        constraints: list[dict[str, Any]] = []

        # 1. 满仓约束: sum(w) = 1.0
        constraints.append({
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        })

        # 2. 行业集中度: sum(w_j) ≤ max_industry
        industries = industry.reindex(codes)
        for ind_name in industries.unique():
            if pd.isna(ind_name):
                continue
            ind_mask = (industries == ind_name).values
            if ind_mask.sum() <= 1:
                continue
            constraints.append({
                "type": "ineq",
                "fun": lambda w, m=ind_mask: self._max_industry - w[m].sum(),
            })

        # 3. 换手率约束: sum|w - w_prev| / 2 ≤ max_turnover
        if prev_weights is not None and not prev_weights.empty:
            prev = prev_weights.reindex(codes).fillna(0).values

            constraints.append({
                "type": "ineq",
                "fun": lambda w, p=prev: self._max_turnover - np.sum(np.abs(w - p)) / 2,
            })

        return constraints

    # ── 启发式回退 ───────────────────────────────────────

    def _fallback(self, scores: pd.Series, industry: pd.Series,
                  prev_weights: pd.Series | None = None) -> pd.Series:
        """优化失败时使用启发式方法回退。"""
        w = scores.copy()
        # 按分数比例分配
        w = w.clip(0)
        total = w.sum()
        if total == 0:
            return w
        w = w / total

        # 行业约束
        for ind in industry.unique():
            if pd.isna(ind):
                continue
            mask = industry == ind
            ind_sum = w[mask].sum()
            if ind_sum > self._max_industry:
                w[mask] *= self._max_industry / ind_sum

        # 单票上限
        w = w.clip(0, self._max_single)

        # 归一化
        total = w.sum()
        if total > 0:
            w /= total

        logger.info(f"[ConvexOptimizer] 使用启发式回退, 持仓 {(w > 0.001).sum()} 只")
        return w
