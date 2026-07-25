from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor


def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    best_f: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement 采集函数。

    EI(x) = (μ - f* - ξ)·Φ(z) + σ·φ(z),  z = (μ - f* - ξ) / σ

    Args:
        X: (n, d) 候选点。
        gp: 已拟合的 GP 模型。
        best_f: 当前最优观测值。
        xi: 探索-利用平衡参数 (越大越探索)。

    Returns:
        (n,) EI 值。
    """
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-12)

    imp = mu - best_f - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def probability_of_improvement(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    best_f: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Probability of Improvement.

    PI(x) = Φ((μ - f* - ξ) / σ)
    """
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    return norm.cdf((mu - best_f - xi) / sigma)


def upper_confidence_bound(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    beta: float = 2.0,
) -> np.ndarray:
    """Upper Confidence Bound (UCB).

    UCB(x) = μ + β·σ
    """
    mu, sigma = gp.predict(X, return_std=True)
    return mu + beta * sigma


def dsr_penalty(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    lambda_: float = 0.05,
    eps: float = 1e-8,
) -> np.ndarray:
    """DSR 惩罚项: λ · σ / (|μ| + ε)

    在高不确定性(σ 大)或低预测收益(μ 接近 0)的区域施加惩罚，
    防止采集函数过度探索无效区域。
    """
    mu, sigma = gp.predict(X, return_std=True)
    return lambda_ * sigma / (np.abs(mu) + eps)


def mixed_acquisition(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    best_f: float,
    xi: float = 0.01,
    dsr_lambda: float = 0.05,
) -> np.ndarray:
    """混合采集函数: EI(x) - DSR_penalty(x)

    既追求高期望改进，又抑制高不确定性/低收益区域。
    """
    ei = expected_improvement(X, gp, best_f, xi)
    penalty = dsr_penalty(X, gp, dsr_lambda)
    return ei - penalty


def optimize_acquisition(
    gp: GaussianProcessRegressor,
    bounds: np.ndarray,
    best_f: float,
    n_restarts: int = 10,
    xi: float = 0.01,
    dsr_lambda: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """最大化采集函数，使用多起点 L-BFGS-B。

    Args:
        gp: 已拟合的 GP。
        bounds: (n_dims, 2) 归一化边界 [[0,1], ...]。
        best_f: 当前最优观测值。
        n_restarts: 随机重启次数。
        xi: EI 探索参数。
        dsr_lambda: DSR 惩罚系数。

    Returns:
        (best_x, best_acq_value): best_x 形状 (n_dims,)。
    """
    from scipy.optimize import minimize

    n_dims = bounds.shape[0]
    rng = np.random.RandomState(random_state)

    best_x = None
    best_val = -np.inf

    # 目标函数（负采集函数，因为 minimize）
    def neg_acq(x: np.ndarray) -> float:
        x = x.reshape(1, -1)
        val = mixed_acquisition(x, gp, best_f, xi, dsr_lambda)[0]
        return -val

    # 多起点优化
    for i in range(n_restarts):
        # 起点: 均匀随机 [0, 1]
        x0 = rng.uniform(0, 1, size=n_dims)

        result = minimize(
            neg_acq,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        if result.success and -result.fun > best_val:
            best_val = -result.fun
            best_x = result.x

    # 如果所有优化都失败，返回起点中最好的那个
    if best_x is None:
        logger = __import__("loguru").logger
        logger.warning("acquisition 优化失败，返回随机点")
        best_x = rng.uniform(0, 1, size=n_dims)
        best_val = float(mixed_acquisition(best_x.reshape(1, -1), gp, best_f, xi, dsr_lambda)[0])

    return best_x, best_val
