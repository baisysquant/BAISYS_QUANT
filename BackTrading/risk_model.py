from __future__ import annotations


import numpy as np
import pandas as pd


def compute_industry_exposure(
    portfolio_weights: pd.Series,
    industry_labels: pd.Series,
) -> dict[str, float]:
    """计算组合的行业暴露（各行业权重之和）。"""
    df = pd.DataFrame({"weight": portfolio_weights, "industry": industry_labels})
    return df.groupby("industry")["weight"].sum().to_dict()


def industry_hhi(industry_exposure: dict[str, float]) -> float:
    """行业 HHI 集中度（Herfindahl Index），越高越集中。"""
    weights = np.array(list(industry_exposure.values()))
    return float((weights ** 2).sum())
