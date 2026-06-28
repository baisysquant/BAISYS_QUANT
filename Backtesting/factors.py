from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger


# ── 因子计算 ──────────────────────────────────────────────────────────────

FACTOR_REGISTRY: dict[str, dict[str, Any]] = {}


def register_factor(name: str, description: str, min_periods: int = 20) -> Callable:
    def decorator(func: Callable) -> Callable:
        FACTOR_REGISTRY[name] = {
            "func": func,
            "description": description,
            "min_periods": min_periods,
        }
        return func
    return decorator


@register_factor("momentum_1m", "过去 21 个交易日收益（动量）", min_periods=21)
def momentum_1m(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    ret = close.pct_change(21)
    return ret


@register_factor("momentum_3m", "过去 63 个交易日收益", min_periods=63)
def momentum_3m(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(63)


@register_factor("momentum_6m", "过去 126 个交易日收益", min_periods=126)
def momentum_6m(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(126)


@register_factor("volatility_1m", "过去 21 个交易日波动率", min_periods=21)
def volatility_1m(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change()
    return ret.rolling(21).std()


@register_factor("volume_ratio", "过去 5 日成交量 / 过去 20 日均量", min_periods=20)
def volume_ratio(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    avg = vol.rolling(20).mean()
    return vol.rolling(5).mean() / avg.replace(0, np.nan)


@register_factor("turnover", "成交额 / 市值 (proxy)", min_periods=1)
def turnover(df: pd.DataFrame) -> pd.Series:
    return df["amount"] / (df["close"] * df["volume"]).replace(0, np.nan)


@register_factor("close_to_high", "收盘价 / 过去 20 日最高价", min_periods=20)
def close_to_high(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["high"].rolling(20).max()


@register_factor("close_to_low", "收盘价 / 过去 20 日最低价", min_periods=20)
def close_to_low(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["low"].rolling(20).min()


@register_factor("ma_divergence_5_20", "(MA5 - MA20) / MA20", min_periods=20)
def ma_divergence_5_20(df: pd.DataFrame) -> pd.Series:
    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    return (ma5 - ma20) / ma20.replace(0, np.nan)


@register_factor("atr_ratio", "ATR(14) / 收盘价", min_periods=14)
def atr_ratio(df: pd.DataFrame) -> pd.Series:
    import pandas_ta as ta
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    return atr / df["close"]


# ── 因子计算主入口 ──────────────────────────────────────────────────────────

def compute_factors(
    kline_df: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """对全量 K 线计算指定因子，返回原 DataFrame 附加因子列。"""
    result = kline_df.copy()
    names = factor_names or list(FACTOR_REGISTRY.keys())

    for sym, grp in kline_df.groupby("symbol"):
        grp = grp.sort_values("trade_date")
        mask = result["symbol"] == sym
        for fname in names:
            info = FACTOR_REGISTRY.get(fname)
            if info is None:
                continue
            if len(grp) < info["min_periods"]:
                continue
            try:
                series = info["func"](grp)
                result.loc[mask, fname] = series.values
            except Exception:
                continue
    return result


# ── IC / IR 分析 ───────────────────────────────────────────────────────────

def compute_ic(
    factor_df: pd.DataFrame,
    forward_days: int = 5,
    factor_col: str = "momentum_1m",
) -> pd.DataFrame:
    """计算截面 IC：每个交易日 factor 值与未来 forward_days 收益的 Spearman 秩相关。

    Returns:
        DataFrame with columns: [trade_date, ic, p_value]
    """
    df = factor_df.copy()
    df["_fwd_ret"] = df.groupby("symbol")["close"].transform(
        lambda x: x.pct_change(periods=forward_days).shift(-forward_days)
    )
    df = df.dropna(subset=[factor_col, "_fwd_ret"])

    results: list[dict[str, Any]] = []
    for dt, grp in df.groupby("trade_date"):
        if len(grp) < 30:
            continue
        f = grp[factor_col].values
        r = grp["_fwd_ret"].values
        from scipy.stats import spearmanr
        stat, _ = spearmanr(f, r)
        results.append({"trade_date": dt, "ic": stat, "factor": factor_col})

    return pd.DataFrame(results)


def compute_ir(ic_df: pd.DataFrame) -> dict[str, float]:
    """计算 IR = mean(IC) / std(IC)。"""
    if ic_df.empty:
        return {"mean_ic": 0, "std_ic": 0, "ir": 0}
    ic = ic_df["ic"].dropna()
    if len(ic) < 2:
        return {"mean_ic": float(ic.mean()), "std_ic": 0, "ir": 0}
    mu = ic.mean()
    sigma = ic.std()
    return {"mean_ic": round(float(mu), 6), "std_ic": round(float(sigma), 6),
            "ir": round(float(mu / sigma), 4) if sigma > 0 else 0}


def ic_heatmap(
    factor_df: pd.DataFrame,
    factor_names: list[str] | None = None,
    forward_days_list: list[int] | None = None,
) -> pd.DataFrame:
    """多因子 × 多持有期的 IC 热力图。"""
    names = factor_names or list(FACTOR_REGISTRY.keys())
    fwd = forward_days_list or [1, 5, 10, 21, 63]

    rows: list[dict[str, Any]] = []
    for fname in names:
        for d in fwd:
            ic_df = compute_ic(factor_df, forward_days=d, factor_col=fname)
            ir_info = compute_ir(ic_df)
            rows.append({
                "factor": fname,
                "forward_days": d,
                "mean_ic": ir_info["mean_ic"],
                "ir": ir_info["ir"],
                "ic_std": ir_info["std_ic"],
            })
    return pd.DataFrame(rows)


# ── 因子组合构建 ───────────────────────────────────────────────────────────

def build_factor_portfolio(
    factor_df: pd.DataFrame,
    factor_col: str = "momentum_1m",
    n_groups: int = 10,
    long_short: bool = True,
) -> pd.DataFrame:
    """在每个截面按因子值分组，构建多空组合收益序列。

    Returns:
        DataFrame: [trade_date, group, avg_return, cum_return]
    """
    df = factor_df.copy()
    df["_fwd_ret"] = df.groupby("symbol")["close"].transform(
        lambda x: x.pct_change().shift(-1)
    )
    df = df.dropna(subset=[factor_col, "_fwd_ret"])

    results: list[dict[str, Any]] = []
    for dt, grp in df.groupby("trade_date"):
        if len(grp) < 2 * n_groups:
            continue
        grp = grp.copy()
        grp["group"] = pd.qcut(grp[factor_col].rank(method="first"), q=n_groups,
                                labels=[f"G{i+1}" for i in range(n_groups)])
        for g, gdata in grp.groupby("group"):
            ret = gdata["_fwd_ret"].mean()
            results.append({"trade_date": dt, "group": g, "return": ret,
                            "factor": factor_col})

    result_df = pd.DataFrame(results)
    if long_short and not result_df.empty:
        groups = result_df["group"].unique()
        top = max(groups)
        bottom = min(groups)
        ls = result_df[result_df["group"].isin([top, bottom])].pivot_table(
            index="trade_date", columns="group", values="return"
        )
        if len(ls.columns) == 2:
            ls["long_short"] = ls[top] - ls[bottom]
            ls["cum_ls"] = (1 + ls["long_short"]).cumprod()
            ls["factor"] = factor_col
            return ls.reset_index()
    return result_df


def list_factors() -> pd.DataFrame:
    """列出所有已注册因子。"""
    return pd.DataFrame([
        {"name": k, "description": v["description"], "min_periods": v["min_periods"]}
        for k, v in FACTOR_REGISTRY.items()
    ])
