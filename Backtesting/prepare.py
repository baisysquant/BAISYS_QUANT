from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from ConfigParser import Config
from LogicAnalyzer.pipeline import MACDAnalyzer


def prepare_backtest_data(
    kline_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """预计算回测信号列。

    按股票分组，对每只股票逐日滚动计算技术指标和管线信号，
    输出合并后的 DataFrame，供 ``QuantPipelineStrategy.on_bar`` 读取。

    Args:
        kline_df: 多股票 K 线 DataFrame。
                  必须含列: symbol, trade_date, open, high, low, close, volume, amount。
        params: 集中参数配置（透传给 pipeline_analysis），
                为空时自动从 Config 加载。

    Returns:
        追加信号列的 DataFrame：
            进场评分, 退出评分, 风险等级, 止损价, 综合评分, score 等。
    """
    if params is None:
        cfg = Config()
        params = _build_params(cfg)

    analyzer = MACDAnalyzer()
    result_rows: list[dict[str, Any]] = []

    from tqdm import tqdm

    symbols = kline_df["symbol"].unique()
    for sym in tqdm(symbols, desc="预计算信号", unit="只", ncols=80):
        stock_df = kline_df[kline_df["symbol"] == sym].sort_values("trade_date").copy()
        if len(stock_df) < 60:
            continue
        try:
            _compute_indicators(stock_df)
        except Exception:
            continue

        for i in range(len(stock_df)):
            bar = stock_df.iloc[: i + 1].reset_index(drop=True)
            try:
                signal = _compute_signal(analyzer, bar, params)
            except Exception:
                continue
            result_rows.append({
                "symbol": sym,
                "trade_date": bar["trade_date"].iloc[-1],
                "进场评分": float(signal.get("进场评分", 0)),
                "退出评分": float(signal.get("退出评分", 0)),
                "风险等级": str(signal.get("风险等级", "LOW")),
                "止损价": float(signal.get("止损价", 0) or 0),
                "综合评分": float(signal.get("score", 0)),
            })

    if not result_rows:
        return kline_df

    signal_df = pd.DataFrame(result_rows)
    result = kline_df.merge(
        signal_df, on=["symbol", "trade_date"], how="left"
    )
    for col in ["进场评分", "退出评分", "综合评分", "止损价"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    result["风险等级"] = result["风险等级"].fillna("LOW")
    return result


def _compute_indicators(df: pd.DataFrame) -> None:
    """计算技术指标（就地修改 df）。"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["DIF"] = macd.iloc[:, 0].values if macd.shape[1] >= 1 else 0
        df["DEA"] = macd.iloc[:, 1].values if macd.shape[1] >= 2 else 0
        hist = macd.iloc[:, 2].values if macd.shape[1] >= 3 else 0
        df["MACD_HIST"] = 2 * hist if isinstance(hist, np.ndarray) else hist
    else:
        df["DIF"] = 0.0
        df["DEA"] = 0.0
        df["MACD_HIST"] = 0.0

    atr_series = ta.atr(high, low, close, length=14)
    df["ATR"] = atr_series if atr_series is not None else 0.0

    # ScoringRules 依赖的金叉/死叉信号列
    if "DIF" in df.columns and "DEA" in df.columns:
        dif = df["DIF"]
        dea = df["DEA"]
        prev_dif = dif.shift(1).fillna(dea.shift(1).fillna(0))
        prev_dea = dea.shift(1).fillna(0)
        golden = (dif > dea) & (prev_dif <= prev_dea)
        dead = (dif < dea) & (prev_dif >= prev_dea)
        df["MACD_SIGNAL_DETAIL"] = None
        df.loc[golden, "MACD_SIGNAL_DETAIL"] = "金叉"
        df.loc[dead, "MACD_SIGNAL_DETAIL"] = "死叉"
        df.loc[~(golden | dead), "MACD_SIGNAL_DETAIL"] = dif[~(golden | dead)].apply(
            lambda v: "多头" if v > 0 else "空头"
        )
        df["MACD_CROSS"] = 0
        df.loc[golden, "MACD_CROSS"] = 1
        df.loc[dead, "MACD_CROSS"] = -1

    for p in (5, 10, 20, 30, 60):
        df[f"MA_{p}"] = close.rolling(p).mean()

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and bb.shape[1] >= 3:
        df["BBU_20_2.0"] = bb.iloc[:, 0].values
        df["BBM_20_2.0"] = bb.iloc[:, 1].values
        df["BBL_20_2.0"] = bb.iloc[:, 2].values
        df["BOLL_BANDWIDTH"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / close


def _compute_signal(
    analyzer: MACDAnalyzer,
    bar: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """对单根 K 线（含历史）计算管线信号。"""
    result = analyzer.pipeline_analysis(bar, params=params)
    exit_strategy = result.get("exit_strategy", {})
    risk_level = result.get("risk_level", "LOW")

    entry_score = float(result.get("score", 0))
    exit_score = _calc_exit_score(bar, exit_strategy, risk_level)
    return {
        "进场评分": entry_score,
        "退出评分": exit_score,
        "风险等级": risk_level,
        "止损价": exit_strategy.get("stop_loss", 0),
        "score": entry_score,
    }


def _calc_exit_score(
    df: pd.DataFrame,
    exit_strategy: dict[str, Any],
    risk_level: str,
) -> float:
    """计算退出评分。"""
    if risk_level in ("HIGH", "D"):
        return 100.0
    stop_loss = exit_strategy.get("stop_loss")
    if stop_loss and len(df) > 0:
        close = df["close"].iloc[-1]
        if close < stop_loss:
            return 90.0
    return 0.0


def _build_params(cfg: Config) -> dict[str, Any]:
    """从 Config 构建 pipeline_analysis 需要的参数字典。"""
    ac = cfg.app_config
    return {
        "regime": ac.regime_detection.model_dump(),
        "divergence": ac.divergence.model_dump(),
        "scoring": ac.scoring_params.model_dump(),
        "technical": ac.technical_constants.model_dump(),
    }
