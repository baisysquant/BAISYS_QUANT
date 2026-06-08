from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BollingerResult:
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series
    current_percent_b: float
    current_bandwidth: float
    bandwidth_pctile: float
    squeeze: bool
    zone: str
    signal: str


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std

    bandwidth = ((upper - lower) / sma).replace([np.inf, -np.inf], np.nan)
    percent_b = ((close - lower) / (upper - lower).replace(0, np.nan)).clip(0, 1)

    _bw = bandwidth.dropna()
    bandwidth_pctile = float(_bw.rank(pct=True).iloc[-1]) if len(_bw) > 1 else 0.5
    squeeze = bandwidth_pctile < 0.10 if len(_bw) > 20 else False

    _close = close.dropna()
    _upper = upper.dropna()
    _lower = lower.dropna()

    if _close.iloc[-1] > _upper.iloc[-1]:
        zone = "overbought"
        signal = "突破上轨"
    elif _close.iloc[-1] < _lower.iloc[-1]:
        zone = "oversold"
        signal = "跌破下轨"
    else:
        zone = "neutral"
        if squeeze:
            signal = "低波/缩口"
        elif _close.iloc[-5:].gt(_upper.iloc[-5:]).any():
            signal = "沿上轨运行"
        elif _close.iloc[-5:].lt(_lower.iloc[-5:]).any():
            signal = "沿下轨运行"
        else:
            signal = "常态波动"

    return BollingerResult(
        upper=upper,
        middle=sma,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b,
        current_percent_b=round(float(percent_b.dropna().iloc[-1]), 3),
        current_bandwidth=round(float(bandwidth.dropna().iloc[-1]), 4),
        bandwidth_pctile=round(bandwidth_pctile, 3),
        squeeze=squeeze,
        zone=zone,
        signal=signal,
    )
