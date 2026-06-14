import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """60 个交易日的模拟 OHLCV 数据（上升趋势后回落）。"""
    np.random.seed(42)
    n = 60
    close = 10.0 * (1 + np.linspace(0, 0.3, n)) + np.random.randn(n) * 0.2
    close = np.maximum(close, 5.0)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_p = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1_000_000, 10_000_000, n)
    amount = volume * close
    return pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "symbol": "sh600000",
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    })


@pytest.fixture
def sample_ohlcv_with_indicators(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """带 MACD / ATR / MA 的 K 线数据。"""
    df = sample_ohlcv.copy()
    close = df["close"]
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = 2 * (df["DIF"] - df["DEA"])
    high_low = df["high"] - df["low"]
    df["ATR"] = high_low.rolling(14).mean()
    for p in [5, 10, 20, 30, 60]:
        df[f"MA_{p}"] = df["close"].rolling(p).mean()
    return df


@pytest.fixture
def sample_boll_bandwidth(sample_ohlcv_with_indicators: pd.DataFrame) -> pd.DataFrame:
    """带布林带宽列的 K 线数据。"""
    df = sample_ohlcv_with_indicators.copy()
    close = df["close"]
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BBU_20_2.0"] = ma20 + 2 * std20
    df["BBM_20_2.0"] = ma20
    df["BBL_20_2.0"] = ma20 - 2 * std20
    df["BOLL_BANDWIDTH"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / close
    return df
