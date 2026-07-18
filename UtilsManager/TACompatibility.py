"""
TACompatibility — pandas_ta replacement using TA-Lib (talib).
Provides the same function signatures and DataFrame accessor (.ta) as pandas_ta.
Works on Windows (no posix module dependency).

NOTE: TA-Lib C extension is NOT thread-safe. A module-level lock serializes
all talib calls to prevent access-violation crashes in multi-threaded usage.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import talib

_talib_lock = threading.Lock()


def _talib_call(func, *args, **kwargs):
    """Thread-safe wrapper for any talib function call."""
    with _talib_lock:
        return func(*args, **kwargs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    result = _talib_call(
        talib.ATR,
        high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float),
        timeperiod=length,
    )
    return pd.Series(result, index=close.index, name=f"ATR_{length}")


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    result = _talib_call(
        talib.ADX,
        high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float),
        timeperiod=length,
    )
    return pd.DataFrame({f"ADX_{length}": result}, index=close.index)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    result = _talib_call(talib.RSI, close.to_numpy(dtype=float), timeperiod=length)
    return pd.Series(result, index=close.index, name=f"RSI_{length}")


def stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 9, d: int = 3,
) -> pd.DataFrame:
    slowk, slowd = _talib_call(talib.STOCH,
        high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float),
        fastk_period=k, slowk_period=3, slowk_matype=0,
        slowd_period=d, slowd_matype=0,
    )
    suffix = f"{k}_{d}_3"
    return pd.DataFrame(
        {f"STOCHk_{suffix}": slowk, f"STOCHd_{suffix}": slowd},
        index=close.index,
    )


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    result = _talib_call(talib.CCI,
        high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float),
        timeperiod=length,
    )
    return pd.Series(result, index=close.index, name=f"CCI_{length}")


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    upper, middle, lower = _talib_call(talib.BBANDS,
        close.to_numpy(dtype=float),
        timeperiod=length, nbdevup=std, nbdevdn=std,
    )
    std_str = f"{std:.1f}"
    return pd.DataFrame(
        {
            f"BBU_{length}_{std_str}": upper,
            f"BBM_{length}_{std_str}": middle,
            f"BBL_{length}_{std_str}": lower,
        },
        index=close.index,
    )


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    macd_val, macdsignal, macdhist = _talib_call(talib.MACD,
        close.to_numpy(dtype=float),
        fastperiod=fast, slowperiod=slow, signalperiod=signal,
    )
    return pd.DataFrame(
        {
            f"MACD_{fast}_{slow}_{signal}": macd_val,
            f"MACDs_{fast}_{slow}_{signal}": macdsignal,
            f"MACDh_{fast}_{slow}_{signal}": macdhist,
        },
        index=close.index,
    )


def cdl_pattern(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, name: str = "all") -> pd.DataFrame | None:
    o = open_.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)

    if name == "all":
        patterns: dict[str, np.ndarray] = {}
        with _talib_lock:
            for func_name in sorted(dir(talib)):
                if func_name.startswith("CDL"):
                    func = getattr(talib, func_name)
                    if callable(func):
                        patterns[func_name] = func(o, h, l, c)
        if not patterns:
            return None
        return pd.DataFrame(patterns, index=close.index)

    func_name = f"CDL{name.upper()}"
    func = getattr(talib, func_name, None)
    if func is None:
        return None
    return pd.DataFrame({func_name: _talib_call(func, o, h, l, c)}, index=close.index)


# ── pandas .ta accessor ──────────────────────────────────────────────

@pd.api.extensions.register_dataframe_accessor("ta")
class _TAAccessor:
    """Mimics pandas_ta's ``df.ta.*`` accessor using TA-Lib under the hood."""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj

    def rsi(
        self, append: bool = True, close: str = "close", length: int = 14,
    ) -> pd.Series | None:
        result = rsi(self._obj[close], length=length)
        if append:
            self._obj[result.name] = result
            return None
        return result

    def stoch(
        self, append: bool = True, close: str = "close",
        high: str = "high", low: str = "low", k: int = 9, d: int = 3,
    ) -> pd.DataFrame | None:
        result = stoch(self._obj[high], self._obj[low], self._obj[close], k=k, d=d)
        if append:
            for col in result.columns:
                self._obj[col] = result[col]
            return None
        return result

    def cci(
        self, append: bool = True, close: str = "close",
        high: str = "high", low: str = "low", length: int = 20,
    ) -> pd.Series | None:
        result = cci(self._obj[high], self._obj[low], self._obj[close], length=length)
        if append:
            self._obj[result.name] = result
            return None
        return result

    def bbands(
        self, append: bool = True, close: str = "close",
        length: int = 20, std: float = 2.0,
    ) -> pd.DataFrame | None:
        result = bbands(self._obj[close], length=length, std=std)
        if append:
            for col in result.columns:
                self._obj[col] = result[col]
            return None
        return result
