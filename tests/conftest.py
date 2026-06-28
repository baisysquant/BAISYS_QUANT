from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ── 测试分层标记 ──
def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "unit: 纯单元测试，无外部依赖")
    config.addinivalue_line("markers", "integration: 集成测试，需要 DB / IO")
    config.addinivalue_line("markers", "e2e: 端到端测试，需要网络和金融数据 API")


@pytest.fixture
def temp_config_ini() -> Path:
    """创建最小配置文件和临时目录供测试使用。"""
    base = Path(tempfile.mkdtemp(prefix="baisys_test_"))
    cfg = base / "config.ini"
    cfg.write_text(
        "[DATABASE]\n"
        "USER = test_user\n"
        "PASSWORD = test_pass\n"
        "HOST = localhost\n"
        "PORT = 5432\n"
        "DB_NAME = test_db\n"
        "\n"
        "[SYSTEM]\n"
        "HOME_DIRECTORY = ~/test_baisys\n"
        "TEMP_DATA_DIR = /tmp/baisys_test\n"
        "\n"
        "[LOGGING]\n"
        "LOG_LEVEL = DEBUG\n"
        "\n"
        "[MULTI_HEAD_ARRANGEMENT]\n"
        "FULL_BULL_THRESHOLD = 80\n"
        "MOVING_AVERAGE_PERIODS = 10,20,30\n"
        "\n"
        "[FILTER_RULES]\n"
        "ENABLE_WEAK_STOCK_FILTER = true\n"
        "EXEMPT_LEVELS = 完全主升,趋势加速\n"
        "\n"
        "[FUND_FLOW]\n"
        "FUND_FLOW_PERIODS = 5,10,20\n"
        "\n"
        "[TECHNICAL_INDICATORS]\n"
        "MACD_PARAMS = 12,26,9\n"
        "\n"
        "[COLUMN_ALIASES]\n"
        "CODE_ALIASES = 股票代码=ts_code\n"
        "\n"
        "[FULL_BULL_SCORING]\n"
        "WEIGHT_ZERO_AXIS = 20\n"
        "\n"
        "[ASHAREHUB]\n"
        "API_KEY = dummy_key\n"
        "\n"
        "[BACKTEST]\n"
        "ENABLED = true\n"
        "OPTIMIZE_FREQUENCY = monthly\n"
        "BACKTEST_START_DATE = 20200101\n"
        "OUT_OF_SAMPLE_DAYS = 20\n"
        "INITIAL_CASH = 1000000\n"
        "\n"
        "[BACKTEST_CALIBRATED]\n"
        "atr_stop_mult = 1.5\n"
        "kelly_fraction = 0.25\n"
        "position_a = 0.3\n"
        "boll_narrow_ratio = 0.8\n"
        "cross_decay_days = 30\n"
        "liq_veto_ratio = 0.05\n"
        "atr_t1_mult = 3.0\n"
        , encoding="utf-8"
    )
    yield cfg
    import shutil
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def temp_cache_dir() -> Path:
    """临时缓存目录。"""
    d = Path(tempfile.mkdtemp(prefix="baisys_cache_"))
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_adj_factor_data() -> pd.DataFrame:
    """模拟含复权因子的 K 线数据。"""
    return pd.DataFrame({
        "symbol": ["sh600000", "sh600000"],
        "trade_date": ["2024-06-27", "2024-06-28"],
        "adj_factor": [1.0, 1.05],
        "close": [10.0, 9.8],
        "close_normal": [10.0, 10.29],
    })


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
