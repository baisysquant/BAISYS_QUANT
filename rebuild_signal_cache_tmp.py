# -*- coding: utf-8 -*-
"""一次性脚本：重建信号缓存（新权重生效，供 A1 验证 IC）。"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"E:\BAISYS")
sys.path.insert(0, str(PROJECT_ROOT))

from UtilsManager.ConfigParser import Config  # noqa: E402
from DataManager.DbEngine import get_engine  # noqa: E402
from BackTrading.runner import _resolve_symbols, _fetch_kline  # noqa: E402
from BackTrading.prepare import prepare_backtest_data, _build_params  # noqa: E402

cfg = Config()
bt = cfg.app_config.backtest
engine = get_engine(cfg)
symbols = _resolve_symbols(engine, cfg)
print(f"[rebuild] symbols={len(symbols)}", flush=True)
kline, _delisted = _fetch_kline(engine, symbols, bt.BACKTEST_START_DATE)
print(f"[rebuild] kline rows={len(kline)}, cols={list(kline.columns)[:15]}", flush=True)
params = _build_params(cfg)
t0 = time.time()
prepared = prepare_backtest_data(
    kline, params=params, compute_exit_strategy=True, vectorized=True,
    backtest_start_date=bt.BACKTEST_START_DATE,
)
print(f"[rebuild] done in {time.time()-t0:.1f}s, rows={len(prepared)}", flush=True)
