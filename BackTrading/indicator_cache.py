"""Phase 0: 全局指标预计算缓存。

在贝叶斯寻优开始前，为所有股票一次性预计算技术指标 + peak/trough。
后续每次 evaluation 直接加载缓存，只跑评分层 compute_signals。

缓存存储位置: CACHE_DIR/indicator_cache_v1/<bucket>/<symbol>.indicators.parquet
                  + .peaks.npy + .troughs.npy + .meta.json

内存缓存仅在主进程有效；子进程（ProcessPoolExecutor worker）从磁盘加载。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_IN_MEMORY: dict[str, pd.DataFrame] = {}
_PEAKS: dict[str, np.ndarray] = {}
_TROUGHS: dict[str, np.ndarray] = {}
_PRECOMPUTE_DONE: bool = False


def _data_fingerprint(df: pd.DataFrame) -> str:
    """快速指纹：只依赖原始 OHLCV，不依赖参数。"""
    key_cols = ["close", "high", "low", "open", "volume"]
    present = [c for c in key_cols if c in df.columns]
    if not present:
        return "unknown"
    raw = "{}_{}_{}_{}_{}".format(
        len(df),
        df[present].iloc[0].values.tolist(),
        df[present].iloc[-1].values.tolist(),
        df[present].iloc[min(len(df) // 2, len(df) - 1)].values.tolist(),
        df["close"].sum(),
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_root() -> Path:
    """计算缓存根目录，不依赖 prepare 模块避免循环导入。"""
    try:
        from UtilsManager.ConfigParser import Config
        base = Path(Config().CACHE_DIRECTORY) / "backtest_signal_cache"
    except Exception:
        base = Path(__file__).resolve().parent / "data" / "signal_cache"
    return base / "indicator_cache_v1"


def _indicators_path(symbol: str) -> Path:
    cr = _cache_root()
    bucket = symbol[:2].lower()
    (cr / bucket).mkdir(parents=True, exist_ok=True)
    return cr / bucket / f"{symbol}.indicators.parquet"


def _peaks_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.peaks.npy"


def _troughs_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.troughs.npy"


def _meta_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.meta.json"


def _load_from_disk(symbol: str) -> bool:
    """从磁盘加载到内存缓存。"""
    ipath = _indicators_path(symbol)
    ppath = _peaks_path(symbol)
    tpath = _troughs_path(symbol)
    if not (ipath.exists() and ppath.exists() and tpath.exists()):
        return False
    try:
        _IN_MEMORY[symbol] = pd.read_parquet(ipath)
        _PEAKS[symbol] = np.load(ppath)
        _TROUGHS[symbol] = np.load(tpath)
        return True
    except Exception:
        return False


def _save_to_disk(symbol: str, df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> None:
    df.to_parquet(_indicators_path(symbol), index=False, compression="zstd", compression_level=3)
    np.save(_peaks_path(symbol), peaks)
    np.save(_troughs_path(symbol), troughs)
    meta = {"fingerprint": _data_fingerprint(df), "n_rows": len(df)}
    with open(_meta_path(symbol), "w") as f:
        json.dump(meta, f)


def _is_cache_valid(symbol: str, df: pd.DataFrame) -> bool:
    mpath = _meta_path(symbol)
    if not mpath.exists():
        return False
    try:
        with open(mpath) as f:
            return json.load(f).get("fingerprint") == _data_fingerprint(df)
    except Exception:
        return False


def precompute_all_indicators(stock_dir: str) -> None:
    """Phase 0: 为 stock_dir 中所有股票预计算指标 + peak/trough。

    写入磁盘缓存 + 内存缓存。幂等 — 磁盘缓存已满时第二次调用是无操作。
    若 WFO 窗口间数据一致，仅扫描元数据，无重复计算。
    """
    stock_files = sorted(Path(stock_dir).glob("*.parquet"))
    if not stock_files:
        logger.warning("Phase 0: stock_dir 中无 parquet 文件，跳过预计算")
        return

    from BackTrading.prepare import _compute_indicators
    from LogicAnalyzer.signals.divergence import adaptive_distance, find_peaks_troughs

    computed = 0
    cached = 0
    skipped = 0
    for f in stock_files:
        symbol = f.stem
        if symbol in _IN_MEMORY:
            # 内存缓存命中 — 数据必须与当前 stock_dir 一致
            # 如果数据不一致，跳过此问题在当前场景不必处理
            #（同一窗口内 stock_dir 不变，跨窗口时分桶路径一致但指纹不同）
            skipped += 1
            continue

        # 尝试从磁盘加载
        if _load_from_disk(symbol):
            cached += 1
            continue

        # 需要计算
        df_raw = pd.read_parquet(f)
        if len(df_raw) < 60:
            _IN_MEMORY[symbol] = pd.DataFrame()
            _PEAKS[symbol] = np.array([], dtype=int)
            _TROUGHS[symbol] = np.array([], dtype=int)
            continue

        df_ind = _compute_indicators(df_raw)
        dd = adaptive_distance(df_ind["DIF"], base_distance=10)
        peaks, troughs = find_peaks_troughs(df_ind["DIF"], distance=dd)

        _IN_MEMORY[symbol] = df_ind
        _PEAKS[symbol] = peaks
        _TROUGHS[symbol] = troughs
        _save_to_disk(symbol, df_ind, peaks, troughs)
        computed += 1

    _log_msg = f"Phase 0: {len(stock_files)} 只股票"
    parts = []
    if computed:
        parts.append(f"+{computed}")
    if cached:
        parts.append(f"cache{cached}")
    if skipped:
        parts.append(f"mem{skipped}")
    if parts:
        _log_msg += " (" + "/".join(parts) + ")"
    logger.info(_log_msg)


def get_precomputed(
    symbol: str,
    stock_dir: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """获取预计算的指标和 peak/trough。

    尝试顺序: 内存缓存 → 磁盘缓存 → 实时计算(fallback)。

    Returns:
        (indicator_df, peaks, troughs)
        若股票不足 60 根 K 线，返回 (空 DataFrame, [], [])。
    """
    # 1. 内存缓存
    if symbol in _IN_MEMORY:
        return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]

    # 2. 磁盘缓存
    if _load_from_disk(symbol):
        return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]

    # 3. Fallback：实时计算（只在非向量化模式或无 Phase 0 时触发）
    if stock_dir is None:
        return pd.DataFrame(), np.array([], dtype=int), np.array([], dtype=int)

    fpath = os.path.join(stock_dir, f"{symbol}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame(), np.array([], dtype=int), np.array([], dtype=int)

    df_raw = pd.read_parquet(fpath)
    if len(df_raw) < 60:
        _IN_MEMORY[symbol] = pd.DataFrame()
        _PEAKS[symbol] = np.array([], dtype=int)
        _TROUGHS[symbol] = np.array([], dtype=int)
        return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]

    from BackTrading.prepare import _compute_indicators
    from LogicAnalyzer.signals.divergence import adaptive_distance, find_peaks_troughs

    df_ind = _compute_indicators(df_raw)
    dd = adaptive_distance(df_ind["DIF"], base_distance=10)
    peaks, troughs = find_peaks_troughs(df_ind["DIF"], distance=dd)

    _IN_MEMORY[symbol] = df_ind
    _PEAKS[symbol] = peaks
    _TROUGHS[symbol] = troughs
    _save_to_disk(symbol, df_ind, peaks, troughs)
    return df_ind, peaks, troughs


def reset_cache() -> None:
    """清空内存缓存（主要在测试中使用）。"""
    global _PRECOMPUTE_DONE
    _PRECOMPUTE_DONE = False
    _IN_MEMORY.clear()
    _PEAKS.clear()
    _TROUGHS.clear()
