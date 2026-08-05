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
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_IN_MEMORY: dict[str, pd.DataFrame] = {}
_PEAKS: dict[str, np.ndarray] = {}
_TROUGHS: dict[str, np.ndarray] = {}
_PRECOMPUTE_DONE: bool = False
# 背离检测缓存：div_type/div_idx/div_strength 只依赖股票自身 DIF 数据，与参数无关，
# 在 Phase 0 一次性预计算，后续每次 evaluation（不同参数组合）直接复用，
# 避免每轮贝叶斯迭代都重跑 O(n²) 的逐 bar Python 循环。
_DIVERGENCE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


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


# ── 背离检测缓存（参数无关，Phase 0 预计算） ──────────────────────

def _divergence_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.divergence.npz"


def precompute_divergence(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """因果逐 bar 背离检测（与 vectorized_signal._divergence_scores 完全一致）。

    结果仅依赖 DIF 数据，与任何评分参数无关，可安全跨参数迭代复用。
    """
    from BackTrading.vectorized_signal import _divergence_scores
    from LogicAnalyzer.signals.divergence import adaptive_distance

    _dd = adaptive_distance(df["DIF"], base_distance=10) if "DIF" in df.columns else 11
    return _divergence_scores(df, base_distance=_dd)


def _load_divergence_from_disk(symbol: str) -> bool:
    p = _divergence_path(symbol)
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=True) as z:
            _DIVERGENCE[symbol] = (
                z["div_type"], z["div_idx"], z["div_strength"],
            )
        return True
    except Exception:
        return False


def _save_divergence_to_disk(symbol: str, div: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    p = _divergence_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, div_type=div[0], div_idx=div[1], div_strength=div[2])


def get_divergence(
    symbol: str,
    df: pd.DataFrame | None = None,
    stock_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """获取预计算的背离检测结果（内存 → 磁盘 → 实时计算兜底）。

    与 get_precomputed 的 fallback 语义一致；df 为空时仅查缓存。
    计算失败时返回 None，由调用方回退到逐 bar 实时计算。
    """
    if symbol in _DIVERGENCE:
        return _DIVERGENCE[symbol]
    if _load_divergence_from_disk(symbol):
        return _DIVERGENCE[symbol]
    if df is None or df.empty:
        return None
    try:
        div = precompute_divergence(df)
    except Exception:
        return None
    _DIVERGENCE[symbol] = div
    try:
        _save_divergence_to_disk(symbol, div)
    except Exception:
        pass
    return div


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


def _precompute_divergences_parallel(symbols: list[str]) -> None:
    """并行预计算背离检测（背离仅依赖 DIF，与参数无关，跨迭代复用）。"""
    from concurrent.futures import ThreadPoolExecutor

    workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(get_divergence, s, _IN_MEMORY[s]) for s in symbols]
        for fut in futures:
            try:
                fut.result()
            except Exception:
                pass


def precompute_all_indicators(stock_dir: str) -> None:
    """Phase 0: 为 stock_dir 中所有股票预计算技术指标（不含 peaks/troughs）。

    peaks/troughs 改为在 _divergence_scores 中滚动计算以避免未来函数。
    背离检测与参数无关，在此并行预计算并落盘（内存 → 磁盘 → 实时计算三级缓存）。
    写入磁盘缓存 + 内存缓存。幂等。
    """
    stock_files = sorted(Path(stock_dir).glob("*.parquet"))
    if not stock_files:
        logger.warning("Phase 0: stock_dir 中无 parquet 文件，跳过预计算")
        return

    from BackTrading.prepare import _compute_indicators

    computed = 0
    cached = 0
    skipped = 0
    for f in stock_files:
        symbol = f.stem
        if symbol in _IN_MEMORY:
            skipped += 1
            continue

        if _load_from_disk(symbol):
            cached += 1
            continue

        df_raw = pd.read_parquet(f)
        if len(df_raw) < 60:
            _IN_MEMORY[symbol] = pd.DataFrame()
            _PEAKS[symbol] = np.array([], dtype=int)
            _TROUGHS[symbol] = np.array([], dtype=int)
            continue

        df_ind = _compute_indicators(df_raw)

        _IN_MEMORY[symbol] = df_ind
        _PEAKS[symbol] = np.array([], dtype=int)
        _TROUGHS[symbol] = np.array([], dtype=int)
        _save_to_disk(symbol, df_ind, np.array([], dtype=int), np.array([], dtype=int))
        computed += 1

    # 背离检测结果只依赖 DIF、与评分参数无关：Phase 0 并行预计算并落盘，
    # 后续每轮贝叶斯迭代（不同参数）直接复用，避免重跑 O(n²) 逐 bar 循环。
    # 缺失背离的股票（含磁盘加载 / 长 K 线股票）一次补齐；已缓存的零开销跳过。
    missing = [
        s for s in _IN_MEMORY
        if s not in _DIVERGENCE and len(_IN_MEMORY[s]) >= 60
    ]
    if missing:
        _t0 = time.perf_counter()
        _precompute_divergences_parallel(missing)
        logger.info(
            f"Phase 0: 背离预计算 {len(missing)} 只耗时 {time.perf_counter() - _t0:.1f}s"
        )

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

    df_ind = _compute_indicators(df_raw)

    _IN_MEMORY[symbol] = df_ind
    _PEAKS[symbol] = np.array([], dtype=int)
    _TROUGHS[symbol] = np.array([], dtype=int)
    _save_to_disk(symbol, df_ind, np.array([], dtype=int), np.array([], dtype=int))
    return df_ind, np.array([], dtype=int), np.array([], dtype=int)
