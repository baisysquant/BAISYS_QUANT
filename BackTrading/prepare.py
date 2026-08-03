from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import date
from pathlib import Path
from typing import Any

from DataCollection.CalendarManager import TradingCalendarAnalyzer

import numpy as np
import pandas as pd
from UtilsManager import TACompatibility as ta
from loguru import logger

from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
from LogicAnalyzer.ml.signal_model import apply_ml_signal

# 向量化信号引擎（延迟导入，在导入时解析以避免 Windows spawn 锁）
try:
    from BackTrading.vectorized_signal import compute_signals as _compute_signals_vec
    _HAVE_VECTORIZED = True
except ImportError:
    _HAVE_VECTORIZED = False
    _compute_signals_vec = None

try:
    from UtilsManager.ConfigParser import Config
    CACHE_DIR = Path(Config().CACHE_DIRECTORY) / "backtest_signal_cache"
except Exception:
    CACHE_DIR = Path(__file__).resolve().parent / "data" / "signal_cache"

from BackTrading.indicator_cache import get_precomputed, precompute_all_indicators


def _clean_stale_tempdirs(max_age_hours: float = 2) -> int:
    """启动时清理残留的 bprep_* 临时目录（上次异常中断遗留）。"""
    import stat
    base = tempfile.gettempdir()
    removed = 0
    now = time.time()
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if not entry.startswith("bprep_") or not os.path.isdir(full):
            continue
        try:
            age_hours = (now - os.path.getmtime(full)) / 3600
            if age_hours > max_age_hours:
                def _onerr(func, path, exc_info):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                shutil.rmtree(full, onerror=_onerr, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    if removed:
        logger.info(f"清理了 {removed} 个残留临时目录")
    return removed


_clean_stale_tempdirs()



def _trade_day_str() -> str:
    try:
        return TradingCalendarAnalyzer().get_last_trading_day()
    except Exception:
        return date.today().isoformat()


# ── 增量缓存（日期后缀 + 每只股票独立写入，支持中断续算） ──

_SIGNAL_PIPELINE_VERSION = "v2"  # 信号管线版本号；管线逻辑变更时手动 +1，自动使旧缓存失效


def _compute_config_hash() -> str:
    """全量 config 哈希 + 管线版本号（用于缓存隔离）。

    计算所有非校准信号参数的全量哈希，不再手动排除字段。
    每次 `_SIGNAL_PIPELINE_VERSION` 变更时旧缓存自动失效。
    """
    try:
        cfg = Config()
        ac = cfg.app_config
        payload = {
            "_version": _SIGNAL_PIPELINE_VERSION,
            "regime": ac.regime_detection.model_dump(),
            "divergence": ac.divergence.model_dump(),
            "scoring": ac.scoring_params.model_dump(),
            "technical": ac.technical_constants.model_dump(),
            "full_bull_scoring": ac.full_bull_scoring.model_dump() if hasattr(ac, 'full_bull_scoring') else {},
            "filter_rules": ac.filter_rules.model_dump(),
            "position_sizing": ac.position_sizing.model_dump(),
        }
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _compute_param_hash(params: dict[str, Any]) -> str:
    """仅对影响信号计算的参数做哈希（用于信号缓存隔离）。

    同时支持扁平 dict（顶层有参数名）和结构化 dict（参数嵌套在 scoring/regime 中）。

    PORTFOLIO/RISK 类参数（atr_stop_mult 等）排除在外，
    它们通过 post-cache 变换注入，不触发信号重算。
    conclusion_full_bull 影响风险等级/进出场阈值（vectorized_signal 消费），
    必须纳入哈希，否则不同阈值会错误复用同一份信号缓存。
    """
    def _get(key: str, default: Any) -> Any:
        """从扁平或结构化 params 中提取信号参数值。"""
        # 扁平 dict：顶层直接有 key
        if key in params:
            return params[key]
        # 结构化 dict：从嵌套路径提取
        if key == "boll_narrow_ratio" and "regime" in params:
            return params["regime"].get(key, default)
        if key == "conclusion_full_bull" and "thresholds" in params:
            return params["thresholds"].get("fully_bull", default)
        if key in ("cross_decay_days", "golden_cross_bonus", "divergence_penalty") and "scoring" in params:
            return params["scoring"].get(key, default)
        return default

    key_params = {
        "boll_narrow_ratio": _get("boll_narrow_ratio", 0.8),
        "cross_decay_days": _get("cross_decay_days", 30),
        "golden_cross_bonus": _get("golden_cross_bonus", 10),
        "divergence_penalty": _get("divergence_penalty", 20),
        "conclusion_full_bull": _get("conclusion_full_bull", 80),
    }
    s = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _cache_dir_for(trade_date: str, param_hash: str | None = None, config_hash: str | None = None) -> Path:
    """信号缓存目录路径。
    
    格式: signal_cache_{trade_date}_{config_hash}_{param_hash}/
    config_hash 自动计算（无需传入），param_hash 区分同一天不同回测参数组合。
    """
    if config_hash is None:
        config_hash = _compute_config_hash()
    base = CACHE_DIR / f"signal_cache_{trade_date}_{config_hash}"
    if param_hash:
        base = CACHE_DIR / f"signal_cache_{trade_date}_{config_hash}_{param_hash}"
    return base


def _symbol_bucket(symbol: str) -> str:
    """按 symbol 前 2 字符分桶，减少单目录文件数。"""
    return symbol[:2].lower()


def _symbol_cache_path(cache_dir: Path, symbol: str) -> Path:
    """获取股票信号缓存文件路径（带分桶）。"""
    bucket = _symbol_bucket(symbol)
    bucket_dir = cache_dir / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    return bucket_dir / f"{symbol}.parquet"


def _completed_symbols(trade_date: str, param_hash: str | None = None, config_hash: str | None = None) -> set[str]:
    cd = _cache_dir_for(trade_date, param_hash, config_hash)
    if not cd.exists():
        return set()
    symbols = set()
    for bucket_dir in cd.iterdir():
        if bucket_dir.is_dir():
            for f in bucket_dir.glob("*.parquet"):
                symbols.add(f.stem)
    return symbols


def _save_stock_signal(cache_dir: Path, symbol: str, rows: list[dict]) -> None:
    path = _symbol_cache_path(cache_dir, symbol)
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd", compression_level=3)


def _load_signal_cache(trade_date: str, param_hash: str | None = None, config_hash: str | None = None) -> pd.DataFrame | None:
    cd = _cache_dir_for(trade_date, param_hash, config_hash)
    if not cd.exists():
        return None
    files = []
    for bucket_dir in cd.iterdir():
        if bucket_dir.is_dir():
            files.extend(sorted(bucket_dir.glob("*.parquet")))
    if not files:
        return None
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception:
            logger.warning("跳过损坏的缓存文件: %s", f)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    # 只重命名确实存在的英文列，避免旧版缓存中文列名冲突
    rename_map = {eng: chn for eng, chn in _REV_SIGNAL_COL_MAP.items() if eng in df.columns and eng != chn}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    # 兼容旧版：若 exit_strategy 列存在且为 dict，提取 stop_loss
    if "exit_strategy" in df.columns and "止损价" not in df.columns:
        df["止损价"] = df["exit_strategy"].apply(
            lambda x: float(x.get("stop_loss", 0)) if isinstance(x, dict) else 0.0
        )
    return df


def _merge_signal(kline_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """Merge K线 with signal data — 只提取信号列，减少内存压力。"""
    # 只保留信号计算产生的列，避免加载不需要的中间指标
    _signal_cols = [c for c in signal_df.columns if c in (
        "进场评分", "退出评分", "综合评分", "止损价", "风险等级",
        "entry_score", "exit_score", "score", "atr", "macd_trend",
        "golden_cross", "hist_momentum", "dif_slope", "divergence",
        "vol_price", "kline", "exit_strategy",
    ) or c.startswith("MACD_") or c.startswith("MA_") or c.startswith("ATR_")]
    if not _signal_cols:
        # fallback: 合并全部信号列
        _signal_cols = list(signal_df.columns)
    signal_subset = signal_df[["symbol", "trade_date"] + _signal_cols].copy()
    result = kline_df.merge(signal_subset, on=["symbol", "trade_date"], how="left")
    del signal_subset, signal_df
    for col in ["进场评分", "退出评分", "综合评分", "止损价"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    if "风险等级" in result.columns:
        result["风险等级"] = result["风险等级"].fillna("LOW")
    return result





def prepare_backtest_data(
    kline_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    signal_param_hash: str | None = None,
    compute_exit_strategy: bool = False,
    vectorized: bool = False,
    backtest_start_date: str | None = None,
) -> pd.DataFrame:
    is_flat = params is not None and (
        "atr_stop_mult" in params
        or "boll_narrow_ratio" in params
        or "conclusion_full_bull" in params or "golden_cross_bonus" in params
    )

    # 在 params 被 convert 为 structured 之前，保存 PORTFOLIO/RISK 单值
    _saved_atr_stop = params.get("atr_stop_mult") if isinstance(params, dict) else None
    _saved_risk_none = params.get("risk_none_multiplier") if isinstance(params, dict) else None

    if signal_param_hash is None:
        signal_param_hash = _compute_param_hash(params)

    if params is None:
        cfg = Config()
        params = _build_params(cfg)
    elif is_flat:
        cfg = Config()
        base = _build_params(cfg)
        base["scoring"].update({k: v for k, v in params.items() if k in (
            "cross_decay_days", "cross_decay_min",
            "vol_norm_denominator", "kline_decay_days", "kline_decay_min",
            "expected_return_lookback",
            "golden_cross_bonus", "divergence_penalty",
        )})
        if "boll_narrow_ratio" in params:
            base["regime"]["boll_narrow_ratio"] = float(params["boll_narrow_ratio"])
        base["thresholds"] = {
            "fully_bull": int(params.get("conclusion_full_bull", base["thresholds"]["fully_bull"])),
            "bullish": base["thresholds"]["bullish"],
            "oscillate": base["thresholds"]["oscillate"],
        }
        params = base

    config_hash = _compute_config_hash()
    cache_tag = f"cfg={config_hash},param={signal_param_hash}"

    random.seed(42)
    np.random.seed(42)

    trade_date = _trade_day_str()
    cache_dir = _cache_dir_for(trade_date, signal_param_hash, config_hash)

    symbols = sorted(kline_df["symbol"].unique())
    done = _completed_symbols(trade_date, signal_param_hash, config_hash)
    missing = [s for s in symbols if s not in done]

    def _finalize(kline, signal) -> pd.DataFrame:
        merged = _merge_signal(kline, signal)
        del kline, signal
        gc.collect()
        merged = apply_ml_signal(merged)
        if _saved_atr_stop is not None and "ATR" in merged.columns:
            stop_raw = merged["close"] - merged["ATR"] * _saved_atr_stop
            merged["止损价"] = np.floor(stop_raw * 100 + 0.5) / 100

        # 截断指标预热缓冲期：仅保留 backtest_start_date 之后的数据
        # 先过滤再返回，避免 .copy() 触发 _consolidate_inplace OOM（~846 MiB）
        if backtest_start_date is not None:
            if pd.api.types.is_datetime64_any_dtype(merged["trade_date"]):
                _cutoff = pd.Timestamp(backtest_start_date)
                mask = merged["trade_date"] >= _cutoff
            else:
                mask = merged["trade_date"] >= backtest_start_date
            n_before = len(merged)
            merged = merged.loc[mask]
            gc.collect()
            n_cut = n_before - len(merged)
            if n_cut > 0:
                logger.info(f"[DIAG] 截断 {n_cut} 行指标预热缓冲数据（< {backtest_start_date}）")

        # 延迟诊断日志到截断后（减少中间切片体积）
        _first_date = merged["trade_date"].iloc[0]
        _first = merged[merged["trade_date"] == _first_date]
        _fe = _first["进场评分"]
        logger.info(f"[DIAG] 首日 {_first_date} 进场评分: min={_fe.min():.1f} max={_fe.max():.1f} mean={_fe.mean():.1f} median={_fe.median():.1f} >=60={(_fe>=60).sum()}/{len(_fe)}")
        _dates = merged["trade_date"].unique()
        if len(_dates) > 100:
            _mid_date = _dates[100]
            _mid = merged[merged["trade_date"] == _mid_date]
            _me = _mid["进场评分"]
            logger.info(f"[DIAG] 第100日 {_mid_date} 进场评分: min={_me.min():.1f} max={_me.max():.1f} mean={_me.mean():.1f} median={_me.median():.1f} >=60={(_me>=60).sum()}/{len(_me)}")
        _last_date = _dates[-1]
        _last = merged[merged["trade_date"] == _last_date]
        _le = _last["进场评分"]
        logger.info(f"[DIAG] 末日 {_last_date} 进场评分: min={_le.min():.1f} max={_le.max():.1f} mean={_le.mean():.1f} median={_le.median():.1f} >=60={(_le>=60).sum()}/{len(_le)}")
        return merged

    if done:
        if not missing:
            logger.info(f"信号缓存全部命中（{len(done)} 只）[{cache_tag}]")
            signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash)
            if signal_df is not None:
                return _finalize(kline_df, signal_df)
        else:
            logger.info(f"信号缓存部分命中（{len(done)}/{len(symbols)}），续算 {len(missing)} 只 [{cache_tag}]")

    if not missing:
        logger.info("无需要计算的股票")
        signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash)
        if signal_df is not None:
            return _finalize(kline_df, signal_df)
        return kline_df

    # ── 需要计算的股票 ──
    _t0 = time.time()
    logger.info(f"信号缓存无效或不存在，开始计算 {len(missing)} 只 [{cache_tag}]...")
    tmpdir = tempfile.mkdtemp(prefix="bprep_")
    stock_dir = os.path.join(tmpdir, "stocks")
    os.mkdir(stock_dir)
    try:
        for sym, grp in kline_df.groupby("symbol"):
            grp.sort_values("trade_date").to_parquet(
                os.path.join(stock_dir, f"{sym}.parquet"), index=False
            )

        # Phase 0: 预计算所有股票的技术指标 + peak/trough（仅一次，后续评估复用）
        if vectorized:
            precompute_all_indicators(stock_dir)

        from tqdm import tqdm
        signal_pipelines = Config().SIGNAL_PIPELINES
        _workers_per_pipeline = max(
            min((os.cpu_count() or 4) // signal_pipelines, 3), 1
        )

        _worker_fn = _stock_worker_vectorized if vectorized else _stock_worker

        def _pipeline(syms: list[str], idx: int) -> None:
            """单管道：ThreadPoolExecutor 并发处理股票（Windows spawn 下 ProcessPoolExecutor 会死锁）。"""
            if not syms:
                return
            pbar = tqdm(total=len(syms), desc=f"管道{idx+1}", unit="只", ncols=50, position=idx)
            pool = ThreadPoolExecutor(max_workers=_workers_per_pipeline)
            try:
                fut_to_sym: dict[Any, str] = {}
                for sym in syms:
                    fut = pool.submit(_worker_fn, sym, stock_dir, params, compute_exit_strategy)
                    fut_to_sym[fut] = sym

                for future in as_completed(fut_to_sym):
                    sym = fut_to_sym[future]
                    try:
                        rows = future.result(timeout=120)
                        if rows:
                            _save_stock_signal(cache_dir, sym, rows)
                    except Exception as e:
                        logger.opt(exception=True).warning(f"  [{sym}] 信号计算失败: {e}")
                    pbar.update(1)
            finally:
                pool.shutdown(wait=False)
            pbar.close()

        chunks = [missing[i::signal_pipelines] for i in range(signal_pipelines)]
        pool_t = ThreadPoolExecutor(max_workers=signal_pipelines)
        try:
            futs = [pool_t.submit(_pipeline, chunk, idx) for idx, chunk in enumerate(chunks)]
            for f in futs:
                try:
                    f.result(timeout=3600)
                except Exception:
                    logger.warning(f" 管道 Future 异常，继续等待其他管道")
        finally:
            pool_t.shutdown(wait=False)
        _t1 = time.time()
        logger.info(f"所有管道完成，耗时 {_t1-_t0:.1f}s")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.debug(f"临时目录已清理: {tmpdir}")

    # ── 加载缓存合并 ──
    logger.info(f"加载信号缓存合并...")
    signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash)
    if signal_df is None or signal_df.empty:
        logger.warning(f"所有信号计算失败（signal_df={type(signal_df).__name__}），返回原始 K 线")
        return kline_df
    logger.info(f"信号合并完成: {len(signal_df)} 行 ({time.time()-_t0:.1f}s)")
    return _finalize(kline_df, signal_df)


def _stock_worker(
    symbol: str,
    stock_dir: str,
    params: dict[str, Any],
    compute_exit_strategy: bool = False,
) -> list[dict[str, Any]]:
    stock_df = pd.read_parquet(os.path.join(stock_dir, f"{symbol}.parquet"), engine="fastparquet")
    if len(stock_df) < 60:
        return []

    # 数据质量检查
    _validate_stock_data(stock_df, symbol)

    # 所有滚动指标（MA, MACD, ATR, BBANDS 等）只向后看，不存在前瞻偏差。
    # 在全量数据上一次性计算，避免每根 bar 重复 800 次。
    stock_df = _compute_indicators(stock_df)

    analyzer = MACDAnalyzer()
    rows: list[dict[str, Any]] = []

    for i in range(len(stock_df)):
        bar = stock_df.iloc[: i + 1]
        try:
            signal = _compute_signal(analyzer, bar, params, compute_exit_strategy)
        except Exception:
            continue
        _details = signal.get("details") or {}
        rows.append({
            "symbol": symbol,
            "trade_date": bar["trade_date"].iloc[-1],
            "entry_score": float(signal.get("进场评分", 0)),
            "exit_score": float(signal.get("退出评分", 0)),
            "risk_level": str(signal.get("风险等级", "LOW")),
            "score": float(signal.get("score", 0)),
            "atr": float(bar["ATR"].iloc[-1]) if "ATR" in bar.columns else 0.0,
            "macd_trend": float(_details.get("MACD趋势", {}).get("score", 0)),
            "golden_cross": float(_details.get("金叉信号", {}).get("score", 0)),
            "hist_momentum": float(_details.get("柱状动能", {}).get("score", 0)),
            "dif_slope": float(_details.get("DIF斜率", {}).get("score", 0)),
            "divergence": float(_details.get("背离信号", {}).get("score", 0)),
            "vol_price": float(_details.get("量价配合", {}).get("score", 0)),
            "kline": float(_details.get("K线形态", {}).get("score", 0)),
        })
    return rows


def _stock_worker_vectorized(
    symbol: str,
    stock_dir: str,
    params: dict[str, Any],
    compute_exit_strategy: bool = False,
) -> list[dict[str, Any]]:
    """全向量化版本的 _stock_worker — 无 per-bar Python 循环。

    使用 Phase 0 预计算的技术指标 + peak/trough 缓存，
    仅运行评分层 compute_signals。

    捕获 BaseException 并写入 stderr（loguru 在 spawn 子进程中可能不可用），
    防止子进程崩溃导致 BrokenProcessPool。
    """
    import sys as _sys
    try:
        stock_df, _peaks, _troughs = get_precomputed(symbol, stock_dir)
        if stock_df.empty or len(stock_df) < 60:
            return []

        _validate_stock_data(stock_df, symbol)

        if stock_df.attrs.get("_invalid", False):
            logger.warning(f"[{symbol}] 数据无效，跳过信号计算")
            return []

        from LogicAnalyzer.signals.divergence import adaptive_distance
        _dd = adaptive_distance(stock_df["DIF"], base_distance=10) if "DIF" in stock_df.columns else 11

        try:
            signal_df = _compute_signals_vec(
                stock_df,
                params=params,
                compute_exit_strategy=compute_exit_strategy,
                diverge_distance=_dd,
            )
        except Exception as e:
            import traceback
            logger.warning(f"  [{symbol}] 向量化信号计算失败({e})\n{traceback.format_exc()}")
            try:
                return _stock_worker(symbol, stock_dir, params, compute_exit_strategy)
            except Exception as f:
                logger.warning(f"  [{symbol}] 原始引擎也失败: {f}")
                return []
        # 将向量化结果格式化为 _stock_worker 相同的 list[dict]
        signal_df["symbol"] = symbol
        rows: list[dict[str, Any]] = []
        for _, row in signal_df.iterrows():
            rows.append({
                "symbol": symbol,
                "trade_date": row["trade_date"],
                "entry_score": float(row["entry_score"]),
                "exit_score": float(row["exit_score"]),
                "risk_level": str(row["risk_level"]),
                "score": float(row["score"]),
                "atr": float(row["atr"]) if pd.notna(row["atr"]) else 0.0,
                "macd_trend": float(row["macd_trend"]),
                "golden_cross": float(row["golden_cross"]),
                "hist_momentum": float(row["hist_momentum"]),
                "dif_slope": float(row["dif_slope"]),
                "divergence": float(row["divergence"]),
                "vol_price": float(row["vol_price"]),
                "kline": float(row["kline"]),
                "exit_strategy": {"stop_loss": float(row["stop_loss"])} if compute_exit_strategy else {"stop_loss": 0.0},
            })
        return rows
    except BaseException:
        import traceback
        _sys.stderr.write(f"\n[FATAL] _stock_worker_vectorized({symbol}) crashed:\n{traceback.format_exc()}\n")
        _sys.stderr.flush()
        return []


# 英文列名映射（用于 parquet 缓存存储，避免中文列名编码问题）
_SIGNAL_COL_MAP = {
    "symbol": "symbol",
    "trade_date": "trade_date",
    "entry_score": "进场评分",
    "exit_score": "退出评分",
    "risk_level": "风险等级",
    "score": "综合评分",
    "atr": "ATR",
    "macd_trend": "MACD趋势分",
    "golden_cross": "金叉信号分",
    "hist_momentum": "柱状动能分",
    "dif_slope": "DIF斜评分",
    "divergence": "背离信号分",
    "vol_price": "量价配合分",
    "kline": "K线形态分",
    "stop_loss": "止损价",
}

# 直接使用 _SIGNAL_COL_MAP 作为重命名映射（英文 -> 中文）
_REV_SIGNAL_COL_MAP = _SIGNAL_COL_MAP


def _validate_stock_data(df: pd.DataFrame, symbol: str) -> None:
    """数据质量检查：零价格、缺失值、涨跌停。标记无效股票让下游跳过。"""
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        logger.warning(f"[{symbol}] 存在非正价格，可能停牌或数据异常，标记为无效")
        df.attrs["_invalid"] = True
        return
    if df["close"].isna().all() or df["volume"].isna().all():
        logger.warning(f"[{symbol}] 全部价格或成交量为空，标记为无效")
        df.attrs["_invalid"] = True
        return
    df.attrs["_invalid"] = False
    nan_frac = df[["open", "high", "low", "close", "volume"]].isna().sum().sum() / (
        len(df) * 5
    )
    if nan_frac > 0.01:
        logger.warning(f"[{symbol}] 缺失值比例 {nan_frac:.1%} > 1%")


def _compute_indicators(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
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

    if "DIF" in df.columns and "DEA" in df.columns:
        dif = df["DIF"]
        dea = df["DEA"]
        prev_dif = dif.shift(1).fillna(dea.shift(1).fillna(0))
        prev_dea = dea.shift(1).fillna(0)
        golden = (dif > dea) & (prev_dif <= prev_dea)
        dead = (dif < dea) & (prev_dif >= prev_dea)
        # 与循环路径 (MACDAnalyzer) 对齐：金叉标注零轴语义（零轴上/下金叉），
        # 供 vectorized golden_cross_score 区分零轴上/下金叉评分
        from LogicAnalyzer.SignalConstants import MACDSignals
        df["MACD_SIGNAL_DETAIL"] = np.where(
            golden,
            MACDSignals.golden_cross_label(dif, dea),
            np.where(dead, MACDSignals.death_cross_label(dead, dif, dea),
                     np.where(dif > 0, "多头", "空头")),
        )
        df["MACD_CROSS"] = 0
        df.loc[golden, "MACD_CROSS"] = 1
        df.loc[dead, "MACD_CROSS"] = -1

    for p in (5, 10, 20, 30, 60):
        df[f"MA_{p}"] = close.rolling(p).mean()

    bb = ta.bbands(close, length=20, std=2)  # type: ignore[arg-type]
    if bb is not None and bb.shape[1] >= 3:
        df["BBU_20_2.0"] = bb.iloc[:, 0].values
        df["BBM_20_2.0"] = bb.iloc[:, 1].values
        df["BBL_20_2.0"] = bb.iloc[:, 2].values
        df["BOLL_BANDWIDTH"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / close

    # 以下指标供 pipeline_analysis._precompute_rule_indicators 使用，
    # 列名必须与 guards 的检查前缀完全匹配，避免每根 bar 重算。
    adx_series = ta.adx(high, low, close, length=14)
    if adx_series is not None:
        df["ADX"] = adx_series.get("ADX_14", 0.0).values  # type: ignore[union-attr]

    df["MA_200"] = close.rolling(200).mean()

    rsi_s = ta.rsi(close, length=14)
    if rsi_s is not None:
        df["RSI_14"] = rsi_s.values if isinstance(rsi_s, pd.Series) else rsi_s

    stoch_df = ta.stoch(high, low, close, k=9, d=3)
    if stoch_df is not None:
        for c in stoch_df.columns:
            df[c] = stoch_df[c].to_numpy()

    df["AMOUNT"] = close * df["volume"]
    df["AMOUNT_MA20"] = df["AMOUNT"].rolling(20).mean()
    df["AMPLITUDE_PCT"] = (high - low) / close

    cci_s = ta.cci(high, low, close, length=20)
    if cci_s is not None:
        df["CCI_20"] = cci_s.values if isinstance(cci_s, pd.Series) else cci_s
    return df


def _compute_signal(
    analyzer: MACDAnalyzer,
    bar: pd.DataFrame,
    params: dict[str, Any],
    compute_exit_strategy: bool = False,
) -> dict[str, Any]:
    result = analyzer.pipeline_analysis(bar, params=params, compute_exit_strategy=compute_exit_strategy)
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
    if risk_level in ("HIGH", "D"):
        return 100.0
    stop_loss = exit_strategy.get("stop_loss")
    if stop_loss and len(df) > 0:
        close = df["close"].iloc[-1]
        if close < stop_loss:
            return 90.0
    return 0.0


def _build_params(cfg: Config) -> dict[str, Any]:
    ac = cfg.app_config
    ps = ac.position_sizing
    fb = ac.full_bull_scoring
    return {
        "regime": ac.regime_detection.model_dump(),
        "divergence": ac.divergence.model_dump(),
        "scoring": ac.scoring_params.model_dump(),
        "position_sizing": {"risk_none_multiplier": ps.RISK_NONE_MULTIPLIER},
        "technical": ac.technical_constants.model_dump(),
        "thresholds": {
            "fully_bull": fb.CONCLUSION_FULL_BULL,
            "bullish": fb.CONCLUSION_BULLISH,
            "oscillate": fb.CONCLUSION_OSCILLATE,
        },
    }
