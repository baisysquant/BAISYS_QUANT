from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from DataCollection.CalendarManager import TradingCalendarAnalyzer

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from ConfigParser import Config
from LogicAnalyzer.pipeline import MACDAnalyzer

try:
    from ConfigParser import Config
    CACHE_DIR = Path(Config().CACHE_DIRECTORY) / "backtest_signal_cache"
except Exception:
    CACHE_DIR = Path(__file__).resolve().parent / "data" / "signal_cache"
CACHE_GROUP_SIZE = 500
PROCESS_BATCH_SIZE = 200
CACHE_MAX_AGE_DAYS = 7


def _trade_day_str() -> str:
    try:
        return TradingCalendarAnalyzer().get_last_trading_day().isoformat()
    except Exception:
        return date.today().isoformat()


def _adjust_signal_workers(current: int, max_workers: int, watermark: float = 0.8) -> int:
    try:
        import psutil
        load = psutil.cpu_percent(interval=0.2) / 100.0
        if load > watermark and current > 1:
            return current - 1
        if load < watermark * 0.7 and current < max_workers:
            return current + 1
    except Exception:
        pass
    return current


def _make_param_hash(params: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def _cache_meta_path() -> Path:
    return CACHE_DIR / "meta.json"


def _check_cache_valid(symbols: list[str], params: dict[str, Any]) -> bool:
    meta_file = _cache_meta_path()
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return False

    if meta.get("date") != _trade_day_str():
        return False
    if meta.get("symbol_count") != len(symbols):
        return False
    if meta.get("param_hash") != _make_param_hash(params):
        return False

    for gidx in range(meta.get("num_groups", 0)):
        if not (CACHE_DIR / f"cache_{gidx}.parquet").exists():
            return False
    return True


def _save_cache_meta(symbols: list[str], params: dict[str, Any], num_groups: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": _trade_day_str(),
        "symbol_count": len(symbols),
        "param_hash": _make_param_hash(params),
        "num_groups": num_groups,
    }
    _cache_meta_path().write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _cleanup_stale_cache() -> None:
    if not CACHE_DIR.exists():
        return
    cutoff = _trade_day_str()
    for f in CACHE_DIR.iterdir():
        if f.is_file():
            try:
                mtime = date.fromtimestamp(f.stat().st_mtime).isoformat()
                if mtime < cutoff:
                    f.unlink()
            except (OSError, ValueError):
                pass
    # 如果清理后 meta.json 已经不在，重建空目录
    if CACHE_DIR.exists() and not (CACHE_DIR / "meta.json").exists():
        for f in CACHE_DIR.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        CACHE_DIR.rmdir()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


def prepare_backtest_data(
    kline_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if params is None:
        cfg = Config()
        params = _build_params(cfg)

    # 锁定随机种子使回测可复现
    random.seed(42)
    np.random.seed(42)

    symbols = sorted(kline_df["symbol"].unique())
    total = len(symbols)

    _cleanup_stale_cache()

    # ── 缓存命中：直接加载合并 ──
    if _check_cache_valid(symbols, params):
        cache_files = sorted(CACHE_DIR.glob("cache_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
        if cache_files:
            logger.info(f"信号缓存有效（{len(cache_files)} 个文件），跳过计算")
            signal_parts = [pd.read_parquet(f) for f in cache_files]
            signal_df = pd.concat(signal_parts, ignore_index=True)
            result = kline_df.merge(signal_df, on=["symbol", "trade_date"], how="left")
            for col in ["进场评分", "退出评分", "综合评分", "止损价"]:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
            result["风险等级"] = result["风险等级"].fillna("LOW")
            return result

    # ── 缓存无效/不存在：重新计算 ──
    logger.info("信号缓存无效或不存在，开始计算...")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    max_workers = 2
    tmpdir = tempfile.mkdtemp(prefix="bprep_")

    from tqdm import tqdm
    pbar = tqdm(total=total, desc="预计算信号", unit="只", ncols=80)

    # 按股票分割写入独立 parquet
    stock_dir = os.path.join(tmpdir, "stocks")
    os.mkdir(stock_dir)
    for sym, grp in kline_df.groupby("symbol"):
        grp.sort_values("trade_date").to_parquet(
            os.path.join(stock_dir, f"{sym}.parquet"), index=False
        )

    # 每 CACHE_GROUP_SIZE 只股票为一个缓存组
    cache_groups = [symbols[i:i + CACHE_GROUP_SIZE] for i in range(0, total, CACHE_GROUP_SIZE)]

    workers = max_workers
    for gidx, group_syms in enumerate(cache_groups):
        group_rows: list[list[dict]] = []

        # 组内按 PROCESS_BATCH_SIZE 分批并行
        group_batches = [
            group_syms[i:i + PROCESS_BATCH_SIZE]
            for i in range(0, len(group_syms), PROCESS_BATCH_SIZE)
        ]
        for batch_syms in group_batches:
            batch_rows: list[list[dict]] = []
            workers = _adjust_signal_workers(workers, max_workers)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_stock_worker, sym, stock_dir, params): sym
                    for sym in batch_syms
                }
                for future in as_completed(futures):
                    try:
                        rows = future.result()
                        if rows:
                            batch_rows.append(rows)
                    except Exception:
                        pass
                    pbar.update(1)

            if batch_rows:
                group_rows.append([r for sub in batch_rows for r in sub])

        # 每 500 只写一次持久化缓存
        if group_rows:
            flat = [r for sub in group_rows for r in sub]
            pd.DataFrame(flat).to_parquet(CACHE_DIR / f"cache_{gidx}.parquet", index=False)

        # 每次缓存组写入后更新元信息（支持中断恢复）
        _save_cache_meta(symbols, params, gidx + 1)

    pbar.close()
    shutil.rmtree(stock_dir, ignore_errors=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 加载缓存合并 ──
    cache_files = sorted(CACHE_DIR.glob("cache_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    if not cache_files:
        logger.warning("所有信号计算失败，返回原始 K 线")
        return kline_df

    signal_parts = [pd.read_parquet(f) for f in cache_files]
    signal_df = pd.concat(signal_parts, ignore_index=True)

    result = kline_df.merge(signal_df, on=["symbol", "trade_date"], how="left")
    for col in ["进场评分", "退出评分", "综合评分", "止损价"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    result["风险等级"] = result["风险等级"].fillna("LOW")
    return result


def _stock_worker(
    symbol: str,
    stock_dir: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    stock_df = pd.read_parquet(os.path.join(stock_dir, f"{symbol}.parquet"))
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
            signal = _compute_signal(analyzer, bar, params)
        except Exception:
            continue
        rows.append({
            "symbol": symbol,
            "trade_date": bar["trade_date"].iloc[-1],
            "进场评分": float(signal.get("进场评分", 0)),
            "退出评分": float(signal.get("退出评分", 0)),
            "风险等级": str(signal.get("风险等级", "LOW")),
            "止损价": float(signal.get("止损价", 0) or 0),
            "综合评分": float(signal.get("score", 0)),
        })
    return rows


def _validate_stock_data(df: pd.DataFrame, symbol: str) -> None:
    """数据质量检查：零价格、缺失值、涨跌停。仅 warn 不阻断。"""
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        logger.warning(f"[{symbol}] 存在非正价格，可能停牌或数据异常")
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

    # 以下指标供 pipeline_analysis._precompute_rule_indicators 使用，
    # 列名必须与 guards 的检查前缀完全匹配，避免每根 bar 重算。
    adx_series = ta.adx(high, low, close, length=14)
    if adx_series is not None:
        df["ADX"] = adx_series.get("ADX_14", 0.0).values

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
) -> dict[str, Any]:
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
    return {
        "regime": ac.regime_detection.model_dump(),
        "divergence": ac.divergence.model_dump(),
        "scoring": ac.scoring_params.model_dump(),
        "technical": ac.technical_constants.model_dump(),
    }
