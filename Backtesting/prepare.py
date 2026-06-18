from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from ConfigParser import Config
from LogicAnalyzer.pipeline import MACDAnalyzer


def _adjust_signal_workers(max_workers: int, watermark: float = 0.8) -> int:
    """信号计算时根据 CPU 负载动态调整进程数。"""
    try:
        import psutil
        load = psutil.cpu_percent(interval=0.2) / 100.0
        if load > watermark:
            return max(1, max_workers - 1)
    except Exception:
        pass
    return max_workers


def prepare_backtest_data(
    kline_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if params is None:
        cfg = Config()
        params = _build_params(cfg)

    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count // 2)
    batch_size = 200
    tmpdir = tempfile.mkdtemp(prefix="bprep_")
    symbols = kline_df["symbol"].unique()
    total = len(symbols)

    from tqdm import tqdm

    pbar = tqdm(total=total, desc="预计算信号", unit="只", ncols=80)
    result_files: list[str] = []

    # write kline_df to parquet once for workers to read
    kline_path = os.path.join(tmpdir, "kline.parquet")
    kline_df.to_parquet(kline_path, index=False)

    batches = [symbols[i:i + batch_size] for i in range(0, total, batch_size)]
    for bidx, batch_syms in enumerate(batches):
        batch_rows: list[list[dict]] = []
        workers = _adjust_signal_workers(max_workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_stock_worker, sym, kline_path, params): sym
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
            flat = [r for sub in batch_rows for r in sub]
            batch_df = pd.DataFrame(flat)
            batch_file = os.path.join(tmpdir, f"batch_{bidx}.parquet")
            batch_df.to_parquet(batch_file, index=False)
            result_files.append(batch_file)

    pbar.close()

    if not result_files:
        logger.warning("所有信号计算失败，返回原始 K 线")
        return kline_df

    signal_parts = [pd.read_parquet(f) for f in result_files]
    signal_df = pd.concat(signal_parts, ignore_index=True)

    # cleanup
    for f in result_files:
        os.remove(f)
    os.remove(kline_path)
    os.rmdir(tmpdir)

    result = kline_df.merge(signal_df, on=["symbol", "trade_date"], how="left")
    for col in ["进场评分", "退出评分", "综合评分", "止损价"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    result["风险等级"] = result["风险等级"].fillna("LOW")
    return result


def _stock_worker(
    symbol: str,
    kline_path: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    df = pd.read_parquet(kline_path)
    stock_df = (
        df[df["symbol"] == symbol].sort_values("trade_date").copy()
    )
    if len(stock_df) < 60:
        return []

    _compute_indicators(stock_df)
    analyzer = MACDAnalyzer()
    rows: list[dict[str, Any]] = []

    for i in range(len(stock_df)):
        bar = stock_df.iloc[: i + 1].reset_index(drop=True)
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


def _compute_indicators(df: pd.DataFrame) -> None:
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
