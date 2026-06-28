from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from DataCollection.CalendarManager import TradingCalendarAnalyzer

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from LogicAnalyzer.pipeline import MACDAnalyzer

try:
    from ConfigParser import Config
    CACHE_DIR = Path(Config().CACHE_DIRECTORY) / "backtest_signal_cache"
except Exception:
    CACHE_DIR = Path(__file__).resolve().parent / "data" / "signal_cache"
PROCESS_BATCH_SIZE = 200


def _trade_day_str() -> str:
    try:
        return TradingCalendarAnalyzer().get_last_trading_day()
    except Exception:
        return date.today().isoformat()


# ── 增量缓存（日期后缀 + 每只股票独立写入，支持中断续算） ──

def _compute_config_hash() -> str:
    """计算所有非校准信号参数的哈希（用于缓存隔离）。
    
    当 config.ini 中任何信号相关参数变更时（如 atr_length、rsi_length 等），
    config_hash 会变化，自动使旧缓存失效。"""
    try:
        cfg = Config()
        ac = cfg.app_config
        # 收集所有信号相关配置（排除仅用于回测的 7 个校准参数）
        payload = {
            "regime": {
                "oscillation_hist_std_ratio": ac.regime_detection.OSCILLATION_HIST_STD_RATIO,
                "top_risk_ma20_deviation": ac.regime_detection.TOP_RISK_MA20_DEVIATION,
                "oscillation_min_bars": ac.regime_detection.OSCILLATION_MIN_BARS,
                "reversal_lookback": ac.regime_detection.REVERSAL_LOOKBACK,
            },
            "divergence": ac.divergence.model_dump(),
            "scoring": {
                k: v for k, v in ac.scoring_params.model_dump().items()
                if k not in ("atr_stop_mult", "atr_t1_mult", "cross_decay_days", "cross_decay_min", "kline_decay_days", "kline_decay_min", "vol_norm_denominator", "atr_t2_mult", "trailing_stop_high_ratio", "trailing_stop_lookback", "trailing_stop_high_lookback", "expected_return_lookback")
            },
            "technical": ac.technical_constants.model_dump(),
            "full_bull_scoring": ac.full_bull_scoring.model_dump() if hasattr(ac, 'full_bull_scoring') else {},
            "filter_rules": {
                k: v for k, v in ac.filter_rules.model_dump().items()
                if k != "liq_veto_ratio"
            },
        }
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _compute_param_hash(params: dict[str, Any]) -> str:
    """计算回测校准参数的哈希（用于同一天多参数组合缓存隔离）。"""
    key_params = {
        "atr_stop_mult": params.get("atr_stop_mult", 1.5),
        "atr_t1_mult": params.get("atr_t1_mult", 3.0),
        "kelly_fraction": params.get("kelly_fraction", 0.25),
        "position_a": params.get("position_a", 0.30),
        "liq_veto_ratio": params.get("liq_veto_ratio", 0.05),
        "boll_narrow_ratio": params.get("boll_narrow_ratio", 0.8),
        "cross_decay_days": params.get("cross_decay_days", 30),
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
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    # 将英文列名映射回中文列名
    df.rename(columns=_REV_SIGNAL_COL_MAP, inplace=True)
    return df


def _merge_signal(kline_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    result = kline_df.merge(signal_df, on=["symbol", "trade_date"], how="left")
    for col in ["进场评分", "退出评分", "综合评分", "止损价"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    result["风险等级"] = result["风险等级"].fillna("LOW")
    return result


def prepare_backtest_data(
    kline_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    signal_param_hash: str | None = None,
    compute_exit_strategy: bool = False,
) -> pd.DataFrame:
    if params is None:
        cfg = Config()
        params = _build_params(cfg)

    if signal_param_hash is None:
        signal_param_hash = _compute_param_hash(params)

    config_hash = _compute_config_hash()
    cache_tag = f"cfg={config_hash},param={signal_param_hash}"

    random.seed(42)
    np.random.seed(42)

    trade_date = _trade_day_str()
    cache_dir = _cache_dir_for(trade_date, signal_param_hash, config_hash)

    symbols = sorted(kline_df["symbol"].unique())
    done = _completed_symbols(trade_date, signal_param_hash, config_hash)
    missing = [s for s in symbols if s not in done]

    if done:
        if not missing:
            logger.info(f"信号缓存全部命中（{len(done)} 只）[{cache_tag}]")
            signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash)
            if signal_df is not None:
                return _merge_signal(kline_df, signal_df)
        else:
            logger.info(f"信号缓存部分命中（{len(done)}/{len(symbols)}），续算 {len(missing)} 只 [{cache_tag}]")

    if not missing:
        logger.info("无需要计算的股票")
        signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash)
        return _merge_signal(kline_df, signal_df) if signal_df is not None else kline_df

    # ── 需要计算的股票 ──
    logger.info(f"信号缓存无效或不存在，开始计算 {len(missing)} 只 [{cache_tag}]...")
    tmpdir = tempfile.mkdtemp(prefix="bprep_")
    stock_dir = os.path.join(tmpdir, "stocks")
    os.mkdir(stock_dir)
    for sym, grp in kline_df.groupby("symbol"):
        grp.sort_values("trade_date").to_parquet(
            os.path.join(stock_dir, f"{sym}.parquet"), index=False
        )

    from tqdm import tqdm
    signal_pipelines = Config().SIGNAL_PIPELINES

    def _pipeline(syms: list[str], idx: int) -> None:
        """单管道：内部自带 ProcessPoolExecutor，逐个 batch 提交。"""
        pbar = tqdm(total=len(syms), desc=f"管道{idx+1}", unit="只", ncols=50, position=idx)
        for i in range(0, len(syms), PROCESS_BATCH_SIZE):
            batch = syms[i:i + PROCESS_BATCH_SIZE]
            with ProcessPoolExecutor(max_workers=2) as pool:
                fut_to_sym = {
                    pool.submit(_stock_worker, sym, stock_dir, params, compute_exit_strategy): sym
                    for sym in batch
                }
                for future in as_completed(fut_to_sym):
                    sym = fut_to_sym[future]
                    try:
                        rows = future.result()
                        if rows:
                            _save_stock_signal(cache_dir, sym, rows)
                    except Exception:
                        pass
                    pbar.update(1)
            pbar.close()

    chunks = [missing[i::signal_pipelines] for i in range(signal_pipelines)]
    with ThreadPoolExecutor(max_workers=signal_pipelines) as pool:
        for idx, chunk in enumerate(chunks):
            pool.submit(_pipeline, chunk, idx)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 加载缓存合并 ──
    signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash)
    if signal_df is None:
        logger.warning("所有信号计算失败，返回原始 K 线")
        return kline_df
    return _merge_signal(kline_df, signal_df)


def _stock_worker(
    symbol: str,
    stock_dir: str,
    params: dict[str, Any],
    compute_exit_strategy: bool = False,
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
            signal = _compute_signal(analyzer, bar, params, compute_exit_strategy)
        except Exception:
            continue
        rows.append({
            "symbol": symbol,
            "trade_date": bar["trade_date"].iloc[-1],
            "entry_score": float(signal.get("进场评分", 0)),
            "exit_score": float(signal.get("退出评分", 0)),
            "risk_level": str(signal.get("风险等级", "LOW")),
            "stop_loss": float(signal.get("止损价", 0) or 0),
            "score": float(signal.get("score", 0)),
        })
    return rows


# 英文列名映射（用于 parquet 缓存存储，避免中文列名编码问题）
_SIGNAL_COL_MAP = {
    "symbol": "symbol",
    "trade_date": "trade_date",
    "entry_score": "进场评分",
    "exit_score": "退出评分",
    "risk_level": "风险等级",
    "stop_loss": "止损价",
    "score": "综合评分",
}

# 直接使用 _SIGNAL_COL_MAP 作为重命名映射（英文 -> 中文）
_REV_SIGNAL_COL_MAP = _SIGNAL_COL_MAP


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
    return {
        "regime": ac.regime_detection.model_dump(),
        "divergence": ac.divergence.model_dump(),
        "scoring": ac.scoring_params.model_dump(),
        "technical": ac.technical_constants.model_dump(),
    }
