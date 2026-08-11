from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import date
from pathlib import Path
from typing import Any

from DataCollection.CalendarManager import TradingCalendarAnalyzer

import numpy as np
import pandas as pd
from loguru import logger

from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
from LogicAnalyzer.ml.signal_model import apply_ml_signal

# ── ML 预测列冻结缓存（P1 特征/参数解耦） ──
# key: (config_hash, data_fp) → DataFrame[symbol, trade_date, 进场评分]
# ML 只按数据版本重训一次（该版本的"首帧"，优化器路径下即默认参数帧），
# 后续任意参数变体直接注入冻结预测，不再重训 XGBoost（~800s → 秒级）。
# 未预测日期（预热期/模型不显著回退）保持原生评分，语义与 apply_ml_signal 一致。
_ML_PRED_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_ML_PRED_LOCK = threading.Lock()

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

from BackTrading.indicator_cache import get_precomputed, get_divergence, precompute_all_indicators
from BackTrading import output_store as _os
from BackTrading import calendar_align as _ca


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


def _data_fingerprint(kline_df: pd.DataFrame, data_version: str | None = None) -> str:
    """信号缓存隔离用数据指纹：区分不同日期范围/股票列表/内容的输入数据。

    修复 WFO 污染：信号缓存 key 原先只有 (trade_date, config, param)，
    窗口切片的 prepare（如 2023-01~2025-11）生成的缓存会被更大范围的
    调用命中（key 相同），导致 2025-11 之后的信号全部 fillna(0)。

    内容哈希：增量同步 / 复权因子改写会改变 OHLC 内容而保持行数与日期范围
    不变，旧指纹下会静默复用脏缓存；现对 OHLCV 采样行做 md5，
    内容变化即整库失效（宁可全废，不可错用）。

    data_version: P3.1 显式数据版本标识（如 kline 表 max(trade_date)+行数
    摘要）。增量同步不改 OHLC 内容时也能让缓存失效，杜绝脏复用。
    """
    try:
        dates = kline_df["trade_date"]
        syms = sorted(kline_df["symbol"].astype(str).dropna().unique())
        content = ""
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in kline_df.columns]
        if cols:
            step = max(1, len(kline_df) // 20000)
            sampled = kline_df.iloc[::step]
            content = hashlib.md5(sampled[cols].values.tobytes()).hexdigest()[:8]
        raw = (
            f"{data_version or ''}|{len(kline_df)}_{dates.min()}_{dates.max()}_{len(syms)}_"
            f"{hashlib.md5('|'.join(syms).encode()).hexdigest()[:8]}_{content}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:10]
    except Exception:
        return "unknown"


def _cache_dir_for(trade_date: str, param_hash: str | None = None, config_hash: str | None = None, data_fp: str | None = None) -> Path:
    """信号缓存目录路径。

    格式: signal_cache_{trade_date}_{config_hash}_{param_hash}_{data_fp}/
    config_hash 自动计算（无需传入），param_hash 区分同一天不同回测参数组合，
    data_fp 区分不同数据范围/股票列表（防止窗口切片缓存污染全量调用）。
    """
    if config_hash is None:
        config_hash = _compute_config_hash()
    base = CACHE_DIR / f"signal_cache_{trade_date}_{config_hash}"
    if param_hash:
        base = CACHE_DIR / f"signal_cache_{trade_date}_{config_hash}_{param_hash}"
    if data_fp:
        base = CACHE_DIR / f"{base.name}_{data_fp}"
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


def _completed_symbols(trade_date: str, param_hash: str | None = None, config_hash: str | None = None, data_fp: str | None = None) -> set[str]:
    cd = _cache_dir_for(trade_date, param_hash, config_hash, data_fp)
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
    if _os.write_mode() == _os.OUTPUT_WRITE_REPLACE:
        # 回退：禁用 upsert，直接替换写（分片前原始行为）
        pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd", compression_level=3)
        return
    # upsert：原子写（tmp + os.replace），同 key 覆写 → 重复运行不产生重复记录，
    # 且不会留下半成品文件被 _completed_symbols 误判为已完成
    _os.atomic_write_parquet(path, pd.DataFrame(rows))


def _load_signal_cache(trade_date: str, param_hash: str | None = None, config_hash: str | None = None, data_fp: str | None = None) -> pd.DataFrame | None:
    cd = _cache_dir_for(trade_date, param_hash, config_hash, data_fp)
    if not cd.exists():
        return None
    files = []
    for bucket_dir in cd.iterdir():
        if bucket_dir.is_dir():
            files.extend(sorted(bucket_dir.glob("*.parquet")))
    if not files:
        return None
    # 分块 concat：一次性读 3155 个文件再整体 concat 会产生双倍内存峰值
    #（parts 列表 + 合并结果），容易在 Windows 上触发 OOM/原生崩溃。
    # 这里按块合并、逐块释放，同时提前丢弃 exit_strategy 字典列（每行一个 dict，
    # 是内存大头，提取出止损价后不再需要）。
    _CHUNK = 400
    parts: list[pd.DataFrame] = []
    df: pd.DataFrame | None = None
    for idx, f in enumerate(files):
        try:
            part = pd.read_parquet(f)
        except Exception:
            logger.warning("跳过损坏的缓存文件: %s", f)
            continue
        if "exit_strategy" in part.columns:
            if "止损价" not in part.columns:
                part["止损价"] = part["exit_strategy"].apply(
                    lambda x: float(x.get("stop_loss", 0)) if isinstance(x, dict) else 0.0
                )
            part = part.drop(columns=["exit_strategy"])
        parts.append(part)
        if len(parts) >= _CHUNK:
            merged = pd.concat(parts, ignore_index=True)
            parts = [merged]
        del part
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    del parts
    # 只重命名确实存在的英文列，避免旧版缓存中文列名冲突
    rename_map = {eng: chn for eng, chn in _REV_SIGNAL_COL_MAP.items() if eng in df.columns and eng != chn}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return _validate_signal_merge(df, cd)


def _validate_signal_merge(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """合并阶段校验唯一性（Task E）：主键 (symbol, trade_date) 重复行去重 + 清单覆盖率。

    重复运行同一片只覆写同名工件，合并后主键唯一；若历史遗留/异常产生重复，
    此处 keep=first 去重并告警，保证下游引擎消费无重复记录。
    """
    _pk = ("symbol", "trade_date")
    if _pk[0] in df.columns and _pk[1] in df.columns:
        _dup = int(df.duplicated(subset=list(_pk), keep="first").sum())
        if _dup:
            logger.warning(f"[output] 信号合并发现 {_dup} 行主键重复，keep=first 去重")
            df = df.drop_duplicates(subset=list(_pk), keep="first").reset_index(drop=True)
    if _os.write_mode() == _os.OUTPUT_WRITE_UPSERT:
        try:
            _manifest = _os.OutputManifest(cache_dir, _os.OutputSchema("signal_cache", _pk, "v1"))
            _records = _manifest.records()
            if _records:
                _found = set(df["symbol"].unique()) if "symbol" in df.columns else set()
                _expected = {r.key for r in _records}
                _missing = sorted(_expected - _found)
                _report = _os.MergeReport(
                    expected=len(_expected),
                    read_ok=len(_expected) - len(_missing),
                    missing=_missing,
                    primary_keys=_pk,
                )
                _os.log_merge(_report, "信号合并")
        except Exception as _e:
            logger.debug(f"[output] 信号清单校验失败: {_e}")
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
    data_version: str | None = None,
    confirmed_suspension_days: set[str] | None = None,
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
    data_fp = _data_fingerprint(kline_df, data_version=data_version)
    cache_tag = f"cfg={config_hash},param={signal_param_hash},data={data_fp}"

    random.seed(42)
    np.random.seed(42)

    # ── Task F 交易日历与停牌标志对齐（CALENDAR_ALIGN_MODE=off 回退老版合并逻辑） ──
    # 真实成交行打标 is_trading/is_suspended（不物化 NaN 行），并按官方日历口径
    # 计算每只股票停牌统计（供 Phase 0 / worker precheck 日历口径 SKIP）。
    # confirmed_suspension_days（官方停牌公告/龙虎榜独立口径）存在时对缺失日做
    # "漏采 vs 停牌"交叉验证，避免数据源漏采被误判为停牌而硬拒。
    _susp_stats: dict[str, dict[str, Any]] = {}
    if _ca.align_enabled():
        _ca.maintain_calendar()
        kline_df = _ca.add_alignment_flags(kline_df)
        _susp_stats = _ca.compute_suspension_stats(
            kline_df, confirmed_suspension_days=confirmed_suspension_days
        )
        try:
            from BackTrading.precheck import suspension_suspects as _suspects

            _susp_suspects = _suspects(_susp_stats)
            if _susp_suspects:
                logger.warning(f"[停牌-疑似漏采] {len(_susp_suspects)} 只高缺失日股票需人工复核"
                               "（真实停牌 vs 数据源漏采，可对比官方停牌公告/龙虎榜）：")
                for _r in _susp_suspects:
                    _cv = (
                        f" 确认停牌={len(_r['confirmed_days'])}天/漏采嫌疑={len(_r['under_collected_days'])}天"
                        if _r["cross_validated"]
                        else " 未交叉验证"
                    )
                    logger.warning(
                        f"  - {_r['symbol']}: 缺失占比={_r['ratio']:.2%}({_r['days']}天) "
                        f"tail={len(_r['tail_days'])}天 interior={len(_r['interior_days'])}天{_cv} "
                        f"缺失日={_r['missing_days'][:8]}{'…' if len(_r['missing_days']) > 8 else ''}"
                    )
        except Exception as _susp_e:
            logger.debug(f"[停牌-疑似漏采] 清单生成跳过: {_susp_e}")

    trade_date = _trade_day_str()
    cache_dir = _cache_dir_for(trade_date, signal_param_hash, config_hash, data_fp)

    symbols = sorted(kline_df["symbol"].unique())
    done = _completed_symbols(trade_date, signal_param_hash, config_hash, data_fp)
    missing = [s for s in symbols if s not in done]

    def _finalize(kline, signal) -> pd.DataFrame:
        merged = _merge_signal(kline, signal)
        del kline, signal
        gc.collect()
        _t_ml = time.time()
        # ── ML 解耦：预测列按数据版本冻结，参数变体不重训 ──
        ml_key = (config_hash, data_fp)
        with _ML_PRED_LOCK:
            ml_pred = _ML_PRED_CACHE.get(ml_key)
        if ml_pred is None:
            logger.info(f"  ML 信号覆写开始（{len(merged)} 行, {merged['symbol'].nunique()} 只），XGBoost 重训一次...")
            merged = apply_ml_signal(merged)
            ml_pred = merged[["symbol", "trade_date", "进场评分"]].copy()
            with _ML_PRED_LOCK:
                _ML_PRED_CACHE[ml_key] = ml_pred
            logger.info(f"  ML 信号覆写完成，耗时 {time.time()-_t_ml:.1f}s（预测已冻结，后续参数变体不再重训）")
        else:
            merged = merged.merge(
                ml_pred.rename(columns={"进场评分": "ML进场评分"}),
                on=["symbol", "trade_date"], how="left",
            )
            _ml_fill = merged["ML进场评分"].notna()
            merged.loc[_ml_fill, "进场评分"] = merged.loc[_ml_fill, "ML进场评分"]
            merged = merged.drop(columns=["ML进场评分"])
            logger.info(f"  ML 预测注入冻结缓存（{int(_ml_fill.sum()):,} 行），耗时 {time.time()-_t_ml:.1f}s")
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
            signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash, data_fp)
            if signal_df is not None:
                return _finalize(kline_df, signal_df)
        else:
            logger.info(f"信号缓存部分命中（{len(done)}/{len(symbols)}），续算 {len(missing)} 只 [{cache_tag}]")

    if not missing:
        logger.info("无需要计算的股票")
        signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash, data_fp)
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
        # fingerprint=data_fp：跨数据批次（WFO 窗口/路径切片）切换时清空指标内存缓存，
        # 防止 worker 复用旧切片的指标导致信号日期错位（2026-08-07 OOS 0 交易根因）。
        if vectorized:
            precompute_all_indicators(stock_dir, fingerprint=data_fp, suspension_stats=_susp_stats)

        from tqdm import tqdm
        signal_pipelines = Config().SIGNAL_PIPELINES
        _workers_per_pipeline = max(
            min((os.cpu_count() or 4) // signal_pipelines, 3), 1
        )

        _worker_fn = _stock_worker_vectorized if vectorized else _stock_worker
        _pipeline_records: list[_os.OutputRecord] = []

        def _pipeline(syms: list[str], idx: int) -> None:
            """单管道：ThreadPoolExecutor 并发处理股票（Windows spawn 下 ProcessPoolExecutor 会死锁）。
            D1 分片：失败片（=失败股票集合）在 SHARD_MAX_ATTEMPTS 内仅重跑失败片。"""
            if not syms:
                return
            pbar = tqdm(total=len(syms), desc=f"管道{idx+1}", unit="只", ncols=50, position=idx)
            pool = ThreadPoolExecutor(max_workers=_workers_per_pipeline)
            _p0 = time.time()
            done = 0
            failures: set[str] = set()
            try:
                def _run_batch(syms_batch: list[str], *, progress: bool = False) -> set[str]:
                    """提交一批符号并收集失败集合（片级失败：仅这批重跑）。"""
                    nonlocal done
                    batch_failures: set[str] = set()
                    fut_to_sym: dict[Any, str] = {}
                    for sym in syms_batch:
                        fut = pool.submit(_worker_fn, sym, stock_dir, params, compute_exit_strategy, _susp_stats)
                        fut_to_sym[fut] = sym

                    for future in as_completed(fut_to_sym):
                        sym = fut_to_sym[future]
                        try:
                            rows = future.result(timeout=120)
                            if rows:
                                _save_stock_signal(cache_dir, sym, rows)
                                _pipeline_records.append(_os.OutputRecord(
                                    shard_id=f"p{idx}", key=sym,
                                    path=str(_symbol_cache_path(cache_dir, sym)),
                                    rows=len(rows),
                                    written_at=time.time(),
                                ))
                        except Exception as e:
                            batch_failures.add(sym)
                            logger.opt(exception=True).warning(f"  [{sym}] 信号计算失败: {e}")
                        if progress:
                            done += 1
                            if done % 500 == 0:
                                logger.info(f"  管道{idx+1} 进度 {done}/{len(syms)} 只（耗时 {time.time()-_p0:.0f}s, 失败 {len(batch_failures)}）")
                            pbar.update(1)
                    return batch_failures

                failures = _run_batch(syms, progress=True)
                # 失败仅重跑失败片（shard=失败股票集合），不重跑成功片
                if failures and Config().SHARD_MODE != "off":
                    max_attempts = max(int(Config().SHARD_MAX_ATTEMPTS), 1)
                    for _a in range(2, max_attempts + 1):
                        if not failures:
                            break
                        time.sleep(0.5)
                        logger.info(f"  管道{idx+1} 重跑失败片（仅失败 {len(failures)} 只，第 {_a}/{max_attempts} 次尝试）...")
                        failures = _run_batch(sorted(failures))
            finally:
                # wait=True：等所有 worker 真正结束再返回，防止残留线程在
                # _load_signal_cache 阶段仍用 pyarrow 写 parquet，与主线程
                # 并发读触发 Windows 原生崩溃（0xC0000005 静默终止）。
                pool.shutdown(wait=True)
            pbar.close()
            logger.info(f"  管道{idx+1} 完成: 成功 {len(syms)-len(failures)}/{len(syms)} 只, 失败 {len(failures)}（耗时 {time.time()-_p0:.0f}s）")
            if failures:
                with open(os.path.join(cache_dir, "_pipeline_failures.txt"), "a", encoding="utf-8") as _f:
                    for _s in sorted(failures):
                        _f.write(f"{_s}\n")

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
            pool_t.shutdown(wait=True)

        # Task E 幂等输出：信号片输出按 (shard_id, key) upsert 到表级清单（同键覆写）
        if _os.write_mode() == _os.OUTPUT_WRITE_UPSERT and _pipeline_records:
            try:
                _schema = _os.OutputSchema("signal_cache", ("symbol", "trade_date"), "v1")
                _manifest = _os.OutputManifest(cache_dir, _schema)
                _manifest.upsert_many(_pipeline_records)
                _merge = _os.validate_artifacts(_manifest.records())
                _os.log_merge(_merge, "信号")
            except Exception as _e:
                logger.warning(f"[output] 信号清单写入失败（不影响计算）: {_e}")
        _t1 = time.time()
        logger.info(f"所有管道完成，耗时 {_t1-_t0:.1f}s")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.debug(f"临时目录已清理: {tmpdir}")

    # ── 加载缓存合并 ──
    logger.info(f"加载信号缓存合并...")
    _t_load = time.time()
    try:
        signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash, data_fp)
    except MemoryError:
        logger.error("读取信号缓存内存不足(MemoryError)，尝试强制 GC 后重试一次")
        gc.collect()
        signal_df = _load_signal_cache(trade_date, signal_param_hash, config_hash, data_fp)
    logger.info(f"读取信号缓存耗时 {time.time()-_t_load:.1f}s")
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
    susp_stats: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    stock_df = pd.read_parquet(os.path.join(stock_dir, f"{symbol}.parquet"), engine="fastparquet")
    if len(stock_df) < 60:
        return []

    # 数据质量检查
    _validate_stock_data(stock_df, symbol)

    # ── 窗口预检（指标计算前）：SKIP → 返回空跳过；NEED_FILL → 限界填充 ──
    # Task F: 日历口径停牌统计（超阈值 → SKIP），无统计回退启发式
    from BackTrading.precheck import apply_precheck as _apply_precheck

    stock_df, _pre_res = _apply_precheck(symbol, stock_df, context="_stock_worker",
                                         suspension_stats=(susp_stats or {}).get(symbol))
    if stock_df.empty:
        return []

    # 所有滚动指标（MA, MACD, ATR, BBANDS 等）只向后看，不存在前瞻偏差。
    # 在全量数据上一次性计算，避免每根 bar 重复 800 次。
    stock_df = _compute_indicators_snapshotted(stock_df, symbol=symbol, context="_stock_worker")

    analyzer = MACDAnalyzer()
    rows: list[dict[str, Any]] = []

    for i in range(len(stock_df)):
        bar = stock_df.iloc[: i + 1]
        try:
            signal = _compute_signal(analyzer, bar, params, compute_exit_strategy)
        except Exception:
            continue
        # ── 置信度消费（指标降级 RELAX/SKIP）：低置信度 bar 抑制进场信号 ──
        from BackTrading.degradation import low_confidence_mask as _low_conf_mask

        if _low_conf_mask(bar).any():
            signal["进场评分"] = 0.0
            signal["score"] = 0.0
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
    susp_stats: dict[str, dict[str, Any]] | None = None,
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
        # 背离检测只依赖 DIF 数据，Phase 0 已缓存，跨参数迭代直接复用
        _div = get_divergence(symbol, stock_df, stock_dir)

        try:
            signal_df = _compute_signals_vec(
                stock_df,
                params=params,
                compute_exit_strategy=compute_exit_strategy,
                diverge_distance=_dd,
                precomputed_divergence=_div,
            )
        except Exception as e:
            import traceback
            logger.warning(f"  [{symbol}] 向量化信号计算失败({e})\n{traceback.format_exc()}")
            try:
                return _stock_worker(symbol, stock_dir, params, compute_exit_strategy, susp_stats)
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


def _compute_indicators_snapshotted(
    df_raw: pd.DataFrame,
    symbol: str = "",
    context: str = "",
) -> pd.DataFrame:
    """_compute_indicators 的失败快照包装（Task A2）。

    指标计算抛异常时，把原始输入快照落盘并返回 snapshot_id 写入日志，
    随后原样重抛，调用方行为不变。
    """
    import traceback as _tb

    from BackTrading.snapshot import save_failure_snapshot

    try:
        return _compute_indicators(df_raw)
    except Exception as exc:
        _sid = save_failure_snapshot(
            ohlcv=df_raw,
            symbol=symbol or None,
            metric_name="compute_indicators",
            error_code="INDICATOR_COMPUTE_FAILED",
            error_message=str(exc),
            traceback_text=_tb.format_exc(),
        )
        _suffix = f" | snapshot_id={_sid}" if _sid else ""
        logger.opt(exception=True).error(
            f"[{symbol or 'unknown'}] 指标计算失败{_suffix}（{context or 'compute'}）: {exc}"
        )
        raise


def _compute_indicators(
    df_raw: pd.DataFrame,
    min_periods: dict[str, int] | None = None,
    confidence_flag: bool = True,
) -> pd.DataFrame:
    """计算全部技术指标（指标降级 + min_periods + 置信度标签）。

    Args:
        min_periods: 每指标最小周期下限（降级时使用），如 {"ma_200": 20, "atr": 5}。
        confidence_flag: 是否生成 bar 级 _IND_CONF 置信度列（策略层消费）。

    Returns:
        df_raw + 指标列；RELAX/SKIP 降级时附加 _IND_CONF（high/low）列与
        df.attrs["_confidence"] = {level, reasons, start_bar}。
    """
    from BackTrading.degradation import (
        degrade_mode,
        safe_adx,
        safe_atr,
        safe_bbands,
        safe_cci,
        safe_ma,
        safe_macd,
        safe_rsi,
        safe_stoch,
    )

    df = df_raw.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    n = len(df)
    mode = degrade_mode()

    def _mp(name: str) -> int | None:
        return (min_periods or {}).get(name)

    def _track(res, degraded_starts: list[int], reasons: list[str]) -> None:
        if res.confidence.is_low:
            degraded_starts.append(res.start_bar)
            reasons.extend(res.confidence.reasons)

    degraded_starts: list[int] = []
    degraded_reasons: list[str] = []

    macd = safe_macd(close, min_periods=_mp("macd"))
    _track(macd, degraded_starts, degraded_reasons)
    if macd.value is not None:
        df["DIF"] = macd.value.iloc[:, 0].values if macd.value.shape[1] >= 1 else 0
        df["DEA"] = macd.value.iloc[:, 1].values if macd.value.shape[1] >= 2 else 0
        hist = macd.value.iloc[:, 2].values if macd.value.shape[1] >= 3 else 0
        df["MACD_HIST"] = 2 * hist if isinstance(hist, np.ndarray) else hist
    else:
        df["DIF"] = 0.0
        df["DEA"] = 0.0
        df["MACD_HIST"] = 0.0

    atr = safe_atr(high, low, close, min_periods=_mp("atr"))
    _track(atr, degraded_starts, degraded_reasons)
    df["ATR"] = atr.value if atr.value is not None else 0.0

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
        res = safe_ma(close, p, mode=mode, min_periods=_mp(f"ma_{p}"))
        _track(res, degraded_starts, degraded_reasons)
        df[f"MA_{p}"] = res.value

    bb = safe_bbands(close, min_periods=_mp("bbands"))
    _track(bb, degraded_starts, degraded_reasons)
    if bb.value is not None and bb.value.shape[1] >= 3:
        df["BBU_20_2.0"] = bb.value.iloc[:, 0].values
        df["BBM_20_2.0"] = bb.value.iloc[:, 1].values
        df["BBL_20_2.0"] = bb.value.iloc[:, 2].values
        df["BOLL_BANDWIDTH"] = (bb.value.iloc[:, 0] - bb.value.iloc[:, 2]) / close

    # 以下指标供 pipeline_analysis._precompute_rule_indicators 使用，
    # 列名必须与 guards 的检查前缀完全匹配，避免每根 bar 重算。
    adx = safe_adx(high, low, close, min_periods=_mp("adx"))
    _track(adx, degraded_starts, degraded_reasons)
    if adx.value is not None:
        df["ADX"] = adx.value.iloc[:, 0].values  # 降级后列名变化，必须按位置取

    res200 = safe_ma(close, 200, mode=mode, min_periods=_mp("ma_200"))
    _track(res200, degraded_starts, degraded_reasons)
    df["MA_200"] = res200.value
    rsi = safe_rsi(close, min_periods=_mp("rsi"))
    _track(rsi, degraded_starts, degraded_reasons)
    if rsi.value is not None:
        df["RSI_14"] = rsi.value.values if isinstance(rsi.value, pd.Series) else rsi.value

    stoch = safe_stoch(high, low, close, min_periods=_mp("stoch"))
    _track(stoch, degraded_starts, degraded_reasons)
    if stoch.value is not None:
        for c in stoch.value.columns:
            df[c] = stoch.value[c].to_numpy()

    df["AMOUNT"] = close * df["volume"]
    df["AMOUNT_MA20"] = df["AMOUNT"].rolling(20).mean()
    df["AMPLITUDE_PCT"] = (high - low) / close

    cci = safe_cci(high, low, close, min_periods=_mp("cci"))
    _track(cci, degraded_starts, degraded_reasons)
    if cci.value is not None:
        df["CCI_20"] = cci.value.values if isinstance(cci.value, pd.Series) else cci.value

    # ── 置信度标签：bar 级 _IND_CONF + attrs 汇总（策略层消费） ──
    if confidence_flag:
        df["_IND_CONF"] = "high"
        if degraded_starts:
            _start = min(degraded_starts)
            conf_col = pd.Series("high", index=df.index)
            conf_col.iloc[max(0, _start):] = "low"
            df["_IND_CONF"] = conf_col
            df.attrs["_confidence"] = {
                "level": "low",
                "reasons": degraded_reasons,
                "start_bar": int(_start),
                "mode": mode.value,
            }
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
