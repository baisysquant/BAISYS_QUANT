"""Task F: 交易日历与停牌标志对齐 — 日历维护 + 对齐层。

职责：
    1. 官方交易日历维护：仅 maintain_calendar() 触发网络拉取（akshare），
       get_official_calendar() 为只读路径（注入覆盖 → 内存 → 磁盘缓存 → 空集，
       绝不拉网），保证回测/引擎在缓存缺失时静默降级、测试环境零网络。
    2. 日线合并对齐：add_alignment_flags() 在合并帧上加 is_trading/is_suspended
       标志列（真实成交行 = 1/0；缺失停牌日不物化 NaN 行，避免污染因子/引擎数值）。
    3. 停牌统计：compute_suspension_stats() 按官方日历口径计算每只股票
       [上市区间内] 的停牌日集合与停牌占比（日历日轴状态向量），供 precheck
       （比例超阈值 → SKIP）与引擎（不可成交）消费。

回退开关: [BACKTEST] CALENDAR_ALIGN_MODE = on（对齐，默认）/ off（老版合并逻辑：
无标志列、precheck 回退零成交+横盘启发式、引擎按数据日轴迭代）。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# 模块级内存缓存：官方交易日集合 + 加载时刻
_OVERRIDE: set[str] | None = None          # 注入覆盖（测试钩子 / 运行时指定）
_mem_dates: set[str] | None = None
_mem_loaded_at: float | None = None


def _calendar_ttl_seconds() -> float:
    try:
        from UtilsManager.ConfigParser import Config
        return float(Config().CALENDAR_TTL_HOURS) * 3600.0
    except Exception:
        return 24.0 * 3600.0


def set_official_calendar(dates: set[str] | None) -> None:
    """注入/清除官方交易日集合（测试钩子；None 清除注入恢复自动模式）。"""
    global _OVERRIDE
    _OVERRIDE = set(dates) if dates else None
    if _OVERRIDE:
        logger.debug(f"[Calendar] 已注入官方日历 {len(_OVERRIDE)} 日")


def _disk_cache_path() -> str | None:
    try:
        from UtilsManager.ConfigParser import Config
        return os.path.join(Config().CACHE_DIRECTORY, "calendar", "official_trading_dates.json")
    except Exception:
        return None


def _read_disk_cache() -> set[str] | None:
    """只读 TradingCalendarAnalyzer 的本地缓存文件（不拉取网络）。"""
    path = _disk_cache_path()
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        dates = {str(d) for d in data.get("dates", [])}
        return dates or None
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as e:
        logger.debug(f"[Calendar] 读取本地日历缓存失败: {e}")
        return None


def _mem_fresh() -> bool:
    if _mem_dates is None or _mem_loaded_at is None:
        return False
    return (time.time() - _mem_loaded_at) < _calendar_ttl_seconds()


def get_official_calendar() -> set[str]:
    """只读获取官方交易日集合（SH+SZ）。绝不触发网络拉取；不可用时返回空集。"""
    global _mem_dates, _mem_loaded_at
    if _OVERRIDE is not None:
        return set(_OVERRIDE)
    if _mem_fresh():
        return set(_mem_dates)
    disk = _read_disk_cache()
    if disk:
        _mem_dates, _mem_loaded_at = set(disk), time.time()
        return set(disk)
    return set()


def maintain_calendar(force: bool = False) -> int:
    """维护官方交易日历：注入覆盖 → 内存/磁盘新鲜缓存 → akshare 拉取并落盘。

    返回日期数量；全部不可用时返回 0（调用方静默降级为老版行为）。
    """
    if _OVERRIDE is not None:
        global _mem_dates, _mem_loaded_at
        _mem_dates, _mem_loaded_at = set(_OVERRIDE), time.time()
        return len(_mem_dates)
    if not force and _mem_fresh():
        return len(_mem_dates)
    disk = _read_disk_cache()
    if disk and not force:
        _mem_dates, _mem_loaded_at = set(disk), time.time()
        logger.info(f"[Calendar] 日历维护命中本地缓存（{len(disk)} 日）")
        return len(disk)
    try:
        from DataCollection.CalendarManager import TradingCalendarAnalyzer
        dates = TradingCalendarAnalyzer().get_official_trading_dates()
    except Exception as e:
        logger.error(f"[Calendar ERROR] 维护拉取失败: {e}")
        dates = None
    if dates:
        _mem_dates, _mem_loaded_at = set(dates), time.time()
        logger.info(f"[Calendar] 日历维护完成（{len(dates)} 日）")
        return len(dates)
    stale = _read_disk_cache()
    if stale:
        _mem_dates, _mem_loaded_at = set(stale), time.time()
        logger.warning("[Calendar WARN] 维护失败，使用过期本地缓存")
        return len(stale)
    logger.warning("[Calendar WARN] 维护失败：缓存与接口均不可用")
    return 0


def align_enabled() -> bool:
    """对齐开关（回退开关）：[BACKTEST] CALENDAR_ALIGN_MODE = on（默认）/ off。"""
    try:
        from UtilsManager.ConfigParser import Config
        return str(Config().CALENDAR_ALIGN_MODE).strip().lower() == "on"
    except Exception:
        return False


def add_alignment_flags(kline_df: pd.DataFrame) -> pd.DataFrame:
    """在合并帧上打标 is_trading / is_suspended（真实成交行 = True/False）。

    幂等：列已存在时不重复添加。不做缺失位置物化（不插入 NaN 行）。
    """
    if "is_trading" not in kline_df.columns:
        kline_df["is_trading"] = True
    if "is_suspended" not in kline_df.columns:
        kline_df["is_suspended"] = False
    return kline_df


def _missing_blocks(span: list[str], present: set[str]) -> list[dict[str, Any]]:
    """span 内连续缺失块。

    说明：span 截止于该股票自身最末观测日（last 必为 present），故所有块都位于
    两端有成交的内部——无法凭位置区分"已复牌的停牌段"与"历史漏采"，需外部
    （停牌公告/龙虎榜）交叉验证或人工复核。
    """
    blocks: list[dict[str, Any]] = []
    run: list[str] = []

    def _flush() -> None:
        nonlocal run
        if not run:
            return
        blocks.append({"start": run[0], "end": run[-1], "days": len(run)})
        run = []

    for d in span:
        if d in present:
            _flush()
        else:
            run.append(d)
    _flush()
    return blocks


def compute_suspension_stats(
    kline_df: pd.DataFrame,
    official_dates: set[str] | None = None,
    confirmed_suspension_days: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """按官方日历口径计算每只股票的停牌统计（日历日轴状态向量）。

    缺失日可能来自两类成因，无法仅凭 K 线区分，故输出分类供下游/人工复核：
      - 真实停牌（停牌日）；
      - 数据源漏采（IncrementalSyncEngine 增量窗口仅 OVERLAP_DAYS=15 天，
        早于窗口的历史缺口永久遗留，被误计为"停牌"）。
    分类输出：
      - suspension_ratio：span（=股票自身首末观测日之间）内缺失占比，口径不变；
      - interior_missing_days：span 内缺失日（两端均有成交，已复牌停牌 或 历史漏采）；
      - tail_missing_days：该股票末观测日之后、全池最末日之前 官方交易日
        （同期其他股票有行情而该股票无行 → 增量同步漏采 或 停牌中，供人工复核）；
      - cross_validated：是否与独立口径（官方停牌公告/龙虎榜）交叉验证。

    Args:
        kline_df: 合并 K 线（须含 symbol/trade_date）。
        official_dates: 官方交易日集合；None 时取 get_official_calendar()。
        confirmed_suspension_days: 已确认停牌日集合。提供时对 span 内缺失日做
            "漏采 vs 停牌"交叉验证：confirmed_days = 缺失 ∩ 确认停牌；
            under_collected_days = 缺失 − 确认停牌。

    Returns:
        {symbol: {
            "span_trading_days": int,
            "suspended_days": [str],
            "suspension_ratio": float,
            "missing_blocks": [{"start","end","days"}],
            "interior_missing_days": [str],
            "tail_missing_days": [str],
            "cross_validated": bool,
            # cross_validated=True 时追加:
            "confirmed_days": [str], "under_collected_days": [str],
            "suspension_ratio_confirmed": float, "under_collection_ratio": float,
        }}
        官方日历不可用或输入缺列时返回 {}（调用方回退启发式）。
    """
    if kline_df is None or kline_df.empty:
        return {}
    if "symbol" not in kline_df.columns or "trade_date" not in kline_df.columns:
        return {}
    if official_dates is None:
        official_dates = get_official_calendar()
    if not official_dates:
        return {}

    cal = np.array(sorted(official_dates))  # ISO 字符串按字典序 = 时间序
    per_sym: dict[str, list[str]] = {}
    sample_last = ""
    for sym, grp in kline_df.groupby("symbol", sort=True):
        ds = sorted({str(d)[:10] for d in grp["trade_date"].astype(str).tolist()})
        if ds:
            per_sym[str(sym)] = ds
            if ds[-1] > sample_last:
                sample_last = ds[-1]

    stats: dict[str, dict[str, Any]] = {}
    for sym, ds in per_sym.items():
        first, last = ds[0], ds[-1]
        lo = int(cal.searchsorted(first, side="left"))
        hi = int(cal.searchsorted(last, side="right"))
        if hi <= lo:
            continue
        span = list(cal[lo:hi])
        present = set(ds)
        suspended = [d for d in span if d not in present]
        tail: list[str] = []
        if sample_last > last:
            hi_t = int(cal.searchsorted(sample_last, side="right"))
            tail = [d for d in cal[hi:hi_t] if d not in present]
        stat: dict[str, Any] = {
            "span_trading_days": int(len(span)),
            "suspended_days": suspended,
            "suspension_ratio": round(len(suspended) / len(span), 6),
            "missing_blocks": _missing_blocks(span, present),
            "interior_missing_days": suspended,
            "tail_missing_days": tail,
        }
        if confirmed_suspension_days:
            confirmed = [d for d in suspended if d in confirmed_suspension_days]
            under_collected = [d for d in suspended if d not in confirmed_suspension_days]
            stat.update({
                "cross_validated": True,
                "confirmed_days": confirmed,
                "under_collected_days": under_collected,
                "suspension_ratio_confirmed": round(len(confirmed) / len(span), 6),
                "under_collection_ratio": round(len(under_collected) / len(span), 6),
            })
        else:
            stat["cross_validated"] = False
        stats[sym] = stat
    return stats
