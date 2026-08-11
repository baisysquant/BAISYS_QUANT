"""Task A2 — 失败快照持久化（Failure Snapshot Persistence）。

窗口计算返回"无效"或抛异常时，将可复现的输入快照写入本地缓存 bucket，
返回 snapshot_id 供日志/告警正文引用；支持按 snapshot_id 本地复现（load_snapshot），
支持过期快照清理（cleanup_snapshots，回退路径：临时落盘 + 告警清理）。

配置项（[BACKTEST] 节）:
    snapshot_enabled = true             # 开关（默认开启）
    snapshot_max_rows = 200             # OHLCV 截断行数（schema: 最近 3×window 或 N=200）
    snapshot_retention_days = 14        # 保留天数，过期自动清理并告警

存储布局（按 bucket 分桶，与 indicator_cache 同风格）:
    <CACHE_DIRECTORY>/failure_snapshots/<yyyy-mm-dd>/<bucket>/<snapshot_id>.parquet  # OHLCV
                                                              <snapshot_id>.json    # 全量元数据
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
import traceback as _traceback_mod
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_SCHEMA_VERSION = 1
# 每次 WFO/管线会话内快照上限：10h 级失败运行中窗口可能数百个，
# 全部落盘会污染缓存目录；超过上限后仅告警不落盘（会话由 begin_snapshot_session 重置）。
_SESSION_CAP = 500
_session_count = 0

# 进程级上下文（runner 入口注入，WFO/worker 无需透传）
_RUN_ID: str = ""
_TASK_ID: str = ""

# 参与快照的行情列（其余指标列不落盘，保证"原始输入可复现"且体积受控）
_OHLCV_COLS = ("trade_date", "symbol", "open", "high", "low", "close", "volume",
               "amount", "pre_close", "turnover_rate")


@dataclass
class FailureSnapshot:
    """失败快照元数据（对应 Task A2 schema 必含字段）。"""

    snapshot_id: str = ""
    run_id: str = ""
    task_id: str = ""
    symbol: str | None = None
    market: str = "未知"
    window_name: str = ""
    window_start: str = ""
    window_end: str = ""
    window_size: int = 0
    metric_name: str = ""
    ohlcv_rows: int = 0
    ohlcv_columns: list[str] = field(default_factory=list)
    trade_calendar_slice: list[str] = field(default_factory=list)
    sample_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    precheck_status: dict[str, Any] | None = None
    adjust_factors: dict[str, Any] | None = None
    worker_id: str = ""
    hostname: str = ""
    pid: int = 0
    container_id: str | None = None
    memory_cpu_snapshot: dict[str, Any] = field(default_factory=dict)
    traceback: str = ""
    error_code: str = ""
    error_message: str = ""
    timestamp: str = ""
    storage_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            **{k: getattr(self, k) for k in (
                "snapshot_id", "run_id", "task_id", "symbol", "market",
                "window_name", "window_start", "window_end", "window_size",
                "metric_name", "ohlcv_rows", "ohlcv_columns",
                "trade_calendar_slice", "sample_counts", "precheck_status",
                "adjust_factors", "worker_id", "hostname", "pid", "container_id",
                "memory_cpu_snapshot", "traceback", "error_code",
                "error_message", "timestamp", "storage_dir",
            )},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FailureSnapshot:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def set_run_context(run_id: str = "", task_id: str = "") -> None:
    """注入进程级 run_id / task_id（runner 入口调用一次，随 log/告警透出）。"""
    global _RUN_ID, _TASK_ID
    if run_id:
        _RUN_ID = run_id
    if task_id:
        _TASK_ID = task_id


def begin_snapshot_session() -> None:
    """开启一次 WFO/管线会话：重置会话计数 + 顺带清理过期快照。"""
    global _session_count
    _session_count = 0
    try:
        cleanup_snapshots()
    except Exception as exc:
        logger.warning(f"失败快照过期清理异常: {exc}")


# ── 配置读取（全部带兜底，任何配置缺失都不影响快照主流程） ────────────

def _snapshot_enabled() -> bool:
    try:
        from UtilsManager.ConfigParser import Config
        return bool(Config().app_config.backtest.SNAPSHOT_ENABLED)
    except Exception:
        return True


def _snapshot_max_rows() -> int:
    try:
        from UtilsManager.ConfigParser import Config
        return int(Config().app_config.backtest.SNAPSHOT_MAX_ROWS)
    except Exception:
        return 200


def _snapshot_retention_days() -> int:
    try:
        from UtilsManager.ConfigParser import Config
        return int(Config().app_config.backtest.SNAPSHOT_RETENTION_DAYS)
    except Exception:
        return 14


def _snapshot_root() -> Path:
    try:
        from UtilsManager.ConfigParser import Config
        base = Path(Config().CACHE_DIRECTORY)
    except Exception:
        base = Path(__file__).resolve().parent / "data"
    return base / "failure_snapshots"


# ── 运行时信息 ────────────────────────────────────────────────────────

def _market_of(symbol: str | None) -> str:
    s = (symbol or "").lower()
    if s.startswith("sh") or s.startswith("6"):
        return "上交所"
    if s.startswith("sz") or s.startswith(("0", "3")):
        return "深交所"
    if s.startswith(("bj", "4", "8")):
        return "北交所"
    return "未知"


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname() or os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown",
        "pid": os.getpid(),
        "worker_id": "",
        "container_id": None,
        "memory_cpu_snapshot": {"rss_mb": None, "cpu_percent": None},
    }
    try:
        import multiprocessing
        info["worker_id"] = f"{multiprocessing.current_process().name}:{threading.current_thread().name}"
    except Exception:
        info["worker_id"] = "main"
    try:
        cgroup = Path("/proc/self/cgroup")
        if cgroup.exists():
            for line in cgroup.read_text(errors="ignore").splitlines():
                cid = line.rsplit("/", 1)[-1].strip()
                if len(cid) == 64 and all(c in "0123456789abcdef" for c in cid):
                    info["container_id"] = cid
                    break
    except Exception:
        pass
    if info["container_id"] is None:
        info["container_id"] = os.environ.get("CONTAINER_ID") or os.environ.get("HOSTNAME") or None
    try:
        import psutil  # type: ignore
        _proc = psutil.Process(os.getpid())
        info["memory_cpu_snapshot"] = {
            "rss_mb": round(_proc.memory_info().rss / (1024 * 1024), 2),
            "cpu_percent": _proc.cpu_percent(interval=None),
        }
    except Exception:
        info["memory_cpu_snapshot"] = {"rss_mb": None, "cpu_percent": None, "note": "psutil 不可用"}
    return info


# ── 数据摘要工具 ──────────────────────────────────────────────────────

def _max_consecutive_non_nan(series: pd.Series) -> int:
    """连续非 NaN 最大长度（标量向量化实现）。"""
    if series.empty:
        return 0
    m = series.notna().astype(int).to_numpy(dtype=np.int8)
    diff = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return 0
    return int((ends - starts).max())


def _sample_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for col in df.columns:
        s = df[col]
        counts[str(col)] = {
            "non_nan": int(s.notna().sum()),
            "max_consecutive_non_nan": _max_consecutive_non_nan(s),
        }
    return counts


def _truncate_ohlcv(df: pd.DataFrame | None, max_rows: int) -> pd.DataFrame:
    """只保留 OHLCV 行情列并截断为最近 max_rows 行（schema: 3×window 或 N=200）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in _OHLCV_COLS if c in df.columns]
    out = df[cols].copy() if cols else df.copy()
    if len(out) > max_rows:
        out = out.iloc[-max_rows:]
    out = out.reset_index(drop=True)
    return out


def _calendar_slice(df: pd.DataFrame) -> list[str]:
    if df.empty or "trade_date" not in df.columns:
        return []
    try:
        return sorted({str(d).split("T")[0][:10] for d in df["trade_date"].dropna().unique()})
    except Exception:
        return []


def _snapshot_id(symbol: str | None, window_name: str, metric_name: str,
                 error_code: str, timestamp: str) -> str:
    raw = f"{symbol or ''}|{window_name}|{metric_name}|{error_code}|{timestamp}|{uuid.uuid4().hex[:8]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ── 核心持久化接口 ────────────────────────────────────────────────────

def save_failure_snapshot(
    *,
    ohlcv: pd.DataFrame | None = None,
    symbol: str | None = None,
    market: str | None = None,
    window_name: str = "",
    window_start: str = "",
    window_end: str = "",
    window_size: int = 0,
    metric_name: str = "",
    error_code: str = "UNKNOWN",
    error_message: str = "",
    traceback_text: str | None = None,
    run_id: str = "",
    task_id: str = "",
    precheck_status: dict[str, Any] | None = None,
    adjust_factors: dict[str, Any] | None = None,
    snapshot_dir: str | Path | None = None,
) -> str | None:
    """把失败输入快照写入本地缓存 bucket，返回 snapshot_id；未启用/超上限返回 None。

    所有参数均有兜底；写盘任何一步失败只告警不抛异常（快照绝不能拖垮主流程）。
    """
    global _session_count
    if not _snapshot_enabled():
        return None
    if _session_count >= _SESSION_CAP:
        if _session_count == _SESSION_CAP:
            logger.warning(
                f"失败快照超过单会话上限 {_SESSION_CAP} 个，后续快照放弃落盘"
                f"（请清理 failure_snapshots 目录或排查窗口系统性失败）"
            )
        _session_count += 1
        return None
    _session_count += 1

    try:
        ts = datetime.now().isoformat(timespec="milliseconds")
        sid = _snapshot_id(symbol, window_name, metric_name, error_code, ts)
        if not run_id:
            run_id = _RUN_ID or uuid.uuid4().hex[:12]
        if not task_id:
            task_id = _TASK_ID or ""

        df = _truncate_ohlcv(ohlcv, _snapshot_max_rows())
        rt = _runtime_info()
        meta = FailureSnapshot(
            snapshot_id=sid,
            run_id=run_id,
            task_id=task_id,
            symbol=symbol,
            market=market or _market_of(symbol),
            window_name=window_name,
            window_start=window_start,
            window_end=window_end,
            window_size=int(window_size) if window_size else len(ohlcv) if ohlcv is not None else 0,
            metric_name=metric_name,
            ohlcv_rows=len(df),
            ohlcv_columns=list(df.columns),
            trade_calendar_slice=_calendar_slice(df),
            sample_counts=_sample_counts(df),
            precheck_status=precheck_status,
            adjust_factors=adjust_factors,
            worker_id=rt["worker_id"],
            hostname=rt["hostname"],
            pid=rt["pid"],
            container_id=rt["container_id"],
            memory_cpu_snapshot=rt["memory_cpu_snapshot"],
            traceback=traceback_text or "",
            error_code=error_code,
            error_message=str(error_message),
            timestamp=ts,
        )

        if snapshot_dir is None:
            root = _snapshot_root() / date.today().isoformat()
            bucket = (symbol or "all")[:2].lower()
            snapshot_dir = root / bucket
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        meta.storage_dir = str(snapshot_dir)

        df_path = snapshot_dir / f"{sid}.parquet"
        meta_path = snapshot_dir / f"{sid}.json"
        if not df.empty:
            df.to_parquet(df_path, index=False, compression="zstd", compression_level=3)
        meta_path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return sid
    except Exception as exc:
        logger.opt(exception=True).warning(f"失败快照写入异常（不影响主流程）: {exc}")
        return None


def load_snapshot(snapshot_id: str, root: str | Path | None = None) -> tuple[pd.DataFrame, FailureSnapshot]:
    """按 snapshot_id 本地复现：返回 (OHLCV DataFrame, 元数据)。不存在时抛 FileNotFoundError。"""
    root = Path(root) if root else _snapshot_root()
    meta_path = next(root.rglob(f"{snapshot_id}.json"), None)
    if meta_path is None:
        raise FileNotFoundError(f"快照 {snapshot_id} 不存在于 {root}")
    meta = FailureSnapshot.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
    df = pd.DataFrame()
    parquet_path = meta_path.with_suffix(".parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    return df, meta


def find_snapshots(root: str | Path | None = None) -> list[FailureSnapshot]:
    """列出全部快照元数据（按时间倒序）。"""
    root = Path(root) if root else _snapshot_root()
    out: list[FailureSnapshot] = []
    if not root.exists():
        return out
    for p in root.rglob("*.json"):
        try:
            out.append(FailureSnapshot.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    out.sort(key=lambda m: m.timestamp, reverse=True)
    return out


def cleanup_snapshots(older_than_days: int | None = None, root: str | Path | None = None) -> int:
    """删除超过保留天数的快照（回退路径：本地落盘 + 告警清理）。

    Returns:
        删除的快照（json 文件）数量。
    """
    retention = older_than_days if older_than_days is not None else _snapshot_retention_days()
    root = Path(root) if root else _snapshot_root()
    if not root.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=max(1, retention))
    removed = 0
    for date_dir in root.iterdir():
        if not date_dir.is_dir():
            continue
        try:
            d = datetime.strptime(date_dir.name, "%Y-%m-%d")
        except ValueError:
            continue
        if d >= cutoff:
            continue
        n = len(list(date_dir.rglob("*.json")))
        try:
            import shutil
            shutil.rmtree(date_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning(f"失败快照清理异常 {date_dir}: {exc}")
            continue
        removed += n
    if removed:
        logger.warning(
            f"已清理 {removed} 个过期失败快照（> {retention} 天，位于 {root}），"
            f"如仍需复现请在此之前导出"
        )
    return removed
