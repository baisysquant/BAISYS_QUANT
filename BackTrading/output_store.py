"""Task E: 幂等输出与中间结果持久化（shard 输出存储）。

分片（shard）输出按 (表, 主键, 版本) 定义存储 schema，写入采用两种模式：
- upsert（默认）：工件先写临时文件再原子 rename（无半成品），随后按
  (shard_id, key) upsert 到表级清单（同键覆写 → 重复运行同一片不产生重复记录）；
- replace（回退）：禁用 upsert，直接替换写（覆盖同名工件，无清单/无校验）。

合并阶段（merge）校验唯一性：
- 清单内重复键（dup_keys）；
- 合并数据主键重复行（dup_rows，默认 keep=first 去重）；
- 工件可读性（missing）→ 合并成功率 = 成功读取 / 期望（验收：正常流程 100%）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_WRITE_UPSERT = "upsert"
OUTPUT_WRITE_REPLACE = "replace"
VALID_WRITE_MODES = (OUTPUT_WRITE_UPSERT, OUTPUT_WRITE_REPLACE)


@lru_cache(maxsize=1)
def write_mode() -> str:
    """输出写入模式：upsert（原子写 + 清单去重）/ replace（直接替换写，禁用 upsert 回退）。"""
    try:
        from UtilsManager.ConfigParser import Config

        m = str(Config().OUTPUT_WRITE_MODE).strip().lower()
        return m if m in VALID_WRITE_MODES else OUTPUT_WRITE_UPSERT
    except Exception:
        return OUTPUT_WRITE_UPSERT


# ── 输出存储 schema：表名 / 主键 / 版本 ──────────────────────────────

@dataclass(frozen=True)
class OutputSchema:
    """输出存储 schema 定义。

    Attributes:
        table: 表名（如 indicator_cache / signal_cache）。
        primary_keys: 合并阶段唯一性校验的主键（如 ("symbol", "trade_date")）。
        version: schema 版本（v1）。
    """

    table: str
    primary_keys: tuple[str, ...]
    version: str = "v1"


@dataclass(frozen=True)
class OutputRecord:
    """一条片输出记录（清单最小单元）。

    Attributes:
        shard_id: 产出该输出的片 ID。
        key: 输出键（如 symbol），与 (shard_id, key) 构成清单主键。
        path: 工件落盘路径（空字符串 = 空输出，如数据不足的股票）。
        rows: 工件行数。
        fingerprint: 输入数据指纹（同键不同数据 → 覆写为新版输出）。
        written_at: 写入时间戳。
    """

    shard_id: str
    key: str
    path: str = ""
    rows: int = 0
    fingerprint: str = ""
    written_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "key": self.key,
            "path": self.path,
            "rows": self.rows,
            "fingerprint": self.fingerprint,
            "written_at": self.written_at,
        }


# ── 原子写入：写临时文件 + os.replace（同 key 覆写，无半成品） ──────────

def _tmp_path(path: Path) -> Path:
    return path.parent / f".{path.name}.tmp{os.getpid()}"


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_parquet(
    path: str | Path,
    df: pd.DataFrame,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p)
    try:
        df.to_parquet(tmp, index=False, compression=compression, compression_level=compression_level)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_npy(path: str | Path, arr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p)
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)  # 文件对象：不会自动追加 .npy 后缀
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_npz(path: str | Path, **arrays: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)  # 文件对象：不会自动追加 .npz 后缀
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


# ── 表级输出清单（upsert 存储） ─────────────────────────────────────

class OutputManifest:
    """表级输出清单（upsert）。

    存储: {root}/manifest/{table}.v{version}.{batch_id}.json
    主键 (shard_id, key)：同键重复写入 = 覆写，重复运行同一片不产生重复记录。
    提交为原子写（tmp + os.replace），线程安全。
    """

    def __init__(self, root: str | Path, schema: OutputSchema, batch_id: str = "") -> None:
        self.schema = schema
        self.batch_id = batch_id
        self._lock = threading.RLock()
        name = f"{schema.table}.v{schema.version}"
        if batch_id:
            name = f"{name}.{batch_id}"
        self._path = Path(root) / "manifest" / f"{name}.json"
        self._records: dict[tuple[str, str], OutputRecord] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if data.get("schema") == self.schema.table and data.get("version") == self.schema.version:
                    for rec in data.get("records", []):
                        key = (rec.get("shard_id", ""), rec.get("key", ""))
                        if not key[1]:
                            continue
                        self._records[key] = OutputRecord(
                            shard_id=rec.get("shard_id", ""),
                            key=rec.get("key", ""),
                            path=rec.get("path", ""),
                            rows=int(rec.get("rows", 0) or 0),
                            fingerprint=rec.get("fingerprint", ""),
                            written_at=float(rec.get("written_at", 0) or 0),
                        )
        except Exception as e:
            logger.warning(f"[output] manifest {self._path.name} 损坏，重置: {e}")

    def upsert(self, record: OutputRecord) -> None:
        with self._lock:
            self._records[(record.shard_id, record.key)] = record
            self._save()

    def upsert_many(self, records: Iterable[OutputRecord]) -> None:
        with self._lock:
            for r in records:
                self._records[(r.shard_id, r.key)] = r
            self._save()

    def _save(self) -> None:
        payload = {
            "schema": self.schema.table,
            "version": self.schema.version,
            "batch_id": self.batch_id,
            "primary_keys": list(self.schema.primary_keys),
            "records": [r.to_dict() for r in self.records()],
        }
        atomic_write_text(self._path, json.dumps(payload, ensure_ascii=False, indent=1))

    def records(self) -> list[OutputRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: (r.shard_id, r.key))

    def keys(self) -> set[str]:
        with self._lock:
            return {r.key for r in self._records.values()}

    def validate_unique(self) -> list[str]:
        """清单文件内的重复键（正常 upsert 不可能产生，防御历史/并发写坏文件）。"""
        dups: list[str] = []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            seen: set[str] = set()
            for rec in data.get("records", []):
                k = rec.get("key", "")
                if k in seen:
                    dups.append(k)
                seen.add(k)
        except Exception:
            pass
        return dups


# ── 合并阶段：校验唯一性并合并 ──────────────────────────────────────

@dataclass
class MergeReport:
    """一次合并校验报告。"""

    expected: int = 0
    read_ok: int = 0
    missing: list[str] = field(default_factory=list)
    dup_keys: list[str] = field(default_factory=list)
    dup_rows: int = 0
    primary_keys: tuple[str, ...] = ()

    @property
    def success_rate(self) -> float:
        if self.expected <= 0:
            return 100.0
        return round(100.0 * self.read_ok / self.expected, 2)

    @property
    def ok(self) -> bool:
        return self.success_rate == 100.0 and not self.dup_keys and self.dup_rows == 0


def merge_outputs(
    records: Sequence[OutputRecord],
    read_fn: Callable[[str], pd.DataFrame | None],
    primary_keys: tuple[str, ...],
    *,
    dedupe: bool = True,
) -> tuple[pd.DataFrame | None, MergeReport]:
    """合并片输出并校验唯一性（合并阶段）。

    read_fn(path) -> DataFrame；读取异常/文件缺失计为 missing（空输出 rows=0 豁免）。
    主键重复行 keep=first 去重并计入 dup_rows；清单内同 key 重复计为 dup_keys。
    返回 (合并结果 或 None, 校验报告)。
    """
    report = MergeReport(expected=len(records), primary_keys=primary_keys)
    parts: list[pd.DataFrame] = []
    seen: set[str] = set()
    for rec in sorted(records, key=lambda r: (r.key, r.shard_id)):
        if rec.key in seen:
            report.dup_keys.append(rec.key)
            continue
        seen.add(rec.key)
        if not rec.path or not Path(rec.path).exists():
            if rec.rows == 0:
                report.read_ok += 1  # 空输出（如数据不足）视为成功
            else:
                report.missing.append(rec.key)
            continue
        try:
            df = read_fn(rec.path)
        except Exception as e:
            logger.warning(f"[output] 工件读取失败 {rec.path}: {e}")
            report.missing.append(rec.key)
            continue
        report.read_ok += 1
        if df is not None and len(df):
            parts.append(df)
    if not parts:
        return None, report
    merged = pd.concat(parts, ignore_index=True)
    if primary_keys and all(c in merged.columns for c in primary_keys):
        dup_mask = merged.duplicated(subset=list(primary_keys), keep="first")
        report.dup_rows = int(dup_mask.sum())
        if dedupe and report.dup_rows:
            merged = merged.drop_duplicates(subset=list(primary_keys), keep="first").reset_index(drop=True)
    return merged, report


def validate_artifacts(records: Sequence[OutputRecord]) -> MergeReport:
    """轻量合并校验（不读内容）：工件存在性 + 清单唯一性 + 空输出豁免。"""
    report = MergeReport(expected=len(records))
    seen: set[str] = set()
    for rec in sorted(records, key=lambda r: (r.key, r.shard_id)):
        if rec.key in seen:
            report.dup_keys.append(rec.key)
            continue
        seen.add(rec.key)
        if (rec.path and Path(rec.path).exists()) or rec.rows == 0:
            report.read_ok += 1
        else:
            report.missing.append(rec.key)
    return report


def log_merge(report: MergeReport, stage: str) -> None:
    """合并校验结果日志（成功率 < 100% 或出现重复 → warning）。"""
    msg = (
        f"[output] {stage} 合并校验: 期望 {report.expected} 工件, "
        f"成功 {report.read_ok}, 缺失 {len(report.missing)}, "
        f"重复键 {len(report.dup_keys)}, 重复行 {report.dup_rows}, "
        f"合并成功率 {report.success_rate}%"
    )
    if report.ok:
        logger.info(msg)
    else:
        logger.warning(msg + (f" missing={report.missing}" if report.missing else ""))
