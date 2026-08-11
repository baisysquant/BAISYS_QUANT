"""D1 分片执行框架（shard）。

大盘任务按 symbol / date 维度切成独立片（shard）：每片独立执行、可断点续跑，
失败仅重跑失败片（同 fingerprint 的 DONE 片直接跳过），并支持临时禁用
（off → 单任务顺序执行，行为与分片前完全一致）。

约定：
- 片间无共享可变状态（调用方保证）。指标/信号缓存均按 symbol 幂等落盘，
  因此按 symbol 分片天然满足幂等性；分片粒度：
    symbol  首选。Phase 0 指标预计算 / 信号计算（prepare 的 pipeline）。
    date    次选。WFO path/window 为天然日期片（v1 提供分区器，执行接入点预留）。
- 断点续跑：任务级 checkpoint JSON 记录每片状态；同 fingerprint 的 DONE 片直接
  跳过；fingerprint 变化 → 全部片重置为 PENDING（输入已变，整批重跑）。
- 重试：FAILED 片在 max_attempts 内重跑（跨进程重启后由 checkpoint 继续累计
  尝试次数）。
- 回退：SHARD_MODE=off → 单任务模式，不写 checkpoint，串行执行全部片。

Windows 注意：进程池（spawn）会死锁，调度器只用线程（与 prepare.py 一致）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

SHARD_DIM_SYMBOL = "symbol"
SHARD_DIM_DATE = "date"

VALID_DIMENSIONS = (SHARD_DIM_SYMBOL, SHARD_DIM_DATE)
VALID_MODES = ("off", "symbol", "hybrid")


class ShardState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class ShardExecutionError(RuntimeError):
    """分片执行失败（尝试次数用尽后仍有失败片，且无原始异常可重抛时）。"""

    def __init__(self, failed_ids: list[str], message: str | None = None) -> None:
        self.failed_ids = list(failed_ids)
        super().__init__(message or f"分片执行失败: {','.join(self.failed_ids)}")


@dataclass(frozen=True)
class ShardSpec:
    """一片最小独立执行单元。

    Attributes:
        shard_id: 片唯一 ID（同任务内稳定，断点续跑按它记账）。
        dimension: 分片维度（symbol / date）。
        keys: 片内键集合（symbol 列表，或 "start..end" 日期范围串）。
        meta: 自由元数据（如 WFO path_idx、窗口参数等），不参与幂等判定。
    """

    shard_id: str
    dimension: str
    keys: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardRunReport:
    """一次 run_shards 的执行报告。"""

    total: int = 0
    done: int = 0
    skipped: int = 0
    failed: int = 0
    retried: int = 0
    failed_ids: list[str] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    results: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def partition_symbols(
    symbols: Sequence[str],
    batch_size: int,
    dimension: str = SHARD_DIM_SYMBOL,
) -> list[ShardSpec]:
    """按 symbol 分批切片（首选维度）。批次内有序、批次间无交集，覆盖全部输入。"""
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
    if dimension not in VALID_DIMENSIONS:
        raise ValueError(f"非法分片维度: {dimension}")
    specs: list[ShardSpec] = []
    ordered = sorted(set(symbols))
    for i in range(0, len(ordered), batch_size):
        keys = tuple(ordered[i : i + batch_size])
        specs.append(
            ShardSpec(
                shard_id=f"{dimension}_{i // batch_size:04d}",
                dimension=dimension,
                keys=keys,
            )
        )
    return specs


def partition_date_ranges(
    ranges: Sequence[tuple[str, str]],
    shard_size: int = 1,
    dimension: str = SHARD_DIM_DATE,
) -> list[ShardSpec]:
    """按日期范围切片（次选维度）。ranges 为 (start_date, end_date) 序列，
    每片含 shard_size 个连续窗口（如 WFO 的 (path, window) 对，1 = 每窗一片）。
    """
    if shard_size < 1:
        raise ValueError(f"shard_size 必须 >= 1，收到 {shard_size}")
    if dimension not in VALID_DIMENSIONS:
        raise ValueError(f"非法分片维度: {dimension}")
    specs: list[ShardSpec] = []
    for i in range(0, len(ranges), shard_size):
        group = list(ranges[i : i + shard_size])
        keys = tuple(f"{s}..{e}" for s, e in group)
        specs.append(
            ShardSpec(
                shard_id=f"{dimension}_{i // shard_size:04d}",
                dimension=dimension,
                keys=keys,
                meta={"ranges": group},
            )
        )
    return specs


def shard_specs(
    keys: Sequence[str],
    *,
    mode: str = "hybrid",
    batch_size: int = 50,
    dimension: str = SHARD_DIM_SYMBOL,
) -> list[ShardSpec]:
    """按模式生成片：off → 全部键合并为单片（单任务）；否则按 batch_size 分批。"""
    if mode == "off":
        return [
            ShardSpec(
                shard_id="single",
                dimension=dimension,
                keys=tuple(sorted(set(keys))),
            )
        ]
    return partition_symbols(keys, batch_size=batch_size, dimension=dimension)


class ShardCheckpoint:
    """任务级分片状态（JSON 文件，线程安全）。

    文件 schema: {"task_id", "dimension", "fingerprint",
                  "shards": {shard_id: {"state", "attempts", "error", "started_at",
                                        "finished_at"}}}
    fingerprint 与当前输入不一致时整批重置为 PENDING（输入变化 → 全量重跑）。
    checkpoint_dir 不可写时降级为纯内存状态（缓存失败不影响任务执行）。
    """

    def __init__(
        self,
        task_id: str,
        checkpoint_dir: str | None,
        fingerprint: str,
        dimension: str = "",
    ) -> None:
        self.task_id = task_id
        self.fingerprint = fingerprint
        self.dimension = dimension
        self._lock = threading.RLock()
        self._persist = False
        self._data: dict[str, Any] = {
            "task_id": task_id,
            "dimension": dimension,
            "fingerprint": fingerprint,
            "shards": {},
        }
        if checkpoint_dir:
            try:
                self._dir = Path(checkpoint_dir)
                self._dir.mkdir(parents=True, exist_ok=True)
                self._path = self._dir / f"{task_id}.json"
                self._persist = True
                self._load()
            except OSError as e:
                logger.warning(f"[shard] checkpoint 目录不可用，降级为纯内存状态: {e}")
                self._persist = False

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if data.get("fingerprint") != self.fingerprint:
                    logger.info(
                        f"[shard] {self.task_id} fingerprint 变化（输入数据已变），全部片重置为 PENDING"
                    )
                    data = {
                        "task_id": self.task_id,
                        "dimension": self.dimension,
                        "fingerprint": self.fingerprint,
                        "shards": {},
                    }
                self._data = data
                self._data.setdefault("shards", {})
        except Exception as e:
            logger.warning(f"[shard] {self.task_id} checkpoint 损坏，重置: {e}")
            self._data = {
                "task_id": self.task_id,
                "dimension": self.dimension,
                "fingerprint": self.fingerprint,
                "shards": {},
            }
        self.save()

    def state_of(self, shard_id: str) -> str:
        with self._lock:
            return self._data["shards"].get(shard_id, {}).get("state", ShardState.PENDING)

    def attempts_of(self, shard_id: str) -> int:
        with self._lock:
            return int(self._data["shards"].get(shard_id, {}).get("attempts", 0))

    def error_of(self, shard_id: str) -> str:
        with self._lock:
            return str(self._data["shards"].get(shard_id, {}).get("error", "") or "")

    def mark_running(self, shard_id: str) -> None:
        with self._lock:
            rec = self._data["shards"].setdefault(shard_id, {})
            rec["state"] = ShardState.RUNNING
            rec["attempts"] = int(rec.get("attempts", 0)) + 1
            rec["started_at"] = time.time()
            rec["finished_at"] = None
            rec["error"] = None
            self.save()

    def mark_done(self, shard_id: str) -> None:
        with self._lock:
            rec = self._data["shards"].setdefault(shard_id, {})
            rec["state"] = ShardState.DONE
            rec["error"] = None
            rec["finished_at"] = time.time()
            self.save()

    def mark_failed(self, shard_id: str, error: str = "") -> None:
        with self._lock:
            rec = self._data["shards"].setdefault(shard_id, {})
            rec["state"] = ShardState.FAILED
            rec["error"] = str(error)[:2000]
            rec["finished_at"] = time.time()
            self.save()

    def mark_pending(self, shard_id: str) -> None:
        with self._lock:
            rec = self._data["shards"].setdefault(shard_id, {})
            rec["state"] = ShardState.PENDING
            rec["error"] = None
            self.save()

    def decrement_attempts(self, shard_id: str) -> None:
        """尝试次数回退一次（仅用于 RUNNING 中断恢复：中断不算失败）。"""
        with self._lock:
            rec = self._data["shards"].setdefault(shard_id, {})
            rec["attempts"] = max(0, int(rec.get("attempts", 1)) - 1)
            self.save()

    def save(self) -> None:
        if not self._persist:
            return
        try:
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            os.replace(tmp, self._path)
        except OSError as e:
            logger.debug(f"[shard] checkpoint 保存失败（降级内存）: {e}")
            self._persist = False


class ShardScheduler:
    """并发调度器：DONE 片跳过（断点续跑），FAILED 片在 max_attempts 内重跑。"""

    def __init__(
        self,
        checkpoint: ShardCheckpoint,
        max_workers: int = 0,
        max_attempts: int = 2,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_workers = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 4)
        self.max_attempts = max(1, int(max_attempts))
        # 片最后一次异常对象（重试用尽后供调用方原样重抛，保留快照/调用方处理语义）
        self._last_errors: dict[str, Exception] = {}

    def last_error(self, shard_id: str) -> Exception | None:
        return self._last_errors.get(shard_id)

    def run(
        self,
        shards: Sequence[ShardSpec],
        worker_fn: Callable[[ShardSpec], Any],
    ) -> ShardRunReport:
        ckpt = self.checkpoint
        report = ShardRunReport(total=len(shards))

        eligible: list[ShardSpec] = []
        for s in shards:
            st = ckpt.state_of(s.shard_id)
            if st == ShardState.DONE:
                report.skipped += 1
                report.skipped_ids.append(s.shard_id)
                continue
            if st == ShardState.RUNNING:
                # 上次运行中断（崩溃/Kill）：重置为 PENDING 续跑，尝试次数回退
                # （中断不算失败，不消耗重试额度）
                logger.warning(f"[shard] {s.shard_id} 上次中断，重置为 PENDING 续跑")
                ckpt.mark_pending(s.shard_id)
                ckpt.decrement_attempts(s.shard_id)
            elif st == ShardState.FAILED:
                # 上次运行失败：本次调用重新尝试（每次调用 = 新的尝试预算），
                # checkpoint 保留 attempts/error 历史供审计
                logger.info(f"[shard] {s.shard_id} 上次失败，本次调用重新尝试")
                ckpt.mark_pending(s.shard_id)
            eligible.append(s)

        remaining = eligible
        for attempt in range(1, self.max_attempts + 1):
            if not remaining:
                break
            if attempt > 1:
                report.retried += len(remaining)
                logger.info(
                    f"[shard] {ckpt.task_id} 第 {attempt}/{self.max_attempts} 次尝试重跑 {len(remaining)} 片"
                )
            try:
                self._run_batch(remaining, worker_fn, report)
            except KeyboardInterrupt:
                for s in remaining:
                    if ckpt.state_of(s.shard_id) == ShardState.RUNNING:
                        ckpt.mark_pending(s.shard_id)
                raise
            remaining = [s for s in remaining if ckpt.state_of(s.shard_id) == ShardState.FAILED]

        for s in shards:
            if ckpt.state_of(s.shard_id) == ShardState.FAILED:
                report.failed += 1
                report.failed_ids.append(s.shard_id)
        report.done = len(shards) - report.skipped - report.failed
        return report

    def _run_batch(
        self,
        batch: list[ShardSpec],
        worker_fn: Callable[[ShardSpec], Any],
        report: ShardRunReport,
    ) -> None:
        ckpt = self.checkpoint
        workers = min(self.max_workers, len(batch))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {}
            for spec in batch:
                ckpt.mark_running(spec.shard_id)
                fut_map[pool.submit(worker_fn, spec)] = spec
            for fut in as_completed(fut_map):
                spec = fut_map[fut]
                try:
                    res = fut.result()
                    ckpt.mark_done(spec.shard_id)
                    report.results.append((spec.shard_id, res))
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    logger.error(f"[shard] {spec.shard_id} 执行失败: {err}")
                    self._last_errors[spec.shard_id] = e
                    ckpt.mark_failed(spec.shard_id, error=err)


def run_serial(
    specs: Sequence[ShardSpec],
    worker_fn: Callable[[ShardSpec], Any],
) -> ShardRunReport:
    """off 模式：单任务顺序执行全部片，不写 checkpoint（行为与分片前一致）。

    异常原样向上传播（不吞），与分片前的串行调用语义完全一致。
    """
    report = ShardRunReport(total=len(specs))
    for spec in specs:
        report.results.append((spec.shard_id, worker_fn(spec)))
        report.done += 1
    return report


def run_shards(
    specs: Sequence[ShardSpec],
    worker_fn: Callable[[ShardSpec], Any],
    *,
    task_id: str,
    fingerprint: str,
    mode: str = "hybrid",
    checkpoint_dir: str | None = None,
    max_workers: int = 0,
    max_attempts: int = 2,
    dimension: str = "",
    raise_on_failed: bool = True,
) -> ShardRunReport:
    """统一入口：mode=off → 单任务串行；否则 checkpoint + 并发 + 失败片重跑。

    片内 worker 抛出的异常记为该片 FAILED（不影响其他片）；重试用尽后仍有失败片时：
    - raise_on_failed=True（默认）：原样重抛首个失败片的原始异常（保留调用方
      快照/异常处理契约），无原始异常（如跨进程续跑）时抛 ShardExecutionError；
    - raise_on_failed=False：返回报告，调用方自行处理（如信号阶段仅记录）。
    """
    mode = (mode or "hybrid").strip().lower()
    if mode not in VALID_MODES:
        logger.warning(f"[shard] 非法分片模式 {mode!r}，回退 off（单任务）")
        mode = "off"
    if mode == "off":
        logger.info(f"[shard] {task_id} off 模式：单任务顺序执行 {len(specs)} 片")
        return run_serial(specs, worker_fn)

    ckpt = ShardCheckpoint(task_id, checkpoint_dir, fingerprint, dimension=dimension)
    scheduler = ShardScheduler(ckpt, max_workers=max_workers, max_attempts=max_attempts)
    report = scheduler.run(specs, worker_fn)
    _fails = f" failed={','.join(report.failed_ids)}" if report.failed_ids else ""
    logger.info(
        f"[shard] {task_id}: total={report.total} done={report.done} "
        f"skipped={report.skipped} failed={report.failed} retried={report.retried}{_fails}"
    )
    if report.failed and raise_on_failed:
        _first = report.failed_ids[0]
        _err = scheduler.last_error(_first)
        if _err is not None:
            raise _err  # 原样重抛：调用方快照/异常处理契约与分片前一致
        raise ShardExecutionError(report.failed_ids)
    return report


def shard_settings() -> dict[str, Any]:
    """从 Config 读取分片设置；读取失败时返回内置默认值（不阻塞任务）。"""
    settings: dict[str, Any] = {
        "mode": "hybrid",
        "batch_size": 50,
        "max_attempts": 2,
        "max_workers": 0,
        "checkpoint_dir": None,
    }
    try:
        from UtilsManager.ConfigParser import Config

        cfg = Config()
        settings.update(
            {
                "mode": cfg.SHARD_MODE,
                "batch_size": cfg.SHARD_SYMBOL_BATCH_SIZE,
                "max_attempts": cfg.SHARD_MAX_ATTEMPTS,
                "max_workers": cfg.SHARD_MAX_WORKERS,
                "checkpoint_dir": os.path.join(cfg.CACHE_DIRECTORY, "shard_cache"),
            }
        )
    except Exception as e:
        logger.debug(f"[shard] 读取 Config 失败，使用默认设置: {e}")
    return settings
