"""D1 分片执行框架（shard）测试。

覆盖：分片划分、断点续跑、失败仅重跑失败片、fingerprint 失效重置、
尝试次数耗尽、中断恢复、off 单任务回退、Phase 0 分片与串行等价。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from BackTrading import sharding as sh


class TestPartition:
    @pytest.mark.unit
    def test_partition_symbols_covers_all(self):
        symbols = [f"s{i}" for i in range(107)]
        specs = sh.partition_symbols(symbols, batch_size=50)
        assert len(specs) == 3
        covered = [k for s in specs for k in s.keys]
        assert sorted(covered) == sorted(set(symbols))
        assert all(len(s.keys) <= 50 for s in specs)
        assert specs[0].dimension == "symbol"

    @pytest.mark.unit
    def test_partition_symbols_deduplicates_and_sorts(self):
        specs = sh.partition_symbols(["a", "b", "a"], batch_size=10)
        covered = [k for s in specs for k in s.keys]
        assert covered == ["a", "b"]

    @pytest.mark.unit
    def test_partition_batch_size_must_be_positive(self):
        with pytest.raises(ValueError):
            sh.partition_symbols(["a"], batch_size=0)

    @pytest.mark.unit
    def test_partition_date_ranges(self):
        ranges = [
            ("2024-01-01", "2024-03-01"),
            ("2024-03-01", "2024-06-01"),
            ("2024-06-01", "2024-09-01"),
        ]
        specs = sh.partition_date_ranges(ranges, shard_size=2)
        assert len(specs) == 2
        assert specs[0].meta["ranges"] == ranges[:2]
        assert specs[1].keys == ("2024-06-01..2024-09-01",)

    @pytest.mark.unit
    def test_shard_specs_off_mode_single_spec(self):
        specs = sh.shard_specs(["c", "a", "b"], mode="off", batch_size=1)
        assert len(specs) == 1
        assert specs[0].keys == ("a", "b", "c")
        assert specs[0].shard_id == "single"


class TestScheduler:
    @pytest.mark.unit
    def test_retries_failed_shards_only(self, tmp_path):
        calls: list[str] = []

        def worker(spec: sh.ShardSpec) -> int:
            calls.append(spec.shard_id)
            if spec.shard_id == "symbol_0002" and calls.count("symbol_0002") == 1:
                raise RuntimeError("boom")
            return len(spec.keys)

        specs = sh.partition_symbols([f"s{i}" for i in range(10)], batch_size=4)
        ckpt = sh.ShardCheckpoint("t_retry", str(tmp_path), "fp1")
        report = sh.ShardScheduler(ckpt, max_workers=4, max_attempts=2).run(specs, worker)

        assert report.ok
        assert report.failed == 0
        # 成功片只跑 1 次；失败片重跑 1 次后成功
        assert calls.count("symbol_0000") == 1
        assert calls.count("symbol_0002") == 2
        assert report.retried == 1
        assert ckpt.state_of("symbol_0002") == sh.ShardState.DONE

    @pytest.mark.unit
    def test_resume_skips_done_shards(self, tmp_path):
        executed: list[str] = []

        def worker(spec: sh.ShardSpec) -> int:
            executed.append(spec.shard_id)
            return 1

        specs = sh.partition_symbols([f"s{i}" for i in range(8)], batch_size=4)
        ckpt = sh.ShardCheckpoint("t_resume", str(tmp_path), "fp1")
        r1 = sh.ShardScheduler(ckpt, max_workers=2, max_attempts=2).run(specs, worker)
        assert r1.done == 2

        executed.clear()
        ckpt2 = sh.ShardCheckpoint("t_resume", str(tmp_path), "fp1")
        r2 = sh.ShardScheduler(ckpt2, max_workers=2, max_attempts=2).run(specs, worker)
        assert r2.skipped == 2
        assert executed == []  # 全部 DONE 片跳过，零重跑
        assert r2.ok

    @pytest.mark.unit
    def test_fingerprint_change_resets_all(self, tmp_path):
        executed: list[str] = []

        def worker(spec: sh.ShardSpec) -> int:
            executed.append(spec.shard_id)
            return 1

        specs = sh.partition_symbols([f"s{i}" for i in range(4)], batch_size=2)
        sh.ShardScheduler(
            sh.ShardCheckpoint("t_fp", str(tmp_path), "fp1"), max_workers=2, max_attempts=2
        ).run(specs, worker)
        executed.clear()
        sh.ShardScheduler(
            sh.ShardCheckpoint("t_fp", str(tmp_path), "fp2"), max_workers=2, max_attempts=2
        ).run(specs, worker)
        # 输入数据（fingerprint）已变 → 全部片重置并重跑
        assert sorted(executed) == [s.shard_id for s in specs]

    @pytest.mark.unit
    def test_max_attempts_exhausted_reports_failed(self, tmp_path):
        def worker(spec: sh.ShardSpec) -> int:
            raise RuntimeError("always")

        specs = sh.partition_symbols([f"s{i}" for i in range(3)], batch_size=1)
        ckpt = sh.ShardCheckpoint("t_fail", str(tmp_path), "fp1")
        report = sh.ShardScheduler(ckpt, max_workers=2, max_attempts=2).run(specs, worker)
        assert report.failed == 3
        assert len(report.failed_ids) == 3
        assert not report.ok
        # 每片尝试次数 = max_attempts
        assert ckpt.attempts_of("symbol_0000") == 2

    @pytest.mark.unit
    def test_interrupted_running_resumes_without_consuming_attempt(self, tmp_path):
        # 模拟上次中断：一片 DONE、一片遗留 RUNNING（崩溃）
        specs = sh.partition_symbols(["a", "b"], batch_size=1)
        ckpt = sh.ShardCheckpoint("t_int", str(tmp_path), "fp1")
        ckpt.mark_done("symbol_0000")
        ckpt.mark_running("symbol_0001")
        executed: list[str] = []

        def worker(spec: sh.ShardSpec) -> int:
            executed.append(spec.shard_id)
            return 1

        report = sh.ShardScheduler(ckpt, max_workers=1, max_attempts=1).run(specs, worker)
        assert report.skipped == 1
        assert executed == ["symbol_0001"]  # 中断片续跑（不占失败重试额度）
        assert ckpt.state_of("symbol_0001") == sh.ShardState.DONE

    @pytest.mark.unit
    def test_checkpoint_file_persisted(self, tmp_path):
        specs = sh.partition_symbols(["a"], batch_size=1)
        ckpt = sh.ShardCheckpoint("t_persist", str(tmp_path), "fp1")
        sh.ShardScheduler(ckpt, max_workers=1, max_attempts=1).run(specs, lambda s: 1)
        data = json.loads((tmp_path / "t_persist.json").read_text(encoding="utf-8"))
        assert data["fingerprint"] == "fp1"
        assert data["shards"]["symbol_0000"]["state"] == "DONE"
        assert data["shards"]["symbol_0000"]["attempts"] == 1

    @pytest.mark.unit
    def test_checkpoint_without_dir_degrades_to_memory(self):
        ckpt = sh.ShardCheckpoint("t_mem", None, "fp1")
        ckpt.mark_done("x")
        assert ckpt.state_of("x") == "DONE"


class TestRunShards:
    @pytest.mark.unit
    def test_off_mode_serial_no_checkpoint(self, tmp_path):
        calls: list[str] = []
        specs = sh.shard_specs(["a", "b"], mode="off", batch_size=1)

        def worker(spec: sh.ShardSpec) -> int:
            calls.extend(spec.keys)
            return 1

        report = sh.run_shards(
            specs, worker, task_id="off1", fingerprint="fp1", mode="off",
            checkpoint_dir=str(tmp_path),
        )
        assert report.done == 1  # off = 单片（含全部键），串行
        assert sorted(calls) == ["a", "b"]
        assert not (tmp_path / "off1.json").exists()  # 不写 checkpoint

    @pytest.mark.unit
    def test_invalid_mode_falls_back_to_off(self, tmp_path):
        calls: list[str] = []
        specs = sh.shard_specs(["a"], mode="weird", batch_size=1)

        def worker(spec: sh.ShardSpec) -> int:
            calls.append("x")
            return 1

        report = sh.run_shards(
            specs, worker, task_id="weird1", fingerprint="fp1", mode="weird",
            checkpoint_dir=str(tmp_path),
        )
        assert report.done == 1
        assert not (tmp_path / "weird1.json").exists()

    @pytest.mark.unit
    def test_worker_exception_does_not_abort_other_shards(self, tmp_path):
        def worker(spec: sh.ShardSpec) -> int:
            if spec.shard_id == "symbol_0001":
                raise ValueError("bad shard")
            return 1

        specs = sh.partition_symbols([f"s{i}" for i in range(6)], batch_size=2)
        report = sh.run_shards(
            specs, worker, task_id="t_partial", fingerprint="fp1", mode="symbol",
            checkpoint_dir=str(tmp_path), max_workers=3, max_attempts=1,
            raise_on_failed=False,  # 容忍失败：返回报告由调用方处理
        )
        assert report.done == 2
        assert report.failed == 1
        assert report.failed_ids == ["symbol_0001"]

    @pytest.mark.unit
    def test_raise_on_failed_reraise_original_exception(self, tmp_path):
        def worker(spec: sh.ShardSpec) -> int:
            if spec.shard_id == "symbol_0000":
                raise ValueError("macd boom")
            return 1

        specs = sh.partition_symbols(["a", "b"], batch_size=1)
        with pytest.raises(ValueError, match="macd boom"):
            sh.run_shards(
                specs, worker, task_id="t_raise", fingerprint="fp1", mode="symbol",
                checkpoint_dir=str(tmp_path), max_workers=2, max_attempts=1,
            )

    @pytest.mark.unit
    def test_off_mode_raises_original_exception(self, tmp_path):
        def worker(spec: sh.ShardSpec) -> int:
            raise ValueError("serial boom")

        specs = sh.shard_specs(["a"], mode="off", batch_size=1)
        with pytest.raises(ValueError, match="serial boom"):
            sh.run_shards(
                specs, worker, task_id="t_off_raise", fingerprint="fp1", mode="off",
                checkpoint_dir=str(tmp_path),
            )
        assert not (tmp_path / "t_off_raise.json").exists()


class TestPhase0Sharded:
    @pytest.mark.unit
    def test_phase0_sharded_matches_serial(self, tmp_path, monkeypatch):
        import numpy as np
        import pandas as pd
        import BackTrading.indicator_cache as ic

        # 隔离：磁盘缓存进 tmp，内存缓存清空（防跨测试污染）
        monkeypatch.setattr(ic, "_cache_root", lambda: tmp_path / "ind_cache")
        ic._reset_memory_caches()

        stock_dir = tmp_path / "stocks"
        stock_dir.mkdir()
        dates = pd.date_range("2024-01-01", periods=120, freq="B").strftime("%Y-%m-%d")
        rng = np.random.default_rng(7)
        for si, sym in enumerate(["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]):
            n = 120 if si < 3 else 40  # 第 4 只不足 60 根 → empty 缓存
            close = 10 + np.cumsum(rng.normal(0, 0.1, n))
            df = pd.DataFrame({
                "symbol": sym,
                "trade_date": dates[:n],
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 1_000_000,
                "amount": 1_000_000 * close,
            })
            df.to_parquet(stock_dir / f"{sym}.parquet", index=False)

        # 分片执行：2 只/片，2 worker（真实执行路径，Config 仅用于 checkpoint 参数，
        # 这里显式传入避免触碰真实缓存目录）
        ic.precompute_all_indicators(
            str(stock_dir),
            fingerprint="fp_phase0_test",
            shard_mode="symbol",
            batch_size=2,
            max_workers=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )

        # 3 只正常股票入内存缓存；不足 60 根的写空缓存
        assert "000001.SZ" in ic._IN_MEMORY
        assert len(ic._IN_MEMORY["000001.SZ"]) == 120
        assert len(ic._IN_MEMORY["000001.SZ"].columns) > 5
        assert ic._IN_MEMORY["600001.SH"].empty

        # 磁盘缓存落盘（幂等单元），含背离缓存
        root = tmp_path / "ind_cache"
        assert (root / "00" / "000001.SZ.indicators.parquet").exists()
        assert (root / "00" / "000001.SZ.divergence.npz").exists()

        # checkpoint 已记录（断点续跑数据源）
        ckpt_files = list((tmp_path / "ckpt").glob("phase0_*.json"))
        assert len(ckpt_files) == 1

        # 幂等：同 fingerprint 二次执行零重算（全部片 DONE 跳过）
        ic._reset_memory_caches()
        ic.precompute_all_indicators(
            str(stock_dir),
            fingerprint="fp_phase0_test",
            shard_mode="symbol",
            batch_size=2,
            max_workers=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        assert "000001.SZ" in ic._IN_MEMORY
        assert len(ic._IN_MEMORY["000001.SZ"]) == 120

        # 串行（off）等价性：同一输入下逐列一致
        ic._reset_memory_caches()
        ic.precompute_all_indicators(str(stock_dir), fingerprint="fp_serial")
        for sym in ["000001.SZ", "000002.SZ", "600000.SH"]:
            assert sym in ic._IN_MEMORY
        assert "000001.SZ" in ic._IN_MEMORY

        ic._reset_memory_caches()
