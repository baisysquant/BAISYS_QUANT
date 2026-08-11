"""Task E 幂等输出与中间结果持久化测试。

覆盖：原子写（无半成品、同 key 覆写）、表级清单 upsert（重复运行不产生重复记录）、
合并阶段唯一性校验（重复键/重复行/缺失工件 → 合并成功率）、
replace 回退（禁用 upsert 直接替换写）、Phase 0 与信号缓存集成。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from BackTrading import output_store as _os


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["a", "b"],
        "trade_date": ["2024-01-01", "2024-01-02"],
        "score": [80, 90],
    })


class TestAtomicWrite:
    @pytest.mark.unit
    def test_atomic_write_parquet_roundtrip(self, tmp_path):
        p = tmp_path / "bucket" / "a.parquet"
        _os.atomic_write_parquet(p, pd.DataFrame({"x": [1, 2]}))
        df = pd.read_parquet(p)
        assert list(df["x"]) == [1, 2]
        assert not list(tmp_path.rglob("*.tmp*"))  # 无半成品残留

    @pytest.mark.unit
    def test_atomic_write_parquet_overwrites_same_key(self, tmp_path):
        p = tmp_path / "a.parquet"
        _os.atomic_write_parquet(p, pd.DataFrame({"x": [1]}))
        _os.atomic_write_parquet(p, pd.DataFrame({"x": [2, 3]}))
        df = pd.read_parquet(p)
        assert list(df["x"]) == [2, 3]  # 同 key 覆写，不累加
        assert len(list(tmp_path.rglob("*.tmp*"))) == 0

    @pytest.mark.unit
    def test_atomic_write_npy_npz_text(self, tmp_path):
        p1 = tmp_path / "a.npy"
        _os.atomic_write_npy(p1, np.array([1, 2, 3]))
        assert list(np.load(p1)) == [1, 2, 3]
        p2 = tmp_path / "d.npz"
        _os.atomic_write_npz(p2, x=np.array([5]), y=np.array([6]))
        with np.load(p2) as z:
            assert list(z["x"]) == [5] and list(z["y"]) == [6]
        p3 = tmp_path / "m.json"
        _os.atomic_write_text(p3, '{"a": 1}')
        assert json.loads(p3.read_text(encoding="utf-8")) == {"a": 1}
        assert len(list(tmp_path.rglob("*.tmp*"))) == 0

    @pytest.mark.unit
    def test_atomic_write_failure_leaves_no_tmp(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        with pytest.raises(OSError):
            _os.atomic_write_text(blocker / "sub" / "f.txt", "data")
        assert len(list(tmp_path.rglob("*.tmp*"))) == 0
        assert blocker.read_text() == "x"  # 原目标未被破坏


class TestOutputManifest:
    @pytest.mark.unit
    def test_upsert_same_key_no_duplicate_records(self, tmp_path):
        schema = _os.OutputSchema("sig", ("symbol", "trade_date"), "v1")
        m = _os.OutputManifest(tmp_path, schema, batch_id="b1")
        m.upsert(_os.OutputRecord(shard_id="p0", key="a", path="a.parquet", rows=2))
        m.upsert(_os.OutputRecord(shard_id="p0", key="a", path="a.parquet", rows=3))  # 重复运行同一片
        m.upsert(_os.OutputRecord(shard_id="p0", key="b", path="b.parquet", rows=1))
        recs = m.records()
        assert len(recs) == 2  # 同 (shard_id, key) 覆写，不重复
        assert m.validate_unique() == []
        by_key = {r.key: r for r in recs}
        assert by_key["a"].rows == 3  # 最新一次写入生效

    @pytest.mark.unit
    def test_manifest_persists_across_instances(self, tmp_path):
        schema = _os.OutputSchema("ind", ("symbol",), "v1")
        m1 = _os.OutputManifest(tmp_path, schema, batch_id="b1")
        m1.upsert(_os.OutputRecord(shard_id="symbol_0000", key="s1", path="p1", rows=5))
        m2 = _os.OutputManifest(tmp_path, schema, batch_id="b1")
        assert [r.key for r in m2.records()] == ["s1"]
        assert m2.keys() == {"s1"}

    @pytest.mark.unit
    def test_manifest_batch_isolation(self, tmp_path):
        schema = _os.OutputSchema("ind", ("symbol",), "v1")
        _os.OutputManifest(tmp_path, schema, batch_id="fpA").upsert(
            _os.OutputRecord(shard_id="s", key="a", rows=1))
        _os.OutputManifest(tmp_path, schema, batch_id="fpB").upsert(
            _os.OutputRecord(shard_id="s", key="b", rows=1))
        files = sorted(p.name for p in (tmp_path / "manifest").glob("*.json"))
        assert len(files) == 2  # 不同数据批次（fingerprint）独立清单，不互相污染
        assert _os.OutputManifest(tmp_path, schema, batch_id="fpA").keys() == {"a"}

    @pytest.mark.unit
    def test_corrupt_manifest_resets(self, tmp_path):
        schema = _os.OutputSchema("ind", ("symbol",), "v1")
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "ind.v1.b1.json").write_text("{corrupt", encoding="utf-8")
        m = _os.OutputManifest(tmp_path, schema, batch_id="b1")
        assert m.records() == []


class TestMerge:
    @pytest.mark.unit
    def test_merge_dedupes_primary_key_rows(self, tmp_path):
        p1 = tmp_path / "a.parquet"
        pd.DataFrame({
            "symbol": ["a", "a", "b"],
            "trade_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "v": [1, 2, 3],
        }).to_parquet(p1, index=False)
        recs = [_os.OutputRecord(shard_id="p0", key="a", path=str(p1), rows=3)]
        merged, rep = _os.merge_outputs(recs, pd.read_parquet, ("symbol", "trade_date"))
        assert rep.read_ok == 1 and rep.success_rate == 100.0
        assert rep.dup_rows == 1  # (a, 2024-01-01) 重复 1 行
        assert len(merged) == 2  # keep=first 去重
        assert merged.iloc[0]["v"] == 1

    @pytest.mark.unit
    def test_merge_missing_artifact_reduces_success_rate(self, tmp_path):
        recs = [
            _os.OutputRecord(shard_id="p0", key="ok", path=str(tmp_path / "ok.parquet"), rows=2),
            _os.OutputRecord(shard_id="p0", key="gone", path=str(tmp_path / "gone.parquet"), rows=2),
        ]
        pd.DataFrame({"symbol": ["a"], "trade_date": ["2024-01-01"]}).to_parquet(
            tmp_path / "ok.parquet", index=False)
        _, rep = _os.merge_outputs(recs, pd.read_parquet, ("symbol", "trade_date"))
        assert rep.missing == ["gone"]
        assert rep.success_rate == 50.0
        assert not rep.ok

    @pytest.mark.unit
    def test_merge_duplicate_keys_detected(self, tmp_path):
        recs = [
            _os.OutputRecord(shard_id="p0", key="a", rows=0),
            _os.OutputRecord(shard_id="p1", key="a", rows=0),
        ]
        _, rep = _os.merge_outputs(recs, pd.read_parquet, ("symbol", "trade_date"))
        assert rep.dup_keys == ["a"]
        assert not rep.ok

    @pytest.mark.unit
    def test_empty_output_exempt_from_missing(self, tmp_path):
        recs = [_os.OutputRecord(shard_id="p0", key="short", path="", rows=0)]
        _, rep = _os.merge_outputs(recs, pd.read_parquet, ("symbol", "trade_date"))
        assert rep.read_ok == 1 and rep.success_rate == 100.0 and rep.ok

    @pytest.mark.unit
    def test_validate_artifacts_light(self, tmp_path):
        (tmp_path / "x.parquet").write_bytes(b"data")
        recs = [
            _os.OutputRecord(shard_id="s0", key="x", path=str(tmp_path / "x.parquet"), rows=1),
            _os.OutputRecord(shard_id="s0", key="gone", path=str(tmp_path / "gone.parquet"), rows=1),
            _os.OutputRecord(shard_id="s0", key="empty", path="", rows=0),
        ]
        rep = _os.validate_artifacts(recs)
        assert rep.read_ok == 2
        assert rep.missing == ["gone"]
        assert rep.success_rate == round(100 * 2 / 3, 2)


class TestWriteMode:
    @pytest.mark.unit
    def test_default_mode_is_upsert(self):
        assert _os.write_mode() == "upsert"

    @pytest.mark.unit
    def test_save_stock_signal_replace_mode_no_manifest(self, tmp_path, monkeypatch):
        import BackTrading.prepare as prepare_mod
        monkeypatch.setattr(_os, "write_mode", lambda: "replace")
        cache_dir = tmp_path / "sig"
        prepare_mod._save_stock_signal(cache_dir, "a", [{"symbol": "a", "v": 1}])
        p = prepare_mod._symbol_cache_path(cache_dir, "a")
        assert p.exists()
        assert not (cache_dir / "manifest").exists()  # 禁用 upsert：无清单
        df = pd.read_parquet(p)
        assert df.iloc[0]["v"] == 1

    @pytest.mark.unit
    def test_save_stock_signal_upsert_overwrites_same_key(self, tmp_path, monkeypatch):
        import BackTrading.prepare as prepare_mod
        monkeypatch.setattr(_os, "write_mode", lambda: "upsert")
        cache_dir = tmp_path / "sig"
        prepare_mod._save_stock_signal(cache_dir, "a", [{"symbol": "a", "v": 1}])
        prepare_mod._save_stock_signal(cache_dir, "a", [{"symbol": "a", "v": 2}])  # 重复运行同一片
        p = prepare_mod._symbol_cache_path(cache_dir, "a")
        df = pd.read_parquet(p)
        assert len(df) == 1 and df.iloc[0]["v"] == 2  # 覆写而非累加
        assert len(list(cache_dir.rglob("*.tmp*"))) == 0


class TestSignalMergeValidation:
    @pytest.mark.unit
    def test_load_signal_cache_dedupes_duplicate_rows(self, tmp_path, monkeypatch):
        import BackTrading.prepare as prepare_mod
        monkeypatch.setattr(prepare_mod, "_cache_dir_for", lambda *a, **k: tmp_path)
        cache_dir = tmp_path
        # 同一 symbol 文件内含主键重复行（模拟历史遗留/异常写入）
        prepare_mod._save_stock_signal(cache_dir, "a", [
            {"symbol": "a", "trade_date": "2024-01-01", "进场评分": 80},
            {"symbol": "a", "trade_date": "2024-01-01", "进场评分": 90},
            {"symbol": "a", "trade_date": "2024-01-02", "进场评分": 70},
        ])
        prepare_mod._save_stock_signal(cache_dir, "b", [
            {"symbol": "b", "trade_date": "2024-01-01", "进场评分": 60},
        ])
        # 清单（upsert 模式）→ 合并覆盖率校验
        _os.OutputManifest(cache_dir, _os.OutputSchema("signal_cache", ("symbol", "trade_date"), "v1")).upsert_many([
            _os.OutputRecord(shard_id="p0", key="a", path=str(prepare_mod._symbol_cache_path(cache_dir, "a")), rows=3),
            _os.OutputRecord(shard_id="p0", key="b", path=str(prepare_mod._symbol_cache_path(cache_dir, "b")), rows=1),
        ])
        df = prepare_mod._load_signal_cache("20991231")
        assert df is not None
        assert not df.duplicated(subset=["symbol", "trade_date"]).any()  # 主键唯一
        assert len(df) == 3  # 2 (a) + 1 (b)，重复行已去重
        assert df[(df["symbol"] == "a") & (df["trade_date"] == "2024-01-01")]["进场评分"].iloc[0] == 80

    @pytest.mark.unit
    def test_load_signal_cache_partial_manifest_warns(self, tmp_path, monkeypatch):
        import BackTrading.prepare as prepare_mod
        monkeypatch.setattr(prepare_mod, "_cache_dir_for", lambda *a, **k: tmp_path)
        prepare_mod._save_stock_signal(tmp_path, "a", [{"symbol": "a", "trade_date": "2024-01-01"}])
        _os.OutputManifest(tmp_path, _os.OutputSchema("signal_cache", ("symbol", "trade_date"), "v1")).upsert_many([
            _os.OutputRecord(shard_id="p0", key="a", rows=1),
            _os.OutputRecord(shard_id="p0", key="missing_b", rows=5),  # 工件不存在
        ])
        df = prepare_mod._load_signal_cache("20991231")
        assert "a" in df["symbol"].values  # 已读工件正常合并


class TestPhase0Manifest:
    @pytest.mark.unit
    def test_phase0_manifest_upsert_and_merge_100(self, tmp_path, monkeypatch):
        import numpy as np
        import BackTrading.indicator_cache as ic
        from BackTrading.indicator_cache import precompute_all_indicators

        monkeypatch.setattr(ic, "_cache_root", lambda: tmp_path / "ind_cache")
        ic._reset_memory_caches()
        stock_dir = tmp_path / "stocks"
        stock_dir.mkdir()
        dates = pd.date_range("2024-01-01", periods=120, freq="B").strftime("%Y-%m-%d")
        rng = np.random.default_rng(7)
        for si, sym in enumerate(["000001.SZ", "000002.SZ", "600000.SH"]):
            close = 10 + np.cumsum(rng.normal(0, 0.1, 120))
            pd.DataFrame({
                "symbol": sym, "trade_date": dates,
                "open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
                "close": close, "volume": 1_000_000, "amount": 1_000_000 * close,
            }).to_parquet(stock_dir / f"{sym}.parquet", index=False)

        kw = dict(shard_mode="symbol", batch_size=2, max_workers=2,
                  checkpoint_dir=str(tmp_path / "ckpt"))
        precompute_all_indicators(str(stock_dir), fingerprint="fpA", **kw)

        schema = _os.OutputSchema("indicator_cache", ("symbol",), "v1")
        import hashlib
        batch = "phase0_" + hashlib.sha1(b"fpA").hexdigest()[:12]
        m_a = _os.OutputManifest(ic._cache_root(), schema, batch_id=batch)
        recs_a = m_a.records()
        assert len(recs_a) == 3
        assert len({r.key for r in recs_a}) == 3  # 无重复键
        rep = _os.validate_artifacts(recs_a)
        assert rep.success_rate == 100.0 and rep.ok  # 合并成功率 100%

        # 第二次运行（同 fingerprint → checkpoint 全部 DONE 跳过）→ 清单不重复累加
        ic._reset_memory_caches()
        precompute_all_indicators(str(stock_dir), fingerprint="fpA", **kw)
        recs_a2 = _os.OutputManifest(ic._cache_root(), schema, batch_id=batch).records()
        assert len(recs_a2) == 3  # upsert：不产生重复记录

        # 新数据批次（不同 fingerprint）→ 独立清单，不互相污染
        ic._reset_memory_caches()
        precompute_all_indicators(str(stock_dir), fingerprint="fpB", **kw)
        batch_b = "phase0_" + hashlib.sha1(b"fpB").hexdigest()[:12]
        recs_b = _os.OutputManifest(ic._cache_root(), schema, batch_id=batch_b).records()
        assert len(recs_b) == 3
        assert _os.OutputManifest(ic._cache_root(), schema, batch_id=batch).keys() == {r.key for r in recs_a2}

        ic._reset_memory_caches()

    @pytest.mark.unit
    def test_phase0_replace_mode_no_manifest(self, tmp_path, monkeypatch):
        import numpy as np
        import BackTrading.indicator_cache as ic
        from BackTrading.indicator_cache import precompute_all_indicators

        monkeypatch.setattr(ic, "_cache_root", lambda: tmp_path / "ind_cache")
        monkeypatch.setattr(_os, "write_mode", lambda: "replace")  # 禁用 upsert → 回退替换写
        ic._reset_memory_caches()
        stock_dir = tmp_path / "stocks"
        stock_dir.mkdir()
        close = np.linspace(10, 13, 120)
        pd.DataFrame({
            "symbol": "a", "trade_date": pd.date_range("2024-01-01", periods=120, freq="B").strftime("%Y-%m-%d"),
            "open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
            "close": close, "volume": 1_000_000, "amount": 1_000_000 * close,
        }).to_parquet(stock_dir / "a.parquet", index=False)

        precompute_all_indicators(str(stock_dir), fingerprint="fpR",
                                  shard_mode="symbol", batch_size=1, max_workers=1,
                                  checkpoint_dir=str(tmp_path / "ckpt"))
        assert not (ic._cache_root() / "manifest").exists()  # 替换写：无清单
        assert ic._indicators_path("a").exists()  # 工件直接落盘
        assert len(list((ic._cache_root()).rglob("*.tmp*"))) == 0
        ic._reset_memory_caches()
