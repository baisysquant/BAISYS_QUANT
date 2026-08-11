"""Task A2 失败快照持久化 — 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def snap_df() -> pd.DataFrame:
    np.random.seed(7)
    n = 1000
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + 0.2
    low = close - 0.2
    df = pd.DataFrame({
        "trade_date": pd.date_range("2023-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "symbol": "sh600000",
        "open": close - 0.05,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    })
    return df


def _save(snap_df, **kw):
    from BackTrading.snapshot import save_failure_snapshot
    return save_failure_snapshot(ohlcv=snap_df, **kw)


def test_roundtrip_full_schema(snap_df, monkeypatch) -> None:
    """保存 → 本地复现：元数据全字段 + OHLCV 可加载。"""
    import BackTrading.snapshot as _snap
    from BackTrading.snapshot import load_snapshot, _snapshot_root
    monkeypatch.setattr(_snap, "_snapshot_max_rows", lambda: 10000)
    sid = _save(
        snap_df,
        symbol="sh600000", market="上交所",
        window_name="1-0",
        window_start="2023-01-02", window_end="2023-06-30", window_size=120,
        metric_name="window_optimize",
        error_code="WINDOW_OPTIMIZE_FAILED",
        error_message="boom", traceback_text="Traceback (most recent call last):\nboom",
        run_id="run_abc", task_id="task_xyz",
        precheck_status={"nan_frac": 0.0},
        adjust_factors={"front_adj": 1.0, "dividend_dates": []},
    )
    assert sid and isinstance(sid, str)
    df, meta = load_snapshot(sid)
    assert len(df) == len(snap_df)
    assert meta.snapshot_id == sid
    assert meta.run_id == "run_abc"
    assert meta.task_id == "task_xyz"
    assert meta.symbol == "sh600000"
    assert meta.market == "上交所"
    assert meta.window_name == "1-0"
    assert meta.window_start == "2023-01-02"
    assert meta.window_end == "2023-06-30"
    assert meta.window_size == 120
    assert meta.metric_name == "window_optimize"
    assert meta.error_code == "WINDOW_OPTIMIZE_FAILED"
    assert meta.error_message == "boom"
    assert "Traceback" in meta.traceback
    assert meta.precheck_status == {"nan_frac": 0.0}
    assert meta.adjust_factors == {"front_adj": 1.0, "dividend_dates": []}
    # 运行时字段
    assert meta.hostname
    assert meta.pid > 0
    assert meta.worker_id
    assert meta.timestamp
    # 样本计数与交易日历
    assert meta.sample_counts["close"]["non_nan"] == 1000
    assert meta.sample_counts["close"]["max_consecutive_non_nan"] == 1000
    assert meta.trade_calendar_slice == sorted({d[:10] for d in snap_df["trade_date"]})
    assert meta.ohlcv_rows == 1000
    # 磁盘文件形态：parquet(行情) + json(元数据)
    assert (Path(meta.storage_dir) / f"{sid}.parquet").exists()
    assert (Path(meta.storage_dir) / f"{sid}.json").exists()
    json.loads((Path(meta.storage_dir) / f"{sid}.json").read_text(encoding="utf-8"))


def test_disabled_returns_none(snap_df, monkeypatch) -> None:
    import BackTrading.snapshot as _snap
    monkeypatch.setattr(_snap, "_snapshot_enabled", lambda: False)
    assert _save(snap_df, metric_name="window_optimize") is None
    assert not _snap._snapshot_root().exists()


def test_truncation_and_summaries(snap_df, monkeypatch) -> None:
    """N=200 截断：只落最近 200 行，sample_counts/trade_calendar 与落盘切片一致。"""
    import BackTrading.snapshot as _snap
    monkeypatch.setattr(_snap, "_snapshot_max_rows", lambda: 200)
    sid = _save(snap_df, symbol="sz000001", metric_name="window_oos", error_code="WINDOW_OOS_FAILED")
    df, meta = _snap.load_snapshot(sid)
    assert len(df) == 200
    assert meta.ohlcv_rows == 200
    # 落盘切片 = 原数据尾部 200 行
    tail = snap_df.iloc[-200:].reset_index(drop=True)
    pd.testing.assert_frame_equal(df, tail)
    assert meta.trade_calendar_slice == sorted({d[:10] for d in tail["trade_date"]})
    assert meta.sample_counts["close"]["non_nan"] == 200
    assert meta.window_size == 1000  # 未传时取输入总行数


def test_market_inference() -> None:
    from BackTrading.snapshot import _market_of
    assert _market_of("sh600000") == "上交所"
    assert _market_of("600000") == "上交所"
    assert _market_of("sz000001") == "深交所"
    assert _market_of("000001") == "深交所"
    assert _market_of("300750") == "深交所"
    assert _market_of("bj830799") == "北交所"
    assert _market_of("qq") == "未知"
    assert _market_of(None) == "未知"


def test_empty_ohlcv_metadata_only() -> None:
    """空输入：仅落元数据，无 parquet，可复现窗口上下文与错误。"""
    from BackTrading.snapshot import load_snapshot
    sid = _save(
        None, window_name="entry", metric_name="window_setup",
        error_code="DATA_INSUFFICIENT", error_message="数据不足",
    )
    df, meta = load_snapshot(sid)
    assert df.empty
    assert meta.error_code == "DATA_INSUFFICIENT"
    assert meta.ohlcv_rows == 0
    assert meta.window_name == "entry"


def test_session_cap(snap_df, monkeypatch) -> None:
    """单会话上限：超过后返回 None 且仅告警。"""
    import BackTrading.snapshot as _snap
    monkeypatch.setattr(_snap, "_SESSION_CAP", 2)
    _snap._session_count = 0
    assert _save(snap_df, metric_name="a") is not None
    assert _save(snap_df, metric_name="b") is not None
    assert _save(snap_df, metric_name="c") is None


def test_cleanup_removes_expired(snap_df) -> None:
    """过期清理：旧日期目录删除，新目录保留。"""
    from BackTrading.snapshot import cleanup_snapshots, _snapshot_root
    root = _snapshot_root()
    old_dir = root / "2020-01-01" / "sh"
    old_dir.mkdir(parents=True)
    (old_dir / "old.json").write_text("{}", encoding="utf-8")
    (old_dir / "old.parquet").write_bytes(b"x")
    recent_dir = root / "2099-01-01" / "sh"
    recent_dir.mkdir(parents=True)
    (recent_dir / "new.json").write_text("{}", encoding="utf-8")

    removed = cleanup_snapshots(older_than_days=7, root=root)
    assert removed == 1
    assert not old_dir.exists()
    assert (recent_dir / "new.json").exists()
    assert cleanup_snapshots(older_than_days=7, root=root) == 0


def test_save_write_failure_does_not_raise(snap_df, monkeypatch) -> None:
    """落盘异常只告警不抛出，主流程不受影响。"""
    import BackTrading.snapshot as _snap
    monkeypatch.setattr(_snap, "_snapshot_root", lambda: Path("Z:/definitely/not/writable"))
    assert _save(snap_df, metric_name="window_optimize") is None


def test_wfo_data_insufficient_snapshot(monkeypatch) -> None:
    """WFO 入口数据不足：抛 ValueError 且快照落盘（含 run/task 上下文）。"""
    from BackTrading.bayesian.meta_optimizer import bayesian_walk_forward_multi
    from BackTrading.snapshot import find_snapshots

    df = pd.DataFrame({
        "trade_date": pd.date_range("2023-01-02", periods=100, freq="B").strftime("%Y-%m-%d"),
        "symbol": "sh600000",
        "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1_000_000,
    })
    with pytest.raises(ValueError, match="数据不足"):
        bayesian_walk_forward_multi(
            kline_df=df, train_period=120, test_period=15,
            num_paths=1, initial_cash=1_000_000,
            run_id="run_wfo_1", task_id="wfo_task",
        )
    snaps = find_snapshots()
    assert len(snaps) == 1
    assert snaps[0].error_code == "DATA_INSUFFICIENT"
    assert snaps[0].metric_name == "window_setup"
    assert snaps[0].run_id == "run_wfo_1"
    assert snaps[0].task_id == "wfo_task"


def test_wfo_empty_slice_snapshot(monkeypatch) -> None:
    """WFO 路径无有效窗口（purge 吃空训练期）：快照落盘 + 兜底帧返回。"""
    from BackTrading.bayesian.meta_optimizer import bayesian_walk_forward_multi
    from BackTrading.snapshot import find_snapshots

    df = pd.DataFrame({
        "trade_date": pd.date_range("2023-01-02", periods=150, freq="B").strftime("%Y-%m-%d"),
        "symbol": "sh600000",
        "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1_000_000,
    })
    result = bayesian_walk_forward_multi(
        kline_df=df, train_period=120, test_period=15,
        num_paths=2, initial_cash=1_000_000, purge_days=200,
    )
    assert not result.empty  # 兜底帧照常返回
    snaps = find_snapshots()
    assert any(s.error_code == "WINDOW_SLICE_EMPTY" for s in snaps)
    assert all(s.symbol is None or s.symbol == "all" for s in snaps)


def test_indicator_cache_failure_snapshot(snap_df, tmp_path, monkeypatch) -> None:
    """指标计算异常：precompute_all_indicators 落盘符号级快照后原样重抛。"""
    import BackTrading.prepare as _prepare
    from BackTrading.indicator_cache import precompute_all_indicators
    from BackTrading.snapshot import find_snapshots

    def _boom(_df):
        raise ValueError("macd boom")

    monkeypatch.setattr(_prepare, "_compute_indicators", _boom)
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    snap_df.to_parquet(stock_dir / "sh600000.parquet", index=False)

    with pytest.raises(ValueError, match="macd boom"):
        precompute_all_indicators(str(stock_dir))

    snaps = find_snapshots()
    assert len(snaps) == 1
    assert snaps[0].symbol == "sh600000"
    assert snaps[0].market == "上交所"
    assert snaps[0].error_code == "INDICATOR_COMPUTE_FAILED"
    assert snaps[0].metric_name == "compute_indicators"
    from BackTrading.snapshot import load_snapshot
    df, meta = load_snapshot(snaps[0].snapshot_id)
    assert meta.ohlcv_rows == 200  # 默认截断 N=200
    assert len(df) == 200
    assert "close" in df.columns
