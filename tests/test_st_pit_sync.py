"""
P0-5 ST/退市状态 PIT（Point-In-Time）同步 测试

背景：stock_st_history 曾为最近快照（旧代码注释自认），历史 ST 期缺失 →
历史 5% 涨跌幅被错按 10%、ST 禁买/强平失效；且 SQL 字符串插值拼接 symbol
（注入风险）。本文件验证：

  - ensure_st_history_table 幂等建表
  - sync_st_pit 每日增量归档（ST 列表 ∩ 池 → True；现货列表 ∩ 池 − ST → False）
  - SZ 简称变更历史 → 历史 ST/退市整理期 PIT（按实际交易日展开）
  - 终止上市日期 → 退市日 PIT
  - 覆盖检查：二次同步（force=False）跳过；force=True 强制回填
  - load_st_pit 参数化 ANY(:syms)：注入串仅当数据处理，不构成 SQL 注入
  - _name_flags / _sz_st_periods 纯逻辑

使用 PostgreSQL 上的临时表（stock_st_history_pit_test），结束即删。
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from DataManager import StPitSync as sp
from DataManager.DbEngine import get_engine
from UtilsManager.ConfigParser import Config


def _db_available() -> bool:
    try:
        with get_engine(Config()).connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="PostgreSQL 不可达，跳过 PIT 表测试"
)


@pytest.fixture()
def engine():
    return get_engine(Config())


@pytest.fixture()
def scratch(engine, monkeypatch):
    name = "stock_st_history_pit_test"
    monkeypatch.setattr(sp, "_ST_TABLE", name)
    with engine.begin() as c:
        c.execute(text(f"DROP TABLE IF EXISTS {name}"))
    yield name
    with engine.begin() as c:
        c.execute(text(f"DROP TABLE IF EXISTS {name}"))


def _no_network(monkeypatch) -> None:
    """全部网络源置空（离线降级路径）。"""
    for f in ("_fetch_st_list", "_fetch_spot_list",
              "_fetch_sz_change_names", "_fetch_sz_delist", "_fetch_sh_delist"):
        monkeypatch.setattr(sp, f, lambda: None)


def _bday_days(engine, symbol, a, b):  # noqa: ANN001
    return list(pd.bdate_range(a, b).date)


# ── 纯逻辑 ──────────────────────────────────────────────


def test_name_flags() -> None:
    assert sp._name_flags("ST平安") == (True, False)
    assert sp._name_flags("*ST金泰") == (True, False)
    assert sp._name_flags("S*ST昌九") == (True, False)
    assert sp._name_flags("平安银行") == (False, False)
    assert sp._name_flags("退市金亚") == (False, True)
    assert sp._name_flags("*ST金泰退") == (True, True)
    assert sp._name_flags(None) == (False, False)
    assert sp._name_flags("") == (False, False)


def test_sz_st_periods() -> None:
    names = pd.DataFrame({
        "证券代码": ["000001", "000001", "000001"],
        "证券简称": ["平安银行", "ST平安", "平安银行"],
        "变更日期": ["2023-06-01", "2024-01-01", "2024-06-01"],
    })
    periods = sp._sz_st_periods(names, {"sz000001"})
    assert periods["sz000001"] == [
        (pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-06-01").date(), True, False)
    ]


def test_sz_st_periods_delist_name_and_pool_filter() -> None:
    names = pd.DataFrame({
        "证券代码": ["000001", "000002", "000003"],
        "证券简称": ["ST平安", "正常股", "退市金亚"],
        "变更日期": ["2024-01-01", "2024-01-01", "2024-02-01"],
    })
    periods = sp._sz_st_periods(names, {"sz000001", "sz000003"})
    assert periods["sz000001"] == [
        (pd.Timestamp("2024-01-01").date(), None, True, False)
    ]
    assert periods["sz000003"] == [
        (pd.Timestamp("2024-02-01").date(), None, False, True)
    ]
    assert "sz000002" not in periods  # 池外股票不处理


# ── 表与同步（真实 PostgreSQL 临时表） ───────────────────


def test_ensure_table_idempotent(engine, scratch) -> None:
    sp.ensure_st_history_table(engine)
    sp.ensure_st_history_table(engine)
    with engine.connect() as c:
        assert c.execute(text(
            "SELECT to_regclass('public.stock_st_history_pit_test')"
        )).scalar() is not None


def test_sync_archive_today(engine, scratch, monkeypatch) -> None:
    _no_network(monkeypatch)
    monkeypatch.setattr(sp, "_pool_covered", lambda *a, **k: False)  # 解锁覆盖守卫，强制走同步路径
    monkeypatch.setattr(
        sp, "_fetch_st_list",
        lambda: pd.DataFrame({"代码": ["000001", "600000"]}),
    )
    monkeypatch.setattr(
        sp, "_fetch_spot_list",
        lambda: pd.DataFrame({"代码": ["000001", "600000", "002726"]}),
    )
    pool = ["sz000001", "sh600000", "sz002726"]
    stats = sp.sync_st_pit(engine, pool, start_date="2024-01-01", end_date="2024-12-31")
    assert stats["archive_today"] == 3
    hist = sp.load_st_pit(engine, pool, "2024-01-01", "2099-12-31")
    today = str(pd.Timestamp.today().date())
    assert hist["sz000001"][today] == (True, False)   # ST 列表 ∩ 池
    assert hist["sh600000"][today] == (True, False)
    assert hist["sz002726"][today] == (False, False)  # 现货 − ST → 显式非 ST


def test_sync_sz_st_periods_expansion(engine, scratch, monkeypatch) -> None:
    names = pd.DataFrame({
        "证券代码": ["000001", "000001", "000001"],
        "证券简称": ["平安银行", "ST平安", "平安银行"],
        "变更日期": ["2023-06-01", "2024-01-01", "2024-06-01"],
    })
    _no_network(monkeypatch)
    monkeypatch.setattr(sp, "_pool_covered", lambda *a, **k: False)  # 解锁覆盖守卫
    monkeypatch.setattr(sp, "_fetch_sz_change_names", lambda: names)
    monkeypatch.setattr(
        sp, "_kline_days_batch",
        lambda engine, specs: {(sym, s, e): _bday_days(engine, sym, s, e)
                               for sym, s, e in specs},
    )
    stats = sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01", end_date="2024-12-31")
    assert stats["sz_st_rows"] > 0
    hist = sp.load_st_pit(engine, ["sz000001"], "2024-01-01", "2024-12-31")
    days = hist["sz000001"]
    assert days["2024-01-01"] == (True, False)   # 变更生效日即 ST
    assert days["2024-03-15"] == (True, False)
    assert days["2024-05-31"] == (True, False)   # 2024-06-01（不含）前仍 ST
    assert "2024-06-03" not in days              # 摘帽后不写入
    assert "2023-06-01" not in days              # 查询窗口外不写入


def test_sync_delist_pit(engine, scratch, monkeypatch) -> None:
    _no_network(monkeypatch)
    monkeypatch.setattr(sp, "_pool_covered", lambda *a, **k: False)  # 解锁覆盖守卫
    monkeypatch.setattr(sp, "_fetch_sz_delist", lambda: pd.DataFrame({
        "证券代码": ["002726"], "终止上市日期": ["2024-03-01"],
    }))
    monkeypatch.setattr(sp, "_fetch_sh_delist", lambda: pd.DataFrame({
        "公司代码": ["600080"], "暂停上市日期": ["2024-04-01"],
    }))
    monkeypatch.setattr(
        sp, "_kline_days_batch",
        lambda engine, specs: {(sym, s, e): _bday_days(engine, sym, s, e)
                               for sym, s, e in specs},
    )
    sp.sync_st_pit(engine, ["sz002726", "sh600080"],
                   start_date="2024-01-01", end_date="2024-12-31")
    hist = sp.load_st_pit(engine, ["sz002726", "sh600080"], "2024-01-01", "2024-12-31")
    # P1-4 修复：退市整理期 = 摘牌日前 N 交易日（2020-12-31 后摘牌 N=15），
    # 2024-03-01（周五）前 15 个交易日 → 2024-02-12 起标记，摘牌日后不再标记
    assert hist["sz002726"]["2024-03-01"] == (False, True)
    assert hist["sz002726"]["2024-02-12"] == (False, True)   # 整理期起点（前 15 交易日）
    assert hist["sz002726"]["2024-02-29"] == (False, True)   # 整理期内（旧实现断言翻转）
    assert "2024-02-09" not in hist["sz002726"]              # 起点前一日非整理期
    assert "2024-03-04" not in hist["sz002726"]              # 摘牌日后不标记（旧实现方向错误）
    assert hist["sh600080"]["2024-04-01"] == (False, True)
    assert "2024-04-02" not in hist["sh600080"]


def test_load_pit_parametrized_any_no_injection(engine, scratch, monkeypatch) -> None:
    _no_network(monkeypatch)
    monkeypatch.setattr(sp, "_pool_covered", lambda *a, **k: False)  # 解锁覆盖守卫
    monkeypatch.setattr(sp, "_fetch_st_list", lambda: pd.DataFrame({"代码": ["000001"]}))
    sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01", end_date="2024-12-31")
    evil = "sz000001'; DROP TABLE stock_st_history_pit_test; --"
    hist = sp.load_st_pit(engine, ["sz000001", evil], "2024-01-01", "2099-12-31")
    assert "sz000001" in hist                      # 正常符号仍可加载
    assert evil not in hist                        # 注入串仅当数据处理
    with engine.connect() as c:
        assert c.execute(text(
            "SELECT to_regclass('public.stock_st_history_pit_test')"
        )).scalar() is not None                    # 表未被 DROP


def test_sync_coverage_guard_and_force(engine, scratch, monkeypatch) -> None:
    calls = {"st": 0, "names": 0}

    def _st_list():
        calls["st"] += 1
        return pd.DataFrame({"代码": ["000001"]})

    def _names():
        calls["names"] += 1
        return None

    monkeypatch.setattr(sp, "_fetch_st_list", _st_list)
    monkeypatch.setattr(sp, "_fetch_sz_change_names", _names)
    for f in ("_fetch_spot_list", "_fetch_sz_delist", "_fetch_sh_delist"):
        monkeypatch.setattr(sp, f, lambda: None)

    # P2P2 修复：空 PIT 表 + 窗口内从未出现 ST/退市标记 → 非 ST 股直接视为覆盖
    # → 跳过同步（不再每次回测全量重拉网络源）
    sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01", end_date="2099-12-31")
    assert calls["st"] == 0 and calls["names"] == 0

    # 覆盖不达标（标记股 PIT 历史缺失）→ 重跑回填源
    monkeypatch.setattr(sp, "_pool_covered", lambda *a, **k: False)
    sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01", end_date="2099-12-31")
    assert calls["st"] == 1 and calls["names"] == 1

    # force=True → 强制回填（今日归档已有行，仅回填源重新拉取）
    sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01",
                   end_date="2099-12-31", force=True)
    assert calls["names"] == 2
    assert calls["st"] == 1


def test_pool_covered_never_st_skips_and_marked_reruns(engine, scratch, monkeypatch) -> None:
    """P2P2 修复：覆盖率只统计"窗口内出现过 ST/退市标记的股票"，非 ST 股直接视为覆盖。"""
    calls = {"st": 0}

    def _st_list():
        calls["st"] += 1
        return pd.DataFrame({"代码": ["000001"]})

    _no_network(monkeypatch)
    monkeypatch.setattr(sp, "_fetch_st_list", _st_list)

    # 空 PIT 表 + 非 ST 股 → 直接视为覆盖 → 跳过同步（不拉任何网络源）
    sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01", end_date="2024-12-31")
    assert calls["st"] == 0

    # 该股窗口内出现 ST 标记（1 行）→ 计入覆盖检查：1 标记行 vs 数百 K 线交易日
    # → 未覆盖 → 重跑同步（标记股历史缺失需回填）
    with engine.begin() as c:
        c.execute(text(
            f"INSERT INTO {scratch} (symbol, trade_date, is_st, is_delisting) "
            "VALUES ('sz000001', '2024-06-03', TRUE, FALSE)"
        ))
    sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01", end_date="2024-12-31")
    assert calls["st"] == 1


def test_sync_offline_degrades_gracefully(engine, scratch, monkeypatch) -> None:
    """全部网络源失败（None）→ 不抛异常、不写行、告警统计为 0。"""
    _no_network(monkeypatch)
    stats = sp.sync_st_pit(engine, ["sz000001"], start_date="2024-01-01", end_date="2024-12-31")
    assert stats == {"archive_today": 0, "sz_st_rows": 0, "delist_rows": 0}
    assert sp.load_st_pit(engine, ["sz000001"], "2024-01-01", "2024-12-31") == {}