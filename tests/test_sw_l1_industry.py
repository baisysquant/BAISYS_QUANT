"""P0-7 ①：申万一级中性化静默失效修复测试。

覆盖：SW2021 全部 31 个一级行业在 MacroFactorFetcher 宏观分类中全覆盖
（宏观 tilt 不再静默为 0）；coordinator._load_sw_l1_map 查询/归一化/缓存/
失败响亮报错；SwIndustrySync 当日快照语义（先删当日再全量插入）。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text


# SW2021 申万一级行业（31 个）
SW2021_L1_NAMES = {
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "家用电器",
    "食品饮料", "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商贸零售", "社会服务", "综合", "建筑材料", "建筑装饰",
    "电力设备", "国防军工", "计算机", "传媒", "通信", "银行", "非银金融",
    "汽车", "机械设备", "煤炭", "石油石化", "环保", "美容护理",
}


def _l1_db() -> tuple:
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE stock_basic_info_sw_l1 ("
            "stock_code TEXT, l1_name TEXT, record_date DATE)"
        ))
        conn.execute(text(
            "INSERT INTO stock_basic_info_sw_l1 VALUES "
            "('600000', '银行', '2026-08-13'), ('600000', '银行', '2026-08-14'), "
            "('sh600519', '食品饮料', '2026-08-14'), "
            "('000001', '银行', '2026-08-14'), "
            "('300750', '电力设备', '2026-08-14')"
        ))
    return engine


class TestMacroTiltL1Coverage:
    def test_all_sw2021_l1_names_covered(self) -> None:
        from DataCollection.MacroFactorFetcher import _SW1_MACRO_CLASS

        missing = SW2021_L1_NAMES - set(_SW1_MACRO_CLASS)
        assert not missing, f"宏观分类缺失申万一级行业: {sorted(missing)}"

    def test_every_l1_name_gets_non_zero_tilt_in_extreme_regime(self) -> None:
        from DataCollection.MacroFactorFetcher import _SW1_MACRO_CLASS, REGIME_INDUSTRY_TILT

        class_tilts = REGIME_INDUSTRY_TILT["boom"]
        for name in SW2021_L1_NAMES:
            cls = _SW1_MACRO_CLASS[name]
            assert cls in class_tilts, f"{name} → 类别 {cls} 无 boom tilt"


class TestLoadSwL1Map:
    def _coord(self, engine, logger=None):
        from Review.coordinator import StockAnalysisCoordinator

        co = object.__new__(StockAnalysisCoordinator)
        co.db_engine = engine
        co.logger = logger
        co._sw_l1_map = None
        return co

    def test_loads_latest_record_date_with_normalization(self) -> None:
        engine = _l1_db()
        co = self._coord(engine)

        m = co._load_sw_l1_map()
        assert m is not None
        # 取最新 record_date；带市场前缀的代码被归一化；非 6 位数字被剔除
        assert m["600000"] == "银行"
        assert m["000001"] == "银行"
        assert m["600519"] == "食品饮料"
        assert "300750" in m and m["300750"] == "电力设备"

    def test_caches_result_across_calls(self) -> None:
        engine = _l1_db()
        co = self._coord(engine)
        first = co._load_sw_l1_map()
        second = co._load_sw_l1_map()
        assert first == second
        assert co._sw_l1_map is not None

    def test_failure_is_loud_and_cached(self) -> None:
        """表缺失 → 返回 None 且记录 error 日志（不再 except: pass 静默）。"""
        engine = create_engine("sqlite://")
        errors: list[str] = []

        class LoggerStub:
            def error(self, msg):  # noqa: ANN001
                errors.append(msg)

        co = self._coord(engine, LoggerStub())
        assert co._load_sw_l1_map() is None
        assert errors and "申万一级" in errors[0]
        # 失败结果同样缓存，避免重复查询刷屏
        assert co._sw_l1_map == {}
        assert co._load_sw_l1_map() is None
        assert len(errors) == 1

    def test_empty_table_returns_none_with_error(self) -> None:
        engine = create_engine("sqlite://")
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE stock_basic_info_sw_l1 ("
                "stock_code TEXT, l1_name TEXT, record_date DATE)"
            ))
        errors: list[str] = []

        class LoggerStub:
            def error(self, msg):  # noqa: ANN001
                errors.append(msg)

        co = self._coord(engine, LoggerStub())
        assert co._load_sw_l1_map() is None
        assert any("无数据" in e for e in errors)


class TestSwIndustrySync:
    def test_snapshot_semantics(self, monkeypatch) -> None:
        from DataManager.SwIndustrySync import sync_sw_l1_industries

        members = pd.DataFrame({
            "l1_code": ["801780", "801120"],
            "l1_name": ["银行", "食品饮料"],
            "stock_code": ["600000", "600519"],
            "stock_name": ["浦发银行", "贵州茅台"],
        })
        monkeypatch.setattr(
            "DataManager.SwIndustrySync.fetch_sw_l1_memberships",
            lambda: members,
        )

        engine = create_engine("sqlite://")
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE stock_basic_info_sw_l1 ("
                "l1_code TEXT, l1_name TEXT, stock_code TEXT, stock_name TEXT, record_date DATE)"
            ))

        n = sync_sw_l1_industries(engine, trade_date=date(2026, 8, 14))
        assert n == 2
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT stock_code, l1_name, record_date FROM stock_basic_info_sw_l1")
            ).fetchall()
        assert len(rows) == 2
        assert {r[0] for r in rows} == {"600000", "600519"}

        # 同日记全量替换（快照语义）：成分变化后旧行消失
        changed = members.assign(stock_name=["浦发银行", "贵州茅台"])
        changed = pd.concat([changed, pd.DataFrame({
            "l1_code": ["801080"], "l1_name": ["电子"],
            "stock_code": ["002415"], "stock_name": ["海康威视"],
        })], ignore_index=True)
        monkeypatch.setattr(
            "DataManager.SwIndustrySync.fetch_sw_l1_memberships",
            lambda: changed,
        )
        n2 = sync_sw_l1_industries(engine, trade_date=date(2026, 8, 14))
        assert n2 == 3
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT stock_code, record_date FROM stock_basic_info_sw_l1")
            ).fetchall()
        assert len(rows) == 3

        # 新日期追加，旧日期保留
        sync_sw_l1_industries(engine, trade_date=date(2026, 8, 15))
        with engine.connect() as conn:
            dates = conn.execute(
                text("SELECT DISTINCT record_date FROM stock_basic_info_sw_l1")
            ).fetchall()
        assert {str(d[0]) for d in dates} == {"2026-08-14", "2026-08-15"}

    def test_normalize_code_drops_invalid(self) -> None:
        from DataManager.SwIndustrySync import _normalize_code

        assert _normalize_code("sh600000") == "600000"
        assert _normalize_code("SZ000001") == "000001"
        assert _normalize_code("600000") == "600000"
        assert _normalize_code("abc") is None