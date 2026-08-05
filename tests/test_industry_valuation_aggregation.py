"""行业估值聚合（个股→申万二级行业市值加权）单元测试"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from LogicAnalyzer.Industrytrending import SWIndustryDataPipeline


def _make_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """行业映射 + 个股估值 + 行业骨架"""
    ah_df = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        "l2_code": ["801111.SI", "801111.SI", "801222.SI", "801222.SI"],
    })
    fundamentals = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        "pe": [10.0, -5.0, 20.0, np.nan],
        "pe_ttm": [11.0, -6.0, 22.0, 30.0],
        "pb": [1.0, 0.5, 2.0, 3.0],
        "dv_ratio": [5.0, 0.0, 2.0, 10.0],
        "total_mv": [100.0, 50.0, 40.0, 60.0],
    })
    df_val = pd.DataFrame({
        "code": ["801111.SI", "801222.SI"],
        "name": ["行业A", "行业B"],
        "pe_static": [None, None],
        "pe_ttm": [None, None],
        "pb": [None, None],
        "div_yield": [None, None],
    })
    return ah_df, fundamentals, df_val


class TestAggregateProfitable:
    """中证口径：剔除亏损股"""

    def test_pe_excludes_loss_makers(self):
        ah, f, dv = _make_data()
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f, dv, "aggregate_profitable")
        # 行业A: 盈利股 000001(mv=100, pe=10) → PE=100/(100/10)=10; 亏损股 000002 剔除
        assert out.loc[out["code"] == "801111.SI", "pe_static"].item() == pytest.approx(10.0)
        # 行业B TTM: 000003(mv=40, 22) + 000004(mv=60, 30) → 100/(40/22+60/30)=26.19
        assert out.loc[out["code"] == "801222.SI", "pe_ttm"].item() == pytest.approx(26.190476, rel=1e-5)

    def test_pb_uses_harmonic(self):
        ah, f, dv = _make_data()
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f, dv, "aggregate_profitable")
        # 行业A PB: 000001(mv=100,pb=1) + 000002(mv=50,pb=0.5) → (150)/(100/1+50/0.5)=150/200=0.75
        assert out.loc[out["code"] == "801111.SI", "pb"].item() == pytest.approx(0.75)

    def test_div_yield_weighted_mean(self):
        ah, f, dv = _make_data()
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f, dv, "aggregate_profitable")
        # 行业B 股息率: (40*2 + 60*10)/(40+60) = 680/100 = 6.8
        assert out.loc[out["code"] == "801222.SI", "div_yield"].item() == pytest.approx(6.8)

    def test_all_loss_industry_pe_is_nan(self):
        ah, f, dv = _make_data()
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f, dv, "aggregate_profitable")
        # 行业B pe_static: 000003=20, 000004=NaN → 仅 000003 盈利 → PE = 40/(40/20)=20（非全亏）
        # 构造全亏: 覆盖为仅 000002 的行业
        ah2 = pd.DataFrame({"symbol": ["000002.SZ"], "l2_code": ["801333.SI"]})
        dv2 = pd.DataFrame({"code": ["801333.SI"], "name": ["行业C"],
                            "pe_static": [None], "pe_ttm": [None],
                            "pb": [None], "div_yield": [None]})
        out2 = SWIndustryDataPipeline._aggregate_industry_valuation(ah2, f, dv2, "aggregate_profitable")
        assert np.isnan(out2.loc[out2["code"] == "801333.SI", "pe_static"].item())


class TestAggregateFull:
    """申万口径：整体法含负利润"""

    def test_pe_counts_loss_makers_in_denominator(self):
        ah, f, dv = _make_data()
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f, dv, "aggregate_full")
        # 行业A: 总市值=150, 总盈利 = 100/10 + 50/(-5) = 10-10 = 0 → 盈亏平衡 → NaN
        assert np.isnan(out.loc[out["code"] == "801111.SI", "pe_static"].item())
        # 行业A TTM: 100/11 + 50/(-6) = 9.09-8.33 = 0.7576 → PE=150/0.7576≈198.0
        pe_ttm = out.loc[out["code"] == "801111.SI", "pe_ttm"].item()
        assert pe_ttm == pytest.approx(150 / (100 / 11 + 50 / (-6)))

    def test_industry_net_loss_pe_negative(self):
        ah, f, dv = _make_data()
        # 行业B pe_static: 总市值=100, 盈利 = 40/20 + NaN(剔除) = 2 → PE=50
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f, dv, "aggregate_full")
        assert out.loc[out["code"] == "801222.SI", "pe_static"].item() == pytest.approx(50.0)
        # 构造整体亏损: 000003(pe=20, mv=40 → E=2) + 000004(pe=-10, mv=60 → E=-6)
        # 总E=-4 → PE = 100/(-4) = -25
        f2 = f.copy()
        f2.loc[f2["symbol"] == "000004.SZ", "pe"] = -10.0
        out2 = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f2, dv, "aggregate_full")
        pe_b = out2.loc[out2["code"] == "801222.SI", "pe_static"].item()
        assert pe_b == pytest.approx(100 / (40 / 20 + 60 / (-10)))

    def test_pb_negative_equity_counts(self):
        ah, f, dv = _make_data()
        f2 = f.copy()
        f2.loc[f2["symbol"] == "000004.SZ", "pb"] = -1.0
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f2, dv, "aggregate_full")
        # 行业B: 总市值=100, 净资产 = 40/2 + 60/(-1) = 20-60 = -40 → PB=100/(-40)=-2.5
        assert out.loc[out["code"] == "801222.SI", "pb"].item() == pytest.approx(-2.5)


class TestEdgeCases:
    def test_empty_fundamentals_returns_unchanged(self):
        ah, f, dv = _make_data()
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f.head(0), dv)
        assert out["pe_static"].isna().all()

    def test_symbol_not_in_industry_map_dropped(self):
        ah = pd.DataFrame({"symbol": ["000001.SZ"], "l2_code": ["801111.SI"]})
        f = pd.DataFrame({"symbol": ["000001.SZ", "999999.SZ"],
                          "pe": [10.0, 5.0], "pe_ttm": [10.0, 5.0],
                          "pb": [1.0, 1.0], "dv_ratio": [2.0, 2.0],
                          "total_mv": [100.0, 100.0]})
        dv = pd.DataFrame({"code": ["801111.SI"], "name": ["行业A"],
                           "pe_static": [None], "pe_ttm": [None],
                           "pb": [None], "div_yield": [None]})
        out = SWIndustryDataPipeline._aggregate_industry_valuation(ah, f, dv)
        assert out.loc[0, "pe_static"] == pytest.approx(10.0)
