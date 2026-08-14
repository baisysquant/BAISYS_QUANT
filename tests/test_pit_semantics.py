"""P0-8 审计测试：PIT（as-of）语义 + 北向资金因子移除。

覆盖：
  ① 财务质量因子 PIT：披露截止日推导、as-of 过滤条件、默认查询日
  ② 资金流 PIT：历史日期只读归档（拒绝把当日最新写入历史日期，防前视）、
     当日拉取按实际交易日归档
  ③ 北向资金因子已从评分链路移除（注册表无权重、fuse_scores 无北向列）
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def config_fixture(tmp_path):
    cfg = MagicMock()
    cfg.ASHAREHUB_API_KEY = "test_key_123"
    cfg.MONEYFLOW_RETRY = 1
    cfg.MONEYFLOW_PAGE_DELAY = 0.1
    cfg.TEMP_DATA_DIRECTORY = str(tmp_path / "mf_cache")
    cfg.FINANCIAL_QUALITY_CACHE_DAYS = 90
    cfg.FINANCIAL_QUALITY_BATCH_SIZE = 500
    cfg.FINANCIAL_QUALITY_BATCH_SLEEP = 20
    cfg.FINANCIAL_QUALITY_FILE_CACHE_DAYS = 30
    cfg.DB_USER = "test"
    cfg.DB_PASSWORD = "test"
    cfg.DB_HOST = "localhost"
    cfg.DB_PORT = 5432
    cfg.DB_NAME = "test"
    cfg.CONFIG_DIR = "config"
    return cfg


# ── ① 质量因子 PIT ─────────────────────────────────────────────

class TestFinancialQualityPit:
    def test_disclosure_deadline_rules(self) -> None:
        from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher

        dl = FinancialQualityFetcher._disclosure_deadline
        assert dl("2023-03-31") == "2023-04-30"
        assert dl("2023-06-30") == "2023-08-31"
        assert dl("2023-09-30") == "2023-10-31"
        assert dl("2023-12-31") == "2024-04-30"  # 年报截止次年 4-30

    def test_fetch_one_contains_disclosure_date(self, config_fixture, monkeypatch) -> None:
        from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher

        raw = pd.DataFrame({
            "选项": ["A", "B", "C", "D", "E"],
            "指标": ["净资产收益率", "毛利率", "销售净利率", "营业总收入增长率", "归属母公司净利润增长率"],
            "2024-03-31": [10.0, 30.0, 8.0, 5.0, 6.0],
            "2023-12-31": [9.0, 28.0, 7.0, 4.0, 5.0],
        })
        monkeypatch.setattr(
            "akshare.stock_financial_abstract",
            lambda symbol: raw,
        )
        fetcher = FinancialQualityFetcher(config_fixture)
        row = fetcher.fetch_one("600000")
        assert row is not None
        assert row["record_date"] == "2024-03-31"
        assert row["disclosure_date"] == "2024-04-30"
        assert row["roe"] == 10.0

    def test_load_quality_filters_by_disclosure_as_of(self, config_fixture, monkeypatch) -> None:
        from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher

        fetcher = FinancialQualityFetcher(config_fixture)
        fake_conn = MagicMock()
        fake_engine = MagicMock()
        fake_engine.connect.return_value.__enter__.return_value = fake_conn
        monkeypatch.setattr(fetcher, "_engine", fake_engine)
        captured: dict = {}

        def fake_read_sql(sql, conn, params=None):  # noqa: ANN001
            captured["sql"] = str(sql)
            captured["params"] = dict(params or {})
            return pd.DataFrame()

        monkeypatch.setattr(pd, "read_sql", fake_read_sql)
        fetcher.load_quality(["600000", "000001"], as_of="2026-08-14")

        assert "disclosure_date" in captured["sql"]
        assert "disclosure_date <= :as_of" in captured["sql"]
        assert "record_date <= :as_of" in captured["sql"]  # 旧数据回退
        assert "DISTINCT ON (symbol)" in captured["sql"]
        assert captured["params"]["as_of"] == "2026-08-14"

    def test_load_quality_normalizes_as_of_formats(self, config_fixture, monkeypatch) -> None:
        from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher

        fetcher = FinancialQualityFetcher(config_fixture)
        fake_conn = MagicMock()
        fake_engine = MagicMock()
        fake_engine.connect.return_value.__enter__.return_value = fake_conn
        monkeypatch.setattr(fetcher, "_engine", fake_engine)
        captured: dict = {}

        def fake_read_sql(sql, conn, params=None):  # noqa: ANN001
            captured["params"] = dict(params or {})
            return pd.DataFrame()

        monkeypatch.setattr(pd, "read_sql", fake_read_sql)
        fetcher.load_quality(as_of="20260814")
        assert captured["params"]["as_of"] == "2026-08-14"


# ── ② 资金流 PIT ───────────────────────────────────────────────

class TestMoneyFlowPit:
    @staticmethod
    def _today_patch(monkeypatch, day: str = "20260814") -> None:
        from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher

        monkeypatch.setattr(MoneyFlowFetcher, "_today", property(lambda self: day))

    def test_historical_without_archive_refuses_fetch(self, config_fixture, monkeypatch, tmp_path) -> None:
        from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher

        self._today_patch(monkeypatch)
        fetcher = MoneyFlowFetcher(config_fixture)
        # 即使有 API 密钥也不允许拉取：历史日期必须命中归档
        result = fetcher.fetch_all("20260810")
        assert result.empty
        # 不得生成目标日期命名的缓存（防前视）
        assert not (tmp_path / "mf_cache" / "moneyflow_20260810.csv").exists()

    def test_historical_with_archive_reads_archive(self, config_fixture, monkeypatch, tmp_path) -> None:
        from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher

        self._today_patch(monkeypatch)
        cache_dir = tmp_path / "mf_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "moneyflow_20260810.csv").write_text(
            "ts_code,trade_date,net_mf_amount\n000001.SZ,2026-08-10,100\n",
            encoding="utf-8",
        )
        fetcher = MoneyFlowFetcher(config_fixture)
        result = fetcher.fetch_all("20260810")
        assert not result.empty
        assert result["net_mf_amount"].iloc[0] == 100

    def test_today_fetch_archives_under_actual_trade_date(self, config_fixture, monkeypatch, tmp_path) -> None:
        from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher

        self._today_patch(monkeypatch)  # 今天 = 2026-08-14
        fetcher = MoneyFlowFetcher(config_fixture)
        fake = MagicMock()
        fake.moneyflow.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ", "600000.SH"],
            "trade_date": ["2026-08-13", "2026-08-13"],  # 接口返回数据实际交易日
            "net_mf_amount": [1.0, 2.0],
        })
        fetcher._client = fake
        result = fetcher.fetch_all("20260814")
        assert not result.empty
        # 按实际交易日归档，而非请求日期
        assert (tmp_path / "mf_cache" / "moneyflow_20260813.csv").exists()
        assert not (tmp_path / "mf_cache" / "moneyflow_20260814.csv").exists()

    def test_today_fetch_archives_when_actual_equals_today(self, config_fixture, monkeypatch, tmp_path) -> None:
        from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher

        self._today_patch(monkeypatch)
        fetcher = MoneyFlowFetcher(config_fixture)
        fake = MagicMock()
        fake.moneyflow.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "trade_date": ["2026-08-14"],
            "net_mf_amount": [5.0],
        })
        fetcher._client = fake
        result = fetcher.fetch_all("20260814")
        assert not result.empty
        assert (tmp_path / "mf_cache" / "moneyflow_20260814.csv").exists()

    def test_today_cache_hit_skips_api(self, config_fixture, monkeypatch, tmp_path) -> None:
        from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher

        self._today_patch(monkeypatch)
        cache_dir = tmp_path / "mf_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "moneyflow_20260814.csv").write_text(
            "ts_code,trade_date,net_mf_amount\n000001.SZ,2026-08-14,42\n",
            encoding="utf-8",
        )
        fetcher = MoneyFlowFetcher(config_fixture)
        fake = MagicMock()
        fetcher._client = fake
        result = fetcher.fetch_all("20260814")
        assert result["net_mf_amount"].iloc[0] == 42
        fake.moneyflow.assert_not_called()


# ── ③ 北向资金因子移除 ─────────────────────────────────────────

class TestNorthFlowRemoved:
    def test_registry_has_no_north_flow(self) -> None:
        from LogicAnalyzer.scoring.factor_registry import FactorRegistry

        reg = FactorRegistry("config/factor_registry.yaml")
        assert "north_flow" not in reg.weights

    def test_fuse_scores_has_no_north_column(self, config_fixture) -> None:
        from LogicAnalyzer.scoring.calculator import FactorCalculator

        calc = FactorCalculator(config_fixture, None)
        report = pd.DataFrame({
            "股票代码": ["600000", "000001"],
            "行业": ["银行", "银行"],
            "综合分析评分": [60, 40],
            "MACD评分": [1.0, -0.5],
            "3日资金流入万元": [100, 50],
            "5日资金流入万元": [200, 90],
            "10日资金流入万元": [300, 150],
            "20日资金流入万元": [400, 200],
        })
        out = calc.fuse_scores(report, macd_score_col="综合分析评分", industry_col="行业",
                               hist_df=pd.DataFrame())
        assert "北向资金评分" not in out.columns
        assert "综合分析评分" in out.columns
        assert out["综合分析评分"].notna().all()