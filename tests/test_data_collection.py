"""DataCollection 模块测试：3 个新增采集器 + 2 个已有采集器的 mock 测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _mock_db_engine():
    """全局 mock 数据库引擎，避免所有采集器初始化时连接真实数据库。"""
    with patch("DataCollection.FinancialQualityFetcher.get_engine") as m1, \
         patch("DataCollection.FinancialValuationFetcher.get_engine") as m2, \
         patch("DataCollection.BenchmarkFetcher.get_engine") as m3:
        mock_eng = MagicMock()
        m1.return_value = mock_eng
        m2.return_value = mock_eng
        m3.return_value = mock_eng
        yield


# ── FinancialQualityFetcher ────────────────────────────────────

class TestFinancialQualityFetcher:
    """akShare 质量因子采集器测试。"""

    @pytest.fixture
    def mock_akshare(self):
        with patch("akshare.stock_financial_abstract") as m:
            # 模拟 stock_financial_abstract 返回格式：col[0]="选项"、col[1]="指标"、col[2:]=报告日期
            rows = [
                ("每股指标", "净资产收益率", 15.5),
                ("每股指标", "毛利率", 45.2),
                ("每股指标", "销售净利率", 12.3),
                ("成长能力", "营业总收入增长率", 8.7),
                ("成长能力", "归属母公司净利润增长率", 10.1),
            ]
            df = pd.DataFrame(rows, columns=["选项", "指标", "2024-12-31"])
            m.return_value = df
            yield m

    @pytest.mark.unit
    def test_parse_one(self, mock_akshare, config_fixture):
        from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher

        fetcher = FinancialQualityFetcher(config_fixture)
        result = fetcher.fetch_one("600519")
        assert result is not None
        assert result["symbol"] == "600519"
        assert result["roe"] == 15.5
        assert result["gross_profit_margin"] == 45.2
        assert result["net_profit_margin"] == 12.3
        assert result["revenue_growth_rate"] == 8.7
        assert result["net_profit_growth_rate"] == 10.1

    @pytest.mark.unit
    def test_empty_input(self, config_fixture):
        with patch("akshare.stock_financial_abstract") as m:
            m.return_value = pd.DataFrame()
            from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher
            result = FinancialQualityFetcher(config_fixture).fetch_one("600519")
            assert result is None


# ── FinancialValuationFetcher ──────────────────────────────────

class TestFinancialValuationFetcher:
    """AShareHub 估值因子采集器测试。"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        df = pd.DataFrame({
            "symbol": ["000001.SZ", "600519.SH"],
            "pe": [5.5, 30.2],
            "pe_ttm": [5.8, 28.5],
            "pb": [0.8, 8.2],
            "total_mv": [25000000.0, 210000000.0],
            "circ_mv": [20000000.0, 180000000.0],
        })
        client.fundamentals.return_value = df
        return client

    @pytest.mark.unit
    def test_fetch_by_date(self, mock_client, config_fixture):
        with patch.object(type(config_fixture), "ENABLE_FUNDAMENTALS", True, create=True):
            from DataCollection.FinancialValuationFetcher import FinancialValuationFetcher
            fetcher = FinancialValuationFetcher(config_fixture)
            fetcher._client = mock_client
            result = fetcher.fetch_by_date("20260105")
            assert not result.empty
            assert "symbol" in result.columns
            assert "pe_ttm" in result.columns
            assert len(result) == 2

    @pytest.mark.unit
    def test_no_api_key(self, config_fixture):
        from DataCollection.FinancialValuationFetcher import FinancialValuationFetcher
        fetcher = FinancialValuationFetcher(config_fixture)
        fetcher._api_key = ""
        result = fetcher.fetch_by_date("20260105")
        assert result.empty


# ── BenchmarkFetcher ───────────────────────────────────────────

class TestBenchmarkFetcher:
    """基准指数采集器测试。"""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        df = pd.DataFrame({
            "ts_code": ["000001.SH", "000001.SH"],
            "trade_date": ["2026-01-02", "2026-01-03"],
            "close": [3200.5, 3215.8],
            "open": [3198.0, 3205.2],
            "high": [3210.3, 3220.1],
            "low": [3195.2, 3202.0],
            "vol": [350000000, 380000000],
            "amount": [420000000000, 450000000000],
        })
        client.index_daily.return_value = df
        return client

    @pytest.mark.unit
    def test_fetch_index(self, mock_client, config_fixture):
        from DataCollection.BenchmarkFetcher import BenchmarkFetcher
        fetcher = BenchmarkFetcher(config_fixture)
        fetcher._client = mock_client
        result = fetcher.fetch_index("000001.SH")
        assert not result.empty
        assert "index_code" in result.columns
        assert len(result) == 2
        assert result["close"].iloc[-1] == 3215.8

    @pytest.mark.unit
    def test_empty_response(self, config_fixture):
        from DataCollection.BenchmarkFetcher import BenchmarkFetcher
        fetcher = BenchmarkFetcher(config_fixture)
        fetcher._api_key = ""
        result = fetcher.fetch_index()
        assert result.empty


# ── MoneyFlowFetcher (已有) ────────────────────────────────────

class TestMoneyFlowFetcher:
    """资金流采集器测试（避免真实网络请求）。"""

    @pytest.mark.unit
    def test_no_api_key(self, config_fixture):
        from DataCollection.MoneyFlowFetcher import MoneyFlowFetcher
        fetcher = MoneyFlowFetcher(config_fixture)
        fetcher.api_key = ""
        result = fetcher.fetch_all("20260105")
        assert result.empty


# ── ChipDistributionFetcher (已有) ─────────────────────────────

class TestChipDistributionFetcher:
    """筹码分布采集器测试。"""

    @pytest.mark.unit
    def test_disabled(self, config_fixture):
        with patch.object(type(config_fixture), "ENABLE_CHIP_DISTRIBUTION", False, create=True):
            from DataCollection.ChipDistributionFetcher import ChipDistributionFetcher
            fetcher = ChipDistributionFetcher(config_fixture)
            result = fetcher.fetch_chip_data(date="20260105")
            assert result.empty


# ── 共享 Fixture ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def config_fixture():
    """返回一个最小配置对象（不依赖真实 DB / API）。"""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.ASHAREHUB_API_KEY = "test_key_123"
    cfg.ENABLE_CHIP_DISTRIBUTION = True
    cfg.ENABLE_FUNDAMENTALS = True
    cfg.FUNDAMENTALS_RETRY = 1
    cfg.MONEYFLOW_RETRY = 1
    cfg.MONEYFLOW_PAGE_DELAY = 0.1
    cfg.FINANCIAL_QUALITY_CACHE_DAYS = 90
    cfg.FINANCIAL_QUALITY_BATCH_SIZE = 500
    cfg.FINANCIAL_QUALITY_BATCH_SLEEP = 20
    cfg.FINANCIAL_QUALITY_FILE_CACHE_DAYS = 30
    cfg.TEMP_DATA_DIRECTORY = "/tmp/test_cache"
    cfg.DB_USER = "test"
    cfg.DB_PASSWORD = "test"
    cfg.DB_HOST = "localhost"
    cfg.DB_PORT = 5432
    cfg.DB_NAME = "test"
    return cfg
