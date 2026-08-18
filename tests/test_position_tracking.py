from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from LogicAnalyzer.portfolio.tracking import COLUMNS, PositionTrackingService


def _make_service(
    xlsx_rows: list[dict] | None = None,
    config_overrides: str = "",
) -> tuple[PositionTrackingService, Path]:
    base = Path(tempfile.mkdtemp(prefix="baisys_test_bt_"))
    cfg_path = base / "config.ini"
    cfg_path.write_text(
        "[SYSTEM]\nHOME_DIRECTORY = ~/test_baisys\n\n"
        "[LOGGING]\nLOG_LEVEL = DEBUG\n\n"
        "[TRADING_COST]\ncommission_rate = 0.0\nstamp_tax_rate = 0.0\ntransfer_fee_rate = 0.0\n\n"
        f"{config_overrides}"
        "[POSITION_BACKTEST]\npool_file_path = 证券交割单.xlsx\n",
        encoding="utf-8",
    )
    xlsx_path = base / "证券交割单.xlsx"
    if xlsx_rows:
        pd.DataFrame(xlsx_rows).to_excel(xlsx_path, sheet_name="交割单", index=False)
    else:
        pd.DataFrame().to_excel(xlsx_path, sheet_name="交割单", index=False)

    cfg = MagicMock()
    cfg.POOL_FILE_PATH = str(xlsx_path)
    cfg.config_file = str(cfg_path)
    cfg.SCORING_PARAMS = {}
    cfg.TRADING_COST_PARAMS = {"commission_rate": 0.0, "stamp_tax_rate": 0.0, "transfer_fee_rate": 0.0}

    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()

    data_provider = MagicMock()
    calendar_mgr = MagicMock()
    calendar_mgr.get_last_trading_day.return_value = "2026-07-03"

    svc = PositionTrackingService(
        config=cfg,
        logger=logger,
        data_provider=data_provider,
        calendar_mgr=calendar_mgr,
        db_engine=None,
    )
    return svc, base


def _make_kline_df(close_prices: list[float], symbol: str = "sh600933") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(close_prices), freq="B")
    high = [c * 1.02 for c in close_prices]
    low = [c * 0.98 for c in close_prices]
    return pd.DataFrame({
        "symbol": symbol,
        "trade_date": dates.strftime("%Y-%m-%d"),
        "open": close_prices,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": [1_000_000] * len(close_prices),
        "amount": [c * 1_000_000 for c in close_prices],
    })


def _row(
    code: str,
    date: str,
    direction: str,
    price: float | None,
    volume: int,
) -> dict:
    return {
        "成交日期": date,
        "证券代码": code,
        "证券名称": "",
        "买卖标志": direction,
        "成交价格": price,
        "成交数量": volume,
    }


class TestLoadRecords:
    @pytest.mark.unit
    def test_basic_buy(self):
        rows = [_row("600933", "20260703", "买入", 15.41, 500)]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert len(records) == 1
        assert records[0]["entry_shares"] == 500
        assert records[0]["entry_price"] == 15.41

    @pytest.mark.unit
    def test_buy_and_sell(self):
        rows = [
            _row("600933", "20260703", "买入", 15.41, 500),
            _row("600933", "20260710", "卖出", 16.0, 200),
        ]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert len(records) == 2
        assert records[0]["entry_shares"] == 500
        assert records[1]["entry_shares"] == -200

    @pytest.mark.unit
    def test_invalid_direction_skipped(self):
        rows = [_row("600933", "20260703", "买", 15.41, 500)]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert len(records) == 0

    @pytest.mark.unit
    def test_price_blank(self):
        rows = [_row("600933", "20260703", "买入", None, 500)]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert len(records) == 1
        assert records[0]["entry_price"] is None
        assert records[0]["entry_shares"] == 500

    @pytest.mark.unit
    def test_invalid_code_skipped(self):
        rows = [
            _row("xxxxx", "20260703", "买入", 15.0, 500),
            _row("600933", "20260703", "买入", 15.0, 500),
        ]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert len(records) == 1

    @pytest.mark.unit
    def test_invalid_date_skipped(self):
        rows = [
            _row("600933", "202607", "买入", 15.0, 500),
            _row("600933", "20260703", "买入", 15.0, 500),
        ]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert len(records) == 1

    @pytest.mark.unit
    def test_zero_shares_skipped(self):
        rows = [
            _row("600933", "20260703", "买入", 15.0, 0),
            _row("600933", "20260703", "买入", 15.0, 500),
        ]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert len(records) == 1

    @pytest.mark.unit
    def test_file_not_found(self):
        svc, _ = _make_service()
        os.remove(svc.config.POOL_FILE_PATH)
        assert svc._load_records(svc.config.POOL_FILE_PATH) == []

    @pytest.mark.unit
    def test_empty_sheet(self):
        svc, _ = _make_service([])
        assert svc._load_records(svc.config.POOL_FILE_PATH) == []


class TestGroupByStock:
    @pytest.mark.unit
    def test_multiple_buys_aggregate(self):
        records = [
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 500, "entry_price": 15.0},
            {"stock_code": "600933", "entry_time": "20260702", "entry_shares": 300, "entry_price": 16.0},
        ]
        svc, _ = _make_service("")
        groups = svc._group_by_stock(records)
        assert len(groups) == 1
        g = groups[0]
        assert g["stock_code"] == "600933"
        assert g["entry_shares"] == 800
        expected_avg = (500 * 15.0 + 300 * 16.0) / 800
        assert g["entry_price"] == round(expected_avg, 2)

    @pytest.mark.unit
    def test_buy_then_partial_sell_fifo(self):
        records = [
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 1000, "entry_price": 10.0},
            {"stock_code": "600933", "entry_time": "20260710", "entry_shares": -300, "entry_price": 12.0},
        ]
        svc, _ = _make_service("")
        groups = svc._group_by_stock(records)
        assert len(groups) == 1
        g = groups[0]
        assert g["entry_shares"] == 700
        assert g["entry_price"] == 10.0
        assert g["total_realized_pnl"] == pytest.approx(600.0)

    @pytest.mark.unit
    def test_full_sell_skipped(self):
        records = [
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 1000, "entry_price": 10.0},
            {"stock_code": "600933", "entry_time": "20260710", "entry_shares": -1000, "entry_price": 12.0},
        ]
        svc, _ = _make_service("")
        groups = svc._group_by_stock(records)
        assert len(groups) == 0

    @pytest.mark.unit
    def test_sell_exceeds_buy_skipped(self):
        records = [
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 500, "entry_price": 10.0},
            {"stock_code": "600933", "entry_time": "20260710", "entry_shares": -800, "entry_price": 12.0},
        ]
        svc, _ = _make_service("")
        groups = svc._group_by_stock(records)
        assert len(groups) == 0

    @pytest.mark.unit
    def test_multiple_stocks(self):
        records = [
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 500, "entry_price": 15.0},
            {"stock_code": "000001", "entry_time": "20260601", "entry_shares": 200, "entry_price": 20.0},
        ]
        svc, _ = _make_service("")
        groups = svc._group_by_stock(records)
        assert len(groups) == 2

    @pytest.mark.unit
    def test_earliest_entry_time_used(self):
        records = [
            {"stock_code": "600933", "entry_time": "20260703", "entry_shares": 500, "entry_price": 15.0},
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 300, "entry_price": 16.0},
        ]
        svc, _ = _make_service("")
        groups = svc._group_by_stock(records)
        assert groups[0]["entry_time"] == "20260701"

    @pytest.mark.unit
    def test_sell_from_multiple_lots_fifo(self):
        records = [
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 500, "entry_price": 10.0},
            {"stock_code": "600933", "entry_time": "20260705", "entry_shares": 500, "entry_price": 12.0},
            {"stock_code": "600933", "entry_time": "20260710", "entry_shares": -600, "entry_price": 14.0},
        ]
        svc, _ = _make_service("")
        groups = svc._group_by_stock(records)
        assert len(groups) == 1
        g = groups[0]
        assert g["entry_shares"] == 400
        assert g["entry_price"] == 12.0
        assert g["total_realized_pnl"] == pytest.approx((14 - 10) * 500 + (14 - 12) * 100)


class TestGroupByStockWithFees:
    @pytest.mark.unit
    def test_fees_reduce_realized_pnl(self):
        svc, _ = _make_service("")
        svc.buy_fee_rate = 0.02
        svc.sell_fee_rate = 0.02
        records = [
            {"stock_code": "600933", "entry_time": "20260701", "entry_shares": 1000, "entry_price": 10.0},
            {"stock_code": "600933", "entry_time": "20260710", "entry_shares": -500, "entry_price": 12.0},
        ]
        groups = svc._group_by_stock(records)
        assert len(groups) == 1
        g = groups[0]
        assert g["total_realized_pnl"] < 1000.0


class TestFillMissingPrices:
    @pytest.mark.unit
    def test_fills_missing_price_from_close(self):
        rows = [_row("600933", "20260703", "买入", None, 500)]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        assert records[0]["entry_price"] is None

        mock_df = _make_kline_df([15.5])
        svc.data_provider.get_kline.return_value = mock_df
        svc._fill_missing_prices(records)
        assert records[0]["entry_price"] == 15.5

    @pytest.mark.unit
    def test_skips_if_price_already_set(self):
        rows = [_row("600933", "20260703", "买入", 15.41, 500)]
        svc, _ = _make_service(rows)
        records = svc._load_records(svc.config.POOL_FILE_PATH)
        svc._fill_missing_prices(records)
        assert records[0]["entry_price"] == 15.41
        svc.data_provider.get_kline.assert_not_called()


class TestProcessGroup:
    @pytest.mark.unit
    def test_only_buys_pnl_positive(self):
        svc, _ = _make_service("")
        group = {
            "stock_code": "600933", "entry_time": "20260701",
            "entry_shares": 500, "entry_price": 10.0,
            "trade_count": 1, "total_realized_pnl": 0.0, "total_invested": 5000.0,
            "total_fees": 0.0,
        }
        mock_df = _make_kline_df([15.0] * 60)
        svc.data_provider.get_kline.return_value = mock_df

        result = svc._process_group(group)
        assert result is not None
        assert result["综合收益率"] == "50.00%"
        assert result["持仓成本"] == 10.0
        assert result["当前收盘价格"] == 15.0

    @pytest.mark.unit
    def test_only_buys_pnl_negative(self):
        svc, _ = _make_service("")
        group = {
            "stock_code": "600933", "entry_time": "20260701",
            "entry_shares": 500, "entry_price": 20.0,
            "trade_count": 1, "total_realized_pnl": 0.0, "total_invested": 10000.0,
            "total_fees": 0.0,
        }
        mock_df = _make_kline_df([15.0] * 60)
        svc.data_provider.get_kline.return_value = mock_df

        result = svc._process_group(group)
        assert result["综合收益率"] == "-25.00%"

    @pytest.mark.unit
    def test_partial_sell_overall_pnl(self):
        svc, _ = _make_service("")
        group = {
            "stock_code": "600933", "entry_time": "20260701",
            "entry_shares": 700, "entry_price": 10.0,
            "trade_count": 2, "total_realized_pnl": 600.0, "total_invested": 10000.0,
            "total_fees": 0.0,
        }
        mock_df = _make_kline_df([12.0] * 60)
        svc.data_provider.get_kline.return_value = mock_df

        result = svc._process_group(group)
        assert result["综合收益率"] == "20.00%"

    @pytest.mark.unit
    def test_holding_days_calculated(self):
        svc, _ = _make_service("")
        group = {
            "stock_code": "600933", "entry_time": "20260601",
            "entry_shares": 500, "entry_price": 10.0,
            "trade_count": 1, "total_realized_pnl": 0.0, "total_invested": 5000.0,
            "total_fees": 0.0,
        }
        mock_df = _make_kline_df([11.0] * 60)
        svc.data_provider.get_kline.return_value = mock_df

        result = svc._process_group(group)
        assert result["持有天数"] == 32  # 2026-07-03 - 2026-06-01

    @pytest.mark.unit
    def test_columns_output(self):
        svc, _ = _make_service("")
        group = {
            "stock_code": "600933", "entry_time": "20260701",
            "entry_shares": 500, "entry_price": 10.0,
            "trade_count": 1, "total_realized_pnl": 100.0, "total_invested": 5000.0,
            "total_fees": 5.0,
        }
        mock_df = _make_kline_df([12.0] * 60)
        svc.data_provider.get_kline.return_value = mock_df
        result = svc._process_group(group)

        for col in COLUMNS:
            assert col in result, f"缺少列: {col}"


class TestFullFlow:
    @pytest.mark.unit
    def test_only_buys(self):
        rows = [_row("600933", "20260703", "买入", 15.41, 500),
                _row("600933", "20260703", "买入", 15.43, 1200),
                _row("600933", "20260703", "买入", 15.39, 700)]
        svc, _ = _make_service(rows)
        mock_df = _make_kline_df([15.5] * 60)
        svc.data_provider.get_kline.return_value = mock_df

        df = svc.run()
        assert not df.empty
        assert len(df) == 2  # 1 stock + 1 summary
        row = df.iloc[0]
        assert row["股票代码"] == "600933"
        assert row["入仓股数"] == 2400

    @pytest.mark.unit
    def test_buy_sell_mixed(self):
        rows = [_row("600933", "20260701", "买入", 10.0, 1000),
                _row("600933", "20260710", "卖出", 12.0, 300)]
        svc, _ = _make_service(rows)
        mock_kline = _make_kline_df([11.0] * 60)
        mock_entry = _make_kline_df([10.0])

        def get_kline_side_effect(symbols, start_date=None, end_date=None, **kwargs):
            if start_date and end_date and start_date == end_date:
                return mock_entry
            return mock_kline

        svc.data_provider.get_kline.side_effect = get_kline_side_effect

        df = svc.run()
        assert not df.empty
        row = df.iloc[0]
        assert row["入仓股数"] == 700
        assert row["持仓成本"] == 10.0
        assert row["操盘次数"] == 2

    @pytest.mark.unit
    def test_file_not_found_returns_empty(self):
        svc, _ = _make_service()
        os.remove(svc.config.POOL_FILE_PATH)
        assert svc.run().empty

    @pytest.mark.unit
    def test_all_invalid_graceful(self):
        rows = [_row("xxxxx", "20260703", "买入", 15.0, 500)]
        svc, _ = _make_service(rows)
        assert svc.run().empty

    @pytest.mark.unit
    def test_buy_and_sell_full_flow(self):
        rows = [_row("600933", "20260701", "买入", 10.0, 1000),
                _row("600933", "20260710", "卖出", 12.0, 300)]
        svc, _ = _make_service(rows)
        mock_kline = _make_kline_df([11.0] * 60)
        mock_entry = _make_kline_df([10.0])

        def get_kline_side_effect(symbols, start_date=None, end_date=None, **kwargs):
            if start_date and end_date and start_date == end_date:
                return mock_entry
            return mock_kline

        svc.data_provider.get_kline.side_effect = get_kline_side_effect

        df = svc.run()
        assert not df.empty

    @pytest.mark.unit
    def test_summary_row_present(self):
        rows = [_row("600933", "20260701", "买入", 10.0, 1000),
                _row("000001", "20260601", "买入", 20.0, 500)]
        svc, _ = _make_service(rows)
        mock_kline = _make_kline_df([11.0] * 60)

        def get_kline_side_effect(symbols, start_date=None, end_date=None, **kwargs):
            return mock_kline

        svc.data_provider.get_kline.side_effect = get_kline_side_effect

        df = svc.run()
        assert not df.empty
        assert df.iloc[-1]["股票代码"] == "组合汇总"
        assert "总交易费用" in df.columns

    @pytest.mark.unit
    def test_column_order_matches_colums(self):
        rows = [_row("600933", "20260703", "买入", 15.0, 500)]
        svc, _ = _make_service(rows)
        mock_df = _make_kline_df([15.5] * 60)
        svc.data_provider.get_kline.return_value = mock_df
        df = svc.run()
        assert list(df.columns[:len(COLUMNS)]) == COLUMNS
