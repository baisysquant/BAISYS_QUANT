from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from DataCollection.CalendarManager import TradingCalendarAnalyzer
from UtilsManager.CodeNormalizer import CodeNormalizer
from UtilsManager.IDataProvider import IDataProvider

COLUMNS = [
    "股票代码",
    "股票名称",
    "所属行业",
    "入仓时间",
    "持仓成本",
    "入仓股数",
    "操盘次数",
    "持有天数",
    "当前收盘价格",
    "已实现盈亏",
    "未实现盈亏",
    "总交易费用",
    "综合收益率",
    "当前T1目标价",
    "当前T2目标价",
    "偏离幅度",
]

SHEET_NAME = "跟仓回测"


class PositionTrackingService:
    def __init__(
        self,
        config: Any,
        logger: Any,
        data_provider: IDataProvider,
        calendar_mgr: TradingCalendarAnalyzer,
        db_engine: Any = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.data_provider = data_provider
        self.calendar_mgr = calendar_mgr
        self.db_engine = db_engine

        tc = getattr(self.config, "TRADING_COST_PARAMS", {})
        self.buy_fee_rate = tc.get("commission_rate", 0.0003) + tc.get("transfer_fee_rate", 0.00001)
        self.sell_fee_rate = tc.get("commission_rate", 0.0003) + tc.get("stamp_tax_rate", 0.0005) + tc.get("transfer_fee_rate", 0.00001)

    # ── 主入口 ────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        pool_path = self.config.POOL_FILE_PATH
        if not pool_path:
            self.logger.info("[跟仓回测] 未配置 pool_file_path（证券交割单路径），跳过")
            return pd.DataFrame()

        raw_records = self._load_records(pool_path)
        if not raw_records:
            return pd.DataFrame()

        overall_realized_pnl = 0.0
        overall_invested = 0.0
        overall_unrealized_pnl = 0.0
        overall_fees = 0.0

        self._fill_missing_prices(raw_records)

        try:
            groups = self._group_by_stock(raw_records)
        except Exception as e:
            self.logger.error(f"[跟仓回测] 股票分组聚合异常: {e}")
            return pd.DataFrame()
        results = [self._safe_process(g) for g in groups]
        results = [r for r in results if r is not None]

        if not results:
            self.logger.warning("[跟仓回测] 所有记录处理失败，无输出")
            return pd.DataFrame()

        for r in results:
            overall_realized_pnl += r.get("_realized_pnl", 0.0)
            overall_invested += r.get("_invested", 0.0)
            overall_unrealized_pnl += r.get("_unrealized_pnl", 0.0)
            overall_fees += r.get("_total_fees", 0.0)

        df = pd.DataFrame(results)
        df = df[COLUMNS]

        df = self._append_summary(df, overall_realized_pnl, overall_invested, overall_unrealized_pnl, overall_fees)

        self.logger.info(f"[跟仓回测] 成功处理 {len(results)} 只股票")
        return df

    # ── 文件解析 ──────────────────────────────────────────────

    def _load_records(self, xlsx_path: str) -> list[dict]:
        if not os.path.isabs(xlsx_path):
            xlsx_path = os.path.join(os.path.dirname(self.config.config_file), xlsx_path)

        if not os.path.exists(xlsx_path):
            self.logger.warning(f"[跟仓回测] 交割单文件不存在: {xlsx_path}，跳过")
            return []

        try:
            df = pd.read_excel(
                xlsx_path,
                sheet_name="交割单",
                dtype={"成交日期": str, "证券代码": str},
            )
        except Exception as e:
            self.logger.error(f"[跟仓回测] 读取交割单失败: {e}")
            return []

        required_cols = {"成交日期", "证券代码", "买卖标志", "成交价格", "成交数量"}
        missing = required_cols - set(df.columns)
        if missing:
            self.logger.error(f"[跟仓回测] 交割单缺少列: {missing}")
            return []

        if df.empty:
            self.logger.warning("[跟仓回测] 交割单无数据行")
            return []

        valid = []
        for idx, row in df.iterrows():
            code = str(row["证券代码"]).strip().zfill(6)
            if not code.isdigit() or len(code) != 6:
                self.logger.error(f"[跟仓回测] 第 {idx + 2} 行证券代码无效: {row['证券代码']}，已跳过")
                continue

            entry_time = str(row["成交日期"]).strip()
            if not (entry_time.isdigit() and len(entry_time) == 8):
                self.logger.error(f"[跟仓回测] 第 {idx + 2} 行成交日期格式错误: {entry_time}，已跳过")
                continue

            direction = str(row["买卖标志"]).strip()
            if direction not in ("买入", "卖出"):
                self.logger.error(f"[跟仓回测] 第 {idx + 2} 行买卖标志无效: {direction}，已跳过")
                continue

            try:
                raw_qty = int(row["成交数量"])
            except (ValueError, TypeError):
                self.logger.error(f"[跟仓回测] 第 {idx + 2} 行成交数量无效: {row['成交数量']}，已跳过")
                continue
            if raw_qty <= 0:
                self.logger.error(f"[跟仓回测] 第 {idx + 2} 行成交数量必须为正数: {raw_qty}，已跳过")
                continue

            shares = raw_qty if direction == "买入" else -raw_qty
            price_val = row.get("成交价格")
            price = None
            if pd.notna(price_val):
                try:
                    price = float(price_val)
                except (ValueError, TypeError):
                    pass
                if price is not None and price <= 0:
                    self.logger.error(f"[跟仓回测] 第 {idx + 2} 行成交价格必须为正数: {price}，已跳过")
                    continue

            valid.append({
                "stock_code": code,
                "entry_time": entry_time,
                "entry_shares": shares,
                "entry_price": price,
            })

        self.logger.info(f"[跟仓回测] 加载 {len(valid)} 条有效交易记录")
        return valid

    # ── 补全缺失价格 ──────────────────────────────────────────

    def _fill_missing_prices(self, records: list[dict]) -> None:
        need_fetch = [(i, r) for i, r in enumerate(records) if r["entry_price"] is None]
        if not need_fetch:
            return

        for idx, rec in need_fetch:
            try:
                prefixed = CodeNormalizer.add_market_prefix(rec["stock_code"])
                trade_date = self._adjust_to_trading_day(rec["entry_time"])
                close_price = self._get_close_on_date(prefixed, trade_date)
                if close_price is None:
                    self.logger.warning(
                        f"[跟仓回测] {rec['stock_code']} 无 {trade_date} 收盘价，"
                        f"该笔记录将用当前价近似"
                    )
                    close_price = self._get_current_close(prefixed)
                if close_price is not None:
                    records[idx]["entry_price"] = close_price
                else:
                    self.logger.error(
                        f"[跟仓回测] {rec['stock_code']} 无任何价格数据，该笔记录将跳过"
                    )
            except Exception as e:
                self.logger.error(
                    f"[跟仓回测] {rec['stock_code']} 获取价格异常: {e}，该笔记录将跳过"
                )

    def _adjust_to_trading_day(self, entry_time: str) -> str:
        dt = datetime.strptime(entry_time, "%Y%m%d")
        return self.calendar_mgr.get_last_trading_day(input_date=dt)

    def _get_close_on_date(self, prefixed: str, date_str: str) -> float | None:
        df = self.data_provider.get_kline(
            symbols=[prefixed],
            start_date=date_str,
            end_date=date_str,
        )
        if df.empty:
            return None
        return float(df.iloc[0]["close"])

    def _get_current_close(self, prefixed: str) -> float | None:
        df = self.data_provider.get_kline(symbols=[prefixed])
        if df.empty:
            return None
        return float(df.iloc[-1]["close"])

    # ── 按股票分组 + FIFO 逐笔盈亏计算（含交易成本） ──────────

    def _group_by_stock(self, records: list[dict]) -> list[dict]:
        groups = defaultdict(list)
        for rec in records:
            if rec["entry_price"] is None:
                continue
            groups[rec["stock_code"]].append(rec)

        result = []
        for code, recs in groups.items():
            recs_sorted = sorted(recs, key=lambda r: r["entry_time"])
            earliest_time = recs_sorted[0]["entry_time"]

            lots = []
            total_realized_pnl = 0.0
            total_invested = 0.0
            total_fees = 0.0
            trade_count = 0

            for r in recs_sorted:
                qty = r["entry_shares"]
                price = r["entry_price"]
                trade_count += 1

                if qty > 0:
                    lot_cost = qty * price
                    buy_fees = lot_cost * self.buy_fee_rate
                    total_purchase = lot_cost + buy_fees
                    lots.append({
                        "shares": qty,
                        "price": price,
                        "total_purchase_cost": total_purchase,
                    })
                    total_invested += total_purchase
                    total_fees += buy_fees
                else:
                    sell_qty = abs(qty)
                    sell_proceeds = sell_qty * price
                    sell_fees = sell_proceeds * self.sell_fee_rate
                    net_proceeds = sell_proceeds - sell_fees
                    total_fees += sell_fees
                    remaining_sell = sell_qty
                    oversold = False

                    for lot in lots:
                        if remaining_sell <= 0:
                            break
                        matched = min(lot["shares"], remaining_sell)
                        ratio = matched / lot["shares"]
                        matched_purchase_cost = lot["total_purchase_cost"] * ratio

                        total_realized_pnl += net_proceeds * (matched / sell_qty) - matched_purchase_cost

                        lot["shares"] -= matched
                        lot["total_purchase_cost"] -= matched_purchase_cost
                        remaining_sell -= matched

                    if remaining_sell > 0:
                        oversold = True

                    lots = [lot for lot in lots if lot["shares"] > 0]

                    if oversold:
                        self.logger.warning(
                            f"[跟仓回测] {code} 卖出数量超过总买入量，已清仓跳过"
                        )
                        total_realized_pnl = 0.0
                        total_invested = 0.0
                        total_fees = 0.0
                        lots = []
                        break

            remaining_shares = sum(lot["shares"] for lot in lots)
            if remaining_shares == 0:
                self.logger.info(f"[跟仓回测] {code} 已清仓，跳过")
                continue

            remaining_cost_basis = sum(lot["price"] * lot["shares"] for lot in lots)
            avg_cost = remaining_cost_basis / remaining_shares

            result.append({
                "stock_code": code,
                "entry_time": earliest_time,
                "entry_shares": remaining_shares,
                "entry_price": round(avg_cost, 2),
                "trade_count": trade_count,
                "total_realized_pnl": total_realized_pnl,
                "total_invested": total_invested,
                "total_fees": total_fees,
            })

        if not result:
            self.logger.warning("[跟仓回测] 分组后无持仓记录")
        else:
            self.logger.info(f"[跟仓回测] 聚合为 {len(result)} 只持仓股票")
        return result

    # ── 单股票盈亏计算 ────────────────────────────────────────

    def _safe_process(self, group: dict) -> dict | None:
        try:
            return self._process_group(group)
        except Exception as e:
            self.logger.error(f"[跟仓回测] 处理 {group.get('stock_code', '?')} 失败: {e}")
            return None

    def _process_group(self, group: dict) -> dict | None:
        code = group["stock_code"]
        prefixed = CodeNormalizer.add_market_prefix(code)

        stock_name, industry = self._get_stock_info(code, prefixed)

        current_close = self._get_current_close(prefixed)
        if current_close is None:
            self.logger.warning(f"[跟仓回测] {code} 无当前收盘价，跳过")
            return None

        remaining_shares = group["entry_shares"]
        avg_cost = group["entry_price"]
        unrealized_pnl = (current_close - avg_cost) * remaining_shares
        realized_pnl = group.get("total_realized_pnl", 0.0)
        total_invested = group.get("total_invested", 0.0)
        total_fees = group.get("total_fees", 0.0)

        total_pnl = realized_pnl + unrealized_pnl
        pnl_pct = total_pnl / total_invested * 100 if total_invested > 0 else 0.0

        today_str = self.calendar_mgr.get_last_trading_day()
        today_dt = datetime.strptime(today_str, "%Y-%m-%d")
        entry_dt = datetime.strptime(group["entry_time"], "%Y%m%d")
        holding_days = (today_dt - entry_dt).days

        t1, t2 = self._calc_targets(prefixed, current_close)

        deviation = ""
        if t1 is not None and t1 > 0:
            dev_pct = (current_close - t1) / t1 * 100
            deviation = f"{dev_pct:.1f}%"

        return {
            "股票代码": code,
            "股票名称": stock_name,
            "所属行业": industry,
            "入仓时间": group["entry_time"],
            "持仓成本": avg_cost,
            "入仓股数": remaining_shares,
            "操盘次数": group.get("trade_count", 1),
            "持有天数": holding_days,
            "当前收盘价格": round(current_close, 2),
            "已实现盈亏": round(realized_pnl, 2),
            "未实现盈亏": round(unrealized_pnl, 2),
            "总交易费用": round(total_fees, 2),
            "综合收益率": f"{pnl_pct:.2f}%",
            "当前T1目标价": round(t1, 2) if t1 is not None else None,
            "当前T2目标价": round(t2, 2) if t2 is not None else None,
            "偏离幅度": deviation,
            "_realized_pnl": realized_pnl,
            "_unrealized_pnl": unrealized_pnl,
            "_invested": total_invested,
            "_total_fees": total_fees,
        }

    def _get_stock_info(self, code: str, prefixed: str) -> tuple[str, str]:
        if self.db_engine is None:
            return "", ""
        try:
            sql = text(
                "SELECT stock_name, industry_name FROM stock_basic_info_sw "
                "WHERE stock_code = :code ORDER BY record_date DESC LIMIT 1"
            )
            with self.db_engine.connect() as conn:
                row = conn.execute(sql, {"code": prefixed}).fetchone()
                if row:
                    return row[0], row[1] or ""
        except Exception:
            pass
        return "", ""

    def _calc_targets(self, prefixed: str, current_close: float) -> tuple:
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        df = self.data_provider.get_kline(symbols=[prefixed], start_date=start)
        if len(df) < 16:
            return None, None

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        if pd.isna(atr) or atr <= 0:
            return None, None

        params = self.config.SCORING_PARAMS
        t1 = current_close + atr * params["atr_t1_mult"]
        t2 = current_close + atr * params["atr_t2_mult"]
        return t1, t2

    # ── 组合层汇总行 ──────────────────────────────────────────

    def _append_summary(
        self,
        df: pd.DataFrame,
        total_realized_pnl: float,
        total_invested: float,
        total_unrealized_pnl: float,
        total_fees: float,
    ) -> pd.DataFrame:
        total_pnl = total_realized_pnl + total_unrealized_pnl
        total_return = total_pnl / total_invested * 100 if total_invested > 0 else 0.0

        summary = {
            "股票代码": "组合汇总",
            "股票名称": "",
            "所属行业": "",
            "入仓时间": "",
            "持仓成本": "",
            "入仓股数": df["入仓股数"].sum() if "入仓股数" in df.columns else "",
            "操盘次数": df["操盘次数"].sum() if "操盘次数" in df.columns else "",
            "持有天数": "",
            "当前收盘价格": "",
            "已实现盈亏": round(total_realized_pnl, 2),
            "未实现盈亏": round(total_unrealized_pnl, 2),
            "总交易费用": round(total_fees, 2),
            "综合收益率": f"{total_return:.2f}%",
            "当前T1目标价": "",
            "当前T2目标价": "",
            "偏离幅度": "",
        }

        df_result = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        return df_result



