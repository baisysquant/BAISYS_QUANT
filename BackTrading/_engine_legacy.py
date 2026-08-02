from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from typing_extensions import TypeAlias

from BackTrading.engine import EngineConfig
from LogicAnalyzer.portfolio.backtest_weights import allocate_weights

ParamsDict: TypeAlias = dict[str, Any]
TradeLog: TypeAlias = list[dict[str, Any]]
EquityCurve: TypeAlias = list[dict[str, Any]]


# 20 日均量（行业口径：不含当日的滚动均值，避免用当日成交量前视）
_ADV_WINDOW = 20
# 注册制板块（创业板 30x / 科创板 688）新股上市前 5 个交易日无涨跌幅限制
_NEW_LISTING_EXEMPT_DAYS = 5
# 主板新股上市首日涨跌幅限制：44% / -36%（次日起 10%）
_MAIN_BOARD_FIRST_DAY_UP = 0.44
_MAIN_BOARD_FIRST_DAY_DOWN = 0.36
# 北交所新股上市首日无涨跌幅限制
_BSE_FIRST_DAY_EXEMPT = True
# 涨跌停四舍五入到分（分位精度）
_LIMIT_PX_PRECISION = 100.0
# 印花税按交易日分段：2023-08-28 财政部减半（卖出单向）
_STAMP_TAX_RECENT = 0.0005
_STAMP_TAX_OLD = 0.001
# 经手费（双边）+ 证管费（双边），通常合并计入佣金；此处单独建模可选
_HANDLING_FEE_RATE = 0.0000341   # 0.00341%（万 0.341）
_CSRC_FEE_RATE = 0.00002         # 0.002%（万 0.2）


def run_full_backtest(
    data: pd.DataFrame,
    params: dict[str, Any],
    engine_cfg: EngineConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if engine_cfg is None:
        engine_cfg = EngineConfig()
    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(data, params, engine_cfg, tl, ec)
    return tl, ec


def _run_single_backtest(
    data: pd.DataFrame,
    params: ParamsDict,
    engine_cfg: EngineConfig,
    trade_log: TradeLog,
    equity_curve: EquityCurve,
) -> float:
    if pd.api.types.is_datetime64_any_dtype(data["trade_date"]):
        data = data.copy()
        data["trade_date"] = data["trade_date"].dt.strftime("%Y-%m-%d")

    date_groups = list(data.groupby("trade_date", sort=True))

    symbols = sorted(data["symbol"].unique().tolist())
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    n_syms = len(symbols)
    pos_value = np.zeros(n_syms, dtype=np.float64)
    pos_shares = np.zeros(n_syms, dtype=np.int32)

    if engine_cfg.point_in_time:
        pit = data.groupby("symbol", sort=False)["trade_date"].min().to_dict()
    else:
        pit = None

    cm = engine_cfg.cost_model
    _adv_state: dict[str, tuple[float, int]] = {}
    _prev_bar: dict[str, tuple[float, float]] = {}
    _buy_date: dict[str, str] = {}
    _sold_today: set[str] = set()

    use_sw = engine_cfg.portfolio_method == "score_weighted"
    max_pos_pct = engine_cfg.max_position_pct
    _max_holdings = engine_cfg.max_holdings
    _buy_threshold = engine_cfg.buy_threshold
    init_cash = engine_cfg.initial_cash

    _none_mult = engine_cfg.risk_none_multiplier
    _kelly = engine_cfg.kelly_fraction
    _pos_a = engine_cfg.position_a
    _atr_stop = engine_cfg.atr_stop_mult
    _max_order_pct = 0.1
    _top_k = 20  # 最大候选买入数（集中资金，避免每只分到极小额度 < 100 股）

    cash = float(init_cash)

    def _update_adv(sym: str, vol: float) -> float:
        if sym not in _adv_state:
            _adv_state[sym] = (vol, 1)
            return vol
        m, n = _adv_state[sym]
        n += 1
        new_m = m + (vol - m) / n
        _adv_state[sym] = (new_m, n)
        return new_m

    def _calc_market_value() -> float:
        held = np.where(pos_shares > 0)[0]
        mtm = 0.0
        for si in held:
            s = symbols[si]
            if s in close_lookup:
                mtm += pos_shares[si] * close_lookup[s]
            else:
                mtm += pos_value[si]
        return mtm

    close_lookup: dict[str, float] = {}

    def _sell_proceeds_and_cost(sym: str, value: float, volume: float) -> tuple[float, float]:
        if cm is not None:
            adv = _adv_state.get(sym, (0.0, 0))[0]
            slip = cm.calc_slippage(volume, adv, side="sell", order_type="market")
            rate = slip + cm.stamp_tax_rate
            return value * (1 - rate), value * rate
        commission = max(value * engine_cfg.commission_rate, engine_cfg.min_commission_per_trade)
        fee = commission + value * (engine_cfg.transfer_fee_rate + engine_cfg.stamp_tax_rate)
        slip_cost = value * engine_cfg.slippage
        total_cost = fee + slip_cost
        return value - total_cost, total_cost

    def _buy_cost(sym: str, value: float, volume: float) -> float:
        if cm is not None:
            adv = _adv_state.get(sym, (0.0, 0))[0]
            return cm.buy_cost(value, volume, adv)
        commission = max(value * engine_cfg.commission_rate, engine_cfg.min_commission_per_trade)
        fee = commission + value * engine_cfg.transfer_fee_rate
        return fee + value * engine_cfg.slippage

    def _process_sell(dt, s_syms, s_idx, s_close, s_vol, partial: bool = False):
        total_sold = 0.0
        for j in range(len(s_syms)):
            si = s_idx[j]
            sh = int(pos_shares[si])
            if sh <= 0:
                continue
            if partial:
                # 申报数量单位：科创板 688 / 创业板 30x（2023-08-28 起）为 200 股，北交所 8xx/92x 为 100 股，主板 100 股
                lot = 200 if s_syms[j].startswith(("688", "30")) else 100
                sell_shares = max(lot, sh // 2) // lot * lot
                if sell_shares >= sh:
                    sell_shares = sh
            else:
                sell_shares = sh
            mv = sell_shares * float(s_close[j])
            pos_shares[si] -= sell_shares
            if pos_shares[si] <= 0:
                pos_value[si] = 0.0
                pos_shares[si] = 0
                _sold_today.add(s_syms[j])
            proc, cst = _sell_proceeds_and_cost(s_syms[j], mv, float(s_vol[j]))
            nonlocal cash
            cash += proc
            total_sold += mv
            trade_log.append({
                "time": dt, "symbol": s_syms[j], "action": "sell" if sell_shares >= sh else "sell_partial",
                "price": float(s_close[j]), "value": round(proc, 2),
                "cost": round(cst, 2),
            })
        return total_sold

    _market_multiplier = 1.0
    _prev_med_score = 0.0  # 上一日中位数评分，用于避免前视偏差

    if use_sw:
        for i_day, (dt, grp) in enumerate(date_groups):
            _sold_today.clear()
            day_data = grp.copy()

            # 市场状态过滤：根据上一日全部股票的中位数评分调整仓位（避免前视偏差）
            if i_day == 0:
                _med_score = 0.0
            else:
                _med_score = _prev_med_score
            if _med_score >= 30:
                _market_multiplier = 1.0
            elif _med_score >= 15:
                _market_multiplier = 0.5
            else:
                _market_multiplier = 0.25
            if pit is not None:
                sym_first = day_data["symbol"].astype(str).map(pit).fillna(dt)
                day_data = day_data[sym_first <= dt]
                if day_data.empty:
                    continue

            # 记录本日中位数评分，供下一日使用（避免前视偏差）
            _day_scores = day_data["进场评分"].values
            _prev_med_score = float(np.median(_day_scores[_day_scores > 0])) if (_day_scores > 0).any() else 0

            syms_str = day_data["symbol"].astype(str).values
            idx = np.array([sym_to_idx[s] for s in syms_str], dtype=np.int32)
            # 不复权价：用于涨跌停判定、止损价比较
            close_raw = day_data["close"].values
            # 复权价：统一用于买入/卖出成交价、市值计算、收益计算
            close_adj = day_data["close_adj"].values if "close_adj" in day_data.columns else close_raw
            # 复权价合法性：负值/NaN 说明上游数据异常（如 sh600076 2024-06-24 负后复权价），
            # 该标的当日禁止买入/卖出/估值，避免负市值污染净资产
            adj_ok = np.isfinite(close_adj) & (close_adj > 0)
            close = day_data["close"].values
            volume = day_data["volume"].values
            buy_score = day_data["进场评分"].values
            sell_score = day_data["退出评分"].values
            risk_str = day_data["风险等级"].astype(str).values

            if i_day % 20 == 0:
                _bs_nonzero = buy_score[buy_score > 0]
                if len(_bs_nonzero) > 0:
                    logger.info(f"[ENGINE-SCORE] {dt}: 进场评分 非零={len(_bs_nonzero)}/{len(buy_score)} mean={_bs_nonzero.mean():.1f} median={float(np.median(_bs_nonzero)):.1f} min={_bs_nonzero.min():.0f} max={_bs_nonzero.max():.0f} >=15={int((buy_score>=15).sum())} >=60={int((buy_score>=60).sum())}")
                else:
                    logger.info(f"[ENGINE-SCORE] {dt}: 进场评分 全为零 ({len(buy_score)} 只)")

            # ── 涨跌停/停牌检查（首日无 prev_bar 时跳过 limit 过滤） ──
            _limit_ratio = np.array([
                0.20 if s.startswith(("30", "688")) else 0.30 if s.startswith("8") else 0.10
                for s in syms_str
            ], dtype=np.float64)
            _have_prev = i_day > 0
            if _have_prev:
                _prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close_raw)])
                limit_up_arr = _prev_close_arr * (1 + _limit_ratio)
                limit_down_arr = _prev_close_arr * (1 - _limit_ratio)
                not_limit_up = close < limit_up_arr
                not_limit_down = close > limit_down_arr
            else:
                not_limit_up = np.ones(len(close_raw), dtype=bool)
                not_limit_down = np.ones(len(close_raw), dtype=bool)
            has_volume = volume > 0

            stop_col = day_data["止损价"].values if "止损价" in day_data.columns else np.zeros(len(day_data))
            stop_hit_col = (stop_col > 0) & (close_raw < stop_col)
            stop_hit_atr = np.zeros(len(day_data), dtype=bool)
            if _atr_stop > 0 and "ATR" in day_data.columns and _have_prev:
                prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close_raw)])
                prev_atr_arr = np.array([_prev_bar.get(s, (0, a))[1] for s, a in zip(syms_str, day_data["ATR"].values)])
                atr_stop = prev_close_arr - prev_atr_arr * _atr_stop
                stop_hit_atr = (atr_stop > 0) & (close_raw < atr_stop)
            if "ATR" in day_data.columns:
                for i_s, s in enumerate(syms_str):
                    _prev_bar[s] = (float(close_raw[i_s]), float(day_data["ATR"].values[i_s]))
            stop_hit = stop_hit_col | stop_hit_atr

            close_lookup = dict(zip(syms_str, close_adj))
            total_value = cash + _calc_market_value()
            daily_sell_value = 0.0

            # ── 卖出（含 T+1 检查 + 分批止盈止损） ──
            held = pos_shares[idx] > 0
            if held.any():
                _t1_ok = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
                exit_high = np.isin(risk_str, ["HIGH", "D"])
                exit_gt = (sell_score > buy_score + 20) & (sell_score > 0)
                exit_score_low = (buy_score > 0) & (buy_score < _buy_threshold // 3)
                sel_all = held & (exit_high | exit_gt | exit_score_low | stop_hit) & not_limit_down & has_volume & _t1_ok & adj_ok
                si_all = np.where(sel_all)[0]
                if len(si_all):
                    sel_stop = held & stop_hit & not_limit_down & has_volume & _t1_ok & adj_ok
                    si_stop = np.where(sel_stop)[0]
                    si_partial = np.setdiff1d(si_all, si_stop)
                    if len(si_stop):
                        daily_sell_value += _process_sell(dt, syms_str[si_stop], idx[si_stop], close_adj[si_stop], volume[si_stop], partial=False)
                    if len(si_partial):
                        daily_sell_value += _process_sell(dt, syms_str[si_partial], idx[si_partial], close_adj[si_partial], volume[si_partial], partial=True)
                    close_lookup = dict(zip(syms_str[adj_ok], close_adj[adj_ok]))
                    total_value = cash + _calc_market_value()

            # ── 买入 ──
            # 动态阈值：仅当非零评分足够多(>10只)时使用百分位，否则用固定阈值
            _non_zero = buy_score[buy_score > 0]
            if len(_non_zero) > 10 and buy_score.max() > _buy_threshold:
                _pct_70 = float(np.percentile(_non_zero, 70))
                _effective_threshold = max(_buy_threshold, _pct_70)
            else:
                _effective_threshold = _buy_threshold
            _not_sold_today = np.array([s not in _sold_today for s in syms_str])
            buy_ok = (buy_score >= _effective_threshold) & (pos_shares[idx] == 0) & (~np.isin(risk_str, ["HIGH", "D", "E"])) & not_limit_up & has_volume & _not_sold_today & adj_ok
            bi = np.where(buy_ok)[0]
            daily_buy_value = 0.0
            if len(bi) == 0 and len(date_groups) > 100 and np.any(buy_score >= _buy_threshold):
                _diag_score = int((buy_score >= _effective_threshold).sum())
                _diag_pos = int((pos_shares[idx] == 0).sum())
                _diag_risk = int((~np.isin(risk_str, ["HIGH", "D", "E"])).sum())
                _diag_limit = int(not_limit_up.sum())
                _diag_vol = int(has_volume.sum())
                _diag_t1 = len(_sold_today)
                logger.info(f"[ENGINE-DIAG] {dt}: 评分≥{_effective_threshold}={_diag_score} 空仓={_diag_pos} 低风险={_diag_risk} 非涨停={_diag_limit} 有量={_diag_vol} T+1禁={_diag_t1} 总={len(buy_ok)}")
            if len(bi):
                b_syms = syms_str[bi]
                b_idx = idx[bi]
                b_close = close_adj[bi]
                b_vol = volume[bi]
                b_risk_str = risk_str[bi]

                # Top-K 等权分配：集中资金到评分最高的 _top_k 只
                if len(bi) > _top_k:
                    b_scores = buy_score[bi]
                    _top_indices = np.argpartition(-b_scores, _top_k)[:_top_k]
                    b_syms = b_syms[_top_indices]
                    b_idx = b_idx[_top_indices]
                    b_close = b_close[_top_indices]
                    b_vol = b_vol[_top_indices]
                    b_risk_str = b_risk_str[_top_indices]
                n_candidates = len(b_syms)
                equal_weight = 1.0 / n_candidates if n_candidates > 0 else 0.0

                existing = int((pos_shares > 0).sum())
                max_new = max(0, _max_holdings - existing) if _max_holdings > 0 else _top_k
                bought = 0
                for j in range(n_candidates):
                    if bought >= max_new:
                        break
                    si = b_idx[j]
                    if pos_shares[si] > 0:
                        continue
                    price = float(b_close[j])
                    if not np.isfinite(price) or price <= 0:
                        continue
                    # 申报数量单位：科创板 688 / 创业板 30x（2023-08-28 起）为 200 股，北交所 8xx/92x 为 100 股，主板 100 股
                    lot = 200 if b_syms[j].startswith(("688", "30")) else 100
                    tv = min(cash * equal_weight, total_value * max_pos_pct * _market_multiplier)
                    shares = int(tv / price) // lot * lot if price > 0 else 0
                    if shares < lot:
                        continue
                    _adv_val = _adv_state.get(b_syms[j], (1e9, 0))[0]
                    if _adv_val > 100:
                        max_shares_vol = int(_adv_val * _max_order_pct) // lot * lot
                        shares = min(shares, max_shares_vol)
                        if shares < lot:
                            continue
                    tv = shares * price
                    cst = _buy_cost(b_syms[j], tv, float(b_vol[j]))
                    if cash >= tv + cst:
                        cash -= tv + cst
                        pos_value[si] = tv
                        pos_shares[si] = shares
                        daily_buy_value += tv
                        _buy_date[b_syms[j]] = str(dt)
                        trade_log.append({
                            "time": dt, "symbol": b_syms[j], "action": "buy",
                            "price": float(b_close[j]), "value": round(tv, 2),
                            "cost": round(cst, 2),
                        })
                        bought += 1

                if bought == 0 and len(date_groups) > 100:
                    _p0 = float(b_close[0]) if n_candidates > 0 else 0
                    _tv0 = min(cash * equal_weight, total_value * max_pos_pct * _market_multiplier) if n_candidates > 0 else 0
                    _s0 = int(_tv0 / _p0) // 100 * 100 if _p0 > 0 else 0
                    logger.info(f"[ENGINE-DIAG] {dt}: {len(bi)}候选→{n_candidates}TopK 0买入  cash={cash:.0f}  tv[0]={_tv0:.0f}  p[0]={_p0:.0f}  s[0]={_s0}  eq_w={equal_weight:.4f}  max_pos_pct={max_pos_pct}")

            for i_sym, i_vol in zip(syms_str, volume):
                _update_adv(i_sym, i_vol)

            total_value = cash + _calc_market_value()
            _turnover = (daily_buy_value + daily_sell_value) / (2 * total_value) if total_value > 0 else 0.0
            equity_curve.append({
                "time": dt, "portfolio_value": round(total_value, 2),
                "turnover": round(_turnover, 6),
            })
    else:
        # ── Legacy path: build cumulative_hist for allocate_weights ──
        cumulative_hist: pd.DataFrame | None = None
        for i_day, (dt, grp) in enumerate(date_groups):
            _sold_today.clear()
            day_data = grp.copy()
            if pit is not None:
                sym_first = day_data["symbol"].astype(str).map(pit).fillna(dt)
                day_data = day_data[sym_first <= dt]
                if day_data.empty:
                    continue

            _day_scores = day_data["进场评分"].values
            # 使用上一日中位数评分（避免前视偏差）
            _med_score = _prev_med_score
            if _med_score >= 30:
                _market_multiplier = 1.0
            elif _med_score >= 15:
                _market_multiplier = 0.5
            else:
                _market_multiplier = 0.25

            # 记录本日中位数评分，供下一日使用（避免前视偏差）
            _prev_med_score = float(np.median(_day_scores[_day_scores > 0])) if (_day_scores > 0).any() else 0

            if cumulative_hist is None:
                cumulative_hist = day_data.copy()
            else:
                cumulative_hist = pd.concat([cumulative_hist, day_data], ignore_index=True)
            if len(cumulative_hist) > 100_000:
                _cut = cumulative_hist["trade_date"].unique()
                if len(_cut) > 252:
                    _keep = sorted(_cut)[-252:]
                    cumulative_hist = cumulative_hist[cumulative_hist["trade_date"].isin(_keep)].copy()

            syms_str = day_data["symbol"].astype(str).values
            idx = np.array([sym_to_idx[s] for s in syms_str], dtype=np.int32)
            # 不复权价：用于涨跌停判定、止损价比较
            close_raw = day_data["close"].values
            # 复权价：统一用于买入/卖出成交价、市值计算、收益计算
            close_adj = day_data["close_adj"].values if "close_adj" in day_data.columns else close_raw
            # 复权价合法性：负值/NaN 说明上游数据异常（如 sh600076 2024-06-24 负后复权价），
            # 该标的当日禁止买入/卖出/估值，避免负市值污染净资产
            adj_ok = np.isfinite(close_adj) & (close_adj > 0)
            close = day_data["close"].values
            volume = day_data["volume"].values
            buy_score = day_data["进场评分"].values
            sell_score = day_data["退出评分"].values
            risk_str = day_data["风险等级"].astype(str).values

            # ── 诊断：每 20 天打印一次 buy_score 分布 ──
            if i_day % 20 == 0:
                _bs_nonzero = buy_score[buy_score > 0]
                if len(_bs_nonzero) > 0:
                    logger.info(f"[ENGINE-SCORE] {dt}: 进场评分 非零={len(_bs_nonzero)}/{len(buy_score)} mean={_bs_nonzero.mean():.1f} median={float(np.median(_bs_nonzero)):.1f} min={_bs_nonzero.min():.0f} max={_bs_nonzero.max():.0f} >=15={int((buy_score>=15).sum())} >=60={int((buy_score>=60).sum())}")
                else:
                    logger.info(f"[ENGINE-SCORE] {dt}: 进场评分 全为零 ({len(buy_score)} 只)")

            # ── 涨跌停/停牌检查（首日无 prev_bar 时跳过 limit 过滤） ──
            _limit_ratio = np.array([
                0.20 if s.startswith(("30", "688")) else 0.30 if s.startswith("8") else 0.10
                for s in syms_str
            ], dtype=np.float64)
            _have_prev = i_day > 0
            if _have_prev:
                _prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close_raw)])
                limit_up_arr = _prev_close_arr * (1 + _limit_ratio)
                limit_down_arr = _prev_close_arr * (1 - _limit_ratio)
                not_limit_up = close < limit_up_arr
                not_limit_down = close > limit_down_arr
            else:
                not_limit_up = np.ones(len(close_raw), dtype=bool)
                not_limit_down = np.ones(len(close_raw), dtype=bool)
            has_volume = volume > 0

            stop_col = day_data["止损价"].values if "止损价" in day_data.columns else np.zeros(len(day_data))
            stop_hit_col = (stop_col > 0) & (close_raw < stop_col)
            stop_hit_atr = np.zeros(len(day_data), dtype=bool)
            if _atr_stop > 0 and "ATR" in day_data.columns and _have_prev:
                prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close_raw)])
                prev_atr_arr = np.array([_prev_bar.get(s, (0, a))[1] for s, a in zip(syms_str, day_data["ATR"].values)])
                atr_stop = prev_close_arr - prev_atr_arr * _atr_stop
                stop_hit_atr = (atr_stop > 0) & (close_raw < atr_stop)
            if "ATR" in day_data.columns:
                for i_s, s in enumerate(syms_str):
                    _prev_bar[s] = (float(close_raw[i_s]), float(day_data["ATR"].values[i_s]))
            stop_hit = stop_hit_col | stop_hit_atr

            close_lookup = dict(zip(syms_str[adj_ok], close_adj[adj_ok]))
            total_value = cash + _calc_market_value()
            daily_sell_value = 0.0

            # ── 卖出（含 T+1 检查） ──
            held = pos_shares[idx] > 0
            if held.any():
                _t1_ok = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
                exit_high = np.isin(risk_str, ["HIGH", "D"])
                exit_gt = (sell_score > buy_score + 20) & (sell_score > 0)
                exit_score_low = (buy_score > 0) & (buy_score < _buy_threshold // 3)
                sel_all = held & (exit_high | exit_gt | exit_score_low | stop_hit) & not_limit_down & has_volume & _t1_ok & adj_ok
                si_all = np.where(sel_all)[0]
                if len(si_all):
                    sel_stop = held & stop_hit & not_limit_down & has_volume & _t1_ok & adj_ok
                    si_stop = np.where(sel_stop)[0]
                    si_partial = np.setdiff1d(si_all, si_stop)
                    if len(si_stop):
                        daily_sell_value += _process_sell(dt, syms_str[si_stop], idx[si_stop], close_adj[si_stop], volume[si_stop], partial=False)
                    if len(si_partial):
                        daily_sell_value += _process_sell(dt, syms_str[si_partial], idx[si_partial], close_adj[si_partial], volume[si_partial], partial=True)
                    close_lookup = dict(zip(syms_str[adj_ok], close_adj[adj_ok]))
                    total_value = cash + _calc_market_value()

            # ── 买入 ──
            _non_zero = buy_score[buy_score > 0]
            if len(_non_zero) > 10 and buy_score.max() > _buy_threshold:
                _pct_70 = float(np.percentile(_non_zero, 70))
                _effective_threshold = max(_buy_threshold, _pct_70)
            else:
                _effective_threshold = max(_buy_threshold, 10)
            _not_sold_today = np.array([s not in _sold_today for s in syms_str])
            buy_ok = (buy_score >= _effective_threshold) & (pos_shares[idx] == 0) & (~np.isin(risk_str, ["HIGH", "D", "E"])) & not_limit_up & has_volume & _not_sold_today & adj_ok
            bi = np.where(buy_ok)[0]
            daily_buy_value = 0.0
            if len(bi):
                b_syms = syms_str[bi]
                b_idx = idx[bi]
                b_close = close_adj[bi]
                b_vol = volume[bi]

                w_dict = allocate_weights(
                    cumulative_hist, method=engine_cfg.portfolio_method,
                    max_weight=max_pos_pct,
                    entry_col="进场评分", risk_col="风险等级",
                    min_entry_score=0,
                )
                w = np.array([w_dict.get(s, 0.0) for s in b_syms], dtype=np.float64)
                total_w = float(w.sum()) or 1.0

                # Top-K 等权分配
                if len(b_syms) > _top_k:
                    b_scores_local = buy_score[bi]
                    _top_indices = np.argpartition(-b_scores_local, _top_k)[:_top_k]
                    b_syms = b_syms[_top_indices]
                    b_idx = b_idx[_top_indices]
                    b_close = b_close[_top_indices]
                    b_vol = b_vol[_top_indices]
                n_candidates = len(b_syms)
                equal_weight = 1.0 / n_candidates if n_candidates > 0 else 0.0

                existing = int((pos_shares > 0).sum())
                max_new = max(0, _max_holdings - existing) if _max_holdings > 0 else _top_k
                bought = 0
                for j in range(n_candidates):
                    if bought >= max_new:
                        break
                    si = b_idx[j]
                    if pos_shares[si] > 0:
                        continue
                    price = float(b_close[j])
                    if not np.isfinite(price) or price <= 0:
                        continue
                    # 申报数量单位：科创板 688 / 创业板 30x（2023-08-28 起）为 200 股，北交所 8xx/92x 为 100 股，主板 100 股
                    lot = 200 if b_syms[j].startswith(("688", "30")) else 100
                    tv = min(cash * equal_weight, total_value * max_pos_pct * _market_multiplier)
                    shares = int(tv / price) // lot * lot if price > 0 else 0
                    if shares < lot:
                        continue
                    _adv_val = _adv_state.get(b_syms[j], (1e9, 0))[0]
                    if _adv_val > 100:
                        max_shares_vol = int(_adv_val * _max_order_pct) // lot * lot
                        shares = min(shares, max_shares_vol)
                        if shares < lot:
                            continue
                    tv = shares * price
                    cst = _buy_cost(b_syms[j], tv, float(b_vol[j]))
                    if cash >= tv + cst:
                        cash -= tv + cst
                        pos_value[si] = tv
                        pos_shares[si] = shares
                        daily_buy_value += tv
                        _buy_date[b_syms[j]] = str(dt)
                        trade_log.append({
                            "time": dt, "symbol": b_syms[j], "action": "buy",
                            "price": float(b_close[j]), "value": round(tv, 2),
                            "cost": round(cst, 2),
                        })
                        bought += 1

            for i_sym, i_vol in zip(syms_str, volume):
                _update_adv(i_sym, i_vol)

            total_value = cash + _calc_market_value()
            _turnover = (daily_buy_value + daily_sell_value) / (2 * total_value) if total_value > 0 else 0.0
            equity_curve.append({
                "time": dt, "portfolio_value": round(total_value, 2),
                "turnover": round(_turnover, 6),
            })

    total_value = cash + _calc_market_value()
    return (total_value / init_cash) - 1
