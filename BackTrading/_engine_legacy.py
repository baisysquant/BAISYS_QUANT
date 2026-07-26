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

    symbols = data["symbol"].unique()
    symbols.sort()
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    n_syms = len(symbols)
    pos_value = np.zeros(n_syms, dtype=np.float64)

    if engine_cfg.point_in_time:
        pit = data.groupby("symbol", sort=False)["trade_date"].min().to_dict()
    else:
        pit = None

    cm = engine_cfg.cost_model
    _adv_state: dict[str, tuple[float, int]] = {}  # symbol -> (expanding_mean, count)
    _prev_bar: dict[str, tuple[float, float]] = {}  # symbol -> (prev_close, prev_atr)
    _buy_date: dict[str, str] = {}  # symbol -> 买入日期（T+1 检查）

    use_sw = engine_cfg.portfolio_method == "score_weighted"
    max_pos_pct = engine_cfg.max_position_pct
    _max_holdings = engine_cfg.max_holdings
    _buy_threshold = engine_cfg.buy_threshold
    init_cash = engine_cfg.initial_cash

    risk_key = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "D": 4}
    _none_mult = engine_cfg.risk_none_multiplier
    risk_mult = np.array([_none_mult, 1.5, 3.0, 5.0, 8.0], dtype=np.float64)
    _kelly = engine_cfg.kelly_fraction
    _pos_a = engine_cfg.position_a
    _atr_stop = engine_cfg.atr_stop_mult

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

    def _process_sell(dt, s_syms, s_idx, s_close, s_vol):
        total_sold = 0.0
        for j in range(len(s_syms)):
            pv = float(pos_value[s_idx[j]])
            if pv <= 0:
                continue
            pos_value[s_idx[j]] = 0.0
            proc, cst = _sell_proceeds_and_cost(s_syms[j], pv, float(s_vol[j]))
            nonlocal cash
            cash += proc
            total_sold += pv
            trade_log.append({
                "time": dt, "symbol": s_syms[j], "action": "sell",
                "price": float(s_close[j]), "value": round(proc, 2),
                "cost": round(cst, 2),
            })
        return total_sold

    if use_sw:
        # ── Fast path: no cumulative_hist needed ──
        for dt, grp in date_groups:
            day_data = grp.copy()
            if pit is not None:
                sym_first = day_data["symbol"].astype(str).map(pit).fillna(dt)
                day_data = day_data[sym_first <= dt]
                if day_data.empty:
                    continue

            syms_str = day_data["symbol"].astype(str).values
            idx = np.array([sym_to_idx[s] for s in syms_str], dtype=np.int32)
            close = day_data["close"].values
            volume = day_data["volume"].values
            buy_score = day_data["进场评分"].values
            sell_score = day_data["退出评分"].values
            risk_str = day_data["风险等级"].astype(str).values

            # ── 涨跌停/停牌检查 ──
            _prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close)])
            _limit_ratio = np.array([
                0.20 if s.startswith(("30", "688")) else 0.10
                for s in syms_str
            ], dtype=np.float64)
            limit_up_arr = _prev_close_arr * (1 + _limit_ratio)
            limit_down_arr = _prev_close_arr * (1 - _limit_ratio)
            not_limit_up = close < limit_up_arr
            not_limit_down = close > limit_down_arr
            has_volume = volume > 0

            # ── 止损：信号列与 ATR 独立生效 ──
            stop_col = day_data["止损价"].values if "止损价" in day_data.columns else np.zeros(len(day_data))
            stop_hit_col = (stop_col > 0) & (close < stop_col)
            stop_hit_atr = np.zeros(len(day_data), dtype=bool)
            if _atr_stop > 0 and "ATR" in day_data.columns:
                prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close)])
                prev_atr_arr = np.array([_prev_bar.get(s, (0, a))[1] for s, a in zip(syms_str, day_data["ATR"].values)])
                atr_stop = prev_close_arr - prev_atr_arr * _atr_stop
                stop_hit_atr = (atr_stop > 0) & (close < atr_stop)
            if "ATR" in day_data.columns:
                for i_s, s in enumerate(syms_str):
                    _prev_bar[s] = (float(close[i_s]), float(day_data["ATR"].values[i_s]))
            stop_hit = stop_hit_col | stop_hit_atr

            total_value = cash + float(pos_value.sum())
            daily_sell_value = 0.0

            # ── 卖出（含 T+1 检查） ──
            held = pos_value[idx] > 0
            if held.any():
                _t1_ok = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
                exit_high = np.isin(risk_str, ["HIGH", "D"])
                exit_gt = (sell_score > buy_score) & (sell_score > 0)
                sel = held & (exit_high | exit_gt | stop_hit) & not_limit_down & has_volume & _t1_ok
                si = np.where(sel)[0]
                if len(si):
                    daily_sell_value = _process_sell(dt, syms_str[si], idx[si], close[si], volume[si])

            # ── 买入 ──
            buy_ok = (buy_score >= _buy_threshold) & (pos_value[idx] == 0) & (~np.isin(risk_str, ["HIGH", "D", "E"])) & not_limit_up & has_volume
            bi = np.where(buy_ok)[0]
            daily_buy_value = 0.0
            if len(bi) == 0 and len(date_groups) > 100 and np.any(buy_score >= _buy_threshold):
                _diag_score = int((buy_score >= _buy_threshold).sum())
                _diag_pos = int((pos_value[idx] == 0).sum())
                _diag_risk = int((~np.isin(risk_str, ["HIGH", "D", "E"])).sum())
                _diag_limit = int(not_limit_up.sum())
                _diag_vol = int(has_volume.sum())
                logger.info(f"[ENGINE-DIAG] {dt}: 评分≥{_buy_threshold}={_diag_score} 空仓={_diag_pos} 低风险={_diag_risk} 非涨停={_diag_limit} 有量={_diag_vol} 总={len(buy_ok)}")
            if len(bi):
                b_syms = syms_str[bi]
                b_idx = idx[bi]
                b_close = close[bi]
                b_vol = volume[bi]
                b_risk_str = risk_str[bi]

                risk_int = np.array([risk_key.get(r, 2) for r in b_risk_str], dtype=np.int32)
                raw_w = 1.0 / risk_mult[risk_int] * _kelly
                w = np.power(raw_w, _pos_a)
                order = np.argsort(-w)
                total_w = float(w.sum()) or 1.0

                existing = int((pos_value > 0).sum())
                max_new = max(0, _max_holdings - existing) if _max_holdings > 0 else min(len(order), 50)
                bought = 0
                for j in order:
                    if bought >= max_new:
                        break
                    si = b_idx[j]
                    if pos_value[si] > 0:
                        continue
                    # 用可用现金分配（扣除 0.2% 税费预留），且单票不超过总资产 max_pos_pct
                    tv = min(cash * 0.998 * (float(w[j]) / total_w), total_value * max_pos_pct)
                    price = float(b_close[j])
                    shares = int(tv / price) // 100 * 100 if price > 0 else 0
                    if shares < 100:
                        continue
                    tv = shares * price
                    cst = _buy_cost(b_syms[j], tv, float(b_vol[j]))
                    if cash >= tv + cst:
                        cash -= tv + cst
                        pos_value[si] = tv
                        daily_buy_value += tv
                        _buy_date[b_syms[j]] = str(dt)
                        trade_log.append({
                            "time": dt, "symbol": b_syms[j], "action": "buy",
                            "price": float(b_close[j]), "value": round(tv, 2),
                            "cost": round(cst, 2),
                        })
                        bought += 1

                if bought == 0 and len(date_groups) > 100:
                    _tv0 = min(cash * 0.998 * (float(w[order[0]]) / total_w), total_value * max_pos_pct)
                    _p0 = float(b_close[order[0]])
                    _s0 = int(_tv0 / _p0) // 100 * 100 if _p0 > 0 else 0
                    logger.info(f"[ENGINE-DIAG] {dt}: {len(bi)}候选 0买入  cash={cash:.0f}  tv[0]={_tv0:.0f}  p[0]={_p0:.0f}  s[0]={_s0}  total_w={total_w:.2f}  max_pos_pct={max_pos_pct}")

            # ── ADV 更新（移后：使用今日数据为明日服务，不干扰今日买卖） ──
            for i_sym, i_vol in zip(syms_str, volume):
                _update_adv(i_sym, i_vol)

            total_value = cash + float(pos_value.sum())
            _turnover = (daily_buy_value + daily_sell_value) / (2 * total_value) if total_value > 0 else 0.0
            equity_curve.append({
                "time": dt, "portfolio_value": round(total_value, 2),
                "turnover": round(_turnover, 6),
            })
    else:
        # ── Legacy path: build cumulative_hist for allocate_weights ──
        cumulative_hist: pd.DataFrame | None = None
        for dt, grp in date_groups:
            day_data = grp.copy()
            if pit is not None:
                sym_first = day_data["symbol"].astype(str).map(pit).fillna(dt)
                day_data = day_data[sym_first <= dt]
                if day_data.empty:
                    continue

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
            close = day_data["close"].values
            volume = day_data["volume"].values
            buy_score = day_data["进场评分"].values
            sell_score = day_data["退出评分"].values
            risk_str = day_data["风险等级"].astype(str).values

            # ── 涨跌停/停牌检查 ──
            _prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close)])
            _limit_ratio = np.array([
                0.20 if s.startswith(("30", "688")) else 0.10
                for s in syms_str
            ], dtype=np.float64)
            limit_up_arr = _prev_close_arr * (1 + _limit_ratio)
            limit_down_arr = _prev_close_arr * (1 - _limit_ratio)
            not_limit_up = close < limit_up_arr
            not_limit_down = close > limit_down_arr
            has_volume = volume > 0

            # ── 止损：信号列与 ATR 独立生效 ──
            stop_col = day_data["止损价"].values if "止损价" in day_data.columns else np.zeros(len(day_data))
            stop_hit_col = (stop_col > 0) & (close < stop_col)
            stop_hit_atr = np.zeros(len(day_data), dtype=bool)
            if _atr_stop > 0 and "ATR" in day_data.columns:
                prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close)])
                prev_atr_arr = np.array([_prev_bar.get(s, (0, a))[1] for s, a in zip(syms_str, day_data["ATR"].values)])
                atr_stop = prev_close_arr - prev_atr_arr * _atr_stop
                stop_hit_atr = (atr_stop > 0) & (close < atr_stop)
            if "ATR" in day_data.columns:
                for i_s, s in enumerate(syms_str):
                    _prev_bar[s] = (float(close[i_s]), float(day_data["ATR"].values[i_s]))
            stop_hit = stop_hit_col | stop_hit_atr

            total_value = cash + float(pos_value.sum())
            daily_sell_value = 0.0

            # ── 卖出（含 T+1 检查） ──
            held = pos_value[idx] > 0
            if held.any():
                _t1_ok = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
                exit_high = np.isin(risk_str, ["HIGH", "D"])
                exit_gt = (sell_score > buy_score) & (sell_score > 0)
                sel = held & (exit_high | exit_gt | stop_hit) & not_limit_down & has_volume & _t1_ok
                si = np.where(sel)[0]
                if len(si):
                    daily_sell_value = _process_sell(dt, syms_str[si], idx[si], close[si], volume[si])

            # ── 买入 ──
            buy_ok = (buy_score >= _buy_threshold) & (pos_value[idx] == 0) & (~np.isin(risk_str, ["HIGH", "D", "E"])) & not_limit_up & has_volume
            bi = np.where(buy_ok)[0]
            daily_buy_value = 0.0
            if len(bi):
                b_syms = syms_str[bi]
                b_idx = idx[bi]
                b_close = close[bi]
                b_vol = volume[bi]

                w_dict = allocate_weights(
                    cumulative_hist, method=engine_cfg.portfolio_method,
                    max_weight=max_pos_pct,
                    entry_col="进场评分", risk_col="风险等级",
                )
                w = np.array([w_dict.get(s, 0.0) for s in b_syms], dtype=np.float64)
                order = np.argsort(-w)
                total_w = float(w.sum()) or 1.0

                existing = int((pos_value > 0).sum())
                max_new = max(0, _max_holdings - existing) if _max_holdings > 0 else min(len(order), 50)
                bought = 0
                for j in order:
                    if bought >= max_new:
                        break
                    si = b_idx[j]
                    if pos_value[si] > 0:
                        continue
                    tv = min(cash * 0.998 * (float(w[j]) / total_w), total_value * max_pos_pct)
                    price = float(b_close[j])
                    shares = int(tv / price) // 100 * 100 if price > 0 else 0
                    if shares < 100:
                        continue
                    tv = shares * price
                    cst = _buy_cost(b_syms[j], tv, float(b_vol[j]))
                    if cash >= tv + cst:
                        cash -= tv + cst
                        pos_value[si] = tv
                        daily_buy_value += tv
                        _buy_date[b_syms[j]] = str(dt)
                        trade_log.append({
                            "time": dt, "symbol": b_syms[j], "action": "buy",
                            "price": float(b_close[j]), "value": round(tv, 2),
                            "cost": round(cst, 2),
                        })
                        bought += 1

            # ── ADV 更新（移后：使用今日数据为明日服务） ──
            for i_sym, i_vol in zip(syms_str, volume):
                _update_adv(i_sym, i_vol)

            total_value = cash + float(pos_value.sum())
            _turnover = (daily_buy_value + daily_sell_value) / (2 * total_value) if total_value > 0 else 0.0
            equity_curve.append({
                "time": dt, "portfolio_value": round(total_value, 2),
                "turnover": round(_turnover, 6),
            })

    final_value = cash + float(pos_value.sum())
    return (final_value / init_cash) - 1
