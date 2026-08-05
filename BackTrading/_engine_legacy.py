from __future__ import annotations

from collections import deque
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig

ParamsDict: TypeAlias = dict[str, Any]
TradeLog: TypeAlias = list[dict[str, Any]]
EquityCurve: TypeAlias = list[dict[str, Any]]


# 20 日均量（行业口径：不含当日的滚动均值，避免用当日成交量前视）
_ADV_WINDOW = 20
# 固定滑点强制下限（1.8 交易摩擦合规：A股隐性成本不低于单边 0.05%，
# 配置/回退路径任何低于此值的基础滑点一律抬升，防止 Alpha 虚高）
_MIN_SLIPPAGE_FLOOR = 0.0005
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
_HANDLING_FEE_RATE = 0.0000341  # 0.00341%（万 0.341）
_CSRC_FEE_RATE = 0.00002  # 0.002%（万 0.2）


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

    # ── ST/退市逐日动态剔除（stock_st_history 由 runner 注入 params） ──
    # 业务规则：
    #   - 退市日：无条件禁止买入并强平持仓（退市股无法交易，必须剔除）。
    #     退市为终态：自最早退市日起该股永久禁买，避免"强平→次日复购→再强平"的
    #     循环刷交易（stock_st_history 常为最近快照，K 线可能延伸到标记日之后）
    #   - ST/*ST 日：仅当 _exclude_st=True 时禁止买入并强平（A股 ST 涨跌幅 5%，
    #     部分策略不允许持仓）；exclude_st=False 时 ST 股全程正常参与交易
    # 预构建 {交易日: 被剔除 symbol 集合}，逐日 O(1) 查询；
    # 掩码用 np.isin 按当日行生成，长度与股票池/PIT 过滤后行数无关
    _st_hist = params.get("_st_history") if isinstance(params, dict) else None
    _exclude_st = bool(params.get("_exclude_st", True)) if isinstance(params, dict) else True
    _st_blocked_syms_by_day: dict[str, set[str]] = {}
    if _st_hist:
        _delisted_on: dict[str, str] = {}
        for _s, _recs in _st_hist.items():
            if not _recs:
                continue
            _del_days = [d for d, (_st_f, _dl) in _recs.items() if _dl]
            if _del_days:
                _delisted_on[_s] = min(_del_days)
        if _delisted_on:
            _all_days = [str(dt) for dt, _g in date_groups]
            for _s, _first_del in _delisted_on.items():
                for _d_str in _all_days:
                    if _d_str >= _first_del:
                        _st_blocked_syms_by_day.setdefault(_d_str, set()).add(_s)
        if _exclude_st:
            for _s, _recs in _st_hist.items():
                if not _recs:
                    continue
                for _d_str, (_st_f, _dl) in _recs.items():
                    if _st_f and not _dl:
                        _st_blocked_syms_by_day.setdefault(_d_str, set()).add(_s)

    pit = data.groupby("symbol", sort=False)["trade_date"].min().to_dict() if engine_cfg.point_in_time else None

    cm = engine_cfg.cost_model
    # 滚动 20 日成交量（不含当日，前视合规）：value = (deque, run_sum)
    _adv_state: dict[str, tuple[Any, float]] = {}
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
        """滚动 _ADV_WINDOW 日均量（不含当日）。当日 bar 结束后入账，供次日使用。"""
        dq, run = _adv_state.get(sym, (None, 0.0))
        if dq is None:
            dq = deque(maxlen=_ADV_WINDOW)
        if len(dq) == dq.maxlen:
            run -= dq[0]
        dq.append(vol)
        run += vol
        _adv_state[sym] = (dq, run)
        return run / len(dq)

    def _current_adv(sym: str) -> float:
        """当前可用 ADV（前一日及之前 _ADV_WINDOW 日滚动均值）；无数据返回 0。"""
        dq, run = _adv_state.get(sym, (None, 0.0))
        return (run / len(dq)) if dq else 0.0

    def _stamp_rate(dt: str | None) -> float:
        """印花税按交易日分段：2023-08-28 财政部减半前 0.1%，其后 0.05%（卖出端）。"""
        if dt is None:
            return cm.stamp_tax_rate if cm is not None else engine_cfg.stamp_tax_rate
        return _STAMP_TAX_RECENT if str(dt) >= "2023-08-28" else _STAMP_TAX_OLD

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

    def _sell_proceeds_and_cost(
        sym: str,
        value: float,
        volume: float,
        amount_ma20: float | None = None,
        dt: str | None = None,
        volatility_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        stamp = _stamp_rate(dt)
        if cm is not None:
            adv = _current_adv(sym)
            total = cm.sell_cost(
                value,
                volume,
                adv,
                amount_ma20=amount_ma20,
                stamp_tax_rate=stamp,
                volatility_multiplier=volatility_multiplier,
            )
            return value - total, total
        commission = max(value * engine_cfg.commission_rate, engine_cfg.min_commission_per_trade)
        slip_rate = max(engine_cfg.slippage, _MIN_SLIPPAGE_FLOOR) * max(volatility_multiplier, 1.0)
        total = commission + value * (engine_cfg.transfer_fee_rate + stamp + slip_rate)
        return value - total, total

    def _buy_cost(
        sym: str, value: float, volume: float, amount_ma20: float | None = None, volatility_multiplier: float = 1.0
    ) -> float:
        if cm is not None:
            adv = _current_adv(sym)
            return cm.buy_cost(value, volume, adv, amount_ma20=amount_ma20, volatility_multiplier=volatility_multiplier)
        commission = max(value * engine_cfg.commission_rate, engine_cfg.min_commission_per_trade)
        slip_rate = max(engine_cfg.slippage, _MIN_SLIPPAGE_FLOOR) * max(volatility_multiplier, 1.0)
        return commission + value * (engine_cfg.transfer_fee_rate + slip_rate)

    def _process_sell(
        dt,
        s_syms,
        s_idx,
        s_close,
        s_vol,
        partial: bool = False,
        s_amount: np.ndarray | None = None,
        s_amp_mult: np.ndarray | None = None,
    ):
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
            proc, cst = _sell_proceeds_and_cost(
                s_syms[j],
                mv,
                float(sell_shares),
                amount_ma20=float(s_amount[j]) if s_amount is not None else None,
                dt=str(dt),
                volatility_multiplier=float(s_amp_mult[j]) if s_amp_mult is not None else 1.0,
            )
            nonlocal cash
            cash += proc
            total_sold += mv
            trade_log.append(
                {
                    "time": dt,
                    "symbol": s_syms[j],
                    "action": "sell" if sell_shares >= sh else "sell_partial",
                    "price": float(s_close[j]),
                    "value": round(proc, 2),
                    "cost": round(cst, 2),
                    # 1.9 流动性拟真字段：实际成交数量（A股最小交易单位整数倍）
                    "qty": int(sell_shares),
                    # 1.7 执行滞后自检字段：同一信号日的复权收盘价（收益乘数锚点）
                    "close_adj": float(s_close[j]),
                }
            )
        return total_sold

    _market_multiplier = 1.0
    _prev_med_score = 0.0  # 上一日中位数评分，用于避免前视偏差

    if use_sw:
        for i_day, (dt, grp) in enumerate(date_groups):
            _sold_today.clear()
            day_data = grp.copy()

            # 市场状态过滤：根据上一日全部股票的中位数评分调整仓位（避免前视偏差）
            _med_score = 0.0 if i_day == 0 else _prev_med_score
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
            # 1.9 流动性拟真：单日振幅 = (high-low)/前收，>5% 剧烈波动日基础滑点翻倍
            high_arr = day_data["high"].values if "high" in day_data.columns else None
            low_arr = day_data["low"].values if "low" in day_data.columns else None
            # 20 日均成交额（元）：用于流动性分档冲击成本
            amount_ma20 = day_data["AMOUNT_MA20"].values if "AMOUNT_MA20" in day_data.columns else None
            buy_score = day_data["进场评分"].values
            sell_score = day_data["退出评分"].values
            risk_str = day_data["风险等级"].astype(str).values

            if i_day % 20 == 0:
                _bs_nonzero = buy_score[buy_score > 0]
                if len(_bs_nonzero) > 0:
                    logger.info(
                        f"[ENGINE-SCORE] {dt}: 进场评分 非零={len(_bs_nonzero)}/{len(buy_score)} mean={_bs_nonzero.mean():.1f} median={float(np.median(_bs_nonzero)):.1f} min={_bs_nonzero.min():.0f} max={_bs_nonzero.max():.0f} >=15={int((buy_score >= 15).sum())} >=60={int((buy_score >= 60).sum())}"
                    )
                else:
                    logger.info(f"[ENGINE-SCORE] {dt}: 进场评分 全为零 ({len(buy_score)} 只)")

            # ── 涨跌停/停牌检查（首日无 prev_bar 时跳过 limit 过滤） ──
            _limit_ratio = np.array(
                [0.20 if s.startswith(("30", "688")) else 0.30 if s.startswith("8") else 0.10 for s in syms_str],
                dtype=np.float64,
            )
            _have_prev = i_day > 0
            if _have_prev:
                _prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close_raw)])
                limit_up_arr = (
                    np.floor(_prev_close_arr * (1 + _limit_ratio) * _LIMIT_PX_PRECISION + 0.5) / _LIMIT_PX_PRECISION
                )
                limit_down_arr = (
                    np.floor(_prev_close_arr * (1 - _limit_ratio) * _LIMIT_PX_PRECISION + 0.5) / _LIMIT_PX_PRECISION
                )
                not_limit_up = close < limit_up_arr
                not_limit_down = close > limit_down_arr
                # 1.9 流动性拟真：单日振幅>5% → 剧烈波动日，基础滑点整体翻倍
                _vol_mult = np.ones(len(close_raw), dtype=np.float64)
                if high_arr is not None and low_arr is not None:
                    _amp = (high_arr - low_arr) / np.maximum(_prev_close_arr, 1e-9)
                    _vol_mult = np.where(_amp > 0.05, 2.0, 1.0)
            else:
                not_limit_up = np.ones(len(close_raw), dtype=bool)
                not_limit_down = np.ones(len(close_raw), dtype=bool)
                _vol_mult = np.ones(len(close_raw), dtype=np.float64)
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

            # ── 卖出（含 T+1 检查 + 分批止盈止损 + ST/退市强平） ──
            held = pos_shares[idx] > 0
            _blocked_syms = _st_blocked_syms_by_day.get(str(dt))
            _st_blocked_idx = (
                np.isin(syms_str, list(_blocked_syms)) if _blocked_syms else None
            )
            if held.any():
                _t1_ok = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
                exit_high = np.isin(risk_str, ["HIGH", "D"])
                exit_gt = (sell_score > buy_score + 20) & (sell_score > 0)
                exit_score_low = (buy_score > 0) & (buy_score < _buy_threshold // 3)
                # ST/退市强平：无视 T+1/跌停/停牌（退市或按策略剔除的标的必须离场）
                force_exit = np.zeros(len(held), dtype=bool)
                if _st_blocked_idx is not None:
                    force_exit = held & _st_blocked_idx & adj_ok
                si_force = np.where(force_exit)[0]
                if len(si_force):
                    daily_sell_value += _process_sell(
                        dt,
                        syms_str[si_force],
                        idx[si_force],
                        close_adj[si_force],
                        volume[si_force],
                        partial=False,
                        s_amount=amount_ma20[si_force] if amount_ma20 is not None else None,
                    )
                    total_value = cash + _calc_market_value()
                sel_all = (
                    held
                    & (exit_high | exit_gt | exit_score_low | stop_hit)
                    & not_limit_down
                    & has_volume
                    & _t1_ok
                    & adj_ok
                    & ~force_exit
                )
                si_all = np.where(sel_all)[0]
                if len(si_all):
                    sel_stop = held & stop_hit & not_limit_down & has_volume & _t1_ok & adj_ok & ~force_exit
                    si_stop = np.where(sel_stop)[0]
                    si_partial = np.setdiff1d(si_all, si_stop)
                    if len(si_stop):
                        daily_sell_value += _process_sell(
                            dt,
                            syms_str[si_stop],
                            idx[si_stop],
                            close_adj[si_stop],
                            volume[si_stop],
                            partial=False,
                            s_amount=amount_ma20[si_stop] if amount_ma20 is not None else None,
                        )
                    if len(si_partial):
                        daily_sell_value += _process_sell(
                            dt,
                            syms_str[si_partial],
                            idx[si_partial],
                            close_adj[si_partial],
                            volume[si_partial],
                            partial=True,
                            s_amount=amount_ma20[si_partial] if amount_ma20 is not None else None,
                        )
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
            _st_ok = (
                ~_st_blocked_idx if _st_blocked_idx is not None
                else np.ones(len(syms_str), dtype=bool)
            )
            buy_ok = (
                (buy_score >= _effective_threshold)
                & (pos_shares[idx] == 0)
                & (~np.isin(risk_str, ["HIGH", "D", "E"]))
                & not_limit_up
                & has_volume
                & _not_sold_today
                & adj_ok
                & _st_ok
            )
            bi = np.where(buy_ok)[0]
            daily_buy_value = 0.0
            if len(bi) == 0 and len(date_groups) > 100 and np.any(buy_score >= _buy_threshold):
                _diag_score = int((buy_score >= _effective_threshold).sum())
                _diag_pos = int((pos_shares[idx] == 0).sum())
                _diag_risk = int((~np.isin(risk_str, ["HIGH", "D", "E"])).sum())
                _diag_limit = int(not_limit_up.sum())
                _diag_vol = int(has_volume.sum())
                _diag_t1 = len(_sold_today)
                logger.info(
                    f"[ENGINE-DIAG] {dt}: 评分≥{_effective_threshold}={_diag_score} 空仓={_diag_pos} 低风险={_diag_risk} 非涨停={_diag_limit} 有量={_diag_vol} T+1禁={_diag_t1} 总={len(buy_ok)}"
                )
            if len(bi):
                b_syms = syms_str[bi]
                b_idx = idx[bi]
                b_close = close_adj[bi]
                b_vol = volume[bi]
                b_amount = amount_ma20[bi] if amount_ma20 is not None else None
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
                    _adv_val = _current_adv(b_syms[j])
                    if _adv_val > 100:
                        max_shares_vol = int(_adv_val * _max_order_pct) // lot * lot
                        shares = min(shares, max_shares_vol)
                        if shares < lot:
                            continue
                    tv = shares * price
                    cst = _buy_cost(
                        b_syms[j], tv, float(shares),
                        amount_ma20=float(b_amount[j]) if b_amount is not None else None,
                        volatility_multiplier=float(_vol_mult[bi[j]]),
                    )
                    if cash >= tv + cst:
                        cash -= tv + cst
                        pos_value[si] = tv
                        pos_shares[si] = shares
                        daily_buy_value += tv
                        _buy_date[b_syms[j]] = str(dt)
                        trade_log.append(
                            {
                                "time": dt,
                                "symbol": b_syms[j],
                                "action": "buy",
                                "price": float(b_close[j]),
                                "value": round(tv, 2),
                                "cost": round(cst, 2),
                                # 1.9 流动性拟真字段：实际成交数量（A股最小交易单位整数倍）
                                "qty": int(shares),
                                # 1.7 执行滞后自检字段：同一信号日的复权收盘价（收益乘数锚点）
                                "close_adj": float(b_close[j]),
                            }
                        )
                        bought += 1

                if bought == 0 and len(date_groups) > 100:
                    _p0 = float(b_close[0]) if n_candidates > 0 else 0
                    _tv0 = (
                        min(cash * equal_weight, total_value * max_pos_pct * _market_multiplier)
                        if n_candidates > 0
                        else 0
                    )
                    _s0 = int(_tv0 / _p0) // 100 * 100 if _p0 > 0 else 0
                    logger.info(
                        f"[ENGINE-DIAG] {dt}: {len(bi)}候选→{n_candidates}TopK 0买入  cash={cash:.0f}  tv[0]={_tv0:.0f}  p[0]={_p0:.0f}  s[0]={_s0}  eq_w={equal_weight:.4f}  max_pos_pct={max_pos_pct}"
                    )

            for i_sym, i_vol in zip(syms_str, volume):
                _update_adv(i_sym, i_vol)

            total_value = cash + _calc_market_value()
            _turnover = (daily_buy_value + daily_sell_value) / (2 * total_value) if total_value > 0 else 0.0
            equity_curve.append(
                {
                    "time": dt,
                    "portfolio_value": round(total_value, 2),
                    "turnover": round(_turnover, 6),
                }
            )
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
            # 1.9 流动性拟真：单日振幅 = (high-low)/前收，>5% 剧烈波动日基础滑点翻倍
            high_arr = day_data["high"].values if "high" in day_data.columns else None
            low_arr = day_data["low"].values if "low" in day_data.columns else None
            # 20 日均成交额（元）：用于流动性分档冲击成本
            amount_ma20 = day_data["AMOUNT_MA20"].values if "AMOUNT_MA20" in day_data.columns else None
            buy_score = day_data["进场评分"].values
            sell_score = day_data["退出评分"].values
            risk_str = day_data["风险等级"].astype(str).values

            # ── 诊断：每 20 天打印一次 buy_score 分布 ──
            if i_day % 20 == 0:
                _bs_nonzero = buy_score[buy_score > 0]
                if len(_bs_nonzero) > 0:
                    logger.info(
                        f"[ENGINE-SCORE] {dt}: 进场评分 非零={len(_bs_nonzero)}/{len(buy_score)} mean={_bs_nonzero.mean():.1f} median={float(np.median(_bs_nonzero)):.1f} min={_bs_nonzero.min():.0f} max={_bs_nonzero.max():.0f} >=15={int((buy_score >= 15).sum())} >=60={int((buy_score >= 60).sum())}"
                    )
                else:
                    logger.info(f"[ENGINE-SCORE] {dt}: 进场评分 全为零 ({len(buy_score)} 只)")

            # ── 涨跌停/停牌检查（首日无 prev_bar 时跳过 limit 过滤） ──
            _limit_ratio = np.array(
                [0.20 if s.startswith(("30", "688")) else 0.30 if s.startswith("8") else 0.10 for s in syms_str],
                dtype=np.float64,
            )
            _have_prev = i_day > 0
            if _have_prev:
                _prev_close_arr = np.array([_prev_bar.get(s, (c, 0))[0] for s, c in zip(syms_str, close_raw)])
                limit_up_arr = (
                    np.floor(_prev_close_arr * (1 + _limit_ratio) * _LIMIT_PX_PRECISION + 0.5) / _LIMIT_PX_PRECISION
                )
                limit_down_arr = (
                    np.floor(_prev_close_arr * (1 - _limit_ratio) * _LIMIT_PX_PRECISION + 0.5) / _LIMIT_PX_PRECISION
                )
                not_limit_up = close < limit_up_arr
                not_limit_down = close > limit_down_arr
                # 1.9 流动性拟真：单日振幅>5% → 剧烈波动日，基础滑点整体翻倍
                _vol_mult = np.ones(len(close_raw), dtype=np.float64)
                if high_arr is not None and low_arr is not None:
                    _amp = (high_arr - low_arr) / np.maximum(_prev_close_arr, 1e-9)
                    _vol_mult = np.where(_amp > 0.05, 2.0, 1.0)
            else:
                not_limit_up = np.ones(len(close_raw), dtype=bool)
                not_limit_down = np.ones(len(close_raw), dtype=bool)
                _vol_mult = np.ones(len(close_raw), dtype=np.float64)
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

            # ── 卖出（含 T+1 检查 + ST/退市强平） ──
            held = pos_shares[idx] > 0
            _blocked_syms = _st_blocked_syms_by_day.get(str(dt))
            _st_blocked_idx = (
                np.isin(syms_str, list(_blocked_syms)) if _blocked_syms else None
            )
            if held.any():
                _t1_ok = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
                exit_high = np.isin(risk_str, ["HIGH", "D"])
                exit_gt = (sell_score > buy_score + 20) & (sell_score > 0)
                exit_score_low = (buy_score > 0) & (buy_score < _buy_threshold // 3)
                # ST/退市强平：无视 T+1/跌停/停牌
                force_exit = np.zeros(len(held), dtype=bool)
                if _st_blocked_idx is not None:
                    force_exit = held & _st_blocked_idx & adj_ok
                si_force = np.where(force_exit)[0]
                if len(si_force):
                    daily_sell_value += _process_sell(
                        dt,
                        syms_str[si_force],
                        idx[si_force],
                        close_adj[si_force],
                        volume[si_force],
                        partial=False,
                        s_amount=amount_ma20[si_force] if amount_ma20 is not None else None,
                    )
                    total_value = cash + _calc_market_value()
                sel_all = (
                    held
                    & (exit_high | exit_gt | exit_score_low | stop_hit)
                    & not_limit_down
                    & has_volume
                    & _t1_ok
                    & adj_ok
                    & ~force_exit
                )
                si_all = np.where(sel_all)[0]
                if len(si_all):
                    sel_stop = held & stop_hit & not_limit_down & has_volume & _t1_ok & adj_ok & ~force_exit
                    si_stop = np.where(sel_stop)[0]
                    si_partial = np.setdiff1d(si_all, si_stop)
                    if len(si_stop):
                        daily_sell_value += _process_sell(
                            dt,
                            syms_str[si_stop],
                            idx[si_stop],
                            close_adj[si_stop],
                            volume[si_stop],
                            partial=False,
                            s_amount=amount_ma20[si_stop] if amount_ma20 is not None else None,
                            s_amp_mult=_vol_mult[si_stop],
                        )
                    if len(si_partial):
                        daily_sell_value += _process_sell(
                            dt,
                            syms_str[si_partial],
                            idx[si_partial],
                            close_adj[si_partial],
                            volume[si_partial],
                            partial=True,
                            s_amount=amount_ma20[si_partial] if amount_ma20 is not None else None,
                            s_amp_mult=_vol_mult[si_partial],
                        )
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
            _st_ok = (
                ~_st_blocked_idx if _st_blocked_idx is not None
                else np.ones(len(syms_str), dtype=bool)
            )
            buy_ok = (
                (buy_score >= _effective_threshold)
                & (pos_shares[idx] == 0)
                & (~np.isin(risk_str, ["HIGH", "D", "E"]))
                & not_limit_up
                & has_volume
                & _not_sold_today
                & adj_ok
                & _st_ok
            )
            bi = np.where(buy_ok)[0]
            daily_buy_value = 0.0
            if len(bi):
                b_syms = syms_str[bi]
                b_idx = idx[bi]
                b_close = close_adj[bi]
                b_vol = volume[bi]
                b_amount = amount_ma20[bi] if amount_ma20 is not None else None

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
                    _adv_val = _current_adv(b_syms[j])
                    if _adv_val > 100:
                        max_shares_vol = int(_adv_val * _max_order_pct) // lot * lot
                        shares = min(shares, max_shares_vol)
                        if shares < lot:
                            continue
                    tv = shares * price
                    cst = _buy_cost(
                        b_syms[j], tv, float(shares),
                        amount_ma20=float(b_amount[j]) if b_amount is not None else None,
                        volatility_multiplier=float(_vol_mult[bi[j]]),
                    )
                    if cash >= tv + cst:
                        cash -= tv + cst
                        pos_value[si] = tv
                        pos_shares[si] = shares
                        daily_buy_value += tv
                        _buy_date[b_syms[j]] = str(dt)
                        trade_log.append(
                            {
                                "time": dt,
                                "symbol": b_syms[j],
                                "action": "buy",
                                "price": float(b_close[j]),
                                "value": round(tv, 2),
                                "cost": round(cst, 2),
                                # 1.9 流动性拟真字段：实际成交数量（A股最小交易单位整数倍）
                                "qty": int(shares),
                                # 1.7 执行滞后自检字段：同一信号日的复权收盘价（收益乘数锚点）
                                "close_adj": float(b_close[j]),
                            }
                        )
                        bought += 1

            for i_sym, i_vol in zip(syms_str, volume):
                _update_adv(i_sym, i_vol)

            total_value = cash + _calc_market_value()
            _turnover = (daily_buy_value + daily_sell_value) / (2 * total_value) if total_value > 0 else 0.0
            equity_curve.append(
                {
                    "time": dt,
                    "portfolio_value": round(total_value, 2),
                    "turnover": round(_turnover, 6),
                }
            )

    total_value = cash + _calc_market_value()

    # ── 1.7 信号执行滞后自检：收益乘数对齐（每笔成交已带同日 close_adj 字段，
    # ── 热路径 O(成交数) 校验；任何开盘/盘中价成交会吃掉当日收益） ──
    try:
        from LogicAnalyzer.ml.execution_lag_integrity import check_price_vs_close_adj

        _lag = check_price_vs_close_adj(trade_log)
        if not _lag.passed:
            logger.warning(f"[执行滞后合规] FAIL: {_lag.details[0] if _lag.details else ''}")
        else:
            logger.info("[执行滞后合规] 收益乘数对齐 PASS（全部成交=信号日收盘价，收益自次日起计）")
    except Exception as _lag_e:
        logger.debug(f"[执行滞后合规] 自检跳过: {_lag_e}")

    # ── 1.8 交易摩擦合规自检（热路径 O(1)：显性成本/滑点下限/动态冲击） ──
    try:
        from LogicAnalyzer.ml.trading_friction_integrity import check_trading_friction_config

        _fric = check_trading_friction_config(engine_cfg)
        if not _fric.passed:
            logger.warning(f"[交易摩擦合规] FAIL: {'；'.join(_fric.details[:3])}")
        else:
            logger.info("[交易摩擦合规] 显性成本+滑点下限+动态冲击全部合规")
    except Exception as _fric_e:
        logger.debug(f"[交易摩擦合规] 自检跳过: {_fric_e}")

    return (total_value / init_cash) - 1
