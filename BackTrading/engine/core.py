from __future__ import annotations

from collections import deque
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig
from BackTrading.calendar_align import get_official_calendar as _cal_get
from BackTrading.limit_pricing import lot_size_for
from BackTrading.domain.models import CostModel

ParamsDict: TypeAlias = dict[str, Any]
TradeLog: TypeAlias = list[dict[str, Any]]
EquityCurve: TypeAlias = list[dict[str, Any]]


# 20 日均量（行业口径：不含当日的滚动均值，避免用当日成交量前视）
_ADV_WINDOW = 20
# 固定滑点强制下限（1.8 交易摩擦合规：A股隐性成本不低于单边 0.05%，
# 配置/回退路径任何低于此值的基础滑点一律抬升，防止 Alpha 虚高）
_MIN_SLIPPAGE_FLOOR = 0.0005
# 挂单过期天数（next_open/vwap 执行模型：信号日 deferred 挂单超此天数未成交则撤销，
# 模拟真实交易日内订单过期机制，避免停牌股数周后以陈旧信号价格成交）
_ORDER_EXPIRY_DAYS = 3


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


def _build_day_limit_model(
    syms_str: np.ndarray,
    close_raw: np.ndarray,
    open_arr: np.ndarray | None,
    high_arr: np.ndarray | None,
    low_arr: np.ndarray | None,
    prev_bar: dict[str, tuple[float, float]],
    st_syms: set[str],
    day_str: str,
    day_idx: dict[str, int],
    listing_map: dict[str, str] | None,
    streak: dict[str, int],
    sim_limits: bool,
    seal_ratio: float,
    tradable_ratio: float,
    seal_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """逐日涨跌停价 + 可成交量比例建模（撮合约束，Task 涨跌停）。

    涨跌停价来自 BackTrading.limit_pricing（主板/创业板/科创板/北交所 + ST 5% +
    上市初期豁免）；无前收（数据首日）的标的按无限制处理（维持原行为）。

    Returns:
        (limit_up, limit_down, at_limit_up, at_limit_down, not_touched_up,
         not_touched_down, touched_up, touched_down, vol_mult, fill_ratio, limit_tag)
        - at_limit_up/at_limit_down: 收盘封板口径（用于连板跟踪与一字板判定）
        - touched_up/touched_down:   盘中触板口径（0.3，high/low/open 触及限价即计，
            用于可成交量折算与买卖限制）
        - fill_ratio: 可成交量比例（sim_limits 关闭时全 1.0）
        - limit_tag:  当日触板方向 "" / "up" / "down"
        - streak:     原地更新连续涨停(+) / 连续跌停(-) 板数
    """
    from BackTrading.limit_pricing import fill_ratio_for, limit_prices_for

    n = len(syms_str)
    limit_up = np.full(n, np.inf)
    limit_down = np.full(n, -np.inf)
    prev_close_arr = np.full(n, np.nan)
    for j, s in enumerate(syms_str):
        pc = prev_bar.get(s)
        if pc is None:
            continue  # 数据首日：无前收 → 豁免（维持原行为）
        prev_close_arr[j] = pc[0]
        _ldays = None
        if listing_map:
            _fs = listing_map.get(s)
            if _fs is not None and _fs in day_idx:
                _ldays = max(1, day_idx[day_str] - day_idx[_fs] + 1)
        info = limit_prices_for(
            pc[0], s, is_st=s in st_syms, listing_days=_ldays, trade_date=day_str
        )
        limit_up[j] = info.limit_up
        limit_down[j] = info.limit_down

    # 收盘封板口径（连板跟踪 / 一字板判定）
    at_limit_up = close_raw >= limit_up - 1e-9
    at_limit_down = close_raw <= limit_down + 1e-9
    # 盘中触板口径（0.3）：high/open 触及涨停价 / low/open 触及跌停价即视为触板
    touched_up = at_limit_up.copy()
    touched_down = at_limit_down.copy()
    if high_arr is not None:
        touched_up = touched_up | (high_arr >= limit_up - 1e-9)
    if low_arr is not None:
        touched_down = touched_down | (low_arr <= limit_down + 1e-9)
    not_touched_up = ~touched_up
    not_touched_down = ~touched_down

    # 1.9 流动性拟真：单日振幅 = (high-low)/前收，>5% 剧烈波动日基础滑点翻倍
    # （无前收的首日回退当日收盘，与原行为一致；全无前收 → 不翻倍）
    vol_mult = np.ones(n, dtype=np.float64)
    if np.isfinite(prev_close_arr).any() and high_arr is not None and low_arr is not None:
        _pc_fb = np.where(np.isnan(prev_close_arr), close_raw, prev_close_arr)
        _amp = (high_arr - low_arr) / np.maximum(_pc_fb, 1e-9)
        vol_mult = np.where(_amp > 0.05, 2.0, 1.0)

    fill_ratio = np.ones(n, dtype=np.float64)
    limit_tag = [""] * n
    if sim_limits:
        for j in range(n):
            if touched_up[j] or touched_down[j]:
                # 连续板数含当日：昨日 streak 0 → 1 板；昨日 1 → 2 板
                boards = max(0, abs(streak.get(syms_str[j], 0))) + 1
                _op = open_arr[j] if open_arr is not None else close_raw[j]
                _hi = high_arr[j] if high_arr is not None else _op
                _lo = low_arr[j] if low_arr is not None else _op
                fill_ratio[j] = fill_ratio_for(
                    close_raw[j], _op, _hi, _lo,
                    limit_up[j], limit_down[j], boards,
                    seal_ratio=seal_ratio, tradable_ratio=tradable_ratio, seal_decay=seal_decay,
                )
                limit_tag[j] = "up" if touched_up[j] else "down"

    # 连板跟踪（收盘后状态，供次日 fill_ratio 用）
    for j, s in enumerate(syms_str):
        cur = streak.get(s, 0)
        if at_limit_up[j]:
            streak[s] = cur + 1 if cur >= 0 else 1
        elif at_limit_down[j]:
            streak[s] = cur - 1 if cur <= 0 else -1
        else:
            streak[s] = 0

    return (
        limit_up, limit_down, at_limit_up, at_limit_down,
        not_touched_up, not_touched_down, touched_up, touched_down,
        vol_mult, fill_ratio, limit_tag,
    )


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

    # ── Task F 引擎日轴对齐官方日历（帧带 is_trading 标志 = prepare 对齐已开启） ──
    # 专业做法：回测日轴 = 交易所日历。数据缺失的官方交易日（全市场无数据）也出现在
    # 日轴与权益曲线上（按上一日市值结转、零换手）；数据中的非日历日视为异常剔除。
    # 日历不可用 / 无标志列（CALENDAR_ALIGN_MODE=off 老版合并）→ 回退按数据日轴迭代。
    _cal_axis = False
    if "is_trading" in data.columns:
        try:
            _cal_dates = _cal_get()
        except Exception:
            _cal_dates = set()
        if _cal_dates:
            _d_min = str(data["trade_date"].min())
            _d_max = str(data["trade_date"].max())
            _axis = sorted(d for d in _cal_dates if _d_min <= d <= _d_max)
            if _axis:
                _grp_map = {str(k): g for k, g in data.groupby("trade_date", sort=True)}
                date_groups = [(str(d), _grp_map.get(str(d), pd.DataFrame())) for d in _axis]
                _cal_axis = True
                _n_data_days = len(_grp_map)
                if len(date_groups) != _n_data_days:
                    logger.info(
                        f"[CALENDAR] 引擎日轴对齐官方日历: {len(date_groups)} 日 "
                        f"（数据 {_n_data_days} 日，补全 {max(0, len(date_groups) - _n_data_days)} 日）"
                    )
            else:
                date_groups = list(data.groupby("trade_date", sort=True))
        else:
            date_groups = list(data.groupby("trade_date", sort=True))
    else:
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

    # ── 撮合约束（simulate_limit_up_down）：涨跌停可成交量规则 ──
    # 开启：触板日按 可成交量比例(一字/盘中 × 连板衰减) 部分成交或未成交（日志可追溯）
    # 关闭：回退简化撮合（触板日一律禁止买入/卖出，等价原行为）
    _sim_limits = bool(getattr(engine_cfg, "simulate_limit_up_down", True))
    _seal_ratio = float(getattr(engine_cfg, "limit_seal_ratio", 0.05))
    _tradable_ratio = float(getattr(engine_cfg, "limit_tradable_ratio", 0.30))
    _seal_decay = float(getattr(engine_cfg, "limit_seal_decay", 0.5))
    # ── 0.6 复牌跳空：停牌后复牌日开盘大幅跳空（补涨兑现卖出 / 补跌日志标记 / 追高禁买） ──
    # 阈值 0 = 关闭该决策（仅识别复牌不动作）
    _resume_gap_up = float(getattr(engine_cfg, "resume_gap_up", 0.05))
    _resume_gap_down = float(getattr(engine_cfg, "resume_gap_down", 0.05))
    # ── 0.1 成交时点模型 ──
    # close=信号日收盘价成交（老行为，偏乐观）/ next_open=信号次日开盘成交（默认，A股T+1）
    # vwap=信号次日VWAP。next_open/vwap 将当日信号挂单，次日开盘按成交模型撮合，
    # 并与 simulate_limit_up_down 联动：次日一字涨停不可买入、一字跌停不可卖出。
    _exec_model = str(getattr(engine_cfg, "execution_model", "next_open")).lower()
    if _exec_model not in ("close", "next_open", "vwap"):
        logger.warning(f"[执行模型] 未知 execution_model={_exec_model!r}，回退 next_open")
        _exec_model = "next_open"
    _defer = _exec_model != "close"
    _limit_streak: dict[str, int] = {}  # 连续涨停(+) / 连续跌停(-) 板数
    # 上市日映射（可选注入 {symbol: "YYYY-MM-DD"}；缺省时仅数据首日豁免）
    _listing_days_map = params.get("_listing_days") if isinstance(params, dict) else None
    _day_idx = {str(dt): i for i, (dt, _g) in enumerate(date_groups)}
    # 逐日 ST/*ST 集合（涨跌幅 5% 用）
    _st_syms_by_day: dict[str, set[str]] = {}
    if _st_hist:
        for _s, _recs in _st_hist.items():
            if not _recs:
                continue
            for _d_str, (_st_f, _dl) in _recs.items():
                if _st_f:
                    _st_syms_by_day.setdefault(str(_d_str), set()).add(_s)

    # 成本单一来源：CostModel（佣金含最低5元 / 印花税日期分段表 / 过户费 / 经手费+证管费 /
    # 滑点 / 流动性分档冲击）。未显式注入时由 EngineConfig 字段自动构建，行为与原回退路径一致。
    cm = engine_cfg.cost_model
    if cm is None:
        cm = CostModel(
            commission_rate=engine_cfg.commission_rate,
            stamp_tax_rate=engine_cfg.stamp_tax_rate,
            market_slippage=max(engine_cfg.slippage, _MIN_SLIPPAGE_FLOOR),
            limit_slippage=max(engine_cfg.slippage, _MIN_SLIPPAGE_FLOOR) * 0.5,
            min_commission_per_trade=engine_cfg.min_commission_per_trade,
            transfer_fee_rate=engine_cfg.transfer_fee_rate,
        )
    # 成本拆解累计（佣金/印花税/过户费/经手费/证管费/滑点/冲击 + 买卖成交额），
    # 回测结束输出 [成本拆解] 报告（各项占总成本百分比）
    _cost_accum: dict[str, float] = {
        "commission": 0.0, "stamp": 0.0, "transfer": 0.0,
        "handling": 0.0, "csrc": 0.0, "slippage": 0.0, "impact": 0.0,
        "buy_value": 0.0, "sell_value": 0.0,
    }
    # 滚动 20 日成交量（不含当日，前视合规）：value = (deque, run_sum)
    _adv_state: dict[str, tuple[Any, float]] = {}
    _prev_bar: dict[str, tuple[float, float]] = {}
    # 0.6 复牌跳空：每只标的最近一次有行情（bar）的交易日，用于识别停牌复牌
    _prev_bar_date: dict[str, str] = {}
    _buy_date: dict[str, str] = {}

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
    _last_total_value = cash  # Task F 日历轴补全日：权益按上一日市值结转

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

    def _calc_market_value() -> float:
        held = np.where(pos_shares > 0)[0]
        mtm = 0.0
        for si in held:
            s = symbols[si]
            px = close_lookup.get(s)
            if px is None or not np.isfinite(px):
                # 0.4 停牌盯市：当日无 bar（停牌/无行情）或 close_adj 缺失的持仓按
                # "停牌前最后收盘价"估值，不再冻结在买入成本价（否则上涨遇停牌净值被低估、
                # 下跌被高估，且停牌期净值"无波动"会虚高 Sharpe）
                px = _last_close.get(s)
            if px is not None and np.isfinite(px):
                mtm += pos_shares[si] * px
            else:
                mtm += pos_value[si]  # 无任何历史收盘（理论不应发生）→ 退回买入成本
        return mtm

    close_lookup: dict[str, float] = {}
    # 0.4 停牌盯市：每只有行情交易日收盘后更新为当日复权收盘价，供停牌日估值回退
    _last_close: dict[str, float] = {}

    def _susp_position_value() -> float:
        """停牌持仓市值（当日无 bar 的持仓按停牌前最后收盘价盯市）。"""
        held = np.where(pos_shares > 0)[0]
        v = 0.0
        for si in held:
            s = symbols[si]
            px = close_lookup.get(s)
            if px is not None and np.isfinite(px):
                continue  # 当日有行情 → 非停牌
            px = _last_close.get(s, pos_value[si] / max(pos_shares[si], 1))
            v += pos_shares[si] * px
        return v

    def _sell_proceeds_and_cost(
        sym: str,
        value: float,
        volume: float,
        amount_ma20: float | None = None,
        dt: str | None = None,
        volatility_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        parts = cm.sell_cost_breakdown(
            value,
            volume,
            _current_adv(sym),
            amount_ma20=amount_ma20,
            dt=dt,
            volatility_multiplier=volatility_multiplier,
        )
        _cost_accum["sell_value"] += value
        for _k in ("commission", "stamp", "transfer", "handling", "csrc", "slippage", "impact"):
            _cost_accum[_k] += parts[_k]
        return value - parts["total"], parts["total"]

    def _buy_cost(
        sym: str, value: float, volume: float, amount_ma20: float | None = None, volatility_multiplier: float = 1.0
    ) -> float:
        parts = cm.buy_cost_breakdown(
            value,
            volume,
            _current_adv(sym),
            amount_ma20=amount_ma20,
            volatility_multiplier=volatility_multiplier,
        )
        _cost_accum["buy_value"] += value
        for _k in ("commission", "transfer", "handling", "csrc", "slippage", "impact"):
            _cost_accum[_k] += parts[_k]
        return parts["total"]

    def _process_sell(
        dt,
        s_syms,
        s_idx,
        s_close,
        s_vol,
        partial: bool = False,
        s_amount: np.ndarray | None = None,
        s_amp_mult: np.ndarray | None = None,
        s_fill_ratio: np.ndarray | None = None,
        s_limit_tag: np.ndarray | None = None,
        s_sig_close: np.ndarray | None = None,
        s_force: bool = False,
    ):
        total_sold = 0.0
        for j in range(len(s_syms)):
            si = s_idx[j]
            sh = int(pos_shares[si])
            if sh <= 0:
                continue
            # 申报数量单位：按板块（科创 200 股/手，其余 100 股/手），一处定义全链路复用
            lot = lot_size_for(s_syms[j])
            if partial:
                # 半仓减仓：取最近整手数（四舍五入而非向下取整，避免 300 股只卖 100 股）
                sell_shares = max(lot, int(sh / 2 / lot + 0.5) * lot)
                if sell_shares >= sh:
                    sell_shares = sh
            else:
                sell_shares = sh
            # 撮合约束：跌停/涨停日按可成交量比例部分成交或未成交（日志可追溯）
            _limit_note = None
            if s_fill_ratio is not None and s_fill_ratio[j] < 1.0:
                _req = sell_shares
                _avail = int(float(s_vol[j]) * float(s_fill_ratio[j])) // lot * lot
                _updown = "涨停" if (s_limit_tag is not None and s_limit_tag[j] == "up") else "跌停"
                if _avail < lot:
                    logger.info(
                        f"[撮合约束] {dt} {s_syms[j]} {_updown} 可成交量不足 → 未成交（卖出） 请求={_req}股 可成交={_avail}股"
                    )
                    continue
                if _avail < sell_shares:
                    sell_shares = _avail
                    _limit_note = s_limit_tag[j] if s_limit_tag is not None else "down"
                    logger.info(
                        f"[撮合约束] {dt} {s_syms[j]} {_updown} 部分成交（卖出） 请求={_req}股 成交={sell_shares}股 fill_ratio={float(s_fill_ratio[j]):.3f}"
                    )
            mv = sell_shares * float(s_close[j])
            pos_shares[si] -= sell_shares
            if pos_shares[si] <= 0:
                pos_value[si] = 0.0
                pos_shares[si] = 0
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
            _extra = (
                {"limit": _limit_note, "fill_ratio": round(float(s_fill_ratio[j]), 3)}
                if _limit_note is not None
                else {}
            )
            _exec_anchor = float(s_sig_close[j]) if s_sig_close is not None else float(s_close[j])
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
                    # 1.7 执行滞后自检字段：信号日复权收盘价锚点（next_open 下为前一日信号收盘）
                    "close_adj": _exec_anchor,
                    # 0.1 成交参考价（成交日开盘/典型价/收盘）——成交时序自检锚点
                    "exec_open": float(s_close[j]),
                    **_extra,
                    **({"force_exit": True} if s_force else {}),
                }
            )
        return total_sold

    # ── 0.1 成交时点模型：挂单队列（next_open/vwap） ──
    # 信号日收盘下单 → 次日开盘按成交模型撮合。先卖后买（卖出回笼资金再买入）。
    _pending_sells: list[dict[str, Any]] = []
    _pending_buys: list[dict[str, Any]] = []

    def _exec_price_for(day_data_ld: pd.DataFrame, j: int, close_adj_local) -> float:
        """成交参考价（0.1 执行时点模型）。

        close=当日收盘 / next_open=开盘 / vwap=日频典型价近似。
        注："vwap" 模式用 (O+H+L+C)/4 典型价作为日频 VWAP 近似，
        非真实日内量价加权 VWAP（需分钟线/逐笔数据），仅适用于日频回测。
        """
        if _exec_model == "next_open":
            if "open_adj" in day_data_ld.columns:
                v = float(day_data_ld["open_adj"].values[j])
            elif "open" in day_data_ld.columns:
                v = float(day_data_ld["open"].values[j])
            else:
                v = float(close_adj_local[j])
            return v if np.isfinite(v) and v > 0 else float(close_adj_local[j])
        if _exec_model == "vwap":
            if "open_adj" in day_data_ld.columns and "high_adj" in day_data_ld.columns and "low_adj" in day_data_ld.columns:
                _o = float(day_data_ld["open_adj"].values[j])
                _h = float(day_data_ld["high_adj"].values[j])
                _l = float(day_data_ld["low_adj"].values[j])
            else:
                _o = float(day_data_ld["open"].values[j]) if "open" in day_data_ld.columns else float(close_adj_local[j])
                _h = float(day_data_ld["high"].values[j]) if "high" in day_data_ld.columns else float(close_adj_local[j])
                _l = float(day_data_ld["low"].values[j]) if "low" in day_data_ld.columns else float(close_adj_local[j])
            _c = float(close_adj_local[j])
            _tp = (_o + _h + _l + _c) / 4.0
            return _tp if np.isfinite(_tp) and _tp > 0 else float(close_adj_local[j])
        return float(close_adj_local[j])

    def _flush_pending(
        dt,
        day_data_ld,
        syms_str_ld,
        idx_ld,
        close_adj_ld,
        close_raw_ld,
        open_arr_ld,
        volume_ld,
        at_limit_up_ld,
        at_limit_down_ld,
        limit_up_ld,
        limit_down_ld,
        adj_ok_ld,
        has_volume_ld,
        amount_ma20_ld,
        _vol_mult_ld,
        _limit_fill_ld,
        _limit_tag_ld,
        resume_gap_up_ld=None,
    ) -> tuple[float, float]:
        """次日开盘撮合（0.1）：先卖后买，一字板联动限制（一字涨停不可买/一字跌停不可卖）。

        Returns:
            (buy_value, sell_value) 当日入账的买卖金额（用于 turnover 统计）。
        """
        nonlocal cash
        if not _defer or (not _pending_sells and not _pending_buys):
            return 0.0, 0.0
        buy_val, sell_val = 0.0, 0.0
        sym_row = {s: j for j, s in enumerate(syms_str_ld)}
        # 一字板判定：开=收=限价（sealed board）
        def _is_seal_up(j: int) -> bool:
            return at_limit_up_ld[j] and open_arr_ld is not None and abs(open_arr_ld[j] - close_raw_ld[j]) <= 1e-9

        def _is_seal_down(j: int) -> bool:
            return at_limit_down_ld[j] and open_arr_ld is not None and abs(open_arr_ld[j] - close_raw_ld[j]) <= 1e-9

        # ── 卖出挂单（先回笼资金） ──
        remaining_sells: list[dict[str, Any]] = []
        for p in _pending_sells:
            # ── P3-3：挂单过期检查（停牌顺延超 _ORDER_EXPIRY_DAYS 个交易日则撤销） ──
            p["_age"] = p.get("_age", 0) + 1
            if p["_age"] > _ORDER_EXPIRY_DAYS:
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 卖出挂单过期"
                    f"（信号日 {p['sig_dt']}，已等待 {p['_age'] - 1} 个交易日）→ 撤销"
                )
                continue
            jj = sym_row.get(p["sym"])
            if jj is None or not adj_ok_ld[jj] or not has_volume_ld[jj]:
                remaining_sells.append(p)  # 停牌/当日无行 → 顺延
                continue
            px = _exec_price_for(day_data_ld, jj, close_adj_ld)
            if _is_seal_down(jj):
                logger.info(f"[执行模型] {dt} {p['sym']} 一字跌停 → 卖出未成交（撤销）")
                continue
            # 0.3 盘中触板：仅在跌停开盘（open ≤ 跌停价）时按日级可成交量折算，
            # 正常开盘不因盘中触板而限制成交（避免误伤正常开盘的单子）
            _open_at_limit_down = (
                open_arr_ld is not None and open_arr_ld[jj] <= limit_down_ld[jj] + 1e-9
            )
            sell_val += _process_sell(
                dt,
                np.array([p["sym"]], dtype=object),
                np.array([idx_ld[jj]], dtype=np.int32),
                np.array([px]),
                np.array([volume_ld[jj]]),
                partial=bool(p["partial"]),
                s_amount=np.array([float(amount_ma20_ld[jj])]) if amount_ma20_ld is not None else None,
                s_amp_mult=np.array([float(_vol_mult_ld[jj])]),
                s_fill_ratio=np.array([float(_limit_fill_ld[jj])]) if (_sim_limits and _open_at_limit_down) else None,
                s_limit_tag=[_limit_tag_ld[jj]] if (_sim_limits and _open_at_limit_down) else None,
                s_sig_close=np.array([float(p["sig_close"])]),
            )
        _pending_sells[:] = remaining_sells
        # ── 买入挂单（按信号日优先级顺序；一字涨停不可买） ──
        filled = 0
        if _max_holdings > 0:
            _slots = max(0, _max_holdings - int((pos_shares > 0).sum()))
        else:
            _slots = _top_k
        remaining_buys: list[dict[str, Any]] = []
        for p in _pending_buys:
            # ── P3-3：挂单过期检查（停牌顺延超 _ORDER_EXPIRY_DAYS 个交易日则撤销） ──
            p["_age"] = p.get("_age", 0) + 1
            if p["_age"] > _ORDER_EXPIRY_DAYS:
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 买入挂单过期"
                    f"（信号日 {p['sig_dt']}，已等待 {p['_age'] - 1} 个交易日）→ 撤销"
                )
                continue
            if filled >= _slots:
                continue  # 无空仓额度 → 撤销
            jj = sym_row.get(p["sym"])
            if jj is None or not adj_ok_ld[jj] or not has_volume_ld[jj]:
                remaining_buys.append(p)  # 停牌/当日无行 → 顺延
                continue
            si = p["si"]
            if pos_shares[si] > 0:
                continue  # 已持仓 → 撤销
            if resume_gap_up_ld is not None and resume_gap_up_ld[jj]:
                logger.info(f"[执行模型] {dt} {p['sym']} 复牌高开 → 买入挂单撤销（追高）")
                continue
            if _is_seal_up(jj):
                logger.info(f"[执行模型] {dt} {p['sym']} 一字涨停 → 买入未成交（撤销）")
                continue
            px = _exec_price_for(day_data_ld, jj, close_adj_ld)
            if not np.isfinite(px) or px <= 0:
                remaining_buys.append(p)
                continue
            lot = lot_size_for(p["sym"])
            shares = int(float(p["tv"]) / px) // lot * lot
            if shares < lot:
                continue
            _adv_val = _current_adv(p["sym"])
            if _adv_val > 100:
                max_shares_vol = int(_adv_val * _max_order_pct) // lot * lot
                shares = min(shares, max_shares_vol)
                if shares < lot:
                    continue
            _limit_note = None
            # 0.3 盘中触板：仅在涨停开盘（open ≥ 涨停价）时按日级可成交量折算，
            # 正常开盘的挂单不因盘中触板而受限
            _open_at_limit_up = (
                open_arr_ld is not None and open_arr_ld[jj] >= limit_up_ld[jj] - 1e-9
            )
            if _sim_limits and _limit_fill_ld[jj] < 1.0 and _open_at_limit_up:
                _req = shares
                _avail = int(float(volume_ld[jj]) * float(_limit_fill_ld[jj])) // lot * lot
                _updown = "涨停" if _limit_tag_ld[jj] == "up" else "跌停"
                if _avail < lot:
                    logger.info(
                        f"[撮合约束/执行模型] {dt} {p['sym']} {_updown} 可成交量不足 → 未成交（买入） 请求={_req}股 可成交={_avail}股"
                    )
                    continue
                if _avail < shares:
                    shares = _avail
                    _limit_note = _limit_tag_ld[jj]
                    logger.info(
                        f"[撮合约束/执行模型] {dt} {p['sym']} {_updown} 部分成交（买入） 请求={_req}股 成交={shares}股 fill_ratio={float(_limit_fill_ld[jj]):.3f}"
                    )
            tv = shares * px
            cst = _buy_cost(
                p["sym"], tv, float(shares),
                amount_ma20=float(amount_ma20_ld[jj]) if amount_ma20_ld is not None else None,
                volatility_multiplier=float(_vol_mult_ld[jj]),
            )
            if cash < tv + cst:
                continue
            cash -= tv + cst
            pos_value[si] = tv
            pos_shares[si] = shares
            buy_val += tv
            _buy_date[p["sym"]] = str(dt)
            _extra_buy = (
                {"limit": _limit_note, "fill_ratio": round(float(_limit_fill_ld[jj]), 3)}
                if _limit_note is not None
                else {}
            )
            trade_log.append(
                {
                    "time": dt,
                    "symbol": p["sym"],
                    "action": "buy",
                    "price": float(px),
                    "value": round(tv, 2),
                    "cost": round(cst, 2),
                    "qty": int(shares),
                    "close_adj": float(p["sig_close"]),
                    "exec_open": float(px),
                    **_extra_buy,
                }
            )
            filled += 1
        _pending_buys[:] = remaining_buys
        return buy_val, sell_val

    _market_multiplier = 1.0
    _prev_med_score = 0.0  # 上一日中位数评分，用于避免前视偏差

    for i_day, (dt, grp) in enumerate(date_groups):
        day_data = grp.copy()

        # Task F 日历轴补全日（全市场无数据的官方交易日）：无成交、无估值变动，
        # 权益按上一日市值结转（保持日轴与交易所日历 100% 对齐）
        if day_data.empty:
            equity_curve.append(
                {
                    "time": dt,
                    "portfolio_value": round(_last_total_value, 2),
                    "turnover": 0.0,
                }
            )
            continue

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
        # 0.4 停牌盯市：记录当日有行情（adj_ok）的复权收盘价，供停牌日估值回退
        for _k, _s in enumerate(syms_str):
            if adj_ok[_k]:
                _last_close[_s] = float(close_adj[_k])
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
        # 涨跌停价 + 可成交量比例：BackTrading.limit_pricing
        # （主板 10% / 创业板·科创板 20% / 北交所 30% + ST 5% + 上市初期豁免）
        open_arr = day_data["open"].values if "open" in day_data.columns else None
        (
            limit_up_arr, limit_down_arr,
            at_limit_up, at_limit_down,
            not_touched_up, not_touched_down,
            _touched_up, _touched_down,
            _vol_mult, _limit_fill, _limit_tag,
        ) = _build_day_limit_model(
            syms_str, close_raw, open_arr, high_arr, low_arr,
            _prev_bar, _st_syms_by_day.get(str(dt), set()),
            str(dt), _day_idx, _listing_days_map,
            _limit_streak, _sim_limits,
            _seal_ratio, _tradable_ratio, _seal_decay,
        )
        _have_prev = i_day > 0
        has_volume = volume > 0

        # ── 0.6 复牌跳空识别：当日有 bar 但上一交易日缺失（停牌复牌）→ 相对停牌前收盘的跳空 ──
        # 补涨（高开≥resume_gap_up）→ 开盘兑现卖出 + 当日禁买（追高）；
        # 补跌（低开）→ 日志标记（风控卖出照常）。阈值 0 = 仅识别不动作。
        resume_gap = np.full(len(syms_str), np.nan)
        if i_day > 0:
            _prev_trade_date = str(date_groups[i_day - 1][0])
            for _k, _s in enumerate(syms_str):
                _lbd = _prev_bar_date.get(_s)
                if _lbd is None or _lbd >= _prev_trade_date:
                    continue
                _pc = _prev_bar.get(_s)
                _op = open_arr[_k] if open_arr is not None else None
                if (_pc is None or _pc[0] <= 0 or _op is None
                        or not np.isfinite(_op) or _op <= 0):
                    continue
                resume_gap[_k] = _op / _pc[0] - 1.0
        resume_gap_up = np.zeros(len(syms_str), dtype=bool)
        resume_gap_down = np.zeros(len(syms_str), dtype=bool)
        if _resume_gap_up > 0:
            resume_gap_up = (resume_gap >= _resume_gap_up) & np.isfinite(resume_gap)
            for _k in np.where(resume_gap_up)[0][:20]:
                logger.info(
                    f"[复牌] {dt} {syms_str[_k]} 高开 {resume_gap[_k]*100:.2f}% → 当日禁买（追高）；持仓则开盘兑现"
                )
        if _resume_gap_down > 0:
            resume_gap_down = (resume_gap <= -_resume_gap_down) & np.isfinite(resume_gap)
            for _k in np.where(resume_gap_down)[0][:20]:
                logger.info(
                    f"[复牌] {dt} {syms_str[_k]} 低开 {resume_gap[_k]*100:.2f}% （补跌，风控卖出照常）"
                )

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
                _prev_bar_date[s] = str(dt)
        stop_hit = stop_hit_col | stop_hit_atr

        close_lookup = dict(zip(syms_str, close_adj))
        total_value = cash + _calc_market_value()
        # 0.1 执行时序：次日开盘撮合昨日挂单（先卖后买；一字板联动）
        daily_buy_value, daily_sell_value = _flush_pending(
            dt, day_data, syms_str, idx, close_adj, close_raw, open_arr,
            volume, at_limit_up, at_limit_down, limit_up_arr, limit_down_arr,
            adj_ok, has_volume,
            amount_ma20, _vol_mult, _limit_fill, _limit_tag,
            resume_gap_up,
        )
        if daily_buy_value or daily_sell_value:
            total_value = cash + _calc_market_value()

        # 0.6 复牌高开兑现：停牌后跳空高开（补涨）→ 复牌日开盘价全部卖出（先于常规卖出）
        if _resume_gap_up > 0 and np.any(resume_gap_up):
            _resume_held = pos_shares[idx] > 0
            _resume_sell = _resume_held & resume_gap_up & adj_ok & has_volume
            si_resume = np.where(_resume_sell)[0]
            if len(si_resume):
                for _k in si_resume:
                    logger.info(
                        f"[复牌] {dt} {syms_str[_k]} 高开 {resume_gap[_k]*100:.2f}% → 开盘兑现卖出"
                    )
                _resume_px = open_arr[si_resume] if open_arr is not None else close_adj[si_resume]
                daily_sell_value += _process_sell(
                    dt,
                    syms_str[si_resume],
                    idx[si_resume],
                    _resume_px,
                    volume[si_resume],
                    partial=False,
                    s_amount=amount_ma20[si_resume] if amount_ma20 is not None else None,
                    s_sig_close=close_adj[si_resume],
                )
                close_lookup = dict(zip(syms_str[adj_ok], close_adj[adj_ok]))
                total_value = cash + _calc_market_value()

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
                    s_force=True,
                )
                total_value = cash + _calc_market_value()
            sel_all = (
                held
                & (exit_high | exit_gt | exit_score_low | stop_hit)
                & (not_touched_down | _sim_limits)
                & has_volume
                & _t1_ok
                & adj_ok
                & ~force_exit
            )
            si_all = np.where(sel_all)[0]
            if _sim_limits:
                # 跌停无量 → 未成交（撮合约束日志，可追溯）
                _zv_sell = np.where(
                    held & _touched_down & ~has_volume & _t1_ok & adj_ok & ~force_exit
                )[0]
                for _j in _zv_sell[:20]:
                    logger.info(
                        f"[撮合约束] {dt} {syms_str[_j]} 跌停无量 → 未成交（卖出）"
                    )
            if len(si_all):
                sel_stop = held & stop_hit & (not_touched_down | _sim_limits) & has_volume & _t1_ok & adj_ok & ~force_exit
                si_stop = np.where(sel_stop)[0]
                si_partial = np.setdiff1d(si_all, si_stop)
                if _defer:
                    # 0.1 执行时序：信号日收盘决策 → 挂单次日开盘成交
                    for _k in si_stop:
                        _pending_sells.append({
                            "sym": syms_str[_k], "si": idx[_k], "partial": False,
                            "sig_dt": str(dt), "sig_close": float(close_adj[_k]),
                        })
                    for _k in si_partial:
                        _pending_sells.append({
                            "sym": syms_str[_k], "si": idx[_k], "partial": True,
                            "sig_dt": str(dt), "sig_close": float(close_adj[_k]),
                        })
                else:
                    if len(si_stop):
                        daily_sell_value += _process_sell(
                            dt,
                            syms_str[si_stop],
                            idx[si_stop],
                            close_adj[si_stop],
                            volume[si_stop],
                            partial=False,
                            s_amount=amount_ma20[si_stop] if amount_ma20 is not None else None,
                            s_fill_ratio=_limit_fill[si_stop] if _sim_limits else None,
                            s_limit_tag=[_limit_tag[k] for k in si_stop] if _sim_limits else None,
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
                            s_fill_ratio=_limit_fill[si_partial] if _sim_limits else None,
                            s_limit_tag=[_limit_tag[k] for k in si_partial] if _sim_limits else None,
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
        _st_ok = (
            ~_st_blocked_idx if _st_blocked_idx is not None
            else np.ones(len(syms_str), dtype=bool)
        )
        buy_ok = (
            (buy_score >= _effective_threshold)
            & (pos_shares[idx] == 0)
            & (~np.isin(risk_str, ["HIGH", "D", "E"]))
            & (not_touched_up | _sim_limits)
            & (~resume_gap_up)  # 0.6 复牌高开当日禁买（追高）
            & has_volume
            & adj_ok
            & _st_ok
        )
        bi = np.where(buy_ok)[0]
        if _sim_limits:
            # 涨停无量 → 未成交（撮合约束日志，可追溯）
            _zv_buy = np.where(
                _touched_up
                & ~has_volume
                & (buy_score >= _effective_threshold)
                & (pos_shares[idx] == 0)
                & (~np.isin(risk_str, ["HIGH", "D", "E"]))
                & adj_ok
                & _st_ok
            )[0]
            for _j in _zv_buy[:20]:
                logger.info(
                    f"[撮合约束] {dt} {syms_str[_j]} 涨停无量 → 未成交（买入）"
                )
        daily_buy_value = daily_buy_value  # 已由 _flush_pending 初始化（收盘模型为 0）
        if len(bi) == 0 and len(date_groups) > 100 and np.any(buy_score >= _buy_threshold):
            _diag_score = int((buy_score >= _effective_threshold).sum())
            _diag_pos = int((pos_shares[idx] == 0).sum())
            _diag_risk = int((~np.isin(risk_str, ["HIGH", "D", "E"])).sum())
            _diag_limit = int(not_touched_up.sum())
            _diag_vol = int(has_volume.sum())
            logger.info(
                f"[ENGINE-DIAG] {dt}: 评分≥{_effective_threshold}={_diag_score} 空仓={_diag_pos} 低风险={_diag_risk} 非涨停={_diag_limit} 有量={_diag_vol} 总={len(buy_ok)}"
            )
        if len(bi):
            b_syms = syms_str[bi]
            b_idx = idx[bi]
            b_close = close_adj[bi]
            b_vol = volume[bi]
            b_amount = amount_ma20[bi] if amount_ma20 is not None else None

            # Top-K 等权分配：集中资金到评分最高的 _top_k 只
            if len(bi) > _top_k:
                b_scores = buy_score[bi]
                _top_indices = np.argpartition(-b_scores, _top_k)[:_top_k]
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
                # 申报数量单位：按板块（科创 200 股/手，其余 100 股/手），一处定义全链路复用
                lot = lot_size_for(b_syms[j])
                tv = min(cash * equal_weight, total_value * max_pos_pct * _market_multiplier)
                shares = int(tv / price) // lot * lot if price > 0 else 0
                if shares < lot:
                    continue
                if _defer:
                    # 0.1 执行时序：挂单次日开盘成交，撮合成本/可成交量在次日判定
                    _pending_buys.append({
                        "sym": b_syms[j], "si": si, "tv": tv,
                        "sig_dt": str(dt), "sig_close": float(b_close[j]),
                    })
                    bought += 1
                    continue
                _adv_val = _current_adv(b_syms[j])
                if _adv_val > 100:
                    max_shares_vol = int(_adv_val * _max_order_pct) // lot * lot
                    shares = min(shares, max_shares_vol)
                    if shares < lot:
                        continue
                # 撮合约束：涨停/跌停日按可成交量比例部分成交或未成交（日志可追溯）
                _limit_note = None
                if _sim_limits and _limit_fill[bi[j]] < 1.0:
                    _req = shares
                    _avail = int(b_vol[j] * _limit_fill[bi[j]]) // lot * lot
                    _updown = "涨停" if _limit_tag[bi[j]] == "up" else "跌停"
                    if _avail < lot:
                        logger.info(
                            f"[撮合约束] {dt} {b_syms[j]} {_updown} 可成交量不足 → 未成交（买入） 请求={_req}股 可成交={_avail}股"
                        )
                        continue
                    if _avail < shares:
                        shares = _avail
                        _limit_note = _limit_tag[bi[j]]
                        logger.info(
                            f"[撮合约束] {dt} {b_syms[j]} {_updown} 部分成交（买入） 请求={_req}股 成交={shares}股 fill_ratio={float(_limit_fill[bi[j]]):.3f}"
                        )
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
                    _extra_buy = (
                        {"limit": _limit_note, "fill_ratio": round(float(_limit_fill[bi[j]]), 3)}
                        if _limit_note is not None
                        else {}
                    )
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
                            "exec_open": float(b_close[j]),
                            **_extra_buy,
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
        _last_total_value = total_value
        _susp_v = _susp_position_value()
        _ec_rec = {
            "time": dt,
            "portfolio_value": round(total_value, 2),
            "turnover": round(_turnover, 6),
        }
        if _susp_v > 0 and total_value > 0:
            # 0.4 流动性风险指标：停牌期持仓市值占比（行业标配）
            _ec_rec["susp_value_ratio"] = round(_susp_v / total_value, 6)
        equity_curve.append(_ec_rec)
    total_value = cash + _calc_market_value()

    # ── 1.7 信号执行滞后自检：收益乘数对齐（每笔成交已带同日 close_adj 字段，
    # ── 热路径 O(成交数) 校验；任何开盘/盘中价成交会吃掉当日收益） ──
    try:
        from LogicAnalyzer.ml.execution_lag_integrity import check_price_vs_close_adj

        _lag = check_price_vs_close_adj(trade_log, exec_mode=_exec_model)
        if not _lag.passed:
            logger.warning(f"[执行滞后合规] FAIL: {_lag.details[0] if _lag.details else ''}")
        else:
            logger.info(f"[执行滞后合规] 成交时序对齐 PASS（execution_model={_exec_model}，收益自成交日起计）")
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

    # ── 成本拆解报告：各项占总成本百分比（单一来源 CostModel 汇总） ──
    try:
        _components = ("commission", "stamp", "transfer", "handling", "csrc", "slippage", "impact")
        _total_cost = sum(_cost_accum[k] for k in _components)
        _buy_v, _sell_v = _cost_accum["buy_value"], _cost_accum["sell_value"]
        if _total_cost > 0:
            _labels = {
                "commission": "佣金", "stamp": "印花税", "transfer": "过户费",
                "handling": "经手费", "csrc": "证管费", "slippage": "滑点", "impact": "冲击",
            }
            _lines = [
                f"[成本拆解] 总成本={_total_cost:.2f} 元（买入成交额 {_buy_v:.0f} / 卖出成交额 {_sell_v:.0f}，"
                f"成本率={_total_cost / max(_buy_v + _sell_v, 1e-9):.4%}）"
            ]
            for _k in _components:
                _v = _cost_accum[_k]
                _lines.append(
                    f"   {_labels[_k]}: {_v:.2f} 元 ({_v / _total_cost * 100:.2f}%)"
                )
            logger.info("\n".join(_lines))
    except Exception as _cost_e:
        logger.debug(f"[成本拆解] 报告跳过: {_cost_e}")

    return (total_value / init_cash) - 1
