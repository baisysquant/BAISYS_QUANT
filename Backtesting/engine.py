from __future__ import annotations

import itertools
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from Backtesting.analytics import compute_risk_metrics, compute_trade_metrics
from Backtesting.portfolio import allocate_weights


@dataclass
class Order:
    symbol: str
    action: Literal["buy", "sell"]
    target_weight: float = 0.0


@dataclass
class TradeRecord:
    time: str
    symbol: str
    action: str
    price: float
    shares: float
    value: float
    cost: float


@dataclass
class BarRow:
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage: float = 0.001
    max_position_pct: float = 0.1
    portfolio_method: str = "score_weighted"  # score_weighted / risk_parity / min_variance / mean_variance
    point_in_time: bool = True
    atr_stop_mult: float = 1.5
    kelly_fraction: float = 0.25
    position_a: float = 0.3
    liq_veto_ratio: float = 0.05
    boll_narrow_ratio: float = 0.8
    cross_decay_days: int = 30


_WALK_FORWARD_RESULTS: list[dict[str, Any]] = []


def get_walk_forward_results() -> list[dict[str, Any]]:
    return list(_WALK_FORWARD_RESULTS)


def clear_walk_forward_results() -> None:
    _WALK_FORWARD_RESULTS.clear()


# 模块级函数（Windows 下 ProcessPoolExecutor 需要可 pickle）
_EVAL_PARAM_KEYS: list[str] = []
_EVAL_CFG: EngineConfig | None = None
_EVAL_DATA_CACHE: pd.DataFrame | None = None
_EVAL_COMBO_IDX: int = 0


def _init_eval_worker(filepath: str, param_keys: list[str], cfg: EngineConfig | None = None) -> None:
    global _EVAL_PARAM_KEYS, _EVAL_CFG, _EVAL_DATA_CACHE
    _EVAL_PARAM_KEYS = param_keys
    _EVAL_CFG = cfg or EngineConfig()
    _EVAL_DATA_CACHE = pd.read_parquet(filepath)


def _eval_one_combo(combo: tuple[Any, ...]) -> tuple[float, dict[str, Any]]:
    global _EVAL_PARAM_KEYS, _EVAL_CFG, _EVAL_DATA_CACHE
    p = dict(zip(_EVAL_PARAM_KEYS, combo))
    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(_EVAL_DATA_CACHE, p, _EVAL_CFG, tl, ec)
    r = compute_risk_metrics(ec) or {}
    s = r.get("sharpe_ratio")
    return (s if s is not None and not (isinstance(s, float) and np.isnan(s)) else -1e10), p


def run_full_backtest(
    data: pd.DataFrame,
    params: dict[str, Any],
    engine_cfg: EngineConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """用给定参数在整个数据集上跑一遍回测，返回 (trade_log, equity_curve)。"""
    if engine_cfg is None:
        engine_cfg = EngineConfig()
    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(data, params, engine_cfg, tl, ec)
    return tl, ec


def _run_single_backtest(
    data: pd.DataFrame,
    params: dict[str, Any],
    engine_cfg: EngineConfig,
    trade_log: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
) -> float:
    cash = engine_cfg.initial_cash
    positions: dict[str, float] = {}

    # 预分组：一次 O(n) 扫描，后续 O(1) 查找（避免每根 bar 扫全表）
    _day_map: dict[pd.Timestamp, pd.DataFrame] = {
        dt: grp.copy() for dt, grp in data.groupby("trade_date")
    }
    dates = sorted(_day_map.keys())

    # Point-in-time: 每个股票首次出现的日期
    _first_date: dict[str, pd.Timestamp] | None = None
    if engine_cfg.point_in_time:
        _first_date = data.groupby("symbol")["trade_date"].min().to_dict()

    buy_signal_col = "进场评分"
    sell_signal_col = "退出评分"
    risk_col = "风险等级"
    stop_loss_col = "止损价"

    # 用于组合优化的累积历史数据
    history: list[pd.DataFrame] = []

    for dt in dates:
        day_data = _day_map[dt].copy()
        if _first_date is not None:
            day_data = day_data[
                day_data["symbol"].apply(lambda s: _first_date.get(str(s), dt) <= dt)
            ]
            if day_data.empty:
                continue
        history.append(day_data)

        total_value = cash + sum(positions.values())

        # 出场
        for _, row in day_data.iterrows():
            sym = str(row.get("symbol", ""))
            if not sym or sym not in positions:
                continue

            exit_score = float(row.get(sell_signal_col, 0))
            entry_score = float(row.get(buy_signal_col, 0))
            risk_str = str(row.get(risk_col, "LOW")).upper()
            stop_loss = float(row.get(stop_loss_col, 0) or 0)
            close = float(row["close"])

            should_sell = (
                risk_str in ("HIGH", "D")
                or (exit_score > entry_score and exit_score > 0)
                or (stop_loss > 0 and close < stop_loss)
            )
            if should_sell:
                sell_value = positions.pop(sym)
                proceeds = sell_value * (1 - engine_cfg.slippage - engine_cfg.stamp_tax_rate)
                cash += proceeds
                trade_log.append({
                    "time": dt, "symbol": sym, "action": "sell",
                    "price": close, "value": round(proceeds, 2),
                    "cost": round(sell_value * (engine_cfg.slippage + engine_cfg.stamp_tax_rate), 2),
                })

        # 入场 — 使用组合优化器分配权重
        candidates = day_data[
            (day_data[buy_signal_col] >= 60)
            & (~day_data["symbol"].isin(positions))
            & (~day_data[risk_col].isin(["HIGH", "D", "E"]))
        ].copy()
        if not candidates.empty and engine_cfg.portfolio_method != "score_weighted":
            hist_df = pd.concat(history, ignore_index=True)
            weights = allocate_weights(
                hist_df, method=engine_cfg.portfolio_method,
                max_weight=engine_cfg.max_position_pct,
                entry_col=buy_signal_col, risk_col=risk_col,
            )
        else:
            risk_map = {"NONE": 1.0, "LOW": 1.5, "MEDIUM": 3.0, "HIGH": 5.0, "D": 8.0}
            weights = {}
            for _, r in candidates.iterrows():
                risk = risk_map.get(str(r.get(risk_col, "MEDIUM")).upper(), 3.0)
                weights[str(r["symbol"])] = min(1.0 / risk, engine_cfg.max_position_pct)

        total_weight = sum(weights.values()) or 1.0
        for _, row in day_data.iterrows():
            sym = str(row.get("symbol", ""))
            if sym not in weights or sym in positions:
                continue
            target_weight = weights[sym] / total_weight
            target_value = total_value * target_weight
            cost = target_value * (engine_cfg.commission_rate + engine_cfg.slippage)
            spend = target_value + cost
            if cash >= spend:
                cash -= spend
                positions[sym] = target_value
                trade_log.append({
                    "time": dt, "symbol": sym, "action": "buy",
                    "price": float(row["close"]), "value": round(target_value, 2), "cost": round(cost, 2),
                })

        total_value = cash + sum(positions.values())
        equity_curve.append({"time": dt, "portfolio_value": round(total_value, 2)})

    final_value = cash + sum(positions.values())
    total_return = (final_value / engine_cfg.initial_cash) - 1
    return total_return


def walk_forward(
    data: pd.DataFrame,
    engine_cfg: EngineConfig | None = None,
    train_period: int = 120,
    test_period: int = 60,
    param_grid: dict[str, list[Any]] | None = None,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    if engine_cfg is None:
        engine_cfg = EngineConfig()

    if param_grid is None:
        param_grid = {
            "atr_stop_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
            "kelly_fraction": [0.1, 0.25, 0.5],
        }

    dates = sorted(data["trade_date"].unique())
    n = len(dates)
    if n < train_period + test_period:
        raise ValueError(f"数据不足: {n} 个交易日，需要至少 {train_period + test_period}")

    windows: list[tuple[int, int]] = []
    start = 0
    while start + train_period + test_period <= n:
        windows.append((start, start + train_period))
        start += test_period

    clear_walk_forward_results()
    results: list[dict[str, Any]] = []

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())

    from tqdm import tqdm as _tqdm
    loop_iter = _tqdm(windows, desc="Walk-Forward", ncols=80) if show_progress else windows

    for win_idx, (train_start, train_end) in enumerate(loop_iter):
        test_start_idx = train_end
        test_end_idx = min(test_start_idx + test_period, n)

        train_dates_list = dates[train_start:train_end]
        test_dates_list = dates[test_start_idx:test_end_idx]

        train_data = data[data["trade_date"].isin(train_dates_list)]
        test_data = data[data["trade_date"].isin(test_dates_list)]

        if train_data.empty or test_data.empty:
            continue

        # Grid search on train (多进程并行)
        from concurrent.futures import ProcessPoolExecutor, as_completed

        best_sharpe = -1e10
        best_combo = {}
        _gcombos = list(itertools.product(*param_values))

        # 写入临时 parquet，各进程独立读取（避免 pickle 序列化 ~1GB DataFrame）
        _tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        _tmp_path = _tmp.name
        _tmp.close()
        train_data.to_parquet(_tmp_path)

        _gdesc = f"窗口{win_idx+1}网格搜索"
        try:
            with ProcessPoolExecutor(max_workers=3, initializer=_init_eval_worker, initargs=(_tmp_path, param_keys, engine_cfg)) as pool:
                futures = {pool.submit(_eval_one_combo, c): c for c in _gcombos}
                _giter = _tqdm(as_completed(futures), desc=_gdesc, ncols=80, total=len(_gcombos), leave=False) if show_progress else as_completed(futures)
                for f in _giter:
                    s, p = f.result()
                    if s > best_sharpe:
                        best_sharpe, best_combo = s, p
        finally:
            try:
                os.unlink(_tmp_path)
            except Exception:
                pass

        # Evaluate on test
        tl_test: list[dict[str, Any]] = []
        ec_test: list[dict[str, Any]] = []
        test_ret = _run_single_backtest(test_data, best_combo, engine_cfg, tl_test, ec_test)
        test_risk = compute_risk_metrics(ec_test) or {}
        test_trade = compute_trade_metrics(tl_test) or {}

        result = {
            "window": win_idx,
            "train_start": train_dates_list[0],
            "train_end": train_dates_list[-1],
            "test_start": test_dates_list[0],
            "test_end": test_dates_list[-1],
            "params": best_combo,
            "train_sharpe": best_sharpe,
            "total_return": test_ret,
            "sharpe_ratio": test_risk.get("sharpe_ratio", 0),
            "sortino_ratio": test_risk.get("sortino_ratio", 0),
            "max_drawdown": test_risk.get("max_drawdown", 0),
        }
        results.append(result)
        _WALK_FORWARD_RESULTS.append(result)

    return results


def _eval_combo_grid(combo: tuple[Any, ...]) -> dict[str, Any]:
    global _EVAL_PARAM_KEYS, _EVAL_CFG, _EVAL_DATA_CACHE
    p = dict(zip(_EVAL_PARAM_KEYS, combo))
    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    ret = _run_single_backtest(_EVAL_DATA_CACHE, p, _EVAL_CFG, tl, ec)
    risk = compute_risk_metrics(ec) or {}
    trade = compute_trade_metrics(tl) or {}
    sr = risk.get("sharpe_ratio")
    sr = sr if sr is not None and not (isinstance(sr, float) and np.isnan(sr)) else -1e10
    return {**p, "total_return": ret, "sharpe_ratio": sr, "max_drawdown": risk.get("max_drawdown", 0)}


def grid_search(
    data: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    engine_cfg: EngineConfig | None = None,
    show_progress: bool = False,
    max_workers: int = 3,
) -> list[dict[str, Any]]:
    if engine_cfg is None:
        engine_cfg = EngineConfig()

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm as _tqdm

    _tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    _tmp_path = _tmp.name
    _tmp.close()
    data.to_parquet(_tmp_path)

    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_eval_worker, initargs=(_tmp_path, param_keys, engine_cfg)) as pool:
            futures = {pool.submit(_eval_combo_grid, c): c for c in combos}
            iterator = _tqdm(as_completed(futures), desc="Grid Search", ncols=80, total=len(combos)) if show_progress else as_completed(futures)
            for f in iterator:
                results.append(f.result())
    finally:
        try:
            os.unlink(_tmp_path)
        except Exception:
            pass

    return results
