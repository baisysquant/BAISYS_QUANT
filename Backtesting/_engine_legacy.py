from __future__ import annotations

import itertools
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from typing_extensions import TypeAlias

from Backtesting.analytics import compute_risk_metrics, compute_trade_metrics
from Backtesting.domain.models import CostModel
from Backtesting.prepare import prepare_backtest_data, _compute_param_hash
from Backtesting.engine import EngineConfig
from Backtesting.portfolio import allocate_weights

from ConfigParser import Config as _Config

ParamsDict: TypeAlias = dict[str, Any]
TradeLog: TypeAlias = list[dict[str, Any]]
EquityCurve: TypeAlias = list[dict[str, Any]]


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


_WALK_FORWARD_RESULTS: list[dict[str, Any]] = []


def get_walk_forward_results() -> list[dict[str, Any]]:
    return list(_WALK_FORWARD_RESULTS)


def clear_walk_forward_results() -> None:
    _WALK_FORWARD_RESULTS.clear()


# 模块级函数（Windows 下 ProcessPoolExecutor 需要可 pickle）
_EVAL_PARAM_KEYS: list[str] = []
_EVAL_CFG: EngineConfig | None = None
_EVAL_DATA_CACHE: pd.DataFrame | None = None


def _init_eval_worker(filepath: str, param_keys: list[str], cfg: EngineConfig | None = None) -> None:
    global _EVAL_PARAM_KEYS, _EVAL_CFG, _EVAL_DATA_CACHE
    _EVAL_PARAM_KEYS = param_keys
    _EVAL_CFG = cfg or EngineConfig()
    _EVAL_DATA_CACHE = pd.read_parquet(filepath)


def _eval_one_combo(combo: tuple[Any, ...]) -> tuple[float, dict[str, Any]]:
    global _EVAL_PARAM_KEYS, _EVAL_CFG, _EVAL_DATA_CACHE
    assert _EVAL_DATA_CACHE is not None and _EVAL_CFG is not None  # 由 _init_eval_worker 保证
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
        # 如果没有提供 engine_cfg，需要从某处获取 Config，这里保留默认行为
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
    cash = engine_cfg.initial_cash
    positions: dict[str, float] = {}

    # 预分组：一次 O(n) 扫描，后续 O(1) 查找（避免每根 bar 扫全表）
    _day_map: dict[pd.Timestamp, pd.DataFrame] = {
        dt: grp.copy() for dt, grp in data.groupby("trade_date")  # type: ignore[misc]
    }
    dates = sorted(_day_map.keys())

    # Point-in-time: 每个股票首次出现的日期
    _first_date: dict[str, pd.Timestamp] | None = None
    if engine_cfg.point_in_time:
        _first_date = data.groupby("symbol")["trade_date"].min().to_dict()  # type: ignore[assignment]

    # 预计算 ADV（日均成交量），供 CostModel 大单冲击成本使用
    cm = engine_cfg.cost_model
    adv_map: dict[str, float] = {}
    if cm is not None:
        adv_map = data.groupby("symbol")["volume"].mean().to_dict()  # type: ignore[assignment]

    buy_signal_col = "进场评分"
    sell_signal_col = "退出评分"
    risk_col = "风险等级"
    stop_loss_col = "止损价"

    # 用于组合优化的累积历史数据
    history: list[pd.DataFrame] = []

    def _sell_proceeds(sym: str, value: float, volume: float) -> tuple[float, float]:
        if cm is not None:
            adv = adv_map.get(sym, 0)
            slip = cm.calc_slippage(volume, adv, side="sell", order_type="market")
            stamp = cm.stamp_tax_rate
            total_rate = slip + stamp
            proceeds = value * (1 - total_rate)
            cost = value * total_rate
        else:
            proceeds = value * (1 - engine_cfg.slippage - engine_cfg.stamp_tax_rate)
            cost = value * (engine_cfg.slippage + engine_cfg.stamp_tax_rate)
        return proceeds, cost

    def _buy_cost(sym: str, value: float, volume: float) -> float:
        if cm is not None:
            adv = adv_map.get(sym, 0)
            return cm.buy_cost(value, volume, adv)
        return value * (engine_cfg.commission_rate + engine_cfg.slippage)

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
                volume = float(row.get("volume", 0))
                proceeds, cost = _sell_proceeds(sym, sell_value, volume)
                cash += proceeds
                trade_log.append({
                    "time": dt, "symbol": sym, "action": "sell",
                    "price": close, "value": round(proceeds, 2),
                    "cost": round(cost, 2),
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
            volume = float(row.get("volume", 0))
            cost = _buy_cost(sym, target_value, volume)
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


def _eval_wf_combo(
    data_path: str,
    train_dates: list,
    combo_dict: dict[str, Any],
    engine_cfg: EngineConfig,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a single param combo on training data (ProcessPoolExecutor worker)."""
    data = pd.read_parquet(data_path)
    train_data = data[data["trade_date"].isin(train_dates)]
    if train_data.empty:
        return -1e10, combo_dict
    tl: TradeLog = []
    ec: EquityCurve = []
    _run_single_backtest(train_data, combo_dict, engine_cfg, tl, ec)
    r = compute_risk_metrics(ec) or {}
    s = r.get("sharpe_ratio")
    score = s if s is not None and not (isinstance(s, float) and np.isnan(s)) else -1e10
    return score, combo_dict


def walk_forward(
    data: pd.DataFrame,
    engine_cfg: EngineConfig | None = None,
    config: _Config | None = None,
    train_period: int = 120,
    test_period: int = 60,
    param_grid: dict[str, list[Any]] | None = None,
    show_progress: bool = False,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    if engine_cfg is None:
        if config is not None:
            engine_cfg = EngineConfig.from_config(config)
        else:
            engine_cfg = EngineConfig()

    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4))

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
    param_combos = list(itertools.product(*param_values))

    # ── 检查数据是否已包含信号列，决定是否需要预计算 ──
    required_signal_cols = {"进场评分", "退出评分", "风险等级", "止损价"}
    has_signals = required_signal_cols.issubset(set(data.columns))

    if has_signals:
        logger.info("输入数据已包含信号列，直接使用，跳过信号预计算")
        prepared_data_by_hash = None
        combo_to_hash = None
    else:
        # ── 预准备所有参数组合的信号数据（带缓存，仅首次计算慢） ──
        logger.info(f"预准备 {len(param_combos)} 个参数组合的信号数据...")
        prepared_data_by_hash = {}
        combo_to_hash = {}

        for combo in param_combos:
            combo_dict = dict(zip(param_keys, combo))
            p_hash = _compute_param_hash(combo_dict)
            combo_to_hash[combo] = p_hash
            if p_hash not in prepared_data_by_hash:
                logger.info(f"准备参数组合 {combo_dict} [hash={p_hash}] 的信号数据...")
                prepared_data_by_hash[p_hash] = prepare_backtest_data(data, params=combo_dict, signal_param_hash=p_hash, compute_exit_strategy=True)

    # ── 将每个参数组合的完整数据写入临时文件，供 worker 进程读取 ──
    tmp_combo_dir = tempfile.mkdtemp(prefix="wf_combos_")
    combo_data_paths: list[str] = []
    try:
        for i, combo in enumerate(param_combos):
            if has_signals:
                prepared = data
            else:
                p_hash = combo_to_hash[combo]
                prepared = prepared_data_by_hash[p_hash]
            path = os.path.join(tmp_combo_dir, f"combo_{i}.parquet")
            prepared.to_parquet(path, compression="zstd", compression_level=1)
            combo_data_paths.append(path)

        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm as _tqdm

        loop_iter = _tqdm(windows, desc="Walk-Forward", ncols=80) if show_progress else windows

        for win_idx, (train_start, train_end) in enumerate(loop_iter):
            test_start_idx = train_end
            test_end_idx = min(test_start_idx + test_period, n)

            clear_walk_forward_results()
            train_dates_list = list(dates[train_start:train_end])

            # ── IS 评估：并行计算所有参数组合的 Sharpe ──
            scored_combos: list[tuple[float, dict[str, Any]]] = []

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for i, combo in enumerate(param_combos):
                    combo_dict = dict(zip(param_keys, combo))
                    future = pool.submit(
                        _eval_wf_combo,
                        combo_data_paths[i],
                        train_dates_list,
                        combo_dict,
                        engine_cfg,
                    )
                    futures[future] = combo_dict

                combo_iter = _tqdm(as_completed(futures), desc=f"  Win-{win_idx} combos", ncols=80, total=len(futures)) if show_progress else as_completed(futures)
                for f in combo_iter:
                    try:
                        score, params_dict = f.result()
                        scored_combos.append((score, params_dict))
                    except Exception:
                        continue

            if not scored_combos:
                continue

            # 按 IS Sharpe 降序排列
            scored_combos.sort(key=lambda x: x[0], reverse=True)
            best_sharpe, best_combo = scored_combos[0]

            # ── 准备 OOS 数据 ──
            if has_signals:
                prepared_full = data
            else:
                best_hash = combo_to_hash.get(tuple(best_combo.items()))
                if best_hash is None:
                    continue
                prepared_full = prepared_data_by_hash[best_hash]

            test_dates_list = dates[train_end:min(train_end + test_period, n)]
            test_data = prepared_full[prepared_full["trade_date"].isin(test_dates_list)]
            if test_data.empty:
                continue

            # ── OOS 评估：对 top-M 组合分别跑回测 ──
            oos_results: list[dict[str, Any]] = []
            top_m = min(5, len(scored_combos))
            for rank_idx in range(top_m):
                is_score, params = scored_combos[rank_idx]
                tl_t: list[dict[str, Any]] = []
                ec_t: list[dict[str, Any]] = []
                _run_single_backtest(test_data, params, engine_cfg, tl_t, ec_t)
                risk_t = compute_risk_metrics(ec_t) or {}
                sr = risk_t.get("sharpe_ratio")
                sr = sr if sr is not None and not (isinstance(sr, float) and np.isnan(sr)) else None
                oos_results.append({
                    "params": params,
                    "is_rank": rank_idx + 1,
                    "is_sharpe": is_score,
                    "oos_sharpe": sr,
                })

            # 用 #1 组合构建主结果（向后兼容 runner.py 的 _extract_best_params）
            best_oos = oos_results[0]
            tl_test: list[dict[str, Any]] = []
            ec_test: list[dict[str, Any]] = []
            _run_single_backtest(test_data, best_combo, engine_cfg, tl_test, ec_test)
            test_risk = compute_risk_metrics(ec_test) or {}
            test_trade = compute_trade_metrics(tl_test) or {}

            results.append({
                "window": win_idx,
                "train_start": dates[train_start],
                "train_end": dates[train_end - 1],
                "test_start": test_dates_list[0] if test_dates_list else "",
                "test_end": test_dates_list[-1] if test_dates_list else "",
                "params": best_combo,
                "train_sharpe": best_sharpe,
                "sharpe_ratio": best_oos["oos_sharpe"] if best_oos["oos_sharpe"] is not None else 0,
                "total_return": test_risk.get("total_return", 0),
                "max_drawdown": test_risk.get("max_drawdown", 0),
                "annual_return": test_risk.get("annual_return", 0),
                "annual_vol": test_risk.get("annual_vol", 0),
                "var_95": test_risk.get("var_95", 0),
                "cvar_95": test_risk.get("cvar_95", 0),
                "win_rate": test_trade.get("win_rate", 0),
                "profit_factor": test_trade.get("profit_factor", 0),
                "total_trades": test_trade.get("total_trades", 0),
                "num_combos": len(param_combos),
                "oos_combos": oos_results,
            })

    finally:
        import shutil
        shutil.rmtree(tmp_combo_dir, ignore_errors=True)

    return results


def _eval_combo_grid(combo: tuple[Any, ...]) -> dict[str, Any]:
    global _EVAL_PARAM_KEYS, _EVAL_CFG, _EVAL_DATA_CACHE
    assert _EVAL_DATA_CACHE is not None and _EVAL_CFG is not None
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
    max_workers: int = 2,
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
