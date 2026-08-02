from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading._engine_legacy import EngineConfig, _run_single_backtest
from BackTrading.bayesian.kernel import GPState
from BackTrading.bayesian.optimizer import optimize_window
from BackTrading.bayesian.space import ParamSpace, build_spaces, split_by_cost
from BackTrading.bayesian.transfer import warm_start_gp
from BackTrading.prepare import prepare_backtest_data


def _datetime64_to_string_guard(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df = df.copy()
        df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    elif df["trade_date"].dtype == object:
        # 已经是字符串，确保格式统一
        sample = str(df["trade_date"].iloc[0]) if len(df) > 0 else ""
        if "T" in sample or len(sample) > 10:
            df = df.copy()
            df["trade_date"] = df["trade_date"].apply(
                lambda x: str(x).split("T")[0][:10] if pd.notna(x) else x
            )
    return df


def _to_date_str(d) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    d_str = str(d)
    return d_str.split("T")[0][:10] if "T" in d_str else d_str[:10]


# 信号上下文预热天数：MACD/ATR/MA 等指标至少需要 120 个交易日的前文，
# 训练/OOS 切片前额外补足这段历史，保证窗口首日起信号有效。
_SIGNAL_WARMUP_DAYS = 120


def _window_dates(
    unique_dates: list,
    train_period: int,
    test_period: int,
    offset: int = 0,
) -> list[tuple[int, int, int, int]]:
    """生成 WFO 窗口切分。

    Returns:
        [(train_start, train_end, test_start, test_end), ...] 索引元组。
    """
    windows = []
    start = offset
    n = len(unique_dates)
    while start + train_period + test_period <= n:
        train_end = start + train_period
        test_start = train_end
        test_end = min(test_start + test_period, n)
        windows.append((start, train_end, test_start, test_end))
        start += test_period
    return windows


def _oos_validate(
    test_data: pd.DataFrame,
    params_list: list[dict[str, float]],
    engine_cfg: EngineConfig,
    top_m: int = 5,
    eval_start_date: str | None = None,
) -> list[dict[str, Any]]:
    """对 top-M 参数组合做 OOS 验证。

    eval_start_date: 信号预热历史已并入 test_data 时，用该日期截断，
    保证回测引擎只交易 [eval_start_date, 末尾] 的 OOS 区间。
    """
    from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics

    _SIGNAL_KEYS = frozenset({
        "boll_narrow_ratio", "cross_decay_days",
        "golden_cross_bonus", "divergence_penalty", "conclusion_full_bull",
    })

    oos_results = []
    for rank_idx, params in enumerate(params_list[:top_m]):
        tl: list[dict[str, Any]] = []
        ec: list[dict[str, Any]] = []
        
        # 使用优化后的组合参数构建引擎配置（而非 base_cfg 默认值）
        from dataclasses import replace as _replace_dc
        _ec = _replace_dc(engine_cfg,
            atr_stop_mult=params.get("atr_stop_mult", engine_cfg.atr_stop_mult),
            buy_threshold=int(params.get("buy_threshold", engine_cfg.buy_threshold)),
            max_holdings=int(params.get("max_holdings", engine_cfg.max_holdings)),
        )

        signal_params = {k: v for k, v in params.items() if k in _SIGNAL_KEYS}
        if signal_params:
            _prepared = prepare_backtest_data(
                test_data, params=signal_params, compute_exit_strategy=True,
                vectorized=True, backtest_start_date=eval_start_date,
            )
        else:
            _prepared = test_data
            # 无信号参数时也按 eval_start_date 截断，避免预热段产生交易
            if eval_start_date and "trade_date" in _prepared.columns:
                _prepared = _prepared[_prepared["trade_date"] >= eval_start_date]
        _run_single_backtest(_prepared, params, _ec, tl, ec)
        risk = compute_risk_metrics(ec) or {}
        trade = compute_trade_metrics(tl) or {}
        sr = risk.get("sharpe_ratio")
        sr = sr if sr is not None and not (isinstance(sr, float) and np.isnan(sr)) else None
        oos_results.append({
            "params": params,
            "is_rank": rank_idx + 1,
            "oos_sharpe": sr,
            "total_return": risk.get("total_return", 0),
            "max_drawdown": risk.get("max_drawdown", 0),
            "annual_return": risk.get("annual_return", 0),
            "annual_vol": risk.get("annual_vol", 0),
            "var_95": risk.get("var_95", 0),
            "cvar_95": risk.get("cvar_95", 0),
            "win_rate": trade.get("win_rate", 0),
            "profit_factor": trade.get("profit_factor", 0),
            "total_trades": trade.get("total_trades", 0),
        })
    return oos_results


def bayesian_walk_forward_multi(
    kline_df: pd.DataFrame,
    param_grid: dict | None = None,    # 保留签名兼容，不再使用
    train_period: int = 120,
    test_period: int = 20,
    num_paths: int = 3,
    initial_cash: float = 1_000_000.0,
    spaces: dict[str, ParamSpace] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """贝叶斯 WFO 主入口 — 多路径 + 多窗口 + 跨窗口迁移学习。

    Args:
        kline_df: K 线数据。
        param_grid: 保留签名兼容，不再使用（由 spaces 替代）。
        train_period: IS 窗口大小（交易日）。
        test_period: OOS 窗口大小。
        num_paths: 多路径数。
        initial_cash: 初始资金。
        spaces: 预构建 ParamSpace（None 时自动从 config 解析）。
        **kwargs: 透传给 EngineConfig 的额外参数（commission, slippage 等）。

    Returns:
        DataFrame，每行一个窗口，与旧 walk_forward 格式兼容。
    """
    from UtilsManager.ConfigParser import Config as _Config

    if spaces is None:
        cfg = _Config().app_config.backtest
        spaces = build_spaces(cfg)

    signal_sp, portfolio_sp = split_by_cost(spaces)

    # ── 构建基座 EngineConfig ──
    base_cfg = EngineConfig(
        initial_cash=initial_cash,
        commission_rate=kwargs.get("commission", 0.0003),
        stamp_tax_rate=kwargs.get("stamp_tax", 0.0005),
        slippage=kwargs.get("slippage", 0.001),
        max_position_pct=kwargs.get("max_position_pct", 0.1),
        portfolio_method=kwargs.get("portfolio_method", "score_weighted"),
        point_in_time=kwargs.get("point_in_time", True),
    )

    # 从 config 读取 BO 预算
    try:
        bt_cfg = _Config().app_config.backtest
        n_init_signal = bt_cfg.BAYESIAN_N_INIT_SIGNAL
        n_iter_signal = bt_cfg.BAYESIAN_N_ITER_SIGNAL
        n_init_portfolio = bt_cfg.BAYESIAN_N_INIT_PORTFOLIO
        n_iter_portfolio = bt_cfg.BAYESIAN_N_ITER_PORTFOLIO
    except Exception:
        n_init_signal, n_iter_signal = 15, 35
        n_init_portfolio, n_iter_portfolio = 20, 150

    # 窗口切分以"正式回测起点"为坐标轴；起点之前为信号预热历史（不参与交易）
    backtest_start_date = kwargs.get("backtest_start_date")
    all_dates_str = sorted({_to_date_str(d) for d in kline_df["trade_date"].unique()})
    if backtest_start_date:
        _cut = backtest_start_date[:10]
        unique_dates = [d for d in all_dates_str if d >= _cut]
    else:
        unique_dates = all_dates_str
    _trim_offset = len(all_dates_str) - len(unique_dates)
    n_dates = len(unique_dates)
    logger.info(f"  n_dates={n_dates}, train_period={train_period}, test_period={test_period}, required={train_period + test_period}")
    if n_dates < train_period + test_period:
        raise ValueError(f"数据不足: {n_dates} 个交易日（正式回测起点后），需要至少 {train_period + test_period}")
    show_progress = kwargs.get("show_progress", False)

    # ── 多路径收集 ──
    all_path_results: dict[int, list[dict[str, Any]]] = {}  # window_id → [path_results]
    _K_FOR_OOS = 3  # OOS 验证 top-K 参数数，供 PBO 计算

    for path_idx in range(num_paths):
        # 确定性偏移：各路径互不重叠，避免 IS/OOS 数据泄露
        offset = path_idx * test_period
        max_offset = max(0, n_dates - train_period - test_period)
        if offset > max_offset:
            logger.warning(f"路径 {path_idx + 1} offset={offset} 超出数据范围 (max={max_offset})，跳过")
            continue

        windows = _window_dates(unique_dates, train_period, test_period, offset)
        logger.info(f"路径 {path_idx + 1}/{num_paths}: offset={offset}, 窗口数={len(windows)}")

        if not windows:
            logger.warning(f"路径 {path_idx + 1} 无有效窗口，检查 n_dates({n_dates}) >= train({train_period})+test({test_period})?")
            continue

        previous_gp_state: GPState | None = None

        for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            train_dates = unique_dates[tr_s:tr_e]
            test_dates = unique_dates[te_s:te_e]

            # ── datetime guard ──
            _df = _datetime64_to_string_guard(kline_df.copy())
            # 统一使用字符串格式进行匹配
            train_dates_str = [_to_date_str(d) for d in train_dates]
            test_dates_str = [_to_date_str(d) for d in test_dates]
            # ── 信号上下文预热：窗口起点前补 _SIGNAL_WARMUP_DAYS 天历史 ──
            # 指标（MACD/ATR/MA）与 ML 需要前文；预热行仅用于信号计算，
            # 通过 eval_start_date 在 prepare/引擎侧截断，不参与交易。
            # 关键：训练切片止于训练期末（tr_e），绝不包含 OOS 区间，
            # 否则训练期目标函数混入样本外收益，OOS/PBO/DSR 全部失真。
            ctx_s_full = _trim_offset + max(0, tr_s - _SIGNAL_WARMUP_DAYS)
            train_mask = _df["trade_date"].isin(all_dates_str[ctx_s_full:_trim_offset + tr_e])
            train_data = _df[train_mask]
            ctx_s_test = _trim_offset + max(0, te_s - _SIGNAL_WARMUP_DAYS)
            test_mask = _df["trade_date"].isin(all_dates_str[ctx_s_test:_trim_offset + te_e])
            test_data = _df[test_mask]
            train_eval_start = train_dates_str[0]
            test_eval_start = test_dates_str[0]

            if train_data.empty:
                logger.warning(f"  [{path_idx + 1}-{win_idx}] 训练数据为空(train_dates={train_dates[0]}~{train_dates[-1]}), 跳过")
                continue

            if show_progress:
                logger.info(f"  [{path_idx + 1}-{win_idx}] 优化窗口: "
                           f"{_to_date_str(train_dates[0])}~{_to_date_str(train_dates[-1])}")
            logger.info(f"  [{path_idx + 1}-{win_idx}] 训练数据: {len(train_data)}行, {train_data['symbol'].nunique()}只股票")

            # ── 贝叶斯优化 IS ──
            try:
                best_params, gp_state, top_k_params = optimize_window(
                    kline_df=train_data,
                    engine_cfg=base_cfg,
                    spaces=spaces,
                    n_init_signal=n_init_signal,
                    n_iter_signal=n_iter_signal,
                    n_init_portfolio=n_init_portfolio,
                    n_iter_portfolio=n_iter_portfolio,
                    previous_gp_state=previous_gp_state,
                    seed=42 + path_idx * 100 + win_idx,
                    eval_start_date=train_eval_start,
                )
            except Exception as opt_err:
                logger.opt(exception=True).warning(f"  [{path_idx + 1}-{win_idx}] 窗口优化失败: {opt_err}")
                continue

            # ── 跨窗口迁移 ──
            n_signal = len(signal_sp)
            previous_gp_state = warm_start_gp(gp_state, n_signal)

            # ── OOS 验证（top-K 参数，PBO 需要多组 OOS 结果） ──
            if not test_dates:
                logger.warning(f"  [{path_idx + 1}-{win_idx}] OOS 区间为空, 跳过 OOS 验证")
                continue

            oos_params = top_k_params[:min(_K_FOR_OOS, len(top_k_params))]
            try:
                oos_results = _oos_validate(
                    test_data,
                    oos_params,
                    base_cfg,
                    top_m=len(oos_params),
                    eval_start_date=test_eval_start,
                )
            except Exception as oos_err:
                logger.opt(exception=True).warning(f"  [{path_idx + 1}-{win_idx}] OOS 验证失败: {oos_err}")
                continue

            # ── 计算 OOS 指标 ──
            oos = oos_results[0] if oos_results else {}
            oos_sharpe = oos.get("oos_sharpe", 0) or 0

            # ── 收集 IS Sharpe ──
            train_sharpe = 0.0

            entry: dict[str, Any] = {
                "window": win_idx,
                "train_start": _to_date_str(train_dates[0]),
                "train_end": _to_date_str(train_dates[-1]),
                "test_start": _to_date_str(test_dates[0]) if test_dates else "",
                "test_end": _to_date_str(test_dates[-1]) if test_dates else "",
                "params": best_params,
                "train_sharpe": train_sharpe,
                "sharpe_ratio": oos_sharpe,
                "total_return": oos.get("total_return", 0),
                "max_drawdown": oos.get("max_drawdown", 0),
                "annual_return": oos.get("annual_return", 0),
                "annual_vol": oos.get("annual_vol", 0),
                "var_95": oos.get("var_95", 0),
                "cvar_95": oos.get("cvar_95", 0),
                "win_rate": oos.get("win_rate", 0),
                "profit_factor": oos.get("profit_factor", 0),
                "total_trades": oos.get("total_trades", 0),
                "num_combos": n_init_signal + n_iter_signal + n_init_portfolio + n_iter_portfolio,
                "oos_combos": oos_results,
            }
            all_path_results.setdefault(win_idx, []).append(entry)

    # ── 按窗口聚合（多路径取中位数） ──
    if not all_path_results:
        logger.warning("所有路径均无有效窗口，使用配置中位数兜底")
        mid = {n: (sp.low + sp.high) / 2 for n, sp in spaces.items()}
        return pd.DataFrame([{
            "window": 0, "params": mid, "sharpe_ratio": 0.0,
            "num_combos": 1, "num_paths": num_paths,
        }])

    agg_rows: list[dict[str, Any]] = []
    for win_id in sorted(all_path_results.keys()):
        entries = all_path_results[win_id]
        # 取第一个作为基础（结构相同）
        base = dict(entries[0])
        # 对 params 取中位数
        all_params = [e["params"] for e in entries if isinstance(e.get("params"), dict)]
        if len(all_params) > 1:
            keys = all_params[0].keys()
            median_params: dict[str, float] = {}
            for k in keys:
                vals = sorted(p[k] for p in all_params)
                median_params[k] = vals[len(vals) // 2]
            base["params"] = median_params
        # sharpe 取中位数
        sharpe_vals = sorted(e["sharpe_ratio"] for e in entries)
        base["sharpe_ratio"] = sharpe_vals[len(sharpe_vals) // 2] if sharpe_vals else 0.0
        base["num_paths"] = len(entries)
        agg_rows.append(base)

    result = pd.DataFrame(agg_rows)
    logger.info(f"贝叶斯 WFO 完成: {len(result)} 个窗口, {num_paths} 条路径")
    return result
