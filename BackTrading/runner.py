from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text

from BackTrading.alert import BacktestAlert
from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics
from BackTrading.calibration import (
    CALIB_PARAM_MAP,
    CalibrationResult,
    apply_calibration_to_config,
    load_calibration,
    run_bayesian_walk_forward as run_walk_forward,
    save_calibration,
    write_calibration_to_ini,
)
from BackTrading.calibration_log import ensure_table, get_last_run, record_run, should_rerun
from UtilsManager.IDataProvider import BacktestDataProvider
from BackTrading.prepare import _build_params, prepare_backtest_data
from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine
from sqlalchemy import text


_BACKTEST_LOCK_KEY = 987654321


def _acquire_lock(engine: Any) -> None:
    """获取回测分布式锁（pg_advisory_xact_lock + NOWAIT，失败则 exit）。"""
    from sqlalchemy import text as _t

    with engine.connect() as conn:
        locked = conn.execute(
            _t(f"SELECT pg_try_advisory_xact_lock({_BACKTEST_LOCK_KEY})")
        ).scalar()
        if locked:
            logger.info("  获取回测分布式锁成功")
        else:
            logger.warning("回测分布式锁被占用，跳过本次执行（可能有另一个进程正在运行）")
            sys.exit(0)


def run_backtest_pipeline(
    config: Config | None = None,
    force: bool = False,
) -> CalibrationResult | None:
    """月度回测管线入口。

    Args:
        config: Config 实例，为空时自动创建。
        force: 是否强制重新运行（忽略 enabled / 频率检查，跳过交互提示）。

    Returns:
        CalibrationResult 或 None（跳过时）。
    """
    if config is None:
        config = Config()

    cfg = config.app_config
    bt = cfg.backtest
    alert = BacktestAlert(config)

    if not force and not bt.ENABLED:
        logger.info("回测未启用 (BACKTEST.enabled=false)，跳过")
        return None

    engine = get_engine(config)
    ensure_table(engine)

    # ── 分布式锁（pg_advisory_xact_lock + NOWAIT 防止阻塞） ──
    _acquire_lock(engine)

    last = get_last_run(engine)
    should_run, reason = should_rerun(last, bt.OPTIMIZE_FREQUENCY)

    if not should_run and not force:
        logger.info(reason)
        answer = input(f"  {reason}。是否强制执行？(y/N): ").strip().lower()
        if answer != "y":
            logger.info("用户取消，跳过回测")
            return load_calibration()
        logger.info("用户确认，强制重新回测")

    logger.info("=" * 50)
    logger.info("开始回测管线 ...")
    logger.info(f"  优化频率: {bt.OPTIMIZE_FREQUENCY}")
    logger.info(f"  数据起始日期: {bt.BACKTEST_START_DATE}")
    logger.info(f"  样本外天数: {bt.OUT_OF_SAMPLE_DAYS}")
    logger.info(f"  初始资金: {bt.INITIAL_CASH:,.0f}")

    _step_times: dict[str, float] = {"start": time.time()}
    def _log_step(name: str) -> None:
        _step_times[name] = time.time()
        _elapsed = _step_times[name] - _step_times.get(list(_step_times.keys())[-2] if len(_step_times) >= 2 else "start", 0)
        _total = _step_times[name] - _step_times["start"]
        logger.info(f"[STEP] {name} ({_elapsed:.1f}s, 累计 {_total:.1f}s)")

    try:
        symbols = _resolve_symbols(engine, config)
        logger.info(f"  股票数量: {len(symbols)}")
        logger.warning("生存偏差: 股票池仅含当前存活股票，已退市/ST 股票的历史负收益未被计入")
        _log_step("resolve_symbols")

        kline_df = _fetch_kline(engine, symbols, bt.BACKTEST_START_DATE)
        if kline_df.empty:
            logger.warning("K 线数据为空，跳过回测")
            return None

        logger.info(f"  K 线行数: {len(kline_df)}")

        # 窗口坐标轴以正式回测起点为准（起点前为信号预热历史，不参与 WFO 交易）
        def _ds(d) -> str:
            return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        _bt_cut = f"{bt.BACKTEST_START_DATE[:4]}-{bt.BACKTEST_START_DATE[4:6]}-{bt.BACKTEST_START_DATE[6:8]}"
        total_trading_days = sum(1 for d in kline_df["trade_date"].unique() if _ds(d) >= _bt_cut)
        _oos = bt.OUT_OF_SAMPLE_DAYS
        # 数据自适应 WFO 配置：路径 p 的 offset = p*OOS，需满足 offset + IS + OOS <= n，
        # 否则路径 2/3 必然越界跳过（如 IS=805+OOS=60 在 865 天数据上只有 1 条路径有效）。
        _np_cfg = max(1, int(bt.WFO_NUM_PATHS))
        _max_np = max(1, (total_trading_days - 120) // _oos) if total_trading_days > _oos + 120 else 1
        _num_paths = min(_np_cfg, _max_np)
        train_period = max(120, min(total_trading_days - _oos, total_trading_days - _oos * _num_paths))
        logger.info(
            f"  交易日数: {total_trading_days} | IS训练窗口: {train_period}天 | OOS: {_oos}天"
            f" | WFO路径数: {_num_paths}（配置 {_np_cfg}，数据上限 {_max_np}）"
        )
        _log_step("fetch_kline")
        wf_result = run_walk_forward(
            kline_df=kline_df,
            num_paths=_num_paths,
            train_period=train_period,
            test_period=bt.OUT_OF_SAMPLE_DAYS,
            initial_cash=bt.INITIAL_CASH,
            commission=bt.COMMISSION_RATE,
            stamp_tax=bt.STAMP_TAX_RATE,
            slippage=bt.SLIPPAGE,
            max_position_pct=bt.MAX_POSITION_PCT,
            portfolio_method=bt.PORTFOLIO_METHOD,
            point_in_time=bt.POINT_IN_TIME,
            show_progress=True,
            backtest_start_date=_bt_cut,
        )
        _log_step("walk_forward")
        logger.info(f"  Walk-Forward 片段数: {len(wf_result)}")

        if not wf_result.empty and wf_result["sharpe_ratio"].max() > 3.0:
            logger.warning(f"akquant 结果异常: Sharpe={wf_result['sharpe_ratio'].max():.2f}>3.0，可能存在前瞻偏差")

        best_params = _extract_best_params(wf_result, config=config)
        logger.info(f"  最佳参数(Sharpe加权前{min(5, len(wf_result))}): {best_params}")

        from BackTrading._engine_legacy import EngineConfig, run_full_backtest
        from BackTrading.domain.models import CostModel

        from UtilsManager.ConfigParser import PositionSizingConfig as _PsCfg
        _ps: _PsCfg = config.app_config.position_sizing
        _sc = config.app_config.scoring_params
        # 组合参数若未被寻优（兜底路径），取配置区间中位，保证最终回测与校准参数一致
        _bt_mid = sum(bt.parse_range("BUY_THRESHOLD_RANGE")[:2]) / 2
        _mh_mid = sum(bt.parse_range("MAX_HOLDINGS_RANGE")[:2]) / 2
        ecfg = EngineConfig(
            initial_cash=bt.INITIAL_CASH,
            commission_rate=bt.COMMISSION_RATE,
            stamp_tax_rate=bt.STAMP_TAX_RATE,
            slippage=bt.SLIPPAGE,
            max_position_pct=bt.MAX_POSITION_PCT,
            portfolio_method=bt.PORTFOLIO_METHOD,
            point_in_time=bt.POINT_IN_TIME,
            atr_stop_mult=best_params.get("atr_stop_mult", _sc.ATR_STOP_MULT),
            buy_threshold=int(best_params.get("buy_threshold", _bt_mid)),
            max_holdings=int(best_params.get("max_holdings", _mh_mid)),
            cost_model=CostModel(
                commission_rate=bt.COMMISSION_RATE,
                stamp_tax_rate=bt.STAMP_TAX_RATE,
                market_slippage=bt.SLIPPAGE,
                min_commission_per_trade=bt.MIN_COMMISSION_PER_TRADE,
                transfer_fee_rate=bt.TRANSFER_FEE_RATE,
            ),
        )
        final_params = _build_params(config)
        final_params["scoring"].update({k: v for k, v in best_params.items() if k in ("atr_stop_mult", "cross_decay_days", "golden_cross_bonus", "divergence_penalty")})
        if "boll_narrow_ratio" in best_params:
            final_params["regime"]["boll_narrow_ratio"] = float(best_params["boll_narrow_ratio"])
        fb_cfg = config.app_config.full_bull_scoring
        final_params["thresholds"] = {
            "fully_bull": int(best_params.get("conclusion_full_bull", fb_cfg.CONCLUSION_FULL_BULL)),
            "bullish": fb_cfg.CONCLUSION_BULLISH,
            "oscillate": fb_cfg.CONCLUSION_OSCILLATE,
        }

        # ── 模拟交易验证：用最近交易日验证参数 OOS 稳定性 ──
        from BackTrading.simulated_trading import validate_params as _sim_validate
        _wf_sharpe = float(wf_result["sharpe_ratio"].mean()) if not wf_result.empty else 0.0
        _sim_verdict = _sim_validate(
            kline_df=kline_df, best_params=best_params,
            oos_sharpe=_wf_sharpe, sim_days=20,
            config=config, engine_cfg=ecfg,
        )
        _promote = _sim_verdict.promote
        if not _promote:
            logger.warning(f"模拟验证不通过，参数不写入 config.ini: {_sim_verdict.reason}")

        # 加载 ST 历史状态（用于逐日动态剔除）
        _bt_start_iso = datetime.strptime(bt.BACKTEST_START_DATE, "%Y%m%d").date().isoformat()
        _end_date = kline_df["trade_date"].max()
        if pd.api.types.is_datetime64_any_dtype(kline_df["trade_date"]):
            _end_date = _end_date.strftime("%Y-%m-%d")
        st_history = _load_st_history(engine, symbols, _bt_start_iso, _end_date)

        _log_step("prepare_final_signals")
        final_prepared = prepare_backtest_data(kline_df, params=final_params, compute_exit_strategy=True, vectorized=True, backtest_start_date=_bt_start_iso)
        _log_step("full_backtest")
        # 将 ST 历史传给引擎
        best_params["_st_history"] = st_history
        trade_log, equity_curve = run_full_backtest(final_prepared, best_params, ecfg)
        _log_step("compute_metrics")
        risk = compute_risk_metrics(equity_curve) or {}
        trade = compute_trade_metrics(trade_log) or {}

        logger.info(f"  ── 绩效分析 ──")
        logger.info(f"  Sharpe={risk.get('sharpe_ratio', 0):.2f} | Sortino={risk.get('sortino_ratio', 0):.2f} | Calmar={risk.get('calmar_ratio', 0):.2f}")
        logger.info(f"  VaR(95%)={risk.get('var_95', 0):.2%} | CVaR(95%)={risk.get('cvar_95', 0):.2%} | MaxDD={risk.get('max_drawdown', 0):.2%}")
        logger.info(f"  交易={trade.get('total_trades', 0)} | 胜率={trade.get('win_rate', 0):.1%} | 盈亏比={trade.get('profit_factor', 0):.2f}")
        logger.info(f"  日均换手率={risk.get('avg_turnover', 0):.2%} | 最高单日换手率={risk.get('max_turnover', 0):.2%}")
        _avg_to = risk.get("avg_turnover", 0)
        if _avg_to and _avg_to > 0.30:
            logger.warning(f"日均换手率 {_avg_to:.2%} > 30%，扣费后实际收益可能打 7 折")
        logger.info(f"  最佳参数(Sharpe加权前{min(5, len(wf_result))}): {best_params}")

        # ── 持仓打分卡：当期持仓的因子分解 ──
        try:
            _holdings = [t for t in trade_log if t.get("action") == "buy"][-20:]  # 最近 20 笔买入
            if _holdings and not final_prepared.empty:
                _last_date = final_prepared["trade_date"].max()
                if pd.api.types.is_datetime64_any_dtype(final_prepared["trade_date"]):
                    _fp = final_prepared.copy()
                    _fp["trade_date"] = _fp["trade_date"].dt.strftime("%Y-%m-%d")
                    _last_date_str = _last_date.strftime("%Y-%m-%d") if hasattr(_last_date, "strftime") else str(_last_date)
                    _latest = _fp[_fp["trade_date"] == _last_date_str]
                else:
                    _latest = final_prepared[final_prepared["trade_date"] == _last_date]
                _score_cols = ["MACD趋势分", "金叉信号分", "柱状动能分", "DIF斜评分",
                               "背离信号分", "量价配合分", "K线形态分"]
                _held_syms = list({t["symbol"] for t in _holdings if t["symbol"] in _latest["symbol"].values})
                if _held_syms:
                    _card = _latest[_latest["symbol"].isin(_held_syms)][
                        ["symbol", "进场评分", "综合评分", "风险等级"] + _score_cols
                    ].copy()
                    _card.columns = ["股票", "进场分", "综合分", "风险"] + [
                        "MACD趋势", "金叉", "动能", "DIF斜率", "背离", "量价", "K线"
                    ]
                    logger.info(f"  ── 持仓因子分解（{_last_date}）──")
                    for _, r in _card.iterrows():
                        _factors = " | ".join(f"{c}={r[c]:.0f}" for c in ["MACD趋势","金叉","动能","DIF斜率","背离","量价","K线"])
                        logger.info(f"    {r['股票']}: 综合{r['综合分']:.0f}/进场{r['进场分']:.0f}/{r['风险']} | {_factors}")
        except Exception:
            pass

        # ── 因子暴露归因 ──
        try:
            _ec_df = pd.DataFrame(equity_curve).set_index("time")
            _ec_df.index = pd.to_datetime(_ec_df.index)
            _port_rets = _ec_df["portfolio_value"].pct_change().dropna()
            if len(_port_rets) > 20:
                from BackTrading.attribution import factor_exposure as _fe
                # 用市场指数收益率作为因子代理
                _index_map = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}
                _factor_data = {}
                for _code, _name in _index_map.items():
                    try:
                        from UtilsManager.IDataProvider import BacktestDataProvider as _Bdp
                        from DataManager.DbEngine import get_engine as _ge
                        _e2 = _ge(config)
                        _p = _Bdp(_e2)
                        _idx = _p.get_index_kline(_code, start=_port_rets.index[0].strftime("%Y-%m-%d"))
                        if _idx is not None and not _idx.empty:
                            _idx = _idx.set_index("trade_date")
                            _idx.index = pd.to_datetime(_idx.index)
                            _factor_data[_name] = _idx["close"].pct_change()
                    except Exception:
                        continue
                if _factor_data:
                    _fdf = pd.DataFrame(_factor_data)
                    _fe_result = _fe(_port_rets, _fdf)
                    _fe_line = " | ".join(
                        f"{k}: β={_fe_result.exposures.get(k, 0):.2f}"
                        f"(p={_fe_result.p_values.get(k, 1):.2f})"
                        for k in _fdf.columns
                    )
                    logger.info(f"  因子暴露[{_fdf.columns.tolist()}]: {_fe_line}")
                    logger.info(f"  回归R²={_fe_result.rsquared:.2%}, adjR²={_fe_result.adj_rsquared:.2%}")
        except Exception:
            pass

        # ── 组合风险暴露（行业 + 风格） ──
        try:
            if pd.api.types.is_datetime64_any_dtype(final_prepared["trade_date"]):
                _fp2 = final_prepared.copy()
                _fp2["trade_date"] = _fp2["trade_date"].dt.strftime("%Y-%m-%d")
                _last_bar = _fp2[_fp2["trade_date"] == _fp2["trade_date"].max()]
            else:
                _last_bar = final_prepared[final_prepared["trade_date"] == final_prepared["trade_date"].max()]
            _risk_holdings = {t["symbol"]: t.get("value", 0) for t in trade_log if t.get("action") == "buy"}
            _total_val = sum(_risk_holdings.values()) or 1
            _pw = pd.Series({k: v / _total_val for k, v in _risk_holdings.items()})
            if len(_pw) > 1 and "行业" in _last_bar.columns:
                from BackTrading.risk_model import compute_industry_exposure, industry_hhi
                _ind_map = _last_bar.set_index("symbol")["行业"].to_dict()
                _ind_exp = compute_industry_exposure(_pw, pd.Series({k: _ind_map.get(k, "未知") for k in _pw.index}))
                _top_ind = sorted(_ind_exp.items(), key=lambda x: -x[1])[:5]
                _hhi = industry_hhi(_ind_exp)
                _ind_line = " | ".join(f"{s}: {w:.1%}" for s, w in _top_ind)
                logger.info(f"  行业暴露 Top5: {_ind_line}")
                if _hhi > 0.3:
                    logger.warning(f"  行业 HHI={_hhi:.2f} > 0.3，集中度偏高")
        except Exception:
            pass

        # ── 因子衰减检查（信号分 vs 前向收益的 Rank IC） ──
        try:
            _fwd_ret = final_prepared.groupby("symbol")["close"].transform(
                lambda s: s.shift(-5) / s - 1
            )
            _ic_cols = ["MACD趋势分", "金叉信号分", "柱状动能分", "DIF斜评分", "背离信号分", "量价配合分", "K线形态分"]
            _ic_factors = {c: "MACD趋势", "金叉信号": "金叉", "柱状动能": "动能",
                           "DIF斜评分": "斜率", "背离信号": "背离", "量价配合": "量价", "K线形态分": "K线"}
            _ics = []
            for _c in _ic_cols:
                if _c not in final_prepared.columns:
                    continue
                _valid = final_prepared[_c].notna() & _fwd_ret.notna()
                if _valid.sum() < 20:
                    continue
                _rho, _ = spearmanr(final_prepared.loc[_valid, _c], _fwd_ret[_valid])
                if not np.isnan(_rho):
                    _ics.append((_ic_factors.get(_c, _c), _rho))
            if _ics:
                _ic_line = " | ".join(f"{n}: IC={r:.3f}" for n, r in _ics)
                logger.info(f"  信号Rank IC（5日前向收益）: {_ic_line}")
        except Exception:
            pass

        top = wf_result.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False).head(5)
        sharpe_avg = float(top["sharpe_ratio"].mean())
        total_return_avg = float(top["total_return"].mean())
        max_dd_avg = float(top["max_drawdown"].mean())

        from BackTrading.calibration import _get_git_commit
        from BackTrading.prepare import _compute_config_hash

        from BackTrading.overfitting import compute_pbo, compute_dsr_from_equity_curve

        wf_results_list = wf_result.to_dict("records") if not wf_result.empty else []
        pbo = compute_pbo(wf_results_list)
        num_combos = int(wf_result["num_combos"].iloc[0]) if not wf_result.empty and "num_combos" in wf_result.columns else 1
        num_trials = num_combos * len(wf_result)
        dsr = compute_dsr_from_equity_curve(equity_curve, num_trials)

        logger.info(f"  Deflated Sharpe Ratio(DSR)={dsr:.2%} | PBO={pbo:.2%} | 试验次数={num_trials}")
        if pbo > 0.5:
            logger.warning(f"PBO={pbo:.2%}>50%，过拟合风险较高，建议缩减参数网格或增加数据")
        if dsr < 0.5:
            logger.warning(f"DSR={dsr:.2%}<50%，统计显著性不足")

        cal_result = CalibrationResult(
            params=best_params,
            score=sharpe_avg,
            sharpe=risk.get("sharpe_ratio", sharpe_avg),
            sortino=risk.get("sortino_ratio", 0),
            calmar=risk.get("calmar_ratio", 0),
            max_drawdown=risk.get("max_drawdown", max_dd_avg),
            max_drawdown_duration=int(risk.get("max_drawdown_duration", 0)),
            total_return=risk.get("total_return", total_return_avg),
            annual_return=risk.get("annual_return", 0),
            annual_vol=risk.get("annual_vol", 0),
            var_95=risk.get("var_95", 0),
            cvar_95=risk.get("cvar_95", 0),
            win_rate=trade.get("win_rate", 0),
            profit_factor=trade.get("profit_factor", 0),
            total_trades=trade.get("total_trades", 0),
            timestamp=datetime.now().isoformat(),
            git_commit=_get_git_commit(),
            config_hash=_compute_config_hash(),
            pbo=round(pbo, 4),
            dsr=round(dsr, 4),
            num_trials=num_trials,
        )
        save_calibration(cal_result)

        # ── 多策略组合回测 ──
        _enable_ms = getattr(bt, "MULTI_STRATEGY_ENABLED", False)
        if _enable_ms:
            try:
                from BackTrading.multi_strategy import run_multi_strategy_backtest as _rms
                _ms_result = _rms(kline_df, ecfg, best_params, trade_log, equity_curve)
                logger.info(f"  多策略组合完成: {len(_ms_result)} 个子策略")
            except Exception as e:
                logger.warning(f"  多策略组合回测异常: {e}")

        # ── 压力测试 ──
        try:
            from BackTrading.stress_test import run_stress_tests as _rst
            _stress_results = _rst(kline_df, ecfg, best_params)
            _worst_dd = min((r.get("max_drawdown", 0) for r in _stress_results.values()), default=0)
            if _worst_dd < -0.3:
                logger.warning(f"  压力测试: 历史极端场景最大回撤 {_worst_dd:.2%} > 30%，建议评估风险")
        except Exception as e:
            logger.warning(f"  压力测试异常: {e}")

        if _promote:
            write_calibration_to_ini(best_params)
            apply_calibration_to_config(config)
            logger.info("模拟验证通过，参数已写入 config.ini 并生效")
        else:
            logger.warning("模拟验证不通过，config.ini 参数保持不变，可作为回测报告参考")
            # 仍将结果写入数据库用于历史追踪

        record_run(
            engine=engine,
            frequency=bt.OPTIMIZE_FREQUENCY,
            backtest_start_date=bt.BACKTEST_START_DATE,
            out_of_sample_days=bt.OUT_OF_SAMPLE_DAYS,
            initial_cash=bt.INITIAL_CASH,
            params=best_params,
            sharpe=cal_result.sharpe,
            total_return=cal_result.total_return,
            max_drawdown=cal_result.max_drawdown,
            extra_metrics=risk | trade | {"pbo": cal_result.pbo, "dsr": cal_result.dsr, "num_trials": cal_result.num_trials},
            git_commit=cal_result.git_commit,
            config_hash=cal_result.config_hash,
        )

        updated_sections = set()
        for k in best_params:
            if k in CALIB_PARAM_MAP:
                updated_sections.add(CALIB_PARAM_MAP[k][0])
        logger.info(f"  寻优结果已写入 calibration_result.json + config.ini [{', '.join(sorted(updated_sections))}]")
        alert.on_success(cal_result)
        return cal_result

    except Exception as exc:
        logger.opt(exception=True).error(f"回测管线失败: {exc}")
        try:
            record_run(
                engine=engine,
                frequency=bt.OPTIMIZE_FREQUENCY,
                backtest_start_date=bt.BACKTEST_START_DATE,
                out_of_sample_days=bt.OUT_OF_SAMPLE_DAYS,
                initial_cash=bt.INITIAL_CASH,
                params={},
                sharpe=0,
                total_return=0,
                max_drawdown=0,
                status="failed",
            )
        except Exception as log_err:
            logger.warning(f"回测失败记录写入异常: {log_err}")
        alert.on_failure(exc)
        return None


def _resolve_symbols(engine: Any, config: Config | None = None) -> list[str]:
    """解析股票列表，支持 main_board_only 过滤。
    
    为消除生存者偏差，股票池包含所有曾有过交易记录的股票（含已退市）。
    ST/*ST/退市的逐日动态剔除由引擎配合 stock_st_history 完成，此处不做静态剔除。
    """
    from UtilsManager.CodeNormalizer import CodeNormalizer

    with engine.connect() as conn:
        # 合并：K 线已有数据的股票 + ST历史表中的股票（含退市）
        rows = conn.execute(text("""
            SELECT DISTINCT symbol FROM stock_daily_kline
            UNION
            SELECT DISTINCT symbol FROM stock_st_history
            ORDER BY symbol
        """)).fetchall()
    raw = sorted({str(r[0]) for r in rows})
    # 尝试从 symbol 中提取纯数字代码用于主板过滤
    if config is not None and config.MAIN_BOARD_ONLY:
        before = len(raw)
        raw = [s for s in raw if s.replace("sh", "").replace("sz", "").startswith(("60", "00"))]
        if len(raw) < before:
            logger.info(f"主板过滤后剩余: {len(raw)} / {before} 只")
    # 注意：不再做静态 ST 剔除，逐日动态剔除由引擎根据 stock_st_history 完成
    # 保留 EXCLUDE_ST 配置兼容性，仅记录日志
    if config is not None and config.app_config.backtest.EXCLUDE_ST:
        logger.info("EXCLUDE_ST=True：将由引擎按 stock_st_history 逐日动态剔除 ST/*ST/退市股票")
    if not raw:
        logger.warning("回测股票池为空，请检查数据库 stock_daily_kline 表")
    return sorted({CodeNormalizer.add_market_prefix(s) if not s.startswith(("sh", "sz")) else s for s in raw})


def _load_st_history(engine: Any, symbols: list[str], start_date: str, end_date: str) -> dict[str, dict[str, tuple[bool, bool]]]:
    """
    加载股票在日期范围内的 ST/退市状态历史。
    
    Returns:
        dict: {symbol: {trade_date: (is_st, is_delisting)}}
    """
    try:
        # 只加载回测股票池中股票的 ST 历史，减少数据量
        sym_placeholders = ",".join([f"'{s}'" for s in symbols])
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT symbol, trade_date, is_st, is_delisting
                FROM stock_st_history
                WHERE symbol IN ({sym_placeholders})
                  AND trade_date >= :start_date
                  AND trade_date <= :end_date
            """), {"start_date": start_date, "end_date": end_date}).fetchall()
        
        st_history = {}
        for symbol, trade_date, is_st, is_delisting in rows:
            if symbol not in st_history:
                st_history[symbol] = {}
            st_history[symbol][str(trade_date)] = (is_st, is_delisting)
        
        logger.info(f"加载 ST 历史状态: {len(st_history)} 只股票，{len(rows)} 条记录")
        return st_history
    except Exception as e:
        logger.warning(f"加载 ST 历史失败，将使用静态剔除: {e}")
        return {}


def _fetch_kline(
    engine: Any,
    symbols: list[str],
    backtest_start_date: str,
) -> pd.DataFrame:
    from DataManager.sync import ensure_table
    from DataManager.IncrementalSyncEngine import IncrementalSyncEngine

    # 将配置日期对齐到首个交易日，与 IncrementalSyncEngine 内部逻辑一致
    aligned_start = IncrementalSyncEngine.align_to_trading_day(backtest_start_date)

    ensure_table(engine)

    # 补齐缺失股票的历史 K 线
    _sync_missing_stocks(engine, symbols, aligned_start)

    end = date.today()
    start = datetime.strptime(aligned_start, "%Y%m%d").date()

    # 前拉缓冲期确保技术指标充分预热（MACD/ATR/MA等需至少 120 个交易日）
    _buffer_trading_days = 180
    _buffer_calendar_days = _buffer_trading_days * 2
    buffer_start = (start - timedelta(days=_buffer_calendar_days)).isoformat()

    provider = BacktestDataProvider(engine)
    df: pd.DataFrame = provider.get_kline(symbols, start_date=buffer_start, end_date=end.isoformat())
    if df.empty:
        return df
    df = df.sort_values(["symbol", "trade_date"])
    return df


def _sync_missing_stocks(engine: Any, symbols: list[str], backtest_start_date: str) -> None:
    """补齐 + 刷新 stock_daily_kline 数据。检查每只股票数据是否齐全，检测除权除息并重拉。

    同时执行一次性"指标预热回填"：已有数据但最早交易日晚于预热起点的股票，
    强制从预热起点回填历史 K 线（MACD/ATR/MA 等指标至少需要 120 个交易日前文）。
    """
    from DataManager.IncrementalSyncEngine import IncrementalSyncEngine

    start = datetime.strptime(backtest_start_date, "%Y%m%d").date()
    _buffer_calendar_days = 360
    buffer_start_iso = (start - timedelta(days=_buffer_calendar_days)).isoformat()

    syncer = IncrementalSyncEngine(engine, default_start=backtest_start_date)

    # 检查哪些股票完全缺失
    with engine.connect() as conn:
        existing = {
            r[0] for r in
            conn.execute(text("SELECT DISTINCT symbol FROM stock_daily_kline")).fetchall()
        }
    missing = [s for s in symbols if s not in existing]
    if missing:
        logger.info(f"  stock_daily_kline 缺少 {len(missing)} 只股票，开始补齐...")
        n = syncer.sync_all(missing, force_start_iso=buffer_start_iso)
        logger.info(f"  补齐完成，新增 {n} 行")

    # 对已有数据的股票执行增量刷新：检查最新日期、除权除息检测
    existing_symbols = [s for s in symbols if s not in missing]
    if existing_symbols:
        logger.info(f"  检查 {len(existing_symbols)} 只股票数据完整性...")
        total = syncer.sync_all(existing_symbols)
        logger.info(f"  刷新完成，新增 {total} 行")

    # 一次性指标预热回填：数据起点晚于预热起点的股票（缺早期历史，指标前文不足）
    if existing_symbols:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT symbol, MIN(trade_date) AS first_d FROM stock_daily_kline "
                    "GROUP BY symbol"
                )).fetchall()
            first_by_symbol = {r[0]: r[1] for r in rows}
            need_warmup = [
                s for s in existing_symbols
                if s in first_by_symbol and first_by_symbol[s] is not None
                and pd.Timestamp(first_by_symbol[s]).strftime("%Y-%m-%d") > buffer_start_iso
            ]
            if need_warmup:
                logger.info(
                    f"  {len(need_warmup)} 只股票历史不足 {buffer_start_iso}（指标预热），"
                    f"强制回填中（示例: {need_warmup[:5]}）..."
                )
                w_total = syncer.sync_all(need_warmup, force_start_iso=buffer_start_iso)
                logger.info(f"  预热回填完成，新增 {w_total} 行")
        except Exception as e:
            logger.warning(f"  预热回填失败（回测继续，指标前文可能不足）: {e}")


def _extract_best_params(wf_result: pd.DataFrame, top_n: int = 5, config: Config | None = None) -> dict[str, float]:
    """
    从 Walk-Forward 结果中提取最佳参数。

    如果提取失败（数据不足、Sharpe 全为 NaN/负值、params 列缺失等），
    返回配置中的默认参数中位数作为兜底，并记录警告。
    """
    # 默认兜底参数（从配置区间取中位数）
    def _fallback_params(cfg: Config | None) -> dict[str, float]:
        if cfg is None:
            return {
                "atr_stop_mult": 2.0,
                "boll_narrow_ratio": 0.9,
                "cross_decay_days": 37,
                "conclusion_full_bull": 80,
                "golden_cross_bonus": 10,
                "divergence_penalty": 20,
                "buy_threshold": 17,
                "max_holdings": 11,
            }
        bt = cfg.app_config.backtest
        return {
            "atr_stop_mult": sum(bt.parse_range("ATR_STOP_MULT_RANGE")[:2]) / 2,
            "boll_narrow_ratio": sum(bt.parse_range("BOLL_NARROW_RATIO_RANGE")[:2]) / 2,
            "cross_decay_days": sum(bt.parse_range("CROSS_DECAY_DAYS_RANGE")[:2]) / 2,
            "conclusion_full_bull": sum(bt.parse_range("CONCLUSION_FULL_BULL_RANGE")[:2]) / 2,
            "golden_cross_bonus": sum(bt.parse_range("GOLDEN_CROSS_BONUS_RANGE")[:2]) / 2,
            "divergence_penalty": sum(bt.parse_range("DIVERGENCE_PENALTY_RANGE")[:2]) / 2,
            "buy_threshold": sum(bt.parse_range("BUY_THRESHOLD_RANGE")[:2]) / 2,
            "max_holdings": sum(bt.parse_range("MAX_HOLDINGS_RANGE")[:2]) / 2,
        }

    if wf_result.empty or "params" not in wf_result.columns:
        logger.warning("Walk-Forward 结果为空或缺少 params 列，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    rows = wf_result.dropna(subset=["sharpe_ratio"])
    if rows.empty:
        logger.warning("Walk-Forward 所有组合 Sharpe 均为 NaN，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    rows = rows.sort_values("sharpe_ratio", ascending=False).head(top_n)
    weights = rows["sharpe_ratio"].values
    total_weight = weights.sum()
    if total_weight <= 0:
        logger.warning("Walk-Forward Top-N 组合 Sharpe 权重和 <= 0，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    all_params: list[dict[str, float]] = []
    for _, r in rows.iterrows():
        if isinstance(r["params"], dict):
            all_params.append({k: float(v) for k, v in r["params"].items()})

    if not all_params:
        logger.warning("Walk-Forward params 列无有效 dict，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    keys = all_params[0].keys()
    weighted: dict[str, float] = {}
    for k in keys:
        vals = [p[k] for p in all_params]
        weighted[k] = sum(v * w for v, w in zip(vals, weights)) / total_weight
    return weighted


def start_scheduler(config: Config | None = None) -> None:
    """启动定时调度（每日检查，按配置频率执行回测）。"""
    import time

    import schedule as _schedule

    if config is None:
        config = Config()

    bt = config.app_config.backtest
    if not bt.ENABLED:
        logger.info("回测未启用，调度器不启动")
        return

    engine = get_engine(config)
    ensure_table(engine)

    logger.info(f"启动回测调度器 (频率={bt.OPTIMIZE_FREQUENCY})")

    def job() -> None:
        logger.info("调度触发：检查回测条件 ...")
        tmp_engine = get_engine(config)
        last = get_last_run(tmp_engine)
        should_run, reason = should_rerun(last, bt.OPTIMIZE_FREQUENCY)
        if should_run:
            run_backtest_pipeline(config, force=True)
        else:
            logger.info(f"调度跳过: {reason}")

    _schedule.every().day.at("02:00").do(job)
    logger.info("  每日 02:00 检查回测条件")

    if bt.OPTIMIZE_FREQUENCY == "initial":
        logger.info("  optimize_frequency=initial，立即执行首次回测")
        run_backtest_pipeline(config, force=True)

    while True:
        _schedule.run_pending()
        time.sleep(3600)


def main() -> None:
    """CLI 入口。

    Usage:
        python -m BackTrading.runner            # 执行回测（交互式判断是否已过期）
        python -m BackTrading.runner --force     # 强制重新回测
        python -m BackTrading.runner --schedule  # 启动常驻调度器
    """
    args = sys.argv[1:]
    config = Config()

    if "--schedule" in args:
        start_scheduler(config)
        return

    force = "--force" in args
    result = run_backtest_pipeline(config, force=force)
    if result is None:
        sys.exit(0)
    logger.info(f"回测完成: Sharpe={result.sharpe:.2f}, Return={result.total_return:.2%}")


if __name__ == "__main__":
    main()
