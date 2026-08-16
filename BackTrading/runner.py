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


_BACKTEST_LOCK_KEY = 987654321
# 会话级 advisory lock 专用连接：回测全程持有，外部增量同步（IncrementalSyncEngine）
# 在写 K 线前探测同一 key，被占用即跳过，防止运行中数据被改写导致缓存内容漂移。
_RUN_LOCK_CONN: Any = None


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

    # 会话级锁：在专用连接上持有整个回测期间（session-level，跨事务存活）。
    # 同一会话重复获取返回 True（回测自身的启动同步不受影响），
    # 外部进程探测同一 key 返回 False → 同步引擎跳过本次执行。
    global _RUN_LOCK_CONN
    _RUN_LOCK_CONN = engine.connect()
    try:
        held = _RUN_LOCK_CONN.execute(
            _t(f"SELECT pg_try_advisory_lock({_BACKTEST_LOCK_KEY})")
        ).scalar()
        if held:
            logger.info("  获取会话级数据隔离锁成功（外部数据同步将让路）")
        else:
            logger.warning("  会话级数据隔离锁被占用，仍继续执行")
    except Exception as exc:
        logger.warning(f"  会话级数据隔离锁获取失败: {exc}")
        _RUN_LOCK_CONN.close()
        _RUN_LOCK_CONN = None


def _release_run_lock() -> None:
    """释放会话级 advisory lock（关闭专用连接即自动释放）。"""
    global _RUN_LOCK_CONN
    if _RUN_LOCK_CONN is not None:
        try:
            _RUN_LOCK_CONN.close()
        except Exception:
            pass
        _RUN_LOCK_CONN = None


def _to_date(v: Any) -> date | None:
    """统一日期归一化：任意时间类型 → datetime.date。

    支持 str / pd.Timestamp / datetime.datetime / numpy.datetime64 / int 等。
    解析失败返回 None（调用方负责处理）。
    """
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.Timestamp(v).date()
    except (ValueError, TypeError, OverflowError):
        return None


def _holdout_equity_slice(
    equity_curve: list[dict[str, Any]] | pd.DataFrame | None,
    final_prepared: pd.DataFrame,
    holdout_days: int,
) -> tuple[list[dict[str, Any]] | pd.DataFrame | None, str | None]:
    """P0-3：按交易日索引切出末段 holdout 净值曲线。

    弃用 len(equity_curve)×ratio 的日历轴切分——净值曲线按日历轴生成（含补全日），
    长度口径与 WFO 交易日口径漂移导致边界错位。这里以 final_prepared 的交易日
    集合定位边界（末 holdout_days 个交易日），再按 time 过滤净值曲线。
    equity_curve 兼容 list 与 DataFrame（旧实现假设 DataFrame，list 上 .empty
    直接 AttributeError 导致整条校准管线 FAILED）。

    Returns:
        (切片后的净值曲线, holdout 起始交易日字符串)；数据不足/类型不支持时
        返回 (None, None)。
    """
    if holdout_days <= 0 or equity_curve is None:
        return None, None
    if not isinstance(equity_curve, (list, pd.DataFrame)):
        return None, None

    # 统一归一化为 date 对象比较，消除 str[:10] 的时区/格式脆弱性
    _fp_dates = [
        d for d in map(_to_date, sorted(pd.unique(final_prepared["trade_date"])))
        if d is not None
    ]
    if len(_fp_dates) < holdout_days:
        return None, None
    _start_date = _fp_dates[-holdout_days]  # datetime.date

    if isinstance(equity_curve, pd.DataFrame):
        if equity_curve.empty:
            return None, None
        # 将 time 列（或 index）归一化为 date 序列后比较
        if "time" in equity_curve.columns:
            _dates = equity_curve["time"].apply(_to_date)
        else:
            _dates = equity_curve.index.to_series().apply(_to_date)
        mask = _dates >= _start_date
        return equity_curve[mask], _start_date.isoformat()

    # list[dict] 分支
    result = [
        e for e in equity_curve
        if _to_date(e.get("time")) is not None and _to_date(e.get("time")) >= _start_date
    ]
    return result, _start_date.isoformat()


def _acceptance_gate(
    *,
    promote: bool,
    oos_decay_pass: bool,
    overfitting_critical: bool,
    sig_pass: bool,
    robust_pass: bool,
    pbo_gate: bool,
    dsr_gate: bool,
) -> tuple[bool, list[str]]:
    """P0-5：统一参数采纳门控（save_calibration 与 write_calibration_to_ini 共用）。

    修复门控不一致：write_calibration_to_ini 曾未应用 PBO/DSR 门控（save_calibration
    有），PBO 过拟合参数集仍可落盘进生产 config.ini。两处采纳决策必须走同一
    门控——任一关键项不通过，calibration_result.json 与 config.ini 均不写入。

    Returns:
        (是否全部通过, 未通过原因列表)。
    """
    reasons: list[str] = []
    if not promote:
        reasons.append("模拟验证未通过")
    if not oos_decay_pass:
        reasons.append("OOS 衰减校验未通过")
    if overfitting_critical:
        reasons.append("多重测试惩罚 CRITICAL")
    if not sig_pass:
        reasons.append("统计显著性未通过")
    if not robust_pass:
        reasons.append("参数稳健性自检不通过")
    if not pbo_gate:
        reasons.append("PBO > 5% 阈值（过拟合风险）")
    if not dsr_gate:
        reasons.append("DSR < 50% 阈值（缩水 Sharpe 不足）")
    return (len(reasons) == 0), reasons


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

    # ── P3.1/P3.2 四方绑定：数据版本 + 配置哈希，变化即强制重跑 ──
    from BackTrading.prepare import _compute_config_hash as _cfg_hash
    _data_version = _compute_kline_data_version(engine)
    _cur_config_hash = _cfg_hash()

    last = get_last_run(engine)
    should_run, reason = should_rerun(
        last, bt.OPTIMIZE_FREQUENCY,
        data_version=_data_version,
        config_hash=_cur_config_hash,
    )

    if not should_run and not force:
        # P0-11：移除阻塞式 input() 交互（生产调度/每日 02:00 DAG 中无终端会挂起）。
        # 默认跳过并提示；需强制重跑时显式传 force=True（或在调度侧传参）。
        logger.info(f"{reason} → 跳过（如需强制重跑请调用 run_backtest_pipeline(force=True)）")
        return load_calibration()

    logger.info("=" * 50)
    logger.info("开始回测管线 ...")
    logger.info(f"  优化频率: {bt.OPTIMIZE_FREQUENCY}")
    logger.info(f"  数据起始日期: {bt.BACKTEST_START_DATE}")
    logger.info(f"  样本外天数: {bt.OUT_OF_SAMPLE_DAYS}")
    logger.info(f"  初始资金: {bt.INITIAL_CASH:,.0f}")

    # ── A2 失败快照：进程级 run_id/task_id 上下文（随日志/告警/快照透出） ──
    import uuid as _uuid

    _run_id = _uuid.uuid4().hex[:12]
    from BackTrading.snapshot import begin_snapshot_session, save_failure_snapshot, set_run_context
    set_run_context(run_id=_run_id, task_id="backtest_pipeline")
    begin_snapshot_session()

    kline_df: pd.DataFrame | None = None

    _step_times: dict[str, float] = {"start": time.time()}
    def _log_step(name: str) -> None:
        _step_times[name] = time.time()
        _elapsed = _step_times[name] - _step_times.get(list(_step_times.keys())[-2] if len(_step_times) >= 2 else "start", 0)
        _total = _step_times[name] - _step_times["start"]
        logger.info(f"[STEP] {name} ({_elapsed:.1f}s, 累计 {_total:.1f}s)")

    try:
        symbols = _resolve_symbols(engine, config)
        logger.info(f"  股票数量: {len(symbols)}")
        _log_step("resolve_symbols")

        kline_df = _fetch_kline(engine, symbols, bt.BACKTEST_START_DATE)
        if kline_df.empty:
            logger.warning("K 线数据为空，跳过回测")
            return None

        logger.info(f"  K 线行数: {len(kline_df)}")

        # P3.1: 从内存 DataFrame 计算数据版本（消除 fetch→version 竞态窗口）
        _data_version = _compute_kline_data_version(engine, kline_df=kline_df)

        # ── ST/退市历史早加载（供 WFO / 模拟验证 / 最终回测全链路使用） ──
        # P0-5: 查询起点覆盖 K 线预热缓冲（_fetch_kline 用 360 日历日缓冲），
        # 否则缓冲期内 ST 涨跌幅 5% 判定缺失。
        _bt_start_iso = datetime.strptime(bt.BACKTEST_START_DATE, "%Y%m%d").date().isoformat()
        _st_query_start = (
            datetime.strptime(bt.BACKTEST_START_DATE, "%Y%m%d").date() - timedelta(days=360)
        ).isoformat()
        _end_date = kline_df["trade_date"].max()
        if pd.api.types.is_datetime64_any_dtype(kline_df["trade_date"]):
            _end_date = _end_date.strftime("%Y-%m-%d")
        # P0-5: ST/退市 PIT 同步（全历史逐日状态回填；网络失败优雅降级，仅告警不阻断）
        try:
            from DataManager.StPitSync import ensure_st_history_table, sync_st_pit
            ensure_st_history_table(engine)
            sync_st_pit(engine, symbols, start_date=_st_query_start, end_date=_end_date)
        except Exception as e:
            logger.warning(f"  ST PIT 同步失败（使用现有 stock_st_history 数据）: {e}")
        st_history = _load_st_history(engine, symbols, _st_query_start, _end_date)

        # ── P0-6 ④: 上市日期表同步（AkShare stock_info_a_code_name → stock_listing_days） ──
        # 显式注入 IPO 日期，引擎禁止从行情数据推断上市日（数据缺口会误判新股，
        # 错误激活"注册制前 5 日无涨跌幅"豁免）。网络失败优雅降级（仅告警不阻断）。
        try:
            from DataManager.ListingDaysSync import (
                ensure_listing_days_table, sync_listing_days,
            )
            ensure_listing_days_table(engine)
            sync_listing_days(engine, symbols)
        except Exception as e:
            logger.warning(f"  上市日期同步失败（引擎将停用新股豁免逻辑）: {e}")
        listing_days = _load_listing_days(engine, symbols, _st_query_start)
        _log_step("load_listing_days")

        # 生存偏差实测评估：池内退市股的历史 K 线是否真实纳入（其退市前负收益才会计入）。
        # P3-5（审计）：评估改用独立数据源（AkShare 交易所退市列表）交叉验证，与
        # stock_st_history PIT 表解耦——PIT 同步失败不应导致"生存偏差受控"误报。
        # 独立源拉取失败 → 降级到 PIT 退市标记口径并注明降级（行为与旧版一致）。
        _kline_syms = set(kline_df["symbol"].astype(str))
        _survival_source = "AkShare 退市列表（独立数据源）"
        try:
            from DataManager.StPitSync import fetch_delisted_symbols
            _delisted_syms = fetch_delisted_symbols() or set()
        except Exception as e:  # noqa: BLE001
            _delisted_syms = set()
            _survival_source = f"PIT 退市标记（独立源拉取失败降级: {e}）"
        if not _delisted_syms:
            _delisted_syms = {
                s for s, recs in st_history.items()
                if any(_is_del for _is_st, _is_del in recs.values())
            }
            if _survival_source.startswith("AkShare"):
                _survival_source = "PIT 退市标记（独立源为空降级）"
        _missing_delisted = sorted(_delisted_syms - _kline_syms)
        if _delisted_syms:
            logger.info(
                f"  独立退市列表（{_survival_source}）识别 {len(_delisted_syms)} 只退市股，"
                f"其中 {len(_delisted_syms & _kline_syms)} 只已纳入 K 线"
            )
            if _missing_delisted:
                logger.warning(
                    f"生存偏差: {len(_missing_delisted)} 只退市股的历史 K 线缺失"
                    f"（如 {_missing_delisted[:5]}），其退市前负收益未被计入，"
                    f"建议扩展数据同步范围（含已退市股票）"
                )
            else:
                logger.info(
                    "生存偏差受控: 退市股历史 K 线已纳入股票池，"
                    "退市前负收益计入回测；ST/退市日的逐日剔除由引擎按 stock_st_history 执行"
                )
        else:
            logger.warning(
                "生存偏差: 独立退市列表与 stock_st_history 均无退市记录，"
                "股票池可能仅含当前存活股票，已退市/ST 股票的历史负收益未被计入"
            )
        _log_step("load_st_history")

        # 窗口坐标轴以正式回测起点为准（起点前为信号预热历史，不参与 WFO 交易）
        # 统一使用 _to_date 归一化，消除 str[:10] 时区/格式脆弱性
        _bt_start = _to_date(bt.BACKTEST_START_DATE)
        assert _bt_start is not None, f"无效的 BACKTEST_START_DATE: {bt.BACKTEST_START_DATE!r}"
        total_trading_days = sum(
            1 for d in kline_df["trade_date"].unique()
            if _to_date(d) is not None and _to_date(d) >= _bt_start
        )
        _oos = bt.OUT_OF_SAMPLE_DAYS
        # ── 末段独立 holdout：终验只在该段进行，WFO 全程禁触 ──
        # holdout_days = round(total * ratio)；钳制 ≥OOS 且 WFO 寻参域须留足 120+OOS，
        # 否则禁用 holdout 回退旧逻辑（自引用，但保证 WFO 至少可跑）。
        _holdout_ratio = float(getattr(bt, "HOLDOUT_RATIO", 0.0))
        _holdout_days = 0
        _holdout_active = False
        if _holdout_ratio > 0:
            _holdout_days = round(total_trading_days * _holdout_ratio)
            _wfo_total = total_trading_days - _holdout_days
            if _holdout_days >= _oos and _wfo_total >= 120 + _oos:
                _holdout_active = True
            else:
                logger.warning(
                    f"  末段 holdout={_holdout_days} 天但条件不满足"
                    f"（holdout≥OOS? {_holdout_days >= _oos} | WFO域={_wfo_total}≥{120 + _oos}? "
                    f"{_wfo_total >= 120 + _oos}），禁用 holdout 回退旧逻辑"
                )
                _holdout_days = 0
        _wfo_total = total_trading_days - _holdout_days
        # 数据自适应 WFO 配置：路径 p 的 offset = p*OOS，需满足 offset + IS + OOS <= n，
        # 否则路径 2/3 必然越界跳过（如 IS=805+OOS=60 在 865 天数据上只有 1 条路径有效）。
        _np_cfg = max(1, int(bt.WFO_NUM_PATHS))
        _max_np = max(1, (_wfo_total - 120) // _oos) if _wfo_total > _oos + 120 else 1
        _num_paths = min(_np_cfg, _max_np)
        train_period = max(120, min(_wfo_total - _oos, _wfo_total - _oos * _num_paths))
        _holdout_label = (
            f" | Holdout: {_holdout_days}天(占比{_holdout_days/total_trading_days:.0%},独立终验)"
            if _holdout_active else " | Holdout: 禁用(自引用回退)"
        )
        logger.info(
            f"  交易日数: {total_trading_days} | IS训练窗口: {train_period}天 | OOS: {_oos}天"
            f" | WFO路径数: {_num_paths}（配置 {_np_cfg}，数据上限 {_max_np}）"
            f"{_holdout_label}"
        )
        _log_step("fetch_kline")
        # ── WFO 系统性失败拦截：捕获 WFOSystematicFailure，记录失败并中断流水线 ──
        from BackTrading.bayesian.meta_optimizer import WFOSystematicFailure

        try:
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
                backtest_start_date=_bt_start_iso,
                st_history=st_history,
                exclude_st=bool(bt.EXCLUDE_ST),
                listing_days=listing_days,
                # P2.1 CPCV 净化+禁运
                purge_days=int(bt.BAYESIAN_CPCV_PURGE_DAYS),
                embargo_days=int(bt.BAYESIAN_CPCV_EMBARGO_DAYS),
                # P2.4 预算制
                time_budget_seconds=float(bt.BAYESIAN_TIME_BUDGET_SECONDS),
                max_no_improve_windows=int(bt.BAYESIAN_MAX_NO_IMPROVE_WINDOWS),
                # 末段独立 holdout：WFO 寻参上界切除末段，供终验独立使用
                holdout_days=_holdout_days,
                # P3.1 数据版本入缓存 key
                data_version=_data_version,
                # A2 失败快照上下文
                run_id=_run_id,
                task_id="backtest_pipeline",
            )
        except WFOSystematicFailure as wfo_err:
            # WFO 系统性失败：策略在当前数据区间无泛化能力，中断流水线
            logger.critical("=" * 60)
            logger.critical(f"[WFO系统性失败] {wfo_err.reason}")
            logger.critical(
                "策略在当前数据区间不具备泛化能力，建议检查：\n"
                "  1. 特征工程是否存在隐性数据泄露\n"
                "  2. 信号逻辑是否适应当前市场 regime\n"
                "  3. 回看区间是否过短/过长导致过拟合\n"
                "本次回测已标记为失败，不会覆盖 config.ini，下次调度将跳过（需手动 force=True 或数据/配置变化后重跑）"
            )
            logger.critical("=" * 60)

            # 记录失败状态，避免下次调度重复执行
            record_run(
                engine=engine,
                frequency=bt.OPTIMIZE_FREQUENCY,
                backtest_start_date=bt.BACKTEST_START_DATE,
                out_of_sample_days=bt.OUT_OF_SAMPLE_DAYS,
                initial_cash=bt.INITIAL_CASH,
                params=wfo_err.fallback_params,
                sharpe=0,
                total_return=0,
                max_drawdown=0,
                status="failed",
                config_hash=_cur_config_hash,
                data_version=_data_version,
            )
            alert.on_failure(wfo_err)
            return None
        _log_step("walk_forward")
        logger.info(f"  Walk-Forward 片段数: {len(wf_result)}")

        if not wf_result.empty and wf_result["sharpe_ratio"].max() > 3.0:
            logger.warning(f"akquant 结果异常: Sharpe={wf_result['sharpe_ratio'].max():.2f}>3.0，可能存在前瞻偏差")

        best_params = _extract_best_params(wf_result, config=config)
        logger.info(f"  最佳参数(Sharpe加权前{min(5, len(wf_result))}): {best_params}")

        # ST/退市逐日动态剔除数据注入（引擎按 params 消费，WFO 已同口径）
        best_params["_st_history"] = st_history
        best_params["_exclude_st"] = bool(bt.EXCLUDE_ST)
        # P0-6 ④：上市日期显式注入（引擎禁止数据推断；空表时豁免逻辑整体停用）
        if listing_days:
            best_params["_listing_days"] = listing_days

        from BackTrading.engine import EngineConfig, run_full_backtest
        from BackTrading.domain.models import CostModel

        from UtilsManager.ConfigParser import PositionSizingConfig as _PsCfg
        _ps: _PsCfg = config.app_config.position_sizing
        _sc = config.app_config.scoring_params
        # 组合参数若未被寻优（兜底路径），取校准覆写值（无校准则配置默认，
        # P0-7 ②：与 [BACKTEST_CALIBRATED] 写回闭环一致，替代旧的区间中位口径）
        ecfg = EngineConfig(
            initial_cash=bt.INITIAL_CASH,
            commission_rate=bt.COMMISSION_RATE,
            stamp_tax_rate=bt.STAMP_TAX_RATE,
            slippage=bt.SLIPPAGE,
            max_position_pct=bt.MAX_POSITION_PCT,
            portfolio_method=bt.PORTFOLIO_METHOD,
            point_in_time=bt.POINT_IN_TIME,
            atr_stop_mult=best_params.get("atr_stop_mult", _sc.ATR_STOP_MULT),
            buy_threshold=int(best_params.get("buy_threshold", bt.BUY_THRESHOLD)),
            max_holdings=int(best_params.get("max_holdings", bt.MAX_HOLDINGS)),
            cost_model=CostModel.from_backtest_config(
                bt, trading_cost=config.app_config.trading_cost
            ),
            execution_model=bt.EXECUTION_MODEL,
            simulate_limit_up_down=bool(bt.SIMULATE_LIMIT_UP_DOWN),
            limit_seal_ratio=float(bt.LIMIT_SEAL_RATIO),
            limit_tradable_ratio=float(bt.LIMIT_TRADABLE_RATIO),
            limit_intraday_ratio=float(bt.LIMIT_INTRADAY_RATIO),
            limit_seal_decay=float(bt.LIMIT_SEAL_DECAY),
            # P0-6 ⑥：开盘集合竞价成交率分档
            auction_fill_ratio=float(bt.AUCTION_FILL_RATIO),
            # P0-6 ⑤：市场状态客观变量（指数20日收益 + 波动率分位）
            regime_ret20_full=float(bt.REGIME_RET20_FULL),
            regime_ret20_half=float(bt.REGIME_RET20_HALF),
            regime_vol_pct_max=float(bt.REGIME_VOL_PCT_MAX),
            resume_gap_up=float(bt.RESUME_GAP_UP),
            resume_gap_down=float(bt.RESUME_GAP_DOWN),
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

        # ── 模拟交易验证：优先用末段独立 holdout 验证集（WFO 全程禁触），
        #    未激活时回退最近交易日（自引用，validate_params 内告警）──
        from BackTrading.simulated_trading import validate_params as _sim_validate
        _wf_sharpe = float(wf_result["sharpe_ratio"].mean()) if not wf_result.empty else 0.0
        _holdout_dates: set[str] | None = None
        if _holdout_active and _holdout_days > 0:
            _k_dates = sorted(pd.Series(kline_df["trade_date"]).astype(str).unique())
            _holdout_dates = set(_k_dates[-_holdout_days:])
        _sim_verdict = _sim_validate(
            kline_df=kline_df, best_params=best_params,
            oos_sharpe=_wf_sharpe, sim_days=20,
            config=config, engine_cfg=ecfg,
            validation_dates=_holdout_dates,
        )
        _promote = _sim_verdict.promote
        if not _promote:
            logger.warning(f"模拟验证不通过，参数不写入 config.ini: {_sim_verdict.reason}")

        _log_step("prepare_final_signals")
        final_prepared = prepare_backtest_data(kline_df, params=final_params, compute_exit_strategy=True, vectorized=True, backtest_start_date=_bt_start_iso, data_version=_data_version)
        _log_step("full_backtest")
        # ST 历史已早加载并注入 best_params（见上方 _st_history 注入）
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

        # ══ 统计显著性基础校验（Statistical Significance）══
        # 三项硬 gate：样本量 / 持仓周期健康度 / 牛熊覆盖
        _sig_pass = True
        try:
            from LogicAnalyzer.statistical_significance import run_significance_check as _sig_check
            _sig_summary = _sig_check(trade_log, kline_df)
            if not _sig_summary.passed:
                _sig_pass = False
                logger.warning(
                    f"[统计显著性] 综合判定 FAIL — {_sig_summary.reason}"
                )
        except Exception as e:
            logger.warning(f"[统计显著性] 自检异常: {e}，不阻断（建议人工复核）")

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
        sharpe_avg = float(top["sharpe_ratio"].mean()) if not top.empty else 0.0
        # 兜底结果帧（无有效窗口时）可能缺少绩效列，逐列防御
        total_return_avg = float(top["total_return"].mean()) if "total_return" in top.columns and not top.empty else 0.0
        max_dd_avg = float(top["max_drawdown"].mean()) if "max_drawdown" in top.columns and not top.empty else 0.0

        # ── Holdout 终验 Sharpe（修正"选优报优"乐观偏差）──
        # holdout 激活时，业绩报告使用 holdout 终验 sharpe（末段 20% 独立回测），
        # WFO Top 5 均值仅用于参数选择，不对外报告。
        holdout_sharpe = None
        if _holdout_active and _holdout_days > 0:
            try:
                holdout_equity, _holdout_start_date = _holdout_equity_slice(
                    equity_curve, final_prepared, _holdout_days
                )
                if holdout_equity is not None and len(holdout_equity) >= 20:
                    holdout_risk = compute_risk_metrics(holdout_equity) or {}
                    holdout_sharpe = holdout_risk.get("sharpe_ratio")
                    logger.info(
                        f"  [Holdout终验] {_holdout_start_date}起末段{_holdout_ratio:.0%}"
                        f"共{len(holdout_equity)}条, Sharpe={holdout_sharpe:.4f}"
                        f"（WFO Top5 均值={sharpe_avg:.4f}）"
                    )
                elif holdout_equity is not None:
                    logger.warning(f"  [Holdout终验] 数据仅{len(holdout_equity)}条<20，回退 WFO 均值")
                else:
                    logger.warning(f"  [Holdout终验] 净值曲线为空或交易日不足，回退 WFO 均值")
            except Exception as e:
                logger.warning(f"  [Holdout终验] 计算异常: {e}，回退 WFO 均值")

        # 业绩报告 sharpe：优先 holdout 终验，其次 WFO Top 5
        report_sharpe = holdout_sharpe if holdout_sharpe is not None else sharpe_avg

        # P3 审计修复：报告层注明区间口径——Sharpe 来自 holdout 末段/WFO Top5
        # （选择期），total_return/max_drawdown 来自全周期最终回测（评估期），
        # 选择期≠评估期，跨期对比指标时必须区分区间，避免口径混淆
        if holdout_sharpe is not None:
            _sharpe_scope = f"末段独立 holdout（{_holdout_days} 个交易日）"
        else:
            _sharpe_scope = "WFO Top5 窗口均值"
        logger.info(
            f"[指标口径] 报告 Sharpe={report_sharpe:.4f} 区间={_sharpe_scope}（选择期）；"
            f"total_return={risk.get('total_return', 0):.2%} / "
            f"max_drawdown={risk.get('max_drawdown', 0):.2%} 为全周期最终回测口径"
            f"（评估期）——两者区间不一致属预期设计，跨期对比请注意"
        )

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

        # ══ 多重测试惩罚（Multiple Testing Deception）══
        # 统计同区间调参次数，超限则对 Sharpe/Sortino 施加统计学硬扣减
        from BackTrading.calibration_log import (
            count_tuning_attempts as _count_attempts,
            apply_multiple_testing_penalty as _apply_penalty,
            MAX_TUNING_ATTEMPTS as _max_attempts,
            MULTIPLE_TESTING_PENALTY as _penalty_rate,
        )
        _raw_sharpe = risk.get("sharpe_ratio", sharpe_avg)
        _raw_sortino = risk.get("sortino_ratio", 0)
        _attempt_count = _count_attempts(engine, bt.BACKTEST_START_DATE, bt.OUT_OF_SAMPLE_DAYS) + 1  # +1 包含本次
        _pun_sharpe, _pun_sortino, _warning_level = _apply_penalty(
            _raw_sharpe, _raw_sortino, _attempt_count,
            bt.BACKTEST_START_DATE, bt.OUT_OF_SAMPLE_DAYS,
        )
        # 用惩罚后的值替代原始值
        if _warning_level != "INFO":
            risk["sharpe_ratio"] = _pun_sharpe
            risk["sortino_ratio"] = _pun_sortino
            logger.warning(
                f"[多重测试惩罚] 原始 Sharpe={_raw_sharpe:.4f} → 惩罚后={_pun_sharpe:.4f} | "
                f"原始 Sortino={_raw_sortino:.4f} → 惩罚后={_pun_sortino:.4f}"
            )
        logger.info(
            f"[多重测试惩罚] 同区间累计调参 {_attempt_count} 次，阈值 {_max_attempts}，"
            f"惩罚率 {_penalty_rate:.0%}，级别={_warning_level}"
        )
        # 高危级别额外阻断
        _overfitting_critical = _warning_level == "CRITICAL"

        # ══ 邻近参数抖动自检（Parameter Robustness Check）══
        # Sharpe > 2.0 时自动触发 ±10% 参数扰动测试
        _robust_pass = True
        try:
            from BackTrading.parameter_robustness import run_robustness_check as _robust_check
            _robust_report = _robust_check(
                kline_df, best_params, _pun_sharpe, config, ecfg,
            )
            if _robust_report.triggered and not _robust_report.overall_robust:
                _robust_pass = False
                if _robust_report.warning_level == "CRITICAL":
                    logger.critical(
                        f"[参数稳健性] 🔴 CRITICAL: {len(_robust_report.failed_params)} 个参数扰动后 "
                        f"Sharpe 断崖式下跌，策略不具备统计稳健性: {_robust_report.failed_params}"
                    )
                else:
                    logger.warning(
                        f"[参数稳健性] ⚠️ {len(_robust_report.failed_params)} 个参数扰动后 "
                        f"Sharpe 显著下跌，建议谨慎: {_robust_report.failed_params}"
                    )
        except Exception as e:
            logger.warning(f"[参数稳健性] 自检异常: {e}，不阻断（建议人工复核）")

        # ── 样本外衰减校验（审计 gate：IS vs OOS 夏普/索提诺衰减 ≤ 30%） ──
        from BackTrading.overfitting import validate_oos_decay as _validate_oos_decay

        _oos_decay_pass = True
        try:
            # 终验 OOS 段：holdout 启用时用末段独立 holdout（WFO 全程禁触），
            # 否则回退旧逻辑从全周期净值尾部切 OOS（自引用）
            if _holdout_active and _holdout_days > 0:
                _oos_n = _holdout_days
                _decay_tag = "独立Holdout"
            else:
                _oos_n = max(bt.OUT_OF_SAMPLE_DAYS, 20)
                _decay_tag = "自引用回退"
            _eq = pd.DataFrame(equity_curve) if isinstance(equity_curve, list) else equity_curve
            if not _eq.empty and "time" in _eq.columns:
                # 确保日期为字符串统一比较
                if pd.api.types.is_datetime64_any_dtype(_eq["time"]):
                    _eq = _eq.copy()
                    _eq["time"] = _eq["time"].dt.strftime("%Y-%m-%d")

                _all_dates = sorted(_eq["time"].unique())
                _total_td = len(_all_dates)
                _is_end = max(_total_td - _oos_n, 1)
                _is_dates = set(_all_dates[:_is_end])
                _oos_dates = set(_all_dates[_is_end:])

                if len(_oos_dates) >= 2:
                    _is_curve = _eq[_eq["time"].isin(_is_dates)]
                    _oos_curve = _eq[_eq["time"].isin(_oos_dates)]

                    _report = _validate_oos_decay(
                        _is_curve, _oos_curve,
                        is_days=len(_is_dates),
                        oos_days=len(_oos_dates),
                    )
                    if not _report.passed:
                        _oos_decay_pass = False
                        logger.warning(
                            f"[OOS衰减校验][{_decay_tag}] FAIL | IS_Sharpe={_report.is_sharpe:.2f} → "
                            f"OOS_Sharpe={_report.oos_sharpe:.2f} (衰减 {_report.sharpe_decay:.1%}) | "
                            f"IS_Sortino={_report.is_sortino:.2f} → OOS_Sortino={_report.oos_sortino:.2f} "
                            f"(衰减 {_report.sortino_decay:.1%})"
                        )
                        logger.warning(f"[OOS衰减校验][{_decay_tag}] {_report.reason}")
                    else:
                        logger.info(
                            f"[OOS衰减校验][{_decay_tag}] PASS | IS_Sharpe={_report.is_sharpe:.2f} → "
                            f"OOS_Sharpe={_report.oos_sharpe:.2f} (衰减 {_report.sharpe_decay:.1%}) | "
                            f"IS_Sortino={_report.is_sortino:.2f} → OOS_Sortino={_report.oos_sortino:.2f} "
                            f"(衰减 {_report.sortino_decay:.1%})"
                        )
                else:
                    logger.info(f"[OOS衰减校验] OOS 交易日仅 {len(_oos_dates)} 天 < 2 天，跳过")
            else:
                logger.info("[OOS衰减校验] 净值曲线为空，跳过")
        except Exception as e:
            logger.warning(f"[OOS衰减校验] 执行异常: {e}，不阻断（建议人工复核）")

        if not _oos_decay_pass:
            logger.warning("=" * 50)
            logger.warning("[OOS衰减校验] 未通过 —— 参数组不予写入 config.ini，结果已废弃")
            logger.warning("=" * 50)

        cal_result = CalibrationResult(
            params=best_params,
            score=report_sharpe,
            sharpe=report_sharpe,
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

        # ── 统一参数采纳门控（P0-5 审计修复） ──
        # save_calibration（calibration_result.json）与 write_calibration_to_ini
        # （生产 config.ini）共用同一门控：统计显著性 + OOS 衰减 + PBO/DSR 硬性
        # 拒绝 + 多重测试惩罚 + 稳健性 + 模拟验证，杜绝门控不一致导致 PBO 过拟合
        # 参数集仍落盘进生产。
        _pbo_gate = pbo <= 0.05
        _dsr_gate = dsr >= 0.5
        if not _pbo_gate:
            logger.warning(
                f"[过拟合防护] PBO={pbo:.4f} > 0.05 阈值，参数组统计显著性不足，拒绝采纳"
            )
        if not _dsr_gate:
            logger.warning(
                f"[过拟合防护] DSR={dsr:.4f} < 0.5 阈值，缩水 Sharpe 比过低，拒绝采纳"
            )
        _gate_pass, _gate_reasons = _acceptance_gate(
            promote=_promote,
            oos_decay_pass=_oos_decay_pass,
            overfitting_critical=_overfitting_critical,
            sig_pass=_sig_pass,
            robust_pass=_robust_pass,
            pbo_gate=_pbo_gate,
            dsr_gate=_dsr_gate,
        )
        if _gate_pass:
            save_calibration(cal_result)
        else:
            logger.warning("=" * 50)
            logger.warning(
                "[采纳门控] 参数组未通过统一采纳门控，calibration_result.json 不予保存: "
                + "；".join(_gate_reasons)
            )
            logger.warning("=" * 50)

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

        if _gate_pass:
            write_calibration_to_ini(best_params)
            apply_calibration_to_config(config)
            logger.info("模拟验证通过，参数已写入 config.ini 并生效")
        else:
            # P0-5：同一统一门控——任一关键项未通过，config.ini 保持不变
            logger.warning("=" * 50)
            logger.warning(
                "[采纳门控] config.ini 参数保持不变（结果可作回测报告参考，已记录数据库）: "
                + "；".join(_gate_reasons)
            )
            logger.warning("=" * 50)
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
            data_version=_data_version,
        )

        updated_sections = set()
        for k in best_params:
            if k in CALIB_PARAM_MAP:
                updated_sections.add(CALIB_PARAM_MAP[k][0])
        if _gate_pass:
            logger.info(f"  寻优结果已采纳并写入 calibration_result.json + config.ini [{', '.join(sorted(updated_sections))}]")
        else:
            logger.info("  寻优结果未通过统一采纳门控，calibration_result.json / config.ini 未写入")
        alert.on_success(cal_result)
        return cal_result

    except Exception as exc:
        logger.opt(exception=True).error(f"回测管线失败: {exc}")
        # A2：管线级兜底快照（窗口级快照由 meta_optimizer 内部落盘）
        import traceback as _tb

        _snap_id = save_failure_snapshot(
            ohlcv=kline_df if kline_df is not None and not kline_df.empty else None,
            metric_name="pipeline",
            error_code="PIPELINE_FAILED",
            error_message=str(exc),
            traceback_text=_tb.format_exc(),
        )
        if _snap_id:
            logger.error(f"回测管线失败快照已保存 | snapshot_id={_snap_id} | run_id={_run_id}")
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
                data_version=_data_version,
            )
        except Exception as log_err:
            logger.warning(f"回测失败记录写入异常: {log_err}")
        alert.on_failure(exc, snapshot_id=_snap_id)
        return None
    finally:
        _release_run_lock()


def _compute_kline_data_version(engine: Any, kline_df: pd.DataFrame | None = None) -> str:
    """数据版本标识：用于信号缓存隔离与 calibration_log 调度。

    优先从内存 DataFrame 计算（runner 主路径），消除 fetch→version 之间的
    竞态窗口（IncrementalSyncEngine 不回听 advisory lock，可能并发写入）；
    无 DataFrame 时回退到数据库查询（should_rerun 调度场景）。

    版本仅使用 max(trade_date) 作为粗粒度失效信号：
    - COUNT 已移除：新增股票全量历史数据会剧烈变动 COUNT，但存量 OHLC
      内容不变，导致不必要的整库重跑；细粒度内容变更由
      _data_fingerprint 的 OHLCV 采样 hash + 行数覆盖。
    - 仅 max(trade_date) 变动才失效：新交易日到达即触发重跑，符合业务直觉。
    """
    # 路径 ①：从内存 DataFrame 计算（与消费数据严格一致，无竞态）
    if kline_df is not None and not kline_df.empty and "trade_date" in kline_df.columns:
        try:
            _max_date = kline_df["trade_date"].max()
            # 安全归一化为 ISO 日期字符串
            _ts = pd.Timestamp(_max_date)
            return _ts.strftime("%Y-%m-%d")
        except Exception as exc:
            logger.warning(f"从 DataFrame 计算数据版本失败: {exc}")

    # 路径 ②：从数据库查询（should_rerun 调度场景，无 DataFrame）
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT COALESCE(MAX(trade_date)::text, '') FROM stock_daily_kline"
            )).fetchone()
        if row is None or not row[0]:
            return ""
        return str(row[0])
    except Exception as exc:
        logger.warning(f"计算 kline 数据版本失败: {exc}")
        return ""


def _resolve_symbols(engine: Any, config: Config | None = None) -> list[str]:
    """解析股票列表，仅保留沪深主板（60x/00x 开头）。

    为消除生存者偏差，股票池包含所有曾有过交易记录的股票（含已退市）。
    ST/*ST/退市的逐日动态剔除由引擎配合 stock_st_history 完成，此处不做静态剔除。
    系统仅覆盖沪深主板，创业板/科创板/北交所已从业务中剔除。
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
    # 硬编码主板过滤：仅保留 60x / 00x 开头代码
    before = len(raw)
    raw = [s for s in raw if s.replace("sh", "").replace("sz", "").startswith(("60", "00"))]
    if len(raw) < before:
        logger.info(f"主板过滤后剩余: {len(raw)} / {before} 只")
    # 注意：不再做静态 ST 剔除，逐日动态剔除由引擎根据 stock_st_history 完成
    # EXCLUDE_ST 控制 ST/*ST 日是否剔除（退市日无条件剔除，见 engine/core.py）
    if config is not None and config.app_config.backtest.EXCLUDE_ST:
        logger.info("EXCLUDE_ST=True：引擎按 stock_st_history 逐日剔除 ST/*ST 日的买入与持仓（退市日无条件剔除）")
    elif config is not None:
        logger.info("EXCLUDE_ST=False：ST/*ST 股全程参与交易（退市日仍强制剔除）")
    if not raw:
        logger.warning("回测股票池为空，请检查数据库 stock_daily_kline 表")
    return sorted({CodeNormalizer.add_market_prefix(s) if not s.startswith(("sh", "sz")) else s for s in raw})


def _load_st_history(engine: Any, symbols: list[str], start_date: str, end_date: str) -> dict[str, dict[str, tuple[bool, bool]]]:
    """
    加载股票在日期范围内的 ST/退市状态历史（PIT 逐日序列）。

    P0-5 审计修复：
      - SQL 注入：旧实现字符串插值拼接 symbol 进 SQL（sym_placeholders），
        已改为 DataManager.StPitSync.load_st_pit 的参数化 = ANY(:syms)。
      - 非 PIT：旧表常为最近快照，历史 ST 期缺失导致 5% 涨跌幅被错按 10%、
        ST 禁买/强平失效；数据由 sync_st_pit 回填的全历史 PIT 序列提供。

    Returns:
        dict: {symbol: {trade_date: (is_st, is_delisting)}}
    """
    from DataManager.StPitSync import load_st_pit

    return load_st_pit(engine, symbols, start_date, end_date)


def _load_listing_days(engine: Any, symbols: list[str], start_date: str) -> dict[str, str]:
    """
    加载股票上市日期（显式注入 IPO 日期，P0-6 ④）。

    P0-6 审计修复：引擎不再从行情数据推断上市日期（数据缺口会误判新股，
    错误激活"注册制前 5 日无涨跌幅"豁免）；上市日期由 stock_listing_days 表
    （AkShare stock_info_a_code_name 上市日期列）提供，缺失时豁免整体停用。
    """
    from DataManager.ListingDaysSync import load_listing_days

    return load_listing_days(engine, symbols, start_date)


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
            # 容差 10 个日历日：预热起点恰逢非交易日/停牌时最早数据略晚于起点属正常，
            # 避免回填完成后因 1-2 天边界差反复触发全市场强制回填
            _warmup_tolerance = 10
            _warmup_cutoff = (
                pd.Timestamp(buffer_start_iso) + timedelta(days=_warmup_tolerance)
            ).strftime("%Y-%m-%d")
            need_warmup = [
                s for s in existing_symbols
                if s in first_by_symbol and first_by_symbol[s] is not None
                and pd.Timestamp(first_by_symbol[s]).strftime("%Y-%m-%d") > _warmup_cutoff
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

    主路径（P2.3 稳健中位数）：优先取 DM 检验显著通过（p<0.05）且
    OOS Sharpe>0 的窗口参数中位数——单窗口 Sharpe 尖峰多为噪声，
    中位数对离群窗口稳健；DM 显著过滤保证"寻优确实优于基准"。
    兜底路径：DM 数据缺失时退化为"OOS 为正窗口的中位数"；
    窗口不足时退回原 Sharpe 加权 Top-N 均值；仍失败则用配置中位数。

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
            # P0-7 ②：组合参数兜底优先取校准覆写值（与日频路径 EngineConfig 一致）
            "buy_threshold": bt.BUY_THRESHOLD,
            "max_holdings": bt.MAX_HOLDINGS,
        }

    if wf_result.empty or "params" not in wf_result.columns:
        logger.warning("Walk-Forward 结果为空或缺少 params 列，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    rows = wf_result.dropna(subset=["sharpe_ratio"])
    if rows.empty:
        logger.warning("Walk-Forward 所有组合 Sharpe 均为 NaN，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    def _median_params(rows_: pd.DataFrame) -> dict[str, float]:
        """对行内 params 取逐参数中位数（稳健主路径核心）。"""
        all_params = [r["params"] for _, r in rows_.iterrows() if isinstance(r["params"], dict)]
        if not all_params:
            return {}
        keys = all_params[0].keys()
        median_params: dict[str, float] = {}
        for k in keys:
            vals = sorted(p[k] for p in all_params)
            median_params[k] = vals[len(vals) // 2]
        return median_params

    # ── 主路径：DM 显著通过（p<0.05）且 OOS>0 的窗口 → 参数中位数 ──
    if "dm_p_value" in rows.columns:
        dm_rows = rows[
            (rows["dm_p_value"] < 0.05) & (rows["sharpe_ratio"] > 0)
        ]
        if len(dm_rows) >= 2:
            med = _median_params(dm_rows)
            if med:
                logger.info(
                    f"[稳健中位数主路径] DM 显著窗口 {len(dm_rows)} 个 "
                    f"(p<0.05 且 OOS>0)，取参数中位数: {med}"
                )
                return med
        logger.warning(
            f"DM 显著窗口仅 {len(dm_rows)} 个(<2)，退化到 OOS 正收益窗口中位数"
        )

    # ── 次级路径：OOS Sharpe>0 窗口的参数中位数（无 DM 列时直接走这里） ──
    pos_rows = rows[rows["sharpe_ratio"] > 0]
    if len(pos_rows) >= 2:
        med = _median_params(pos_rows)
        if med:
            logger.info(f"[稳健中位数] OOS 正收益窗口 {len(pos_rows)} 个，取参数中位数: {med}")
            return med

    # ── 兜底：原 Sharpe 加权 Top-N 均值 ──
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
