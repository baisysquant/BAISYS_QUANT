from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text

from Backtesting.alert import BacktestAlert
from Backtesting.calibration import (
    CALIB_PARAM_MAP,
    CalibrationResult,
    apply_calibration_to_config,
    load_calibration,
    run_walk_forward,
    save_calibration,
    write_calibration_to_ini,
)
from Backtesting.calibration_log import ensure_table, get_last_run, record_run, should_rerun
from Backtesting.data_provider import BacktestDataProvider
from Backtesting.prepare import prepare_backtest_data
from ConfigParser import Config
from DataManager.DbEngine import get_engine


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

    try:
        symbols = _resolve_symbols(engine)
        logger.info(f"  股票数量: {len(symbols)}")

        kline_df = _fetch_kline(engine, symbols, bt.BACKTEST_START_DATE, config)
        if kline_df.empty:
            logger.warning("K 线数据为空，跳过回测")
            return None

        logger.info(f"  K 线行数: {len(kline_df)}")

        total_trading_days = int(kline_df["trade_date"].nunique())
        train_period = max(total_trading_days - bt.OUT_OF_SAMPLE_DAYS, 30)
        logger.info(f"  交易日数: {total_trading_days} | 训练窗口: {train_period}天")

        prepared = prepare_backtest_data(kline_df)
        signal_prefixes = ('进场', '退出', '风险', '止损', '综合')
        signal_cols = [c for c in prepared.columns if c.startswith(signal_prefixes)]
        logger.info(f"  预计算信号列: {signal_cols}")

        wf_result = run_walk_forward(
            kline_df=prepared,
            train_period=train_period,
            test_period=bt.OUT_OF_SAMPLE_DAYS,
            initial_cash=bt.INITIAL_CASH,
            show_progress=True,
        )
        logger.info(f"  Walk-Forward 片段数: {len(wf_result)}")

        best_params = _extract_best_params(wf_result)
        logger.info(f"  最佳参数: {best_params}")

        sharpe = float(wf_result.iloc[0].get("sharpe_ratio", 0))
        total_return = float(wf_result.iloc[0].get("total_return", 0))
        max_dd = float(wf_result.iloc[0].get("max_drawdown", 0))

        cal_result = CalibrationResult(
            params=best_params,
            score=sharpe,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_return,
            timestamp=datetime.now().isoformat(),
        )
        save_calibration(cal_result)
        write_calibration_to_ini(best_params)
        apply_calibration_to_config(config)

        record_run(
            engine=engine,
            frequency=bt.OPTIMIZE_FREQUENCY,
            backtest_start_date=bt.BACKTEST_START_DATE,
            out_of_sample_days=bt.OUT_OF_SAMPLE_DAYS,
            initial_cash=bt.INITIAL_CASH,
            params=best_params,
            sharpe=sharpe,
            total_return=total_return,
            max_drawdown=max_dd,
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
        except Exception:
            pass
        alert.on_failure(exc)
        return None


def _resolve_symbols(engine: Any) -> list[str]:
    """解析全 A 股股票列表（~5000 只），从 stock_basic_info_sw 表获取。"""
    from UtilsManager.CodeNormalizer import CodeNormalizer

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT stock_code FROM stock_basic_info_sw
            ORDER BY stock_code
        """)).fetchmany(6000)
    if rows:
        normalized = sorted({
            CodeNormalizer.add_market_prefix(str(r[0]).strip().zfill(6))
            for r in rows
        })
        if normalized:
            logger.info(f"从本地股票信息表获取 {len(normalized)} 只股票")
            return normalized

    logger.warning("股票信息表为空，回退到 stock_daily_kline 已有数据")
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT symbol FROM stock_daily_kline
        """)).fetchmany(5000)
    return sorted({r[0] for r in rows})


def _fetch_kline(
    engine: Any,
    symbols: list[str],
    backtest_start_date: str,
    config: Config,
) -> pd.DataFrame:
    from Backtesting.sync import ensure_table

    ensure_table(engine)

    # 补齐缺失股票的历史 K 线
    _sync_missing_stocks(engine, symbols, config, backtest_start_date)

    end = date.today()
    start = datetime.strptime(backtest_start_date, "%Y%m%d").date()

    provider = BacktestDataProvider(engine)
    df = provider.get_kline(symbols, start_date=start.isoformat(), end_date=end.isoformat())
    if df.empty:
        return df
    df = df.sort_values(["symbol", "trade_date"])
    return df


def _sync_missing_stocks(engine: Any, symbols: list[str], config: Config, backtest_start_date: str) -> None:
    """补齐 + 刷新 stock_daily_kline 数据。检查每只股票数据是否齐全，检测除权除息并重拉。"""
    from DataManager.IncrementalSyncEngine import IncrementalSyncEngine

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
        n = syncer.sync_all(missing)
        logger.info(f"  补齐完成，新增 {n} 行")

    # 对所有股票执行增量刷新：检查最新日期、除权除息检测
    logger.info(f"  检查 {len(symbols)} 只股票数据完整性...")
    total = syncer.sync_all(symbols)
    logger.info(f"  刷新完成，新增 {total} 行")


def _extract_best_params(wf_result: pd.DataFrame) -> dict[str, float]:
    if wf_result.empty or "params" not in wf_result.columns:
        return {}
    best_row = wf_result.iloc[0]
    if isinstance(best_row["params"], dict):
        return {k: float(v) for k, v in best_row["params"].items()}
    return {}


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
        run_backtest_pipeline(config)

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
        python -m Backtesting.runner            # 执行回测（交互式判断是否已过期）
        python -m Backtesting.runner --force     # 强制重新回测
        python -m Backtesting.runner --schedule  # 启动常驻调度器
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
