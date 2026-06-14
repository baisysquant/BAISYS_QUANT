from __future__ import annotations

import sys

from loguru import logger

from LogicAnalyzer.StockAnalysisCoordinator import StockAnalysisCoordinatorFactory


def main() -> None:
    """BAISYS_QUANT 统一入口 — 回测校准 + 每日复盘一体化。

    CLI 参数:
        --force              强制重新回测（忽略频率检查）
        --pipeline-only      仅执行每日复盘管线，跳过回测
        --backtest-only      仅执行回测，跳过每日复盘
        --schedule           启动回测定时调度器（常驻进程）
    """
    import io

    # 强制 Windows UTF-8 输出，防止 UnicodeEncodeError
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    args = [a.lstrip("-").replace("-", "_") for a in sys.argv[1:]]
    force = "force" in args
    pipeline_only = "pipeline_only" in args
    backtest_only = "backtest_only" in args
    schedule = "schedule" in args

    logger.info("=" * 80)
    logger.info("BAISYS_QUANT - A股量化复盘分析系统")
    logger.info("=" * 80)

    # ── 回测定时调度器 ──────────────────────────────────────
    if schedule:
        from Backtesting.runner import start_scheduler

        start_scheduler()
        return

    # ── 回测校准阶段 ────────────────────────────────────────
    if not pipeline_only:
        from Backtesting.calibration_log import ensure_table, get_last_run, should_rerun
        from ConfigParser import Config
        from DataManager.DbEngine import get_engine

        cfg = Config()
        bt = cfg.app_config.backtest

        if bt.ENABLED:
            engine = get_engine(cfg)
            ensure_table(engine)

            last = get_last_run(engine)
            should, reason = should_rerun(last, bt.OPTIMIZE_FREQUENCY)

            if should or force:
                if force:
                    logger.info("--force 指定，强制回测校准")
                else:
                    logger.info(reason)
                    logger.info("到期，校准完成后自动进入复盘流程")

                from Backtesting.runner import run_backtest_pipeline

                result = run_backtest_pipeline(cfg, force=True)
                if result is None:
                    logger.warning("回测未完成，使用现有参数继续复盘")
                else:
                    logger.info("回测校准完成，参数已写入 config.ini")
                    # 重新加载配置，让复盘模块使用最新参数
                    cfg = Config()
            else:
                logger.info(f"回测未到期（{reason}），跳过校准，直接复盘")
        else:
            logger.info("回测未启用 (BACKTEST.enabled=false)，跳过校准")

    # ── 每日复盘阶段 ────────────────────────────────────────
    if not backtest_only:
        logger.info("")
        try:
            coordinator = StockAnalysisCoordinatorFactory.create(config_file="config.ini")
            coordinator.run()

            logger.info("")
            logger.info("=" * 80)
            logger.info("[OK] 分析流程完成！")
            logger.info("=" * 80)
            logger.info("   - Excel报告: temp_data/审计报告_YYYYMMDD.xlsx")
            logger.info("   - 日志文件: logs/Corenews_Main_YYYYMMDD.log")

        except Exception as e:
            logger.error("")
            logger.error("=" * 80)
            logger.error(f"[FAIL] 分析流程失败: {type(e).__name__}")
            logger.error(f"   错误信息: {e}")
            logger.error("=" * 80)
            raise


if __name__ == "__main__":
    main()
