from LogicAnalyzer.StockAnalysisCoordinator import StockAnalysisCoordinatorFactory
from UtilsManager.LoggerManager import get_logger


def main():
    """
    主函数 - 执行完整的股票分析流程
 

    Raises:
        DatabaseConnectionError: 数据库连接失败
        ReportGenerationError: 报告生成失败
        Exception: 其他未预期的错误
    """
    import sys
    import io

    # 强制在 Windows 终端下支持 UTF-8 编码，防止特殊 Unicode/Emoji 字符导致 UnicodeEncodeError
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    logger = get_logger()

    logger.info("=" * 80)
    logger.info("BAISYS_QUANT - A股量化复盘分析系统")
    logger.info("=" * 80)
    logger.info("")

    try:
        # 使用工厂类创建协调器实例（完全依赖注入）
        coordinator = StockAnalysisCoordinatorFactory.create(config_file="config.ini")

        # 执行完整的分析流程
        coordinator.run()

        logger.info("")
        logger.info("=" * 80)
        logger.info("[OK] 分析流程完成！")
        logger.info("=" * 80)
        logger.info("")
        logger.info("   - Excel报告: temp_data/审计报告_YYYYMMDD.xlsx")
        logger.info("   - 日志文件: logs/Corenews_Main_YYYYMMDD.log")
        logger.info("")

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"[FAIL] 分析流程失败: {type(e).__name__}")
        logger.error(f"   错误信息: {e}")
        logger.error("=" * 80)
        logger.error("")
        raise


if __name__ == "__main__":
    main()
