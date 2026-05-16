"""
股票分析主程序（新架构入口）

这是BAISYS_QUANT系统的主入口文件，使用新的分层架构。

新架构特点：
- 模块化设计：数据获取、处理、分析、报告生成分离
- 依赖注入：通过构造函数注入依赖，便于测试
- 高内聚低耦合：每个服务类职责单一
- 易于扩展：添加新功能只需新增服务类

旧版本（2005行）已迁移到以下模块：
- LogicAnalyzer/DataAcquisitionService.py - 数据获取
- DataManager/DataProcessingService.py - 数据处理
- LogicAnalyzer/AnalysisService.py - 业务分析
- DataManager/ReportService.py - 报告生成
- StockAnalysisCoordinator.py - 主协调器（根目录）
- UtilsManager/CodeNormalizer.py - 代码标准化
- UtilsManager/PriceExtractor.py - 价格提取

使用方法：
    python MainShareAnalysis.py

或者直接使用协调器：
    python StockAnalysisCoordinator.py
"""

from StockAnalysisCoordinator import StockAnalysisCoordinator


def main():
    """
    主函数 - 执行完整的股票分析流程
    
    流程步骤：
    1. 同步历史数据到数据库
    2. 获取待分析股票代码列表
    3. 获取所有原始数据（资金流、技术指标、行业板块等）
    4. 获取K线数据并提取最新价格
    5. 处理技术指标信号（MACD、KDJ、CCI、RSI、BOLL）
    6. 运行行业深度分析
    7. 处理均线突破数据并筛选多头排列
    8. 合并和处理所有数据
    9. 映射行业信号到个股
    10. 剔除弱势且加速下跌的个股
    11. 生成Excel审计报告
    12. 同步结果到数据库
    
    Raises:
        DatabaseConnectionError: 数据库连接失败
        ReportGenerationError: 报告生成失败
        Exception: 其他未预期的错误
    """
    print("=" * 80)
    print("BAISYS_QUANT - A股量化复盘分析系统")
    print("新架构版本 v2.0")
    print("=" * 80)
    print()
    
    try:
        # 创建协调器实例（自动初始化所有服务）
        coordinator = StockAnalysisCoordinator(config_file="config.ini")
        
        # 执行完整的分析流程
        coordinator.run()
        
        print()
        print("=" * 80)
        print("✅ 分析流程完成！")
        print("=" * 80)
        print()
        print("📊 查看报告:")
        print("   - Excel报告: temp_data/审计报告_YYYYMMDD.xlsx")
        print("   - 日志文件: logs/Corenews_Main_YYYYMMDD.log")
        print("   - 技术指标: temp_data/*_Signals_YYYYMMDD.txt")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ 分析流程失败: {type(e).__name__}")
        print(f"   错误信息: {e}")
        print("=" * 80)
        print()
        print("💡 建议:")
        print("   1. 检查配置文件 config.ini")
        print("   2. 查看日志文件获取详细信息")
        print("   3. 确保数据库连接正常")
        print()
        raise


if __name__ == "__main__":
    main()
