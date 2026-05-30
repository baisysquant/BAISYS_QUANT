"""
技术指标调试脚本
用于诊断技术指标计算失败的原因
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sqlalchemy import create_engine, text
from ConfigParser import Config

def debug_technical_indicators():
    """调试技术指标处理流程"""
    
    # 配置初始化
    config_path = "config.ini"
    config = Config(config_path)
    DB_URI = f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
    
    print("=" * 80)
    print("【技术指标诊断工具】")
    print("=" * 80)
    
    # 1. 检查数据库连接
    print("\n[步骤1] 检查数据库连接...")
    try:
        engine = create_engine(DB_URI)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM stock_daily_kline"))
            count = result.scalar()
            print(f"✅ 数据库连接成功，K线表数据量: {count}")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return
    
    # 2. 获取最新交易日期的股票代码
    print("\n[步骤2] 获取最新交易日期...")
    try:
        with engine.connect() as conn:
            query = text("SELECT MAX(trade_date) FROM stock_daily_kline")
            latest_date = conn.execute(query).scalar()
            print(f"✅ 最新交易日期: {latest_date}")
            
            # 获取该日期的股票代码
            query = text(f"""
                SELECT DISTINCT symbol 
                FROM stock_daily_kline 
                WHERE trade_date = '{latest_date}'
                LIMIT 5
            """)
            symbols = conn.execute(query).fetchall()
            print(f"✅ 该日期有 {len(symbols)} 只股票，样本: {[s[0] for s in symbols]}")
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return
    
    # 3. 检查历史数据
    print("\n[步骤3] 检查单只股票的历史数据...")
    if symbols:
        test_symbol = symbols[0][0]
        print(f"测试股票: {test_symbol}")
        
        try:
            with engine.connect() as conn:
                query = text(f"""
                    SELECT trade_date, symbol, "open", "close", high, low, volume
                    FROM stock_daily_kline 
                    WHERE symbol = '{test_symbol}'
                    ORDER BY trade_date DESC
                    LIMIT 10
                """)
                df = pd.read_sql(query, conn)
                print(f"✅ 查询到 {len(df)} 条历史数据")
                print(f"  列名: {list(df.columns)}")
                print(f"  数据类型:\n{df.dtypes}")
                print(f"  最近数据:\n{df.head(3)}")
                
                # 检查数据完整性
                print(f"\n  数据完整性检查:")
                print(f"    - close 非空率: {df['close'].notna().sum() / len(df) * 100:.1f}%")
                print(f"    - open 非空率: {df['open'].notna().sum() / len(df) * 100:.1f}%")
                print(f"    - high 非空率: {df['high'].notna().sum() / len(df) * 100:.1f}%")
                print(f"    - low 非空率: {df['low'].notna().sum() / len(df) * 100:.1f}%")
                
        except Exception as e:
            print(f"❌ 查询失败: {e}")
    
    # 4. 测试技术指标计算
    print("\n[步骤4] 测试技术指标计算...")
    if symbols and len(df) >= 30:
        try:
            from LogicAnalyzer.SignalManager import _process_single_stock
            
            # 提取纯代码
            pure_code = test_symbol[2:] if test_symbol.startswith(('sh', 'sz', 'bj')) else test_symbol
            
            print(f"开始处理: {test_symbol} (纯代码: {pure_code})")
            result = _process_single_stock(test_symbol, df, "9186", config)
            
            print(f"\n技术指标计算结果:")
            print(f"  - MACD_12269: {'✅ 有数据' if result['macd_12269'] else '❌ 无数据'}")
            print(f"  - MACD_9186: {'✅ 有数据' if result['macd_9186'] else '❌ 无数据'}")
            print(f"  - MACD_COMBINED_DIVERGENCE: {'✅ 有数据' if result['macd_divergence'] else '❌ 无数据'}")
            print(f"  - KDJ: {'✅ 有数据' if result['kdj'] else '❌ 无数据'}")
            print(f"  - CCI: {'✅ 有数据' if result['cci'] else '❌ 无数据'}")
            print(f"  - RSI: {'✅ 有数据' if result['rsi'] else '❌ 无数据'}")
            print(f"  - BOLL: {'✅ 有数据' if result['boll'] else '❌ 无数据'}")
            print(f"  - MACD_momentum: {'✅ 有数据' if result['macd_momentum'] else '❌ 无数据'}")
            print(f"  - MACD_full_bull: {'✅ 有数据' if result['macd_full_bull'] else '❌ 无数据'}")
            
            # 打印详细结果
            print(f"\n详细结果:")
            for key, val in result.items():
                if val:
                    print(f"  {key}: {val}")
                    
        except Exception as e:
            print(f"❌ 技术指标计算失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. 完整流程测试
    print("\n[步骤5] 完整流程测试...")
    try:
        from StockAnalysisCoordinator import StockAnalysisCoordinator
        from DataCollection.CalendarManager import TradingCalendarAnalyzer
        from LogicAnalyzer.AnalysisService import AnalysisService
        from UtilsManager.LoggerManager import get_logger
        
        calendar_mgr = TradingCalendarAnalyzer()
        today_str = calendar_mgr.get_last_trading_day()
        logger = get_logger(config.LOG_DIR, f"debug_{today_str}.log", "INFO")
        
        # 获取K线数据
        with engine.connect() as conn:
            query = text(f"""
                SELECT symbol FROM stock_daily_kline 
                WHERE trade_date = '{latest_date}'
                LIMIT 10
            """)
            symbols_list = [row[0] for row in conn.execute(query).fetchall()]
            
            # 获取历史数据
            symbols_str = ",".join([f"'{s}'" for s in symbols_list])
            query = text(f"""
                SELECT * FROM stock_daily_kline 
                WHERE symbol IN ({symbols_str})
                ORDER BY trade_date
            """)
            hist_df = pd.read_sql(query, conn)
        
        # 测试信号处理
        analysis_service = AnalysisService(config, logger, engine)
        spot_data = pd.DataFrame()  # 空即可
        ta_signals = analysis_service.process_technical_signals(symbols_list, hist_df, spot_data)
        
        print(f"\n处理结果统计:")
        for key, df in ta_signals.items():
            if isinstance(df, pd.DataFrame):
                print(f"  {key}: {len(df)} 条")
            else:
                print(f"  {key}: {type(df)}")
                
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("【诊断完成】")
    print("=" * 80)

if __name__ == "__main__":
    debug_technical_indicators()
