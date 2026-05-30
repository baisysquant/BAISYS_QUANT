#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K线数据获取天数配置测试脚本
用于验证新添加的K线数据获取天数配置功能是否正常工作
"""

import sys
import os
from ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer

def test_config_loading():
    """测试配置加载功能"""
    print("=" * 60)
    print("测试1: K线数据获取天数配置加载")
    print("=" * 60)
    
    try:
        config = Config("config.ini")
        
        # 检查新增的配置项是否存在
        assert hasattr(config, 'KLINE_HISTORY_DAYS'), "缺少 KLINE_HISTORY_DAYS 配置项"
        
        print(f"✓ KLINE_HISTORY_DAYS: {config.KLINE_HISTORY_DAYS}")
        print("✓ 配置加载测试通过\n")
        
        return True
    except Exception as e:
        print(f"✗ 配置加载测试失败: {e}\n")
        return False

def test_config_values():
    """测试配置值的正确性"""
    print("=" * 60)
    print("测试2: K线数据获取天数配置值验证")
    print("=" * 60)
    
    try:
        config = Config("config.ini")
        
        # 验证默认值
        assert isinstance(config.KLINE_HISTORY_DAYS, int), "KLINE_HISTORY_DAYS 应该是整数"
        assert config.KLINE_HISTORY_DAYS > 0, "KLINE_HISTORY_DAYS 应该大于0"
        assert config.KLINE_HISTORY_DAYS <= 1000, "KLINE_HISTORY_DAYS 不应该超过1000"
        
        print(f"✓ 配置类型验证通过")
        print(f"✓ 配置范围验证通过 (当前值: {config.KLINE_HISTORY_DAYS})")
        print("✓ 配置值验证测试通过\n")
        
        return True
    except Exception as e:
        print(f"✗ 配置值验证测试失败: {e}\n")
        return False

def test_date_calculation():
    """测试日期计算功能"""
    print("=" * 60)
    print("测试3: 交易日期偏移计算")
    print("=" * 60)
    
    try:
        calendar_mgr = TradingCalendarAnalyzer()
        
        # 获取最后交易日
        last_trading_day = calendar_mgr.get_last_trading_day()
        print(f"最后交易日: {last_trading_day}")
        
        # 测试往前推200天
        start_date_200 = calendar_mgr.get_trading_day_offset(-200, last_trading_day)
        print(f"往前推200天: {start_date_200}")
        
        # 测试往前推100天
        start_date_100 = calendar_mgr.get_trading_day_offset(-100, last_trading_day)
        print(f"往前推100天: {start_date_100}")
        
        # 验证日期格式
        assert len(start_date_200) == 10, f"日期格式错误: {start_date_200}"
        assert len(start_date_100) == 10, f"日期格式错误: {start_date_100}"
        
        # 验证日期的合理性（往前推的日期应该早于最后交易日）
        assert start_date_200 < last_trading_day, f"日期逻辑错误: {start_date_200} >= {last_trading_day}"
        assert start_date_100 < last_trading_day, f"日期逻辑错误: {start_date_100} >= {last_trading_day}"
        
        print("✓ 日期计算测试通过\n")
        return True
    except Exception as e:
        print(f"✗ 日期计算测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """集成测试：验证配置和日期计算的整合"""
    print("=" * 60)
    print("测试4: 配置与日期计算集成测试")
    print("=" * 60)
    
    try:
        config = Config("config.ini")
        calendar_mgr = TradingCalendarAnalyzer()
        
        # 使用配置的天数计算起始日期
        last_trading_day = calendar_mgr.get_last_trading_day()
        start_date = calendar_mgr.get_trading_day_offset(-config.KLINE_HISTORY_DAYS, last_trading_day)
        
        # 转换为YYYYMMDD格式
        start_date_str = start_date.replace('-', '')
        
        print(f"配置天数: {config.KLINE_HISTORY_DAYS}")
        print(f"最后交易日: {last_trading_day}")
        print(f"起始日期: {start_date_str}")
        print(f"日期范围: {start_date_str} 至 {last_trading_day.replace('-', '')}")
        
        # 验证结果
        assert len(start_date_str) == 8, f"日期格式错误: {start_date_str}"
        assert start_date_str < last_trading_day.replace('-', ''), f"日期逻辑错误"
        
        print("✓ 集成测试通过\n")
        return True
    except Exception as e:
        print(f"✗ 集成测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始K线数据获取天数配置功能测试...\n")
    
    tests = [
        test_config_loading,
        test_config_values,
        test_date_calculation,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("✓ 所有测试通过！K线数据获取天数配置功能已正确实现。")
        return 0
    else:
        print("✗ 部分测试失败，请检查实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())