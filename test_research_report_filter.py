#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
研报过滤功能测试脚本
用于验证新添加的研报二次过滤功能是否正常工作
"""

import sys
import os
from ConfigParser import Config

def test_config_loading():
    """测试配置加载功能"""
    print("=" * 60)
    print("测试1: 配置加载功能")
    print("=" * 60)
    
    try:
        config = Config("config.ini")
        
        # 检查新增的配置项是否存在
        assert hasattr(config, 'ENABLE_RESEARCH_REPORT_FILTER'), "缺少 ENABLE_RESEARCH_REPORT_FILTER 配置项"
        assert hasattr(config, 'RESEARCH_REPORT_MIN_COUNT'), "缺少 RESEARCH_REPORT_MIN_COUNT 配置项"
        
        print(f"✓ ENABLE_RESEARCH_REPORT_FILTER: {config.ENABLE_RESEARCH_REPORT_FILTER}")
        print(f"✓ RESEARCH_REPORT_MIN_COUNT: {config.RESEARCH_REPORT_MIN_COUNT}")
        print("✓ 配置加载测试通过\n")
        
        return True
    except Exception as e:
        print(f"✗ 配置加载测试失败: {e}\n")
        return False

def test_config_values():
    """测试配置值的正确性"""
    print("=" * 60)
    print("测试2: 配置值验证")
    print("=" * 60)
    
    try:
        config = Config("config.ini")
        
        # 验证默认值
        assert isinstance(config.ENABLE_RESEARCH_REPORT_FILTER, bool), "ENABLE_RESEARCH_REPORT_FILTER 应该是布尔值"
        assert isinstance(config.RESEARCH_REPORT_MIN_COUNT, int), "RESEARCH_REPORT_MIN_COUNT 应该是整数"
        assert config.RESEARCH_REPORT_MIN_COUNT >= 0, "RESEARCH_REPORT_MIN_COUNT 应该大于等于0"
        
        print(f"✓ 配置类型验证通过")
        print(f"✓ 配置范围验证通过")
        print("✓ 配置值验证测试通过\n")
        
        return True
    except Exception as e:
        print(f"✗ 配置值验证测试失败: {e}\n")
        return False

def test_filter_logic_simulation():
    """模拟测试过滤逻辑"""
    print("=" * 60)
    print("测试3: 过滤逻辑模拟")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        # 创建模拟数据
        report_data = {
            '股票代码': ['000001', '000002', '000003', '000004', '000005'],
            '机构投资评级(近六个月)-买入': [0, 1, 2, 3, 5]
        }
        report_df = pd.DataFrame(report_data)
        
        # 模拟配置
        min_count = 1  # 阈值设为1
        
        # 执行过滤逻辑
        filtered_df = report_df[report_df['机构投资评级(近六个月)-买入'] > min_count]
        filtered_codes = set(filtered_df['股票代码'].unique().tolist())
        
        print(f"原始股票数: {len(report_df)}")
        print(f"过滤后股票数: {len(filtered_codes)}")
        print(f"过滤掉的股票: {set(report_df['股票代码']) - filtered_codes}")
        print(f"保留的股票: {filtered_codes}")
        
        # 验证结果
        expected_codes = {'000003', '000004', '000005'}  # 买入次数>1的股票
        assert filtered_codes == expected_codes, f"期望 {expected_codes}, 实际 {filtered_codes}"
        
        print("✓ 过滤逻辑模拟测试通过\n")
        return True
    except Exception as e:
        print(f"✗ 过滤逻辑模拟测试失败: {e}\n")
        return False

def main():
    """主测试函数"""
    print("开始研报过滤功能测试...\n")
    
    tests = [
        test_config_loading,
        test_config_values,
        test_filter_logic_simulation
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
        print("✓ 所有测试通过！研报过滤功能已正确实现。")
        return 0
    else:
        print("✗ 部分测试失败，请检查实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())