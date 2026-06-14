"""
Pandera 数据契约测试脚本

验证 Pandera Schema 定义和校验功能测试。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger

from DataManager.DataSchemas import SchemaValidator


def test_stock_basic_schema() -> None:
    """测试股票基础信息 Schema"""
    logger.info("=" * 60)
    logger.info("测试股票基础信息 Schema")
    logger.info("=" * 60)

    # 测试有效数据
    valid_data = pd.DataFrame(
        {
            "股票代码": ["000001", "600000", "300001"],
            "股票简称": ["平安银行", "浦发银行", "宁德时代"],
            "行业": ["银行", "银行", "电力设备"],
        }
    )

    is_valid, errors = SchemaValidator.validate_stock_basic(valid_data)
    if is_valid:
        logger.info("有效数据校验: [OK] 通过")
    else:
        logger.info(f"有效数据校验: [FAIL] 失败: {errors}")
    assert is_valid, "有效数据应该通过校验"

    # 测试无效数据 - 股票代码格式错误
    invalid_data = pd.DataFrame(
        {
            "股票代码": ["00001", "600000", "abc123"],
            "股票简称": ["平安银行", "浦发银行", "宁德时代"],
            "行业": ["银行", "银行", "电力设备"],
        }
    )

    is_valid, errors = SchemaValidator.validate_stock_basic(invalid_data)
    logger.info("无效数据校验: [OK] 正确检测到错误" if not is_valid else "[FAIL] 应该失败但通过了")
    logger.info(f"检测到的错误: {errors}")

    logger.info("[OK] 股票基础信息 Schema 测试完成\n")


def test_stock_price_schema() -> None:
    """测试股票价格 Schema"""
    logger.info("=" * 60)
    logger.info("测试股票价格 Schema")
    logger.info("=" * 60)

    valid_data = pd.DataFrame(
        {
            "股票代码": ["000001", "600000"],
            "最新价": [10.5, 8.2],
        }
    )

    is_valid, errors = SchemaValidator.validate_stock_price(valid_data)
    if is_valid:
        logger.info("有效数据校验: [OK] 通过")
    else:
        logger.info(f"有效数据校验: [FAIL] 失败: {errors}")
    assert is_valid, "有效数据应该通过校验"

    logger.info("[OK] 股票价格 Schema 测试完成\n")


def test_industry_board_schema() -> None:
    """测试行业板块 Schema"""
    logger.info("=" * 60)
    logger.info("测试行业板块 Schema")
    logger.info("=" * 60)

    valid_data = pd.DataFrame(
        {
            "板块名称": ["银行", "电力设备", "计算机"],
            "板块代码": ["BK0465", "BK0479", "BK0473"],
            "涨跌幅": [2.5, -1.2, 0.8],
        }
    )

    is_valid, errors = SchemaValidator.validate_industry_board(valid_data)
    if is_valid:
        logger.info("有效数据校验: [OK] 通过")
    else:
        logger.info(f"有效数据校验: [FAIL] 失败: {errors}")
    assert is_valid, "有效数据应该通过校验"

    logger.info("[OK] 行业板块 Schema 测试完成\n")


def test_main_cost_schema() -> None:
    """测试主力成本 Schema"""
    logger.info("=" * 60)
    logger.info("测试主力成本 Schema")
    logger.info("=" * 60)

    valid_data = pd.DataFrame(
        {
            "股票代码": ["000001", "600000"],
            "主力成本": [10.0, 7.5],
            "主力成本差价": [0.5, 0.7],
            "成本位置": ["低位", "中位"],
            "主力控盘强度": ["强", "中"],
        }
    )

    is_valid, errors = SchemaValidator.validate_main_cost(valid_data)
    if is_valid:
        logger.info("有效数据校验: [OK] 通过")
    else:
        logger.info(f"有效数据校验: [FAIL] 失败: {errors}")
    assert is_valid, "有效数据应该通过校验"

    logger.info("[OK] 主力成本 Schema 测试完成\n")


def test_coerce_types() -> None:
    """测试类型自动转换"""
    logger.info("=" * 60)
    logger.info("测试类型自动转换")
    logger.info("=" * 60)

    # 测试字符串数字自动转换为 float
    data = pd.DataFrame(
        {
            "股票代码": ["000001", "600000"],
            "最新价": ["10.5", "8.2"],
        }
    )

    is_valid, errors = SchemaValidator.validate_stock_price(data)
    if is_valid:
        logger.info("类型转换校验: [OK] 通过")
    else:
        logger.info(f"类型转换校验: [FAIL] 失败: {errors}")

    logger.info("[OK] 类型转换测试完成\n")


def main() -> int:
    """运行所有测试"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "Pandera 数据契约测试" + " " * 28 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")

    try:
        test_stock_basic_schema()
        test_stock_price_schema()
        test_industry_board_schema()
        test_main_cost_schema()
        test_coerce_types()

        logger.info("=" * 60)
        logger.info("[OK] 所有测试通过！")
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
