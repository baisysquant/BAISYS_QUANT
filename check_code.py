#!/usr/bin/env python3
"""
代码质量检查脚本
使用 Ruff 和 MyPy 进行代码检查
"""

import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    """运行命令并返回结果"""
    print(f"\n{'=' * 60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("代码质量检查工具")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 检查是否安装了所需工具
    try:
        import ruff
        import mypy
    except ImportError as e:
        print(f"\n[FAIL] 缺少依赖: {e}")
        print("请运行: pip install ruff mypy")
        return 1

    print("\n[OK] 检查 Ruff 和 MyPy 已安装")

    # 运行 Ruff 检查
    ruff_code = run_command(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format", "concise"], "Ruff 代码规范检查"
    )

    # 运行 Ruff 格式化检查
    ruff_format_code = run_command([sys.executable, "-m", "ruff", "format", ".", "--check"], "Ruff 代码格式化检查")

    # 运行 MyPy 类型检查
    mypy_code = run_command(
        [sys.executable, "-m", "mypy", "StockAnalysisCoordinator.py", "MainShareAnalysis.py"], "MyPy 类型检查"
    )

    # 汇总结果
    print("\n" + "=" * 60)
    print("检查结果汇总:")
    print("=" * 60)
    print(f"  Ruff 检查: {'[OK] 通过' if ruff_code == 0 else '[FAIL] 失败'}")
    print(f"  Ruff 格式化: {'[OK] 通过' if ruff_format_code == 0 else '[FAIL] 失败'}")
    print(f"  MyPy 类型检查: {'[OK] 通过' if mypy_code == 0 else '[FAIL] 失败'}")
    print("=" * 60)

    # 提示修复方法
    if ruff_code != 0 or ruff_format_code != 0:
        print("\n 提示:")
        print("  - 运行 `ruff check . --fix` 自动修复部分问题")
        print("  - 运行 `ruff format .` 自动格式化代码")

    return 0 if (ruff_code == 0 and ruff_format_code == 0 and mypy_code == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
