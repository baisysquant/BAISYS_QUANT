"""
ScoringRules — 门控评分规则包

按功能拆分为：
  - base.py:  条件函数 + 动作函数（按 Gate 0-4 分组）
  - rules.py: Rule 数据类型 + RULES 列表 + 执行入口

向后兼容：所有公开符号通过 __init__.py 重新导出。
"""

from LogicAnalyzer.scoring.rules.base import *
from LogicAnalyzer.scoring.rules.rules import *
