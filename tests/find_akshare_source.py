"""追踪哪个模块第一个触发 import akshare"""
import sys
import warnings

# 在 import 前先记录所有已有模块
before = set(sys.modules.keys())

from LogicAnalyzer.StockAnalysisCoordinator import StockAnalysisCoordinatorFactory

after = set(sys.modules.keys())
new_modules = after - before

akshare_or_miniracer = [m for m in new_modules if 'akshare' in m.lower() or 'mini_racer' in m.lower()]
if akshare_or_miniracer:
    print("=== Modules that loaded akshare / py_mini_racer ===")
    for m in sorted(akshare_or_miniracer):
        mod = sys.modules[m]
        f = getattr(mod, '__file__', 'unknown')
        print(f"  {m}: {f}")

# 找出 akshare 是否被间接加载
importers = []
for mod_name in sorted(after):
    mod = sys.modules[mod_name]
    if mod_name == 'akshare':
        importers.append(mod_name)
    # Check common akshare-dependant modules

# 反过来查：哪些模块 import 了 akshare
print("\n=== Module-level imports of akshare ===")
import ast, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for mod_name in sorted(after):
    mod = sys.modules[mod_name]
    f = getattr(mod, '__file__', None)
    if f and f.startswith(BASE) and f.endswith('.py'):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(
                    alias.name == 'akshare' or alias.name.startswith('akshare.')
                    for alias in node.names
                ):
                    rel = os.path.relpath(f, BASE)
                    print(f"  {rel}: import {', '.join(a.name for a in node.names)}")
        except Exception:
            pass
