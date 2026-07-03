"""检查 akshare 的加载链"""
import sys
from LogicAnalyzer.DataAcquisitionService import DataAcquisitionService

print("=== akshare import chain ===")
for mod_name in sorted(sys.modules):
    if 'akshare' in mod_name.lower() and not mod_name.startswith('LogicAnalyzer'):
        mod = sys.modules[mod_name]
        f = getattr(mod, '__file__', 'unknown')
        print(f'  {mod_name}: {f}')

print("\n=== py_mini_racer modules ===")
for mod_name in sorted(sys.modules):
    if 'mini_racer' in mod_name:
        print(f'  {mod_name}')
