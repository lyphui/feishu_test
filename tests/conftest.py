"""pytest 共享配置：把 backtest/ 放入 sys.path，使 `import config` / `import engine`
等回测同级模块可被测试导入（脚本运行时由 `python backtest/x.py` 自动放入 path）。"""

import os
import sys

_BACKTEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest")
if _BACKTEST_DIR not in sys.path:
    sys.path.insert(0, _BACKTEST_DIR)
