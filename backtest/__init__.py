# 本文件必须保持为空（只允许注释）。
#
# tests/test_engine_no_matplotlib.py 断言 `import backtest.engine` 不会把
# matplotlib 拖进进程。包化后 `import backtest.engine` 会先执行本 `__init__`，
# 一旦在这里写一句便利 re-export（尤其 `from .report import ...`），引擎就
# 重新被绑上 matplotlib。要加 re-export 请先改掉那条测试。
