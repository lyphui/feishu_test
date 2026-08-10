"""
守住"引擎不含绘图"这条边界。

`plot_backtest` 曾经住在 `engine.py` 里，导致**任何** `import engine` 都连带拖进
matplotlib——批量回测、参数扫描、pytest 这些不出图的场景全都要付这个代价，无 GUI
环境还得先操心 backend。拆到 `report.py` 之后，这条测试防止它被重新塞回去。
"""

import pathlib
import subprocess
import sys
import textwrap

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)


def test_importing_engine_does_not_load_matplotlib():
    """在**子进程**里验证：import engine 之后 sys.modules 里不该出现 matplotlib。

    必须起子进程——同一个 pytest 会话里其他测试早就把 matplotlib 导进来了，
    在本进程内检查 `sys.modules` 永远会误判为失败。
    """
    code = textwrap.dedent("""
        import os, sys
        root = sys.argv[1]
        sys.path[:0] = [os.path.join(root, "backtest"), root]
        import engine
        assert "matplotlib" not in sys.modules, \\
            "engine 又把 matplotlib 拖进来了——绘图应该留在 report.py"
        assert callable(engine.run_backtest)
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", code, _ROOT],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_report_still_exposes_plot_backtest():
    """拆分不能弄丢公开接口：report 与 macd_analysis 两条路径都要能拿到。"""
    import report
    assert callable(report.plot_backtest)
    import macd_analysis
    assert macd_analysis.plot_backtest is report.plot_backtest
