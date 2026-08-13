"""防复发验证：backtest/scripts/ 下每个脚本都必须能被真正启动。

必须用子进程而非 `import`——测试进程里仓库根本来就在 sys.path 上，
`import` 永远测不出「bootstrap 顺序错误导致 ModuleNotFoundError」这类
只有真正启动才暴露的问题。

策略：
- 有 argparse 的脚本用 `python -m backtest.scripts.X --help`（argparse 打印
  帮助后以 0 退出，且导入全部顶层模块，正好覆盖 import 顺序）；
- 无 argparse 的脚本（backtest_lu_macd / backtest_lu_macd_bull）用 runpy 以
  非 __main__ 名执行，只跑顶层导入、跳过 `if __name__ == "__main__"` 块，
  避免真的启动回测。
"""

import glob
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(_ROOT, "backtest", "scripts")


def _script_paths():
    return sorted(
        p for p in glob.glob(os.path.join(SCRIPTS_DIR, "*.py"))
        if os.path.basename(p) != "__init__.py"
    )


def _has_argparse(path):
    with open(path, encoding="utf-8") as fh:
        return "ArgumentParser" in fh.read()


@pytest.mark.parametrize(
    "script", _script_paths(), ids=lambda p: os.path.basename(p)
)
def test_script_starts(script):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    module = "backtest.scripts." + os.path.splitext(os.path.basename(script))[0]
    if _has_argparse(script):
        cmd = [sys.executable, "-m", module, "--help"]
    else:
        # run_name 非 "__main__"：只执行顶层导入，不触发 main()
        cmd = [sys.executable, "-c",
               f"import runpy; runpy.run_path({script!r}, run_name='_probe')"]
    proc = subprocess.run(
        cmd, cwd=_ROOT, env=env, capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, (
        f"{os.path.basename(script)} 启动失败 (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
    assert "ModuleNotFoundError" not in proc.stderr, (
        f"{os.path.basename(script)} stderr 出现 ModuleNotFoundError：\n"
        f"{proc.stderr[-4000:]}"
    )
