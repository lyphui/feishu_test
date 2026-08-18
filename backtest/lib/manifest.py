"""
run.json：一次批量运行的可复现清单（评审 `docs/backtest-review.md` 项 2）。

`output/` 下此前只有结果，没有「谁跑的、用什么参数、数据截止到哪天」——
同一条命令两周后重跑得到不同数字时，无从对账。每个输出目录落一份 run.json：

- git sha + 是否 dirty（代码版本）
- 完整 CLI argv（参数版本）
- `costs` 常量快照（成本口径版本）
- 各标的 `price_store` meta 的 data_end / rows（数据版本）
- 耗时

用法
----
    from backtest.lib.manifest import write_run_manifest

    t0 = time.time()                     # main() 开头
    ...
    write_run_manifest(output_dir, symbols=codes, started_at=t0)
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

from backtest.lib import costs
from backtest.lib.market_data import DEFAULT_ADJUST
from backtest.lib.price_store import read_meta

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: `dirty` 只看**代码**路径。`data/market/daily/` 是被 git 跟踪的，而每次
#: `auto_update` 都会重写那些 CSV（外加新标的落成未跟踪文件），于是全仓库
#: `git status --porcelain` 永远非空——`dirty` 恒为 true，就再也无法表达
#: 「这次是带着未提交的代码跑的」这唯一有用的信息。
_CODE_PATHS = ["backtest", "jcy", "tests", "prepare_jcy_data.py", "pytest.ini"]


def _git_info() -> dict:
    """git 版本信息；不在仓库内或 git 不可用时降级为 None，不阻断输出。"""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_BASE_DIR,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--"] + _CODE_PATHS, cwd=_BASE_DIR,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        sha, dirty = None, None
    return {"sha": sha, "dirty": dirty, "dirty_scope": list(_CODE_PATHS)}


def costs_snapshot() -> dict:
    """成本与利率假设快照——结果数字的解释依赖这组值。"""
    return {
        "commission_rate": costs.COMMISSION_RATE,
        "min_commission":  costs.MIN_COMMISSION,
        "stamp_duty":      costs.STAMP_DUTY,
        "slippage":        costs.SLIPPAGE,
        "risk_free_rate":  costs.RISK_FREE_RATE,
        "cash_rate":       costs.CASH_RATE,
    }


def write_run_manifest(output_dir: str, *, argv: list | None = None,
                       symbols=(), started_at: float | None = None,
                       extra: dict | None = None,
                       filename: str = "run.json") -> str:
    """
    在 output_dir 下写 run.json，返回文件路径。

    symbols : 标的代码；元素可以是 "601857"（按默认复权口径查 meta）或
              (symbol, adjust) 二元组（如 ("000300", "none")）。
              meta 缺失的标的记 null，不报错——offline 首次跑也允许落清单。
    started_at : main() 开头的 time.time()，用于记耗时。
    filename : 清单文件名。同一个输出目录下有多种运行形态时（如
               `compare_ma_cross` 的 `--codes` 临时问答产出 `custom_*.csv`），
               清单要跟着结果文件一起改名，否则清单描述的运行与旁边的 CSV
               对不上号。
    """
    os.makedirs(output_dir, exist_ok=True)

    data_meta = {}
    for item in symbols:
        symbol, adjust = (item if isinstance(item, tuple)
                          else (item, DEFAULT_ADJUST))
        meta = read_meta(symbol, adjust)
        data_meta[f"{symbol}_{adjust}"] = (
            {"data_end": meta.get("data_end"), "rows": meta.get("rows")}
            if meta else None
        )

    manifest = {
        "run_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git":     _git_info(),
        "argv":    list(sys.argv if argv is None else argv),
        "costs":   costs_snapshot(),
        "data":    data_meta,
        "elapsed_sec": (round(time.time() - started_at, 1)
                        if started_at is not None else None),
    }
    if extra:
        manifest["extra"] = extra

    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
