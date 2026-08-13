"""
入口脚本共享的 argparse 选项（parent parser）。

`backtest/scripts/` 下 12 个 CLI 此前各写一份 `--offline` / `--output` /
`--start` / `--capital`（有的还没有），统一到这里之后：
- 选项定义只有一处，新增脚本直接 `parents=[cli.base_parser()]`；
- 每个脚本用 `parser.set_defaults(...)` 覆盖自己的默认输出目录等；
- `--offline` 对所有脚本一致：不联网、只读本地缓存（`lib/price_store` /
  `lib/intraday_store`），无本地缓存时按各脚本的取数路径报错或跳过。
"""

import argparse


def base_parser() -> argparse.ArgumentParser:
    """公共选项：offline / output / start / capital。默认值由各脚本 set_defaults。"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--offline", action="store_true", help="不联网，只读本地缓存")
    p.add_argument("--output", type=str, default=None, help="输出目录")
    p.add_argument("--start", type=str, default=None, help="统计窗口起点 YYYYMMDD")
    p.add_argument("--capital", type=float, default=None, help="本金")
    return p
