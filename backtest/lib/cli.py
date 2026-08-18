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


def base_parser(*, start: bool = True, capital: bool = True) -> argparse.ArgumentParser:
    """
    公共选项：offline / output（+ 可选 start / capital）。默认值由各脚本 set_defaults。

    start / capital 可以关掉，**用不上的脚本必须关掉**：`parents=[...]` 是
    无条件继承的，脚本不读 `args.start` 时这个选项照样出现在 `--help` 里、
    照样被接受——传 `--capital 500000` 却毫无效果，比没有这个选项更糟。
    典型例子：`backtest_jcy_pool` / `sweep_params` 的回测起点由每只票各自的
    推荐日决定，统一的 `--start` 无从谈起。
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--offline", action="store_true", help="不联网，只读本地缓存")
    p.add_argument("--output", type=str, default=None, help="输出目录")
    if start:
        p.add_argument("--start", type=str, default=None, help="统计窗口起点 YYYYMMDD")
    if capital:
        p.add_argument("--capital", type=float, default=None, help="本金")
    return p
