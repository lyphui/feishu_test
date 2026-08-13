"""
控制台输出编码修正与共享的表格打印辅助。

**每个 backtest 入口脚本的 main() 第一行都该调用 `use_utf8()`。**

为什么需要
----------
本项目的报表大量使用 `▶ ★ ✗ ⚠ − ¥ ❌` 这类符号（14 个脚本里都有），而简体中文
Windows 的默认 locale 编码是 **GBK**——这些字符 GBK 编码不出来，`print` 直接抛
`UnicodeEncodeError` 把脚本打断。

坑在于**它只在输出不是控制台时才犯**：

* 直接在终端跑 → Python 走 `WriteConsoleW`（UTF-16），`sys.stdout.encoding` 是
  utf-8，一切正常，所以平时看不出问题；
* `> log.txt`、`| more`、CI、被别的程序捕获 → stdout 退化成普通管道，改用 locale
  编码（gbk），第一个 `▶` 就崩。

也就是说「本地跑得好好的，一存日志就挂」。别靠 `PYTHONIOENCODING=utf-8` 环境变量
补救——那要求每个调用方都记得设，忘一次就断一次。

表格打印（`fmt_table` / `print_wide`）也是跨脚本重复的小工具：`compare_exec_plans`、
`compare_ma_cross`、`compare_playbooks` 曾各写一份 `_fmt` / `fmt` / `print_wide`，
收在这里，三处共用同一份实现。
"""

import sys


def use_utf8() -> None:
    """把 stdout / stderr 强制成 UTF-8。已经是 UTF-8 时是空操作。"""
    for stream in (sys.stdout, sys.stderr):
        # 被重定向成 StringIO 之类的对象时没有 reconfigure，跳过即可
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def fmt_table(df, floatfmt: str = "{:7.2f}") -> str:
    """DataFrame 排版：无索引、按列宽对齐，用于控制台表格。"""
    return df.to_string(index=False,
                        float_format=lambda v: floatfmt.format(v))


def print_wide(table, chunk: int = 5, floatfmt: str = "{:9.1f}") -> None:
    """列太多时按列分块打印，避免 pandas 折行把表拆得没法看。"""
    cols = list(table.columns)
    for i in range(0, len(cols), chunk):
        part = table[cols[i:i + chunk]]
        print(part.to_string(float_format=lambda v: floatfmt.format(v)))
        if i + chunk < len(cols):
            print()
