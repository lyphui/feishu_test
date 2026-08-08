# 更新日志（索引）

本目录记录对回测引擎与报告层有行为影响的改动。格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

**一条改动一个文件**，命名 `YYYY-MM-DD-<英文短横线标题>.md`；同一天有多条时用
不同的标题后缀区分。本文件只做索引，不放正文——所有内容写进各自的条目文件。

新增一条时：在 `changelog/` 下建文件，并在下表**最上方**加一行。

| 日期 | 条目 | 主题 | 含行为变更 |
|------|------|------|:----------:|
| 2026-08-09 | [卖出端下单窗口](2026-08-09-sell-side-execution-window.md) | `execution.py` 买卖双向化、新增 `exec_bench.py` 与分时仓库；两池卖出侧实测（结论排序相反） | **是**（`death_cross` 改连续算，影响隔夜死叉日） |
| 2026-08-09 | [分时执行口径修正](2026-08-09-intraday-exec-price-caliber.md) | 可成交价改取下一根开盘价（去前视）；分时无 GO 不再跳过建仓 | **是** |
| 2026-08-09 | [日内下单测算](2026-08-09-intraday-execution-benchmark.md) | 新增 `lib/execution.py`（VWAP 基准的成交价测算）；`jcy_intraday_timing` GO 窗口实测无效，仅改建议文案 | 否（回测数值不变，仅打印文案） |
| 2026-08-08 | [油价→股价传导性分析](2026-08-08-oil-price-transmission.md) | 新增 `lib/oil_price.py`（新浪源 Brent/WTI/SC）与 `oil_track.py` 传导相关性报告 | 否（纯新增报告小节） |
| 2026-08-08 | [长期跟踪与分批建仓](2026-08-08-oil-tracking-ladder.md) | 本地行情仓库、波段/回撤剖面、市场状态识别、分批建仓模拟器与 `oil_track.py` | 否（全为新增模块） |
| 2026-08-08 | [回测口径修正](2026-08-08-backtest-cost-calibration.md) | 净额收益、挂单年龄语义、日均超额排序、样本外验证 `--oos-frac` | **是** |

> 标「含行为变更」的条目会改变已有回测的输出数值，**重跑历史结论前请先读**。
