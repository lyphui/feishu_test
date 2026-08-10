# [已发布] — 2026-08-10（输出编码修复 + 成交质量口径澄清）

上一条提交后自查出的四个待办，处理了三个，第四个明确记为「不做」。
全部为修复与口径澄清，**所有已发布的实测数字逐个复核未变**。

## 修复：重定向输出会 `UnicodeEncodeError` 崩掉

- **新增 `backtest/lib/console.py`**，`use_utf8()` 把 stdout/stderr 强制成 UTF-8；
  已在 10 个入口脚本的 `main()` 首行调用（`exec_bench` / `fatfinger_bench` /
  `hk_oil_etf_signal` / `jcy_intraday_timing` / `jcy_macd_bull_batch` /
  `lu_macd_analysis` / `lu_macd_bull_analysis` / `macd_analysis` / `oil_track` /
  `param_sweep`）。

- **为什么以前没发现**：报表里的 `▶ ★ ✗ ⚠ − ¥ ❌` GBK 编码不出来，但简体中文
  Windows 上 Python **只在输出不是控制台时**才退回 locale 编码——
  * 终端里直接跑 → 走 `WriteConsoleW`，`sys.stdout.encoding` 是 utf-8，一切正常；
  * `> log.txt`、`| more`、CI、被别的程序捕获 → stdout 是普通管道，改用 gbk，
    第一个 `▶` 就抛 `UnicodeEncodeError` 把脚本打断。

  也就是"本地跑得好好的，一存日志就挂"。14 个脚本都有这类字符，属于全仓库问题，
  不是新模块引入的。已验证 `hk_oil_etf_signal` / `fatfinger_bench` / `oil_track`
  重定向到文件均 exit 0，且不再需要 `PYTHONIOENCODING=utf-8`。

## 顺带修复：6 个入口脚本按文档跑根本起不来

为验证上面那条而做 `--help` 冒烟测试时发现的，与本轮改动无关的存量问题：

`python backtest/x.py` 只会把 `backtest/` 放进 `sys.path[0]`，**仓库根不在**——
Python 不会把 cwd 加进去。于是凡是要 `from strategies import` / `from jcy.lib...`
的脚本一律 `ModuleNotFoundError`：`jcy_intraday_timing` / `jcy_macd_bull_batch` /
`lu_macd_analysis` / `lu_macd_bull_analysis` / `macd_analysis` / `param_sweep`
六个全中。`exec_bench.py` 早就自己补了这段，只是没推广出去。

CLAUDE.md 里「仓库根已在 `sys.path`」这句话是错的——它只在 `pytest`（`conftest.py`
补的）和从仓库根 `import` 包时成立，对 `python backtest/x.py` 不成立。

已把 `exec_bench.py` 那段 4 行 bootstrap 复制到六个脚本，并改正 CLAUDE.md 的表述。
（这段**必须**内联在 import 之前，不能收进 `lib/` 的某个函数——它要先跑完，
后面的 `from strategies import` 才有得可导。）

冒烟测试：10 个入口脚本 `--help` 全部 exit 0。

## 修复：`fill_edge` 的回补价不再靠猜

`simulate_fatfinger` 原先事后按「成交次日开盘价」推算回补价。但再平衡真赶上停牌
会顺延，猜出来的价对应一个没发生的交易。

改法：把 `pending_rebalance: bool` 换成直接持有那条成交记录的引用 `pending_fill`，
**在真正执行再平衡的那一天就地写回成交价**。样本末尾那笔还欠着回补的成交现在
留 NaN 并被 `fill_edge` 的 `dropna` 排除，而不是编一个价出来；`n` 相应改为只计
回补价已知的成交。

复核：601857 / 600938 六档 k 的敞口对齐超额与 t 统计量**逐个未变**
（−108.8 / −82.8 / −37.1 / −7.2 / +7.9 / −7.8pp 与 −135.9 / −98.2 / −29.8 /
−19.5 / −16.5 / −11.7pp；唯二接近显著的仍是 t=−1.97 / −1.99 买单侧，
正向最大 t=1.71）。停牌本来就少，改的是口径的诚实度不是结论。

## 澄清：`edge` 是毛价差，门槛不是 0

`edge` 的回补价取不含滑点的裸价，佣金印花税也不在里面——这是**故意的**，摩擦
成本已经完整体现在净值曲线里，再扣一遍就是双重计费。代价是它天然偏乐观。

新增 `ROUND_TRIP_BP`（≈26bp，由 `lib.ladder` 的成本常量算出）并在模块 docstring、
`fill_edge()` docstring 与 `fatfinger_bench` 输出里点明：**判断"捡没捡到"要拿
edge 跟这条线比，不是跟 0 比**。原先只在 docs 的结论文字里提过一句，容易误读。

## 明确不做：`simulate_grid` 的锚价通用化

`ratchet=True` 目前只在 `level==0` 时上移，因为卖出触发价是从
`anchor − level×step` 推的，持仓期间抬锚会和当初买入的那一格对不上。

通用解法已写进 docstring 备查：把 `stack` 从「只存股数」改成「存 (股数, 该格买入
价)」，卖出按各格自己的买入价 ×(1+step) 触发，买卖对称之后锚怎么动都不影响已持有
的格子。**不实施**——实测网格在这两只票上没有稳定的敞口对齐超额（54 组中位数
≈ −3pp，见 [限价单与网格](2026-08-10-limit-order-and-grid-bench.md)），
为一个不赚钱的策略重构撮合逻辑不划算。真要用网格再回来改。

## 行为变更

无。`use_utf8()` 只改输出编码；`fill_edge` 的回补价口径修正在两只票全样本上
未改变任何已发布数字；`ROUND_TRIP_BP` 是新增的只读常量。测试 **283 passed**。
