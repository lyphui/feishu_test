# [已发布] — 2026-08-10（四处真耦合的解耦重构）

起因是一个架构问题："JCY 信息源 / A 股油气 / 港股 ETF / 回测引擎都堆在一起，
要不要重构文件夹让它们解耦？"

**先量了依赖图，答案是不要。** 四条业务线在 import 层面**已经互不引用**：JCY 用
`bull_backtest`，A 股油气用 `oil_price/regime/swings/ladder/fatfinger`，港股 ETF 用
`trend_stop`，下单测算用 `execution/intraday_store`；跨线共享的只有
`console/market_data/plotting/price_store` 四个名副其实的基础设施。今天加港股 ETF
那条线时没被迫动前三条——这是"结构能不能扩展"的唯一检验，它通过了。搬目录只是改
标签，代价却是 script 模式的 `sys.path` 要再改一轮、8 个 docs 与全部 changelog 的
路径引用集体失效（changelog 是历史记录，本不该被追溯修改）。

**但"耦合"的感觉是真的，位置不在目录上。** 本条修掉四处，全部**不改任何已发布数字**。

## ① 分时取数从两份实现收成一份

`jcy_intraday_timing` 自带一份 `_fetch_intraday_baostock`（52 行），与
`lib/intraday_store.fetch_intraday_raw` 是两份独立代码，**而且复权口径还不一样**
（前者 `adjustflag="2"` 前复权、后者 `"3"` 不复权），靠人看着两处维持一致。

两个口径本身**都对**，这是关键：

* 分时 **MACD** 要复权——MACD 走连续跨日序列，不复权在除权日有假跳空会带偏指标；
* **VWAP 基准**要不复权——它由 `amount/volume` 算出，那两个字段恒为原始值，
  混用是几百上千 bp 的系统性错位（`execution._check_same_basis()` 专门拦这个，
  `docs/execution-bench.md` 里那个 −1500bp 就是这么来的）。

所以正确形态是**同一个函数的两种参数**，不是两份实现：
`fetch_intraday_raw(..., adjust="none"|"qfq"|"hfq")`，`ADJUST_FLAGS` 做映射。
`jcy_intraday_timing.fetch_intraday` 保留 akshare 优先，baostock 那条路改为委派，
口径仍是 qfq——**信号完全不变**。

`load_intraday` 明确拒绝非 none：qfq 会随分红回溯改写历史，增量追加会把两种口径
缝在一起（与 `price_store` 同一个坑）；而且每次都得整表重建，缓存本身没有意义。

> 注意 `tests/test_intraday_exec_price.py` 那个"逐日比对"守的是**逻辑**对齐
> （同一份合成数据喂两条路径），它从来看不见数据源口径的差异。新增
> `tests/test_intraday_single_source.py` 8 项补这个缺口。

## ② A 股成本模型从两份字面量收成一处

`engine.py` 的函数默认值与 `lib/ladder.py` 的模块常量，同样的
`0.0003 / 5.0 / 0.001 / 0.001` 写了两遍，靠 ladder 的 docstring 写一句"与 engine.py
保持一致"人肉同步。这个项目全部价值建立在"同口径可比"上（`docs/tracking-and-ladder.md`
的敞口对齐口径直接依赖它），这里恰恰最脆。

新增 `lib/costs.py` 收口：常量 + `commission()` + `infer_limit_pct()`。
`engine.py` / `lib/ladder.py` / `config.py` 的 `.ini` 缺省值三处全部引用它；
`lib/fatfinger.py` 经 ladder 转引，自动在同一口径内。`_commission` 与
`infer_limit_pct` 保留旧名 re-export，历史导入不受影响。

`tests/test_costs.py` 13 项守住同源，含"引擎签名默认值必须等于 costs 的值"
（用 `inspect.signature` 查，改回字面量就会红）。

## ③ `param_sweep` 解绑 JCY 池

名字是通用参数扫描器，`main()` 却直接读 `jcy_insights.json`（`JSON_PATH` /
`LONG_RATINGS` / `load_candidates`），想给油气池做同样的过拟合检验就得改代码。
这是全仓库唯一一处**业务线泄漏进通用工具**。

新增 `resolve_universe(args)`：扫描机制与"票从哪来"之间的唯一接口，把任何来源归一成
`[{"code", "date"}]`（`date` = 该票的回测起点，JCY 池里就是推荐日）。下游
`evaluate_combo()` 本来也只用这两个键，所以契约天然成立，没有为解耦而加的抽象。

    python backtest/param_sweep.py                                 # 仍是 JCY 池，行为不变
    python backtest/param_sweep.py --codes 601857 600938 --codes-start 20180101

`--codes` 池所有票共用同一起点，按推荐日的时序留出无从谈起，`--oos-frac` 会自动
退化为纯样本内**并说明原因**（原来的提示只讲"票太少"，对这条路径是误导）。

## ④ `engine.py` 不再挂着绘图

`plot_backtest` + matplotlib 住在引擎里，导致**任何** `import engine` 都连带拖进
matplotlib——批量回测、参数扫描、pytest 这些不出图的场景全都要付这个代价，无 GUI
环境还得先操心 backend。项目里 `bull_report.py` / `batch_report.py` 早就是报告层
分离的，engine 只是没跟上。

拆出 `backtest/report.py`（125 行）。`setup_matplotlib()` 随之搬过去，与拆分前
`import engine` 的副作用等价，中文标签渲染不变。`macd_analysis.py` 继续 re-export
`plot_backtest`，旧导入路径不受影响。

`tests/test_engine_no_matplotlib.py` 在**子进程**里断言 `import engine` 后
`sys.modules` 不含 matplotlib（同进程内检查必然误判，其他测试早把它导进来了）。

## 行为变更

**无。** 逐项实测复核：

| 验证 | 结果 |
|---|---|
| `engine` 12 组回测（3 只票 × 2 策略 × 2 止损止盈），全部指标 + 成本明细 | 与重构前 **逐字一致**（`git stash` 前后对跑 JSON diff） |
| `fatfinger_bench --offline` 两只票六档 k 的敞口对齐超额 | **逐字一致** |
| `oil_track --offline` 完整输出 | **逐字一致** |
| `hk_oil_etf_signal --offline` | 年化 4.9% / 回撤 −52.1%，与文档一致 |
| 10 个入口脚本 `--help` | 全部 exit 0 |
| 全量测试 | **309 passed**（283 → 309，新增 26 项守护测试） |

## 文档

- `CLAUDE.md`：文件层次加 `report.py` / `lib/costs.py`；模块表更新 engine 定位；
  运行顺序加 `--codes` 示例。
- `docs/backtest-engine.md`：引擎"不含绘图"、成本来自 `lib/costs.py`，各注明守护测试。
- `docs/batch-and-sweep.md`：新增"股票池是可注入的"一节。
- `docs/lib-reference.md`：新增 `costs.py` 一行；`intraday_store.py` 补"唯一分时取数
  实现 + adjust 口径"。
