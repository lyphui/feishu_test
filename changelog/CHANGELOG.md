# 更新日志（索引）

本目录记录对回测引擎与报告层有行为影响的改动。格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

**一条改动一个文件**，命名 `YYYY-MM-DD-<英文短横线标题>.md`；同一天有多条时用
不同的标题后缀区分。本文件只做索引，不放正文——所有内容写进各自的条目文件。

新增一条时：在 `changelog/` 下建文件，并在下表**最上方**加一行。

| 日期 | 条目 | 主题 | 含行为变更 |
|------|------|------|:----------:|
| 2026-08-13 | [单票打法对比台](2026-08-13-stock-playbook-bench.md) | 新增 `stock_playbook.py`：把 `engine`/`ladder`/`trend_stop` 三套模拟器的 14 种打法摆进同一张表（含科创板 ±20% 涨跌停与整手本金两处口径修正）；寒武纪实测留档——两个起点都显示择时打不赢「始终在场」、分批是唯一稳定改良、止盈止损在高波动票上是灾难 | 否（纯新增入口脚本） |
| 2026-08-11 | [MA5/8 金叉死叉证伪](2026-08-11-ma-cross-5-8-falsified.md) | 新增 `strategies/ma_cross.py`（含只过滤进场的量能条件）+ `ma_cross_bench.py` 分品类实测台（6 桶 × 8 变体）+ ETF 取数通道 `fetch_etf_data`/`kind="etf"`（并修 `_to_yfinance_ticker` 的基金号段误判）；实测结论：MA5/8 在 30 个标的上 0 个跑赢买入持有，零成本对照下仍全负，回撤反而更深 | 否（全为新增模块与新增取数分支） |
| 2026-08-10 | [四处真耦合的解耦重构](2026-08-10-decouple-shared-layers.md) | 依赖图证明四条业务线已解耦、**不搬目录**；改修四处真耦合：分时取数两份实现（口径还不同）收成一份 `fetch_intraday_raw(adjust=)`、A 股成本两份字面量收进 `lib/costs.py`、`param_sweep` 用 `resolve_universe()` 解绑 JCY 池、`plot_backtest` 拆到 `report.py` 让引擎不再拖 matplotlib | 否（engine/fatfinger/oil_track 输出逐字一致） |
| 2026-08-10 | [输出编码 + 成交质量口径](2026-08-10-console-utf8-and-fill-edge-cleanup.md) | 新增 `lib/console.py` 的 `use_utf8()`（10 个入口脚本调用），修掉「重定向输出就 `UnicodeEncodeError`」；补上 6 个入口脚本缺失的仓库根 `sys.path` bootstrap（按文档跑本来就 `ModuleNotFoundError`）；`fill_edge` 回补价改为执行当天就地记录、不再猜次日开盘；新增 `ROUND_TRIP_BP` 点明 edge 是毛价差 | 否（数字逐个复核未变） |
| 2026-08-10 | [港股原油 ETF 月频信号](2026-08-10-hk-oil-etf-trend-stop.md) | 新增 `lib/trend_stop.py` + `hk_oil_etf_signal.py`（月末均线 + 移动止损，含港股最低佣金成本模型）；`price_store` 接入港股行情（`kind="hk"`）；发布前修掉净值仓位差一天的未来函数（年化 9.5%→4.9%，「止损提升收益」结论作废）；留档两个否定结论：展期收益不可择时、港股油气股不是油价工具 | 否（全为新增模块） |
| 2026-08-10 | [乌龙指与网格双双证伪](2026-08-10-limit-order-and-grid-bench.md) | 新增 `lib/fatfinger.py` + `fatfinger_bench.py`（50/50 两侧挂远距离限价单等错价）与 `simulate_grid(ratchet=)`；确立**敞口对齐**评价口径，六档 k + 54 组网格参数实测均无稳定超额，有效的是按状态调仓位 | 否（新增模块 + 新参数默认关闭） |
| 2026-08-10 | [评级池与归因](2026-08-10-rating-pool-and-attribution.md) | 候选池纳入「买入」、`--control` 看空对照组、选股/择时两个 alpha 分开报、在场比例、扫描改用日均超额选参 | **是**（池子 239→248 只，summary 列变化，最优参数格可能不同） |
| 2026-08-10 | [零轴出场](2026-08-10-bull-zero-axis-exit.md) | `shrink_exit=True` 补上 `hist ≤ 0` 离场——柱子直接跌破 0 后旧实现没有任何出场信号 | **是**（信号层必变；两只缓存标的因在场仅 2.5% 成交未变） |
| 2026-08-10 | [指数预热解耦](2026-08-10-index-warmup-decoupled.md) | 牛市过滤器的指数取数起点改为绝对日期，不再随候选池变化 | **是**（`bull_market` 序列修正，历史结论需重跑） |
| 2026-08-09 | [PositionTracker 统一 T+1](2026-08-09-position-tracker-t1-execution.md) | 修掉三套执行口径并存（零延迟成交 + 分界线取决于 `--lookback`），全部排到下一个交易日 | **是**（平均 −1.97pp，动量票影响最大） |
| 2026-08-09 | [卖出端下单窗口](2026-08-09-sell-side-execution-window.md) | `execution.py` 买卖双向化、新增 `exec_bench.py` 与分时仓库；两池卖出侧实测（结论排序相反） | **是**（`death_cross` 改连续算，影响隔夜死叉日） |
| 2026-08-09 | [分时执行口径修正](2026-08-09-intraday-exec-price-caliber.md) | 可成交价改取下一根开盘价（去前视）；分时无 GO 不再跳过建仓 | **是** |
| 2026-08-09 | [日内下单测算](2026-08-09-intraday-execution-benchmark.md) | 新增 `lib/execution.py`（VWAP 基准的成交价测算）；`jcy_intraday_timing` GO 窗口实测无效，仅改建议文案 | 否（回测数值不变，仅打印文案） |
| 2026-08-08 | [油价→股价传导性分析](2026-08-08-oil-price-transmission.md) | 新增 `lib/oil_price.py`（新浪源 Brent/WTI/SC）与 `oil_track.py` 传导相关性报告 | 否（纯新增报告小节） |
| 2026-08-08 | [长期跟踪与分批建仓](2026-08-08-oil-tracking-ladder.md) | 本地行情仓库、波段/回撤剖面、市场状态识别、分批建仓模拟器与 `oil_track.py` | 否（全为新增模块） |
| 2026-08-08 | [回测口径修正](2026-08-08-backtest-cost-calibration.md) | 净额收益、挂单年龄语义、日均超额排序、样本外验证 `--oos-frac` | **是** |

> 标「含行为变更」的条目会改变已有回测的输出数值，**重跑历史结论前请先读**。
