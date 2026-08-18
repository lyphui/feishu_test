# backtest/ 系统评审与优化路线图

> 从 [CLAUDE.md](../CLAUDE.md) 的模块表进入。评审对象：
> [`backtest/`](../backtest/) 全部（engine / strategies / lib / scripts / reports）。
> 本文只做评审与路线图，不改代码；所有结论标注 `文件:行号`（2026-08-14 核对）。

`backtest/` 已经是一套方法论纪律很硬的**研究工具**：hfq 单一口径、T+1 时序铁律、
`eval_start` 预热窗口、`costs.py` 作为成本与成交约束的唯一真值源、把证伪结论
（MA5/8、展期收益择时、网格）一并留档。这些在个人量化项目里罕见。

但它离**能运营一个真实账户的系统**还差几层：所有回测都是「单票满仓 10 万独立跑」，
`batch_report` 自己就标注等权曲线不是可投资组合；收益口径含税前股息且假设无成本再投；
跑过的策略×参数×品类组合已上百，却没有多重检验校正；最大的两个批量任务绕过本地
行情仓库直连网络，重跑一次结论就可能对不上。

四个方向（结论可信度 / 成本与税收口径 / 工程整固 / 组合与账户层）全部纳入，
按「依赖关系 + 单位工作量收益」排序，而不是按重要性排序——组合层最重要，
但它必须站在前三项之上。

---

## 1. 现状基线（先立住做对的部分，后面每条改动都不能破坏它）

- **成本与成交约束单一真值源**。`backtest/lib/costs.py` docstring 开篇即声明
  「唯一真值源」（`costs.py:2`）：`COMMISSION_RATE`（:35）、`MIN_COMMISSION`（:37）、
  `STAMP_DUTY`（:39）、`SLIPPAGE`（:41）、`commission()`（:52）、`tradability()`（:59）、
  `infer_limit_pct()`（:99）。engine 与 ladder 都从这儿取（`engine.py:23-26`、
  `ladder.py:30-32`），且有机器守护：`tests/test_costs.py:18`（engine 默认值来自共享源）、
  `:27`（ladder 与 engine 同常量）、`:85`（tradability 全库只有一个实现）。
- **行情可注入**。`engine.run_backtest` 签名带 `df=` 参数（`engine.py:64`，
  消费点 :103-104），测试与复用场景不必联网；`compare_playbooks.py:231` 已在用。
- **周期重采样不打错标签**。`BaseStrategy._resample_period`
  （`backtest/strategies/base.py:39`）防住了 pandas "MS" 规则「月初标签用月末数据」
  这一类未来函数（docstring :27-36 自陈）。
- **月频信号滞后一根**。`trend_stop` 用 `held = position.shift(1)`
  （`backtest/lib/trend_stop.py:169`）；:163-168 记录了不滞后的代价——3175 实测
  年化 9.5% 虚增 vs 实际 4.9%，「彻头彻尾的未来函数」。
- **挂单老化与作废**。engine 的 `pending / pending_age / abandoned`
  （`engine.py:129-131`）：同向信号不重置年龄（:187-194），
  超 `max_pending_days=3`（:62）未成交即作废（:240-245）。
- **市场状态分类无未来函数**。`regime.classify`（`backtest/lib/regime.py:30`）
  全部 rolling/expanding（:51-54），新状态连续 5 天才切换的滞回（:61-69）。
- **选股 alpha 与择时 alpha 分列**。`backtest/reports/batch_report.py:34-45`
  （`选股alpha% = 基准收益% − 指数收益%`、`超额收益% = 策略收益% − 基准收益%`）；
  等权曲线自标「非可投资组合」（图例 :320、图注 :336、docstring :22）。
- **参数扫描的样本外纪律**。`sweep_params` 的 `--oos-frac` 时序留出
  （`sweep_params.py:359`，切分 :292-312）、「样本外只有一次」提醒（:515-516）、
  `--codes` 池共用起点时退化为纯样本内的显式提示（:387-394）。

---

## 2. P0 — 取数统一与可复现（低成本，且是其余三项的前提）

### 2.1 取数路径分裂

`price_store.load_daily`（`backtest/lib/price_store.py:225`：本地仓库 +
`OVERLAP_DAYS=10` 重叠对账 :51/:204-211 + `auto_update=False` 纯离线 :232）
与 `market_data.fetch_*`（`backtest/lib/market_data.py`：`fetch_stock_data` :185、
`fetch_etf_data` :257、`fetch_hk_data` :316、`fetch_index_data` :350，直连网络）
两条路并存：

- **走仓库**：`track_oil.py:107`、`compare_playbooks.py:393`、`compare_ma_cross.py:215`、
  `track_hk_oil_etf.py:155`、`backtest_fatfinger.py:150`
- **直连网络**：`engine.py:103-104`（run_backtest 默认取数）、
  `backtest_jcy_pool.py:327`（个股经引擎默认路径同直连）、
  `sweep_params.py:132` 与 :416、`backtest_jcy_intraday.py:262` 与 :269、
  `backtest_lu_macd_bull.py:102`

偏偏**规模最大的两个任务在直连那一侧**：jcy 池（`load_candidates(ratings=LONG_RATINGS)`
去重后 **248 只标的**，见 `jcy/lib/common.py:80`；去重规则「保留首次落入所选评级的最早记录」
在 :83）每次重下，sweep 是 8 轴网格 × N 只。后果有三：慢到 P2 的重复实验做不起来、没有 `--offline`、
以及「同一条命令两周后重跑得到不同数字」——这与仓库坚持 hfq 的理由是同一件事。

**建议**：这些入口统一改走 `load_daily`，`cli.base_parser()` 的 `--offline`
（`backtest/lib/cli.py:18`，目前只有 4 个脚本用 parents 接入，另 2 个手写）
全线生效；`engine.run_backtest` 的默认取数保持 `fetch_stock_data`
（库不该假设有本地仓库），由调用方注入 `df=`（现成先例：`compare_playbooks.py:231`）。

### 2.2 无 run manifest

`output/` 下只有结果，没有「谁跑的、用什么参数、数据截止到哪天」。
建议每个输出目录写 `run.json`：git sha + 是否 dirty、完整 CLI argv、
各标的 `price_store` meta 的 `data_end`/`rows`、`costs` 常量快照、耗时。
这是几十行的事，但它让所有历史结论从「记得当时是这么跑的」变成可复算。

### 2.3 假设常量的第二个漂移点

`costs.py` 收编了费率，但无风险利率与现金利率仍散在字面量里：

- `engine.py:389` `_calc_sharpe(..., rf=0.02)`
- `ladder.py:75` summarize 里硬编码 `- 0.02`
- `compare_playbooks.py:105` `_sharpe(ret, rf=0.02)`（docstring 自陈与 engine 公式一致）
- `ladder.py:86` `_run(..., cash_rate=0.015)`，另 `compare_playbooks.py:199` 显式传同值

同一张对比表里的夏普若哪天有一处被改，排序就变成按公式排序。这是目前
**唯一没有测试守护的常量漂移面**（费率有 `tests/test_costs.py`，它们没有）。
建议同样收进 `costs.py`（或 `assumptions.py`），并由 `tests/test_costs.py`
的同源检查一起守住。

---

## 3. P1 — 成本与税收口径（改的是数值本身，越早做后面越不用重算）

### 3.1 hfq 含税前股息：现场实算后的修正结论

hfq 口径的边界是：税前股息、无成本、无税、即时再投。A 股红利税随持有期分档
（≤1 个月 20% / 1 个月–1 年 10% / >1 年 0%）。直觉上「高换手策略 × 高股息标的
（601857 TTM 每股税前分红 0.47 元 = 2025-09-17 的 0.22 + 2026-06-26 的 0.25，
对 2026-08-07 盘面价 10.77 元即**股息率 4.4%**）= 系统性高估」，
但**用 `data/market/dividend/601857.csv` 实算一次后，这个直觉需要修正**：

- 以主力打法（LuMACDBull `shrink_exit=True`）全样本跑 601857（2018-01-02 →
  2026-08-07，本地 hfq 仓库 + 000300 指数过滤）：20 笔完整交易，平均持仓 3.5 天，
  在场比例 3.4%，总收益 12.45%。
- 逐笔匹配除息日：**没有一笔持仓跨越除息日**。20 笔持仓合计 107 个自然日
  （单笔 1~14 天），全区间 3140 天内除息 17 次，随机撒点的期望跨越次数
  = 17 × 107/3140 ≈ **0.58**——即「一次都没跨到」并不反常，是这个量级的正常结果。
  税后修正 **≈ 0**。
- 修正量级上限：以近期仓位（~6,800 股 ≈ 10 万元）计，单次除息每股 ~0.23–0.25 元，
  即便按最重的 20% 档，每跨越一次除息约扣 300–350 元 ≈ **0.3% 本金/次**。
- benchmark 腿（买入持有 8.6 年）适用 >1 年 0% 档，修正同样 ≈ 0。

即这条风险的真正落点不是高换手策略，而是**持仓 1 个月–1 年的打法**
（ladder 波段、trend_stop 月频调仓中段持仓），适用 10% 档；以及 hfq
「无成本即时再投」假设本身（量级更小）。

**建议**：先在 `docs/` 写清 hfq 的口径边界（税前、无成本即时再投，本节数字即依据），
再视中长持仓打法的实测持仓分布决定是否加「税后修正」列。
`price_store` 已存好分红数据（`price_store.py:46`、写入 :292-293），落地条件具备。
复算命令见第 7 节。

### 3.2 选股 alpha 的分母口径不对称（现场核实：成立）

`基准收益%` 来自个股 hfq（全收益，含股息再投），`指数收益%` 来自
`fetch_index_data("000300")`（`backtest_jcy_pool.py:327`，akshare 价格指数，
**不含股息**）。现场比对本地 `data/market/daily/000300_none.csv` 与同期沪深 300
全收益指数（H00300，中证指数官网源）：

- 区间 2018-01-02 → 2026-08-07（8.59 年，2086 个共同交易日）
- 价格指数：累计 +14.85%，年化 **+1.62%**
- 全收益指数：累计 +40.90%，年化 **+4.07%**
- **年化差 2.45%/年**——与沪深 300 股息率量级一致，证实取回的是价格指数

于是 `选股alpha% = 基准 − 指数` 被系统性抬高同一量级，而这一列正是用来回答
「研报推荐值不值钱」的。**建议**：改取全收益指数（H00300 已可经 akshare
`stock_zh_index_hist_csindex` 取到），或在表头明确标注并给出年化修正幅度。
这条影响 `batch_report` 与 `compare_rating_pools` 的全部结论。复算命令见第 7 节。

> ⚠ **实施陷阱：这里必须取两条指数，不能只换一处。**
> `backtest_jcy_pool.py:327` 取回的**同一个 `index_df` 对象**被喂给两个用途：
> `BullStrategyAdapter(inner_strategy, index_df)`（:160-161，牛市过滤器，
> **决定买卖信号**）和 `result_to_row(..., index_df=index_df)`（:264，**alpha 分母**）。
> 顺手把这一处换成全收益指数，会连带改写全部个股的 `bull_market` 历史
> ——一次口径修正就变成了一次策略变更，且两种效应在结果里无法拆开。
> 牛市过滤器按惯例应留在**价格指数**（「大盘在不在牛市」讲的是点位不是全收益），
> 只有 alpha 分母改用全收益。落地时应把两者拆成两个变量分别取数，
> 并用一次「只改分母、策略信号逐日 diff 为空」的回归确认没有串味。

### 3.3 港股成本模型没进 `costs.py`

`hk_trade_cost`（佣金 0.25% **最低 HK$100**、ETF 免印花税、平台费 HK$30）
活在 `backtest/lib/trend_stop.py:53-60`（费率常量 :45-50，ETF 豁免 :76），
与月频决策耦合。用户有真实港币账户（最低佣金决定了调仓频率上限），
港股迟早不止 3175.HK 一只。**建议**：把成本层改成 market-aware
（`costs.for_market("A"|"HK")` 返回一组费率 + 撮合约束 + `LOT`），
`trend_stop` 改为消费者而非定义者。

### 3.4 ST 股涨跌停无法从代码前缀推断

`infer_limit_pct` 的 docstring 已自陈（`costs.py:107-109`）：ST/*ST 为 5%，
但从代码看不出来，按主板 10% 处理 → 对 ST 偏乐观。建议在 P1 里只做一件小事：
本地维护一份 ST 区间映射或从 akshare 名称字段取，缺失时在汇总里报
「本池含 N 只无法判定的标的」。

---

## 4. P2 — 结论可信度

### 4.1 多重检验完全缺位

已跑过的组合：`sweep_params` 网格（8 个轴任选 2，`sweep_params.py:87-98`、:343）、
`compare_ma_cross`（6 品类桶 :74-112 × 8 变体 :147-156 × 30 标的）、
`compare_playbooks`（**15 种打法**，:204-241：7 个 ladder 系 + 月频趋势 6 + MA 交叉 2
——`CLAUDE.md` 原写「14 种」，本次一并订正）、
`trend_stop.sweep`（7 个 MA 长度 × 6 档止损 = 42 格，`trend_stop.py:251`，
网格定义 `track_hk_oil_etf.py:36-37`）。跑得越多，「最优格」是噪声的概率越高，
而现在唯一的防线是「整片偏绿 vs 孤立亮点」的目视判断。

**建议**：新增 `lib/stats.py`：deflated Sharpe（按实际尝试次数折减）、
以及在每个 sweep 结尾打印「本次共 N 组，纯随机下预期有 X 组达到该水平」。

### 4.2 描述性对比升级为检验

`compare_rating_pools`（`reports/batch_report.py:425`）的 docstring 自己写着
「这是描述性对比，不是显著性检验」（:436-438，运行时打印同款警告 :470-471）——
正确但可以补上：两池选股 alpha 的差值配置换检验 p 值（打乱评级标签重采样），
`print_summary_table` 的「跑赢比例」与「日均超额中位」配 block bootstrap
置信区间（块长取平均持仓天数，保留自相关）。一个共享实现（`lib/stats.py`），两处消费。

### 4.3 缺单标的时间轴上的 walk-forward

`sweep_params --oos-frac` 切的是**横截面**（较晚被推荐的票留作验证），
这在 JCY 池上是对的，但 `--codes` 池所有票共用起点时它会自动退化
（脚本已提示，`sweep_params.py:387-394`）。对油气、寒武纪这类长期跟踪标的，
需要的是滚动 walk-forward：滚动窗口内选参 → 下一段按该参数实盘化交易 →
拼成一条**真实可交易**的净值曲线。这条曲线才是「这套打法上不上实盘」的唯一诚实估计。

### 4.4 幸存者偏差已警告但未量化

`run_pool` 对失败标的只打印一句「剩下的样本带幸存者偏差」
（`backtest_jcy_pool.py:269-271`，失败行 :261-263 直接 continue）。
建议落 `summary_failures.csv`（代码/名称/推荐日/失败原因）并在汇总头部报失败率
——退市与长期停牌恰恰是最坏的那批。

---

## 5. P3 — 组合与账户层（最大的缺口，但依赖前三项）

### 5.1 `lib/portfolio.py`：固定资金池的多标的模拟

现状是 N 只票各自满仓 10 万，`batch_report` 的等权曲线明确声明不可投资
（`reports/batch_report.py:320`、:336）。需要的是：一个资金池、信号到来时按规则分配
（等权/波动率倒数/按信号强度）、单票上限、相关性或行业集中度约束、现金计息、
以及**信号多于资金时的排队规则**。分层建议：不合并 `engine`（二元单票）与
`ladder`（连续单票）两套骨架——两者的 docstring 已论证过不可互相塞入——而是
在其上加一层组合调度器，成交与成本一律复用 `costs.tradability` / `costs.commission`。
`track_oil.py:417-432` 那段「两票相关系数 + 分散化效果」是这一层的雏形，可以升格。

### 5.2 `lib/book.py` + `data/book/`：真实持仓账本

目前系统不知道用户实际持有什么。需要：成交流水录入 → 持仓/成本/已实现未实现 →
与回测口径对账（同一套 `costs`）。有了它，`track_oil` 的「建仓阶梯」才能从
「假设你从零开始」变成「按你当前 3 档仓位，下一档挂在哪」。

### 5.3 `scripts/daily_brief.py`：跨市场的今日行动单

把 `track_oil`（A 股油气）、`track_hk_oil_etf`（港股月频）、JCY 池信号汇成一张单：
标的、动作、挂单价（换算成盘面不复权价）、手数（按账本现金与 `LOT`/港股每手股数）、
触发条件、以及「今天什么都不用做」的显式结论。现有各脚本已经各自能出
「今天该做什么」，缺的是统一入口与账本联动。

### 5.4 `engine` / `ladder` 双撮合骨架的残余漂移面

`tradability` 与费率已统一（见第 1 节），但整手回退逻辑
（`engine.py:201-208` vs `ladder.py:109-114`）、滑点施加点
（`engine.py:199` vs `ladder.py:108`/:124）、佣金扣减顺序
（`engine.py:205`/:212 vs `ladder.py:113`/:126）、T+1 判定
（`engine.py:117`/:248-249 vs `ladder.py:139-145`）仍是两份代码。
建议提取一个共享的成交原语（`costs.fill(...)` 返回成交股数/成交价/费用），
两套骨架各自调用——不合并骨架，只收敛「一笔成交怎么算」。

---

## 6. 路线图汇总表

| # | 项 | 优先级 | 改变已有回测数值？ | 落点文件 | 依赖 | 粗略工作量 |
|---|----|--------|--------------------|----------|------|-----------|
| 1 | 取数统一走 price_store + `--offline` 全线 | P0 | **待验证**（见表下注） | 5 个直连脚本 + `lib/cli.py` | 无 | 中 |
| 2 | run.json manifest | P0 | 否 | 各脚本输出处 + 公共 helper | 无 | 小 |
| 3 | rf / cash_rate 收编进 costs.py | P0 | 否（值不变，只收敛出处） | `costs.py` + 4 处消费点 + `tests/test_costs.py` | 无 | 小 |
| 4 | hfq 口径边界文档 + 中长持仓税后修正 | P1 | **是**（加税后修正列时） | `costs.py`/引擎/ladder + docs | P0（取数与 manifest 先行） | 中 |
| 5 | 全收益指数替换价格指数的选股 alpha 分母 | P1 | **是**（选股alpha% 全表下移 ~2.4%/年） | `batch_report.py` / `backtest_jcy_pool.py` / price_store | 项 1（指数先入仓库） | 小（但**须拆两条指数**，见 3.2 陷阱框） |
| 6 | 港股成本 market-aware 收编 | P1 | **是**（港股回测数值口径归一） | `costs.py` + `trend_stop.py` | 项 3（同层改造） | 中 |
| 7 | ST 涨跌停映射 | P1 | 是（仅涉 ST 标的，且偏保守方向） | `costs.py` + 汇总侧 | 无 | 小 |
| 8 | `lib/stats.py` deflated Sharpe / 多重检验计数 | P2 | 否（只加打印与列） | `lib/stats.py` + 各 sweep 结尾 | 项 1（重跑要快） | 中 |
| 9 | 置换检验 + block bootstrap | P2 | 否（只加 p 值/区间） | `lib/stats.py` + `batch_report.py` | 项 8 | 中 |
| 10 | 单标的滚动 walk-forward | P2 | 否（新曲线，不改旧表） | `sweep_params.py` 或新脚本 | 项 1 | 大 |
| 11 | summary_failures.csv + 失败率 | P2 | 否（只加落盘与报头） | `backtest_jcy_pool.py` + `batch_report.py` | 无 | 小 |
| 12 | `lib/portfolio.py` 组合调度器 | P3 | 否（新层） | `lib/portfolio.py` | P0–P2 全部 | 大 |
| 13 | `lib/book.py` 真实持仓账本 | P3 | 否（新层） | `lib/book.py` + `data/book/` | 项 6（成本层归一） | 中 |
| 14 | `scripts/daily_brief.py` 今日行动单 | P3 | 否（新入口） | `scripts/daily_brief.py` | 项 12、13 | 中 |
| 15 | `costs.fill(...)` 共享成交原语 | P3 | 不应变（需用测试钉死现值） | `costs.py` + engine + ladder | 项 3 | 中 |

项 4、5、6 落地时会**改变已有输出数值**，届时需按 CLAUDE.md 约定写 changelog
并在索引标「含行为变更」，条目里单列「行为变更」一节。项 15 属重构，
合并前后应先用 `tests/` 钉死成交数值再动。

**项 1 为什么标「待验证」而不是「否」。** 直觉上换取数路径不该改数，但
`price_store` 里的数据未必与 `fetch_stock_data` 此刻返回的一致：`market_data.py:228-242`
的 baostock 回退源与 akshare 的 hfq 基准不同，仓库里某只票可能是回退源写入的；
`_overlap_matches`（`price_store.py:204-211`）也可能在某次更新时触发过整表重建。
这恰恰是「会静默改数」的典型改动。落地时应先跑一次 A/B：同一批标的、同一组参数，
新旧路径各出一份 `summary.csv` 做逐列 diff，**确认为空之后**才允许把这一项标成
「不改变数值」；若非空，需查明是哪只票、哪个源，并在 changelog 里说明。

---

## 7. 两处现场核实的复算命令

两段都在**仓库根**用 Git Bash 执行（heredoc 语法 PowerShell 不认）；
必须走 `.venv/Scripts/python.exe`——系统 PATH 上的 python 版本过低，
`jcy/lib/common.py:21` 的 `str | None` 标注会直接 `TypeError`。

### 7.1 项 5：000300 是价格指数（年化差 2.45%）

```bash
.venv/Scripts/python.exe - <<'EOF'
import pandas as pd, akshare as ak
tr = ak.stock_zh_index_hist_csindex(symbol="H00300", start_date="20180101", end_date="20260807")
tr.columns = ['date','code','name','name2','en','en2','open','high','low','close','chg','pct','vol','amt','n','pe']
tr['date'] = pd.to_datetime(tr['date'])
px = pd.read_csv('data/market/daily/000300_none.csv', parse_dates=['date'])
m = px[['date','close']].merge(tr[['date','close']], on='date', suffixes=('_px','_tr'))
yrs = (m.date.max()-m.date.min()).days/365.25
print(f"price ann: {(m.close_px.iloc[-1]/m.close_px.iloc[0])**(1/yrs)-1:+.2%}")
print(f"total return ann: {(m.close_tr.iloc[-1]/m.close_tr.iloc[0])**(1/yrs)-1:+.2%}")
EOF
# 实测输出：price +1.62% / total return +4.07% → 年化差 2.45%（2026-08-07 截止）
```

### 7.2 项 4：601857 股息税三档实算（高换手策略修正 ≈ 0）

```bash
.venv/Scripts/python.exe - <<'EOF'
import pandas as pd
from backtest.lib.price_store import load_daily
from backtest.engine import run_backtest
from backtest.strategies import LuMACDBullStrategy
from backtest.strategies.bull_backtest import BullStrategyAdapter

df  = load_daily("601857", "20180101", auto_update=False, verbose=False)
idx = load_daily("000300", "20180101", auto_update=False, adjust="none", kind="index", verbose=False)
r = run_backtest("601857", "20180101", "20260807",
                 strategy=BullStrategyAdapter(LuMACDBullStrategy(shrink_exit=True), idx), df=df)
tr = r["trades"].copy(); tr["date"] = pd.to_datetime(tr["date"])
div = pd.read_csv("data/market/dividend/601857.csv", parse_dates=["ex_date"])
buys, sells = tr.iloc[::2].reset_index(drop=True), tr.iloc[1::2].reset_index(drop=True)
tax = 0.0
for i in range(min(len(buys), len(sells))):
    b, s = buys.iloc[i], sells.iloc[i]
    days = (s.date - b.date).days
    rate = 0.20 if days <= 30 else (0.10 if days <= 365 else 0.0)
    for _, row in div[(div.ex_date > b.date) & (div.ex_date <= s.date)].iterrows():
        tax += row.cash_before_tax * b.shares * rate
print(f"trades={r['total_trades']} avg_hold={r['avg_holding_days']}d tax={tax:.0f}")
EOF
# 实测输出：trades=20 avg_hold=3.55d exposure=3.40% total_return=12.45% tax=0
#           20 笔持仓合计 107 个自然日，跨越除息 0 次
```

注意这段**没有传 `stop_loss`**，而 `backtest_jcy_pool` 的 CLI 默认是 `0.10`
（`backtest_jcy_pool.py:213`）。加上止损会改变笔数与持仓分布，但只会让持仓更短、
跨越除息的概率更低，结论方向不变。要严格对齐生产口径，补 `stop_loss=0.10` 重跑。

---

## 8. 本评审自身的局限

- **只有 2 条结论经过实测**（3.1 的股息税、3.2 的指数口径，复算命令见第 7 节）。
  其余 13 项路线图条目是**设计判断**，依据是代码阅读与既有 docs/changelog，
  没有跑数支撑。按本仓库一贯标准，它们的证据等级低于 `docs/ma-cross-5-8.md`
  那类实测留档，采纳前应各自补一次小样本验证。
- **3.1 的税负结论不可外推**。它只覆盖「601857 × LuMACDBull × 全样本」这一个组合。
  600938 没测；`ladder`（波段持仓，常落在 1 个月–1 年的 10% 档）与 `trend_stop`
  （月频，单段持仓动辄数月）恰恰是最可能受影响的两类，**都没测**。
  第 3.1 节末尾指出的「真正落点是中长持仓打法」是推断，不是结论。
- **3.2 的 2.45%/年是单区间单指数的测量值**，取决于 2018-01-02 → 2026-08-07
  这一段样本；换区间会变，且只对沪深 300 成立。若把基准换成中证 500 或行业指数，
  需重测各自的股息率缺口。
- **行号引用会随代码变动失效**。全文行号按 2026-08-14 的工作区核对，
  未来重构（尤其是路线图项 1、15 这类跨文件改动）之后需重新核对，
  或改用函数名定位。
- **优先级排序反映的是「依赖关系 + 单位工作量收益」，不是重要性**。
  P3 的组合与账户层才是「能不能用来管真实账户」的决定项，
  它排在最后只因为它依赖前三层的口径与数据先立住。
