# 包内工具层参考表（`jcy/lib/` 与 `backtest/lib/`）

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。快速查表用；各模块的设计动机分散在对应的功能文档里
> （[[jcy-pipeline]]、[[tracking-and-ladder]]、[[execution-bench]]）。

工具按归属沉入各自包的 `lib/` 子目录，不再有根级 `utils/` 伪共享包。每个模块只有单一归属者，import 语义清晰。

backtest 侧的分层判据：**import matplotlib 的进 `backtest/reports/`，import 具体策略类的进
`backtest/strategies/`，其余计算与数据工具进 `backtest/lib/`**（规则全文见 CLAUDE.md）。

## `jcy/lib/`

被流水线步骤调用，`from jcy.lib.x import`；跨包时 backtest 也 `from jcy.lib.common import`

| 模块 | 核心内容 |
|------|----------|
| `common.py` | `title_to_date()`、`title_to_filename()`、`record_key()`、`safe_title()`、`load_docs()`、`load_candidates(json, ratings=)`、`is_ashare_code()`、评级常量 `LONG_RATINGS`（买入+增持）/ `CONTROL_RATINGS`（减持/卖出/回避）、路径常量（`JSON_PATH/DOCS_FILE/ADVICE_DIR`） |
| `text.py` | `blocks_to_text()`（飞书 block → 文本）、`parse_json_loose()`（宽松 JSON 解析） |
| `pplx.py` | `PerplexityAPI` 客户端（`chat()`、`sonar_deep_research()`） |

## `backtest/lib/`

被引擎/入口脚本复用，`from backtest.lib.x import`

| 模块 | 核心内容 |
|------|----------|
| `market_data.py` | `fetch_stock_data()` + `fetch_index_data()` + `fetch_hk_data()` + `fetch_etf_data()` — 个股/指数/场内基金行情，akshare → baostock → yfinance 三源（含限流重试）；场内基金（`fetch_etf_data`，`price_store` 里 `kind="etf"`）只能走 akshare `fund_etf_hist_em` 或 yfinance：个股接口不含 ETF，baostock 对 `sh.510300` 直接返回空表。`_to_yfinance_ticker(is_fund=True)` 按**基金号段**判沪深（沪 5 开头 / 深 1 开头），用个股规则会把 `510300` 错判成深市；`ADJUST` 统一三源复权口径，回测默认 **hfq 后复权**（qfq 会随分红回溯改写历史价，结果不可复现）。港股（`fetch_hk_data`）只有 yfinance 一条源，口径是**前复权**，故须以 `adjust="qfq"` 入库；yfinance 的 `end` 是开区间，内部已多要一天以免丢掉当天 K 线 |
| `price_store.py` | `load_daily()` / `update_daily()` / `load_dividends()` — 本地行情仓库（`data/market/`）。首次全量、之后只补增量；只有 hfq 能安全追加（qfq 强制整表重建），每次增量重叠 `OVERLAP_DAYS` 天对账收盘价，不一致即重建；`*.meta.json` 记**请求过的区间**而非数据首尾日。头尾增量骨架与 `intraday_store` 共用，抽在 `store_base.py` |
| `intraday_store.py` | `load_intraday()` / `update_intraday()` / `fetch_intraday_raw()` — 本地分时仓库（`data/market/intraday/`）。存 **不复权**（`adjustflag="3"`）并带 `amount`，因为 VWAP 由 `amount/volume` 算出、必须与价格同口径；不复权价不回溯改写，追加比 hfq 还安全。只有 baostock 一条源（akshare 分钟接口打 eastmoney，本机被阻断）。缓存不入库，可重建。**`fetch_intraday_raw` 是全仓库唯一的分时取数实现**，`adjust=none\|qfq\|hfq` 区分口径——`backtest_jcy_intraday` 以 `qfq` 复用它（分时 MACD 要复权，否则除权日的假跳空会带偏指标），但**非 none 不入库**（qfq 回溯改写历史，且 `amount` 仍是原始值不可用作 VWAP 基准）。增量骨架同 `price_store`，抽在 `store_base.py` |
| `store_base.py` | `incremental_update()` — `price_store`（日线）与 `intraday_store`（分时）共享的增量更新骨架：「头缺口 / 尾缺口 + `OVERLAP_DAYS` 重叠对账 + `PRICE_RTOL` 容差不符则整表重建」。怎么抓数、文件放哪、列集合、按索引还是按 dt 列合并、重叠几天，全部由调用方注入。`oil_price.py` 是同一模式的有意简化变体（每次整表覆盖、无头尾段逻辑），不复用本模块 |
| `swings.py` | `zigzag()` / `swing_table()` / `drawdown_episodes()` / `drawdown_profile()` — 波段分解与独立回撤事件（按"创新高→再创新高"切分，不逐日比对阈值） |
| `regime.py` | `classify()` / `regime_stats()` / `regime_episodes()` — 市场状态分类（趋势上行/宽幅震荡/趋势下行），严格只用当日及以前数据，带 `confirm_days` 滞回 |
| `ladder.py` | `simulate_buy_hold/dca/ladder/grid/adaptive()` + `summarize()` + `PLAYBOOK` — 分批建仓模拟器，成本常量与涨跌停/停牌成交判定（`costs.tradability`）均取自 `costs.py`，与 `engine.py` 严格同口径；闲置现金计息，额外报 `avg_exposure` / `deployed_return` |
| `trend_stop.py` | `simulate()` / `buy_hold()` / `sweep()` / `hk_trade_cost()` / `month_end_flags()` / `next_decision_date()` — 月频均线 + 移动止损（港股口径）。月末信号**次日**执行，止损锚在入场后最高收盘价、触发当日离场，止损后须等下个月末才重入。**净值用 `position.shift(1)`**——`pos[i]` 是按第 i 天收盘价成交才拿到的，不滞后就等于让止损躲掉触发当天那根阴线（踩过，年化虚增一倍）。内置港股成本模型（佣金 0.25% **最低 HK$100**、ETF 免印花税）——策略走月频正是被这个最低佣金逼出来的，两者耦合故同放一个模块 |
| `costs.py` | `COMMISSION_RATE` / `MIN_COMMISSION` / `STAMP_DUTY` / `SLIPPAGE` / `LOT` / `LIMIT_PCT_*` + `commission()` / `infer_limit_pct()` / `tradability()` —— **A 股成本与交易约束的唯一真值源**。`engine.py`（二元仓位，走函数参数）与 `ladder.py`（连续仓位，走模块常量）是两套撮合骨架，但费率必须逐位相同，否则「网格 vs 静态同敞口」这类横向比较比的是费率不是策略；`fatfinger.py` 也直取本模块常量。涨跌停/停牌成交判定曾有两份行为不同的实现（ladder 侧容差松 10 倍、漏判负值/NaN 成交量），已统一收进 `tradability()`（engine 的严格口径），两套骨架因此真正可比。`tests/test_costs.py` 守住同源 |
| `fatfinger.py` | `simulate_fatfinger()` / `simulate_static_mix()` — 乌龙指双侧远价限价单模拟，与同敞口静态持有基准（敞口对齐评价口径的基准曲线，见 [tracking-and-ladder.md](tracking-and-ladder.md)）。成本常量直取 `costs.py`，结果容器 `LadderResult` 与 `summarize()` 复用 `ladder.py` 的公开名 |
| `position_tracker.py` | `PositionTracker` / `TradeRecord` — JCY 分级买入 + 三级递进卖出的仓位状态机（统一 T+1、佣金印花税）。从 `backtest_jcy_intraday` 拆出的纯计算模块，测试直接导入，不必 `importorskip` 进 CLI 脚本 |
| `console.py` | `use_utf8()` — 把 stdout/stderr 强制成 UTF-8。**每个入口脚本 `main()` 第一行都要调**：报表里的 `▶ ★ ✗ ⚠ −` GBK 编码不出来，而 Windows 只在**输出不是控制台时**（`> log.txt`、`\| more`、CI）才退回 GBK，于是"终端跑得好好的，一存日志就 `UnicodeEncodeError`"。另收编三个 `compare_*` 脚本曾各写一份的表格打印：`fmt_table()` / `print_wide()` |
| `cli.py` | `base_parser()` — 入口脚本共享的 argparse parent parser（`--offline` / `--output` / `--start` / `--capital`），选项定义只有一处；各脚本用 `parser.set_defaults(...)` 覆盖自己的默认值 |
| `oil_price.py` | `fetch_oil_price()` / `update_oil()` / `load_oil()`（Brent/WTI/SC，新浪源，整表覆盖——`store_base` 模式的有意简化变体）、`transmission_table()`（油价→股价领先滞后相关系数，纯描述性）。常量 `OIL_SYMBOLS`（商品代码 WTI/BRENT/SC）与 `OIL_STOCKS`（A 股油气代码→名称，track_oil / compare_exec_plans / backtest_fatfinger 共用）**同名不同义**，已改名区分 |
| `execution.py` | `intraday_macd()` / `daily_panel(side=)` / `add_limit_plan(side=)` / `benchmark(side=)` / `wait_value()` / `split_by_go()` — 日内下单方案的成交价测算，基准为当日 VWAP，单位 bp，**买卖双向**（原始 bp 中性，`优势bp` 才分侧）。**与标的、与策略无关**：两条工作流共用同一套度量，各自跑各自的数 |

## `backtest/reports/`

一切出图与报告导出。**import matplotlib 的模块都进这里**；`engine.py` / `config.py` 不碰
matplotlib（`import backtest.engine` 不拖进它，`tests/test_engine_no_matplotlib.py` 守住，
`backtest/__init__.py` 因此必须留空、不做便利 re-export）。

| 模块 | 核心内容 |
|------|----------|
| `report.py` | `plot_backtest()` — 标准 4 面板图（价格+信号、指标、权益、回撤） |
| `batch_report.py` | 批量回测汇总：`build_summary()` / `print_summary_table()` / `build_portfolio_curve()` / `plot_portfolio()` / `write_batch_report()` / `compare_rating_pools()` |
| `bull_report.py` | `plot_bull_backtest()` / `export_bull_daily_status()` — 牛市策略图表与逐日状态 CSV |
| `intraday_report.py` | `plot_intraday_chart()` — `backtest_jcy_intraday` 的三面板分时图（价格+MACD+仓位） |
| `lu_macd_report.py` | `plot_lu_backtest()` / `export_daily_status()` — lu_macd 单票回测图与逐日状态 CSV |
| `plotting.py` | `setup_matplotlib()` / `style_ax(ax)`、GitHub Dark 配色 `COLORS` |

策略适配器 `BullStrategyAdapter` 在 `backtest/strategies/bull_backtest.py`——它 import 具体
策略类，按分层判据归 `strategies/` 而非 `lib/`（见 [strategies.md](strategies.md)）。
