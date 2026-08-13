# 包内工具层参考表（`jcy/lib/` 与 `backtest/lib/`）

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。快速查表用；各模块的设计动机分散在对应的功能文档里
> （[[jcy-pipeline]]、[[tracking-and-ladder]]、[[execution-bench]]）。

工具按归属沉入各自包的 `lib/` 子目录，不再有根级 `utils/` 伪共享包。每个模块只有单一归属者，import 语义清晰。

## `jcy/lib/`

被流水线步骤调用，`from jcy.lib.x import`；跨包时 backtest 也 `from jcy.lib.common import`

| 模块 | 核心内容 |
|------|----------|
| `common.py` | `title_to_date()`、`title_to_filename()`、`record_key()`、`safe_title()`、`load_docs()`、`load_candidates(json, ratings=)`、`is_ashare_code()`、评级常量 `LONG_RATINGS`（买入+增持）/ `CONTROL_RATINGS`（减持/卖出/回避）、路径常量（`JSON_PATH/DOCS_FILE/ADVICE_DIR`） |
| `text.py` | `blocks_to_text()`（飞书 block → 文本）、`parse_json_loose()`（宽松 JSON 解析） |
| `pplx.py` | `PerplexityAPI` 客户端（`chat()`、`sonar_deep_research()`） |

## `backtest/lib/`

被引擎/入口脚本复用，脚本模式下 `from lib.x import`

| 模块 | 核心内容 |
|------|----------|
| `market_data.py` | `fetch_stock_data()` + `fetch_index_data()` + `fetch_hk_data()` + `fetch_etf_data()` — 个股/指数/场内基金行情，akshare → baostock → yfinance 三源（含限流重试）；场内基金（`fetch_etf_data`，`price_store` 里 `kind="etf"`）只能走 akshare `fund_etf_hist_em` 或 yfinance：个股接口不含 ETF，baostock 对 `sh.510300` 直接返回空表。`_to_yfinance_ticker(is_fund=True)` 按**基金号段**判沪深（沪 5 开头 / 深 1 开头），用个股规则会把 `510300` 错判成深市；`ADJUST` 统一三源复权口径，回测默认 **hfq 后复权**（qfq 会随分红回溯改写历史价，结果不可复现）。港股（`fetch_hk_data`）只有 yfinance 一条源，口径是**前复权**，故须以 `adjust="qfq"` 入库；yfinance 的 `end` 是开区间，内部已多要一天以免丢掉当天 K 线 |
| `price_store.py` | `load_daily()` / `update_daily()` / `load_dividends()` — 本地行情仓库（`data/market/`）。首次全量、之后只补增量；只有 hfq 能安全追加（qfq 强制整表重建），每次增量重叠 `OVERLAP_DAYS` 天对账收盘价，不一致即重建；`*.meta.json` 记**请求过的区间**而非数据首尾日 |
| `swings.py` | `zigzag()` / `swing_table()` / `drawdown_episodes()` / `drawdown_profile()` — 波段分解与独立回撤事件（按"创新高→再创新高"切分，不逐日比对阈值） |
| `regime.py` | `classify()` / `regime_stats()` / `regime_episodes()` — 市场状态分类（趋势上行/宽幅震荡/趋势下行），严格只用当日及以前数据，带 `confirm_days` 滞回 |
| `ladder.py` | `simulate_buy_hold/dca/ladder/grid/adaptive()` + `PLAYBOOK` — 分批建仓模拟器，成本与 T+1 口径对齐 `engine.py`，闲置现金计息，额外报 `avg_exposure` / `deployed_return` |
| `trend_stop.py` | `simulate()` / `buy_hold()` / `sweep()` / `hk_trade_cost()` / `month_end_flags()` / `next_decision_date()` — 月频均线 + 移动止损（港股口径）。月末信号**次日**执行，止损锚在入场后最高收盘价、触发当日离场，止损后须等下个月末才重入。**净值用 `position.shift(1)`**——`pos[i]` 是按第 i 天收盘价成交才拿到的，不滞后就等于让止损躲掉触发当天那根阴线（踩过，年化虚增一倍）。内置港股成本模型（佣金 0.25% **最低 HK$100**、ETF 免印花税）——策略走月频正是被这个最低佣金逼出来的，两者耦合故同放一个模块 |
| `plotting.py` | `COLORS` 字典（GitHub Dark 配色）、`setup_matplotlib()`、`style_ax(ax)` |
| `costs.py` | `COMMISSION_RATE` / `MIN_COMMISSION` / `STAMP_DUTY` / `SLIPPAGE` / `LOT` / `LIMIT_PCT_*` + `commission()` / `infer_limit_pct()` —— **A 股成本与交易约束的唯一真值源**。`engine.py`（二元仓位，走函数参数）与 `ladder.py`（连续仓位，走模块常量）是两套撮合骨架，但费率必须逐位相同，否则「网格 vs 静态同敞口」这类横向比较比的是费率不是策略。`fatfinger.py` 经 ladder 转引。`tests/test_costs.py` 守住同源 |
| `console.py` | `use_utf8()` — 把 stdout/stderr 强制成 UTF-8。**每个入口脚本 `main()` 第一行都要调**：报表里的 `▶ ★ ✗ ⚠ −` GBK 编码不出来，而 Windows 只在**输出不是控制台时**（`> log.txt`、`\| more`、CI）才退回 GBK，于是"终端跑得好好的，一存日志就 `UnicodeEncodeError`" |
| `bull_backtest.py` | `BullStrategyAdapter`（牛市策略通用适配器；绘图/CSV 在 `backtest/reports/bull_report.py`） |
| `oil_price.py` | `fetch_oil_price()` / `update_oil()` / `load_oil()`（Brent/WTI/SC，新浪源，整表覆盖）、`transmission_table()`（油价→股价领先滞后相关系数，纯描述性） |
| `execution.py` | `intraday_macd()` / `daily_panel(side=)` / `add_limit_plan(side=)` / `benchmark(side=)` / `wait_value()` / `split_by_go()` — 日内下单方案的成交价测算，基准为当日 VWAP，单位 bp，**买卖双向**（原始 bp 中性，`优势bp` 才分侧）。**与标的、与策略无关**：两条工作流共用同一套度量，各自跑各自的数 |
| `intraday_store.py` | `load_intraday()` / `update_intraday()` / `fetch_intraday_raw()` — 本地分时仓库（`data/market/intraday/`）。存 **不复权**（`adjustflag="3"`）并带 `amount`，因为 VWAP 由 `amount/volume` 算出、必须与价格同口径；不复权价不回溯改写，追加比 hfq 还安全。只有 baostock 一条源（akshare 分钟接口打 eastmoney，本机被阻断）。缓存不入库，可重建。**`fetch_intraday_raw` 是全仓库唯一的分时取数实现**，`adjust=none\|qfq\|hfq` 区分口径——`backtest_jcy_intraday` 以 `qfq` 复用它（分时 MACD 要复权，否则除权日的假跳空会带偏指标），但**非 none 不入库**（qfq 回溯改写历史，且 `amount` 仍是原始值不可用作 VWAP 基准）|
