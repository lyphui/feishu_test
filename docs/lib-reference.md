# 包内工具层参考表（`jcy/lib/` 与 `backtest/lib/`）

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。快速查表用；各模块的设计动机分散在对应的功能文档里
> （[[jcy-pipeline]]、[[tracking-and-ladder]]、[[execution-bench]]）。

工具按归属沉入各自包的 `lib/` 子目录，不再有根级 `utils/` 伪共享包。每个模块只有单一归属者，import 语义清晰。

## `jcy/lib/`

被流水线步骤调用，`from jcy.lib.x import`；跨包时 backtest 也 `from jcy.lib.common import`

| 模块 | 核心内容 |
|------|----------|
| `common.py` | `title_to_date()`、`title_to_filename()`、`record_key()`、`safe_title()`、`load_docs()`、`load_candidates()`、`is_ashare_code()`、路径常量（`JSON_PATH/DOCS_FILE/ADVICE_DIR`） |
| `text.py` | `blocks_to_text()`（飞书 block → 文本）、`parse_json_loose()`（宽松 JSON 解析） |
| `pplx.py` | `PerplexityAPI` 客户端（`chat()`、`sonar_deep_research()`） |

## `backtest/lib/`

被引擎/入口脚本复用，脚本模式下 `from lib.x import`

| 模块 | 核心内容 |
|------|----------|
| `market_data.py` | `fetch_stock_data()` + `fetch_index_data()` — 个股/指数行情，akshare → baostock → yfinance 三源（含限流重试）；`ADJUST` 统一三源复权口径，回测默认 **hfq 后复权**（qfq 会随分红回溯改写历史价，结果不可复现） |
| `price_store.py` | `load_daily()` / `update_daily()` / `load_dividends()` — 本地行情仓库（`data/market/`）。首次全量、之后只补增量；只有 hfq 能安全追加（qfq 强制整表重建），每次增量重叠 `OVERLAP_DAYS` 天对账收盘价，不一致即重建；`*.meta.json` 记**请求过的区间**而非数据首尾日 |
| `swings.py` | `zigzag()` / `swing_table()` / `drawdown_episodes()` / `drawdown_profile()` — 波段分解与独立回撤事件（按"创新高→再创新高"切分，不逐日比对阈值） |
| `regime.py` | `classify()` / `regime_stats()` / `regime_episodes()` — 市场状态分类（趋势上行/宽幅震荡/趋势下行），严格只用当日及以前数据，带 `confirm_days` 滞回 |
| `ladder.py` | `simulate_buy_hold/dca/ladder/grid/adaptive()` + `PLAYBOOK` — 分批建仓模拟器，成本与 T+1 口径对齐 `engine.py`，闲置现金计息，额外报 `avg_exposure` / `deployed_return` |
| `plotting.py` | `COLORS` 字典（GitHub Dark 配色）、`setup_matplotlib()`、`style_ax(ax)` |
| `bull_backtest.py` | `BullStrategyAdapter`（牛市策略通用适配器；绘图/CSV 在 `backtest/bull_report.py`） |
| `oil_price.py` | `fetch_oil_price()` / `update_oil()` / `load_oil()`（Brent/WTI/SC，新浪源，整表覆盖）、`transmission_table()`（油价→股价领先滞后相关系数，纯描述性） |
| `execution.py` | `intraday_macd()` / `daily_panel(side=)` / `add_limit_plan(side=)` / `benchmark(side=)` / `wait_value()` / `split_by_go()` — 日内下单方案的成交价测算，基准为当日 VWAP，单位 bp，**买卖双向**（原始 bp 中性，`优势bp` 才分侧）。**与标的、与策略无关**：两条工作流共用同一套度量，各自跑各自的数 |
| `intraday_store.py` | `load_intraday()` / `update_intraday()` / `fetch_intraday_raw()` — 本地分时仓库（`data/market/intraday/`）。存 **不复权**（`adjustflag="3"`）并带 `amount`，因为 VWAP 由 `amount/volume` 算出、必须与价格同口径；不复权价不回溯改写，追加比 hfq 还安全。只有 baostock 一条源（akshare 分钟接口打 eastmoney，本机被阻断）。缓存不入库，可重建 |
