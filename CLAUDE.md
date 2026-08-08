# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# feishu_test — 代码库架构文档

## 环境要求

- Python **≥ 3.11**
- **纯源码运行，无需安装本项目**：所有命令从仓库根目录执行，`jcy/`、`strategies/` 等包直接被 Python 解析（仓库根已在 `sys.path`），`backtest/` 脚本以 `python backtest/x.py` 运行（脚本目录自动入 path）。
- 仅需安装第三方依赖：
  ```bash
  pip install pandas numpy matplotlib requests openai python-dotenv pyyaml akshare yfinance pytest
  # 可选但推荐：行情第二数据源（akshare 失败时自动回退，且能给出正确的 hfq 后复权口径）
  pip install baostock
  ```
- `authorize/` 下的一次性授权脚本另需 `flask flask-sslify pyOpenSSL`（不在主流水线依赖内，仅手动获取/刷新飞书 token 时用到）。

## 项目概述

飞书股市分析 + MACD 量化回测一体化平台。

**完整流水线：**
1. **数据采集**：从飞书多维表格读取（JCY）股市分析文章
2. **AI 分析**：用 Perplexity sonar-reasoning-pro 生成投资建议（面向小白）
3. **结构化提取**：用 LLM（默认 DashScope，可回退 Custom/Azure/Coze）提取公司/代码/评级等结构化信息
4. **量化回测**：对推荐股票执行 MACD 策略回测，验证实际收益

---

## 文件层次

```
feishu_test/
├── CLAUDE.md                      # 本文件
├── .env                           # 环境变量（API 密钥等，不提交）
│
├── ── 入口脚本 ────────────────────────────────
├── prepare_jcy_data.py            # 薄入口：re-export jcy 包并提供 main（python prepare_jcy_data.py）
│
├── ── JCY 流水线包 ────────────────────────────
├── jcy/
│   ├── config.py                  # 常量 / 环境变量 / system prompt / logging
│   ├── store.py                   # 单一真值源读写、复合键索引、Step 跳过判断、advice 路径解析
│   ├── feishu.py                  # Step 1：飞书采集（_feishu_get 统一 GET + 分页 + 增量缓存）
│   ├── advice.py                  # Step 2：Perplexity 投资建议（先落文件后写 record 原子性）
│   ├── extract.py                 # Step 3：LLM 结构化提取（DashScope/Azure/Coze/Custom 回退）
│   ├── pipeline.py                # main 编排（--strict / --log-file）
│   ├── migrate_compound_key.py    # 运维：存量数据复合键迁移（零 API 调用）
│   ├── prompts/                   # jcy 自包含的 system prompt（被 config.read_prompt 读取）
│   │   ├── step2_advice_system.md # Step 2 Perplexity system prompt
│   │   └── step3_extract_system.md # Step 3 结构化提取 system prompt
│   └── lib/                       # jcy 内部工具（被流水线步骤调用，from jcy.lib.x import）
│       ├── common.py              # 日期解析 / 文件命名 / 复合键 / 候选股筛选 / 路径常量（原 jcy_common）
│       ├── text.py                # 飞书 block → 文本、宽松 JSON 解析（原 jcy_text）
│       └── pplx.py                # Perplexity API 客户端封装
│
├── ── 回测脚本 ─────────────────────────────────
├── backtest/                      # 脚本模式（非包），以 python backtest/x.py 运行，同级裸导入
│   ├── engine.py                  # 核心回测引擎：run_backtest / plot_backtest（无 CLI，含 A 股成交约束）
│   ├── config.py                  # 共享配置层：BacktestConfig / load_backtest_config / execution_kwargs / OutputPaths
│   ├── bull_report.py             # 牛市策略报告：plot_bull_backtest / export_bull_daily_status
│   ├── batch_report.py            # 批量回测横截面汇总：summary.csv / 等权组合净值 vs 指数
│   ├── param_sweep.py             # 参数敏感性网格扫描（判断是否过拟合）
│   ├── jcy_macd_bull_batch.py     # Step 4: 批量 MACD 牛市策略回测（读 jcy_insights.json）
│   ├── jcy_intraday_timing.py     # 日线信号 + 分时择时（多周期共振）
│   ├── macd_analysis.py           # 薄入口：re-export engine + MACDStrategy CLI
│   ├── lu_macd_analysis.py        # 单股卢式 MACD 三级底部策略回测
│   ├── lu_macd_bull_analysis.py   # 单股卢式 MACD 牛市动能截取策略回测
│   ├── presets/                   # 单股回测输入预设 .ini（symbol/日期区间/止损止盈等）
│   │   ├── jxty_jcy_260104.ini    # 单股 MACD 回测示例预设
│   │   ├── lu_macd_config.ini     # 卢式 MACD 策略回测预设
│   │   ├── lu_macd_bull_config.ini # 卢式牛市策略回测预设
│   │   └── rjgd_syr_260130.ini    # 其他回测预设示例
│   └── lib/                       # backtest 内部工具（被引擎/入口复用，from lib.x import）
│       ├── market_data.py         # 行情数据获取：个股 + 指数（akshare → baostock → yfinance 三源，统一后复权 hfq）
│       ├── plotting.py            # 绘图样式（GitHub Dark 配色 + matplotlib 配置）
│       └── bull_backtest.py       # 牛市策略通用适配器 BullStrategyAdapter
│
├── ── 策略包 ──────────────────────────────────
├── strategies/
│   ├── __init__.py                # 导出三个策略类
│   ├── base.py                    # BaseStrategy 抽象基类（含共享 _ema() 方法）
│   ├── macd.py                    # MACDStrategy（金叉/死叉，教科书版）
│   ├── lu_macd.py                 # LuMACDStrategy（三级底部确认，长线建仓）
│   └── lu_macd_bull.py            # LuMACDBullStrategy（牛市截陡坡，高频战术）
│
├── ── 数据目录 ─────────────────────────────────
├── data/jcy/
│   ├── jcy_table.json             # 飞书多维表格原始记录（JSON）
│   ├── jcy_docs.yaml              # 飞书文档内容（YAML，含正文）
│   ├── jcy_insights.json          # 结构化提取结果 + 单一真值源（含 advice_file/extracted_at 状态）
│   └── advice/                    # Perplexity 生成的投资建议 Markdown 文件
│       └── YYYY-MM-DD__标题.md     # 复合命名（date + 安全化 title），对应每期 JCY 文章
│
├── authorize/                     # 飞书 OAuth 授权工具（手动运行一次性获取/刷新 token，不属于自动化流水线）
│   ├── get_usr_key.py             # 完整授权（打开浏览器换 code→token）/ 静默刷新（用 refresh_token），写入 feishu_key/*.txt
│   ├── feishu_callback.py         # 本地自签名 HTTPS 回调服务（Flask，:8080/callback），接收飞书授权 code 供手动复制
│   └── feishu_key/
│       ├── feishu_token.txt       # 飞书 Bearer Token（get_usr_key.py 写入 / jcy/feishu.py 读取）
│       ├── feishu_token_refresh.txt # refresh_token（get_usr_key.py 静默刷新用，不提交）
│       └── localhost.crt / .key   # feishu_callback.py 自动生成的自签名证书（不提交）
│
├── ── 测试 ────────────────────────────────────
├── tests/                         # pytest（不联网、不读真实 data/）
│
└── output/                        # 回测图表和 CSV 输出目录
```

---

## 核心功能模块

### 0. 飞书 OAuth 授权工具 (`authorize/`，手动运行，独立于自动化流水线)

**职责：** 首次获取 / 过期后刷新 `jcy/feishu.py` 采集数据所需的飞书 `user_access_token`，产出 `authorize/feishu_key/feishu_token.txt`（`TOKEN_FILE` 指向的文件）。不在 Step 1-4 的自动执行链路中，需要人工触发。

- `get_usr_key.py` — 主入口，交互式二选一：
  1. **完整授权**：打开浏览器走飞书 OAuth 页面 → 用户手动复制回调 URL 中的 `code` 粘贴回终端 → 换取 `access_token`/`refresh_token`，写入 `feishu_token.txt` / `feishu_token_refresh.txt`
  2. **静默刷新**：用已保存的 `refresh_token` 换新 token，免去重新授权；失败时自动回退到完整授权
- `feishu_callback.py` — 可选的本地 HTTPS 回调服务（Flask + 自签名证书，监听 `:8080/callback`），把飞书跳转回来的 `code` 打印/展示在页面上，方便手动复制（`get_usr_key.py` 的完整授权流程也可以只用浏览器地址栏读 code，不强依赖此服务）

### 1. 数据准备一体化流水线 (`jcy/` 包，入口 `prepare_jcy_data.py`)

**职责：** 整合 Step 1-3，完成飞书数据采集 → AI 投资建议 → 结构化提取。实现拆分在 `jcy/`（config/store/feishu/advice/extract/pipeline），`prepare_jcy_data.py` 为薄入口。

**Step 1 — 飞书数据采集：**
```
wiki_token → bitable app_token
    → get_all_records()            # 分页读取多维表格所有记录
    → extract_links_from_records() # 正则提取飞书文档 URL
    → fetch_doc_content_json()     # 按文档类型（docx/wiki）调用对应 API
    → save_data()                  # 表格 → JSON，文档 → YAML
```

**Step 2 — AI 投资建议生成（Perplexity）：**
- 模型：`sonar-reasoning-pro`（联网搜索 + 推理，去掉 `<think>` 标签后保留正文）
- system prompt 外置于 `jcy/prompts/step2_advice_system.md`
- 增量机制：以 `jcy_insights.json` 为单一真值源；record 的 `advice_file` 字段存在且文件实际存在即跳过
- 原子性：先落 advice 文件 → 再写 record 的 `advice_file` 字段
- 输出格式：今日核心观点 / 股票行业详解 / 投资小白行动建议 / 风险提示 / 一句话总结
- 超时保护：连续超时 `MAX_CONSECUTIVE_TIMEOUTS` 次时终止，不写入无效响应

**Step 3 — 结构化信息提取（LLM，按 provider 回退）：**
- `S3_PROVIDERS` 回退顺序：**DashScope**（默认启用）→ Azure OpenAI（默认注释关闭）→ Coze（默认注释关闭）→ **Custom**（OpenAI 兼容端点，设置 `CUSTOM_API_KEY` 即自动启用，未注释）
- 原文 + 建议文档 → `response_format=json_object` → 结构化 JSON
- system prompt 外置于 `jcy/prompts/step3_extract_system.md`
- 输出 schema：
```json
{
  "companies": [{"name", "code", "exchange", "rating", "rating_reason"}],
  "markets": ["A股", "港股", ...],
  "tendency": "整体投资倾向（一句话）",
  "key_advice": ["建议1", "建议2", ...]
}
```
- 增量机制：以 `(date, title)` 复合键去重；record 含 `extracted_at` 即跳过
- 去重统一以 `jcy_insights.json` 为单一真值源；`advice_cache.json` 已废弃

**环境变量：**
- `TOKEN_FILE` — 飞书 Bearer Token 文件路径（`authorize/get_usr_key.py` 写入，`jcy/feishu.py` 读取）
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` — 飞书应用凭证，仅 `authorize/get_usr_key.py` 换取/刷新 token 时使用
- `JCY_WIKI_TOKEN` — 飞书 Wiki 节点 token
- `JCY_APP_TABLE_ID` — 多维表格 ID
- `JCY_VIEW_ID` — 视图 ID（可选）
- `PPLX_API_KEY` — Perplexity API 密钥
- `PPLX_GROUP_ID` — API Group ID
- `DASHSCOPE_API_KEY` — DashScope API 密钥（Step 3 默认 provider）
- `DASHSCOPE_BASE_URL` — DashScope 端点（默认 compatible-mode/v1）
- `DASHSCOPE_MODEL` — DashScope 模型名
- `CUSTOM_API_KEY` / `CUSTOM_BASE_URL` / `CUSTOM_MODEL` — 自定义 OpenAI 兼容 provider（Step 3 回退候选，`CUSTOM_API_KEY` 非空即启用）
- `AZURE_OPENAI_*` / `COZE_*` — 可选 provider，默认在 `S3_PROVIDERS` 中注释关闭

---

### 2. MACD 策略回测引擎 (`backtest/engine.py`)

**职责：** 核心回测引擎（纯函数，无 CLI），被所有回测入口脚本复用。`macd_analysis.py` 为薄入口，re-export `run_backtest`/`plot_backtest`/`fetch_stock_data` 以兼容历史 `from macd_analysis import ...` 导入。

**关键函数：**
- `fetch_stock_data(symbol, start, end)` — 从 `lib.market_data` 再导出（canonical 位置在 `backtest/lib/market_data.py`）
- `run_backtest(symbol, strategy, capital, stop_loss, take_profit, ...)` — 执行回测
  - 按信号买卖，100 股整数手，T 日信号在 **T+1 开盘**成交（`signal.shift(1)`）
  - **单根 K 线内的事件顺序**：① 开盘执行挂单 → ② 盘中止损/止盈 → ③ 收盘估值。
    当日刚建仓的不检查止损（A 股 T+1，当天买当天卖不掉）
  - **A 股成交约束**：开盘涨停买不进、开盘跌停卖不掉、`volume==0` 停牌不可成交；
    未成交的信号顺延最多 `max_pending_days` 天，之后作废（避免一字连板追到天上）。
    涨跌停幅度由 `infer_limit_pct(symbol)` 按代码前缀推断（主板 10% / 双创 20% / 北交所 30%）
  - **成本**：佣金（双边万三，单笔最低 5 元）+ 印花税（单边千一）+ 双边滑点（默认千一）
  - `eval_start="YYYYMMDD"` — 统计窗口起点，把指标预热期排除在收益/回撤/夏普/基准之外
  - `df=` — 直接注入行情，跳过网络请求（测试与参数扫描复用）
  - 基准（买入持有）与策略同口径：窗口首日**开盘价**买入、末日收盘估值
- `plot_backtest(result, save_path)` — 标准 4 面板图（价格+信号、指标、权益、回撤）

**共享配置层（`backtest/config.py`）：** 三个单股入口共用
- `load_backtest_config(filename, *, defaults)` → `BacktestConfig`：统一解析 `backtest/presets/*.ini` 的 `[backtest]` 段（end_date 默认今日、止损止盈空值转 None、proxy 写环境变量、缺失时按 defaults 写出）；策略专属参数经 `cfg.get_int/get_bool/get_float` 从 `.extra` 读取
- `execution_kwargs(cfg)` → dict：从 `.ini` 读成本与交易约束（commission_rate / min_commission / stamp_duty / slippage / limit_move_check / max_pending_days），直接 `**` 展开给 `run_backtest`，保证三个入口的成本假设一致
- `OutputPaths(save_dir, prefix, name, symbol, end_date)`：统一输出路径（`.chart/.csv/.status`），`OutputPaths.safe()` 清洗文件名

**CLI：** `python backtest/macd_analysis.py --config jxty_jcy_260104.ini`

---

### 3. 策略体系 (`strategies/`)

| 策略类 | 文件 | 适用场景 |
|--------|------|----------|
| `MACDStrategy` | `macd.py` | 教科书金叉/死叉，无过滤 |
| `LuMACDStrategy` | `lu_macd.py` | 三级底部确认（0 轴上，底背离，金叉），长线建仓 |
| `LuMACDBullStrategy` | `lu_macd_bull.py` | 牛市过滤（大盘月线）+ 截取红柱最陡段，高频战术 |

**BaseStrategy 接口（必须实现）：**
```python
prepare(df) -> df          # 计算指标，生成 signal 列（1/-1/0）
plot_indicators(ax, df, colors) -> None
name: str                   # 策略名（图表标题）
params: dict               # 参数字典（展示用）
```

**BaseStrategy 共享方法：**
- `_ema(series, period)` — 静态方法，EMA 计算，所有 MACD 策略子类共用（避免各子类重复定义）
- `_resample_period(df, rule, agg, drop_incomplete=True)` — **跨周期重采样必须走这里**。
  以区间内最后一个**真实交易日**为标签（不是 `"MS"` 月初），并丢掉末尾未走完的那根 K 线。
  直接用 `df.resample("MS")` 会把月末收盘价打上月初标签，构成整月的未来函数
- `_align_to_daily(series, daily_index)` — 低频 → 日线对齐。先在**并集**索引上 ffill 再收敛回日线；
  直接 `series.reindex(daily_index).ffill()` 会把标签不在交易日上的 K 线整根丢掉

> ⚠️ 多周期策略的时序铁律：周/月线信号只在该区间**收盘当天**生效，再由引擎 `shift(1)` 到 T+1 成交。
> `tests/test_strategy_lookahead.py` 用"截断数据重算、历史信号不得改变"的属性测试守住这一点。

**LuMACDBullStrategy 特殊设计：**
- 构造函数接受 `index_df`（大盘数据），`prepare()` 参数中的 `index_df` 优先，否则 fallback 到构造函数传入的值
- 牛市判断：大盘**已收盘**月线 DIF > 0 且 DIF > DEA
- 买入：近 `cross_window`(默认3) 根内出现金叉 **且** 红柱连续 `expand_bars`(默认2) 根拉长。
  单根"红柱拉长"在金叉当根恒为真（hist 由 ≤0 翻正），起不到过滤作用，故必须连续确认；
  `expand_bars=1, cross_window=1` 可复现旧口径
- 卖出模式：`shrink_exit=True`（红柱缩短即走）或 `False`（等死叉）

---

### 4. 批量回测 (`backtest/jcy_macd_bull_batch.py` + `backtest/batch_report.py`)

**职责：** 读取 `jcy_insights.json` → 筛选 A 股推荐 → 批量执行牛市策略回测 → 横截面汇总

**关键逻辑：**
- `is_ashare_code(code)` — 过滤 A 股代码（沪深交易所）
- `BullStrategyAdapter` — 包装 `LuMACDBullStrategy`，将推荐日期前的信号清零（避免未来数据）
- `backtest_one()` — 单股回测，`eval_start=推荐日`，预热期不计入统计；返回 result dict
- 大盘指数只取一次，全部个股共用（原来每只票各拉一次）
- 预热 600 自然日：牛市过滤器要算大盘**月线** MACD，EMA-26 需要 26 根月线 ≈ 2 年

**汇总输出（`batch_report.py`）：**
- `output/summary.csv` — 每股一行（收益/基准/超额/回撤/夏普/胜率/受阻次数），按超额收益排序
- `output/summary_portfolio.csv|.png` — 等权组合净值 vs 大盘。各股推荐日不同，采用
  "平均在场净值"：每只票从自己的统计起点归一为 1.0，按日历对齐后对当日已在场的取算术平均
- 控制台汇总：跑赢基准比例、超额收益均值/**中位数**、最好/最差各 5 只。
  收益分布右偏，均值会被个别翻倍股拉高，中位数才是典型结果

**CLI：** `python backtest/jcy_macd_bull_batch.py [--output output/]`

---

### 4b. 参数敏感性扫描 (`backtest/param_sweep.py`)

**职责：** 在推荐股票池上网格遍历两个参数轴，判断当前参数是"策略有效"还是"恰好挑中幸运点"。

- 可扫轴见 `AXES`：`expand_bars` / `cross_window` / `fast` / `slow` / `signal_period` / `shrink_exit` / `stop_loss` / `take_profit`
- 判读看**稳健性**不看最大值：整片偏绿=结论稳健；孤立亮点、邻居全负=过拟合
- 输出 `output/sweep/sweep_results.csv` + `sweep_heatmap.png`（默认参数格用金框标出）
- 行情在进程内缓存，同一只票整个网格只拉一次

**CLI：** `python backtest/param_sweep.py [--axis stop_loss take_profit] [--limit 20]`

---

### 5. 包内工具层（`jcy/lib/` 与 `backtest/lib/`）

工具按归属沉入各自包的 `lib/` 子目录，不再有根级 `utils/` 伪共享包。每个模块只有单一归属者，import 语义清晰。

**`jcy/lib/`**（被流水线步骤调用，`from jcy.lib.x import`；跨包时 backtest 也 `from jcy.lib.common import`）

| 模块 | 核心内容 |
|------|----------|
| `common.py` | `title_to_date()`、`title_to_filename()`、`record_key()`、`safe_title()`、`load_docs()`、`load_candidates()`、`is_ashare_code()`、路径常量（`JSON_PATH/DOCS_FILE/ADVICE_DIR`） |
| `text.py` | `blocks_to_text()`（飞书 block → 文本）、`parse_json_loose()`（宽松 JSON 解析） |
| `pplx.py` | `PerplexityAPI` 客户端（`chat()`、`sonar_deep_research()`） |

**`backtest/lib/`**（被引擎/入口脚本复用，脚本模式下 `from lib.x import`）

| 模块 | 核心内容 |
|------|----------|
| `market_data.py` | `fetch_stock_data()` + `fetch_index_data()` — 个股/指数行情，akshare → baostock → yfinance 三源（含限流重试）；`ADJUST` 统一三源复权口径，回测默认 **hfq 后复权**（qfq 会随分红回溯改写历史价，结果不可复现） |
| `plotting.py` | `COLORS` 字典（GitHub Dark 配色）、`setup_matplotlib()`、`style_ax(ax)` |
| `bull_backtest.py` | `BullStrategyAdapter`（牛市策略通用适配器；绘图/CSV 在 `backtest/bull_report.py`） |

---

## 数据流图

```
飞书多维表格
    │
    ▼ prepare_jcy_data.py（Step 1）
data/jcy/jcy_docs.yaml          （原始文档正文）
    │
    ├─ prepare_jcy_data.py（Step 2，Perplexity sonar）
    │                    ──────────► data/jcy/advice/YYYY-MM-DD.md
    │
    └─ prepare_jcy_data.py（Step 3，DashScope/Custom LLM）
                         ──────────► data/jcy/jcy_insights.json
                                         {companies, rating, markets, key_advice}
                                                 │
                                  backtest/jcy_macd_bull_batch.py（Step 4）
                                                 │
                                      akshare/yfinance 行情
                                                 │
                                      LuMACDBullStrategy 回测
                                                 │
                                      output/batch_YYYYMMDD/
                                          ├── *.png  （5面板图表）
                                          ├── *.csv  （交易记录）
                                          └── *_daily_status.csv
```

---

## 回测预设格式 (`backtest/presets/*.ini`)

```ini
[backtest]
symbol      = 600519       # 股票代码
name        = maotai       # 名称（用于文件名）
start_date  = 20180101
end_date    =              # 留空 = 今日
index_symbol = 000300      # 大盘指数（牛市判断）
capital     = 100000
stop_loss   =              # 如 0.10 = 10%，留空不设
take_profit =
save_chart_dir = output/
proxy       =              # 如 http://127.0.0.1:7890

# LuMACDBull 专属
shrink_exit = true         # true=红柱缩短即走，false=等死叉
```

---

## 环境变量 (`.env`)

```
TOKEN_FILE=...                     # 飞书 Bearer Token 文件路径
FEISHU_APP_ID=...                  # 飞书应用 ID（仅 authorize/get_usr_key.py 用）
FEISHU_APP_SECRET=...              # 飞书应用密钥（仅 authorize/get_usr_key.py 用）
PPLX_API_KEY=pplx-...
PPLX_GROUP_ID=...
JCY_WIKI_TOKEN=...
JCY_APP_TABLE_ID=...
JCY_VIEW_ID=...

# Step 3 默认 provider：DashScope
DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=deepseek-v4-pro

# Step 3 回退 provider：Custom（OpenAI 兼容端点，非空即启用，未注释）
CUSTOM_API_KEY=...
CUSTOM_BASE_URL=...
CUSTOM_MODEL=...

# 可选 provider（默认在 S3_PROVIDERS 中注释关闭）
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=gpt-5
AZURE_OPENAI_API_VERSION=2024-12-01-preview
COZE_URL=...
```

---

## 运行方式（纯源码，无需安装本项目）

项目不打包、不依赖 `pip install -e .`。**所有命令从仓库根目录执行**：

- `jcy/`、`strategies/` 等包直接被 Python 解析（仓库根已在 `sys.path`），如 `python prepare_jcy_data.py`、`pytest`。
- `backtest/` 为脚本模式（无 `__init__.py`），以 `python backtest/x.py` 运行，脚本目录自动入 `sys.path[0]`；测试经 `tests/conftest.py` 把 `backtest/` 加入 path。

只需安装第三方依赖（见「环境要求」），无本项目安装步骤。

---

## 运行顺序

```bash
# 完整流水线（按序执行）
python prepare_jcy_data.py                  # Step 1-3：拉取数据 → AI建议 → 结构化提取
python backtest/jcy_macd_bull_batch.py      # Step 4：批量量化回测

# 分时择时（日线信号 + 分时 MACD 共振）
python backtest/jcy_intraday_timing.py                  # 全部候选股
python backtest/jcy_intraday_timing.py --code 600519    # 单股分析
python backtest/jcy_intraday_timing.py --period 60      # 60min K 线

# 独立单股回测
python backtest/macd_analysis.py --config jxty_jcy_260104.ini
python backtest/lu_macd_analysis.py
python backtest/lu_macd_bull_analysis.py

# 参数敏感性（判断是否过拟合，建议先 --limit 试跑）
python backtest/param_sweep.py --limit 20
python backtest/param_sweep.py --axis stop_loss take_profit

# 测试（离线，不联网、不读真实 data/）
pytest
```
