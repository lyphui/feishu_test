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
  ```

## 项目概述

飞书股市分析 + MACD 量化回测一体化平台。

**完整流水线：**
1. **数据采集**：从飞书多维表格读取（JCY）股市分析文章
2. **AI 分析**：用 Perplexity sonar-reasoning-pro 生成投资建议（面向小白）
3. **结构化提取**：用 Azure OpenAI GPT 提取公司/代码/评级等结构化信息
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
│   ├── extract.py                 # Step 3：LLM 结构化提取（DashScope/Azure/Coze 回退）
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
│   ├── engine.py                  # 核心回测引擎：run_backtest / plot_backtest（无 CLI）
│   ├── config.py                  # 共享配置层：BacktestConfig / load_backtest_config / OutputPaths
│   ├── bull_report.py             # 牛市策略报告：plot_bull_backtest / export_bull_daily_status
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
│       ├── market_data.py         # 行情数据获取：个股 + 指数（akshare → yfinance 双源，含限流重试）
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
├── authorize/feishu_key/
│   └── feishu_token.txt           # 飞书 Bearer Token（手动维护）
│
├── ── 测试 ────────────────────────────────────
├── tests/                         # pytest（不联网、不读真实 data/）
│
└── output/                        # 回测图表和 CSV 输出目录
```

---

## 核心功能模块

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
- 默认启用 **DashScope**（`S3_PROVIDERS` 中 Azure / Coze 默认注释关闭，可按需开启）
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
- `TOKEN_FILE` — 飞书 Bearer Token 文件路径
- `JCY_WIKI_TOKEN` — 飞书 Wiki 节点 token
- `JCY_APP_TABLE_ID` — 多维表格 ID
- `JCY_VIEW_ID` — 视图 ID（可选）
- `PPLX_API_KEY` — Perplexity API 密钥
- `PPLX_GROUP_ID` — API Group ID
- `DASHSCOPE_API_KEY` — DashScope API 密钥（Step 3 默认 provider）
- `DASHSCOPE_BASE_URL` — DashScope 端点（默认 compatible-mode/v1）
- `DASHSCOPE_MODEL` — DashScope 模型名
- `AZURE_OPENAI_*` / `COZE_*` — 可选 provider，默认在 `S3_PROVIDERS` 中注释关闭

---

### 2. MACD 策略回测引擎 (`backtest/engine.py`)

**职责：** 核心回测引擎（纯函数，无 CLI），被所有回测入口脚本复用。`macd_analysis.py` 为薄入口，re-export `run_backtest`/`plot_backtest`/`fetch_stock_data` 以兼容历史 `from macd_analysis import ...` 导入。

**关键函数：**
- `fetch_stock_data(symbol, start, end)` — 从 `lib.market_data` 再导出（canonical 位置在 `backtest/lib/market_data.py`）
- `run_backtest(symbol, strategy, capital, stop_loss, take_profit)` — 执行回测
  - 按信号买卖，100 股整数手，A 股印花税（单边 0.1%）+ 佣金（双边 0.03%）
  - 持仓期间按收盘价估值，生成权益曲线和回撤序列
- `plot_backtest(result, save_path)` — 标准 4 面板图（价格+信号、指标、权益、回撤）

**共享配置层（`backtest/config.py`）：** 三个单股入口共用
- `load_backtest_config(filename, *, defaults)` → `BacktestConfig`：统一解析 `backtest/presets/*.ini` 的 `[backtest]` 段（end_date 默认今日、止损止盈空值转 None、proxy 写环境变量、缺失时按 defaults 写出）；策略专属参数经 `cfg.get_int/get_bool/get_float` 从 `.extra` 读取
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

**LuMACDBullStrategy 特殊设计：**
- 构造函数接受 `index_df`（大盘数据），`prepare()` 参数中的 `index_df` 优先，否则 fallback 到构造函数传入的值
- 牛市判断：大盘月线 DIF > 0 且 DIF > DEA
- 卖出模式：`shrink_exit=True`（红柱缩短即走）或 `False`（等死叉）

---

### 4. 批量回测 (`backtest/jcy_macd_bull_batch.py`)

**职责：** 读取 `jcy_insights.json` → 筛选 A 股推荐 → 批量执行牛市策略回测

**关键逻辑：**
- `is_ashare_code(code)` — 过滤 A 股代码（沪深交易所）
- `BullStrategyAdapter` — 包装 `LuMACDBullStrategy`，将推荐日期前的信号清零（避免未来数据）
- `backtest_one()` — 单股回测，从 `trade_start_date`（推荐日）开始计算
- 批量输出：图表 + 交易 CSV + 每日状态 CSV 到 `output/batch_YYYYMMDD/`

**CLI：** `python backtest/jcy_macd_bull_batch.py [--date 20260226] [--output output/]`

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
| `market_data.py` | `fetch_stock_data()` + `fetch_index_data()` — 个股/指数行情，akshare → yfinance 双源（含限流重试） |
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
    └─ prepare_jcy_data.py（Step 3，Azure GPT）
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
PPLX_API_KEY=pplx-...
PPLX_GROUP_ID=...
JCY_WIKI_TOKEN=...
JCY_APP_TABLE_ID=...
JCY_VIEW_ID=...

# Step 3 默认 provider：DashScope
DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=deepseek-v4-pro

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
```
