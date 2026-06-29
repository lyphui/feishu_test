# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# feishu_test — 代码库架构文档

## 环境要求

- Python **≥ 3.11**
- 安装依赖（可编辑模式，使 `strategies/`、`utils/` 可直接 import）：
  ```bash
  pip install -e .
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
├── pyproject.toml                 # 可编辑安装配置（pip install -e .）
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
│   └── pipeline.py                # main 编排（--strict / --log-file）
│
├── ── 回测脚本 ─────────────────────────────────
├── backtest/
│   ├── jcy_macd_bull_batch.py     # Step 4: 批量 MACD 牛市策略回测（读 jcy_insights.json）
│   ├── jcy_intraday_timing.py     # 日线信号 + 分时择时（多周期共振）
│   ├── macd_analysis.py           # 单股 MACD 教科书策略回测 + 核心回测引擎
│   ├── lu_macd_analysis.py        # 单股卢式 MACD 三级底部策略回测
│   └── lu_macd_bull_analysis.py   # 单股卢式 MACD 牛市动能截取策略回测
│
├── ── 策略包 ──────────────────────────────────
├── strategies/
│   ├── __init__.py                # 导出三个策略类
│   ├── base.py                    # BaseStrategy 抽象基类（含共享 _ema() 方法）
│   ├── macd.py                    # MACDStrategy（金叉/死叉，教科书版）
│   ├── lu_macd.py                 # LuMACDStrategy（三级底部确认，长线建仓）
│   └── lu_macd_bull.py            # LuMACDBullStrategy（牛市截陡坡，高频战术）
│
├── ── 工具包 ──────────────────────────────────
├── utils/
│   ├── plotting.py                # 共享绘图样式（GitHub Dark 配色 + matplotlib 配置）
│   ├── market_data.py             # 共享行情数据获取：个股 + 指数（akshare → yfinance 双源，含限流重试）
│   ├── bull_backtest.py           # 牛市策略专用：Adapter + 绘图 + CSV 导出
│   ├── jcy_common.py              # JCY 流水线共享工具（日期解析、文件命名、候选股筛选、YAML 加载）
│   └── pplx.py                    # Perplexity API 客户端封装
│
├── ── 配置文件 ─────────────────────────────────
├── config/
│   ├── jxty_jcy_260104.ini        # 单股 MACD 回测示例配置
│   ├── lu_macd_config.ini         # 卢式 MACD 策略回测配置
│   ├── lu_macd_bull_config.ini    # 卢式牛市策略回测配置
│   └── rjgd_syr_260130.ini        # 其他回测配置示例
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
├── ── 提示词 ──────────────────────────────────
├── prompts/
│   ├── step2_advice_system.md     # Step 2 Perplexity system prompt
│   └── step3_extract_system.md    # Step 3 结构化提取 system prompt
│
├── ── 运维脚本 ────────────────────────────────
├── scripts/
│   └── migrate_compound_key.py    # 存量数据复合键迁移（零 API 调用）
│
├── ── 测试 ────────────────────────────────────
├── tests/                         # pytest（不联网、不读真实 data/）
│
├── deprecated/                    # 已废弃脚本（被 prepare_jcy_data.py 整合替代）
│   ├── get_jcy_data.py
│   ├── analyze_jcy.py
│   ├── extract_jcy_insights.py
│   └── ...
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
- system prompt 外置于 `prompts/step2_advice_system.md`
- 增量机制：以 `jcy_insights.json` 为单一真值源；record 的 `advice_file` 字段存在且文件实际存在即跳过
- 原子性：先落 advice 文件 → 再写 record 的 `advice_file` 字段
- 输出格式：今日核心观点 / 股票行业详解 / 投资小白行动建议 / 风险提示 / 一句话总结
- 超时保护：连续超时 `MAX_CONSECUTIVE_TIMEOUTS` 次时终止，不写入无效响应

**Step 3 — 结构化信息提取（LLM，按 provider 回退）：**
- 默认启用 **DashScope**（`S3_PROVIDERS` 中 Azure / Coze 默认注释关闭，可按需开启）
- 原文 + 建议文档 → `response_format=json_object` → 结构化 JSON
- system prompt 外置于 `prompts/step3_extract_system.md`
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

### 2. MACD 策略回测引擎 (`backtest/macd_analysis.py`)

**职责：** 核心回测引擎，被所有回测入口脚本复用

**关键函数：**
- `fetch_stock_data(symbol, start, end)` — 从 `utils.market_data` 再导出（canonical 位置已移至 `utils/market_data.py`）
- `run_backtest(symbol, strategy, capital, stop_loss, take_profit)` — 执行回测
  - 按信号买卖，100 股整数手，A 股印花税（单边 0.1%）+ 佣金（双边 0.03%）
  - 持仓期间按收盘价估值，生成权益曲线和回撤序列
- `plot_backtest(result, save_path)` — 标准 4 面板图（价格+信号、指标、权益、回撤）

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

### 5. 共享工具包 (`utils/`)

| 模块 | 核心内容 |
|------|----------|
| `plotting.py` | `COLORS` 字典（GitHub Dark 配色）、`setup_matplotlib()`、`style_ax(ax)` |
| `market_data.py` | `fetch_stock_data()` + `fetch_index_data()` — 个股/指数行情，akshare → yfinance 双源（含限流重试） |
| `bull_backtest.py` | `BullStrategyAdapter`、`plot_bull_backtest()`、`export_bull_daily_status()` |
| `jcy_common.py` | `title_to_date()`、`title_to_filename()`、`load_docs()`、`load_candidates()`、`is_ashare_code()`、路径常量（`JSON_PATH`） |
| `pplx.py` | `PerplexityAPI` 客户端（`chat()`、`sonar_deep_research()`） |

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

## 配置文件格式 (`config/*.ini`)

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

## 包安装

项目使用 `pyproject.toml` 支持可编辑安装，消除 `sys.path.insert` hack：

```bash
pip install -e .
```

安装后 `strategies/`、`utils/` 等包可在任意目录直接 import。

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
