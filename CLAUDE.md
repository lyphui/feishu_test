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
  # 强烈建议装上：行情第二数据源（akshare 失败时自动回退，且能给出正确的 hfq 后复权口径）
  # 也是**派息数据的唯一来源**（`lib/price_store.update_dividends`），缺了它算不出股息率
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
├── CLAUDE.md                      # 本文件（索引）
├── docs/                          # 各模块详细设计文档（按需读取，见下方「详细文档索引」）
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
│   ├── exec_bench.py              # 日内下单方案实测台：买/卖两侧，按股票池分别出数
│   ├── oil_track.py               # 油气双雄长期跟踪：增量更新 + 当前状态与打法 + 连续全样本回测
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
│       ├── price_store.py         # 本地行情仓库：增量更新 + 重叠对账 + 分红记录（data/market/）
│       ├── intraday_store.py      # 本地分时仓库：**不复权**含 amount，供 execution.py 用（不入库）
│       ├── swings.py              # ZigZag 波段分解、独立回撤事件与修复耗时剖面
│       ├── regime.py              # 市场状态分类（趋势上行/宽幅震荡/趋势下行，严格只用历史数据）
│       ├── ladder.py              # 分批建仓模拟器：梯度加仓/定投/网格/按状态自适应切换
│       ├── plotting.py            # 绘图样式（GitHub Dark 配色 + matplotlib 配置）
│       ├── bull_backtest.py       # 牛市策略通用适配器 BullStrategyAdapter
│       ├── oil_price.py           # Brent/WTI/SC 原油价格（新浪源）+ 油价→股价传导相关性分析
│       └── execution.py           # 日内下单方案测算（VWAP 基准，买卖双向，与标的/策略无关的度量层）
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
├── data/market/                   # 本地行情仓库（price_store.py 维护，增量追加）
│   ├── daily/{symbol}_{adjust}.csv       # 日线 OHLCV，hfq=回测口径 / none=盘面实际价
│   ├── daily/{symbol}_{adjust}.meta.json # 已覆盖的**请求区间** + 最后更新时间
│   ├── dividend/{symbol}.csv             # 派息记录（baostock），算股息率用
│   └── intraday/{symbol}_{p}m_none.csv   # 分时 K 线，**不复权**含 amount（.gitignore，可重建）
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
├── ── 更新日志 ──────────────────────────────────
├── changelog/                     # **一条改动一个文件**，不要写进同一个 CHANGELOG.md
│   ├── CHANGELOG.md               # 只做索引：日期 / 链接 / 主题 / 是否含行为变更
│   └── YYYY-MM-DD-<英文短横线标题>.md  # 条目正文（同一天多条用不同标题后缀区分）
│
├── ── 测试 ────────────────────────────────────
├── tests/                         # pytest（不联网、不读真实 data/）
│
└── output/                        # 回测图表和 CSV 输出目录
```

---

## 核心功能模块（速览 + 详细文档索引）

下表只给每个模块一句话定位；**具体实现细节、设计取舍、踩过的坑都拆到 `docs/` 里**，
按需读取对应文档，不要求每次都通读全部——这正是本次重构要解决的"CLAUDE.md 太臃肿"问题。

| # | 模块 | 一句话 | 详细文档 |
|---|------|--------|----------|
| 0-1 | 飞书授权 + JCY 数据流水线 | `authorize/` 换取 token；`jcy/` 完成 Step1-3（采集→AI建议→结构化提取），含全部 provider 与环境变量 | [docs/jcy-pipeline.md](docs/jcy-pipeline.md) |
| 2 | MACD 回测引擎 | `backtest/engine.py` 核心 `run_backtest`：A股成交约束、T+1、成本口径、`eval_start` 窗口；`backtest/config.py` 共享配置层；预设 `.ini` 格式 | [docs/backtest-engine.md](docs/backtest-engine.md) |
| 3 | 策略体系 | `strategies/`：MACDStrategy / LuMACDStrategy / LuMACDBullStrategy，BaseStrategy 共享方法与时序铁律 | [docs/strategies.md](docs/strategies.md) |
| 4, 4b | 批量回测 + 参数扫描 | `jcy_macd_bull_batch.py` 批量跑 + `batch_report.py` 横截面汇总；`param_sweep.py` 网格扫描判断过拟合，含样本外验证 | [docs/batch-and-sweep.md](docs/batch-and-sweep.md) |
| 4c | 长期跟踪 + 分批建仓 | `oil_track.py` + `lib/price_store\|swings\|regime\|ladder`：本地行情仓库、市场状态分类、分批建仓模拟器 | [docs/tracking-and-ladder.md](docs/tracking-and-ladder.md) |
| 4d | 日内下单测算 | `lib/execution.py` + `exec_bench.py`：VWAP 基准、买卖双向度量、JCY/油气两池实测结论表 | [docs/execution-bench.md](docs/execution-bench.md) |
| 5 | 包内工具层参考表 | `jcy/lib/` 与 `backtest/lib/` 全部模块的速查表 | [docs/lib-reference.md](docs/lib-reference.md) |

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

每个变量的用途与所属 provider 见 [docs/jcy-pipeline.md](docs/jcy-pipeline.md)。

---

## 更新日志约定

改动记录放在 `changelog/`，**一条改动一个文件**，命名 `YYYY-MM-DD-<英文短横线标题>.md`。
不要把多条改动追加进同一个 `CHANGELOG.md` —— 那个文件只作索引表，正文一律写进条目文件。

新增一条时：建条目文件（正文用 `#` 一级标题），并在 `changelog/CHANGELOG.md`
索引表**最上方**加一行，标明是否「含行为变更」。会改变已有回测输出数值的改动
必须标为「是」，并在条目里单列「行为变更」一节。

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

# 日内下单方案实测（买/卖两侧，两个池子分别出数，结论不可互相外推）
python backtest/exec_bench.py --universe jcy --side both --limit 45
python backtest/exec_bench.py --universe oil --side sell
python backtest/exec_bench.py --universe oil --offline   # 不联网，只读分时缓存

# 独立单股回测
python backtest/macd_analysis.py --config jxty_jcy_260104.ini
python backtest/lu_macd_analysis.py
python backtest/lu_macd_bull_analysis.py

# 油气双雄长期跟踪（增量更新本地行情 → 当前状态与挂单价）
python backtest/oil_track.py                      # 增量更新 + 跟踪报告
python backtest/oil_track.py --offline            # 不联网，只读本地缓存
python backtest/oil_track.py --backtest --chart   # 连续全样本回测 + 出图

# 参数敏感性（判断是否过拟合，建议先 --limit 试跑）
python backtest/param_sweep.py --limit 20
python backtest/param_sweep.py --axis stop_loss take_profit
python backtest/param_sweep.py --oos-frac 0.3          # 留最近 30% 推荐做样本外验证

# 测试（离线，不联网、不读真实 data/）
pytest
```

---

## 详细文档索引 (`docs/`)

| 文档 | 内容 |
|------|------|
| [docs/jcy-pipeline.md](docs/jcy-pipeline.md) | 飞书 OAuth 授权 + JCY 数据流水线 Step1-3，含全部环境变量 |
| [docs/backtest-engine.md](docs/backtest-engine.md) | `engine.py` 核心回测逻辑、`config.py` 共享配置层、预设 `.ini` 格式 |
| [docs/strategies.md](docs/strategies.md) | 策略体系（MACD / LuMACD / LuMACDBull）与时序铁律 |
| [docs/batch-and-sweep.md](docs/batch-and-sweep.md) | 批量回测横截面汇总 + 参数敏感性扫描（过拟合判断） |
| [docs/tracking-and-ladder.md](docs/tracking-and-ladder.md) | 长期跟踪（行情仓库/波段/市场状态）+ 分批建仓模拟器 |
| [docs/execution-bench.md](docs/execution-bench.md) | 日内下单方案测算方法论 + JCY/油气两池实测结论 |
| [docs/lib-reference.md](docs/lib-reference.md) | `jcy/lib/` 与 `backtest/lib/` 全部模块速查表 |
