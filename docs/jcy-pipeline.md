# JCY 数据准备流水线（`authorize/` + `jcy/`）

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。覆盖飞书授权、Step 1-3（采集 → AI 建议 → 结构化提取）。

## 0. 飞书 OAuth 授权工具 (`authorize/`，手动运行，独立于自动化流水线)

**职责：** 首次获取 / 过期后刷新 `jcy/feishu.py` 采集数据所需的飞书 `user_access_token`，产出 `authorize/feishu_key/feishu_token.txt`（`TOKEN_FILE` 指向的文件）。不在 Step 1-4 的自动执行链路中，需要人工触发。

- `get_usr_key.py` — 主入口，交互式二选一：
  1. **完整授权**：打开浏览器走飞书 OAuth 页面 → 用户手动复制回调 URL 中的 `code` 粘贴回终端 → 换取 `access_token`/`refresh_token`，写入 `feishu_token.txt` / `feishu_token_refresh.txt`
  2. **静默刷新**：用已保存的 `refresh_token` 换新 token，免去重新授权；失败时自动回退到完整授权
- `feishu_callback.py` — 可选的本地 HTTPS 回调服务（Flask + 自签名证书，监听 `:8080/callback`），把飞书跳转回来的 `code` 打印/展示在页面上，方便手动复制（`get_usr_key.py` 的完整授权流程也可以只用浏览器地址栏读 code，不强依赖此服务）

## 1. 数据准备一体化流水线 (`jcy/` 包，入口 `prepare_jcy_data.py`)

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

## 环境变量（本流水线相关）

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

见 [CLAUDE.md](../CLAUDE.md) 的 `.env` 完整清单获取所有变量的汇总视图。

**关键模块（`jcy/lib/`）**，被流水线步骤调用，`from jcy.lib.x import`；跨包时 backtest 也 `from jcy.lib.common import`：

| 模块 | 核心内容 |
|------|----------|
| `common.py` | `title_to_date()`、`title_to_filename()`、`record_key()`、`safe_title()`、`load_docs()`、`load_candidates()`、`is_ashare_code()`、路径常量（`JSON_PATH/DOCS_FILE/ADVICE_DIR`） |
| `text.py` | `blocks_to_text()`（飞书 block → 文本）、`parse_json_loose()`（宽松 JSON 解析） |
| `pplx.py` | `PerplexityAPI` 客户端（`chat()`、`sonar_deep_research()`） |
