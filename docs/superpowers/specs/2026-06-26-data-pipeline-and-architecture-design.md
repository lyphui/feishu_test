# 数据流水线诊断与架构改造设计

日期：2026-06-26
范围：数据流水线（`prepare_jcy_data.py`）+ 整体架构走向
**不含**：回测引擎 / 策略层 / 可视化 —— 这些已由
[2026-06-26-backtest-engine-refactor-design.md](2026-06-26-backtest-engine-refactor-design.md) 覆盖，本 spec 不重复。

---

## 1. 背景与目标

回测引擎的不足已由独立 spec 系统覆盖并分阶段实施中。该 spec 明确把
数据采集流水线（`prepare_jcy_data.py`）列为非目标。本 spec 补上这块空白，
聚焦两个层面：

1. **数据流水线的具体缺陷**——增量去重、错误处理、配置漂移、可测性。
2. **整体架构走向**——这套"采集→AI→回测"流水线值不值得动、动到什么程度。

**诊断口径（已与用户确认）**：所有问题统一按"对结果/可维护性的实际扰动
程度"排序，**正确性/健壮性与架构可维护性并重**，不预设维度偏好。

**核心约束**：
- **不重跑现有数据**。改去重键与文件命名的同时，提供迁移脚本，以
  **最小化 pplx / llm 调用**的方式迁移存量数据（纯重命名 + 重建索引）。
- **不推倒架构**。保持"脚本化流水线 + 文件数据层"形态，不引入
  编排器（Airflow/Prefect）、数据库、或服务化。投入集中在
  *契约显式化、状态单一化、关键纯逻辑可测*。

---

## 2. 关键设计决策（已与用户确认）

| 决策点 | 选定方案 |
|---|---|
| 去重键 | `date` 单键改为 **`date + title` 复合键**。一天可能多条数据，单 `date` 键会互相覆盖。 |
| 文件命名 | `advice/` 文件名从 `YYYY-MM-DD.md` 改为 **`YYYY-MM-DD__<安全化title>.md`**，与复合键一致，避免一天多条互相覆盖文件。 |
| 一天多条的来源 | **面向未来的防御性设计**。历史数据每天仅一条，**不存在**"单文件名塞了多条被覆盖"的损坏情况——迁移脚本只需干净的重命名 + 重建索引。 |
| 去重机制 | **一次到位收敛成单一真值源**：以 `jcy_insights.json` 为权威清单，废弃 `advice_cache.json`，Step 2/3 的跳过判断都基于权威清单。不分阶段。 |
| 迁移调用成本 | 存量 `advice/*.md`（pplx 产物）与 `jcy_insights.json`（llm 产物）**一律不重新调 API**，只重命名 + 重写索引。仅当某 `(date,title)` 完全找不到对应产物时才补调；迁移脚本先 dry-run 报告。 |
| 架构走向 | **不上 DB / 编排器 / 服务化**。补契约（轻量 schema 校验）+ 状态单一化 + 纯逻辑可测。生成数据移出 git 追踪。 |

---

## 3. 数据流水线缺陷清单

按对结果/可维护性的实际扰动程度排序。

### 高 —— 静默产出坏数据 / 去重脆弱

1. **去重键 `date` 单键，一天多条互相覆盖**
   （`prepare_jcy_data.py:578` `_s3_load_output` 的 `date_index`；
   `:745` record 的 `date` 字段）。
   同一天的多篇文章共用同一 `date` 键 → JSON 索引互相覆盖，
   `advice/` 文件名（`title_to_filename` 当前产 `YYYY-MM-DD.md`）也互相覆盖。
   **这是数据丢失，不只是去重失效。**
   修复：复合键 `(date, title)`；文件名带 title。

2. **`date` fallback 成完整 `title` 污染键空间**
   （`prepare_jcy_data.py:745` `"date": date_str or title`）。
   `title_to_date` 解析失败时 `date` 写成整条标题，与正常记录不在同一
   命名空间，**永不判重，每次运行重复 append**。
   改复合键后该问题一并消解（键里 title 始终存在；`date` 缺失时显式标记
   `date=None` 而非塞 title）。

3. **三套去重状态互相打架**
   - Step 2 默认扫 `advice/` 目录文件名（`S2_SKIP_MODE="files"`）。
   - Step 3 用 `jcy_insights.json` 的 `date` 字段。
   - `advice_cache.json` 存"链接→文件名"映射，默认不用于跳过。
   三处可能不一致（文件被删但 cache 仍在等）。
   修复：见 §4.2 单一真值源。

4. **`title_to_date` 假设 `20xx` 世纪 + 贪婪匹配**
   （`utils/jcy_common.py:20` `f"20{ymd[:2]}"`；`\d{6}` 匹配标题里
   *第一个* 6 位数字）。硬编码世纪前缀；标题含其它 6 位编号会误判。
   本轮：复合键弱化了对完美日期解析的依赖；额外加注释说明假设，
   并在解析失败时返回 `None`（不再 fallback 成 title）。

### 高 —— 错误处理掩盖问题

5. **Step 1 网络请求零容错**
   （`_s1_get_app_token` / `_s1_get_all_records` / `_s1_get_docx_content` 等）。
   直接 `r.json()["code"]`，不查 HTTP 状态、不 catch `requests` 异常或
   `JSONDecodeError`、无重试。飞书返回 5xx / HTML 错误页时抛
   `KeyError`/`JSONDecodeError`。对比 Step 2/3 有连续超时保护，Step 1 裸奔。
   修复：抽统一的 `_feishu_get(url, ...)` 包装——查 HTTP 状态、解析失败
   给清晰错误、对 5xx/超时做有限重试。

6. **`run_step1` 失败仅返回 `False`，主流程静默继续**
   （`prepare_jcy_data.py:782`）。"用旧缓存继续"的意图导致采集失败被降级
   成一条 print，用户易忽略，拿旧数据跑了 Step 2/3 仍显示"成功"。
   修复：失败时打印**醒目警告 + 摘要**（"⚠️ Step 1 采集失败，后续步骤将
   使用 N 天前的旧缓存"），并在最终摘要里复述。是否硬中止由 `--strict`
   开关控制（默认继续，保留现有行为）。

### 中 —— 配置 / 文档与现实漂移

7. **CLAUDE.md 与代码脱节**
   - 文档称 Step 3 用 Azure GPT，实际 `S3_PROVIDERS` 默认只启用
     **DashScope**，Azure/Coze 被注释（`prepare_jcy_data.py:90-94`）。
   - 环境变量表缺 `DASHSCOPE_API_KEY`、`TOKEN_FILE`；列出的
     `AZURE_OPENAI_*` 当前不生效。
   修复：更新 CLAUDE.md 的 Step 3 描述与环境变量表，使其反映实际
   provider 回退顺序与所需 env。

8. **超长 prompt + 运行参数硬编码在主文件**
   两段共 ~90 行 system prompt 嵌在 `.py` 顶部
   （`prepare_jcy_data.py:51-138`）；sleep 间隔、`max_tokens`、provider 列表
   皆模块级常量。改 prompt（此类 AI 流水线最高频迭代动作）= 改源码。
   修复：prompt 抽到 `prompts/`（独立 `.txt`/`.md`），运行参数集中到一个
   config 区块或 dataclass。**YAGNI**：不引入配置框架，纯文件 + 常量集中。

### 中 —— 可测性 / 可观测性

9. **零测试**。仓库无 `tests/`（回测 spec 将新建该目录，本 spec 复用）。
   纯逻辑（`title_to_date`、`_s1_blocks_to_text`、`_s3_parse_json`、
   `load_candidates`）与 IO 缠绕，无法单测。
   修复：见 §4.3，先抽纯函数再补单测。

10. **全靠 `print`，无日志分级**。批量跑无法区分 INFO/WARN/ERROR，
    不落盘；连续失败终止后，已成功部分与失败原因无结构化记录。
    修复：**轻量**引入 `logging`（控制台 + 可选文件 handler），保留现有
    中文输出风格。不引入结构化日志框架。

---

## 4. 改造方案细节

### 4.1 复合键 + 文件命名

- **键**：内部统一用 `record_key(date, title)`，实现为
  `f"{date or 'NODATE'}__{title}"`（title 安全化后）。`_s3_load_output`
  的索引改为 `{record_key: i}`。
- **文件名**：`title_to_filename` 改产 `{date}__{safe_title}.md`
  （`date` 缺失时用 `NODATE__{safe_title}.md`）。安全化沿用现有
  `re.sub(r'[\\/:*?"<>|]', '_', ...)`，并截断过长 title（如 80 字符）
  避免文件名超长。
- **record 字段**：`date` 解析失败时存 `null` + 保留 `title`，不再
  `date or title`。

### 4.2 单一真值源（一次到位）

`jcy_insights.json` 的 `articles` 成为权威清单，每条 record 显式带：
- `date`、`title`、`link`
- `advice_file`（advice md 的路径，产物存在性的依据）
- `insights`（companies/markets/tendency/key_advice；未提取则缺失）

跳过判断：
- **Step 2 跳过** = 该 `(date,title)` 的 record 存在 `advice_file` 且文件实际存在。
- **Step 3 跳过** = 该 record 已含 `insights` 字段。

`advice_cache.json` **废弃**（信息被 record 字段覆盖）。`S2_SKIP_MODE`
两套分支删除，只保留权威清单逻辑。

> 注：Step 2 产生 advice 时需要先有 record 占位。若 Step 1 尚未为某文档建
> record，Step 2 先创建 `{date,title,link,advice_file}` 的占位 record，
> Step 3 再补 `insights`。这样三步共用同一清单。

### 4.3 纯逻辑抽离 + 测试

把以下纯函数从 IO 中剥离到可单测位置（`utils/jcy_common.py` 或新
`utils/jcy_text.py`）：
- `title_to_date` / `title_to_filename` / `record_key`
- `_s1_blocks_to_text`（飞书 block → 文本）
- `_s3_parse_json`（容错 JSON 解析）

`tests/`（与回测 spec 共用目录）新增：
- `test_jcy_keys.py`：`record_key` 复合键、一天多条不冲突、`date=None`
  的命名；`title_to_filename` 长 title 截断与非法字符。
- `test_jcy_parse.py`：`title_to_date` 解析成功/失败（失败返回 `None`
  不返回 title）；`_s3_parse_json` 裸 JSON / 带 ```json 围栏 / 嵌正文。
- `test_blocks_to_text.py`：heading/text block 提取，空 block 跳过。

### 4.4 迁移脚本 `scripts/migrate_compound_key.py`

**目标**：把存量数据从旧命名/单键迁移到复合键，**零 API 调用**。

输入真值：`jcy_docs.yaml`（每篇的 `(title, link)`）+ `title_to_date`。

流程：
1. **dry-run（默认）**：
   - 遍历 docs，对每篇算新 `record_key` 与新文件名。
   - 旧 advice 文件名（`YYYY-MM-DD.md`）→ 新文件名映射。
   - 报告：能纯重命名迁移的 N 条；旧 JSON 里能重建索引的 M 条；
     **缺产物（需补调）的 K 条**——预期 K=0（历史每天一条）。
   - 不写任何文件，打印计划。
2. **`--apply`**：
   - 重命名 `advice/*.md` 为新命名（旧文件保留备份或 git 可回滚）。
   - 重写 `jcy_insights.json`：键改复合键，`advice_file` 指向新路径，
     `date or title` 修正为 `date=null` + title。
   - 删除（或归档）`advice_cache.json`。
   - 全程不调用 pplx / Azure / DashScope。
3. 若 dry-run 报告 K>0，列出具体缺哪些产物，由用户决定是否人工补调，
   迁移脚本本身**不自动补调**。

### 4.5 错误处理与日志

- `_feishu_get(url, params, token, retries=2)`：统一飞书 GET，查 HTTP
  状态、`code != 0` 给结构化错误、5xx/超时有限重试。Step 1 各处改调它。
- `logging`：模块级 logger，控制台 handler 默认 INFO；`--log-file`
  可选写文件。现有 emoji/中文风格保留（改成 `log.info(...)`）。
- 最终摘要复述各步成功/跳过/失败计数与是否用了旧缓存。

---

## 5. 架构走向结论（不改造，仅记录方向与触发条件）

这套架构**本质健康**：采集/AI/回测三段分层清晰 + strategies/utils 复用。
架构债集中在*契约缺失*与*状态分散*，不是分层错误。故：

- **保持脚本化流水线 + 文件数据层**，不引入编排器/DB/服务化。
- **够用的契约化**（本 spec 已含）：三步可独立调用的基础已在（各 `run_stepN`
  函数），加轻量 schema 校验把"隐式文件约定"变显式。
  - 范围控制：本轮 schema 校验**仅做 record 结构的最小校验**
    （必填字段存在性），不引入 pydantic 重型依赖；用手写 dataclass 或
    `assert` + 清晰报错。
- **生成数据移出 git**：`data/jcy/advice/` 与生成的 JSON 加入 `.gitignore`，
  git 只留代码与少量样本。（独立小改动，可随本 spec 或单独提交。）

**观察项（不进本计划，记录触发条件）**：
- `load_candidates` 全量线性扫 JSON：记录数到 ~数千条再考虑索引/SQLite。
- 文件即数据库：同上量级触发时再评估。
- 回测入口脚本 config 解析重复：由回测 spec 的 Phase 4（`utils/runner.py`）
  覆盖，本 spec 不重复，仅引用边界。

---

## 6. 实施阶段（可独立验证）

- **Phase A（纯逻辑 + 测试台）**：抽 `record_key`/`title_to_*`/
  `_s1_blocks_to_text`/`_s3_parse_json` 到可测位置 + 单测。无行为变更。
- **Phase B（复合键 + 命名 + 单一真值源）**：§4.1 + §4.2 一次到位。
  改 `prepare_jcy_data.py` 的去重/命名/跳过逻辑，废弃 `advice_cache.json`。
  **代码改完后立即写 §4.4 迁移脚本并跑 dry-run**，确认存量可零调用迁移。
- **Phase C（迁移执行）**：跑 `migrate_compound_key.py --apply`，git 留可
  回滚点。迁移后跑一次 `prepare_jcy_data.py` 冒烟，确认无重复 append、
  无文件覆盖。
- **Phase D（健壮性 + 可观测性）**：§4.5 `_feishu_get` 重试、`run_step1`
  醒目失败警告 + `--strict`、`logging` 接入。
- **Phase E（配置/文档去漂移）**：§3.7 更新 CLAUDE.md；§3.8 prompt 抽到
  `prompts/`、运行参数集中。
- **Phase F（git 卫生）**：生成数据加 `.gitignore`。

每个 Phase 跑全量单测；Phase B/C 额外人工核对迁移后 JSON 与 advice 目录。

---

## 7. 非目标（YAGNI）

- 不引入编排器（Airflow/Prefect）、消息队列、数据库。
- 不引入配置框架 / 结构化日志框架 / pydantic（手写最小校验即可）。
- 不重写架构分层；不改三步串行的基本形态（只让其可独立调用 + 加契约）。
- 不动回测引擎 / 策略 / 可视化（归属另一份 spec）。
- 迁移脚本不自动补调 API（dry-run 报告缺口，由用户决定）。
- 不优化 `load_candidates` 全量扫描（量级未到，列为观察项）。
