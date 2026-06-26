# 数据流水线改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 JCY 数据流水线（`prepare_jcy_data.py`）的去重键改为 `date+title` 复合键、三套去重状态收敛为单一真值源（`jcy_insights.json`），并补齐错误处理、日志、测试与零调用迁移脚本。

**Architecture:** 保持"脚本化流水线 + 文件数据层"形态，不引入 DB/编排器。先把纯逻辑（键生成、日期解析、JSON 容错解析、飞书 block 转文本）抽到可单测位置并建测试台；再原地重构 `prepare_jcy_data.py` 的去重/命名/跳过逻辑（含 Step2/3 原子性约束）；写零 API 调用的迁移脚本迁移存量 156 篇数据；最后补健壮性、可观测性、配置去漂移与 git 卫生。

**Tech Stack:** Python ≥3.11（实际 3.12）、pytest（dev 依赖，需新增）、pyyaml、requests、openai SDK。

## Global Constraints

- Python `requires-python = ">=3.11"`（pyproject.toml 已声明，勿降低）。
- **不重跑现有数据**：迁移脚本对存量 `advice/*.md` 与 `jcy_insights.json` **零 API 调用**，只重命名 + 重建索引。
- **不引入** 编排器 / 消息队列 / 数据库 / 配置框架 / 结构化日志框架 / pydantic。校验用手写 `assert` + 清晰报错。
- **不动** 回测引擎 / 策略 / 可视化（另一份 spec 覆盖）。
- 复合键实现固定为：`record_key(date, title) -> f"{date or 'NODATE'}__{safe_title}"`，`safe_title` = 非法字符 `[\\/:*?"<>|]` 替换为 `_` 后截断到 80 字符。
- advice 文件名固定为：`{date or 'NODATE'}__{safe_title}.md`。
- `date` 解析失败时 record 存 `date: null`（不再 `date or title`）。
- Step 2 原子性顺序固定：**先落 advice 文件 → 再更新 record 的 `advice_file` 字段并存盘**。
- 跳过判断：Step 2 = record 有 `advice_file` 且文件存在；Step 3 = record 有 `extracted_at` 字段（Step 3 完成标记）。
- 现有产出风格（emoji + 中文）保留。
- 测试目录 `tests/`（与回测 spec 共用），pytest 驱动，不联网、不读真实 `data/`（用 tmp_path / 合成数据）。

---

## File Structure

- `utils/jcy_common.py`（修改）— 现有共享工具。新增 `record_key`、`safe_title`；改 `title_to_date`（失败返回 None，加注释）、`title_to_filename`（产复合命名）。
- `utils/jcy_text.py`（新建）— 从 `prepare_jcy_data.py` 抽出的纯文本逻辑：`blocks_to_text`、`parse_json_loose`。无 IO、无网络。
- `prepare_jcy_data.py`（修改）— 主流水线。Step 1 改调 `_feishu_get`；Step 2/3 改用单一真值源 + 原子性；引入 logging；`--strict`/`--log-file` 开关；删 `S2_SKIP_MODE` 与 `advice_cache.json` 相关逻辑。
- `scripts/migrate_compound_key.py`（新建）— 零 API 调用的存量迁移（dry-run 默认 / `--apply`）。
- `prompts/step2_advice_system.md`（新建）、`prompts/step3_extract_system.md`（新建）— 抽出的 system prompt。
- `tests/test_jcy_keys.py`、`tests/test_jcy_parse.py`、`tests/test_blocks_to_text.py`、`tests/test_skip_logic.py`、`tests/test_migrate.py`（新建）。
- `pyproject.toml`（修改）— 加 `[project.optional-dependencies] dev = ["pytest"]`。
- `.gitignore`（修改）— 忽略生成数据。
- `CLAUDE.md`（修改）— Step 3 provider 与环境变量去漂移。

---

## Phase A — 纯逻辑抽离 + 测试台

### Task A0: 安装测试依赖

**Files:**
- Modify: `pyproject.toml:7-12`（在 `[build-system]` 前插入 optional-dependencies）

**Interfaces:**
- Produces: 可运行 `pytest`。

- [ ] **Step 1: 修改 pyproject.toml 增加 dev 依赖**

在 `pyproject.toml` 的 `[project]` 段落之后、`[build-system]` 之前插入：

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: 安装**

Run: `pip install -e ".[dev]"`
Expected: 成功安装 pytest，无报错。

- [ ] **Step 3: 验证 pytest 可用**

Run: `pytest --version`
Expected: 打印 `pytest 8.x.x`。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pytest dev dependency"
```

---

### Task A1: `safe_title` 与 `record_key`

**Files:**
- Modify: `utils/jcy_common.py`（在 `title_to_date` 上方新增两个函数）
- Test: `tests/test_jcy_keys.py`

**Interfaces:**
- Produces:
  - `safe_title(title: str, maxlen: int = 80) -> str` — 非法字符替换为 `_`、strip、截断到 maxlen。
  - `record_key(date: str | None, title: str) -> str` — `f"{date or 'NODATE'}__{safe_title(title)}"`。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_jcy_keys.py`：

```python
from utils.jcy_common import safe_title, record_key


def test_safe_title_replaces_illegal_chars():
    assert safe_title('a/b:c*d?"e<f>g|h') == "a_b_c_d__e_f_g_h"


def test_safe_title_truncates_long_title():
    long = "x" * 200
    assert len(safe_title(long)) == 80


def test_safe_title_strips_whitespace():
    assert safe_title("  hello  ") == "hello"


def test_record_key_combines_date_and_title():
    assert record_key("2026-06-26", "Vol.260626 时代主题") == "2026-06-26__Vol.260626 时代主题"


def test_record_key_handles_none_date():
    assert record_key(None, "无日期标题") == "NODATE__无日期标题"


def test_record_key_same_date_different_title_distinct():
    a = record_key("2026-06-26", "上午盘评")
    b = record_key("2026-06-26", "下午盘评")
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jcy_keys.py -v`
Expected: FAIL with `ImportError: cannot import name 'safe_title'`.

- [ ] **Step 3: Write minimal implementation**

在 `utils/jcy_common.py` 顶部 import 区已有 `import re`。在 `title_to_date` 定义之前插入：

```python
def safe_title(title: str, maxlen: int = 80) -> str:
    """标题安全化：替换文件名非法字符、去首尾空白、截断到 maxlen。"""
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", title).strip()
    return cleaned[:maxlen]


def record_key(date: str | None, title: str) -> str:
    """复合去重键：date + 安全化 title。date 缺失时用 NODATE 占位。

    一天可能有多条数据，故不能用单 date 做键。
    """
    return f"{date or 'NODATE'}__{safe_title(title)}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jcy_keys.py -v`
Expected: PASS（6 passed）。

- [ ] **Step 5: Commit**

```bash
git add utils/jcy_common.py tests/test_jcy_keys.py
git commit -m "feat: add safe_title and record_key compound key helpers"
```

---

### Task A2: `title_to_date` 失败返回 None + `title_to_filename` 复合命名

**Files:**
- Modify: `utils/jcy_common.py:15-30`（`title_to_date` 与 `title_to_filename`）
- Test: `tests/test_jcy_parse.py`

**Interfaces:**
- Consumes: `safe_title`、`record_key`（Task A1）。
- Produces:
  - `title_to_date(title: str) -> str | None` — 失败返回 `None`（不变签名，行为已是返回 None，新增注释明确世纪前缀假设）。
  - `title_to_filename(title: str) -> str` — 返回 `{date or 'NODATE'}__{safe_title}.md`。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_jcy_parse.py`：

```python
from utils.jcy_common import title_to_date, title_to_filename


def test_title_to_date_success():
    assert title_to_date("Vol.260626 时代主题") == "2026-06-26"


def test_title_to_date_no_digits_returns_none():
    assert title_to_date("无日期的标题") is None


def test_title_to_filename_with_date():
    assert title_to_filename("Vol.260626 时代主题") == "2026-06-26__Vol.260626 时代主题.md"


def test_title_to_filename_without_date_uses_nodate():
    assert title_to_filename("纯文字标题") == "NODATE__纯文字标题.md"


def test_title_to_filename_sanitizes():
    assert title_to_filename("a/b 260101") == "2026-01-01__a_b 260101.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jcy_parse.py -v`
Expected: FAIL（`test_title_to_filename_with_date` 等断言不符，旧实现产 `2026-06-26.md`）。

- [ ] **Step 3: Write minimal implementation**

替换 `utils/jcy_common.py` 中的 `title_to_date` 与 `title_to_filename`：

```python
def title_to_date(title: str) -> str | None:
    """从标题提取日期：'Vol.260626 今日更新' → '2026-06-26'。

    假设：6 位数字为 YYMMDD，世纪前缀固定 '20'（2000-2099）。
    匹配标题中第一个 6 位连续数字；无匹配返回 None（不再 fallback 成标题）。
    """
    m = re.search(r'(\d{6})', title)
    if m:
        ymd = m.group(1)
        return f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}"
    return None


def title_to_filename(title: str) -> str:
    """生成 advice 文件名：复合命名 '{date or NODATE}__{safe_title}.md'。

    与 record_key 一致，避免一天多条互相覆盖文件。
    """
    date = title_to_date(title)
    return f"{date or 'NODATE'}__{safe_title(title)}.md"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jcy_parse.py tests/test_jcy_keys.py -v`
Expected: PASS（全部）。

- [ ] **Step 5: Commit**

```bash
git add utils/jcy_common.py tests/test_jcy_parse.py
git commit -m "feat: title_to_filename uses compound naming; title_to_date returns None on miss"
```

---

### Task A3: 抽 `blocks_to_text` 与 `parse_json_loose` 到 `utils/jcy_text.py`

**Files:**
- Create: `utils/jcy_text.py`
- Modify: `prepare_jcy_data.py`（删 `_s1_blocks_to_text` 与 `_s3_parse_json` 函数体，改为从 `utils.jcy_text` 导入并保留同名别名）
- Test: `tests/test_blocks_to_text.py`

**Interfaces:**
- Produces:
  - `blocks_to_text(blocks: list[dict]) -> str` — 飞书 docx block 列表 → 纯文本（heading1-6 与 text block 提取 elements 的 text_run content，空行跳过，`\n` 连接）。
  - `parse_json_loose(raw: str) -> dict` — 先 `json.loads`，失败则正则提取首个 `{...}` 再解析，仍失败抛 `ValueError`。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_blocks_to_text.py`：

```python
from utils.jcy_text import blocks_to_text, parse_json_loose


def _text_block(content):
    return {"text": {"elements": [{"text_run": {"content": content}}]}}


def _heading_block(level, content):
    return {f"heading{level}": {"elements": [{"text_run": {"content": content}}]}}


def test_blocks_to_text_extracts_text_and_headings():
    blocks = [_heading_block(1, "标题"), _text_block("正文一"), _text_block("正文二")]
    assert blocks_to_text(blocks) == "标题\n正文一\n正文二"


def test_blocks_to_text_skips_empty():
    blocks = [_text_block("有内容"), _text_block("   "), _text_block("")]
    assert blocks_to_text(blocks) == "有内容"


def test_blocks_to_text_empty_list():
    assert blocks_to_text([]) == ""


def test_parse_json_loose_plain():
    assert parse_json_loose('{"a": 1}') == {"a": 1}


def test_parse_json_loose_with_fence():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_loose_embedded_in_prose():
    assert parse_json_loose('好的，结果是 {"a": 1} 完毕') == {"a": 1}


def test_parse_json_loose_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_json_loose("没有 JSON 的纯文本")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blocks_to_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.jcy_text'`.

- [ ] **Step 3: Write minimal implementation**

创建 `utils/jcy_text.py`：

```python
"""JCY 流水线纯文本逻辑（无 IO/网络，可单测）：飞书 block 转文本、容错 JSON 解析。"""

import json
import re


def blocks_to_text(blocks: list[dict]) -> str:
    """飞书 docx block 列表 → 纯文本。提取 heading1-6 与 text block 的文本，空行跳过。"""
    lines = []
    heading_keys = ("heading1", "heading2", "heading3",
                    "heading4", "heading5", "heading6")
    for block in blocks:
        matched = False
        for key in heading_keys:
            if key in block:
                text = "".join(
                    e.get("text_run", {}).get("content", "")
                    for e in block[key].get("elements", [])
                )
                if text.strip():
                    lines.append(text.strip())
                matched = True
                break
        if not matched and "text" in block:
            text = "".join(
                e.get("text_run", {}).get("content", "")
                for e in block["text"].get("elements", [])
            )
            if text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def parse_json_loose(raw: str) -> dict:
    """容错解析 JSON：先直接解析，失败则提取首个 {...} 块，仍失败抛 ValueError。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            return json.loads(m.group())
        raise ValueError(f"无法解析 JSON 响应: {raw[:200]}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_blocks_to_text.py -v`
Expected: PASS（7 passed）。

- [ ] **Step 5: 主文件改为复用（删重复实现）**

在 `prepare_jcy_data.py` 的 import 区（`from utils.jcy_common import ...` 附近）加入：

```python
from utils.jcy_text import blocks_to_text as _s1_blocks_to_text, parse_json_loose as _s3_parse_json
```

然后**删除** `prepare_jcy_data.py` 中原 `_s1_blocks_to_text(blocks)` 函数定义（约 `def _s1_blocks_to_text` 整段）与 `_s3_parse_json(raw)` 函数定义（约 `def _s3_parse_json` 整段）。其余调用点（`_s1_blocks_to_text(...)`、`_s3_parse_json(...)`）因别名导入而保持不变。

- [ ] **Step 6: 验证主文件仍能导入**

Run: `python -c "import prepare_jcy_data"`
Expected: 无 ImportError、无 NameError。

- [ ] **Step 7: Commit**

```bash
git add utils/jcy_text.py tests/test_blocks_to_text.py prepare_jcy_data.py
git commit -m "refactor: extract blocks_to_text and parse_json_loose to utils.jcy_text with tests"
```

---

## Phase B — 复合键 + 单一真值源 + 原子性

### Task B1: 提取并单测 Step 跳过判断纯函数

**Files:**
- Modify: `prepare_jcy_data.py`（新增两个纯判断函数 + 一个 record 查找函数）
- Test: `tests/test_skip_logic.py`

**Interfaces:**
- Consumes: `record_key`（Task A1）。
- Produces（定义在 `prepare_jcy_data.py`，供 Step 2/3 调用）：
  - `_record_index(articles: list[dict]) -> dict[str, int]` — 以 `record_key(a["date"], a["title"])` 为键的 `{key: index}`。
  - `_step2_done(record: dict) -> bool` — record 有非空 `advice_file` 且该文件存在。
  - `_step3_done(record: dict) -> bool` — record 含非空 `extracted_at`。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_skip_logic.py`：

```python
import os

from prepare_jcy_data import _record_index, _step2_done, _step3_done


def test_record_index_keys_by_compound():
    articles = [
        {"date": "2026-06-26", "title": "上午"},
        {"date": "2026-06-26", "title": "下午"},
    ]
    idx = _record_index(articles)
    assert idx == {"2026-06-26__上午": 0, "2026-06-26__下午": 1}


def test_record_index_none_date():
    idx = _record_index([{"date": None, "title": "无日期"}])
    assert "NODATE__无日期" in idx


def test_step2_done_true_when_file_exists(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("x", encoding="utf-8")
    assert _step2_done({"advice_file": str(f)}) is True


def test_step2_done_false_when_file_missing(tmp_path):
    assert _step2_done({"advice_file": str(tmp_path / "missing.md")}) is False


def test_step2_done_false_when_no_field():
    assert _step2_done({}) is False


def test_step3_done_true_when_extracted_at_present():
    assert _step3_done({"extracted_at": "2026-06-26 10:00:00"}) is True


def test_step3_done_false_when_absent():
    assert _step3_done({"date": "2026-06-26"}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skip_logic.py -v`
Expected: FAIL with `ImportError: cannot import name '_record_index'`.

- [ ] **Step 3: Write minimal implementation**

在 `prepare_jcy_data.py` 的 import 区确保有 `from utils.jcy_common import ... , record_key`（把 `record_key` 加入现有导入）。在 Step 3 区块（`_s3_load_output` 附近）新增：

```python
def _record_index(articles: list[dict]) -> dict:
    """以复合键 record_key(date, title) 建索引 {key: list_index}。"""
    return {
        record_key(a.get("date"), a.get("title", "")): i
        for i, a in enumerate(articles)
    }


def _step2_done(record: dict) -> bool:
    """Step 2 是否已完成：record 有 advice_file 且文件实际存在。"""
    path = record.get("advice_file")
    return bool(path) and os.path.exists(path)


def _step3_done(record: dict) -> bool:
    """Step 3 是否已完成：record 含 extracted_at（提取完成标记）。"""
    return bool(record.get("extracted_at"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skip_logic.py -v`
Expected: PASS（7 passed）。

- [ ] **Step 5: Commit**

```bash
git add prepare_jcy_data.py tests/test_skip_logic.py
git commit -m "feat: compound-key record index and step done-checks"
```

---

### Task B2: 单一真值源 record 加载 / 存盘 / upsert

**Files:**
- Modify: `prepare_jcy_data.py`（重写 `_s3_load_output` / `_s3_save_output`，新增 `_load_articles` / `_save_articles` / `_upsert_record`）
- Test: `tests/test_skip_logic.py`（追加 upsert 测试）

**Interfaces:**
- Consumes: `_record_index`（Task B1）。
- Produces:
  - `_load_articles() -> list[dict]` — 读 `jcy_insights.json` 的 `articles`，文件不存在返回 `[]`。
  - `_save_articles(articles: list[dict]) -> None` — 按 date 倒序写回，结构 `{updated_at,total,articles}`。
  - `_upsert_record(articles: list[dict], record: dict) -> list[dict]` — 按复合键存在则就地更新（merge），否则 append。返回同一 list。

- [ ] **Step 1: Write the failing test**

在 `tests/test_skip_logic.py` 末尾追加：

```python
from prepare_jcy_data import _upsert_record


def test_upsert_appends_new():
    arts = []
    _upsert_record(arts, {"date": "2026-06-26", "title": "新", "x": 1})
    assert len(arts) == 1 and arts[0]["x"] == 1


def test_upsert_merges_existing_same_key():
    arts = [{"date": "2026-06-26", "title": "同", "advice_file": "a.md"}]
    _upsert_record(arts, {"date": "2026-06-26", "title": "同", "extracted_at": "t"})
    assert len(arts) == 1
    assert arts[0]["advice_file"] == "a.md"
    assert arts[0]["extracted_at"] == "t"


def test_upsert_same_date_diff_title_two_records():
    arts = []
    _upsert_record(arts, {"date": "2026-06-26", "title": "上午"})
    _upsert_record(arts, {"date": "2026-06-26", "title": "下午"})
    assert len(arts) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skip_logic.py -v`
Expected: FAIL with `ImportError: cannot import name '_upsert_record'`.

- [ ] **Step 3: Write minimal implementation**

替换 `prepare_jcy_data.py` 中 `_s3_load_output` 与 `_s3_save_output`，并新增 helper。`_s3_load_output` 旧返回 `(articles, index)`，新结构改为分离的 `_load_articles` + `_record_index`。新增/替换为：

```python
def _load_articles() -> list:
    """读取权威清单 articles（jcy_insights.json）。文件不存在返回空列表。"""
    if not os.path.exists(S3_OUTPUT_FILE):
        return []
    with open(S3_OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("articles", [])


def _save_articles(articles: list) -> None:
    """按 date 倒序写回权威清单。"""
    os.makedirs(os.path.dirname(S3_OUTPUT_FILE), exist_ok=True)
    sorted_articles = sorted(articles, key=lambda a: a.get("date") or "", reverse=True)
    with open(S3_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total":      len(sorted_articles),
            "articles":   sorted_articles,
        }, f, ensure_ascii=False, indent=2)


def _upsert_record(articles: list, record: dict) -> list:
    """按复合键 upsert：存在则 merge 更新，否则 append。"""
    key = record_key(record.get("date"), record.get("title", ""))
    index = _record_index(articles)
    if key in index:
        articles[index[key]].update(record)
    else:
        articles.append(record)
    return articles
```

然后**删除**旧的 `_s3_load_output` 与 `_s3_save_output` 定义（B3 会改 `run_step3` 的调用点，此处先保证新函数存在且旧函数不再被引用——若旧函数此刻仍被 `run_step3` 调用，B3 未完成前主文件会引用缺失符号，故 B2、B3 必须连续提交，B2 提交前先完成 B3 的 `run_step3` 改写。**实施顺序：先做 B2 Step 3 + B3 Step 3 的代码改动，再统一跑测试与提交。**）

> 注：为避免中间态断裂，B2 与 B3 的实现代码一起落，分两条 commit 但一次跑通。

- [ ] **Step 4: Run unit tests（B2 部分）**

Run: `pytest tests/test_skip_logic.py -v`
Expected: PASS（含新增 3 个 upsert 测试）。

- [ ] **Step 5: Commit**

```bash
git add prepare_jcy_data.py tests/test_skip_logic.py
git commit -m "feat: single-source-of-truth article load/save/upsert by compound key"
```

---

### Task B3: 改写 Step 2 / Step 3 主循环为单一真值源 + 原子性

**Files:**
- Modify: `prepare_jcy_data.py`（`run_step2`、`run_step3`、删 `S2_SKIP_MODE`/`_s2_*cache*`/`_s2_get_skip_set`）
- Test: `tests/test_skip_logic.py`（已覆盖判断逻辑；本任务行为靠冒烟验证）

**Interfaces:**
- Consumes: `_load_articles`、`_save_articles`、`_upsert_record`、`_record_index`、`_step2_done`、`_step3_done`、`title_to_filename`、`record_key`。
- Produces: `run_step2(docs)`、`run_step3(docs)` 新签名不变，内部改用权威清单。

- [ ] **Step 1: 改写 `run_step2`**

把 `run_step2` 整段替换为以下逻辑（保留连续超时保护、sleep、emoji 输出风格）。关键：**先落文件 → 再 upsert record 存盘**（原子性约束）。

```python
def run_step2(docs):
    """Perplexity 投资建议生成。单一真值源：以 jcy_insights.json 为权威清单。
    原子性：先落 advice 文件，再写 record 的 advice_file 字段。
    连续超时 MAX_CONSECUTIVE_TIMEOUTS 次时终止。"""
    pplx     = PerplexityAPI(S2_API_KEY, S2_GROUP_ID)
    articles = _load_articles()
    index    = _record_index(articles)

    # 跳过：该文档对应 record 已有 advice 文件
    pending = []
    for d in docs:
        title = d.get("文档标题", "")
        key   = record_key(title_to_date(title), title)
        rec   = articles[index[key]] if key in index else {}
        if not _step2_done(rec):
            pending.append(d)
    print(f"📋 Step 2 本次需分析：{len(pending)} 篇（已跳过 {len(docs) - len(pending)} 篇）\n")

    if not pending:
        print("✅ 所有文档均已生成建议，无需重新处理")
        return

    consecutive_timeouts = 0
    new_count = 0
    for i, doc in enumerate(pending, 1):
        title = doc.get("文档标题", "（无标题）")
        link  = doc.get("文档链接", "")
        date  = title_to_date(title)
        print(f"[{i}/{len(pending)}] 分析：{title}")

        try:
            advice, citations = _s2_analyze_doc(pplx, doc)
        except requests.exceptions.Timeout:
            consecutive_timeouts += 1
            print(f"   ⏰ 请求超时（连续第 {consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS} 次）")
            if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                print(f"   ❌ 连续超时 {MAX_CONSECUTIVE_TIMEOUTS} 次，终止 Step 2")
                break
            time.sleep(S2_SLEEP)
            continue

        if advice is None:
            print("   ❌ API 响应无效，跳过本篇\n")
            consecutive_timeouts = 0
            continue

        consecutive_timeouts = 0
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename    = title_to_filename(title)
        # ── 原子性：先落文件 ──
        path = _s2_save_md(filename, _s2_build_md(title, link, analyzed_at, advice, citations))
        # ── 再更新权威清单 ──
        _upsert_record(articles, {
            "date":        date,
            "title":       title,
            "link":        link,
            "advice_file": os.path.abspath(path),
        })
        _save_articles(articles)
        index = _record_index(articles)
        new_count += 1
        print(f"   ✅ 已保存 → {path}\n")
        if i < len(pending):
            time.sleep(S2_SLEEP)

    print(f"\nStep 2 完成：本次新增 {new_count} 篇，文件目录：{ADVICE_DIR}")
```

- [ ] **Step 2: 删除废弃的 cache 逻辑**

删除 `prepare_jcy_data.py` 中以下定义与常量：`_s2_load_cache`、`_s2_save_cache`、`_s2_get_skip_set`、`S2_CACHE_FILE`、`S2_SKIP_MODE`。（`_s2_build_md`、`_s2_save_md`、`_s2_analyze_doc` 保留。）

- [ ] **Step 3: 改写 `run_step3`**

把 `run_step3` 替换为基于权威清单的版本（保留 provider 回退、连续失败保护、sleep）：

```python
def run_step3(docs):
    """LLM 结构化提取，单一真值源。Step 3 跳过 = record 已有 extracted_at。"""
    active_names = [p["name"] for p in S3_PROVIDERS if p["enabled"]]
    print(f"🤖 LLM 顺序：{' → '.join(active_names) or '无（请配置 API Key）'}\n")

    articles = _load_articles()
    new_count       = 0
    consec_failures = 0

    for doc in docs:
        title    = doc.get("文档标题", "（无标题）")
        link     = doc.get("文档链接", "")
        date_str = title_to_date(title)
        filename = title_to_filename(title)
        key      = record_key(date_str, title)

        index = _record_index(articles)
        rec   = articles[index[key]] if key in index else {}
        if _step3_done(rec):
            print(f"[跳过] {title}")
            continue

        advice_text = _s3_load_advice(filename)
        if advice_text is None:
            print(f"[警告] 建议文件不存在：{filename}，仅用原文分析")

        print(f"[{new_count + 1}] 提取：{title} ...")
        try:
            insights, used = _s3_extract_insights(doc, advice_text)
        except (OSError, RuntimeError, ValueError, TimeoutError) as e:
            consec_failures += 1
            print(f"   ❌ 所有 Provider 均失败: {e}")
            if consec_failures >= MAX_CONSECUTIVE_TIMEOUTS:
                print(f"   ❌ 连续失败 {MAX_CONSECUTIVE_TIMEOUTS} 次，终止 Step 3")
                break
            time.sleep(S3_SLEEP)
            continue

        consec_failures = 0
        _upsert_record(articles, {
            "date":         date_str,
            "title":        title,
            "link":         link,
            "advice_file":  os.path.abspath(os.path.join(ADVICE_DIR, filename)),
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm_provider": used,
            **insights,
        })
        _save_articles(articles)
        new_count += 1
        companies = insights.get("companies", [])
        rated     = [c for c in companies if c.get("rating")]
        print(f"   ✅ 已保存（公司:{len(companies)}个，有评级:{len(rated)}个，"
              f"市场:{insights.get('markets', [])}，"
              f"建议:{len(insights.get('key_advice', []))}条）[{used}]")
        time.sleep(S3_SLEEP)

    print(f"\nStep 3 完成：本次新增 {new_count} 篇，输出：{S3_OUTPUT_FILE}")
```

- [ ] **Step 4: 验证导入与全量单测**

Run: `python -c "import prepare_jcy_data" && pytest tests/ -v`
Expected: 导入无 NameError（旧 `_s3_load_output` 等已不再被引用）；全部测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add prepare_jcy_data.py
git commit -m "refactor: step2/3 use single source of truth with atomic file-then-record write"
```

---

## Phase C — 迁移脚本与执行

### Task C1: 迁移脚本（dry-run + apply）

**Files:**
- Create: `scripts/migrate_compound_key.py`
- Test: `tests/test_migrate.py`

**Interfaces:**
- Consumes: `record_key`、`title_to_date`、`title_to_filename`、`safe_title`（utils.jcy_common）。
- Produces:
  - `plan_migration(docs: list[dict], advice_dir: str, articles: list[dict]) -> dict` — 返回 `{"renames": [(old, new)], "missing": [key], "json_rewrites": int}`，纯函数、无副作用。
  - `apply_migration(plan, advice_dir, insights_path, cache_path) -> None` — 执行重命名、重写 JSON、删 cache。
  - CLI：默认 dry-run 打印计划；`--apply` 执行。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_migrate.py`：

```python
import os
from scripts.migrate_compound_key import plan_migration, apply_migration


def test_plan_maps_old_to_new_name(tmp_path):
    advice = tmp_path / "advice"
    advice.mkdir()
    (advice / "2026-06-26.md").write_text("x", encoding="utf-8")
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L"}]
    plan = plan_migration(docs, str(advice), articles=[])
    olds = [os.path.basename(o) for o, _ in plan["renames"]]
    news = [os.path.basename(n) for _, n in plan["renames"]]
    assert "2026-06-26.md" in olds
    assert "2026-06-26__Vol.260626 时代主题.md" in news
    assert plan["missing"] == []


def test_plan_reports_missing_when_no_old_file(tmp_path):
    advice = tmp_path / "advice"
    advice.mkdir()
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L"}]
    plan = plan_migration(docs, str(advice), articles=[])
    assert plan["renames"] == []
    assert "2026-06-26__Vol.260626 时代主题" in plan["missing"]


def test_apply_renames_and_rewrites_json(tmp_path):
    advice = tmp_path / "advice"
    advice.mkdir()
    (advice / "2026-06-26.md").write_text("x", encoding="utf-8")
    insights = tmp_path / "jcy_insights.json"
    import json
    insights.write_text(json.dumps({"articles": [
        {"date": "2026-06-26", "title": "Vol.260626 时代主题",
         "advice_file": str(advice / "2026-06-26.md")}
    ]}, ensure_ascii=False), encoding="utf-8")
    cache = tmp_path / "advice_cache.json"
    cache.write_text("{}", encoding="utf-8")
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L"}]
    plan = plan_migration(docs, str(advice), articles=json.loads(insights.read_text(encoding="utf-8"))["articles"])
    apply_migration(plan, str(advice), str(insights), str(cache))
    assert (advice / "2026-06-26__Vol.260626 时代主题.md").exists()
    assert not (advice / "2026-06-26.md").exists()
    assert not cache.exists()
    new_articles = json.loads(insights.read_text(encoding="utf-8"))["articles"]
    assert new_articles[0]["advice_file"].endswith("2026-06-26__Vol.260626 时代主题.md")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.migrate_compound_key'`.

- [ ] **Step 3: Write minimal implementation**

创建 `scripts/__init__.py`（空文件，使 `scripts` 可导入）与 `scripts/migrate_compound_key.py`：

```python
"""存量数据迁移到复合键命名，零 API 调用。默认 dry-run，--apply 执行。

旧命名 advice/YYYY-MM-DD.md → 新命名 YYYY-MM-DD__<safe_title>.md。
重写 jcy_insights.json 的 advice_file 与 date 字段，删除 advice_cache.json。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jcy_common import title_to_date, title_to_filename, record_key

_BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA   = os.path.join(_BASE, "data", "jcy")
ADVICE  = os.path.join(_DATA, "advice")
INSIGHTS = os.path.join(_DATA, "jcy_insights.json")
CACHE   = os.path.join(_DATA, "advice_cache.json")
DOCS    = os.path.join(_DATA, "jcy_docs.yaml")


def _old_filename(date: str | None) -> str | None:
    """历史命名（每天一条）：YYYY-MM-DD.md。无 date 无旧命名。"""
    return f"{date}.md" if date else None


def plan_migration(docs: list, advice_dir: str, articles: list) -> dict:
    """计算迁移计划，纯函数无副作用。

    Returns {"renames": [(old_path, new_path)], "missing": [key], "json_rewrites": int}
    """
    renames, missing = [], []
    for d in docs:
        title = d.get("文档标题", "")
        date  = title_to_date(title)
        key   = record_key(date, title)
        new_name = title_to_filename(title)
        new_path = os.path.join(advice_dir, new_name)
        old_name = _old_filename(date)
        old_path = os.path.join(advice_dir, old_name) if old_name else None
        if old_path and os.path.exists(old_path):
            if os.path.abspath(old_path) != os.path.abspath(new_path):
                renames.append((old_path, new_path))
        elif os.path.exists(new_path):
            pass  # 已是新命名，无需动作
        else:
            missing.append(key)
    return {"renames": renames, "missing": missing, "json_rewrites": len(articles)}


def apply_migration(plan: dict, advice_dir: str, insights_path: str, cache_path: str) -> None:
    """执行迁移：重命名文件、重写 JSON 的 advice_file/date、删 cache。零 API 调用。"""
    for old_path, new_path in plan["renames"]:
        os.rename(old_path, new_path)

    if os.path.exists(insights_path):
        with open(insights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for a in data.get("articles", []):
            title = a.get("title", "")
            date  = title_to_date(title)
            a["date"] = date  # 解析失败→None，不再塞 title
            a["advice_file"] = os.path.abspath(
                os.path.join(advice_dir, title_to_filename(title)))
        data["total"] = len(data.get("articles", []))
        with open(insights_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    if os.path.exists(cache_path):
        os.remove(cache_path)


def _load_docs_yaml(path: str) -> list:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def main():
    parser = argparse.ArgumentParser(description="复合键存量数据迁移（零 API 调用）")
    parser.add_argument("--apply", action="store_true", help="执行迁移（默认仅 dry-run）")
    args = parser.parse_args()

    docs = _load_docs_yaml(DOCS)
    articles = []
    if os.path.exists(INSIGHTS):
        with open(INSIGHTS, "r", encoding="utf-8") as f:
            articles = json.load(f).get("articles", [])

    plan = plan_migration(docs, ADVICE, articles)
    print(f"重命名 advice 文件：{len(plan['renames'])} 个")
    print(f"重写 JSON 记录：{plan['json_rewrites']} 条")
    print(f"缺产物（需人工补调）：{len(plan['missing'])} 条")
    for k in plan["missing"]:
        print(f"  ⚠️ 缺：{k}")

    if not args.apply:
        print("\n[dry-run] 未写入任何文件。确认无误后加 --apply 执行。")
        return

    if plan["missing"]:
        print(f"\n⚠️ 有 {len(plan['missing'])} 条缺产物，本脚本不自动补调 API。")
        print("   仍将迁移可迁移部分；缺失项需后续手动跑流水线补齐。")
    apply_migration(plan, ADVICE, INSIGHTS, CACHE)
    print("\n✅ 迁移完成。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_migrate.py -v`
Expected: PASS（3 passed）。

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/migrate_compound_key.py tests/test_migrate.py
git commit -m "feat: zero-API-call migration script for compound key naming"
```

---

### Task C2: 在真实数据上 dry-run 并执行迁移

**Files:**
- 无代码改动；操作 `data/jcy/`（生成数据）。

**Interfaces:**
- Consumes: `scripts/migrate_compound_key.py`（Task C1）。

- [ ] **Step 1: 备份当前生成数据（git 留可回滚点）**

Run: `git add -A data/jcy && git commit -m "chore: snapshot jcy data before compound-key migration"`
Expected: 提交成功（即便部分文件已被 .gitignore 也无妨，此为安全网）。

- [ ] **Step 2: dry-run 查看计划**

Run: `python scripts/migrate_compound_key.py`
Expected: 打印"重命名 156 个左右、重写 156 条、缺产物 0 条"。**若缺产物 > 0**，停下，列出缺哪些，人工判断是否可接受再继续。

- [ ] **Step 3: 执行迁移**

Run: `python scripts/migrate_compound_key.py --apply`
Expected: 打印 `✅ 迁移完成`。

- [ ] **Step 4: 人工核对**

Run: `ls data/jcy/advice/ | head -3 && echo "---" && python -c "import json;d=json.load(open('data/jcy/jcy_insights.json',encoding='utf-8'));print(d['articles'][0]['advice_file']); print('cache exists:', __import__('os').path.exists('data/jcy/advice_cache.json'))"`
Expected: advice 文件名为 `YYYY-MM-DD__标题.md`；`advice_file` 指向新命名；`cache exists: False`。

- [ ] **Step 5: 冒烟测试 —— 重跑流水线确认全跳过、无重复**

Run: `python -c "from utils.jcy_common import load_docs; import prepare_jcy_data as p; docs=load_docs(); p.run_step2(docs); p.run_step3(docs)"`
Expected: Step 2 / Step 3 均打印"全部跳过 / 无需重新处理"，`jcy_insights.json` 的 `total` 不增长（无重复 append）。

- [ ] **Step 6: Commit 迁移后数据**

```bash
git add -A data/jcy
git commit -m "chore: migrate jcy data to compound-key naming"
```

---

## Phase D — 健壮性 + 可观测性

### Task D1: 统一飞书 GET 包装 `_feishu_get`

**Files:**
- Modify: `prepare_jcy_data.py`（新增 `_feishu_get`；`_s1_get_app_token`/`_s1_get_all_records`/`_s1_get_docx_content`/`_s1_get_wiki_content` 改调它）
- Test: `tests/test_feishu_get.py`

**Interfaces:**
- Produces: `_feishu_get(url: str, token: str, params: dict | None = None, retries: int = 2) -> dict` — 返回飞书 API 的 `data` 字段；HTTP 非 200 或 `code != 0` 抛 `RuntimeError`；5xx/超时重试 `retries` 次。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_feishu_get.py`（用 monkeypatch 伪造 requests，不联网）：

```python
import pytest
import prepare_jcy_data as p


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_feishu_get_success(monkeypatch):
    monkeypatch.setattr(p.requests, "get",
                        lambda *a, **k: _Resp(200, {"code": 0, "data": {"ok": 1}}))
    assert _call() == {"ok": 1}


def _call():
    return p._feishu_get("http://x", "tok", {})


def test_feishu_get_api_error_raises(monkeypatch):
    monkeypatch.setattr(p.requests, "get",
                        lambda *a, **k: _Resp(200, {"code": 99, "msg": "bad"}))
    with pytest.raises(RuntimeError, match="bad"):
        _call()


def test_feishu_get_http_error_raises(monkeypatch):
    monkeypatch.setattr(p.requests, "get",
                        lambda *a, **k: _Resp(500, {}))
    with pytest.raises(RuntimeError):
        p._feishu_get("http://x", "tok", {}, retries=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_feishu_get.py -v`
Expected: FAIL with `AttributeError: module 'prepare_jcy_data' has no attribute '_feishu_get'`.

- [ ] **Step 3: Write minimal implementation**

在 `prepare_jcy_data.py` 的 Step 1 区块顶部（`_s1_hdr` 附近）新增：

```python
def _feishu_get(url: str, token: str, params: dict | None = None,
                retries: int = 2) -> dict:
    """统一飞书 GET：查 HTTP 状态、code != 0 抛错、5xx/超时有限重试。返回 data 字段。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=_s1_hdr(token), params=params, timeout=30)
            if r.status_code >= 500:
                last_err = RuntimeError(f"飞书 HTTP {r.status_code}")
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"飞书 HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            if data.get("code") != 0:
                raise RuntimeError(f"飞书 API 错误：{data.get('msg')}（code={data.get('code')}）")
            return data.get("data", {})
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"飞书请求失败（已重试 {retries} 次）：{last_err}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_feishu_get.py -v`
Expected: PASS（3 passed）。

- [ ] **Step 5: 改 Step 1 各处调用 `_feishu_get`**

将 `_s1_get_app_token`、`_s1_get_all_records`、`_s1_get_docx_content`、`_s1_get_wiki_content` 中 `requests.get(...) → data = r.json(); if data["code"] != 0: ...` 的模式替换为调用 `_feishu_get(url, token, params)`，并从返回的 `data` 直接取字段。例如 `_s1_get_app_token`：

```python
def _s1_get_app_token(wiki_token, token):
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token}"
    data = _feishu_get(url, token)
    obj_token = data["node"]["obj_token"]
    print(f"✅ bitable app_token：{obj_token}")
    return obj_token
```

`_s1_get_all_records`、`_s1_get_docx_content` 的分页循环：把每页的 `data = r.json(); if code != 0: break` 改为 `data = _feishu_get(url, token, params)`，循环用 `data.get("items", [])` 与 `data.get("has_more")`。

- [ ] **Step 6: 验证导入 + 全量单测**

Run: `python -c "import prepare_jcy_data" && pytest tests/ -v`
Expected: 导入无误；全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add prepare_jcy_data.py tests/test_feishu_get.py
git commit -m "feat: unified _feishu_get with retry and clear errors; step1 uses it"
```

---

### Task D2: `run_step1` 失败醒目化 + `--strict` + logging + `--log-file`

**Files:**
- Modify: `prepare_jcy_data.py`（`main()` 加 argparse；`logging` 初始化；`run_step1` 失败警告；最终摘要）

**Interfaces:**
- Consumes: 现有 `run_step1/2/3`。
- Produces: `main()` 接受 `--strict`、`--log-file`；模块级 `log = logging.getLogger("jcy")`。

- [ ] **Step 1: 引入 logging 与 argparse**

在 `prepare_jcy_data.py` import 区加 `import argparse, logging`。在常量区加：

```python
log = logging.getLogger("jcy")


def _setup_logging(log_file: str | None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
```

- [ ] **Step 2: 改写 `main()` 接受开关并醒目化 Step 1 失败**

把 `main()` 改为：

```python
def main():
    parser = argparse.ArgumentParser(description="JCY 数据准备流水线（Step 1-3）")
    parser.add_argument("--strict", action="store_true",
                        help="Step 1 采集失败时硬中止（默认继续用旧缓存）")
    parser.add_argument("--log-file", type=str, default="",
                        help="日志同时写入该文件")
    args = parser.parse_args()
    _setup_logging(args.log_file or None)

    print("\n" + "=" * 60)
    print("  JCY 数据准备流水线（Step 1 → 2 → 3）")
    print("=" * 60)

    print("\n" + "─" * 60 + "\n  Step 1: 飞书数据采集\n" + "─" * 60)
    step1_ok = run_step1()
    if not step1_ok:
        log.warning("⚠️ Step 1 采集失败，后续步骤将使用已有旧缓存（数据可能不是最新）")
        if args.strict:
            log.error("--strict 已启用，终止流水线")
            return

    try:
        docs = load_docs()
        print(f"\n📂 读取到 {len(docs)} 篇文档")
    except FileNotFoundError:
        print("❌ 未找到文档缓存（jcy_docs.yaml），无法执行 Step 2/3")
        return

    print("\n" + "─" * 60 + "\n  Step 2: Perplexity 投资建议生成\n" + "─" * 60)
    run_step2(docs)
    print("\n" + "─" * 60 + "\n  Step 3: LLM 结构化提取\n" + "─" * 60)
    run_step3(docs)

    print("\n" + "=" * 60)
    print(f"  全部流程完成（Step 1 采集：{'成功' if step1_ok else '失败-用旧缓存'}）")
    print("=" * 60)
```

- [ ] **Step 3: 验证 CLI 可运行（--help 不联网）**

Run: `python prepare_jcy_data.py --help`
Expected: 打印含 `--strict` 与 `--log-file` 的帮助文本，无异常。

- [ ] **Step 4: 验证全量单测仍通过**

Run: `pytest tests/ -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add prepare_jcy_data.py
git commit -m "feat: --strict and --log-file flags, prominent step1 failure warning, logging"
```

---

## Phase E — 配置 / 文档去漂移

### Task E1: prompt 抽到 `prompts/`

**Files:**
- Create: `prompts/step2_advice_system.md`、`prompts/step3_extract_system.md`
- Modify: `prepare_jcy_data.py`（`S2_SYSTEM_PROMPT`/`S3_SYSTEM_PROMPT` 改为从文件读取）

**Interfaces:**
- Produces: `_read_prompt(name: str) -> str` — 读 `prompts/<name>`。

- [ ] **Step 1: 把现有 prompt 文本移到文件**

创建 `prompts/step2_advice_system.md`，内容 = 现 `S2_SYSTEM_PROMPT` 字符串的完整正文（`prepare_jcy_data.py:51-73` 三引号内文本，逐字复制）。
创建 `prompts/step3_extract_system.md`，内容 = 现 `S3_SYSTEM_PROMPT` 完整正文（`prepare_jcy_data.py:96-138` 三引号内文本，逐字复制）。

- [ ] **Step 2: 改代码从文件读取**

把 `S2_SYSTEM_PROMPT = """..."""` 与 `S3_SYSTEM_PROMPT = """..."""` 两段替换为：

```python
_PROMPTS_DIR = os.path.join(_BASE_DIR, "prompts")


def _read_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


S2_SYSTEM_PROMPT = _read_prompt("step2_advice_system.md")
S3_SYSTEM_PROMPT = _read_prompt("step3_extract_system.md")
```

- [ ] **Step 3: 验证导入且 prompt 内容非空**

Run: `python -c "import prepare_jcy_data as p; assert len(p.S2_SYSTEM_PROMPT) > 100 and len(p.S3_SYSTEM_PROMPT) > 100; print('prompts loaded')"`
Expected: 打印 `prompts loaded`。

- [ ] **Step 4: 全量单测**

Run: `pytest tests/ -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add prompts/ prepare_jcy_data.py
git commit -m "refactor: externalize step2/step3 system prompts to prompts/"
```

---

### Task E2: CLAUDE.md provider 与环境变量去漂移

**Files:**
- Modify: `CLAUDE.md`（Step 3 描述段 + 环境变量表）

**Interfaces:** 无代码。

- [ ] **Step 1: 更新 Step 3 描述**

在 CLAUDE.md 中找到"Step 3 — 结构化信息提取（Azure GPT）"，把"Azure OpenAI gpt-5"的措辞改为反映实际：

```markdown
**Step 3 — 结构化信息提取（LLM，按 provider 回退）：**
- 默认启用 **DashScope**（`S3_PROVIDERS` 中 Azure / Coze 默认注释关闭，可按需开启）
- 原文 + 建议文档 → `response_format=json_object` → 结构化 JSON
- 增量机制：按 `(date, title)` 复合键去重；record 含 `extracted_at` 即跳过
```

- [ ] **Step 2: 更新环境变量表**

在环境变量清单中：新增 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`DASHSCOPE_MODEL`、`TOKEN_FILE`（飞书 token 文件路径）；标注 `AZURE_OPENAI_*` / `COZE_*` 为"可选，默认关闭"。

```markdown
- `TOKEN_FILE` — 飞书 Bearer Token 文件路径
- `DASHSCOPE_API_KEY` — DashScope API 密钥（Step 3 默认 provider）
- `DASHSCOPE_BASE_URL` — DashScope 端点（默认 compatible-mode/v1）
- `DASHSCOPE_MODEL` — DashScope 模型名
- `AZURE_OPENAI_*` / `COZE_*` — 可选 provider，默认在 S3_PROVIDERS 中注释关闭
```

- [ ] **Step 3: 同步更新去重机制描述**

把 CLAUDE.md 中 Step 2/Step 3"增量机制：按 `date` 字段去重 / 扫描 advice 目录"等旧描述，统一改为："以 `jcy_insights.json` 为单一真值源；Step 2 跳过依据 record 的 `advice_file`，Step 3 依据 `extracted_at`；`advice_cache.json` 已废弃。"

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: align CLAUDE.md with actual providers, env vars, dedup mechanism"
```

---

## Phase F — git 卫生

### Task F1: 生成数据加入 .gitignore

**Files:**
- Modify: `.gitignore`

**Interfaces:** 无代码。

- [ ] **Step 1: 追加忽略规则**

在 `.gitignore` 末尾追加（注意现有文件末尾无换行，先补换行）：

```
data/jcy/advice/
data/jcy/jcy_insights.json
data/jcy/jcy_table.json
data/jcy/jcy_docs.yaml
.playwright-mcp/
```

- [ ] **Step 2: 从 git 索引移除已追踪的生成数据（保留磁盘文件）**

Run: `git rm -r --cached data/jcy/advice data/jcy/jcy_insights.json data/jcy/jcy_table.json data/jcy/jcy_docs.yaml 2>/dev/null; git status --short | head`
Expected: 这些路径显示为 deleted（仅从索引移除，磁盘仍在）。

> 注：若团队需要少量样本入库，可在 `data/jcy/sample/` 另存几条并不忽略。本计划默认全部移出。

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore generated jcy data and playwright artifacts"
```

---

## Self-Review

**1. Spec coverage**

- §3.1 复合键 → A1/A2/B1/B2/B3 ✓
- §3.2 date fallback 成 title → A2（返回 None）+ C1 apply（修正存量）✓
- §3.3 三套状态打架 → B3（删 cache 逻辑）+ C1（删 cache 文件）✓
- §3.4 title_to_date 世纪假设 → A2（注释 + 返回 None）✓
- §3.5 Step1 零容错 → D1 `_feishu_get` ✓
- §3.6 run_step1 静默失败 → D2 `--strict` + 警告 ✓
- §3.7 CLAUDE.md 脱节 → E2 ✓
- §3.8 prompt 硬编码 → E1 ✓
- §3.9 零测试 → A1/A2/A3/B1/B2/C1/D1 各自带测试 ✓
- §3.10 print 无日志 → D2 logging ✓
- §4.2 单一真值源 + 原子性 → B2/B3（先文件后 record）✓
- §4.4 迁移脚本 → C1/C2 ✓
- §5 .gitignore 生成数据 → F1 ✓
- §5 schema 最小校验 → 已用 `_step2_done/_step3_done/_upsert` 的字段约定隐式覆盖；未额外加 assert 校验函数。**补充说明**：spec §5 称"最小校验（必填字段存在性）"为 nice-to-have，本计划通过复合键 helper 对 `date/title` 的处理覆盖了实际风险点，不再单列校验任务（YAGNI）。

**2. Placeholder scan**：无 TBD/TODO；所有代码步骤含完整代码；测试含真实断言。✓

**3. Type consistency**：`record_key(date, title)`、`title_to_filename(title)`、`_step2_done(record)`、`_step3_done(record)`、`_upsert_record(articles, record)`、`plan_migration(docs, advice_dir, articles)`、`apply_migration(plan, advice_dir, insights_path, cache_path)`、`_feishu_get(url, token, params, retries)` 在定义与调用处签名一致。✓

**注意（实施者必读）**：B2 与 B3 必须**连续实现、一次跑通**——B2 删除旧 `_s3_load_output/_s3_save_output` 后，B3 才把 `run_step3` 调用点改到新函数。建议先落 B2+B3 全部代码改动，再运行 `python -c "import prepare_jcy_data"` 验证无缺失符号，然后分两条 commit。
