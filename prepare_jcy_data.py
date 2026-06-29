"""
JCY 数据准备一体化流水线（Step 1-3）

Step 1  飞书 API 数据采集
Step 2  Perplexity AI 投资建议生成（sonar-reasoning-pro）
Step 3  Azure GPT 结构化信息提取

超时保护：Step 2/3 中 API 连续超时 MAX_CONSECUTIVE_TIMEOUTS 次时终止当前步骤，
         不将无效响应写入 data/ 目录。
"""
import os
import re
import json
import time
import argparse
import logging
import yaml
import requests
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI, OpenAI

from utils.pplx import PerplexityAPI
from utils.jcy_common import title_to_date, title_to_filename, load_docs, record_key, ADVICE_DIR
from utils.jcy_text import blocks_to_text as _s1_blocks_to_text, parse_json_loose as _s3_parse_json

load_dotenv(find_dotenv())

# ════════════════════════════════════════════════════════════════
#  全局常量
# ════════════════════════════════════════════════════════════════

MAX_CONSECUTIVE_TIMEOUTS = 3   # 连续超时达到此次数时终止当前步骤

log = logging.getLogger("jcy")


def _setup_logging(log_file: str | None):
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)


_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_BASE_DIR, "data", "jcy")

# ─── Step 1 ──────────────────────────────────────────────────────────────────
S1_TOKEN_FILE   = os.getenv("TOKEN_FILE")
S1_TABLE_FILE   = os.path.join(_DATA_DIR, "jcy_table.json")
S1_DOCS_FILE    = os.path.join(_DATA_DIR, "jcy_docs.yaml")
S1_WIKI_TOKEN   = os.getenv("JCY_WIKI_TOKEN")
S1_APP_TABLE_ID = os.getenv("JCY_APP_TABLE_ID")
S1_VIEW_ID      = os.getenv("JCY_VIEW_ID")

# ─── Step 2 ──────────────────────────────────────────────────────────────────
S2_API_KEY    = os.getenv("PPLX_API_KEY")
S2_GROUP_ID   = os.getenv("PPLX_GROUP_ID")
S2_SLEEP      = 2

_PROMPTS_DIR = os.path.join(_BASE_DIR, "prompts")


def _read_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


S2_SYSTEM_PROMPT = _read_prompt("step2_advice_system.md")

# ─── Step 3 ──────────────────────────────────────────────────────────────────
S3_AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
S3_AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")
S3_AZURE_API_KEY    = os.getenv("AZURE_OPENAI_KEY", "")
S3_AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
S3_COZE_API_KEY     = os.getenv("COZE_API_KEY", "")
S3_COZE_URL         = os.getenv("COZE_URL", "")
S3_COZE_MODEL       = os.getenv("COZE_MODEL", "doubao-pro")
S3_DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY", "")
S3_DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
S3_DASHSCOPE_MODEL    = os.getenv("DASHSCOPE_MODEL", "deepseek-v4-pro")
S3_OUTPUT_FILE      = os.path.join(_DATA_DIR, "jcy_insights.json")
S3_SLEEP            = 2

# LLM providers in fallback order; disable by leaving the API key env var empty
S3_PROVIDERS = [
    {"name": "DashScope", "type": "dashscope", "enabled": bool(S3_DASHSCOPE_API_KEY)},
    # {"name": "Azure OpenAI", "type": "azure", "enabled": bool(S3_AZURE_API_KEY)},
    # {"name": "Coze",         "type": "coze",  "enabled": bool(S3_COZE_API_KEY)},
]

S3_SYSTEM_PROMPT = _read_prompt("step3_extract_system.md")


# ════════════════════════════════════════════════════════════════
#  Step 1 — 飞书数据采集
# ════════════════════════════════════════════════════════════════

def _s1_hdr(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


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


def _s1_read_token():
    with open(S1_TOKEN_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def _s1_load_existing():
    """返回 {链接: 文档条目} 字典"""
    if not os.path.exists(S1_DOCS_FILE):
        print("📂 未找到旧数据文件，将全量读取")
        return {}
    with open(S1_DOCS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    existing = {item["文档链接"]: item for item in data if item.get("文档链接")}
    print(f"📂 读取到旧数据：{len(existing)} 条已缓存文档")
    return existing


def _s1_save(records, doc_list, record_title_map):
    os.makedirs(_DATA_DIR, exist_ok=True)
    table_output = {
        "更新时间":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "表格记录总数":   len(records),
        "表格原始数据":   records,
    }
    with open(S1_TABLE_FILE, "w", encoding="utf-8") as f:
        json.dump(table_output, f, ensure_ascii=False, indent=2)

    yaml_list = []
    for item in doc_list:
        rid   = item.get("record_id", "")
        title = item.get("文档标题") or record_title_map.get(rid, "")
        text  = (item.get("文档内容正文")
                 or (item.get("文档数据") or {}).get("content", {}).get("text", ""))
        yaml_list.append({
            "record_id":    rid,
            "文档标题":     title,
            "文档链接":     item.get("文档链接", ""),
            "文档内容正文": text,
        })
    with open(S1_DOCS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(yaml_list, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"💾 表格 → {S1_TABLE_FILE}")
    print(f"💾 文档 → {S1_DOCS_FILE}（共 {len(yaml_list)} 条）")


def _s1_get_app_token(wiki_token, token):
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token}"
    try:
        data = _feishu_get(url, token)
    except RuntimeError as e:
        print(f"❌ 获取wiki节点失败：{e}")
        return None
    obj_token = data["node"]["obj_token"]
    print(f"✅ bitable app_token：{obj_token}")
    return obj_token


def _s1_get_all_records(app_token, table_id, view_id, token):
    all_records = []
    page_token  = ""
    page        = 1
    print(f"\n📊 读取多维表格（{table_id}）...")

    while True:
        url = (f"https://open.feishu.cn/open-apis/bitable/v1/apps/"
               f"{app_token}/tables/{table_id}/records")
        params = {"page_size": 100}
        if view_id:
            params["view_id"] = view_id
        if page_token:
            params["page_token"] = page_token

        try:
            data = _feishu_get(url, token, params)
        except RuntimeError as e:
            print(f"❌ 读取表格失败：{e}")
            break

        items = data.get("items", [])
        all_records.extend(items)
        print(f"   第{page}页：{len(items)} 条")

        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")
        page += 1
        time.sleep(0.2)

    print(f"✅ 共 {len(all_records)} 条记录")
    return all_records


def _s1_extract_links(records):
    pattern = re.compile(
        r'https://[a-zA-Z0-9\-]+\.feishu\.cn/'
        r'(?:docx|wiki|base|sheets|drive/file|mindnotes|minutes|board)'
        r'/[^\s\"\'\]\[<>]+'
    )
    seen  = set()
    items = []
    for record in records:
        row_id = record.get("record_id", "未知")
        fields = record.get("fields", {})
        for field_name, field_value in fields.items():
            raw = json.dumps(field_value, ensure_ascii=False)
            for url in pattern.findall(raw):
                url = url.rstrip('",')
                if url not in seen:
                    seen.add(url)
                    items.append({"record_id": row_id, "字段名": field_name, "文档链接": url})
    print(f"✅ 提取到 {len(items)} 个不重复文档链接")
    return items


def _s1_parse_doc_type(url):
    patterns = [
        (r'feishu\.cn/docx/([a-zA-Z0-9]+)', 'docx'),
        (r'feishu\.cn/wiki/([a-zA-Z0-9]+)', 'wiki'),
        (r'feishu\.cn/sheets/([a-zA-Z0-9]+)', 'sheets'),
        (r'feishu\.cn/base/([a-zA-Z0-9]+)', 'bitable'),
        (r'feishu\.cn/drive/file/([a-zA-Z0-9]+)', 'file'),
    ]
    for pat, doc_type in patterns:
        m = re.search(pat, url)
        if m:
            return doc_type, m.group(1)
    return 'unknown', None


def _s1_get_docx_content(doc_token, token):
    all_blocks = []
    page_token = ""
    while True:
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks"
        params = {"page_size": 200}
        if page_token:
            params["page_token"] = page_token
        try:
            data = _feishu_get(url, token, params)
        except RuntimeError as e:
            return {"error": str(e)}
        all_blocks.extend(data.get("items", []))
        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")
        time.sleep(0.1)
    return {"text": _s1_blocks_to_text(all_blocks)}


def _s1_get_wiki_content(wiki_token_val, token):
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token_val}"
    try:
        data = _feishu_get(url, token)
    except RuntimeError as e:
        return {"error": str(e)}
    node      = data["node"]
    obj_type  = node.get("obj_type", "")
    obj_token = node.get("obj_token", "")
    if obj_type in ("doc", "docx"):
        return _s1_get_docx_content(obj_token, token)
    return {"note": f"obj_type={obj_type}，暂不支持读取正文"}


def _s1_fetch_doc(url, token):
    doc_type, doc_token = _s1_parse_doc_type(url)
    if doc_type == 'docx':
        return {"doc_type": "docx", "doc_token": doc_token,
                "content": _s1_get_docx_content(doc_token, token)}
    elif doc_type == 'wiki':
        return {"doc_type": "wiki", "doc_token": doc_token,
                "content": _s1_get_wiki_content(doc_token, token)}
    elif doc_type == 'sheets':
        return {"doc_type": "sheets", "doc_token": doc_token,
                "content": None, "note": "sheets类型暂不支持读取正文"}
    elif doc_type == 'bitable':
        return {"doc_type": "bitable", "doc_token": doc_token,
                "content": None, "note": "bitable类型暂不支持读取正文"}
    else:
        return {"doc_type": "unknown", "content": None, "note": "无法识别的文档类型"}


def run_step1():
    """飞书数据采集。失败时返回 False，但后续步骤仍会尝试使用已有缓存。"""
    try:
        token = _s1_read_token()
    except FileNotFoundError:
        print(f"❌ 未找到 token 文件：{S1_TOKEN_FILE}")
        return False
    print(f"✅ token：{token[:10]}...")

    existing_cache = _s1_load_existing()

    app_token = _s1_get_app_token(S1_WIKI_TOKEN, token)
    if not app_token:
        return False

    records = _s1_get_all_records(app_token, S1_APP_TABLE_ID, S1_VIEW_ID, token)
    if not records:
        print("❌ 未读取到任何记录")
        return False

    link_items = _s1_extract_links(records)
    new_items  = [item for item in link_items if item["文档链接"] not in existing_cache]
    skip_count = len(link_items) - len(new_items)
    print(f"\n📋 增量分析：已缓存（跳过）{skip_count} 个，新增（需读取）{len(new_items)} 个")

    new_results = []
    if new_items:
        print(f"\n📖 开始读取 {len(new_items)} 个新文档...")
        for i, item in enumerate(new_items, 1):
            url = item["文档链接"]
            print(f"   [{i}/{len(new_items)}] {url[:70]}...")
            doc_data = _s1_fetch_doc(url, token)
            new_results.append({
                "record_id": item["record_id"],
                "字段名":    item["字段名"],
                "文档链接":  url,
                "读取时间":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "文档数据":  doc_data,
            })
            time.sleep(1.0)
    else:
        print("✅ 所有文档均已缓存，无需重新读取")

    merged = dict(existing_cache)
    for item in new_results:
        merged[item["文档链接"]] = item
    final_list = [merged[item["文档链接"]] for item in link_items if item["文档链接"] in merged]

    record_title_map = {}
    for record in records:
        rid = record.get("record_id", "")
        content_field = record.get("fields", {}).get("内容", {})
        if isinstance(content_field, dict):
            record_title_map[rid] = content_field.get("text", "")
        else:
            record_title_map[rid] = str(content_field) if content_field else ""

    _s1_save(records, final_list, record_title_map)
    print(f"\n表格记录：{len(records)} 条，文档总数：{len(final_list)} 个"
          f"（新增 {len(new_results)} 个，跳过 {skip_count} 个）")
    return True


# ════════════════════════════════════════════════════════════════
#  Step 2 — Perplexity 投资建议生成
# ════════════════════════════════════════════════════════════════

def _s2_build_md(title, link, analyzed_at, advice, citations):
    lines = [
        f"# {title}", "",
        f"> **原文链接：** [{link}]({link})  ",
        f"> **分析时间：** {analyzed_at}",
        "", "---", "", advice.strip(),
    ]
    if citations:
        lines += ["", "---", "", "## 引用来源", ""]
        for idx, url in enumerate(citations, 1):
            lines.append(f"{idx}. {url}")
    return "\n".join(lines) + "\n"


def _s2_save_md(filename, content):
    os.makedirs(ADVICE_DIR, exist_ok=True)
    path = os.path.join(ADVICE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _s2_analyze_doc(pplx, doc):
    """调用 sonar-reasoning-pro 分析文档。

    Returns (advice, citations) on success, (None, []) on non-timeout failure.
    Raises requests.exceptions.Timeout on timeout.
    """
    content = doc.get("文档内容正文", "").strip()
    if not content:
        return None, []

    title = doc.get("文档标题", "")
    user_prompt = (
        f"以下是一篇股市分析文章，标题是「{title}」：\n\n---\n{content}\n---\n\n"
        "请根据上面的内容，结合你搜索到的最新资讯，为投资小白生成详细的投资分析报告。"
    )
    # pplx.chat() 会 re-raise requests.exceptions.Timeout
    result = pplx.chat(
        model="sonar-reasoning-pro",
        messages=[
            {"role": "system", "content": S2_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=30000,
        temperature=0.3,
    )
    if not result:
        return None, []
    choices = result.get("choices", [])
    if not choices:
        return None, []
    raw = choices[0]["message"]["content"]
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].strip()
    return raw, result.get("citations", [])


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


# ════════════════════════════════════════════════════════════════
#  Step 3 — Azure GPT 结构化提取
# ════════════════════════════════════════════════════════════════

def _s3_load_advice(filename):
    path = os.path.join(ADVICE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _record_index(articles: list) -> dict:
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


def _s3_coze_extract_content(data):
    """Extract text content from Coze response, trying common response shapes."""
    # OpenAI-compatible
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    # Coze bot/workflow formats
    for key in ("output", "answer", "content", "text", "result"):
        if key in data and isinstance(data[key], str):
            return data[key].strip()
    if "data" in data and isinstance(data["data"], dict):
        inner = data["data"]
        for key in ("output", "answer", "content", "text"):
            if key in inner and isinstance(inner[key], str):
                return inner[key].strip()
    # Last resort: dump the whole response so we can inspect it
    raise ValueError(f"无法从 Coze 响应中提取内容，完整响应：{json.dumps(data, ensure_ascii=False)[:500]}")


def _s3_call_provider(provider, user_prompt):
    """Call one LLM provider. Returns dict or raises on any error."""
    if provider["type"] == "dashscope":
        client = OpenAI(
            api_key=S3_DASHSCOPE_API_KEY,
            base_url=S3_DASHSCOPE_BASE_URL,
        )
        response = client.chat.completions.create(
            model=S3_DASHSCOPE_MODEL,
            messages=[
                {"role": "system", "content": S3_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=16000,
            response_format={"type": "json_object"},
        )
        return _s3_parse_json(response.choices[0].message.content.strip())

    if provider["type"] == "azure":
        client = AzureOpenAI(
            api_version=S3_AZURE_API_VER,
            azure_endpoint=S3_AZURE_ENDPOINT,
            api_key=S3_AZURE_API_KEY,
        )
        response = client.chat.completions.create(
            model=S3_AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": S3_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_completion_tokens=16000,
            response_format={"type": "json_object"},
        )
        return _s3_parse_json(response.choices[0].message.content.strip())

    if provider["type"] == "coze":
        r = requests.post(
            S3_COZE_URL,
            headers={"Authorization": f"Bearer {S3_COZE_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": S3_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "model": S3_COZE_MODEL,
                "temperature": 0.3,
                "max_tokens": 16000,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        raw = _s3_coze_extract_content(data)
        return _s3_parse_json(raw)

    raise ValueError(f"未知 provider 类型: {provider['type']}")


def _s3_extract_insights(doc, advice_text):
    """Try each enabled LLM provider in order.

    Returns (insights_dict, provider_name) or raises if all fail.
    """
    title   = doc.get("文档标题", "")
    content = doc.get("文档内容正文", "").strip()
    user_prompt = (
        f"【原文标题】{title}\n\n"
        f"【原文内容】\n{content}\n\n"
        f"【AI生成的投资建议】\n{advice_text or '（暂无建议文档）'}"
    )
    active = [p for p in S3_PROVIDERS if p["enabled"]]
    if not active:
        raise RuntimeError("没有可用的 LLM Provider（请检查 API Key 环境变量）")
    last_err = None
    for provider in active:
        try:
            return _s3_call_provider(provider, user_prompt), provider["name"]
        except (OSError, RuntimeError, ValueError, TimeoutError) as e:
            print(f"   ⚠️  {provider['name']} 失败: {type(e).__name__}: {e}")
            last_err = e
    raise last_err


def run_step3(docs):
    """LLM 结构化提取，按 S3_PROVIDERS 顺序自动回退。"""
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


# ════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════

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


if __name__ == "__main__":
    main()
