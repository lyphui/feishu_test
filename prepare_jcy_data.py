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
import yaml
import requests
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI

from utils.pplx import PerplexityAPI
from utils.jcy_common import title_to_date, title_to_filename, load_docs, ADVICE_DIR

load_dotenv(find_dotenv())

# ════════════════════════════════════════════════════════════════
#  全局常量
# ════════════════════════════════════════════════════════════════

MAX_CONSECUTIVE_TIMEOUTS = 3   # 连续超时达到此次数时终止当前步骤

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_BASE_DIR, "data", "jcy")

# ─── Step 1 ──────────────────────────────────────────────────────────────────
S1_TOKEN_FILE   = os.path.join(_BASE_DIR, "authorize", "feishu_key", "feishu_token.txt")
S1_TABLE_FILE   = os.path.join(_DATA_DIR, "jcy_table.json")
S1_DOCS_FILE    = os.path.join(_DATA_DIR, "jcy_docs.yaml")
S1_WIKI_TOKEN   = os.getenv("JCY_WIKI_TOKEN")
S1_APP_TABLE_ID = os.getenv("JCY_APP_TABLE_ID")
S1_VIEW_ID      = os.getenv("JCY_VIEW_ID")

# ─── Step 2 ──────────────────────────────────────────────────────────────────
S2_API_KEY    = os.getenv("PPLX_API_KEY")
S2_GROUP_ID   = os.getenv("PPLX_GROUP_ID")
S2_CACHE_FILE = os.path.join(_DATA_DIR, "advice_cache.json")
S2_SLEEP      = 2
S2_SKIP_MODE  = "files"   # "files": 扫描 advice/ 目录；"cache": 按 JSON 缓存跳过

S2_SYSTEM_PROMPT = """你是一位善于用简单语言解释投资的顾问，同时会结合最新的互联网资讯进行验证和补充。
你的读者是完全不了解股票和金融的普通人，请用通俗易懂的语言，避免专业术语，必要时用括号解释。

请严格按以下格式输出：

## 今日核心观点
（2-4句话概括：当前市场在担心什么？看好什么？整体氛围如何？）

## 文章提到的股票/行业详解
（对文章中每一个具体股票或行业，分别说明以下三点：
  - 【是什么】这家公司/行业做什么的，用一句话让外行听懂
  - 【为什么被提到】作者为什么看好或提及它，背后的逻辑是什么
  - 【值不值得关注】结合你搜索到的最新信息，这家公司/行业目前的真实情况如何，有没有实质性的投资价值或风险）

## 投资小白行动建议
（普通人看完这篇文章后，可以做什么、应该注意什么？
  请给出具体可操作的建议，例如：可以关注哪个方向、目前不适合追高、应该等待什么信号再行动等）

## 风险提示
（这篇文章有哪些判断可能是错的？投资有哪些风险是小白容易忽视的？）

## 一句话总结
（用最简单的一句话，帮小白记住今天最重要的一个结论）"""

# ─── Step 3 ──────────────────────────────────────────────────────────────────
S3_AZURE_ENDPOINT   = "https://llm-east-us2-test.openai.azure.com/"
S3_AZURE_DEPLOYMENT = "gpt-5"
S3_AZURE_API_KEY    = os.getenv("AZURE_OPENAI_KEY", "")
S3_AZURE_API_VER    = "2024-12-01-preview"
S3_COZE_API_KEY     = os.getenv("COZE_API_KEY", "")
S3_COZE_URL         = os.getenv("COZE_URL", "https://99p6x2gyv9.coze.site/run")
S3_COZE_MODEL       = os.getenv("COZE_MODEL", "doubao-pro")
S3_OUTPUT_FILE      = os.path.join(_DATA_DIR, "jcy_insights.json")
S3_SLEEP            = 2

# LLM providers in fallback order; disable by leaving the API key env var empty
S3_PROVIDERS = [
    # {"name": "Azure OpenAI", "type": "azure", "enabled": bool(S3_AZURE_API_KEY)},
    {"name": "Coze",         "type": "coze",  "enabled": bool(S3_COZE_API_KEY)},
]

S3_SYSTEM_PROMPT = """你是一位专业的股票市场分析助手。请从提供的股市分析文章和投资建议中，提取关键的结构化信息。

严格按照以下 JSON 格式输出，不要输出任何其他内容：

{
  "companies": [
    {
      "name": "公司中文名",
      "code": "股票代码（如600000、000001、0700.HK、NVDA, 若不确定则为null）",
      "exchange": "具体交易所，从以下选择：上交所/深交所/北交所/港交所/纳斯达克/纽交所/其他，若不确定则为null",
      "rating": "投资评级，从以下选择：买入/增持/持有/减持/卖出/回避，若文章无明确倾向则为null",
      "rating_reason": "评级依据，一句话说明（不超过30字，若rating为null则省略）"
    }
  ],
  "markets": ["A股", "港股", "美股", "等"],
  "tendency": "整体投资倾向，例如：看涨科技/看涨周期/防御/观望/多元配置等（一句话）",
  "key_advice": [
    "建议1（简洁，不超过50字）",
    "建议2",
    "建议3"
  ]
}

说明：
- companies：文章中明确提到的股票或公司，仅当文章涉及股市分析时填写，否则为空数组 []
- exchange 判断规则（A股代码）：
    * 600xxx / 601xxx / 603xxx / 605xxx / 688xxx → 上交所（科创板在上交所）
    * 000xxx / 001xxx / 002xxx / 003xxx / 300xxx / 301xxx → 深交所（创业板在深交所）
    * 430xxx / 830xxx / 83xxxx / 87xxxx / 88xxxx / 899xxx → 北交所
    * 末尾含 .HK 或 .hk → 港交所
    * 无数字代码的美股（如 NVDA、AAPL）→ 纳斯达克 或 纽交所（根据常识判断）
- rating 投资评级含义：
    * 买入（Strong Buy）：强烈推荐，预期涨幅显著高于市场
    * 增持（Overweight/Add）：看好，建议适度加仓
    * 持有（Hold/Neutral）：中性，维持现有仓位
    * 减持（Underweight/Reduce）：谨慎，建议降低仓位
    * 卖出（Sell）：明确看空，建议清仓
    * 回避（Avoid）：风险较高，不建议介入
- markets：文章主要讨论的投资市场，从以下选择：A股、港股、美股、期货、基金、其他
- tendency：用一句话总结整体投资倾向，要具体（如"看好国产半导体，规避美股AI"比"看涨"更好）
- key_advice：精炼的3-5条核心建议，每条简洁明了

只输出 JSON，不要 markdown 代码块，不要任何解释。"""


# ════════════════════════════════════════════════════════════════
#  Step 1 — 飞书数据采集
# ════════════════════════════════════════════════════════════════

def _s1_hdr(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


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
    r = requests.get(url, headers=_s1_hdr(token))
    data = r.json()
    if data["code"] != 0:
        print(f"❌ 获取wiki节点失败：{data['msg']}（code={data['code']}）")
        return None
    obj_token = data["data"]["node"]["obj_token"]
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

        r = requests.get(url, headers=_s1_hdr(token), params=params)
        data = r.json()
        if data["code"] != 0:
            print(f"❌ 读取表格失败：{data['msg']}（code={data['code']}）")
            break

        items = data["data"].get("items", [])
        all_records.extend(items)
        print(f"   第{page}页：{len(items)} 条")

        if not data["data"].get("has_more"):
            break
        page_token = data["data"].get("page_token", "")
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


def _s1_blocks_to_text(blocks):
    lines = []
    heading_keys = ("heading1", "heading2", "heading3", "heading4", "heading5", "heading6")
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


def _s1_get_docx_content(doc_token, token):
    all_blocks = []
    page_token = ""
    while True:
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks"
        params = {"page_size": 200}
        if page_token:
            params["page_token"] = page_token
        r = requests.get(url, headers=_s1_hdr(token), params=params)
        data = r.json()
        if data["code"] != 0:
            return {"error": data["msg"], "code": data["code"]}
        all_blocks.extend(data["data"].get("items", []))
        if not data["data"].get("has_more"):
            break
        page_token = data["data"].get("page_token", "")
        time.sleep(0.1)
    return {"text": _s1_blocks_to_text(all_blocks)}


def _s1_get_wiki_content(wiki_token_val, token):
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token_val}"
    r = requests.get(url, headers=_s1_hdr(token))
    data = r.json()
    if data["code"] != 0:
        return {"error": data["msg"], "code": data["code"]}
    node      = data["data"]["node"]
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

def _s2_load_cache():
    if not os.path.exists(S2_CACHE_FILE):
        return {}
    with open(S2_CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _s2_save_cache(cache):
    os.makedirs(os.path.dirname(S2_CACHE_FILE), exist_ok=True)
    with open(S2_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _s2_get_skip_set():
    if S2_SKIP_MODE == "files":
        if not os.path.isdir(ADVICE_DIR):
            return set()
        return {f for f in os.listdir(ADVICE_DIR) if f.endswith(".md")}
    else:
        return set(_s2_load_cache().keys())


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
    """Perplexity 投资建议生成。连续超时 MAX_CONSECUTIVE_TIMEOUTS 次时终止。"""
    pplx     = PerplexityAPI(S2_API_KEY, S2_GROUP_ID)
    cache    = _s2_load_cache()
    skip_set = _s2_get_skip_set()

    mode_label = "目录文件扫描" if S2_SKIP_MODE == "files" else "JSON缓存"
    print(f"📂 跳过模式：{mode_label}，已处理：{len(skip_set)} 个")

    if S2_SKIP_MODE == "files":
        new_docs = [d for d in docs if title_to_filename(d.get("文档标题", "")) not in skip_set]
    else:
        new_docs = [d for d in docs if d.get("文档链接") not in skip_set]
    print(f"📋 本次需分析：{len(new_docs)} 篇\n")

    if not new_docs:
        print("✅ 所有文档均已分析，无需重新处理")
        return

    consecutive_timeouts = 0
    new_count = 0

    for i, doc in enumerate(new_docs, 1):
        title = doc.get("文档标题", "（无标题）")
        link  = doc.get("文档链接", "")
        print(f"[{i}/{len(new_docs)}] 分析：{title}")

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
            print(f"   ❌ API 响应无效，跳过本篇\n")
            consecutive_timeouts = 0
            continue

        consecutive_timeouts = 0
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename    = title_to_filename(title)
        path        = _s2_save_md(filename, _s2_build_md(title, link, analyzed_at, advice, citations))
        cache[link] = filename
        _s2_save_cache(cache)
        new_count += 1
        print(f"   ✅ 已保存 → {path}\n")

        if i < len(new_docs):
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


def _s3_load_output():
    if not os.path.exists(S3_OUTPUT_FILE):
        return [], {}
    with open(S3_OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    index    = {a["date"]: i for i, a in enumerate(articles) if "date" in a}
    return articles, index


def _s3_save_output(articles):
    os.makedirs(os.path.dirname(S3_OUTPUT_FILE), exist_ok=True)
    sorted_articles = sorted(articles, key=lambda a: a.get("date", ""), reverse=True)
    with open(S3_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total":      len(sorted_articles),
            "articles":   sorted_articles,
        }, f, ensure_ascii=False, indent=2)


def _s3_parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            return json.loads(m.group())
        raise ValueError(f"无法解析 JSON 响应: {raw[:200]}")


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
        except Exception as e:
            print(f"   ⚠️  {provider['name']} 失败: {type(e).__name__}: {e}")
            last_err = e
    raise last_err


def run_step3(docs):
    """LLM 结构化提取，按 S3_PROVIDERS 顺序自动回退。"""
    active_names = [p["name"] for p in S3_PROVIDERS if p["enabled"]]
    print(f"🤖 LLM 顺序：{' → '.join(active_names) or '无（请配置 API Key）'}\n")

    articles, date_index = _s3_load_output()
    print(f"已提取：{len(articles)} 篇，本次跳过已有记录\n")

    new_count       = 0
    consec_failures = 0

    for doc in docs:
        title    = doc.get("文档标题", "（无标题）")
        link     = doc.get("文档链接", "")
        date_str = title_to_date(title)
        filename = title_to_filename(title)

        if date_str and date_str in date_index:
            print(f"[跳过] {title}")
            continue

        advice_text = _s3_load_advice(filename)
        if advice_text is None:
            print(f"[警告] 建议文件不存在：{filename}，仅用原文分析")

        print(f"[{new_count + 1}] 提取：{title} ...")

        try:
            insights, used = _s3_extract_insights(doc, advice_text)
        except Exception as e:
            consec_failures += 1
            print(f"   ❌ 所有 Provider 均失败: {e}")
            if consec_failures >= MAX_CONSECUTIVE_TIMEOUTS:
                print(f"   ❌ 连续失败 {MAX_CONSECUTIVE_TIMEOUTS} 次，终止 Step 3")
                break
            time.sleep(S3_SLEEP)
            continue

        consec_failures = 0
        record = {
            "date":         date_str or title,
            "title":        title,
            "link":         link,
            "advice_file":  os.path.abspath(os.path.join(ADVICE_DIR, filename)),
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm_provider": used,
            **insights,
        }
        articles.append(record)
        if date_str:
            date_index[date_str] = len(articles) - 1
        _s3_save_output(articles)
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
    print("\n" + "=" * 60)
    print("  JCY 数据准备流水线（Step 1 → 2 → 3）")
    print("=" * 60)

    # ── Step 1 ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 1: 飞书数据采集")
    print("─" * 60)
    run_step1()   # 失败时仍尝试后续步骤（利用已有缓存）

    # ── 加载文档（Steps 2 & 3 共用）────────────────────────────
    try:
        docs = load_docs()
        print(f"\n📂 读取到 {len(docs)} 篇文档")
    except FileNotFoundError:
        print("❌ 未找到文档缓存（jcy_docs.yaml），无法执行 Step 2/3")
        return

    # ── Step 2 ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 2: Perplexity 投资建议生成")
    print("─" * 60)
    run_step2(docs)

    # ── Step 3 ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 3: Azure GPT 结构化提取")
    print("─" * 60)
    run_step3(docs)

    print("\n" + "=" * 60)
    print("  全部流程完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
