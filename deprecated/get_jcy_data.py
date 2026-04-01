"""
飞书多维表格 + 文档内容读取器
功能：
1. 读取多维表格所有记录
2. 提取每行的飞书文档链接
3. 读取每个文档的原始JSON内容（块结构）
4. 增量更新：已缓存的链接跳过，只读取新增链接
"""
import requests
import json
import re
import time
import os
import yaml
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ======================== 配置项 ========================
TOKEN_FILE   = os.path.join(os.path.dirname(__file__), "authorize", "feishu_key", "feishu_token.txt")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "data", "jcy")   # 输出子目录
TABLE_FILE   = os.path.join(OUTPUT_DIR, "jcy_table.json")                # 表格数据
DOCS_FILE    = os.path.join(OUTPUT_DIR, "jcy_docs.yaml")                 # 文档内容

WIKI_TOKEN   = os.getenv("JCY_WIKI_TOKEN")
APP_TABLE_ID = os.getenv("JCY_APP_TABLE_ID")
VIEW_ID      = os.getenv("JCY_VIEW_ID")
# =======================================================


# ─── 工具函数 ────────────────────────────────────────────────────────────────

def read_token():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def hdr(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def load_existing_data():
    """读取已有YAML文件，返回 {链接: 文档条目} 字典"""
    if not os.path.exists(DOCS_FILE):
        print("📂 未找到旧数据文件，将全量读取")
        return {}
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    existing = {item["文档链接"]: item for item in data if item.get("文档链接")}
    print(f"📂 读取到旧数据：{len(existing)} 条已缓存文档")
    return existing

def save_data(records, doc_list, record_title_map):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 表格写入 JSON ──────────────────────────────────────
    table_output = {
        "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "表格记录总数": len(records),
        "表格原始数据": records,
    }
    with open(TABLE_FILE, "w", encoding="utf-8") as f:
        json.dump(table_output, f, ensure_ascii=False, indent=2)

    # ── 文档内容写入 YAML ──────────────────────────────────
    yaml_list = []
    for item in doc_list:
        rid   = item.get("record_id", "")
        title = item.get("文档标题") or record_title_map.get(rid, "")
        text  = (item.get("文档内容正文")
                 or (item.get("文档数据") or {}).get("content", {}).get("text", ""))
        yaml_list.append({
            "record_id": rid,
            "文档标题":   title,
            "文档链接":   item.get("文档链接", ""),
            "文档内容正文": text,
        })
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(yaml_list, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"💾 表格已保存到 {TABLE_FILE}")
    print(f"💾 文档已保存到 {DOCS_FILE}（共 {len(yaml_list)} 条）")


# ─── Step 1: wiki → bitable app_token ───────────────────────────────────────

def get_bitable_token_from_wiki(wiki_token, token):
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token}"
    r = requests.get(url, headers=hdr(token))
    data = r.json()
    if data["code"] != 0:
        print(f"❌ 获取wiki节点失败：{data['msg']}（code={data['code']}）")
        return None
    obj_token = data["data"]["node"]["obj_token"]
    print(f"✅ bitable app_token：{obj_token}")
    return obj_token


# ─── Step 2: 读取多维表格所有记录 ────────────────────────────────────────────

def get_all_records(app_token, table_id, view_id, token):
    all_records = []
    page_token = ""
    page = 1
    print(f"\n📊 读取多维表格（{table_id}）...")

    while True:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        params = {"page_size": 100}
        if view_id:
            params["view_id"] = view_id
        if page_token:
            params["page_token"] = page_token

        r = requests.get(url, headers=hdr(token), params=params)
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


# ─── Step 3: 从记录中提取文档链接 ────────────────────────────────────────────

def extract_links_from_records(records):
    pattern = re.compile(
        r'https://[a-zA-Z0-9\-]+\.feishu\.cn/'
        r'(?:docx|wiki|base|sheets|drive/file|mindnotes|minutes|board)'
        r'/[^\s\"\'\]\[<>]+'
    )
    seen_links = set()
    link_items = []

    for record in records:
        row_id = record.get("record_id", "未知")
        fields = record.get("fields", {})
        for field_name, field_value in fields.items():
            raw = json.dumps(field_value, ensure_ascii=False)
            for url in pattern.findall(raw):
                url = url.rstrip('",')
                if url not in seen_links:
                    seen_links.add(url)
                    link_items.append({
                        "record_id": row_id,
                        "字段名": field_name,
                        "文档链接": url,
                    })

    print(f"✅ 提取到 {len(link_items)} 个不重复文档链接")
    return link_items


# ─── Step 4: 读取文档内容 ────────────────────────────────────────────────────

def parse_doc_type(url):
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



def blocks_to_text(blocks):
    """将 blocks 列表拼接成纯文本字符串，跳过图片等非文本块"""
    lines = []
    heading_keys = ("heading1","heading2","heading3","heading4","heading5","heading6")
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


def get_docx_content_json(doc_token, token):
    """读取飞书文档原始块结构（JSON）"""
    all_blocks = []
    page_token = ""

    while True:
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks"
        params = {"page_size": 200}
        if page_token:
            params["page_token"] = page_token

        r = requests.get(url, headers=hdr(token), params=params)
        data = r.json()

        if data["code"] != 0:
            return {"error": data["msg"], "code": data["code"]}

        all_blocks.extend(data["data"].get("items", []))
        if not data["data"].get("has_more"):
            break
        page_token = data["data"].get("page_token", "")
        time.sleep(0.1)

    return {"text": blocks_to_text(all_blocks)}


def get_wiki_content_json(wiki_token_val, token):
    """wiki节点 → 找背后文档 → 读取内容"""
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token_val}"
    r = requests.get(url, headers=hdr(token))
    data = r.json()

    if data["code"] != 0:
        return {"error": data["msg"], "code": data["code"]}

    node = data["data"]["node"]
    obj_type  = node.get("obj_type", "")
    obj_token = node.get("obj_token", "")

    if obj_type in ("doc", "docx"):
        content = get_docx_content_json(obj_token, token)
        # content["wiki_node"] = node
        return content
    else:
        return { "note": f"obj_type={obj_type}，暂不支持读取正文"}


def fetch_doc_content_json(url, token):
    doc_type, doc_token = parse_doc_type(url)

    if doc_type == 'docx':
        return {"doc_type": "docx", "doc_token": doc_token,
                "content": get_docx_content_json(doc_token, token)}
    elif doc_type == 'wiki':
        return {"doc_type": "wiki", "doc_token": doc_token,
                "content": get_wiki_content_json(doc_token, token)}
    elif doc_type == 'sheets':
        return {"doc_type": "sheets", "doc_token": doc_token,
                "content": None, "note": "sheets类型暂不支持读取正文"}
    elif doc_type == 'bitable':
        return {"doc_type": "bitable", "doc_token": doc_token,
                "content": None, "note": "bitable类型暂不支持读取正文"}
    else:
        return {"doc_type": "unknown", "content": None, "note": "无法识别的文档类型"}


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    # 1. 读取 token
    token = read_token()
    print(f"✅ token：{token[:10]}...")

    # 2. 加载旧数据缓存
    existing_cache = load_existing_data()  # {链接: 文档条目}

    # 3. wiki → app_token
    app_token = get_bitable_token_from_wiki(WIKI_TOKEN, token)
    if not app_token:
        return

    # 4. 读取所有表格记录
    records = get_all_records(app_token, APP_TABLE_ID, VIEW_ID, token)
    if not records:
        print("❌ 未读取到任何记录")
        return

    # 5. 提取文档链接
    link_items = extract_links_from_records(records)

    # 6. 增量判断
    new_items  = [item for item in link_items if item["文档链接"] not in existing_cache]
    skip_count = len(link_items) - len(new_items)

    print(f"\n📋 增量分析：")
    print(f"   已缓存（跳过）：{skip_count} 个")
    print(f"   新增（需读取）：{len(new_items)} 个")

    # 7. 读取新文档内容
    new_results = []
    if new_items:
        print(f"\n📖 开始读取 {len(new_items)} 个新文档...")
        for i, item in enumerate(new_items, 1):
            url = item["文档链接"]
            print(f"   [{i}/{len(new_items)}] {url[:70]}...")
            doc_data = fetch_doc_content_json(url, token)
            new_results.append({
                "record_id": item["record_id"],
                "字段名":    item["字段名"],
                "文档链接":  url,
                "读取时间":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "文档数据":  doc_data,
            })
            time.sleep(1.0)
    else:
        print("\n✅ 所有文档均已缓存，无需重新读取")

    # 8. 合并旧缓存 + 新数据，按当前表格行顺序排列
    merged = dict(existing_cache)
    for item in new_results:
        merged[item["文档链接"]] = item

    final_list = [merged[item["文档链接"]] for item in link_items if item["文档链接"] in merged]

    # 9. 构建 record_id → 标题 映射
    record_title_map = {}
    for record in records:
        rid = record.get("record_id", "")
        content_field = record.get("fields", {}).get("内容", {})
        if isinstance(content_field, dict):
            record_title_map[rid] = content_field.get("text", "")
        else:
            record_title_map[rid] = str(content_field) if content_field else ""

    # 10. 保存
    save_data(records, final_list, record_title_map)

    # 10. 终端摘要
    print("\n==================== 完成 ====================")
    print(f"表格记录：{len(records)} 条")
    print(f"文档总数：{len(final_list)} 个（新增 {len(new_results)} 个，跳过 {skip_count} 个）")
    if new_results:
        print("\n--- 新增文档预览（前3个）---")
        for item in new_results[:3]:
            blocks = item["文档数据"].get("content", {})
            if isinstance(blocks, dict):
                block_count = blocks.get("block_count", 0)
            else:
                block_count = 0
            print(f"\n🔗 {item['文档链接']}")
            print(f"   类型：{item['文档数据'].get('doc_type')}  块数量：{block_count}")


if __name__ == "__main__":
    main()
