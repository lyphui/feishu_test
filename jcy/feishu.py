"""Step 1 — 飞书数据采集：统一 GET 包装、分页读表、文档正文抓取、增量缓存写盘。"""

import json
import os
import re
import time
from datetime import datetime

import requests
import yaml

from utils.jcy_text import blocks_to_text
from jcy import config

log = config.log


def _hdr(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def feishu_get(url: str, token: str, params: dict | None = None,
               retries: int = 2) -> dict:
    """统一飞书 GET：查 HTTP 状态、code != 0 抛错、5xx/超时有限重试。返回 data 字段。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=_hdr(token), params=params, timeout=30)
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


def _read_token():
    with open(config.S1_TOKEN_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_existing():
    """返回 {链接: 文档条目} 字典"""
    if not os.path.exists(config.S1_DOCS_FILE):
        log.info("📂 未找到旧数据文件，将全量读取")
        return {}
    with open(config.S1_DOCS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    existing = {item["文档链接"]: item for item in data if item.get("文档链接")}
    log.info(f"📂 读取到旧数据：{len(existing)} 条已缓存文档")
    return existing


def _save(records, doc_list, record_title_map):
    os.makedirs(config._DATA_DIR, exist_ok=True)
    table_output = {
        "更新时间":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "表格记录总数":   len(records),
        "表格原始数据":   records,
    }
    with open(config.S1_TABLE_FILE, "w", encoding="utf-8") as f:
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
    with open(config.S1_DOCS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(yaml_list, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    log.info(f"💾 表格 → {config.S1_TABLE_FILE}")
    log.info(f"💾 文档 → {config.S1_DOCS_FILE}（共 {len(yaml_list)} 条）")


def _get_app_token(wiki_token, token):
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token}"
    try:
        data = feishu_get(url, token)
    except RuntimeError as e:
        log.error(f"❌ 获取wiki节点失败：{e}")
        return None
    obj_token = data["node"]["obj_token"]
    log.info(f"✅ bitable app_token：{obj_token}")
    return obj_token


def _get_all_records(app_token, table_id, view_id, token):
    all_records = []
    page_token  = ""
    page        = 1
    log.info(f"\n📊 读取多维表格（{table_id}）...")

    while True:
        url = (f"https://open.feishu.cn/open-apis/bitable/v1/apps/"
               f"{app_token}/tables/{table_id}/records")
        params = {"page_size": 100}
        if view_id:
            params["view_id"] = view_id
        if page_token:
            params["page_token"] = page_token

        try:
            data = feishu_get(url, token, params)
        except RuntimeError as e:
            log.error(f"❌ 读取表格失败：{e}")
            break

        items = data.get("items", [])
        all_records.extend(items)
        log.info(f"   第{page}页：{len(items)} 条")

        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")
        page += 1
        time.sleep(0.2)

    log.info(f"✅ 共 {len(all_records)} 条记录")
    return all_records


def _extract_links(records):
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
    log.info(f"✅ 提取到 {len(items)} 个不重复文档链接")
    return items


def _parse_doc_type(url):
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


def _get_docx_content(doc_token, token):
    all_blocks = []
    page_token = ""
    while True:
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks"
        params = {"page_size": 200}
        if page_token:
            params["page_token"] = page_token
        try:
            data = feishu_get(url, token, params)
        except RuntimeError as e:
            return {"error": str(e)}
        all_blocks.extend(data.get("items", []))
        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")
        time.sleep(0.1)
    return {"text": blocks_to_text(all_blocks)}


def _get_wiki_content(wiki_token_val, token):
    url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token={wiki_token_val}"
    try:
        data = feishu_get(url, token)
    except RuntimeError as e:
        return {"error": str(e)}
    node      = data["node"]
    obj_type  = node.get("obj_type", "")
    obj_token = node.get("obj_token", "")
    if obj_type in ("doc", "docx"):
        return _get_docx_content(obj_token, token)
    return {"note": f"obj_type={obj_type}，暂不支持读取正文"}


def _fetch_doc(url, token):
    doc_type, doc_token = _parse_doc_type(url)
    if doc_type == 'docx':
        return {"doc_type": "docx", "doc_token": doc_token,
                "content": _get_docx_content(doc_token, token)}
    elif doc_type == 'wiki':
        return {"doc_type": "wiki", "doc_token": doc_token,
                "content": _get_wiki_content(doc_token, token)}
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
        token = _read_token()
    except FileNotFoundError:
        log.error(f"❌ 未找到 token 文件：{config.S1_TOKEN_FILE}")
        return False
    log.info(f"✅ token：{token[:10]}...")

    existing_cache = _load_existing()

    app_token = _get_app_token(config.S1_WIKI_TOKEN, token)
    if not app_token:
        return False

    records = _get_all_records(app_token, config.S1_APP_TABLE_ID, config.S1_VIEW_ID, token)
    if not records:
        log.error("❌ 未读取到任何记录")
        return False

    link_items = _extract_links(records)
    new_items  = [item for item in link_items if item["文档链接"] not in existing_cache]
    skip_count = len(link_items) - len(new_items)
    log.info(f"\n📋 增量分析：已缓存（跳过）{skip_count} 个，新增（需读取）{len(new_items)} 个")

    new_results = []
    if new_items:
        log.info(f"\n📖 开始读取 {len(new_items)} 个新文档...")
        for i, item in enumerate(new_items, 1):
            url = item["文档链接"]
            log.info(f"   [{i}/{len(new_items)}] {url[:70]}...")
            doc_data = _fetch_doc(url, token)
            new_results.append({
                "record_id": item["record_id"],
                "字段名":    item["字段名"],
                "文档链接":  url,
                "读取时间":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "文档数据":  doc_data,
            })
            time.sleep(1.0)
    else:
        log.info("✅ 所有文档均已缓存，无需重新读取")

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

    _save(records, final_list, record_title_map)
    log.info(f"\n表格记录：{len(records)} 条，文档总数：{len(final_list)} 个"
          f"（新增 {len(new_results)} 个，跳过 {skip_count} 个）")
    return True
