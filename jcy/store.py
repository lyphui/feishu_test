"""单一真值源（jcy_insights.json）读写、复合键索引、Step 跳过判断、advice 路径解析。"""

import json
import os
from datetime import datetime

from utils.jcy_common import record_key, title_to_date
from jcy import config


def advice_path(advice_file: str | None) -> str | None:
    """把 record 的 advice_file 解析为本机可用路径。

    advice 文件统一平铺在 ADVICE_DIR，故只取 basename 再拼当前 ADVICE_DIR——
    既支持新存的相对文件名，也兼容历史存量的绝对路径（跨机器可移植）。
    """
    if not advice_file:
        return None
    return os.path.join(config.ADVICE_DIR, os.path.basename(advice_file.replace("\\", "/")))


def detect_doc_key_collisions(docs: list) -> dict:
    """检测文档清单中折叠到同一复合键 (date,title) 的项。

    Step 1 按 URL 去重抓取，Step 2/3 按 (date,title) 复合键去重；二者口径不同。
    若同一逻辑文章以两个 URL 重复出现、或标题内日期被编辑导致键漂移，
    会出现"多个文档 → 同一复合键"，Step 2/3 的 upsert 会静默互相覆盖。
    此函数把这类冲突显式暴露出来（返回 {key: [titles]}，仅含冲突项）。
    """
    by_key: dict = {}
    for d in docs:
        title = d.get("文档标题", "")
        key = record_key(title_to_date(title), title)
        by_key.setdefault(key, []).append(title)
    return {k: v for k, v in by_key.items() if len(v) > 1}


def record_index(articles: list) -> dict:
    """以复合键 record_key(date, title) 建索引 {key: list_index}。"""
    return {
        record_key(a.get("date"), a.get("title", "")): i
        for i, a in enumerate(articles)
    }


def step2_done(record: dict) -> bool:
    """Step 2 是否已完成：record 有 advice_file 且文件实际存在。"""
    path = advice_path(record.get("advice_file"))
    return bool(path) and os.path.exists(path)


def step3_done(record: dict) -> bool:
    """Step 3 是否已完成：record 含 extracted_at（提取完成标记）。"""
    return bool(record.get("extracted_at"))


def load_articles() -> list:
    """读取权威清单 articles（jcy_insights.json）。文件不存在返回空列表。"""
    if not os.path.exists(config.S3_OUTPUT_FILE):
        return []
    with open(config.S3_OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("articles", [])


def save_articles(articles: list) -> None:
    """按 date 倒序写回权威清单。原子写：先写临时文件再 os.replace，避免半写损坏。"""
    os.makedirs(os.path.dirname(config.S3_OUTPUT_FILE), exist_ok=True)
    sorted_articles = sorted(articles, key=lambda a: a.get("date") or "", reverse=True)
    tmp_path = f"{config.S3_OUTPUT_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total":      len(sorted_articles),
            "articles":   sorted_articles,
        }, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, config.S3_OUTPUT_FILE)


def upsert_record(articles: list, record: dict) -> list:
    """按复合键 upsert：存在则 merge 更新，否则 append。"""
    key = record_key(record.get("date"), record.get("title", ""))
    index = record_index(articles)
    if key in index:
        articles[index[key]].update(record)
    else:
        articles.append(record)
    return articles
