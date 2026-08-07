"""单一真值源（jcy_insights.json）读写、复合键索引、Step 跳过判断、advice 路径解析。"""

import json
import os
from datetime import datetime

from jcy.lib.common import record_key, title_to_date, title_to_filename
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


#: advice 正文（去掉 front-matter 后）的最小长度；低于此视为截断/失败的残留文件
ADVICE_MIN_BODY_CHARS = 200


def advice_complete(path: str | None) -> bool:
    """advice 文件是否存在且内容完整。

    完整 = 具备 _build_md 写出的结构（H1 标题 + 分析时间元信息 + 分隔线），
    且分隔线之后的正文足够长。用于识别 API 中断/写了一半的残留文件，
    这类文件应当重跑而不是被当成已完成。
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    if not text.lstrip().startswith("# ") or "**分析时间：**" not in text:
        return False
    head, sep, body = text.partition("\n---\n")
    return bool(sep) and len(body.strip()) >= ADVICE_MIN_BODY_CHARS


def step2_done(record: dict, title: str = "") -> bool:
    """Step 2 是否已完成：advice 文件存在且完整即算完成。

    以磁盘上的文件为准，不依赖 jcy_insights.json 中是否有对应 record——
    record 可能因写入中断/键漂移而缺失，但只要分析文档已完整生成，
    就不该再消耗一次 Perplexity 调用重跑。
    """
    path = advice_path(record.get("advice_file"))
    if advice_complete(path):
        return True
    # record 缺失或没有 advice_file 时，按标题推导出的规范文件名再查一次磁盘
    if title:
        return advice_complete(advice_path(title_to_filename(title)))
    return False


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
