"""JCY 流水线共享工具（日期解析、文件命名、候选股筛选、YAML 加载）。"""

import json
import os
import re
import yaml

# 数据路径（相对于项目根目录）
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_FILE  = os.path.join(_BASE_DIR, "data", "jcy", "jcy_docs.yaml")
ADVICE_DIR = os.path.join(_BASE_DIR, "data", "jcy", "advice")
JSON_PATH  = os.path.join(_BASE_DIR, "data", "jcy", "jcy_insights.json")


def title_to_date(title: str) -> str | None:
    """从标题提取日期：'Vol.260226 今日更新' → '2026-02-26'"""
    m = re.search(r'(\d{6})', title)
    if m:
        ymd = m.group(1)
        return f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}"
    return None


def title_to_filename(title: str) -> str:
    """从标题生成文件名：'Vol.260226 今日更新' → '2026-02-26.md'"""
    date = title_to_date(title)
    if date:
        return f"{date}.md"
    safe = re.sub(r'[\\/:*?"<>|]', '_', title).strip()
    return f"{safe}.md"


def load_docs(docs_file: str = DOCS_FILE) -> list[dict]:
    """读取飞书文档 YAML 文件。"""
    with open(docs_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def is_ashare_code(code) -> bool:
    """判断是否为 6 位纯数字的 A 股代码。"""
    return bool(code and re.fullmatch(r"\d{6}", str(code)))


def load_candidates(json_path: str = JSON_PATH) -> list[dict]:
    """
    从 jcy_insights.json 筛选增持 A 股，去重后返回候选列表。
    同一股票多次出现时，保留 rating=增持 的最早记录。

    返回格式：[{"code": ..., "name": ..., "date": "YYYYMMDD", "reason": ...}, ...]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    articles = sorted(data.get("articles", []), key=lambda a: a.get("date", ""))

    seen: dict[str, dict] = {}
    for article in articles:
        for company in article.get("companies", []):
            code   = company.get("code")
            rating = company.get("rating", "")
            if rating != "增持" or not is_ashare_code(code):
                continue
            if code not in seen:
                seen[code] = {
                    "code":   code,
                    "name":   company.get("name", ""),
                    "date":   article.get("date", "").replace("-", ""),
                    "reason": company.get("rating_reason", ""),
                }

    return list(seen.values())
