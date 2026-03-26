"""analyze_jcy.py 和 extract_jcy_insights.py 共享的工具函数。"""

import os
import re
import yaml

# 数据路径（相对于项目根目录）
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_FILE  = os.path.join(_BASE_DIR, "data", "jcy", "jcy_docs.yaml")
ADVICE_DIR = os.path.join(_BASE_DIR, "data", "jcy", "advice")


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
