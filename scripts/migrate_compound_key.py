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
    claimed = set()  # 已被认领的旧文件（一天多条时旧命名只有一份，先到先得）
    for d in docs:
        title = d.get("文档标题", "")
        date  = title_to_date(title)
        key   = record_key(date, title)
        new_name = title_to_filename(title)
        new_path = os.path.join(advice_dir, new_name)
        old_name = _old_filename(date)
        old_path = os.path.join(advice_dir, old_name) if old_name else None
        old_abs  = os.path.abspath(old_path) if old_path else None
        if os.path.exists(new_path):
            continue  # 已是新命名，无需动作
        if old_abs and old_abs not in claimed and os.path.exists(old_path):
            if old_abs != os.path.abspath(new_path):
                renames.append((old_path, new_path))
                claimed.add(old_abs)
        else:
            # 旧文件不存在，或同日旧文件已被另一篇认领 → 缺产物
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
