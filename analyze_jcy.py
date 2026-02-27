"""
基于飞书文档内容，调用 Perplexity sonar-reasoning-pro 生成投资建议
面向投资小白，逐篇分析，增量跳过已分析文档
每篇输出为独立 Markdown 文件，按日期命名
"""
import os
import sys
import re
import json
import time
import yaml
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 把 utils 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from pplx import PerplexityAPI

# ======================== 配置项 ========================
API_KEY   = os.getenv("PPLX_API_KEY")
GROUP_ID  = os.getenv("PPLX_GROUP_ID")

DOCS_FILE   = os.path.join(os.path.dirname(__file__), "data", "jcy", "jcy_docs.yaml")
ADVICE_DIR  = os.path.join(os.path.dirname(__file__), "data", "jcy", "advice")   # MD 文件目录
CACHE_FILE  = os.path.join(os.path.dirname(__file__), "data", "jcy", "advice_cache.json")  # 增量缓存

SLEEP_BETWEEN = 2   # 每次请求间隔（秒），避免触发限速

# 跳过模式：
#   "cache" - 按 advice_cache.json 中记录的文档链接跳过（精确）
#   "files" - 按 advice/ 目录中已存在的 .md 文件名跳过（适合缓存丢失时重建）
SKIP_MODE = "files"
# =======================================================

SYSTEM_PROMPT = """你是一位善于用简单语言解释投资的顾问，同时会结合最新的互联网资讯进行验证和补充。
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


# ─── 文件名工具 ──────────────────────────────────────────────────────────────

def title_to_filename(title):
    """
    从标题提取日期生成文件名。
    "Vol.260226 今日更新" → "2026-02-26.md"
    无日期时用标题做文件名（去除非法字符）。
    """
    m = re.search(r'(\d{6})', title)
    if m:
        ymd = m.group(1)
        return f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}.md"
    safe = re.sub(r'[\\/:*?"<>|]', '_', title).strip()
    return f"{safe}.md"


# ─── 缓存读写 ─────────────────────────────────────────────────────────────────

def load_cache():
    """返回 {文档链接: md文件名} 字典"""
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_skip_set():
    """
    根据 SKIP_MODE 返回已处理项的集合：
      "cache" → 集合内容为已记录的文档链接
      "files" → 集合内容为 advice/ 目录中已存在的 .md 文件名
    """
    if SKIP_MODE == "files":
        if not os.path.isdir(ADVICE_DIR):
            return set()
        return {f for f in os.listdir(ADVICE_DIR) if f.endswith(".md")}
    else:  # "cache"
        return set(load_cache().keys())


# ─── Markdown 生成与保存 ──────────────────────────────────────────────────────

def build_md(title, link, analyzed_at, advice, citations):
    lines = [
        f"# {title}",
        "",
        f"> **原文链接：** [{link}]({link})  ",
        f"> **分析时间：** {analyzed_at}",
        "",
        "---",
        "",
        advice.strip(),
    ]
    if citations:
        lines += [
            "",
            "---",
            "",
            "## 引用来源",
            "",
        ]
        for idx, url in enumerate(citations, 1):
            lines.append(f"{idx}. {url}")
    return "\n".join(lines) + "\n"


def save_md(filename, content):
    os.makedirs(ADVICE_DIR, exist_ok=True)
    path = os.path.join(ADVICE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ─── API 调用 ─────────────────────────────────────────────────────────────────

def load_docs():
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def analyze_doc(pplx, doc):
    """调用 sonar-reasoning-pro 分析单篇文档，返回 (建议文本, 引用列表)"""
    title   = doc.get("文档标题", "")
    content = doc.get("文档内容正文", "").strip()

    if not content:
        return "（文档内容为空，无法分析）", []

    user_prompt = f"""以下是一篇股市分析文章，标题是「{title}」：

---
{content}
---

请根据上面的内容，结合你搜索到的最新资讯，为投资小白生成详细的投资分析报告。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    result = pplx.chat(
        model="sonar-reasoning-pro",
        messages=messages,
        max_tokens=30000,
        temperature=0.3,
    )

    if not result:
        return "（API 请求失败）", []

    choices = result.get("choices", [])
    if not choices:
        return "（响应中无内容）", []

    raw = choices[0]["message"]["content"]
    # sonar-reasoning-pro 会在 <think>...</think> 后输出正文
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].strip()

    citations = result.get("citations", [])
    return raw, citations


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  飞书文档 → Perplexity 投资建议生成器")
    print("=" * 55)

    docs = load_docs()
    print(f"📂 共读取到 {len(docs)} 篇文档")

    cache = load_cache()  # 始终加载缓存，用于写回（无论哪种跳过模式）
    skip_set = get_skip_set()
    mode_label = "目录文件扫描" if SKIP_MODE == "files" else "JSON缓存"
    print(f"📂 跳过模式：{mode_label}，已跳过项：{len(skip_set)} 个\n")

    if SKIP_MODE == "files":
        new_docs = [d for d in docs if title_to_filename(d.get("文档标题", "")) not in skip_set]
    else:
        new_docs = [d for d in docs if d.get("文档链接") not in skip_set]
    print(f"📋 本次需分析：{len(new_docs)} 篇\n")

    if not new_docs:
        print("✅ 所有文档均已分析，无需重新处理")
        return

    pplx = PerplexityAPI(API_KEY, GROUP_ID)

    for i, doc in enumerate(new_docs, 1):
        title = doc.get("文档标题", "（无标题）")
        link  = doc.get("文档链接", "")
        print(f"[{i}/{len(new_docs)}] 分析：{title}")

        advice, citations = analyze_doc(pplx, doc)
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        filename = title_to_filename(title)
        md_content = build_md(title, link, analyzed_at, advice, citations)
        path = save_md(filename, md_content)

        cache[link] = filename
        save_cache(cache)

        print(f"   ✅ 已保存 → {path}\n")

        if i < len(new_docs):
            time.sleep(SLEEP_BETWEEN)

    print(f"\n==================== 完成 ====================")
    print(f"本次新增分析：{len(new_docs)} 篇")
    print(f"累计总计：{len(cache)} 篇")
    print(f"文件目录：{ADVICE_DIR}")


if __name__ == "__main__":
    main()
