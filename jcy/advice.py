"""Step 2 — Perplexity 投资建议生成。单一真值源 + 先落文件后写 record 的原子性。"""

import os
import time
from datetime import datetime

import requests

from jcy.lib.pplx import PerplexityAPI
from jcy.lib.common import title_to_date, title_to_filename, record_key
from jcy import config, store

log = config.log


def _build_md(title, link, analyzed_at, advice, citations):
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


def _save_md(filename, content):
    os.makedirs(config.ADVICE_DIR, exist_ok=True)
    path = os.path.join(config.ADVICE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _analyze_doc(pplx, doc):
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
            {"role": "system", "content": config.S2_SYSTEM_PROMPT},
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
    连续超时达 config.MAX_CONSECUTIVE_TIMEOUTS 次时终止。"""
    pplx     = PerplexityAPI(config.S2_API_KEY, config.S2_GROUP_ID)
    articles = store.load_articles()
    index    = store.record_index(articles)

    # 跳过：该文档对应 record 已有 advice 文件；或正文为空（无法分析，避免每次空跑重试）
    pending = []
    empty_count = 0
    for d in docs:
        title = d.get("文档标题", "")
        if not d.get("文档内容正文", "").strip():
            empty_count += 1
            continue
        key   = record_key(title_to_date(title), title)
        rec   = articles[index[key]] if key in index else {}
        if not store.step2_done(rec):
            pending.append(d)
    done_count = len(docs) - len(pending) - empty_count
    log.info(f"📋 Step 2 本次需分析：{len(pending)} 篇"
          f"（已完成 {done_count} 篇，空正文跳过 {empty_count} 篇）\n")

    if not pending:
        log.info("✅ 所有文档均已生成建议，无需重新处理")
        return

    consecutive_timeouts = 0
    new_count = 0
    for i, doc in enumerate(pending, 1):
        title = doc.get("文档标题", "（无标题）")
        link  = doc.get("文档链接", "")
        date  = title_to_date(title)
        log.info(f"[{i}/{len(pending)}] 分析：{title}")

        try:
            advice, citations = _analyze_doc(pplx, doc)
        except requests.exceptions.Timeout:
            consecutive_timeouts += 1
            log.info(f"   ⏰ 请求超时（连续第 {consecutive_timeouts}/{config.MAX_CONSECUTIVE_TIMEOUTS} 次）")
            if consecutive_timeouts >= config.MAX_CONSECUTIVE_TIMEOUTS:
                log.error(f"   ❌ 连续超时 {config.MAX_CONSECUTIVE_TIMEOUTS} 次，终止 Step 2")
                break
            time.sleep(config.S2_SLEEP)
            continue

        if advice is None:
            log.error("   ❌ API 响应无效，跳过本篇\n")
            consecutive_timeouts = 0
            continue

        consecutive_timeouts = 0
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename    = title_to_filename(title)
        # ── 原子性：先落文件 ──
        path = _save_md(filename, _build_md(title, link, analyzed_at, advice, citations))
        # ── 再更新权威清单（存相对文件名，跨机器可移植）──
        store.upsert_record(articles, {
            "date":        date,
            "title":       title,
            "link":        link,
            "advice_file": filename,
        })
        store.save_articles(articles)
        index = store.record_index(articles)
        new_count += 1
        log.info(f"   ✅ 已保存 → {path}\n")
        if i < len(pending):
            time.sleep(config.S2_SLEEP)

    log.info(f"\nStep 2 完成：本次新增 {new_count} 篇，文件目录：{config.ADVICE_DIR}")
