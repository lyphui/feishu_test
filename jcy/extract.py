"""Step 3 — LLM 结构化提取，按 S3_PROVIDERS 顺序自动回退（DashScope / Azure / Coze）。"""

import json
import os
import time
from datetime import datetime

import requests
from openai import AzureOpenAI, OpenAI

from jcy.lib.text import parse_json_loose
from jcy.lib.common import title_to_date, title_to_filename, record_key
from jcy import config, store

log = config.log


def _load_advice(filename):
    path = os.path.join(config.ADVICE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _coze_extract_content(data):
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


def _call_provider(provider, user_prompt):
    """Call one LLM provider. Returns dict or raises on any error."""
    if provider["type"] == "dashscope":
        client = OpenAI(
            api_key=config.S3_DASHSCOPE_API_KEY,
            base_url=config.S3_DASHSCOPE_BASE_URL,
        )
        response = client.chat.completions.create(
            model=config.S3_DASHSCOPE_MODEL,
            messages=[
                {"role": "system", "content": config.S3_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=16000,
            response_format={"type": "json_object"},
        )
        return parse_json_loose(response.choices[0].message.content.strip())

    if provider["type"] == "azure":
        client = AzureOpenAI(
            api_version=config.S3_AZURE_API_VER,
            azure_endpoint=config.S3_AZURE_ENDPOINT,
            api_key=config.S3_AZURE_API_KEY,
        )
        response = client.chat.completions.create(
            model=config.S3_AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": config.S3_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_completion_tokens=16000,
            response_format={"type": "json_object"},
        )
        return parse_json_loose(response.choices[0].message.content.strip())

    if provider["type"] == "coze":
        r = requests.post(
            config.S3_COZE_URL,
            headers={"Authorization": f"Bearer {config.S3_COZE_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": config.S3_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "model": config.S3_COZE_MODEL,
                "temperature": 0.3,
                "max_tokens": 16000,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        raw = _coze_extract_content(data)
        return parse_json_loose(raw)

    raise ValueError(f"未知 provider 类型: {provider['type']}")


def _extract_insights(doc, advice_text):
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
    active = [p for p in config.S3_PROVIDERS if p["enabled"]]
    if not active:
        raise RuntimeError("没有可用的 LLM Provider（请检查 API Key 环境变量）")
    last_err = None
    for provider in active:
        try:
            return _call_provider(provider, user_prompt), provider["name"]
        except (OSError, RuntimeError, ValueError, TimeoutError) as e:
            log.warning(f"   ⚠️  {provider['name']} 失败: {type(e).__name__}: {e}")
            last_err = e
    raise last_err or RuntimeError("所有 LLM Provider 均失败（无具体异常）")


def run_step3(docs):
    """LLM 结构化提取，按 S3_PROVIDERS 顺序自动回退。Step 3 跳过 = record 已有 extracted_at。"""
    active_names = [p["name"] for p in config.S3_PROVIDERS if p["enabled"]]
    log.info(f"🤖 LLM 顺序：{' → '.join(active_names) or '无（请配置 API Key）'}\n")

    articles = store.load_articles()
    new_count       = 0
    consec_failures = 0

    for doc in docs:
        title    = doc.get("文档标题", "（无标题）")
        link     = doc.get("文档链接", "")
        date_str = title_to_date(title)
        filename = title_to_filename(title)
        key      = record_key(date_str, title)

        index = store.record_index(articles)
        rec   = articles[index[key]] if key in index else {}
        if store.step3_done(rec):
            log.info(f"[跳过] {title}")
            continue

        advice_text = _load_advice(filename)
        if advice_text is None:
            log.warning(f"[警告] 建议文件不存在：{filename}，仅用原文分析")

        log.info(f"[{new_count + 1}] 提取：{title} ...")
        try:
            insights, used = _extract_insights(doc, advice_text)
        except (OSError, RuntimeError, ValueError, TimeoutError) as e:
            consec_failures += 1
            log.error(f"   ❌ 所有 Provider 均失败: {e}")
            if consec_failures >= config.MAX_CONSECUTIVE_FAILURES:
                log.error(f"   ❌ 连续失败 {config.MAX_CONSECUTIVE_FAILURES} 次，终止 Step 3")
                break
            time.sleep(config.S3_SLEEP)
            continue

        consec_failures = 0
        store.upsert_record(articles, {
            "date":         date_str,
            "title":        title,
            "link":         link,
            "advice_file":  filename,
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm_provider": used,
            **insights,
        })
        store.save_articles(articles)
        new_count += 1
        companies = insights.get("companies", [])
        rated     = [c for c in companies if c.get("rating")]
        log.info(f"   ✅ 已保存（公司:{len(companies)}个，有评级:{len(rated)}个，"
              f"市场:{insights.get('markets', [])}，"
              f"建议:{len(insights.get('key_advice', []))}条）[{used}]")
        time.sleep(config.S3_SLEEP)

    log.info(f"\nStep 3 完成：本次新增 {new_count} 篇，输出：{config.S3_OUTPUT_FILE}")
