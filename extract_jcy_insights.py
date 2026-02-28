"""
对每篇文章及对应建议文档，调用 Azure OpenAI 提取：
  - 公司和股票代码
  - 投资市场代码
  - 投资倾向
  - 精炼建议

增量处理：已提取的文章自动跳过
输出：data/jcy/jcy_insights.json
"""
import os
import re
import json
import time
import yaml
from datetime import datetime
from openai import AzureOpenAI

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ======================== 配置项 ========================
AZURE_ENDPOINT   = "https://llm-east-us2-test.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_API_VER    = "2024-12-01-preview"

DOCS_FILE    = os.path.join(os.path.dirname(__file__), "data", "jcy", "jcy_docs.yaml")
ADVICE_DIR   = os.path.join(os.path.dirname(__file__), "data", "jcy", "advice")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "data", "jcy", "jcy_insights.json")

SLEEP_BETWEEN = 2   # 每次请求间隔（秒）
# =======================================================

SYSTEM_PROMPT = """你是一位专业的股票市场分析助手。请从提供的股市分析文章和投资建议中，提取关键的结构化信息。

严格按照以下 JSON 格式输出，不要输出任何其他内容：

{
  "companies": [
    {
      "name": "公司中文名",
      "code": "股票代码（如600000、000001、0700.HK、NVDA，若不确定则为null）",
      "exchange": "具体交易所，从以下选择：上交所/深交所/北交所/港交所/纳斯达克/纽交所/其他，若不确定则为null",
      "rating": "投资评级，从以下选择：买入/增持/持有/减持/卖出/回避，若文章无明确倾向则为null",
      "rating_reason": "评级依据，一句话说明（不超过30字，若rating为null则省略）"
    }
  ],
  "markets": ["A股", "港股","美股"，"等"],
  "tendency": "整体投资倾向，例如：看涨科技/看涨周期/防御/观望/多元配置等（一句话）",
  "key_advice": [
    "建议1（简洁，不超过50字）",
    "建议2",
    "建议3"
  ]
}

说明：
- companies：文章中明确提到的股票或公司，仅当文章涉及股市分析时填写，否则为空数组 []
- exchange 判断规则（A股代码）：
    * 600xxx / 601xxx / 603xxx / 605xxx / 688xxx → 上交所（科创板在上交所）
    * 000xxx / 001xxx / 002xxx / 003xxx / 300xxx / 301xxx → 深交所（创业板在深交所）
    * 430xxx / 830xxx / 83xxxx / 87xxxx / 88xxxx / 899xxx → 北交所
    * 末尾含 .HK 或 .hk → 港交所
    * 无数字代码的美股（如 NVDA、AAPL）→ 纳斯达克 或 纽交所（根据常识判断）
- rating 投资评级含义：
    * 买入（Strong Buy）：强烈推荐，预期涨幅显著高于市场
    * 增持（Overweight/Add）：看好，建议适度加仓
    * 持有（Hold/Neutral）：中性，维持现有仓位
    * 减持（Underweight/Reduce）：谨慎，建议降低仓位
    * 卖出（Sell）：明确看空，建议清仓
    * 回避（Avoid）：风险较高，不建议介入
- markets：文章主要讨论的投资市场，从以下选择：A股、港股、美股、期货、基金、其他
- tendency：用一句话总结整体投资倾向，要具体（如"看好国产半导体，规避美股AI"比"看涨"更好）
- key_advice：精炼的3-5条核心建议，每条简洁明了

只输出 JSON，不要 markdown 代码块，不要任何解释。"""


def title_to_date(title):
    """从标题提取日期：'Vol.260226 今日更新' → '2026-02-26'"""
    m = re.search(r'(\d{6})', title)
    if m:
        ymd = m.group(1)
        return f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}"
    return None


def title_to_filename(title):
    """'Vol.260226 今日更新' → '2026-02-26.md'"""
    date = title_to_date(title)
    if date:
        return f"{date}.md"
    safe = re.sub(r'[\\/:*?"<>|]', '_', title).strip()
    return f"{safe}.md"


def load_docs():
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def load_advice(filename):
    path = os.path.join(ADVICE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_output():
    """加载已有输出，返回 (结果列表, {date: index} 查找表)"""
    if not os.path.exists(OUTPUT_FILE):
        return [], {}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    index = {a["date"]: i for i, a in enumerate(articles) if "date" in a}
    return articles, index


def save_output(articles):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(articles),
        "articles": articles,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_insights(client, doc, advice_text):
    """调用 GPT 提取结构化信息，返回 dict"""
    title   = doc.get("文档标题", "")
    content = doc.get("文档内容正文", "").strip()

    user_prompt = f"""【原文标题】{title}

【原文内容】
{content}

【AI生成的投资建议】
{advice_text or "（暂无建议文档）"}"""

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_completion_tokens=2000,
        temperature=0.2,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            return json.loads(m.group())
        raise ValueError(f"无法解析 JSON 响应: {raw[:200]}")


def main():
    print("=" * 55)
    print("  飞书文档 → 结构化洞察提取器（Azure GPT-4.1）")
    print("=" * 55)

    docs = load_docs()
    print(f"共读取到 {len(docs)} 篇文档")

    articles, date_index = load_output()
    print(f"已提取：{len(articles)} 篇，本次跳过已有记录\n")

    client = AzureOpenAI(
        api_version=AZURE_API_VER,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )

    new_count = 0
    for i, doc in enumerate(docs):
        if i>3:
            break
        title    = doc.get("文档标题", "（无标题）")
        link     = doc.get("文档链接", "")
        date_str = title_to_date(title)
        filename = title_to_filename(title)

        # 增量跳过
        if date_str and date_str in date_index:
            print(f"[跳过] {title}")
            continue

        advice_text = load_advice(filename)
        if advice_text is None:
            print(f"[警告] 建议文件不存在：{filename}，仅用原文分析")

        print(f"[{new_count + 1}] 提取：{title} ...")

        try:
            insights = extract_insights(client, doc, advice_text)
        except Exception as e:
            print(f"   !! 失败: {e}")
            insights = {
                "companies": [],
                "markets": [],
                "tendency": "提取失败",
                "key_advice": [str(e)[:100]],
            }

        record = {
            "date":       date_str or title,
            "title":      title,
            "link":       link,
            "advice_file": os.path.abspath(os.path.join(ADVICE_DIR, filename)),
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **insights,
        }

        articles.append(record)
        if date_str:
            date_index[date_str] = len(articles) - 1

        save_output(articles)
        new_count += 1
        companies = insights.get('companies', [])
        rated = [c for c in companies if c.get('rating')]
        print(f"   ✅ 已保存（公司:{len(companies)}个，有评级:{len(rated)}个，"
              f"市场:{insights.get('markets', [])}，"
              f"建议:{len(insights.get('key_advice', []))}条）")

        if new_count < len(docs):
            time.sleep(SLEEP_BETWEEN)

    print(f"\n{'=' * 55}")
    print(f"本次新增提取：{new_count} 篇")
    print(f"输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()