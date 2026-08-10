"""JCY 流水线共享工具（日期解析、文件命名、候选股筛选、YAML 加载）。"""

import json
import os
import re
import yaml

# 数据路径（相对于项目根目录）。本文件在 jcy/lib/ 下，故向上 3 层到仓库根。
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_FILE  = os.path.join(_BASE_DIR, "data", "jcy", "jcy_docs.yaml")
ADVICE_DIR = os.path.join(_BASE_DIR, "data", "jcy", "advice")
JSON_PATH  = os.path.join(_BASE_DIR, "data", "jcy", "jcy_insights.json")


def safe_title(title: str, maxlen: int = 80) -> str:
    """标题安全化：替换文件名非法字符、去首尾空白、截断到 maxlen。"""
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", title).strip()
    return cleaned[:maxlen]


def record_key(date: str | None, title: str) -> str:
    """复合去重键：date + 安全化 title。date 缺失时用 NODATE 占位。

    一天可能有多条数据，故不能用单 date 做键。
    """
    return f"{date or 'NODATE'}__{safe_title(title)}"


def title_to_date(title: str) -> str | None:
    """从标题提取日期：'Vol.260626 今日更新' → '2026-06-26'。

    假设：6 位数字为 YYMMDD，世纪前缀固定 '20'（2000-2099）。
    匹配标题中第一个 6 位连续数字；无匹配返回 None（不再 fallback 成标题）。
    """
    m = re.search(r'(\d{6})', title)
    if m:
        ymd = m.group(1)
        return f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}"
    return None


def title_to_filename(title: str) -> str:
    """生成 advice 文件名：复合命名 '{date or NODATE}__{safe_title}.md'。

    与 record_key 一致，避免一天多条互相覆盖文件。
    """
    date = title_to_date(title)
    return f"{date or 'NODATE'}__{safe_title(title)}.md"


def load_docs(docs_file: str = DOCS_FILE) -> list[dict]:
    """读取飞书文档 YAML 文件。"""
    with open(docs_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def is_ashare_code(code) -> bool:
    """判断是否为 6 位纯数字的 A 股代码。"""
    return bool(code and re.fullmatch(r"\d{6}", str(code)))


# 正向池：看多评级。旧实现只收「增持」，把 51 只「买入」（Strong Buy，更强的
# 信号）整个排除在回测之外——用中间档定义股票池而丢掉最强档，没有理由。
LONG_RATINGS = ("买入", "增持")

# 对照池：看空/回避评级。单跑正向池只能说明「这批票涨没涨」，回答不了
# 「这个评级体系有没有区分度」——那需要看空池同口径跑一遍作对照。
CONTROL_RATINGS = ("减持", "卖出", "回避")


def parse_ratings(value: str) -> tuple[str, ...]:
    """逗号分隔的评级列表 → tuple；空串 → 空 tuple（表示不启用）。

    给 argparse 的 type= 用。放在这里而不是各入口脚本里各写一份：
    两个脚本的 --ratings 必须同义，复制出来的解析函数迟早会漂移。
    """
    return tuple(r.strip() for r in (value or "").split(",") if r.strip())


def load_candidates(json_path: str = JSON_PATH,
                    ratings: tuple[str, ...] | list[str] = LONG_RATINGS) -> list[dict]:
    """
    从 jcy_insights.json 按评级筛选 A 股，去重后返回候选列表。
    同一股票多次出现时，保留**首次落入 ratings** 的最早记录。

    ratings : 收录哪些评级，默认 LONG_RATINGS（买入 + 增持）。
              传 CONTROL_RATINGS 可得到看空对照池。

    返回格式：
      [{"code":..., "name":..., "date": "YYYYMMDD", "rating":..., "reason":...}, ...]

    已知边界：同一只票可能在正向池和对照池里各出现一次（先增持、后回避），
    两个池子的日期不同，这是真实情况，不做特殊处理。
    """
    ratings = tuple(ratings)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # date 可能是 None（Step 3 未能从标题解析出日期）。用 `or ""` 兜底：
    # 既保证排序不因 None 与 str 比较而崩，也让无日期的文章排在最前、
    # 随后被下面的 `if not date` 跳过——没有推荐日就无法确定回测起点。
    articles = sorted(data.get("articles", []), key=lambda a: a.get("date") or "")

    seen: dict[str, dict] = {}
    for article in articles:
        date = (article.get("date") or "").replace("-", "")
        if not date:
            continue
        for company in article.get("companies") or []:
            code   = company.get("code")
            rating = company.get("rating", "")
            if rating not in ratings or not is_ashare_code(code):
                continue
            if code not in seen:
                seen[code] = {
                    "code":   code,
                    "name":   company.get("name", ""),
                    "date":   date,
                    "rating": rating,
                    "reason": company.get("rating_reason", ""),
                }

    return list(seen.values())
