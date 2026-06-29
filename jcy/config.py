"""JCY 流水线配置：常量、环境变量、system prompt、logging 初始化。

注意：`ADVICE_DIR` 从 jcy.lib.common 复用（与候选股筛选共用同一路径常量）。
"""

import logging
import os

from dotenv import load_dotenv, find_dotenv

from jcy.lib.common import ADVICE_DIR  # noqa: F401  (re-export for pipeline modules)

load_dotenv(find_dotenv())

# ════════════════════════════════════════════════════════════════
#  全局
# ════════════════════════════════════════════════════════════════

MAX_CONSECUTIVE_TIMEOUTS = 3   # Step 2：连续 API 超时达到此次数时终止
MAX_CONSECUTIVE_FAILURES = 3   # Step 3：连续提取失败达到此次数时终止

log = logging.getLogger("jcy")
# 默认即可输出（即便未调用 setup_logging，如测试/REPL 直接调用 run_step*）。
# setup_logging 会清空并重配 handler，故这里只装一个干净的控制台 handler。
if not log.handlers:
    log.setLevel(logging.INFO)
    log.propagate = False
    _default_console = logging.StreamHandler()
    _default_console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_default_console)


def setup_logging(log_file: str | None):
    """配置 jcy logger：控制台保留干净的中文/emoji 风格（仅 message），
    文件（若指定）带完整时间戳+级别，便于排查。幂等：重复调用不叠加 handler。"""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False  # 不冒泡到 root，避免重复输出

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)


_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR    = os.path.join(_BASE_DIR, "data", "jcy")
# prompts 随 jcy 包内置（jcy/prompts/），故取本文件所在目录，不走仓库根
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def read_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


# ─── Step 1 ──────────────────────────────────────────────────────
S1_TOKEN_FILE   = os.getenv("TOKEN_FILE")
S1_TABLE_FILE   = os.path.join(_DATA_DIR, "jcy_table.json")
S1_DOCS_FILE    = os.path.join(_DATA_DIR, "jcy_docs.yaml")
S1_WIKI_TOKEN   = os.getenv("JCY_WIKI_TOKEN")
S1_APP_TABLE_ID = os.getenv("JCY_APP_TABLE_ID")
S1_VIEW_ID      = os.getenv("JCY_VIEW_ID")

# ─── Step 2 ──────────────────────────────────────────────────────
S2_API_KEY      = os.getenv("PPLX_API_KEY")
S2_GROUP_ID     = os.getenv("PPLX_GROUP_ID")
S2_SLEEP        = 2
S2_SYSTEM_PROMPT = read_prompt("step2_advice_system.md")

# ─── Step 3 ──────────────────────────────────────────────────────
S3_AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
S3_AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")
S3_AZURE_API_KEY    = os.getenv("AZURE_OPENAI_KEY", "")
S3_AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
S3_COZE_API_KEY     = os.getenv("COZE_API_KEY", "")
S3_COZE_URL         = os.getenv("COZE_URL", "")
S3_COZE_MODEL       = os.getenv("COZE_MODEL", "doubao-pro")
S3_DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY", "")
S3_DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
S3_DASHSCOPE_MODEL    = os.getenv("DASHSCOPE_MODEL", "deepseek-v4-pro")
S3_OUTPUT_FILE      = os.path.join(_DATA_DIR, "jcy_insights.json")
S3_SLEEP            = 2

# LLM providers in fallback order; disable by leaving the API key env var empty
S3_PROVIDERS = [
    {"name": "DashScope", "type": "dashscope", "enabled": bool(S3_DASHSCOPE_API_KEY)},
    # {"name": "Azure OpenAI", "type": "azure", "enabled": bool(S3_AZURE_API_KEY)},
    # {"name": "Coze",         "type": "coze",  "enabled": bool(S3_COZE_API_KEY)},
]

S3_SYSTEM_PROMPT = read_prompt("step3_extract_system.md")
