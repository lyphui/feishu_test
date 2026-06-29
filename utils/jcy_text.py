"""JCY 流水线纯文本逻辑（无 IO/网络，可单测）：飞书 block 转文本、容错 JSON 解析。"""

import json
import re


def blocks_to_text(blocks: list[dict]) -> str:
    """飞书 docx block 列表 → 纯文本。提取 heading1-6 与 text block 的文本，空行跳过。"""
    lines = []
    heading_keys = ("heading1", "heading2", "heading3",
                    "heading4", "heading5", "heading6")
    for block in blocks:
        matched = False
        for key in heading_keys:
            if key in block:
                text = "".join(
                    e.get("text_run", {}).get("content", "")
                    for e in block[key].get("elements", [])
                )
                if text.strip():
                    lines.append(text.strip())
                matched = True
                break
        if not matched and "text" in block:
            text = "".join(
                e.get("text_run", {}).get("content", "")
                for e in block["text"].get("elements", [])
            )
            if text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def parse_json_loose(raw: str) -> dict:
    """容错解析 JSON：先直接解析，失败则提取首个 {...} 块，仍失败抛 ValueError。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            return json.loads(m.group())
        raise ValueError(f"无法解析 JSON 响应: {raw[:200]}")
