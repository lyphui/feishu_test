from utils.jcy_text import blocks_to_text, parse_json_loose


def _text_block(content):
    return {"text": {"elements": [{"text_run": {"content": content}}]}}


def _heading_block(level, content):
    return {f"heading{level}": {"elements": [{"text_run": {"content": content}}]}}


def test_blocks_to_text_extracts_text_and_headings():
    blocks = [_heading_block(1, "标题"), _text_block("正文一"), _text_block("正文二")]
    assert blocks_to_text(blocks) == "标题\n正文一\n正文二"


def test_blocks_to_text_skips_empty():
    blocks = [_text_block("有内容"), _text_block("   "), _text_block("")]
    assert blocks_to_text(blocks) == "有内容"


def test_blocks_to_text_empty_list():
    assert blocks_to_text([]) == ""


def test_parse_json_loose_plain():
    assert parse_json_loose('{"a": 1}') == {"a": 1}


def test_parse_json_loose_with_fence():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_loose_embedded_in_prose():
    assert parse_json_loose('好的，结果是 {"a": 1} 完毕') == {"a": 1}


def test_parse_json_loose_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_json_loose("没有 JSON 的纯文本")
