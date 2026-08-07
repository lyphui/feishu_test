from jcy.feishu import _extract_links, _parse_doc_type


def _rec(rid, link):
    return {"record_id": rid, "fields": {"内容": {"link": link, "text": "标题"}}}


def test_extract_links_with_subdomain():
    items = _extract_links([_rec("r1", "https://abc.feishu.cn/wiki/Tok123")])
    assert [i["文档链接"] for i in items] == ["https://abc.feishu.cn/wiki/Tok123"]


def test_extract_links_bare_domain():
    """裸域名（无子域名）形式也须被提取，否则该记录不会有文档正文。"""
    items = _extract_links([_rec("r2", "https://feishu.cn/wiki/Tok456")])
    assert [i["文档链接"] for i in items] == ["https://feishu.cn/wiki/Tok456"]


def test_extract_links_dedupes_across_records():
    url = "https://feishu.cn/docx/TokSame"
    items = _extract_links([_rec("r1", url), _rec("r2", url)])
    assert len(items) == 1


def test_extract_links_ignores_non_feishu():
    items = _extract_links([_rec("r1", "https://example.com/wiki/Tok789")])
    assert items == []


def test_parse_doc_type_bare_domain():
    assert _parse_doc_type("https://feishu.cn/wiki/Tok456") == ("wiki", "Tok456")
    assert _parse_doc_type("https://abc.feishu.cn/docx/Tok789") == ("docx", "Tok789")
