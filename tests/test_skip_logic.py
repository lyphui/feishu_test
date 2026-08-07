import jcy.config as config
from jcy.store import (
    detect_doc_key_collisions,
    record_index,
    step2_done,
    step3_done,
    upsert_record,
)


def test_detect_doc_key_collisions_none():
    docs = [
        {"文档标题": "Vol.260626 上午"},
        {"文档标题": "Vol.260626 下午"},
    ]
    assert detect_doc_key_collisions(docs) == {}


def test_detect_doc_key_collisions_same_date_same_title():
    docs = [
        {"文档标题": "Vol.260626 时代主题", "文档链接": "L1"},
        {"文档标题": "Vol.260626 时代主题", "文档链接": "L2"},  # 同标题不同 URL
    ]
    coll = detect_doc_key_collisions(docs)
    assert "2026-06-26__Vol.260626 时代主题" in coll
    assert len(coll["2026-06-26__Vol.260626 时代主题"]) == 2


def test_record_index_keys_by_compound():
    articles = [
        {"date": "2026-06-26", "title": "上午"},
        {"date": "2026-06-26", "title": "下午"},
    ]
    idx = record_index(articles)
    assert idx == {"2026-06-26__上午": 0, "2026-06-26__下午": 1}


def test_record_index_none_date():
    idx = record_index([{"date": None, "title": "无日期"}])
    assert "NODATE__无日期" in idx


def _complete_md(title="标题"):
    """构造一份结构完整的 advice 文件内容（同 advice._build_md 的结构）。"""
    return (
        f"# {title}\n\n"
        "> **原文链接：** [L](L)  \n"
        "> **分析时间：** 2026-08-07 10:00:00\n\n"
        "---\n\n" + "正文内容。" * 60 + "\n"
    )


def test_step2_done_true_when_file_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    (tmp_path / "a.md").write_text(_complete_md(), encoding="utf-8")
    assert step2_done({"advice_file": "a.md"}) is True


def test_step2_done_false_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    assert step2_done({"advice_file": "missing.md"}) is False


def test_step2_done_false_when_no_field():
    assert step2_done({}) is False


def test_step2_done_false_when_file_truncated(tmp_path, monkeypatch):
    """半写/截断的残留文件不算完成，应重跑。"""
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    (tmp_path / "a.md").write_text("# 标题\n\n> **分析时间：** t\n\n---\n\n太短", encoding="utf-8")
    assert step2_done({"advice_file": "a.md"}) is False


def test_step2_done_true_from_title_when_record_missing(tmp_path, monkeypatch):
    """核心回归：insights.json 里没有该 record，但分析文档已完整存在 → 跳过。"""
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    title = "Vol.260806 好在没放量"
    (tmp_path / "2026-08-06__Vol.260806 好在没放量.md").write_text(
        _complete_md(title), encoding="utf-8")
    assert step2_done({}, title) is True


def test_step2_done_title_fallback_handles_trailing_space(tmp_path, monkeypatch):
    """标题带尾随空格时，文件名经 safe_title 去空格，仍应命中。"""
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    (tmp_path / "2026-08-04__Vol.260804 美股又站出来.md").write_text(
        _complete_md(), encoding="utf-8")
    assert step2_done({}, "Vol.260804 美股又站出来 ") is True


def test_step2_done_false_when_title_has_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    assert step2_done({}, "Vol.260807 尚未分析") is False


def test_step3_done_true_when_extracted_at_present():
    assert step3_done({"extracted_at": "2026-06-26 10:00:00"}) is True


def test_step3_done_false_when_absent():
    assert step3_done({"date": "2026-06-26"}) is False


def test_upsert_appends_new():
    arts = []
    upsert_record(arts, {"date": "2026-06-26", "title": "新", "x": 1})
    assert len(arts) == 1 and arts[0]["x"] == 1


def test_upsert_merges_existing_same_key():
    arts = [{"date": "2026-06-26", "title": "同", "advice_file": "a.md"}]
    upsert_record(arts, {"date": "2026-06-26", "title": "同", "extracted_at": "t"})
    assert len(arts) == 1
    assert arts[0]["advice_file"] == "a.md"
    assert arts[0]["extracted_at"] == "t"


def test_upsert_same_date_diff_title_two_records():
    arts = []
    upsert_record(arts, {"date": "2026-06-26", "title": "上午"})
    upsert_record(arts, {"date": "2026-06-26", "title": "下午"})
    assert len(arts) == 2
