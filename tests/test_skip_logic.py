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


def test_step2_done_true_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    assert step2_done({"advice_file": "a.md"}) is True


def test_step2_done_false_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    assert step2_done({"advice_file": "missing.md"}) is False


def test_step2_done_false_when_no_field():
    assert step2_done({}) is False


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
