import os

from prepare_jcy_data import (
    _record_index,
    _step2_done,
    _step3_done,
    _upsert_record,
)


def test_record_index_keys_by_compound():
    articles = [
        {"date": "2026-06-26", "title": "上午"},
        {"date": "2026-06-26", "title": "下午"},
    ]
    idx = _record_index(articles)
    assert idx == {"2026-06-26__上午": 0, "2026-06-26__下午": 1}


def test_record_index_none_date():
    idx = _record_index([{"date": None, "title": "无日期"}])
    assert "NODATE__无日期" in idx


def test_step2_done_true_when_file_exists(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("x", encoding="utf-8")
    assert _step2_done({"advice_file": str(f)}) is True


def test_step2_done_false_when_file_missing(tmp_path):
    assert _step2_done({"advice_file": str(tmp_path / "missing.md")}) is False


def test_step2_done_false_when_no_field():
    assert _step2_done({}) is False


def test_step3_done_true_when_extracted_at_present():
    assert _step3_done({"extracted_at": "2026-06-26 10:00:00"}) is True


def test_step3_done_false_when_absent():
    assert _step3_done({"date": "2026-06-26"}) is False


def test_upsert_appends_new():
    arts = []
    _upsert_record(arts, {"date": "2026-06-26", "title": "新", "x": 1})
    assert len(arts) == 1 and arts[0]["x"] == 1


def test_upsert_merges_existing_same_key():
    arts = [{"date": "2026-06-26", "title": "同", "advice_file": "a.md"}]
    _upsert_record(arts, {"date": "2026-06-26", "title": "同", "extracted_at": "t"})
    assert len(arts) == 1
    assert arts[0]["advice_file"] == "a.md"
    assert arts[0]["extracted_at"] == "t"


def test_upsert_same_date_diff_title_two_records():
    arts = []
    _upsert_record(arts, {"date": "2026-06-26", "title": "上午"})
    _upsert_record(arts, {"date": "2026-06-26", "title": "下午"})
    assert len(arts) == 2
