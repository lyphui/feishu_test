"""候选池筛选测试（load_candidates，离线，不读真实 data/）。

守两件事：
  1. 看多池必须收「买入」——旧实现只收「增持」，把 Strong Buy 整个丢在回测外
  2. 对照池能用同一个函数取出看空评级，否则没法回答"评级有没有区分度"
"""

import json

import pytest

from jcy.lib.common import CONTROL_RATINGS, LONG_RATINGS, load_candidates


def _write(tmp_path, articles):
    p = tmp_path / "insights.json"
    p.write_text(json.dumps({"articles": articles}, ensure_ascii=False),
                 encoding="utf-8")
    return str(p)


def _article(date, companies):
    return {"date": date, "title": f"T{date}", "companies": companies}


def _co(name, code, rating):
    return {"name": name, "code": code, "rating": rating,
            "rating_reason": f"{rating}理由"}


@pytest.fixture
def sample(tmp_path):
    return _write(tmp_path, [
        _article("2026-01-05", [_co("甲", "600001", "买入"),
                                _co("乙", "600002", "增持"),
                                _co("丙", "600003", "回避")]),
        _article("2026-02-05", [_co("丁", "000004", "减持"),
                                _co("戊", "300005", "持有"),
                                # 同一只票后来被再次提及，不得覆盖最早记录
                                _co("甲", "600001", "增持")]),
        _article("2026-03-05", [_co("己", "NVDA", "买入"),      # 非 A 股代码
                                _co("庚", None, "增持"),        # 无代码
                                _co("辛", "600007", "卖出")]),
    ])


def test_long_pool_includes_strong_buy(sample):
    """「买入」是最强评级，必须进池子——旧实现只收「增持」把它整个丢了。"""
    codes = {c["code"] for c in load_candidates(sample)}
    assert "600001" in codes, "买入评级的票必须进回测池"
    assert "600002" in codes
    assert codes == {"600001", "600002"}


def test_candidate_carries_its_rating(sample):
    by_code = {c["code"]: c for c in load_candidates(sample)}
    assert by_code["600001"]["rating"] == "买入"
    assert by_code["600002"]["rating"] == "增持"


def test_control_pool_picks_bearish_ratings(sample):
    """看空对照池：同一个函数换一组评级即可，保证两池口径完全一致。"""
    codes = {c["code"] for c in load_candidates(sample, ratings=CONTROL_RATINGS)}
    assert codes == {"600003", "000004", "600007"}          # 回避/减持/卖出


def test_pools_are_disjoint_on_ratings(sample):
    assert set(LONG_RATINGS).isdisjoint(CONTROL_RATINGS)


def test_neutral_ratings_are_in_neither_pool(sample):
    """「持有」既不看多也不看空，两个池子都不该收。"""
    long_codes = {c["code"] for c in load_candidates(sample)}
    ctrl_codes = {c["code"] for c in load_candidates(sample,
                                                     ratings=CONTROL_RATINGS)}
    assert "300005" not in long_codes | ctrl_codes


def test_keeps_earliest_record_per_code(sample):
    """甲在 1 月是买入、2 月是增持，只保留最早那条。"""
    甲 = next(c for c in load_candidates(sample) if c["code"] == "600001")
    assert 甲["date"] == "20260105"
    assert 甲["rating"] == "买入"


def test_non_ashare_and_missing_codes_are_dropped(sample):
    codes = {c["code"] for c in load_candidates(sample)}
    assert "NVDA" not in codes and None not in codes


def test_articles_without_date_are_skipped(tmp_path):
    """没有推荐日就定不出回测起点，只能跳过（不能默默按今天算）。"""
    path = _write(tmp_path, [
        {"date": None, "title": "无日期", "companies": [_co("甲", "600001", "买入")]},
        _article("2026-01-05", [_co("乙", "600002", "增持")]),
    ])
    assert {c["code"] for c in load_candidates(path)} == {"600002"}


def test_explicit_single_rating(tmp_path, sample):
    assert {c["code"] for c in load_candidates(sample, ratings=("买入",))} == {"600001"}
