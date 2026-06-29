import os
from scripts.migrate_compound_key import plan_migration, apply_migration


def test_plan_maps_old_to_new_name(tmp_path):
    advice = tmp_path / "advice"
    advice.mkdir()
    (advice / "2026-06-26.md").write_text("x", encoding="utf-8")
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L"}]
    plan = plan_migration(docs, str(advice), articles=[])
    olds = [os.path.basename(o) for o, _ in plan["renames"]]
    news = [os.path.basename(n) for _, n in plan["renames"]]
    assert "2026-06-26.md" in olds
    assert "2026-06-26__Vol.260626 时代主题.md" in news
    assert plan["missing"] == []


def test_plan_reports_missing_when_no_old_file(tmp_path):
    advice = tmp_path / "advice"
    advice.mkdir()
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L"}]
    plan = plan_migration(docs, str(advice), articles=[])
    assert plan["renames"] == []
    assert "2026-06-26__Vol.260626 时代主题" in plan["missing"]


def test_plan_same_date_two_docs_only_one_claims_old_file(tmp_path):
    # 旧命名一天只有一份文件；两篇同日文档只有一篇能认领，另一篇报 missing
    advice = tmp_path / "advice"
    advice.mkdir()
    (advice / "2026-06-25.md").write_text("x", encoding="utf-8")
    docs = [
        {"文档标题": "Vol.260625 上午", "文档链接": "L1"},
        {"文档标题": "Vol.260625 下午", "文档链接": "L2"},
    ]
    plan = plan_migration(docs, str(advice), articles=[])
    assert len(plan["renames"]) == 1
    assert len(plan["missing"]) == 1


def test_apply_renames_and_rewrites_json(tmp_path):
    advice = tmp_path / "advice"
    advice.mkdir()
    (advice / "2026-06-26.md").write_text("x", encoding="utf-8")
    insights = tmp_path / "jcy_insights.json"
    import json
    insights.write_text(json.dumps({"articles": [
        {"date": "2026-06-26", "title": "Vol.260626 时代主题",
         "advice_file": str(advice / "2026-06-26.md")}
    ]}, ensure_ascii=False), encoding="utf-8")
    cache = tmp_path / "advice_cache.json"
    cache.write_text("{}", encoding="utf-8")
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L"}]
    plan = plan_migration(docs, str(advice), articles=json.loads(insights.read_text(encoding="utf-8"))["articles"])
    apply_migration(plan, str(advice), str(insights), str(cache))
    assert (advice / "2026-06-26__Vol.260626 时代主题.md").exists()
    assert not (advice / "2026-06-26.md").exists()
    assert not cache.exists()
    new_articles = json.loads(insights.read_text(encoding="utf-8"))["articles"]
    assert new_articles[0]["advice_file"].endswith("2026-06-26__Vol.260626 时代主题.md")
