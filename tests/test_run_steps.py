"""run_step2 循环逻辑 + 路径解析 + 原子写 的离线测试（不联网）。"""

import json
import os

import jcy.config as config
import jcy.store as store
import jcy.advice as advice


# ── advice_path 解析 ─────────────────────────────────────────────

def test_advice_path_resolves_bare_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    assert store.advice_path("a.md") == os.path.join(str(tmp_path), "a.md")


def test_advice_path_reduces_legacy_absolute_path(monkeypatch, tmp_path):
    # 历史存量的绝对路径（甚至是另一台机器的 Windows 路径）应被还原到当前 ADVICE_DIR
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    legacy = r"C:\old\machine\data\jcy\advice\2026-06-26__x.md"
    assert store.advice_path(legacy) == os.path.join(str(tmp_path), "2026-06-26__x.md")


def test_advice_path_none():
    assert store.advice_path(None) is None
    assert store.advice_path("") is None


def _complete_md(title="标题"):
    return (
        f"# {title}\n\n"
        "> **原文链接：** [L](L)  \n"
        "> **分析时间：** 2026-08-07 10:00:00\n\n"
        "---\n\n" + "正文内容。" * 60 + "\n"
    )


def test_step2_done_uses_current_advice_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path))
    (tmp_path / "a.md").write_text(_complete_md(), encoding="utf-8")
    assert store.step2_done({"advice_file": "a.md"}) is True
    assert store.step2_done({"advice_file": "missing.md"}) is False


# ── save_articles 原子写 ─────────────────────────────────────────

def test_save_articles_atomic_no_tmp_left(monkeypatch, tmp_path):
    out = tmp_path / "jcy_insights.json"
    monkeypatch.setattr(config, "S3_OUTPUT_FILE", str(out))
    store.save_articles([{"date": "2026-06-26", "title": "t"}])
    assert out.exists()
    assert not (tmp_path / "jcy_insights.json.tmp").exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total"] == 1


# ── run_step2 循环 ──────────────────────────────────────────────

class _FakePplx:
    def __init__(self, *a, **k):
        pass


def _setup_step2(monkeypatch, tmp_path, analyze_result):
    """把 ADVICE_DIR / S3_OUTPUT_FILE 指向 tmp，桩掉网络。"""
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path / "advice"))
    monkeypatch.setattr(config, "S3_OUTPUT_FILE", str(tmp_path / "insights.json"))
    monkeypatch.setattr(advice, "PerplexityAPI", _FakePplx)
    monkeypatch.setattr(advice, "_analyze_doc", lambda pplx, doc: analyze_result)
    monkeypatch.setattr(advice.time, "sleep", lambda *a: None)


def test_step2_skips_empty_content(monkeypatch, tmp_path):
    calls = []
    _setup_step2(monkeypatch, tmp_path, ("建议正文", []))
    monkeypatch.setattr(advice, "_analyze_doc",
                        lambda pplx, doc: calls.append(doc) or ("建议正文", []))
    docs = [{"文档标题": "Vol.260626 空", "文档链接": "L", "文档内容正文": "   "}]
    advice.run_step2(docs)
    assert calls == []  # 空正文不应触发分析
    assert not os.path.exists(str(tmp_path / "insights.json"))


def test_step2_success_writes_file_then_record(monkeypatch, tmp_path):
    _setup_step2(monkeypatch, tmp_path, ("建议正文", ["http://cite"]))
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}]
    advice.run_step2(docs)

    advice_file = tmp_path / "advice" / "2026-06-26__Vol.260626 时代主题.md"
    assert advice_file.exists()
    data = json.loads((tmp_path / "insights.json").read_text(encoding="utf-8"))
    rec = data["articles"][0]
    assert rec["advice_file"] == "2026-06-26__Vol.260626 时代主题.md"  # 相对文件名
    assert rec["date"] == "2026-06-26"


def test_step2_skips_already_done(monkeypatch, tmp_path):
    _setup_step2(monkeypatch, tmp_path, ("建议正文", []))
    # 预置已完成 record + 真实 advice 文件
    advice_dir = tmp_path / "advice"
    advice_dir.mkdir()
    (advice_dir / "2026-06-26__Vol.260626 时代主题.md").write_text(
        _complete_md("Vol.260626 时代主题"), encoding="utf-8")
    (tmp_path / "insights.json").write_text(json.dumps({"articles": [
        {"date": "2026-06-26", "title": "Vol.260626 时代主题",
         "advice_file": "2026-06-26__Vol.260626 时代主题.md"}
    ]}, ensure_ascii=False), encoding="utf-8")

    called = []
    monkeypatch.setattr(advice, "_analyze_doc", lambda pplx, doc: called.append(1) or ("x", []))
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}]
    advice.run_step2(docs)
    assert called == []  # 已完成应跳过，不再调用


def test_step2_consecutive_timeout_aborts(monkeypatch, tmp_path):
    import requests
    _setup_step2(monkeypatch, tmp_path, None)

    def _always_timeout(pplx, doc):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(advice, "_analyze_doc", _always_timeout)
    docs = [{"文档标题": f"Vol.26062{i} 标题{i}", "文档链接": f"L{i}",
             "文档内容正文": "正文"} for i in range(5)]
    advice.run_step2(docs)  # 不应抛异常；连续超时达上限即 break
    # 没有任何成功写入
    assert not os.path.exists(str(tmp_path / "insights.json"))
    # 失败时不得留下任何 md 文件（否则下次会被误判成已完成而永久跳过）
    advice_dir = tmp_path / "advice"
    assert not advice_dir.exists() or list(advice_dir.glob("*.md")) == []


def test_step2_api_failure_writes_no_md(monkeypatch, tmp_path):
    """pplx 返回无效（None）时不落 md，下次仍会重跑。"""
    _setup_step2(monkeypatch, tmp_path, (None, []))
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}]
    advice.run_step2(docs)
    advice_dir = tmp_path / "advice"
    assert not advice_dir.exists() or list(advice_dir.glob("*.md")) == []
    assert not os.path.exists(str(tmp_path / "insights.json"))


def test_step2_rejects_too_short_response(monkeypatch, tmp_path):
    """API 成功返回但内容过短（截断）→ 视为无效，不落 md。"""

    class _Pplx:
        def chat(self, **k):
            return {"choices": [{"message": {"content": "太短了"}}]}

    doc = {"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}
    assert advice._analyze_doc(_Pplx(), doc) == (None, [])


def test_step2_reruns_truncated_md(monkeypatch, tmp_path):
    """磁盘上是半写/截断的 md → 不算完成，应重新调用 API。"""
    _setup_step2(monkeypatch, tmp_path, ("建议正文" * 100, []))
    advice_dir = tmp_path / "advice"
    advice_dir.mkdir()
    (advice_dir / "2026-06-26__Vol.260626 时代主题.md").write_text(
        "# 半个文件", encoding="utf-8")

    called = []
    monkeypatch.setattr(advice, "_analyze_doc",
                        lambda pplx, doc: called.append(1) or ("建议正文" * 100, []))
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}]
    advice.run_step2(docs)
    assert len(called) == 1  # 截断文件应触发重跑


def test_step2_skips_when_md_exists_but_record_missing(monkeypatch, tmp_path):
    """核心回归：insights.json 无该 record，但完整 md 已存在 → 不再调用 API。"""
    _setup_step2(monkeypatch, tmp_path, ("建议正文", []))
    advice_dir = tmp_path / "advice"
    advice_dir.mkdir()
    (advice_dir / "2026-06-26__Vol.260626 时代主题.md").write_text(
        _complete_md("Vol.260626 时代主题"), encoding="utf-8")
    # 注意：不写 insights.json，模拟 record 缺失

    called = []
    monkeypatch.setattr(advice, "_analyze_doc", lambda pplx, doc: called.append(1) or ("x", []))
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}]
    advice.run_step2(docs)
    assert called == []


def test_save_md_atomic_no_tmp_left(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ADVICE_DIR", str(tmp_path / "advice"))
    path = advice._save_md("a.md", _complete_md())
    assert os.path.exists(path)
    assert list((tmp_path / "advice").glob("*.tmp")) == []
