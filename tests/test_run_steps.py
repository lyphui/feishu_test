"""run_step2 / run_step3 循环逻辑 + 路径解析 + 原子写 的离线测试（不联网）。"""

import json
import os

import prepare_jcy_data as p


# ── _advice_path 解析 ────────────────────────────────────────────

def test_advice_path_resolves_bare_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "ADVICE_DIR", str(tmp_path))
    assert p._advice_path("a.md") == os.path.join(str(tmp_path), "a.md")


def test_advice_path_reduces_legacy_absolute_path(monkeypatch, tmp_path):
    # 历史存量的绝对路径（甚至是另一台机器的 Windows 路径）应被还原到当前 ADVICE_DIR
    monkeypatch.setattr(p, "ADVICE_DIR", str(tmp_path))
    legacy = r"C:\old\machine\data\jcy\advice\2026-06-26__x.md"
    assert p._advice_path(legacy) == os.path.join(str(tmp_path), "2026-06-26__x.md")


def test_advice_path_none():
    assert p._advice_path(None) is None
    assert p._advice_path("") is None


def test_step2_done_uses_current_advice_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "ADVICE_DIR", str(tmp_path))
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    assert p._step2_done({"advice_file": "a.md"}) is True
    assert p._step2_done({"advice_file": "missing.md"}) is False


# ── _save_articles 原子写 ────────────────────────────────────────

def test_save_articles_atomic_no_tmp_left(monkeypatch, tmp_path):
    out = tmp_path / "jcy_insights.json"
    monkeypatch.setattr(p, "S3_OUTPUT_FILE", str(out))
    p._save_articles([{"date": "2026-06-26", "title": "t"}])
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
    monkeypatch.setattr(p, "ADVICE_DIR", str(tmp_path / "advice"))
    monkeypatch.setattr(p, "S3_OUTPUT_FILE", str(tmp_path / "insights.json"))
    monkeypatch.setattr(p, "PerplexityAPI", _FakePplx)
    monkeypatch.setattr(p, "_s2_analyze_doc", lambda pplx, doc: analyze_result)
    monkeypatch.setattr(p.time, "sleep", lambda *a: None)


def test_step2_skips_empty_content(monkeypatch, tmp_path):
    calls = []
    _setup_step2(monkeypatch, tmp_path, ("建议正文", []))
    monkeypatch.setattr(p, "_s2_analyze_doc",
                        lambda pplx, doc: calls.append(doc) or ("建议正文", []))
    docs = [{"文档标题": "Vol.260626 空", "文档链接": "L", "文档内容正文": "   "}]
    p.run_step2(docs)
    assert calls == []  # 空正文不应触发分析
    assert not os.path.exists(str(tmp_path / "insights.json"))


def test_step2_success_writes_file_then_record(monkeypatch, tmp_path):
    _setup_step2(monkeypatch, tmp_path, ("建议正文", ["http://cite"]))
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}]
    p.run_step2(docs)

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
    (advice_dir / "2026-06-26__Vol.260626 时代主题.md").write_text("x", encoding="utf-8")
    (tmp_path / "insights.json").write_text(json.dumps({"articles": [
        {"date": "2026-06-26", "title": "Vol.260626 时代主题",
         "advice_file": "2026-06-26__Vol.260626 时代主题.md"}
    ]}, ensure_ascii=False), encoding="utf-8")

    called = []
    monkeypatch.setattr(p, "_s2_analyze_doc", lambda pplx, doc: called.append(1) or ("x", []))
    docs = [{"文档标题": "Vol.260626 时代主题", "文档链接": "L", "文档内容正文": "有正文"}]
    p.run_step2(docs)
    assert called == []  # 已完成应跳过，不再调用


def test_step2_consecutive_timeout_aborts(monkeypatch, tmp_path):
    import requests
    _setup_step2(monkeypatch, tmp_path, None)

    def _always_timeout(pplx, doc):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(p, "_s2_analyze_doc", _always_timeout)
    docs = [{"文档标题": f"Vol.26062{i} 标题{i}", "文档链接": f"L{i}",
             "文档内容正文": "正文"} for i in range(5)]
    p.run_step2(docs)  # 不应抛异常；连续超时达上限即 break
    # 没有任何成功写入
    assert not os.path.exists(str(tmp_path / "insights.json"))
