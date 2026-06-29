import pytest
import prepare_jcy_data as p


class _Resp:
    def __init__(self, status, payload, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _call():
    return p._feishu_get("http://x", "tok", {})


def test_feishu_get_success(monkeypatch):
    monkeypatch.setattr(p.requests, "get",
                        lambda *a, **k: _Resp(200, {"code": 0, "data": {"ok": 1}}))
    assert _call() == {"ok": 1}


def test_feishu_get_api_error_raises(monkeypatch):
    monkeypatch.setattr(p.requests, "get",
                        lambda *a, **k: _Resp(200, {"code": 99, "msg": "bad"}))
    with pytest.raises(RuntimeError, match="bad"):
        _call()


def test_feishu_get_http_error_raises(monkeypatch):
    monkeypatch.setattr(p.requests, "get",
                        lambda *a, **k: _Resp(500, {}))
    with pytest.raises(RuntimeError):
        p._feishu_get("http://x", "tok", {}, retries=0)
