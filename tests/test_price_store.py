"""price_store 增量更新逻辑（不联网：抓取函数全部打桩）。"""

import json
import os

import pandas as pd
import pytest

from lib import price_store as ps


def _bars(start: str, n: int, base: float = 10.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "open": [base + i * 0.1 for i in range(n)],
        "high": [base + i * 0.1 + 0.2 for i in range(n)],
        "low": [base + i * 0.1 - 0.2 for i in range(n)],
        "close": [base + i * 0.1 for i in range(n)],
        "volume": [1000] * n,
    }, index=idx)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(ps, "DAILY_DIR", str(tmp_path / "daily"))
    monkeypatch.setattr(ps, "DIVIDEND_DIR", str(tmp_path / "dividend"))
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    """记录每次 _fetch 的请求区间，并返回该区间内的合成数据。"""
    log = []
    full = _bars("2024-01-01", 200)

    def fake_fetch(symbol, start, end, kind, adjust, proxy=""):
        log.append((start, end))
        lo = pd.to_datetime(start, format="%Y%m%d")
        hi = pd.to_datetime(end, format="%Y%m%d")
        return full.loc[lo:hi]

    monkeypatch.setattr(ps, "_fetch", fake_fetch)
    return log


def test_first_call_fetches_full_range(store, calls):
    df = ps.update_daily("TEST", "20240101", "20240401", verbose=False)
    assert len(calls) == 1
    assert not df.empty
    assert os.path.exists(ps.daily_path("TEST", "hfq"))


def test_second_call_only_fetches_tail(store, calls):
    ps.update_daily("TEST", "20240101", "20240301", verbose=False)
    calls.clear()
    ps.update_daily("TEST", "20240101", "20240401", verbose=False)
    # 只补尾段，且起点回退了重叠对账窗口，不是从 20240101 重来
    assert len(calls) == 1
    assert calls[0][0] > "20240101"
    assert calls[0][1] == "20240401"


def test_no_refetch_of_head_on_non_trading_start(store, calls):
    """请求起点落在非交易日时，不能每次都去补一个永远为空的头段。"""
    ps.update_daily("TEST", "20231231", "20240301", verbose=False)   # 12-31 是周日
    calls.clear()
    ps.update_daily("TEST", "20231231", "20240301", verbose=False)
    assert all(s >= "20240101" for s, _ in calls), f"重复补空头段: {calls}"


def test_meta_records_requested_range(store, calls):
    ps.update_daily("TEST", "20240101", "20240301", verbose=False)
    meta = json.loads(open(ps.meta_path("TEST", "hfq"), encoding="utf-8").read())
    assert meta["requested_start"] == "20240101"
    assert meta["requested_end"] == "20240301"
    assert meta["rows"] > 0


def test_rows_are_merged_not_duplicated(store, calls):
    ps.update_daily("TEST", "20240101", "20240301", verbose=False)
    ps.update_daily("TEST", "20240101", "20240401", verbose=False)
    df = ps.read_daily("TEST", "hfq")
    assert df.index.is_unique
    assert df.index.is_monotonic_increasing


def test_source_rewriting_history_triggers_rebuild(store, monkeypatch):
    """数据源改了复权口径 → 重叠对账失败 → 整表重建，而不是把两段缝起来。"""
    v1 = _bars("2024-01-01", 60, base=10.0)
    v2 = _bars("2024-01-01", 90, base=20.0)      # 整段历史价都变了
    state = {"df": v1}

    def fake_fetch(symbol, start, end, kind, adjust, proxy=""):
        lo = pd.to_datetime(start, format="%Y%m%d")
        hi = pd.to_datetime(end, format="%Y%m%d")
        return state["df"].loc[lo:hi]

    monkeypatch.setattr(ps, "_fetch", fake_fetch)
    ps.update_daily("TEST", "20240101", "20240215", verbose=False)
    state["df"] = v2
    out = ps.update_daily("TEST", "20240101", "20240401", verbose=False)
    # 全部来自新口径，没有残留旧价
    assert out["close"].min() >= 20.0


def test_gap_fetch_failure_keeps_local_cache(store, monkeypatch):
    """补增量时数据源挂了，应沿用本地缓存而不是抛异常。"""
    ok = _bars("2024-01-01", 60)

    def fetch_ok(symbol, start, end, kind, adjust, proxy=""):
        lo = pd.to_datetime(start, format="%Y%m%d")
        hi = pd.to_datetime(end, format="%Y%m%d")
        return ok.loc[lo:hi]

    monkeypatch.setattr(ps, "_fetch", fetch_ok)
    ps.update_daily("TEST", "20240101", "20240215", verbose=False)

    def fetch_boom(*a, **k):
        raise RuntimeError("数据源挂了")

    monkeypatch.setattr(ps, "_fetch", fetch_boom)
    out = ps.update_daily("TEST", "20240101", "20240401", verbose=False)
    assert not out.empty


def test_qfq_always_rebuilds(store, calls):
    """前复权无法安全追加，必须整表重建。"""
    ps.update_daily("TEST", "20240101", "20240301", adjust="qfq", verbose=False)
    calls.clear()
    ps.update_daily("TEST", "20240101", "20240401", adjust="qfq", verbose=False)
    assert calls[0][0] == "20240101"
