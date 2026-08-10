"""
分时取数只能有**一处**实现。

`jcy_intraday_timing` 曾经自带一份 `_fetch_intraday_baostock`，与
`lib/intraday_store.fetch_intraday_raw` 是两份独立代码，而且**复权口径还不一样**
（前者 qfq、后者 none），靠人看着两处维持一致。两个口径本身都对——分时 MACD 要
复权（不复权在除权日有假跳空），VWAP 基准要不复权（amount/volume 恒为原始值）——
所以正确形态是同一个函数的两种参数，而不是两份实现。

这些测试全部离线：用假的 baostock 模块拦住网络调用，只检查参数与形状。
"""

import sys
import types

import pandas as pd
import pytest

from lib import intraday_store as store

jit = pytest.importorskip("jcy_intraday_timing")


class _FakeRS:
    """最小可用的 baostock 结果集：吐两根 K 线。"""
    fields = ["date", "time", "open", "high", "low", "close", "volume", "amount"]
    error_code = "0"

    def __init__(self):
        self._rows = [
            ["2026-03-05", "20260305100000000", "10.0", "10.2", "9.9", "10.1",
             "1000", "10100"],
            ["2026-03-05", "20260305103000000", "10.1", "10.4", "10.0", "10.3",
             "2000", "20600"],
        ]

    def next(self):
        return bool(self._rows)

    def get_row_data(self):
        return self._rows.pop(0)


@pytest.fixture
def fake_bs(monkeypatch):
    """装一个假 baostock，记录 query 收到的 adjustflag。"""
    seen = {}

    def query(code, fields, start_date, end_date, frequency, adjustflag):
        seen.update(code=code, frequency=frequency, adjustflag=adjustflag)
        return _FakeRS()

    mod = types.SimpleNamespace(
        login=lambda: None, logout=lambda: None,
        query_history_k_data_plus=query)
    monkeypatch.setitem(sys.modules, "baostock", mod)
    return seen


# ── 单一实现 ──────────────────────────────────────────────────────────────────

def test_jcy_no_longer_has_its_own_baostock_fetcher():
    assert not hasattr(jit, "_fetch_intraday_baostock"), \
        "分时取数又出现第二份实现了——应当复用 intraday_store.fetch_intraday_raw"


def test_jcy_delegates_to_the_shared_fetcher(monkeypatch):
    """akshare 失败时必须走仓库那一份，且传 qfq。"""
    monkeypatch.setattr(jit, "_fetch_intraday_akshare", lambda *a, **k: None)
    calls = {}

    def spy(symbol, start, end, period=30, adjust="none"):
        calls.update(symbol=symbol, period=period, adjust=adjust)
        return pd.DataFrame({
            "dt": pd.to_datetime(["2026-03-05 10:00", "2026-03-05 10:30"]),
            "date": pd.to_datetime(["2026-03-05"] * 2),
            "open": [10.0, 10.1], "high": [10.2, 10.4], "low": [9.9, 10.0],
            "close": [10.1, 10.3], "volume": [1000, 2000],
            "amount": [10100, 20600]})

    monkeypatch.setattr(jit, "fetch_intraday_raw", spy)
    out = jit.fetch_intraday("601857", "20260301", "20260306", period=30)

    assert calls == {"symbol": "601857", "period": 30, "adjust": "qfq"}
    # 本模块按 datetime 索引消费，且不要 amount
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2


def test_jcy_returns_empty_frame_when_both_sources_fail(monkeypatch):
    """取数失败必须是空表而不是异常——上层靠"无分时数据也照常建仓"兜底。"""
    monkeypatch.setattr(jit, "_fetch_intraday_akshare", lambda *a, **k: None)
    monkeypatch.setattr(jit, "fetch_intraday_raw",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert jit.fetch_intraday("601857", "20260301", "20260306").empty


# ── 复权口径 ──────────────────────────────────────────────────────────────────

def test_adjust_maps_to_the_right_baostock_flag(fake_bs):
    store.fetch_intraday_raw("601857", "20260301", "20260306", 30, adjust="none")
    assert fake_bs["adjustflag"] == "3"
    store.fetch_intraday_raw("601857", "20260301", "20260306", 30, adjust="qfq")
    assert fake_bs["adjustflag"] == "2"
    store.fetch_intraday_raw("601857", "20260301", "20260306", 30, adjust="hfq")
    assert fake_bs["adjustflag"] == "1"


def test_default_stays_unadjusted(fake_bs):
    """默认必须仍是不复权——仓库里已入库的全是这个口径。"""
    store.fetch_intraday_raw("601857", "20260301", "20260306", 30)
    assert fake_bs["adjustflag"] == "3"


def test_unknown_adjust_is_rejected(fake_bs):
    with pytest.raises(ValueError, match="未知复权口径"):
        store.fetch_intraday_raw("601857", "20260301", "20260306", 30, adjust="xx")


def test_store_refuses_to_cache_adjusted_bars():
    """qfq 会随分红回溯改写历史，增量追加会把两种口径缝在一起——必须拒绝入库。"""
    with pytest.raises(ValueError, match="只存不复权"):
        store.load_intraday("601857", adjust="qfq")


def test_fetch_shape_is_the_flat_panel(fake_bs):
    df = store.fetch_intraday_raw("601857", "20260301", "20260306", 30)
    assert list(df.columns) == store.COLUMNS
    assert len(df) == 2
    assert df["amount"].gt(0).all()
