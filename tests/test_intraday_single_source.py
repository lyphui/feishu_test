"""
分时取数只能有**一处**实现。

`backtest_jcy_intraday` 曾经自带一份 `_fetch_intraday_baostock`（以及
akshare/baostock 回退 + 重试的 `fetch_intraday`），与
`lib/intraday_store.fetch_intraday_raw` 是两份独立代码，而且**复权口径还不一样**
（前者 qfq、后者 none），靠人看着两处维持一致。两个口径本身都对——分时 MACD 要
复权（不复权在除权日有假跳空），VWAP 基准要不复权（amount/volume 恒为原始值）——
所以正确形态是同一个函数的两种参数，而不是两份实现。现在脚本的 `fetch_intraday`
整段已删，改用 `store.fetch_intraday_indexed`（= `fetch_intraday_raw` + 索引转换）。

这些测试全部离线：用假的 baostock 模块拦住网络调用，只检查参数与形状。
"""

import sys
import types

import pandas as pd
import pytest

from backtest.lib import intraday_store as store
from backtest.scripts import backtest_jcy_intraday as jit


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

def test_script_no_longer_has_its_own_fetcher():
    """脚本不得再自带分时取数实现——akshare/baostock 回退那一套已删除。"""
    assert not hasattr(jit, "_fetch_intraday_akshare")
    assert not hasattr(jit, "_fetch_intraday_baostock")
    assert not hasattr(jit, "fetch_intraday_raw"), \
        "脚本里不应再出现独立的 fetch_intraday_raw——应当复用 intraday_store"


def test_script_delegates_to_the_shared_fetcher():
    """脚本用的 `fetch_intraday_indexed` 必须就是 intraday_store 那一个对象。"""
    assert jit.fetch_intraday_indexed is store.fetch_intraday_indexed


def test_fetch_intraday_indexed_returns_indexed_panel(fake_bs):
    """`fetch_intraday_indexed` 默认 qfq（MACD 要复权），返回 datetime 索引宽表。"""
    out = store.fetch_intraday_indexed("601857", "20260301", "20260306", 30)
    assert fake_bs["adjustflag"] == "2"            # qfq
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2


def test_fetch_intraday_indexed_passes_none_adjust_through(fake_bs):
    """不复权口径（VWAP 测算用）也允许显式传。"""
    store.fetch_intraday_indexed("601857", "20260301", "20260306", 30, adjust="none")
    assert fake_bs["adjustflag"] == "3"


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
