"""
取数源顺序的守护：个股必须 baostock 优先（评审后续，2026-08）。

为什么值得一条测试
------------------
源顺序是个**口径决定**，不是实现细节：东财的 hfq 与 baostock 的分红再投累积
方式不同，601225 实测 8.6 年年化差 4.1pp。谁排第一，决定了整个仓库的 hfq
是不是真·全收益口径。而顺序又恰恰是最容易在"顺手调一下回退逻辑"时被改掉的
东西——改回去不会有任何报错，只会让以后新抓的标的与存量不同口径。
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest.lib import market_data


def _bs_frame():
    return pd.DataFrame({
        "date": ["2026-08-05", "2026-08-06"],
        "open": [10.0, 10.5], "high": [10.8, 10.9],
        "low": [9.9, 10.2], "close": [10.5, 10.7],
        "volume": [1000.0, 1100.0],
    })


def _ak_frame():
    return pd.DataFrame({
        "日期": ["2026-08-05", "2026-08-06"],
        "开盘": [1.0, 1.0], "收盘": [1.0, 1.0],
        "最高": [1.0, 1.0], "最低": [1.0, 1.0],
        "成交量": [1.0, 1.0], "成交额": [1.0, 1.0],
    })


@pytest.fixture
def both_sources(monkeypatch):
    """两个源都可用且返回**可区分**的数据，看实际命中哪个。"""
    calls = []

    def fake_bs_query(code, start, end, frequency="d", fields=None, adjustflag="2"):
        calls.append(("baostock", code, adjustflag))
        return _bs_frame()

    monkeypatch.setattr(market_data, "_baostock_query", fake_bs_query)

    ak = MagicMock()
    ak.stock_zh_a_hist.side_effect = lambda **kw: (calls.append(("akshare", kw["symbol"], kw["adjust"]))
                                                   or _ak_frame())
    monkeypatch.setitem(sys.modules, "akshare", ak)
    return calls


def test_stock_prefers_baostock(both_sources):
    """
    个股：baostock 必须**先**被调用，且 akshare 完全不被碰。

    东财 hfq 不是全收益口径（601225：年化 +18.17% vs 真值 +22.27%），
    所以只要 baostock 有数据就不该退到 akshare。
    """
    df = market_data.fetch_stock_data("601225", "20260801", "20260807")
    assert [c[0] for c in both_sources] == ["baostock"]
    assert df.attrs["source"] == "baostock"
    assert df["close"].iloc[-1] == pytest.approx(10.7)      # 来自 baostock 那份


def test_stock_hfq_maps_to_baostock_adjustflag_1(both_sources):
    """hfq 必须映射成 baostock 的 adjustflag=1，别退化成前复权。"""
    market_data.fetch_stock_data("601225", "20260801", "20260807", adjust="hfq")
    assert both_sources[0][2] == market_data.ADJUST["hfq"]["baostock"] == "1"


def test_stock_falls_back_to_akshare_with_caliber_warning(monkeypatch):
    """
    baostock 返回空 → 退 akshare，但必须**告警**：该票与仓库其余标的不同口径。
    """
    monkeypatch.setattr(market_data, "_baostock_query",
                        lambda *a, **k: pd.DataFrame())
    ak = MagicMock()
    ak.stock_zh_a_hist.side_effect = lambda **kw: _ak_frame()
    monkeypatch.setitem(sys.modules, "akshare", ak)

    with pytest.warns(UserWarning, match="baostock 不同"):
        df = market_data.fetch_stock_data("601225", "20260801", "20260807")
    assert df.attrs["source"] == "akshare"


def test_index_still_prefers_akshare(monkeypatch):
    """
    指数**不跟着翻**：无复权概念，且 akshare 指数接口走新浪不经东财，
    本机可达、两源实测逐位一致。翻了只会平白引入变化。
    """
    calls = []
    monkeypatch.setattr(market_data, "_baostock_query",
                        lambda *a, **k: calls.append("baostock") or _bs_frame())
    ak = MagicMock()
    ak.stock_zh_index_daily.side_effect = lambda **kw: (
        calls.append("akshare") or pd.DataFrame({
            "date": ["2026-08-05", "2026-08-06"],
            "open": [1.0, 1.0], "high": [1.0, 1.0],
            "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1.0, 1.0],
        }))
    monkeypatch.setitem(sys.modules, "akshare", ak)

    df = market_data.fetch_index_data("000300", "20260801", "20260807")
    assert calls == ["akshare"]
    assert df.attrs["source"] == "akshare"
