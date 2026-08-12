"""场内基金（ETF/LOF）取数通道（不联网：数据源全部打桩）。"""

import sys
import types

import pandas as pd
import pytest

from lib import market_data as md
from lib import price_store as ps


# ── yfinance ticker 后缀 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("symbol,expected", [
    ("510300", "510300.SS"),    # 沪市 ETF：5 开头
    ("518880", "518880.SS"),
    ("159915", "159915.SZ"),    # 深市 ETF：1 开头
    ("161226", "161226.SZ"),
])
def test_fund_ticker_uses_fund_number_ranges(symbol, expected):
    """场内基金号段与个股不同：按个股规则判断会把 510300 错判成深市。"""
    assert md._to_yfinance_ticker(symbol, is_fund=True) == expected


def test_stock_ticker_rules_unchanged():
    assert md._to_yfinance_ticker("600519") == "600519.SS"
    assert md._to_yfinance_ticker("000001") == "000001.SZ"
    assert md._to_yfinance_ticker("399006", is_index=True) == "399006.SZ"


# ── akshare 基金接口 ─────────────────────────────────────────────────────────

def _fake_akshare(monkeypatch, df: pd.DataFrame):
    """把一个只有 fund_etf_hist_em 的假 akshare 塞进 sys.modules。"""
    calls = {}

    def fund_etf_hist_em(symbol, period, start_date, end_date, adjust):
        calls.update(symbol=symbol, adjust=adjust, period=period)
        return df.copy()

    mod = types.ModuleType("akshare")
    mod.fund_etf_hist_em = fund_etf_hist_em
    monkeypatch.setitem(sys.modules, "akshare", mod)
    return calls


def _ak_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03"],
        "开盘": [4.10, 4.05], "收盘": [4.05, 4.04],
        "最高": [4.11, 4.06], "最低": [4.05, 4.03],
        "成交量": [9_429_306, 10_617_503],
        "成交额": [3.27e9, 3.65e9],
    })


def test_etf_columns_renamed_and_indexed(monkeypatch):
    calls = _fake_akshare(monkeypatch, _ak_frame())
    out = md.fetch_etf_data("510300", "20240101", "20240201", adjust="qfq")

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert out["close"].iloc[0] == pytest.approx(4.05)
    # 复权口径必须原样透传给数据源，不能悄悄换成默认值
    assert calls["adjust"] == "qfq"
    assert calls["symbol"] == "510300"


def test_etf_falls_back_to_yfinance_when_akshare_empty(monkeypatch):
    _fake_akshare(monkeypatch, pd.DataFrame())
    seen = {}

    def fake_download(ticker, start, end, **kw):
        seen["ticker"] = ticker
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame({"open": [1.0, 1.1], "high": [1.2, 1.3],
                             "low": [0.9, 1.0], "close": [1.1, 1.2],
                             "volume": [100, 200]}, index=idx)

    monkeypatch.setattr(md, "_yfinance_download", fake_download)
    with pytest.warns(UserWarning):        # hfq 请求退化成 yfinance 的前复权口径
        out = md.fetch_etf_data("510300", "20240101", "20240201", adjust="hfq")
    assert seen["ticker"] == "510300.SS"   # 走的是基金号段，不是个股规则
    assert len(out) == 2


# ── price_store 分发 ─────────────────────────────────────────────────────────

def test_price_store_routes_etf_kind(monkeypatch):
    """kind="etf" 必须走基金接口——个股接口不含场内基金，baostock 更是返回空表。"""
    seen = {}

    def fake_etf(symbol, start, end, adjust=md.DEFAULT_ADJUST, proxy=""):
        seen.update(symbol=symbol, adjust=adjust)
        idx = pd.to_datetime(["2024-01-02"])
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                             "close": [1.0], "volume": [1]}, index=idx)

    def boom(*a, **k):
        raise AssertionError("ETF 不应该走个股/指数通道")

    monkeypatch.setattr(ps, "fetch_etf_data", fake_etf)
    monkeypatch.setattr(ps, "fetch_stock_data", boom)
    monkeypatch.setattr(ps, "fetch_index_data", boom)

    out = ps._fetch("510300", "20240101", "20240201", "etf", "qfq")
    assert not out.empty
    assert seen == {"symbol": "510300", "adjust": "qfq"}
