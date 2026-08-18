"""全收益指数取数（评审项 5）的归一化守护：mock akshare，不依赖网络。"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest.lib.market_data import fetch_index_tr_data

#: ak.stock_zh_index_hist_csindex 的真实列名（2026-08 实测）。
#: 测试**必须**用真名而不是 a..p 占位：取数早先按位置赋 16 个列名，
#: 用占位名的 mock 永远是绿的，而线上换一次列序就会把「涨跌幅」静默
#: 映射成 close——数值仍是合理量级，没有任何地方会报错。
CSINDEX_COLUMNS = [
    "日期", "指数代码", "指数中文全称", "指数中文简称",
    "指数英文全称", "指数英文简称",
    "开盘", "最高", "最低", "收盘", "涨跌", "涨跌幅",
    "成交量", "成交金额", "样本数量", "滚动市盈率",
]


def _row(date, o, h, l, c, chg, vol):
    return [date, "H00300", "沪深300全收益", "300收益", "x", "x",
            o, h, l, c, chg, 0.0, vol, 1e9, 300, 12.5]


def _fake_csindex(**kw):
    """三根正常 K 线 + 一根休市补行（与前一行 close/涨跌/成交量逐位相同）。"""
    rows = [
        _row("2026-08-05", 5800.1, 5850.2, 5790.0, 5840.5, 1.0, 12345),
        _row("2026-08-06", 5840.5, 5860.0, 5800.0, 5810.3, -30.2, 12346),
        # 休市补行：中证会把上一交易日整行重复一遍（实测 2026-08-01 周六）
        _row("2026-08-07", 5810.3, 5860.0, 5800.0, 5810.3, -30.2, 12346),
        _row("2026-08-10", 5810.3, 5900.0, 5805.0, 5895.8, 85.5, 12347),
    ]
    return pd.DataFrame(rows, columns=CSINDEX_COLUMNS)


def _fake_csindex_close_only(**kw):
    """该源多数交易日不发布 OHL（实测 87.8% 的行 open/high/low 为 NaN）。"""
    rows = [
        _row("2026-08-05", None, None, None, 5840.5, 1.0, 12345),
        _row("2026-08-06", None, None, None, 5810.3, -30.2, 12346),
    ]
    return pd.DataFrame(rows, columns=CSINDEX_COLUMNS)


def _mock(monkeypatch, side_effect):
    ak = MagicMock()
    ak.stock_zh_index_hist_csindex.side_effect = side_effect
    monkeypatch.setitem(sys.modules, "akshare", ak)
    return ak


@pytest.fixture
def mock_akshare(monkeypatch):
    return _mock(monkeypatch, _fake_csindex)


def test_fetch_index_tr_normalizes_to_ohlcv(mock_akshare):
    df = fetch_index_tr_data("H00300", "20260801", "20260810")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert str(df.index[0].date()) == "2026-08-05"
    assert df["close"].iloc[-1] == pytest.approx(5895.8)
    # 调用参数必须原样透传给 akshare（日期格式 YYYYMMDD）
    _, kwargs = mock_akshare.stock_zh_index_hist_csindex.call_args
    assert kwargs == {"symbol": "H00300", "start_date": "20260801",
                      "end_date": "20260810"}


def test_padded_non_trading_rows_are_dropped(mock_akshare):
    """
    休市补行必须丢掉，不能进日线仓库。

    识别依据是 (close, 涨跌, 成交量) 与上一行**逐位相同**——真实平盘日的
    「涨跌」是 0，不会重复上一行的非零值。
    """
    df = fetch_index_tr_data("H00300", "20260801", "20260810")
    assert [str(d.date()) for d in df.index] == [
        "2026-08-05", "2026-08-06", "2026-08-10"]        # 08-07 补行被丢


def test_ohl_filled_from_close_when_source_omits_them(monkeypatch):
    """
    该源只发布收盘价时，OHL 填成 close 而不是留 NaN。

    留 NaN 会在 `data/market/daily/` 里放一份「只有 close 是真的」的日线，
    而 `price_store._overlap_matches` 只对账 close，永远发现不了；下游
    `costs.tradability` 读 `row["open"]` 会直接拿到 NaN。
    """
    _mock(monkeypatch, _fake_csindex_close_only)
    df = fetch_index_tr_data("H00300", "20260801", "20260807")
    assert df[["open", "high", "low", "close"]].notna().all().all()
    for col in ("open", "high", "low"):
        assert (df[col] == df["close"]).all()


def test_unexpected_columns_fail_loudly(monkeypatch):
    """
    列名对不上必须**报错**，不能静默错位。

    这是改用中文列名 rename 的全部理由：按位置赋名时，源换一次列序就会把
    「涨跌幅」映射成 close，量级看着还挺合理，没有任何地方会响。
    """
    def _renamed(**kw):
        df = _fake_csindex()
        return df.rename(columns={"收盘": "收市价"})
    _mock(monkeypatch, _renamed)
    with pytest.raises(ValueError, match="收盘"):
        fetch_index_tr_data("H00300", "20260801", "20260810")


def test_source_is_tagged(mock_akshare):
    """meta.json 要靠 attrs["source"] 记下数据源（评审项 6）。"""
    df = fetch_index_tr_data("H00300", "20260801", "20260810")
    assert df.attrs["source"] == "csindex"


def test_price_store_dispatches_index_tr(monkeypatch):
    """price_store._fetch 的 kind='index_tr' 必须路由到全收益取数。"""
    from backtest.lib import price_store
    sentinel = pd.DataFrame({"close": [1.0]},
                            index=pd.to_datetime(["2026-08-07"]))
    monkeypatch.setattr("backtest.lib.price_store.fetch_index_tr_data",
                        lambda *a, **k: sentinel)
    out = price_store._fetch("H00300", "20260801", "20260807", "index_tr", "none")
    assert not out.empty
