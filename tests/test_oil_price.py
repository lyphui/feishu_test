"""oil_price 存取与传导性分析（不联网：抓取函数打桩，transmission_table 用合成数据）。"""

import json

import numpy as np
import pandas as pd
import pytest

from backtest.lib import oil_price as op


def _oil_bars(start: str, n: int, base: float = 80.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = base + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": 1000,
    }, index=idx)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "OIL_DIR", str(tmp_path))
    return tmp_path


def test_update_oil_writes_csv_and_meta(store, monkeypatch):
    bars = _oil_bars("2024-01-01", 50)
    monkeypatch.setattr(op, "fetch_oil_price", lambda symbol: bars)

    df = op.update_oil("WTI", verbose=False)

    assert len(df) == 50
    assert op.oil_path("WTI").startswith(str(store))
    with open(op.meta_path("WTI"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["rows"] == 50
    assert meta["data_start"] == "20240101"


def test_load_oil_offline_reads_local_cache(store, monkeypatch):
    bars = _oil_bars("2024-01-01", 30)
    monkeypatch.setattr(op, "fetch_oil_price", lambda symbol: bars)
    op.update_oil("BRENT", verbose=False)

    df = op.load_oil("BRENT", auto_update=False)
    assert len(df) == 30


def test_load_oil_offline_missing_raises(store):
    with pytest.raises(FileNotFoundError):
        op.load_oil("SC", auto_update=False)


def test_fetch_oil_price_unknown_symbol_raises():
    with pytest.raises(ValueError):
        op.fetch_oil_price("NOT_A_SYMBOL")


def test_transmission_table_finds_planted_lag():
    """油价领先股价 3 天的合成数据，lag=3 处的相关系数应明显高于其它 lag。"""
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(42)
    oil_ret = rng.normal(0, 0.01, len(idx))
    oil_close = 80 * np.exp(np.cumsum(oil_ret))
    # 股票收益 = 3 天前的油价收益 * 2 + 噪音
    stock_ret = np.zeros(len(idx))
    stock_ret[3:] = oil_ret[:-3] * 2 + rng.normal(0, 0.02, len(idx) - 3)
    stock_close = 10 * np.exp(np.cumsum(stock_ret))

    oil = pd.DataFrame({"close": oil_close}, index=idx)
    stock = pd.DataFrame({"close": stock_close}, index=idx)

    t = op.transmission_table(oil, stock, lags=(0, 1, 2, 3, 5, 10))
    best = t.loc[t["corr"].idxmax()]
    assert best["lag_days"] == 3
    assert best["corr"] > 0.5


def test_fetch_rejects_missing_close_column():
    """列名变了要当场报错，不能补 NA 让 close 整列变空、最后静默无输出。"""
    import akshare as ak
    bad = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=40),
                        "open": 80.0, "closing_price": 81.0})
    orig = getattr(ak, "futures_foreign_hist", None)
    try:
        ak.futures_foreign_hist = lambda symbol: bad
        with pytest.raises(RuntimeError, match="close"):
            op.fetch_oil_price("WTI")
    finally:
        if orig is not None:
            ak.futures_foreign_hist = orig


def test_update_refuses_truncated_response(store, monkeypatch):
    """数据源半残只吐回几百行时，不能把本地几千行历史覆盖掉。"""
    full = _oil_bars("2020-01-01", 500)
    monkeypatch.setattr(op, "fetch_oil_price", lambda symbol: full)
    op.update_oil("WTI", verbose=False)

    monkeypatch.setattr(op, "fetch_oil_price", lambda symbol: full.tail(50))
    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        op.update_oil("WTI", verbose=False)
    assert len(op.read_oil("WTI")) == 500          # 本地未被破坏


def test_update_refuses_rewritten_history(store, monkeypatch):
    """上游换了合约拼接口径（历史价整段变了）时拒绝覆盖。"""
    full = _oil_bars("2020-01-01", 500, base=80.0)
    monkeypatch.setattr(op, "fetch_oil_price", lambda symbol: full)
    op.update_oil("SC", verbose=False)

    monkeypatch.setattr(op, "fetch_oil_price",
                        lambda symbol: _oil_bars("2020-01-01", 500, base=200.0))
    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        op.update_oil("SC", verbose=False)


def test_force_allows_rebuild(store, monkeypatch):
    full = _oil_bars("2020-01-01", 500, base=80.0)
    monkeypatch.setattr(op, "fetch_oil_price", lambda symbol: full)
    op.update_oil("SC", verbose=False)

    monkeypatch.setattr(op, "fetch_oil_price",
                        lambda symbol: _oil_bars("2020-01-01", 500, base=200.0))
    op.update_oil("SC", force=True, verbose=False)
    assert op.read_oil("SC")["close"].mean() > 150


def test_stale_oil_cache_shrinks_sample_not_just_correlation():
    """
    油价缓存过期时，超出容差的交易日必须被剔除（n 下降），
    而不是靠 merge_asof 前向填充悄悄制造一堆 0 收益率。
    """
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(7)
    oil = pd.DataFrame({"close": 80 + np.cumsum(rng.normal(0, 0.5, 400))}, index=idx)
    stock = pd.DataFrame({"close": 10 + np.cumsum(rng.normal(0, 0.1, 400))}, index=idx)

    full_n = op.transmission_table(oil, stock, lags=(1,))["n"].iloc[0]
    stale_n = op.transmission_table(oil.iloc[:300], stock, lags=(1,))["n"].iloc[0]
    assert stale_n < full_n - 50


def test_transmission_table_flags_noise_lags():
    """纯噪声的 lag 必须被标成 signif=False，不能让人当成微弱传导。"""
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(3)
    oil = pd.DataFrame({"close": 80 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))}, index=idx)
    stock = pd.DataFrame({"close": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))}, index=idx)

    t = op.transmission_table(oil, stock)
    assert not t["signif"].any(), "不相关的两条序列不应出现显著 lag"
    assert (t["ci95"] > 0).all()


def test_transmission_table_empty_input_returns_empty_df():
    empty = pd.DataFrame(columns=["close"])
    idx = pd.bdate_range("2020-01-01", periods=10)
    stock = pd.DataFrame({"close": np.arange(10) + 10.0}, index=idx)
    assert op.transmission_table(empty, stock).empty


def test_transmission_table_aligns_different_calendars():
    """油价与股价交易日历不完全重合（周末/节假日不同）时仍能算出结果。"""
    oil_idx = pd.date_range("2020-01-01", periods=60, freq="D")  # 每天都有报价
    stock_idx = pd.bdate_range("2020-01-01", periods=45)          # 只有工作日

    rng = np.random.default_rng(1)
    oil = pd.DataFrame({"close": 80 + np.cumsum(rng.normal(0, 0.3, len(oil_idx)))},
                       index=oil_idx)
    stock = pd.DataFrame({"close": 10 + np.cumsum(rng.normal(0, 0.1, len(stock_idx)))},
                         index=stock_idx)

    t = op.transmission_table(oil, stock, lags=(0, 1))
    assert not t.empty
    assert (t["n"] > 0).all()
