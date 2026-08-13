"""乌龙指捕捉策略模拟器（离线，全部用合成行情）。

守住的是三件会直接翻转结论的事：挂单价受涨跌停约束、成交价是自己挂的那个价
（跳空时才改善为开盘价）、买入当日不得卖出。
"""

import numpy as np
import pandas as pd
import pytest

from backtest.lib.fatfinger import fill_edge, simulate_fatfinger, simulate_static_mix


def make_df(rows, start="2020-01-01") -> pd.DataFrame:
    """rows = [(open, high, low, close, volume?), ...]"""
    idx = pd.bdate_range(start, periods=len(rows))
    data = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for r in rows:
        o, h, l, c = r[:4]
        data["open"].append(o); data["high"].append(h)
        data["low"].append(l); data["close"].append(c)
        data["volume"].append(r[4] if len(r) > 4 else 1_000_000)
    return pd.DataFrame(data, index=idx, dtype=float)


def flat(px, n, high_mult=1.001, low_mult=0.999):
    return [(px, px * high_mult, px * low_mult, px)] * n


# ── 挂单可行性：涨跌停 ────────────────────────────────────────────────────────

def test_orders_beyond_price_limit_are_rejected():
    """±30% 的单子在价格平稳时根本挂不出去，不是「挂着没成交」。"""
    df = make_df(flat(100.0, 60, 1.09, 0.91))     # 日内摆幅 ±9%，够不到 ±30%
    r = simulate_fatfinger(df, 100_000, k_up=0.30, k_dn=0.30)
    assert r.diag["n_fills"] == 0
    assert r.diag["n_rejected_sell_days"] > 0
    assert r.diag["n_rejected_buy_days"] > 0


def test_wide_order_becomes_placeable_after_price_drifts():
    """锚价固定、股价自己走出趋势之后，同一个 k 相对新前收就落回涨跌停带内了。

    这正是 ±30% 在实测里仍有零星成交的原因——成交的是趋势，不是谁敲错。
    """
    # 从 100 一路涨到 145：锚锁在 ~100，前收涨上去后 130 的卖单就挂得出去了
    px = list(np.linspace(100, 145, 80))
    df = make_df([(p, p * 1.005, p * 0.995, p) for p in px])
    r = simulate_fatfinger(df, 100_000, k_up=0.30, k_dn=0.30)
    assert r.diag["n_sell"] >= 1
    # 成交那天股价确实已经涨到 130 附近，而不是从 100 直接被敲到 130
    assert float(r.fills["fill"].iloc[0]) == pytest.approx(130, rel=0.05)


# ── 撮合与成交价 ──────────────────────────────────────────────────────────────

def test_fill_price_is_the_limit_price_not_the_high():
    """限价单成交价就是自己报的价，不能拿当日最高价当成交价。"""
    rows = flat(100.0, 3) + [(100.0, 108.0, 99.5, 101.0)] + flat(101.0, 3)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05)
    sells = r.fills[r.fills["side"] == "sell"]
    assert len(sells) == 1
    anchor = float(sells["anchor"].iloc[0])
    assert float(sells["fill"].iloc[0]) == pytest.approx(anchor * 1.05)


def test_gap_through_order_fills_at_the_open():
    """跳空高开越过挂单价时成交在开盘价（价格改善），不是仍按挂单价。"""
    rows = flat(100.0, 3) + [(107.0, 108.0, 106.0, 107.0)] + flat(107.0, 3)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05)
    sells = r.fills[r.fills["side"] == "sell"]
    assert len(sells) == 1
    assert float(sells["fill"].iloc[0]) == pytest.approx(107.0)


def test_halted_day_does_not_fill():
    """停牌（volume==0）当天不成交，哪怕 OHLC 看上去够到了挂单价。"""
    rows = flat(100.0, 3) + [(100.0, 120.0, 80.0, 100.0, 0)] + flat(100.0, 3)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05)
    assert r.diag["n_fills"] == 0


def test_both_sides_same_day_fills_at_most_once():
    """日线看不出先后，两侧同日触发时只能认一笔，不能自成一对无风险套利。"""
    rows = flat(100.0, 3) + [(100.0, 112.0, 88.0, 100.0)] + flat(100.0, 3)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05)
    assert r.diag["n_both_sides_days"] == 1
    assert r.diag["n_fills"] == 1


# ── 仓位与 T+1 ────────────────────────────────────────────────────────────────

def test_returns_to_target_mix_after_a_fill():
    """成交后要回到 50/50，而不是停在 100% 现金或 100% 持仓。"""
    rows = flat(100.0, 3) + [(100.0, 108.0, 99.5, 106.0)] + flat(106.0, 10)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05)
    assert r.diag["n_sell"] == 1
    assert float(r.result.exposure.iloc[-1]) == pytest.approx(0.5, abs=0.02)


def test_buy_fill_cannot_be_sold_same_day_even_in_fast_mode():
    """A 股当日买入不可卖出：买单成交日不得出现卖出腿。"""
    rows = flat(100.0, 3) + [(100.0, 100.5, 92.0, 94.0)] + flat(94.0, 5)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05,
                           fast_sell_rebalance=True)
    buys = r.fills[r.fills["side"] == "buy"]
    assert len(buys) == 1
    fill_date = buys["date"].iloc[0]
    same_day = [t for t in r.result.trades
                if t["date"] == fill_date and t["action"] == "sell"]
    assert same_day == []


def test_no_negative_cash_or_shares():
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    df = make_df([(p, p * 1.03, p * 0.97, p) for p in px])
    r = simulate_fatfinger(df, 100_000, k_up=0.03, k_dn=0.03)
    assert (r.result.exposure >= -1e-9).all()
    assert (r.result.exposure <= 1.0 + 1e-6).all()
    assert (r.result.equity > 0).all()


# ── 成交质量指标 ──────────────────────────────────────────────────────────────

def test_edge_is_positive_when_the_spike_snaps_back():
    """尖峰被打回 → 卖单 edge 为正；这才叫「捡到乌龙指」。"""
    rows = flat(100.0, 3) + [(100.0, 108.0, 99.5, 100.0)] + flat(100.0, 5)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05)
    e = fill_edge(r.fills)
    assert float(e.loc[e["side"] == "sell", "edge_mean_bp"].iloc[0]) > 0


def test_edge_is_negative_when_the_move_is_real():
    """价格冲上去就不回来 → 卖单 edge 为负，是被趋势带走的，不是捡漏。"""
    rows = flat(100.0, 3) + [(100.0, 108.0, 99.5, 108.0)] + flat(112.0, 5)
    r = simulate_fatfinger(make_df(rows), 100_000, k_up=0.05, k_dn=0.05)
    e = fill_edge(r.fills)
    assert float(e.loc[e["side"] == "sell", "edge_mean_bp"].iloc[0]) < 0


# ── 对照组 ────────────────────────────────────────────────────────────────────

def test_entry_is_not_filled_on_the_first_bar():
    """首根 K 线只做决策，成交排到下一根开盘（T+1，同 ladder._run）。

    不这样做的话，首根开盘价会被当成可成交价——对新股来说那是上市当日的开盘，
    白拿一段涨幅，基准被抬高之后任何策略的「超额」都是负的。
    """
    rows = [(10.0, 10.0, 10.0, 10.0)] + flat(20.0, 20)
    for r in (simulate_static_mix(make_df(rows), 100_000, target=0.5),
              simulate_fatfinger(make_df(rows), 100_000,
                                 k_up=0.5, k_dn=0.5).result):
        assert r.trades[0]["price"] == pytest.approx(20.0, rel=0.01)


def test_static_mix_lies_flat():
    df = make_df(flat(100.0, 200, 1.05, 0.95))
    r = simulate_static_mix(df, 100_000, target=0.5)
    assert r.stats["n_trades"] == 1
    assert r.stats["avg_exposure"] == pytest.approx(0.5, abs=0.03)


def test_static_mix_rebalances_on_schedule():
    # 用 10 元的票：一手 1000 元，10 万本金下整手粒度才细到能看出定期再平衡。
    # 100 元的票一手就 1 万，半手的死区比 21 天的漂移还大，本来就不该动。
    px = list(np.linspace(10, 20, 210))
    df = make_df([(p, p * 1.01, p * 0.99, p) for p in px])
    r = simulate_static_mix(df, 100_000, target=0.5, rebalance_days=21)
    assert r.stats["n_trades"] > 5
    # 定期再平衡的意义就是把仓位钉住，不让它随涨幅漂上去
    assert r.stats["avg_exposure"] == pytest.approx(0.5, abs=0.03)
