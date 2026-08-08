"""jcy_intraday_timing 的两条执行口径（2026-08-09 修正）。

① 可成交价 = 首个 GO 柱的**下一根开盘价**，不是 GO 柱自己的收盘价（后者是前视）
② 分时条件只决定「几点买」，不决定「买不买」——无 GO / 无分时数据都必须照常建仓
"""

import numpy as np
import pandas as pd
import pytest

jit = pytest.importorskip("jcy_intraday_timing")
from lib import execution as ex

BARS = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]


def make_exec_bars(closes, opens=None):
    """造单日 8 根分时 K 线，index 为 datetime。"""
    n = len(closes)
    opens = opens if opens is not None else [c * 0.998 for c in closes]
    idx = pd.to_datetime([f"2026-03-05 {BARS[i]}" for i in range(n)])
    return pd.DataFrame({"open": opens, "high": [c * 1.01 for c in closes],
                         "low": [c * 0.99 for c in closes], "close": closes,
                         "volume": [1e5] * n}, index=idx)


def summary_at(bars, pos):
    return jit.TimingSummary(has_go=True, go_times=[bars.index[pos]],
                             first_go=bars.index[pos], go_count=1,
                             total_bars=len(bars))


# ── ① 可成交价口径 ───────────────────────────────────────────────────────────

def test_exec_price_is_next_bar_open_not_go_bar_close():
    bars = make_exec_bars([10.0, 10.2, 10.5, 10.4, 10.6, 10.8, 10.7, 10.9],
                          opens=[9.9, 10.1, 10.3, 10.45, 10.5, 10.7, 10.75, 10.85])
    px = jit._executable_price(bars, summary_at(bars, 2))
    assert px == pytest.approx(10.45)          # 第 4 根的开盘价
    assert px != pytest.approx(10.5)           # 不是 GO 柱(第3根)的收盘价


def test_exec_price_falls_back_to_close_when_go_is_last_bar():
    bars = make_exec_bars([10.0, 10.2, 10.5, 10.4, 10.6, 10.8, 10.7, 10.9])
    px = jit._executable_price(bars, summary_at(bars, len(bars) - 1))
    assert px == pytest.approx(10.9)           # 身后没有 K 线了，只能收盘价


def test_exec_price_falls_back_to_close_when_no_go():
    bars = make_exec_bars([10.0, 10.2, 10.5, 10.4, 10.6, 10.8, 10.7, 10.9])
    s = jit.TimingSummary(has_go=False, go_times=[], first_go=None,
                          go_count=0, total_bars=len(bars))
    assert jit._executable_price(bars, s) == pytest.approx(10.9)


def test_exec_price_matches_lib_execution_go_price():
    """与 lib.execution.daily_panel 的 go_price 必须同口径，防两处漂移。"""
    rng = np.random.default_rng(4)
    rows, px = [], 10.0
    for d in pd.bdate_range("2026-01-01", periods=40):
        for hm in BARS:
            px *= 1 + rng.normal(0, 0.004)
            vol = 1e5
            rows.append({"dt": pd.Timestamp(f"{d.date()} {hm}"), "date": d,
                         "open": px * 0.999, "high": px * 1.004, "low": px * 0.996,
                         "close": px, "volume": vol, "amount": px * vol})
    flat = pd.DataFrame(rows)
    panel = ex.daily_panel(ex.intraday_macd(flat), warmup_bars=0)

    wide = flat.set_index("dt")[["open", "high", "low", "close", "volume"]]
    wide = jit.add_macd(wide)

    checked = 0
    for _, r in panel.iterrows():
        bars = jit.classify_timing(wide[wide.index.normalize() == r["date"]], "buy")
        mine = jit._executable_price(bars, jit.summarize_timing(bars))
        assert mine == pytest.approx(r["go_price"]), f"{r['date']} 口径不一致"
        checked += 1
    assert checked >= 30


# ── ② 分时不得决定是否建仓 ───────────────────────────────────────────────────

def _df_sig(n=6, close=10.0, closes=None):
    idx = pd.bdate_range("2026-03-02", periods=n)
    return pd.DataFrame({
        "close": closes if closes is not None else [close] * n,
        "DIF": [0.5] * n, "DEA": [0.2] * n,
        "signal": [1] + [0] * (n - 1),
        "hist_expanding": [False] * n, "hist_shrinking": [False] * n,
    }, index=idx)


def test_buys_at_given_exec_price():
    df = _df_sig()
    d0 = df.index[0]
    t = jit.PositionTracker(capital=100_000)
    t.run(df, {d0: {"exec_date": d0, "exec_price": 9.5, "action": "buy", "dif": 0.5}})
    buys = [x for x in t.trades if x.action == "初仓"]
    assert len(buys) == 1
    assert buys[0].price == pytest.approx(9.5)


def test_missing_intraday_price_still_buys_using_daily_close():
    """exec_price=None（分时数据缺失）不能再让这一笔凭空消失。"""
    df = _df_sig(close=12.0)
    d0 = df.index[0]
    t = jit.PositionTracker(capital=100_000)
    t.run(df, {d0: {"exec_date": d0, "exec_price": None, "action": "buy", "dif": 0.5}})
    buys = [x for x in t.trades if x.action == "初仓"]
    assert len(buys) == 1, "分时数据缺失不该改变是否建仓"
    assert buys[0].price == pytest.approx(12.0)      # 兜底走日线收盘价


def test_pending_buy_uses_exec_day_close_not_signal_day_close():
    """兜底价要取执行日的收盘价——拿信号日的价去成交次日的单是错配。"""
    df = _df_sig(closes=[8.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    d0, d1 = df.index[0], df.index[1]
    t = jit.PositionTracker(capital=100_000)
    t.run(df, {d0: {"exec_date": d1, "exec_price": None, "action": "buy", "dif": 0.5}})
    buys = [x for x in t.trades if x.action == "初仓"]
    assert len(buys) == 1
    assert buys[0].date == d1
    assert buys[0].price == pytest.approx(9.0)      # 执行日价，不是信号日的 8.0
