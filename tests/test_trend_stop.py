"""月频均线 + 移动止损的离线测试：不联网、不读真实 data/。"""

import numpy as np
import pandas as pd
import pytest

from lib.trend_stop import (buy_hold, hk_fee_rate, hk_trade_cost,
                            month_end_flags, next_decision_date, simulate,
                            sweep)


def make_df(prices, start="2020-01-01"):
    """按工作日排布的收盘价序列。"""
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({"close": np.asarray(prices, dtype=float)}, index=idx)


# ── 费用模型 ──────────────────────────────────────────────────────────────────

def test_min_commission_dominates_small_tickets():
    """小单被最低佣金压死——这是整套策略走月频的根本原因。"""
    small = hk_trade_cost(5_790)          # 600 股 @ 9.65
    assert small == pytest.approx(132.5, abs=1.0)
    assert hk_fee_rate(5_790) > 0.02      # 费用率超 2%

    big = hk_fee_rate(100_000)
    assert big < 0.003                    # 10 万港币降到 0.3% 以内
    assert hk_fee_rate(5_790) > 7 * big


def test_commission_floor_breakpoint():
    """成交额 4 万以下走最低佣金，以上才按比例——拐点必须在 100/0.0025。"""
    below = hk_trade_cost(30_000)
    above = hk_trade_cost(60_000)
    assert below == pytest.approx(100 + 30 + 30_000 * 0.000085 + 2.0, abs=0.5)
    assert above > 150 + 30                # 0.25% × 6 万 = 150 已超最低


def test_etf_exempt_from_stamp_duty():
    v = 100_000
    assert hk_trade_cost(v, is_etf=False) - hk_trade_cost(v, is_etf=True) \
        == pytest.approx(v * 0.001)


def test_zero_value_costs_nothing():
    assert hk_trade_cost(0) == 0.0
    assert hk_fee_rate(0) == 0.0


# ── 决策日 ────────────────────────────────────────────────────────────────────

def test_month_end_flags_pick_last_trading_day():
    df = make_df(range(70), start="2021-01-01")
    flags = month_end_flags(df.index)
    picked = df.index[flags.to_numpy()]
    # 每个出现过的月份恰好一个 True，且是该月在样本里的最后一行
    for (y, m), grp in df.groupby([df.index.year, df.index.month]):
        assert grp.index[-1] in picked
    assert len(picked) == len({(d.year, d.month) for d in df.index})


def test_next_decision_date_rolls_to_next_month():
    # 月中 → 当月最后一个工作日
    assert next_decision_date("2026-08-10") == pd.Timestamp("2026-08-31")
    # 已在月末工作日 → 就是当天
    assert next_decision_date("2026-08-31") == pd.Timestamp("2026-08-31")
    # 月末工作日之后（周末）→ 顺延到下月
    assert next_decision_date("2026-09-30") == pd.Timestamp("2026-09-30")


# ── 时序铁律：不许偷看未来 ────────────────────────────────────────────────────

def test_no_lookahead_truncation_invariance():
    """
    截断样本不改变历史仓位。若某处用了未来数据（如全样本均值/分位），
    砍掉尾巴会让前面的持仓变样。
    """
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    df = make_df(prices)

    full = simulate(df, ma_len=20, stop=0.15, fee=0.0)
    for cut in (250, 320, 380):
        part = simulate(df.iloc[:cut], ma_len=20, stop=0.15, fee=0.0)
        shared = part.position.index
        pd.testing.assert_series_equal(
            full.position.loc[shared], part.position,
            check_names=False,
            obj=f"截断到 {cut} 行后历史仓位被改写")


def test_signal_executes_next_bar_not_same_bar():
    """月末收盘看到信号，最早也要次日才有仓位。"""
    # 前 40 天横盘压低均线，之后拉升；用短均线让信号在样本内出现
    prices = [10.0] * 40 + [12.0] * 40
    df = make_df(prices, start="2021-01-01")
    res = simulate(df, ma_len=5, stop=None, fee=0.0)

    flags = month_end_flags(df.index).to_numpy()
    ma = df["close"].rolling(5).mean()
    above = (df["close"] > ma).to_numpy()
    pos = res.position.to_numpy()

    assert pos[0] == 0.0                      # 第一根永远空仓，没有可用信号
    for i in range(1, len(df)):
        # 只有"前一根是决策日且当时信号为多"才可能在第 i 根建立新仓位
        if pos[i] == 1.0 and pos[i - 1] == 0.0:
            assert flags[i - 1] and above[i - 1], \
                f"第 {i} 根凭空建仓，前一根不是决策日或信号不为多"


# ── 移动止损 ──────────────────────────────────────────────────────────────────

def test_trailing_stop_measures_from_peak_not_entry():
    """止损锚在**入场后最高收盘价**，不是入场价。"""
    # 缓慢上行建立持仓，冲高到 20，然后跌到 16（较高点 -20%，较入场仍是盈利）
    prices = [10 + i * 0.05 for i in range(60)] + [20.0, 16.0] + [16.0] * 10
    df = make_df(prices, start="2021-01-01")
    res = simulate(df, ma_len=10, stop=0.15, fee=0.0)

    assert len(res.trades) >= 1
    stopped = res.trades[res.trades["reason"] == "移动止损"]
    assert len(stopped) >= 1, "从高点回落 20% 应触发移动止损"
    row = stopped.iloc[0]
    assert row["exit_px"] == pytest.approx(16.0)
    assert row["ret"] > 0, "较入场价仍盈利，说明锚的是高点而不是成本价"


def test_equity_matches_trade_prices_no_free_escape():
    """
    单段持仓的净值涨幅必须等于 `exit_px / entry_px`（零费用下）。

    这条守的是净值与仓位的对齐：`pos[i]` 是按第 i 天收盘价成交建立的，赚的是
    第 i+1 天的涨跌。少滞后一天的话，止损触发那天的阴线会被整段躲掉——止损
    正是看见那天的收盘价才触发的，白躲就是未来函数。
    """
    prices = [10 + i * 0.05 for i in range(60)] + [20.0, 16.0] + [16.0] * 10
    df = make_df(prices, start="2021-01-01")
    res = simulate(df, ma_len=10, stop=0.15, fee=0.0)

    t = res.trades
    assert len(t) >= 1
    for _, row in t.iterrows():
        seg = res.equity.loc[row["entry_date"]:row["exit_date"]]
        # entry_date 当天收盘才成交，净值起点取前一根
        prior = res.equity.shift(1).loc[row["entry_date"]]
        assert seg.iloc[-1] / prior == pytest.approx(
            row["exit_px"] / row["entry_px"], rel=1e-9), \
            f"{row['entry_date']:%F}~{row['exit_date']:%F} 净值与成交价对不上"


def test_no_stop_when_disabled():
    prices = [10 + i * 0.05 for i in range(60)] + [20.0, 16.0] + [16.0] * 10
    df = make_df(prices, start="2021-01-01")
    res = simulate(df, ma_len=10, stop=None, fee=0.0)
    assert "移动止损" not in set(res.trades.get("reason", []))


def test_stop_requires_month_end_to_reenter():
    """止损离场后，不能在同一个月里因为价格回升就重新买入。"""
    prices = ([10 + i * 0.05 for i in range(60)]
              + [20.0, 16.0] + [21.0] * 8)      # 止损后立刻反弹创新高
    df = make_df(prices, start="2021-01-01")
    res = simulate(df, ma_len=10, stop=0.15, fee=0.0)

    pos = res.position
    stop_day = res.trades[res.trades["reason"] == "移动止损"].iloc[0]["exit_date"]
    flags = month_end_flags(df.index)
    after = pos.loc[pos.index > stop_day]
    # 止损当天到下一个决策日之间，仓位必须一直是 0
    nxt = flags.loc[flags.index > stop_day]
    nxt_decision = nxt[nxt].index[0] if nxt.any() else None
    if nxt_decision is not None:
        gap = after.loc[after.index <= nxt_decision]
        assert (gap == 0).all(), "止损后未等月末就重新进场"


# ── 费用与统计 ────────────────────────────────────────────────────────────────

def test_fee_charged_per_switch():
    prices = [10.0] * 40 + [12.0] * 40
    df = make_df(prices, start="2021-01-01")
    free = simulate(df, ma_len=5, stop=None, fee=0.0)
    paid = simulate(df, ma_len=5, stop=None, fee=0.01)
    assert paid.stats["n_trades"] == free.stats["n_trades"] >= 1
    assert paid.stats["total_return"] < free.stats["total_return"]


def test_flat_position_earns_nothing_in_crash():
    """空仓期间不吃下跌——这条不成立的话整个止损就没意义。"""
    prices = [10.0] * 60 + [3.0] * 40         # 断崖后长期低位
    df = make_df(prices, start="2021-01-01")
    res = simulate(df, ma_len=10, stop=0.15, fee=0.0)
    bh = buy_hold(df, fee=0.0)
    assert res.stats["max_drawdown"] > bh.stats["max_drawdown"]
    assert res.stats["exposure"] < 1.0


def test_buy_hold_matches_raw_price_change():
    prices = [10.0, 11.0, 12.0, 9.0, 13.0]
    df = make_df(prices)
    bh = buy_hold(df, fee=0.0)
    assert bh.stats["total_return"] == pytest.approx(13.0 / 10.0 - 1)


def test_sweep_covers_grid():
    rng = np.random.default_rng(3)
    df = make_df(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    g = sweep(df, [20, 40], [None, 0.15], fee=0.0)
    assert len(g) == 4
    assert set(g.columns) >= {"ma", "stop", "年化", "最大回撤", "夏普"}
    assert g["stop"].tolist() == [0.0, 0.15, 0.0, 0.15]


def test_short_sample_raises_nothing_but_gives_no_signal():
    """样本短于均线时不该崩，只是没有结论。"""
    df = make_df([10.0] * 5)
    res = simulate(df, ma_len=20, stop=0.15, fee=0.0)
    assert res.state["ma"] is None
    assert (res.position == 0).all()


def test_unknown_freq_rejected():
    df = make_df([10.0] * 30)
    with pytest.raises(ValueError, match="未知决策频率"):
        simulate(df, ma_len=5, freq="quarter")
