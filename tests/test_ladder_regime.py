"""分批建仓模拟器 + 市场状态分类器（离线，全部用合成行情）。"""

import numpy as np
import pandas as pd
import pytest

from backtest.lib import regime as rg
from backtest.lib.ladder import (simulate_adaptive, simulate_buy_hold, simulate_dca,
                        simulate_grid, simulate_ladder)
from backtest.lib.swings import drawdown_episodes, swing_table


def make_df(closes, start="2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": 1_000_000}, index=idx)


def ramp(a, b, n):
    return list(np.linspace(a, b, n))


# ── swings ────────────────────────────────────────────────────────────────────

def test_zigzag_finds_swings_without_a_gap_day():
    """方向未确认时若共用极值变量，只有单日跳空能确认拐点，缓涨缓跌会整段丢失。"""
    closes = ramp(10, 15, 60) + ramp(15, 11, 60) + ramp(11, 16, 60)
    t = swing_table(make_df(closes)["close"], 0.08)
    assert len(t) >= 3
    assert (t["pct"] > 0).any() and (t["pct"] < 0).any()


def test_drawdown_episodes_counts_one_dip_once():
    """价格在阈值附近抖动时，同一轮回撤不能被记成很多次。"""
    closes = ramp(10, 12, 30) + [11.0, 10.5, 11.0, 10.4, 11.0, 10.6] + ramp(11, 14, 30)
    ep = drawdown_episodes(make_df(closes)["close"])
    assert len(ep[ep["depth"] <= -0.05]) == 1


def test_drawdown_episode_records_recovery():
    closes = ramp(10, 12, 20) + ramp(12, 9, 20) + ramp(9, 13, 20)
    ep = drawdown_episodes(make_df(closes)["close"])
    deep = ep[ep["depth"] <= -0.20].iloc[0]
    assert deep["recover_days"] > 0
    assert pd.notna(deep["recover_date"])


# ── ladder ────────────────────────────────────────────────────────────────────

def test_buy_hold_tracks_price():
    df = make_df(ramp(10, 20, 250))
    r = simulate_buy_hold(df, 100_000)
    assert 0.85 < r.stats["total_return"] / 1.0 < 1.0     # ~+100% 扣成本
    assert r.stats["n_trades"] == 1


def test_ladder_honors_the_shared_tradability_rule():
    """
    停牌与一字涨停的判定来自 `lib/costs.tradability`，ladder 转调它、不自存一份。

    这条是**接线测试**：`tests/test_costs.py` 保证那个函数本身算得对，这里保证
    ladder 的撮合循环真的走它。买单 T+1 开盘执行，所以约束要放在第 2 根 K 线上。
    """
    df = make_df(ramp(10, 20, 60)).astype({"volume": float})
    assert len(simulate_buy_hold(df, 100_000).trades) == 1          # 基准：正常成交

    for vol in (0.0, -5.0, float("nan")):
        halted = df.copy()
        halted.loc[halted.index[1], "volume"] = vol
        assert not simulate_buy_hold(halted, 100_000).trades, f"停牌日不该成交（volume={vol}）"

    limit_up = df.copy()
    limit_up.loc[limit_up.index[1], "open"] = float(df["close"].iloc[0]) * 1.10
    assert not simulate_buy_hold(limit_up, 100_000).trades, "开盘一字涨停买不进"


def test_ladder_buys_more_tranches_on_deeper_fall():
    df = make_df(ramp(10, 6, 200))                        # 一路跌 40%
    shallow = simulate_ladder(df, 100_000, n_tranches=4, step=0.08)
    deep = simulate_ladder(df, 100_000, n_tranches=4, step=0.20)
    assert shallow.stats["n_trades"] > deep.stats["n_trades"]


def test_ladder_never_exceeds_capital():
    df = make_df(ramp(10, 4, 300))
    r = simulate_ladder(df, 100_000, n_tranches=5, step=0.05)
    bought = sum(t["amount"] + t["fee"] for t in r.trades if t["action"] == "buy")
    assert bought <= 100_000 * 1.02                       # 仅现金利息带来的极小溢出


def test_take_profit_sells_after_gain():
    df = make_df(ramp(10, 20, 200))
    r = simulate_ladder(df, 100_000, n_tranches=2, step=0.05, take_profit=0.20)
    assert any(t["action"] == "sell" for t in r.trades)


def test_ma_exit_does_not_churn():
    """止损后必须锁住再入场，否则下跌途中会刷出成百上千笔来回交易。"""
    df = make_df(ramp(20, 8, 400))
    r = simulate_ladder(df, 100_000, n_tranches=4, step=0.08, ma_exit=120)
    assert r.stats["n_trades"] < 30, f"疑似反复进出：{r.stats['n_trades']} 笔"


def test_ma_exit_limits_bear_market_loss():
    df = make_df(ramp(20, 8, 400))
    stop = simulate_ladder(df, 100_000, n_tranches=4, step=0.08, ma_exit=120)
    hold = simulate_buy_hold(df, 100_000)
    assert stop.stats["total_return"] > hold.stats["total_return"]


def test_grid_sells_exactly_the_shares_it_bought():
    """每格买到的股数不同，卖出必须按真实股数走，不能按金额比例折算。"""
    closes = ramp(10, 7, 80) + ramp(7, 11, 80) + ramp(11, 8, 80)
    r = simulate_grid(make_df(closes), 100_000, base_position=0.5,
                      n_grids=4, grid_step=0.07)
    sells = [t for t in r.trades if t["action"] == "sell"]
    buys = [t for t in r.trades if t["action"] == "buy"]
    assert sells, "网格应当在反弹时卖出"
    # 底仓永远不卖：卖出股数合计 < 买入股数合计
    assert sum(t["shares"] for t in sells) < sum(t["shares"] for t in buys)


def test_grid_never_arms_on_a_stock_that_only_goes_up():
    """锚价钉死在首根收盘时，一路上涨的标的会让网格一次都装不上膛。

    这不是实现缺陷而是默认行为，但它会让参数扫描出现整片一模一样的结果
    （600938 上 36 组参数全是 1 笔交易）。判断网格好不好用之前先看 n_trades。
    """
    df = make_df(ramp(10, 25, 300))
    r = simulate_grid(df, 100_000, base_position=0.5, n_grids=5, grid_step=0.07)
    assert r.stats["n_trades"] == 1


def test_grid_ratchet_re_arms_after_a_rally():
    """ratchet=True 时锚随新高上移，涨完再回调才有格子可买。"""
    closes = ramp(10, 25, 300) + ramp(25, 18, 60) + ramp(18, 24, 60)
    df = make_df(closes)
    plain = simulate_grid(df, 100_000, base_position=0.5, n_grids=5, grid_step=0.07)
    rat = simulate_grid(df, 100_000, base_position=0.5, n_grids=5, grid_step=0.07,
                        ratchet=True)
    assert plain.stats["n_trades"] == 1
    assert rat.stats["n_trades"] > 3


def test_grid_ratchet_does_not_move_the_anchor_while_holding_grid_lots():
    """持有网格仓位时抬锚会让卖出触发价对不上买入格，必须只在 level==0 时上移。"""
    # 先跌破一格建仓，再小幅反弹但不到卖出线，锚若被抬高就会提前/错价卖出
    closes = ramp(10, 8.5, 40) + ramp(8.5, 8.9, 40)
    r = simulate_grid(make_df(closes), 100_000, base_position=0.5,
                      n_grids=5, grid_step=0.07, ratchet=True)
    sells = [t for t in r.trades if t["action"] == "sell"]
    assert sells == []


def test_dca_spreads_purchases():
    df = make_df(ramp(10, 12, 300))
    r = simulate_dca(df, 100_000, n_tranches=10, every_days=21)
    assert r.stats["n_trades"] == 10
    assert r.stats["avg_exposure"] < 0.95                 # 建仓期不可能一直满仓


def test_equity_is_positive_and_finite():
    df = make_df(ramp(10, 3, 300))
    for r in [simulate_buy_hold(df), simulate_ladder(df), simulate_grid(df),
              simulate_dca(df)]:
        assert np.isfinite(r.equity).all()
        assert (r.equity > 0).all()


# ── regime ────────────────────────────────────────────────────────────────────

def test_classify_is_causal():
    """截断后重算，历史标签不得改变——否则就是用了未来数据。"""
    closes = ramp(10, 18, 400) + ramp(18, 11, 300) + ramp(11, 16, 300)
    df = make_df(closes)
    full = rg.classify(df)["regime"]
    cut = rg.classify(df.iloc[:700])["regime"]
    assert (full.iloc[:700].values == cut.values).all()


def test_classify_labels_uptrend_and_downtrend():
    up = rg.classify(make_df(ramp(10, 30, 600)))["regime"]
    assert up.iloc[-1] == rg.TREND_UP
    down = rg.classify(make_df(ramp(30, 10, 600)))["regime"]
    assert down.iloc[-1] == rg.BEAR


def test_classify_hysteresis_suppresses_flapping():
    """价格在年线上下反复穿越时，状态不能天天翻。"""
    base = ramp(10, 20, 300)
    noisy = base + [20 + (1.5 if i % 2 else -1.5) for i in range(200)]
    reg = rg.classify(make_df(noisy), confirm_days=5)["regime"]
    switches = (reg != reg.shift()).sum()
    assert switches < 20, f"状态切换过于频繁：{switches} 次"


def test_adaptive_cuts_position_in_downtrend():
    df = make_df(ramp(10, 25, 400) + ramp(25, 9, 400))
    reg = rg.classify(df)["regime"]
    r = simulate_adaptive(df, reg, 100_000)
    tail_exposure = r.exposure.iloc[-50:].mean()
    assert tail_exposure < 0.5, f"下行段仓位仍有 {tail_exposure:.0%}"


def test_adaptive_drawdown_is_smaller_than_holding():
    """
    自适应的设计目标是控回撤，不是在任意路径上都跑赢满仓。

    默认打法在趋势下行档保留 30% 底仓（历史上这两只票的下行段后常跟反弹，
    清仓版反复踏空），所以单边下跌里它必然还是要吃一部分亏损——
    该守住的是"回撤显著小于满仓"，而不是"总收益一定更高"。
    """
    df = make_df(ramp(10, 25, 400) + ramp(25, 9, 400))
    reg = rg.classify(df)["regime"]
    ad = simulate_adaptive(df, reg, 100_000)
    bh = simulate_buy_hold(df, 100_000)
    assert ad.stats["max_drawdown"] > bh.stats["max_drawdown"] + 0.10


def test_regime_stats_returns_all_present_labels():
    df = make_df(ramp(10, 25, 400) + ramp(25, 9, 400) + ramp(9, 14, 200))
    reg = rg.classify(df)
    st = rg.regime_stats(df, reg, horizon=20)
    assert len(st) >= 2
    assert set(st.index) <= set(rg.LABELS)
