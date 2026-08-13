"""PositionTracker 的执行时点：统一 T+1，且不受 `--lookback` 影响。

日线的 signal / 红柱缩短 / 死叉 / DIF<0 都要等当日收盘才算得出来，
所以每一条都必须排到下一个交易日成交——买卖两侧、窗口内外，一视同仁。

直接从 `lib/position_tracker` 导入（纯计算模块），不再 importorskip 进
那个 1300+ 行的 CLI 脚本。
"""

import pandas as pd
import pytest

from backtest.lib.position_tracker import PositionTracker


def frame(*, closes, signal=None, expanding=None, shrinking=None,
          dif=None, dea=None):
    n = len(closes)
    idx = pd.bdate_range("2026-03-02", periods=n)
    return pd.DataFrame({
        "close": closes,
        "signal": signal if signal is not None else [0] * n,
        "hist_expanding": expanding if expanding is not None else [False] * n,
        "hist_shrinking": shrinking if shrinking is not None else [False] * n,
        "DIF": dif if dif is not None else [0.5] * n,
        "DEA": dea if dea is not None else [0.2] * n,
    }, index=idx)


def by_action(tracker, action):
    return [t for t in tracker.trades if t.action == action]


# ── 买入侧 ───────────────────────────────────────────────────────────────────

def test_buy_executes_next_day_not_signal_day():
    """signal 要等 T 日收盘才算得出来，不能在 T 日收盘价成交。"""
    df = frame(closes=[10.0, 11.0, 12.0, 13.0], signal=[1, 0, 0, 0])
    t = PositionTracker(capital=100_000)
    t.run(df)
    buys = by_action(t, "初仓")
    assert len(buys) == 1
    assert buys[0].date == df.index[1]          # T+1
    assert buys[0].price == pytest.approx(11.0)  # T+1 的收盘价


def test_signal_on_last_bar_never_trades():
    """最后一根 K 线之后没有 T+1，该操作只能作废，不许当日抢成交。"""
    df = frame(closes=[10.0, 11.0, 12.0], signal=[0, 0, 1])
    t = PositionTracker(capital=100_000)
    t.run(df)
    assert t.trades == []


def test_buy_add_also_deferred():
    """加仓同样是收盘后才知道的条件，也要顺延一天。"""
    df = frame(closes=[10.0, 11.0, 12.0, 13.0, 14.0],
               signal=[1, 0, 0, 0, 0],
               expanding=[False, False, True, False, False])
    t = PositionTracker(capital=100_000)
    t.run(df)
    adds = by_action(t, "加仓")
    assert len(adds) == 1
    assert adds[0].date == df.index[3]          # idx2 判定 → idx3 成交
    assert adds[0].price == pytest.approx(13.0)


# ── 卖出侧 ───────────────────────────────────────────────────────────────────

def _full_position_frame():
    """idx0 买信号 → idx1 初仓 → idx2 红柱拉长 → idx3 满仓，之后可测卖出。"""
    closes = [10.0, 10.0, 10.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    n = len(closes)
    return frame(
        closes=closes,
        signal=[1] + [0] * (n - 1),
        expanding=[False, False, True] + [False] * (n - 3),
        shrinking=[False] * 4 + [True] + [False] * (n - 5),
        dif=[0.5] * 5 + [0.1, -0.1, -0.1, -0.1],
        dea=[0.2] * 5 + [0.3, 0.3, 0.3, 0.3],
    )


def test_three_stage_sells_are_all_deferred_one_day():
    """一级红柱缩短、二级死叉、三级 DIF<0 —— 三级全部 T+1 成交。"""
    df = _full_position_frame()
    t = PositionTracker(capital=100_000)
    t.run(df)

    sells = [x for x in t.trades if x.action in ("减仓", "清仓")]
    assert [s.reason for s in sells] == ["红柱缩短", "死叉", "DIF<0"]
    # idx4 判定缩短 → idx5 成交；idx5 判定死叉 → idx6；idx6 判定 DIF<0 → idx7
    assert [s.date for s in sells] == [df.index[5], df.index[6], df.index[7]]
    assert [s.price for s in sells] == pytest.approx([30.0, 40.0, 50.0])


def test_sell_price_is_exec_day_not_signal_day():
    """价格必须取执行日的——拿信号日的价成交次日的单是把已知价错配到另一天。"""
    df = _full_position_frame()
    t = PositionTracker(capital=100_000)
    t.run(df)
    first_sell = [x for x in t.trades if x.reason == "红柱缩短"][0]
    assert first_sell.price != pytest.approx(df["close"].iloc[4])   # 信号日 20.0
    assert first_sell.price == pytest.approx(df["close"].iloc[5])   # 执行日 30.0


# ── lookback 不变性 ──────────────────────────────────────────────────────────

def test_lookback_window_does_not_change_execution_dates():
    """
    `intraday_map` 只换成交价、不换成交日。

    早先窗口内的信号 T+1 成交、窗口外的当日收盘成交，等于同一次回测前后两段
    用两套规则——改个只该影响打印的参数就能改变历史收益。
    """
    df = _full_position_frame()
    d0 = df.index[0]

    bare = PositionTracker(capital=100_000)
    bare.run(df)

    # 同一个信号进了 lookback 窗口，但分时价缺失（exec_price=None）
    covered = PositionTracker(capital=100_000)
    covered.run(df, {d0: {"exec_date": df.index[1], "exec_price": None,
                          "action": "buy", "dif": 0.5}})

    assert [(t.date, t.action, t.price) for t in bare.trades] == \
           [(t.date, t.action, t.price) for t in covered.trades]


def test_intraday_price_changes_price_only_not_date():
    df = _full_position_frame()
    d0 = df.index[0]
    t = PositionTracker(capital=100_000)
    t.run(df, {d0: {"exec_date": df.index[1], "exec_price": 9.5,
                    "action": "buy", "dif": 0.5}})
    buy = by_action(t, "初仓")[0]
    assert buy.date == df.index[1]              # 日子还是 T+1
    assert buy.price == pytest.approx(9.5)      # 只有价换成了分时价


def test_exec_day_same_mode_still_fills_same_day():
    """`--exec_day same` 是使用者明确选的盘中实时口径，保留当日成交。"""
    df = _full_position_frame()
    d0 = df.index[0]
    t = PositionTracker(capital=100_000)
    t.run(df, {d0: {"exec_date": d0, "exec_price": 9.9,
                    "action": "buy", "dif": 0.5}})
    buy = by_action(t, "初仓")[0]
    assert buy.date == d0
    assert buy.price == pytest.approx(9.9)
