"""
回测引擎成交细节测试（不联网：行情通过 run_backtest(df=...) 直接注入）
=====================================================================
覆盖 A 股实盘约束：T+1 开盘成交、T+1 当日不可卖、涨跌停/停牌无法成交、
佣金最低 5 元、双边滑点、印花税单边、统计窗口 eval_start。
"""

import numpy as np
import pandas as pd
import pytest

from engine import run_backtest, infer_limit_pct, _commission, _tradability
from strategies.base import BaseStrategy


# ── 测试脚手架 ────────────────────────────────────────────────────────────────

class ScriptedStrategy(BaseStrategy):
    """按预先给定的 {日期偏移: 信号} 发信号，把引擎行为与指标逻辑解耦。"""

    def __init__(self, signals: dict[int, int]):
        self._signals = signals

    name = "scripted"
    params: dict = {}

    def prepare(self, df):
        df = df.copy()
        df["signal"] = 0
        for pos, sig in self._signals.items():
            df.iloc[pos, df.columns.get_loc("signal")] = sig
        return df

    def plot_indicators(self, ax, df, colors):
        pass


def _flat_df(n=80, price=10.0, start="2024-01-01"):
    """恒定价格行情：任何收益都只可能来自成本，便于精确核对费用。"""
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price,
         "volume": 1e6},
        index=idx,
    )


def _run(df, signals, **kw):
    kw.setdefault("initial_capital", 100_000.0)
    return run_backtest("600000", "20240101", "20241231",
                        strategy=ScriptedStrategy(signals), df=df, **kw)


# ── T+1 成交时序 ──────────────────────────────────────────────────────────────

def test_signal_executes_next_day_at_open():
    df = _flat_df()
    # 信号在第 10 根产生，成交应发生在第 11 根：把两根的开盘价区分开
    df.iloc[10, df.columns.get_loc("open")] = 12.0
    df.iloc[11, df.columns.get_loc("open")] = 11.0
    r = _run(df, {10: 1}, slippage=0.0, limit_move_check=False)

    trades = r["trades"]
    buy = trades[trades["action"] == "买入"].iloc[0]
    assert buy["date"] == df.index[11], "T 日信号必须在 T+1 成交"
    assert buy["price"] == pytest.approx(11.0), "必须用 T+1 的开盘价，不是信号日的"


def test_position_opened_today_cannot_stop_out_same_day():
    """A 股 T+1：当天买入当天不能卖，哪怕盘中已跌破止损线。"""
    df = _flat_df()
    buy_day = 11
    df.iloc[buy_day, df.columns.get_loc("low")] = 5.0   # 建仓当日盘中腰斩

    r = _run(df, {10: 1}, stop_loss=0.10, slippage=0.0, limit_move_check=False)
    stops = r["trades"][r["trades"]["action"] == "止损卖出"]
    assert stops.empty or stops.iloc[0]["date"] > df.index[buy_day]


def test_stop_loss_triggers_on_a_later_bar():
    df = _flat_df()
    df.iloc[15, df.columns.get_loc("low")] = 8.5        # 第16根跌破 10% 止损
    r = _run(df, {10: 1}, stop_loss=0.10, slippage=0.0, limit_move_check=False)

    stops = r["trades"][r["trades"]["action"] == "止损卖出"]
    assert len(stops) == 1
    assert stops.iloc[0]["date"] == df.index[15]
    assert stops.iloc[0]["price"] == pytest.approx(9.0)  # min(open=10, 止损价=9)


def test_no_rebuy_before_the_exit_that_precedes_it():
    """
    时序回归：止损出场发生在盘中，同一根 K 线的开盘价买入必须先于它。
    旧实现先跑止损再跑买入，会在出场后用更早的开盘价重新建仓。
    """
    df = _flat_df()
    df.iloc[15, df.columns.get_loc("low")] = 8.5
    r = _run(df, {10: 1, 14: 1}, stop_loss=0.10, slippage=0.0,
             limit_move_check=False)

    same_bar = r["trades"][r["trades"]["date"] == df.index[15]]
    actions = list(same_bar["action"])
    assert "买入" not in actions, f"止损当根不应再建仓：{actions}"


# ── 涨跌停 / 停牌 ─────────────────────────────────────────────────────────────

def test_limit_up_open_blocks_buy_and_defers_it():
    df = _flat_df()
    # 第11根（T+1）一字涨停：前收 10，开盘 11
    for col in ("open", "high", "low", "close"):
        df.iloc[11, df.columns.get_loc(col)] = 11.0
    # 第12根回落到可成交
    for col in ("open", "high", "low", "close"):
        df.iloc[12, df.columns.get_loc(col)] = 10.5

    r = _run(df, {10: 1}, slippage=0.0)
    buys = r["trades"][r["trades"]["action"] == "买入"]
    assert len(buys) == 1
    assert buys.iloc[0]["date"] == df.index[12], "涨停当日不该成交，应顺延"
    assert not r["blocked_trades"].empty


def test_limit_down_open_blocks_sell():
    df = _flat_df()
    for col in ("open", "high", "low", "close"):
        df.iloc[20, df.columns.get_loc(col)] = 9.0     # 一字跌停
    r = _run(df, {10: 1, 19: -1}, slippage=0.0)

    sells = r["trades"][r["trades"]["action"] == "卖出"]
    assert sells.empty or sells.iloc[0]["date"] > df.index[20]
    assert (r["blocked_trades"]["action"] == "卖出受阻").any()


def test_pending_order_expires_after_max_pending_days():
    df = _flat_df()
    for i in range(11, 40):                             # 连续一字涨停
        for col in ("open", "high", "low", "close"):
            df.iloc[i, df.columns.get_loc(col)] = df.iloc[i - 1]["close"] * 1.1

    r = _run(df, {10: 1}, slippage=0.0, max_pending_days=3)
    assert r["trades"].empty, "连板期间不应追进去"
    assert len(r["blocked_trades"]) <= 4


def test_suspended_day_is_untradable():
    df = _flat_df()
    df.iloc[11, df.columns.get_loc("volume")] = 0       # 停牌
    r = _run(df, {10: 1}, slippage=0.0)
    buys = r["trades"][r["trades"]["action"] == "买入"]
    assert buys.iloc[0]["date"] == df.index[12]


def test_limit_move_check_can_be_disabled():
    df = _flat_df()
    for col in ("open", "high", "low", "close"):
        df.iloc[11, df.columns.get_loc(col)] = 11.0
    r = _run(df, {10: 1}, slippage=0.0, limit_move_check=False)
    assert r["trades"].iloc[0]["date"] == df.index[11]


@pytest.mark.parametrize("symbol,expected", [
    ("600519", 0.10), ("000001", 0.10),
    ("300750", 0.20), ("301029", 0.20), ("688981", 0.20),
    ("830799", 0.30), ("430047", 0.30),
])
def test_infer_limit_pct(symbol, expected):
    assert infer_limit_pct(symbol) == expected


def test_tradability_first_bar_without_prev_close():
    row = pd.Series({"open": 10.0, "high": 10.0, "low": 10.0,
                     "close": 10.0, "volume": 1e6})
    assert _tradability(row, float("nan"), 0.10) == (True, True)


# ── 费用 ──────────────────────────────────────────────────────────────────────

def test_min_commission_floor_applies():
    assert _commission(1_000.0, 0.0003, 5.0) == 5.0        # 0.3 元 → 抬到 5
    assert _commission(1_000_000.0, 0.0003, 5.0) == 300.0  # 超过下限则按比例


def test_round_trip_cost_matches_hand_calculation():
    """价格恒定时，一买一卖的亏损必须精确等于手续费 + 滑点。"""
    df = _flat_df(price=10.0)
    cap = 100_000.0
    slip, rate, duty, minc = 0.001, 0.0003, 0.001, 5.0

    r = _run(df, {10: 1, 30: -1}, initial_capital=cap, slippage=slip,
             commission_rate=rate, stamp_duty=duty, min_commission=minc,
             limit_move_check=False)

    buy_px = 10.0 * (1 + slip)
    lots = int(cap / buy_px / 100)
    shares = lots * 100
    cost = shares * buy_px
    buy_fee = max(cost * rate, minc)
    sell_px = 10.0 * (1 - slip)
    proceeds = shares * sell_px
    sell_fee = max(proceeds * rate, minc) + proceeds * duty
    expected_equity = cap - cost - buy_fee + proceeds - sell_fee

    assert r["final_equity"] == pytest.approx(expected_equity, rel=1e-9)
    assert r["total_return"] < 0                      # 恒定价格下只能亏成本


def test_slippage_is_adverse_on_both_sides():
    df = _flat_df(price=10.0)
    r = _run(df, {10: 1, 30: -1}, slippage=0.01, limit_move_check=False)
    t = r["trades"]
    assert t[t["action"] == "买入"].iloc[0]["price"] == pytest.approx(10.1)
    assert t[t["action"] == "卖出"].iloc[0]["price"] == pytest.approx(9.9)


def test_never_spends_more_cash_than_available():
    df = _flat_df(price=10.0)
    r = _run(df, {10: 1}, initial_capital=1_050.0, slippage=0.001,
             limit_move_check=False)
    assert (r["equity_curve"]["equity"] > 0).all()
    trades = r["trades"]
    if not trades.empty:
        assert trades.iloc[0]["cash"] >= 0


# ── 统计口径 ──────────────────────────────────────────────────────────────────

def test_eval_start_excludes_warmup_from_stats():
    """预热期不交易，权益恒定；计入统计会拉低夏普、抬高样本数。"""
    n = 500
    idx = pd.bdate_range("2023-01-02", periods=n)
    px = pd.Series(np.linspace(10.0, 20.0, n), index=idx)
    df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                       "close": px, "volume": 1e6}, index=idx)

    warm_end = 300
    signals = {warm_end: 1}
    eval_date = idx[warm_end].strftime("%Y%m%d")

    full = _run(df, signals, slippage=0.0, limit_move_check=False)
    windowed = _run(df, signals, slippage=0.0, limit_move_check=False,
                    eval_start=eval_date)

    assert len(windowed["equity_curve"]) < len(full["equity_curve"])
    assert len(windowed["equity_curve_full"]) == len(full["equity_curve"])
    # 总收益不受影响（预热期权益不变），但夏普必须变高（不再被平段稀释）
    assert windowed["total_return"] == pytest.approx(full["total_return"])
    assert windowed["sharpe_ratio"] > full["sharpe_ratio"]
    # 基准也必须只算窗口内的涨幅
    assert windowed["benchmark_return"] < full["benchmark_return"]


def test_benchmark_uses_open_of_first_bar():
    """基准与策略同口径：首日开盘买入、末日收盘估值。"""
    n = 60
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({"open": 10.0, "high": 12.0, "low": 9.0,
                       "close": 11.0, "volume": 1e6}, index=idx)
    r = _run(df, {}, slippage=0.0, limit_move_check=False)

    assert r["benchmark_base"] == pytest.approx(10.0)
    assert r["benchmark_return"] == pytest.approx(10.0)   # 10 开盘 → 11 收盘


def test_eval_start_beyond_data_raises():
    df = _flat_df()
    with pytest.raises(ValueError, match="之后没有任何交易日"):
        _run(df, {}, eval_start="20991231")
