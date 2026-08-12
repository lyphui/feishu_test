"""
MACrossStrategy 单元测试（不联网，全部用构造数据）
=================================================
覆盖三件事：
  1. 金叉/死叉的判定与位置（含"预热期不得出幽灵信号"）
  2. 量能过滤只作用于进场，不影响出场
  3. 无未来函数：截断重算后历史信号不变（与 test_strategy_lookahead.py 同法）
"""

import numpy as np
import pandas as pd
import pytest

from strategies import MACrossStrategy


def _frame(close: list[float], volume=None) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    s = pd.Series(close, index=idx, dtype="float64")
    vol = pd.Series(volume if volume is not None else 1e6, index=idx,
                    dtype="float64")
    return pd.DataFrame({"open": s.shift(1).fillna(s.iloc[0]),
                         "high": s * 1.01, "low": s * 0.99,
                         "close": s, "volume": vol}, index=idx)


def _wave(n: int = 240, seed: int = 3) -> pd.DataFrame:
    """带趋势与波动的日线，保证 MA5/MA8 反复穿越。"""
    t = np.arange(n)
    base = 50 + 8 * np.sin(2 * np.pi * t / 55) + 0.02 * t
    noise = np.random.default_rng(seed).normal(0, 0.4, n).cumsum() * 0.2
    close = np.abs(base + noise) + 5
    vol = np.random.default_rng(seed + 1).uniform(5e5, 5e6, n)
    return _frame(list(close), volume=vol)


# ── 参数校验 ──────────────────────────────────────────────────────────────────

def test_fast_must_be_shorter_than_slow():
    with pytest.raises(ValueError):
        MACrossStrategy(fast=8, slow=5)
    with pytest.raises(ValueError):
        MACrossStrategy(fast=5, slow=5)


def test_unknown_ma_type_rejected():
    with pytest.raises(ValueError):
        MACrossStrategy(ma_type="wma")


# ── 信号判定 ──────────────────────────────────────────────────────────────────

def test_golden_and_death_cross_positions():
    """先跌后涨再跌：应恰好出现一次金叉、一次死叉，且金叉在死叉之前。"""
    close = [20 - 0.5 * i for i in range(20)]          # 下跌 20 天
    close += [close[-1] + 0.8 * i for i in range(1, 31)]   # 上涨 30 天
    close += [close[-1] - 0.9 * i for i in range(1, 31)]   # 再跌 30 天
    out = MACrossStrategy(fast=5, slow=8).prepare(_frame(close))

    buys = out.index[out["signal"] == 1]
    sells = out.index[out["signal"] == -1]
    assert len(buys) == 1, f"应只有一次金叉，实际 {len(buys)}"
    assert len(sells) == 1, f"应只有一次死叉，实际 {len(sells)}"
    assert buys[0] < sells[0]
    # 金叉当日 MA5 必须在 MA8 之上，前一日在其下
    i = out.index.get_loc(buys[0])
    assert out["MA_FAST"].iloc[i] > out["MA_SLOW"].iloc[i]
    assert out["MA_FAST"].iloc[i - 1] <= out["MA_SLOW"].iloc[i - 1]


def test_warmup_rows_dropped_and_no_phantom_signal():
    """预热期整行被 dropna 掉；首根有效 K 线不得凭空报信号。"""
    df = _wave(60)
    out = MACrossStrategy(fast=5, slow=8).prepare(df)
    assert len(out) == len(df) - (8 - 1)          # MA8 需要 8 根
    assert out[["MA_FAST", "MA_SLOW"]].notna().all().all()
    assert out["signal"].iloc[0] == 0


def test_monotonic_trend_gives_no_repeat_signals():
    """单边上涨中金叉只报一次，不是天天报买入。"""
    out = MACrossStrategy(fast=5, slow=8).prepare(
        _frame([10 + 0.3 * i for i in range(80)]))
    assert (out["signal"] == 1).sum() <= 1
    assert (out["signal"] == -1).sum() == 0


# ── 量能过滤 ──────────────────────────────────────────────────────────────────

def _cross_frame(vol_at_cross: float, other_vol: float = 1e6) -> tuple:
    """构造一次金叉 + 一次死叉，并把金叉当日的量设成给定值。"""
    close = [20 - 0.5 * i for i in range(20)]
    close += [close[-1] + 0.8 * i for i in range(1, 31)]
    close += [close[-1] - 0.9 * i for i in range(1, 31)]
    df = _frame(close)
    plain = MACrossStrategy(fast=5, slow=8).prepare(df)
    cross_day = plain.index[plain["signal"] == 1][0]
    df["volume"] = other_vol
    df.loc[cross_day, "volume"] = vol_at_cross
    return df, cross_day


def test_volume_filter_blocks_shrinking_golden_cross():
    df, cross_day = _cross_frame(vol_at_cross=2e5)      # 金叉当日缩量
    out = MACrossStrategy(fast=5, slow=8, vol_window=5, vol_ratio=1.0).prepare(df)
    assert out.loc[cross_day, "signal"] == 0, "缩量金叉应被过滤掉"
    # 死叉不受量能影响，仍然照卖
    assert (out["signal"] == -1).sum() == 1


def test_volume_filter_keeps_expanding_golden_cross():
    df, cross_day = _cross_frame(vol_at_cross=5e6)      # 金叉当日放量
    out = MACrossStrategy(fast=5, slow=8, vol_window=5, vol_ratio=1.0).prepare(df)
    assert out.loc[cross_day, "signal"] == 1


def test_volume_filter_never_adds_buy_signals():
    """量能过滤只能减少买点，不能凭空生出新的买点。"""
    df = _wave()
    plain = MACrossStrategy(fast=5, slow=8).prepare(df)
    filtered = MACrossStrategy(fast=5, slow=8, vol_window=5,
                               vol_ratio=1.2).prepare(df)
    buys_plain = set(plain.index[plain["signal"] == 1])
    buys_filtered = set(filtered.index[filtered["signal"] == 1])
    assert buys_filtered <= buys_plain
    # 卖点完全一致
    assert (set(plain.index[plain["signal"] == -1])
            == set(filtered.index[filtered["signal"] == -1]))


# ── 未来函数 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    {"fast": 5, "slow": 8},
    {"fast": 5, "slow": 8, "vol_window": 5, "vol_ratio": 1.0},
    {"fast": 20, "slow": 60},
    {"fast": 5, "slow": 8, "ma_type": "ema"},
])
def test_no_lookahead(kw):
    """截断重算：只喂 df[:t] 算出的末根信号，必须等于喂全量时 t 那天的信号。"""
    df = _wave(300)
    full = MACrossStrategy(**kw).prepare(df)
    for d in full.index[-40:]:
        partial = MACrossStrategy(**kw).prepare(df.loc[:d])
        assert partial.index[-1] == d
        assert partial["signal"].iloc[-1] == full.loc[d, "signal"], (
            f"{d.date()} 的信号随未来数据改变 → 存在未来函数")


def test_ema_type_changes_signals_but_stays_valid():
    df = _wave()
    sma = MACrossStrategy(fast=5, slow=8, ma_type="sma").prepare(df)
    ema = MACrossStrategy(fast=5, slow=8, ma_type="ema").prepare(df)
    assert set(ema["signal"].unique()) <= {-1, 0, 1}
    assert (ema["signal"] != 0).sum() > 0
    assert not sma["signal"].equals(ema["signal"])
