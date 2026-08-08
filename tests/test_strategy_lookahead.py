"""
策略层未来函数 / 多周期对齐测试（不联网，全部用构造数据）
==========================================================

核心是 `_assert_no_lookahead` 这条属性测试：
    对每个历史日 t，只喂 df[:t] 重算出来的最后一根信号，
    必须等于喂完整 df 后 t 那天的信号。

不相等就意味着策略用到了 t 之后才知道的信息。这一条能通用地抓住
resample 标签打在区间起点、周线信号提前几天生效、以及各种 shift 方向
写反的问题，比逐个 case 断言更耐改。
"""

import numpy as np
import pandas as pd
import pytest

from strategies import LuMACDStrategy, LuMACDBullStrategy, MACDStrategy
from strategies.base import BaseStrategy


# ── 构造数据 ──────────────────────────────────────────────────────────────────

def _make_daily(n: int = 900, seed: int = 0, start: str = "2019-01-02") -> pd.DataFrame:
    """构造带波动的日线数据（工作日索引，模拟交易日）。"""
    rng = pd.bdate_range(start, periods=n)
    t = np.arange(n)
    # 长周期趋势 + 中周期波动 + 噪声，保证各周期 MACD 都能反复穿越 0 轴
    base = 60 + 22 * np.cos(2 * np.pi * t / 430) + 6 * np.sin(2 * np.pi * t / 65)
    noise = np.random.default_rng(seed).normal(0, 0.35, n).cumsum() * 0.3
    close = pd.Series(np.abs(base + noise) + 5, index=rng)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": np.random.default_rng(seed + 1).uniform(1e6, 4e6, n),
        },
        index=rng,
    )


def _assert_no_lookahead(strategy_factory, df: pd.DataFrame,
                         probe_days: int = 45, col: str = "signal"):
    """
    截断重算法：对最后 probe_days 个交易日逐个截断，断言历史信号不随未来数据改变。

    strategy_factory 每次返回全新实例，避免策略内部状态（如 LuMACD 的
    _bottom_price）跨次污染。
    """
    full = strategy_factory().prepare(df)
    probe_dates = full.index[-probe_days:]

    for d in probe_dates:
        truncated = df.loc[:d]
        partial = strategy_factory().prepare(truncated)
        assert partial.index[-1] == d, f"截断后末日应为 {d}"
        assert partial[col].iloc[-1] == full.loc[d, col], (
            f"{d.date()} 的 {col} 在补全未来数据后发生变化："
            f"截断={partial[col].iloc[-1]} 完整={full.loc[d, col]} → 存在未来函数"
        )


# ── _resample_period / _align_to_daily ───────────────────────────────────────

class _Dummy(BaseStrategy):
    name = "dummy"
    params: dict = {}

    def prepare(self, df):
        return df

    def plot_indicators(self, ax, df, colors):
        pass


def test_resample_period_labels_are_real_trading_days():
    """周/月线标签必须落在真实交易日上，且等于该区间最后一个交易日。"""
    df = _make_daily(400)
    s = _Dummy()

    for rule in ("W-FRI", "ME"):
        out = s._resample_period(df, rule, {"close": "last"})
        assert len(out) > 0
        assert out.index.isin(df.index).all(), f"{rule} 存在落在非交易日的标签"
        # 每根 K 线的 close 必须等于标签日当天的 close（区间最后一个交易日收盘）
        for label, row in out.iterrows():
            assert row["close"] == pytest.approx(df.loc[label, "close"]), (
                f"{rule} {label.date()} 的 close 不是标签日当天的收盘价"
            )


def test_resample_period_survives_holiday_at_period_edge():
    """周五整体休市时，整根周线不允许丢失（旧 reindex 写法会全丢）。"""
    df = _make_daily(300)
    no_friday = df[df.index.weekday != 4]      # 每周五都休市
    s = _Dummy()

    weekly = s._resample_period(no_friday, "W-FRI", {"close": "last"},
                                drop_incomplete=False)
    n_weeks = len(no_friday.index.to_period("W").unique())
    assert len(weekly) == n_weeks, "有周线 K 线被静默丢弃"
    assert weekly.index.isin(no_friday.index).all()
    # 对照：朴素写法在周五休市时一根都对不上
    assert weekly["close"].reindex(no_friday.index).notna().sum() == n_weeks


def test_resample_period_drops_only_the_unfinished_bar():
    """未走完的区间只丢最后一根，历史区间一根不少。"""
    df = _make_daily(300)
    s = _Dummy()

    full = s._resample_period(df, "W-FRI", {"close": "last"}, drop_incomplete=False)
    trimmed = s._resample_period(df, "W-FRI", {"close": "last"})

    ends_on_friday = df.index[-1].weekday() == 4
    expected = len(full) if ends_on_friday else len(full) - 1
    assert len(trimmed) == expected
    assert list(trimmed.index) == list(full.index[:len(trimmed)])


def test_incomplete_bar_would_make_signals_flip_flop():
    """
    守住回归：把未收盘的周线当成完整周线，会让同一天的信号随数据增长变脸。
    这里直接对比"含进行中 K 线"与"仅已收盘 K 线"两种口径下的周线根数。
    """
    df = _make_daily(300)
    s = _Dummy()
    # 截到周三：本周还没收盘
    wed = [d for d in df.index if d.weekday() == 2][-1]
    part = df.loc[:wed]

    naive = s._resample_period(part, "W-FRI", {"close": "last"}, drop_incomplete=False)
    strict = s._resample_period(part, "W-FRI", {"close": "last"})

    assert naive.index[-1] == wed          # 周三被当成"周线收盘日"
    assert strict.index[-1] < wed          # 严格口径不认这根
    assert len(strict) == len(naive) - 1


def test_align_to_daily_does_not_drop_bars():
    """低频标签不在日线索引里时，其值仍必须传播给之后的交易日。"""
    daily = pd.bdate_range("2024-01-01", periods=60)
    # 故意把低频标签放在周末（不在 daily 里）
    low = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2024-01-06", "2024-01-20", "2024-02-03"]),
    )
    out = _Dummy()._align_to_daily(low, daily)

    assert out.loc[pd.Timestamp("2024-01-08")] == 1.0
    assert out.loc[pd.Timestamp("2024-01-22")] == 2.0
    assert out.loc[pd.Timestamp("2024-02-05")] == 3.0
    # 首根低频 K 线之前保持缺失，不允许用后面的值回填
    assert out.loc[pd.Timestamp("2024-01-03")] != out.loc[pd.Timestamp("2024-01-08")]
    assert pd.isna(out.loc[pd.Timestamp("2024-01-03")])


def test_align_to_daily_naive_reindex_would_lose_data():
    """守住回归：朴素 reindex().ffill() 会丢数据，本测试固定住这个差异。"""
    daily = pd.bdate_range("2024-01-01", periods=60)
    low = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-06", "2024-01-20"]))

    naive = low.reindex(daily).ffill()
    fixed = _Dummy()._align_to_daily(low, daily)

    assert naive.notna().sum() == 0          # 全丢
    assert fixed.notna().sum() > 0


# ── 未来函数：三个策略 ────────────────────────────────────────────────────────

def test_macd_strategy_no_lookahead():
    df = _make_daily(400)
    _assert_no_lookahead(MACDStrategy, df)


def test_lu_macd_no_lookahead():
    """周线/月线三级确认不得提前生效（旧实现周一即使用本周五收盘）。"""
    df = _make_daily(900)
    _assert_no_lookahead(lambda: LuMACDStrategy(require_green_bar=False), df)


def test_lu_macd_bull_no_lookahead():
    """牛市过滤器不得让月初就知道当月月末收盘。"""
    df = _make_daily(900)
    index_df = _make_daily(900, seed=42)
    _assert_no_lookahead(
        lambda: LuMACDBullStrategy(index_df=index_df), df
    )


def test_lu_macd_bull_bull_market_flag_no_lookahead():
    """单独盯住 bull_market 这一列，避免信号恰好全 0 掩盖问题。"""
    df = _make_daily(900)
    index_df = _make_daily(900, seed=42)
    _assert_no_lookahead(
        lambda: LuMACDBullStrategy(index_df=index_df), df, col="bull_market"
    )


def test_lu_macd_weekly_events_land_on_week_close():
    """LuMACD 的买入信号只能出现在周线收盘日（该周最后一个交易日）。"""
    df = _make_daily(900)
    s = LuMACDStrategy(require_green_bar=False)
    out = s.prepare(df)
    week_close_dates = set(s._resample_ohlcv(df, "W-FRI").index)

    buys = out.index[out["signal"] == 1]
    for d in buys:
        assert d in week_close_dates, f"买入信号 {d.date()} 不在周线收盘日上"


# ── 买入条件不再恒真 ──────────────────────────────────────────────────────────

def test_bull_buy_requires_more_than_a_golden_cross():
    """
    金叉当根 hist 必然由 ≤0 翻正，"本根>上根"恒成立。
    默认 expand_bars=2 后，金叉当根不得再单独构成买点。
    """
    df = _make_daily(900)
    index_df = _make_daily(900, seed=42)
    s = LuMACDBullStrategy(index_df=index_df)
    out = s.prepare(df)

    dif, dea = out["DIF"], out["DEA"]
    golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))

    assert golden.sum() > 0, "测试数据没有金叉，用例失效"
    # 单根口径确实恒真 —— 正是旧实现没有过滤力的原因
    assert (golden & out["hist_expanding"]).sum() == golden.sum()
    # 连续口径必须能过滤掉金叉当根
    assert (golden & out["hist_expand_run"]).sum() == 0
    # 买点数量必须严格少于金叉数量
    assert (out["signal"] == 1).sum() < golden.sum()


def test_bull_expand_bars_one_reproduces_old_behaviour():
    """expand_bars=1 时退化为旧口径，便于对照与复现历史结果。"""
    df = _make_daily(600)
    index_df = _make_daily(600, seed=42)
    s = LuMACDBullStrategy(index_df=index_df, expand_bars=1, cross_window=1)
    out = s.prepare(df)

    dif, dea = out["DIF"], out["DEA"]
    golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    assert (golden & out["hist_expand_run"]).sum() == golden.sum()


def test_bull_buy_and_sell_never_collide():
    """连续拉长与红柱缩短互斥，同一根不应既是买点又是卖点。"""
    df = _make_daily(600)
    index_df = _make_daily(600, seed=42)
    out = LuMACDBullStrategy(index_df=index_df).prepare(df)
    assert not (out["hist_expand_run"] & out["hist_shrinking"]).any()


def test_bull_without_index_warns_and_disables_filter():
    df = _make_daily(300)
    s = LuMACDBullStrategy()
    with pytest.warns(UserWarning, match="牛市过滤器已禁用"):
        out = s.prepare(df)
    assert out["bull_market"].all()
