"""lib/execution.py：日内下单方案测算（全合成数据，不联网、不读 data/）。"""

import numpy as np
import pandas as pd
import pytest

from lib import execution as ex

BARS = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]


def make_bars(n_days: int = 30, base: float = 10.0, seed: int = 0,
              drift: float = 0.0) -> pd.DataFrame:
    """造 n_days × 8 根 30min K 线，amount/volume 与 close 严格同口径。"""
    rng = np.random.default_rng(seed)
    rows = []
    px = base
    for d in pd.bdate_range("2024-01-01", periods=n_days):
        for hm in BARS:
            px *= 1 + rng.normal(drift, 0.004)
            vol = float(rng.integers(1000, 5000) * 100)
            rows.append({
                "dt": pd.Timestamp(f"{d.date()} {hm}"), "date": d,
                "open": px * 0.999, "high": px * 1.004, "low": px * 0.996,
                "close": px, "volume": vol, "amount": px * vol,
            })
    return pd.DataFrame(rows)


def test_vwap_matches_amount_over_volume():
    bars = make_bars(20)
    p = ex.daily_panel(ex.intraday_macd(bars), warmup_bars=0)
    day = p.iloc[5]["date"]
    g = bars[bars["date"] == day]
    assert p.iloc[5]["vwap"] == pytest.approx(g["amount"].sum() / g["volume"].sum())


def test_rejects_mismatched_adjustment_basis():
    """价格做过复权、amount/volume 没有 → 必须报错，不能默默算出错位的 bp。"""
    bars = make_bars(20)
    bars = bars.assign(**{c: bars[c] * 0.85 for c in ("open", "high", "low", "close")})
    with pytest.raises(ValueError, match="close/vwap"):
        ex.daily_panel(ex.intraday_macd(bars), warmup_bars=0)


def test_go_price_uses_next_bar_open_not_go_bar_close():
    """信号柱的收盘价是前视价；可成交的是下一根开盘价。"""
    bars = ex.intraday_macd(make_bars(40, seed=3))
    p = ex.daily_panel(bars, warmup_bars=0)
    hit = p[p["has_go"] & p["go_time"].notna() & (p["go_time"] != "15:00")]
    assert len(hit) > 0, "构造的数据里应当出现 GO"
    row = hit.iloc[0]
    day = bars[bars["date"] == row["date"]].reset_index(drop=True)
    i = day.index[day["dt"].dt.strftime("%H:%M") == row["go_time"]][0]
    assert row["go_price"] == pytest.approx(day.loc[i + 1, "open"])
    assert row["go_price"] != pytest.approx(day.loc[i, "close"])


def test_go_price_falls_back_to_close_when_no_go():
    bars = ex.intraday_macd(make_bars(30, seed=11))
    p = ex.daily_panel(bars, warmup_bars=0)
    no_go = p[~p["has_go"]]
    if no_go.empty:
        pytest.skip("该随机种子下每天都有 GO")
    assert (no_go["go_price"] == no_go["close"]).all()


def test_limit_plan_requires_fallback():
    p = ex.daily_panel(ex.intraday_macd(make_bars(20)), warmup_bars=0)
    with pytest.raises(ValueError, match="兜底"):
        ex.add_limit_plan(p, offset=-0.005, fallback="不存在的列")


def test_limit_plan_forces_fill_when_untouched():
    """挂得离谱的限价单不能"不成交就不买"，必须按兜底价成交。"""
    p = ex.daily_panel(ex.intraday_macd(make_bars(25)), warmup_bars=0)
    out = ex.add_limit_plan(p, offset=-0.5, fallback="close", name="deep")
    assert not out["deep_filled"].any()
    assert (out["deep"] == out["close"]).all()          # 全部走兜底，无一缺失
    assert out["deep"].notna().all()


def test_limit_plan_fill_uses_rest_low_not_day_low():
    """开盘那一刻还没挂上单，用首根的下影线判成交会高估成交率。"""
    p = ex.daily_panel(ex.intraday_macd(make_bars(25, seed=5)), warmup_bars=0)
    p = p.copy()
    p.loc[p.index[0], "day_low"] = p.loc[p.index[0], "open"] * 0.5   # 首根挖深坑
    out = ex.add_limit_plan(p, offset=-0.2, fallback="close")
    assert not out.iloc[0][out.columns[-1]]              # 仍判为未成交


def test_benchmark_zero_bp_when_buying_at_vwap():
    p = ex.daily_panel(ex.intraday_macd(make_bars(60)), warmup_bars=0)
    p["at_vwap"] = p["vwap"]
    t = ex.benchmark(p, ["at_vwap"])
    assert t.iloc[0]["均值bp"] == pytest.approx(0.0, abs=1e-9)
    assert not bool(t.iloc[0]["signif"])


def test_benchmark_reports_fill_rate_and_signif():
    p = ex.daily_panel(ex.intraday_macd(make_bars(80, seed=7)), warmup_bars=0)
    p = ex.add_limit_plan(p, offset=-0.005, fallback="close", name="lim")
    t = ex.benchmark(p, ["open", "lim"]).set_index("方案")
    assert t.loc["open", "成交率"] == "100%"
    assert t.loc["lim", "成交率"].endswith("%")
    assert set(t["signif"]) <= {True, False}


def test_daily_panel_missing_column_raises():
    bars = make_bars(10).drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        ex.daily_panel(bars, warmup_bars=0)


def test_split_by_go_returns_both_branches():
    p = ex.daily_panel(ex.intraday_macd(make_bars(80, seed=13)), warmup_bars=0)
    t = ex.split_by_go(p)
    assert "天数" in t.columns
    assert t["天数"].sum() == len(p)


def test_go_buy_matches_jcy_intraday_timing():
    """与 jcy_intraday_timing 的买入 GO 定义必须一致，否则两处会悄悄漂移。"""
    jit = pytest.importorskip("jcy_intraday_timing")
    bars = make_bars(30, seed=17)
    mine = ex.intraday_macd(bars)

    theirs = bars.set_index("dt")[["open", "high", "low", "close", "volume"]]
    theirs = jit.classify_timing(jit.add_macd(theirs), "buy")
    assert (mine["go_buy"].values == (theirs["timing"].values == "GO")).all()
