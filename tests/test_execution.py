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


# ── 卖出侧 ───────────────────────────────────────────────────────────────────

def test_go_sell_matches_jcy_intraday_timing():
    """卖出 GO 同样要与 jcy_intraday_timing 对齐（红柱缩短 或 死叉）。"""
    jit = pytest.importorskip("jcy_intraday_timing")
    bars = make_bars(30, seed=17)
    mine = ex.intraday_macd(bars)

    theirs = bars.set_index("dt")[["open", "high", "low", "close", "volume"]]
    theirs = jit.classify_timing(jit.add_macd(theirs), "sell")
    assert (mine["go_sell"].values == (theirs["timing"].values == "GO")).all()


def test_sell_go_is_looser_than_buy_go():
    """卖侧只要动能转弱就放行，天数占比必然高于买侧——两边 bp 不可横向比。"""
    bars = ex.intraday_macd(make_bars(120, seed=21))
    buy = ex.daily_panel(bars, side="buy", warmup_bars=0)
    sell = ex.daily_panel(bars, side="sell", warmup_bars=0)
    assert sell["has_go"].mean() > buy["has_go"].mean()


def test_neutral_columns_identical_across_sides():
    """开盘/收盘/VWAP 是中性的日内形状，换侧不该变；变的只是好坏。"""
    bars = ex.intraday_macd(make_bars(60, seed=23))
    buy = ex.daily_panel(bars, side="buy", warmup_bars=0)
    sell = ex.daily_panel(bars, side="sell", warmup_bars=0)
    for c in ("vwap", "open", "close", "day_low", "day_high"):
        assert (buy[c].values == sell[c].values).all()


def test_edge_sign_flips_between_sides():
    """同一个固定时点方案，买卖两侧的优势bp 互为相反数。"""
    bars = ex.intraday_macd(make_bars(80, seed=27))
    p = ex.daily_panel(bars, warmup_bars=0)
    b = ex.benchmark(p, ["open"], side="buy").iloc[0]
    s = ex.benchmark(p, ["open"], side="sell").iloc[0]
    assert b["均值bp"] == pytest.approx(s["均值bp"])          # 原始偏离中性
    assert b["优势bp"] == pytest.approx(-s["优势bp"])         # 好坏才分侧


def test_sell_limit_plan_fills_on_rest_high():
    """卖侧挂高价，成交与否看首根之后的最高价，不是最低价。"""
    p = ex.daily_panel(ex.intraday_macd(make_bars(40, seed=31)), warmup_bars=0)
    out = ex.add_limit_plan(p, offset=0.005, fallback="close", side="sell")
    col = "limit_+50bp"
    expected = p["rest_high"] >= p["open"] * 1.005
    assert (out[col + "_filled"].values == expected.values).all()
    assert out.loc[expected, col].values == pytest.approx(
        (p.loc[expected, "open"] * 1.005).values)


def test_sell_limit_plan_forces_fill_when_untouched():
    """卖侧「没成交就继续拿着」= 把择时失败变成留仓，代价不在表里，必须兜底。"""
    p = ex.daily_panel(ex.intraday_macd(make_bars(30, seed=33)), warmup_bars=0)
    out = ex.add_limit_plan(p, offset=0.5, fallback="close", name="high",
                            side="sell")
    assert not out["high_filled"].any()
    assert (out["high"] == out["close"]).all()


def test_limit_plan_rejects_wrong_side_offset():
    """挂在错误一侧会立刻成交，测的就不是「等价格」这件事了。"""
    p = ex.daily_panel(ex.intraday_macd(make_bars(20)), warmup_bars=0)
    with pytest.raises(ValueError, match="买侧"):
        ex.add_limit_plan(p, offset=+0.005, fallback="close", side="buy")
    with pytest.raises(ValueError, match="卖侧"):
        ex.add_limit_plan(p, offset=-0.005, fallback="close", side="sell")


def test_wait_value_excludes_rows_where_plan_equals_close():
    """无 GO 的日子方案价兜底成收盘价，恒等于 0，留着会稀释均值和 t 值。"""
    bars = ex.intraday_macd(make_bars(150, seed=37))
    p = ex.daily_panel(bars, side="sell", warmup_bars=0)
    n_diff = int((p["go_price"] != p["close"]).sum())
    wv = ex.wait_value(p, side="sell")
    assert not wv.empty and wv.iloc[0]["n"] == n_diff
    assert n_diff < len(p), "该样本应当存在无 GO 的日子"


def test_wait_value_measures_move_after_fill():
    """均值就是"成交价 → 收盘价"的平均涨跌，正 = 成交后价格继续上行。"""
    bars = ex.intraday_macd(make_bars(150, seed=41))
    p = ex.daily_panel(bars, side="sell", warmup_bars=0)
    d = p[p["go_price"] != p["close"]]
    expect = ((d["close"] / d["go_price"] - 1) * 1e4).mean()
    assert ex.wait_value(p, side="sell").iloc[0]["均值bp"] == pytest.approx(expect)


def test_unknown_side_rejected():
    p = ex.daily_panel(ex.intraday_macd(make_bars(20)), warmup_bars=0)
    with pytest.raises(ValueError, match="side"):
        ex.benchmark(p, ["open"], side="hold")
