"""批量汇总层测试：横截面表与等权组合曲线（不联网、不绘图落盘到项目目录）。"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from backtest.reports.batch_report import (
    SUMMARY_COLUMNS, build_portfolio_curve, build_summary, compare_rating_pools,
    index_window_return, normalized_equity, result_to_row, write_batch_report,
)


def _fake_result(total, bench, *, n=100, capital=100_000.0, start="2024-01-01",
                 exposure=40.0):
    idx = pd.bdate_range(start, periods=n)
    equity = pd.Series(np.linspace(capital, capital * (1 + total / 100), n),
                       index=idx)
    return {
        "initial_capital": capital,
        "equity_base": capital,
        "equity_curve": pd.DataFrame({"equity": equity}, index=idx),
        "exposure_pct": exposure,
        "avg_holding_days": 4.5,
        "total_return": total,
        "benchmark_return": bench,
        "max_drawdown": -8.0,
        "sharpe_ratio": 1.1,
        "total_trades": 3,
        "win_rate": 66.7,
        "profit_factor": 1.8,
        "blocked_trades": pd.DataFrame(),
        "costs": {"cost_drag_pct": 0.42},
    }


def _cand(code, name, date="20240102", rating="增持"):
    return {"code": code, "name": name, "date": date,
            "rating": rating, "reason": "x"}


def _index(start="2024-01-01", n=100, first=3000.0, last=3300.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"close": np.linspace(first, last, n)}, index=idx)


def test_result_to_row_has_all_columns_and_excess():
    row = result_to_row(_cand("600000", "浦发"), _fake_result(20.0, 8.0, n=100))
    assert set(row) == set(SUMMARY_COLUMNS)
    assert row["超额收益%"] == pytest.approx(12.0)
    assert row["日均超额bp"] == pytest.approx(12.0 * 100 / 100)
    assert row["成本占比%"] == pytest.approx(0.42)


def test_pick_alpha_measures_the_recommendation_not_the_timing():
    """
    两个 alpha 不能混：
      选股alpha% = 基准收益% − 指数收益%  → 研报推荐本身值不值钱
      超额收益%  = 策略收益% − 基准收益%  → MACD 择时加不加分
    这里构造"推荐的票跑赢指数、但策略择时反而拖后腿"，两列必须一正一负。
    """
    # 指数 +10%，个股买入持有 +25%（选股 +15%），策略只做出 +8%（择时 −17%）
    result = _fake_result(8.0, 25.0, n=100)
    row = result_to_row(_cand("600000", "浦发"), result,
                        index_df=_index(first=3000.0, last=3300.0))

    assert row["指数收益%"] == pytest.approx(10.0, abs=0.01)
    assert row["选股alpha%"] == pytest.approx(15.0, abs=0.01)   # 推荐是好的
    assert row["超额收益%"] == pytest.approx(-17.0)             # 择时是差的


def test_pick_alpha_is_none_without_index():
    row = result_to_row(_cand("600000", "浦发"), _fake_result(20.0, 8.0))
    assert row["指数收益%"] is None and row["选股alpha%"] is None
    assert row["超额收益%"] == pytest.approx(12.0)      # 择时口径不依赖指数


def test_index_window_return_tolerates_mismatched_calendars():
    """个股停牌 / 指数休市导致交易日对不齐时，不能整只票的选股 alpha 变空值。"""
    idx = _index(n=100)
    window = pd.bdate_range("2024-01-03", periods=40)   # 起点晚于指数首日
    window = window.union([pd.Timestamp("2024-03-16")])  # 掺一个指数没有的日子
    assert index_window_return(idx, window) is not None
    assert index_window_return(None, window) is None
    assert index_window_return(idx, window[:1]) is None


def test_exposure_columns_come_from_the_engine():
    row = result_to_row(_cand("600000", "浦发"), _fake_result(20.0, 8.0,
                                                              exposure=12.5))
    assert row["在场比例%"] == pytest.approx(12.5)
    assert row["平均持仓天数"] == pytest.approx(4.5)
    assert row["评级"] == "增持"


def test_daily_excess_normalizes_window_length():
    """同样 +12% 超额，持有两年的日均超额必须远低于持有两个月的。"""
    short = result_to_row(_cand("600000", "短"), _fake_result(20.0, 8.0, n=40))
    long_ = result_to_row(_cand("600001", "长"), _fake_result(20.0, 8.0, n=480))

    assert short["超额收益%"] == long_["超额收益%"]
    assert short["日均超额bp"] > long_["日均超额bp"] * 10


def test_build_summary_sorted_by_daily_excess_desc():
    """
    排序键是日均超额而非总超额：窗口长的标的不该仅因为跑得久就排前面。
    B 总超额最高（+20%，400 日 → 5bp/日），C 总超额只有 +2% 但窗口只有
    20 日（→ 10bp/日），按日均排 C 必须在 B 前面。
    """
    rows = [
        result_to_row(_cand("600000", "A"), _fake_result(5.0, 10.0, n=100)),
        result_to_row(_cand("600001", "B"), _fake_result(30.0, 10.0, n=400)),
        result_to_row(_cand("600002", "C"), _fake_result(12.0, 10.0, n=20)),
    ]
    df = build_summary(rows)
    assert list(df["代码"]) == ["600002", "600001", "600000"]
    assert df["日均超额bp"].is_monotonic_decreasing


def test_normalized_equity_starts_at_one():
    r = _fake_result(25.0, 5.0)
    norm = normalized_equity(r)
    assert norm.iloc[0] == pytest.approx(1.0)
    assert norm.iloc[-1] == pytest.approx(1.25)


def test_normalized_equity_uses_equity_base_not_initial_capital():
    """窗口起点权益 != 初始资金时（预热期有成交），曲线仍须从 1.0 起步。"""
    r = _fake_result(25.0, 5.0)
    r["equity_base"] = r["initial_capital"] * 1.5     # 预热期已赚了 50%
    r["equity_curve"]["equity"] *= 1.5

    norm = normalized_equity(r)
    assert norm.iloc[0] == pytest.approx(1.0)


def test_portfolio_curve_averages_only_active_names():
    """不同推荐日的股票：未入场的不得参与当日平均。"""
    a = pd.Series([1.0, 1.1, 1.2], index=pd.bdate_range("2024-01-01", periods=3))
    b = pd.Series([1.0, 2.0], index=pd.bdate_range("2024-01-02", periods=2))

    port = build_portfolio_curve({"A": a, "B": b})

    assert port.loc[a.index[0], "n_active"] == 1
    assert port.loc[a.index[0], "portfolio"] == pytest.approx(1.0)   # 只有 A
    assert port.loc[a.index[1], "n_active"] == 2
    assert port.loc[a.index[1], "portfolio"] == pytest.approx((1.1 + 1.0) / 2)
    assert port.loc[a.index[2], "portfolio"] == pytest.approx((1.2 + 2.0) / 2)


def test_portfolio_curve_forward_fills_suspended_names():
    """个股停牌导致缺日时，用前值延续，不能让它退出平均。"""
    idx = pd.bdate_range("2024-01-01", periods=4)
    a = pd.Series([1.0, 1.1, 1.2, 1.3], index=idx)
    b = pd.Series([1.0, 1.5], index=[idx[0], idx[3]])      # 中间两天停牌

    port = build_portfolio_curve({"A": a, "B": b})
    assert (port["n_active"] == 2).all()
    assert port.loc[idx[2], "portfolio"] == pytest.approx((1.2 + 1.0) / 2)


def test_portfolio_curve_empty_input():
    assert build_portfolio_curve({}).empty


def test_write_batch_report_creates_files(tmp_path):
    rows = [
        result_to_row(_cand("600000", "A"), _fake_result(30.0, 10.0)),
        result_to_row(_cand("600001", "B"), _fake_result(-5.0, 10.0)),
    ]
    curves = {
        "600000 A": normalized_equity(_fake_result(30.0, 10.0)),
        "600001 B": normalized_equity(_fake_result(-5.0, 10.0)),
    }
    idx = pd.bdate_range("2024-01-01", periods=100)
    index_df = pd.DataFrame({"close": np.linspace(3000, 3300, 100)}, index=idx)

    summary = write_batch_report(rows, curves, str(tmp_path), index_df=index_df)

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary_portfolio.csv").exists()
    assert (tmp_path / "summary_portfolio.png").exists()
    assert len(summary) == 2


def test_write_batch_report_handles_no_results(tmp_path):
    summary = write_batch_report([], {}, str(tmp_path))
    assert summary.empty
    assert not (tmp_path / "summary.csv").exists()


# ── 引擎 → 汇总层的契约 ───────────────────────────────────────────────────────

def test_real_engine_result_feeds_the_report_layer(tmp_path):
    """
    用真实 run_backtest 的输出跑一遍汇总，守住两层之间的字段契约：
    汇总层读的 equity_base / costs.cost_drag_pct 等键必须真的由引擎产出。
    _fake_result 是手写的，改了引擎它不会报错——这个测试才会。
    """
    from backtest.engine import run_backtest
    from backtest.strategies.base import BaseStrategy

    class _Scripted(BaseStrategy):
        name = "scripted"
        params: dict = {}

        def prepare(self, df):
            df = df.copy()
            df["signal"] = 0
            df.iloc[10, df.columns.get_loc("signal")] = 1
            df.iloc[60, df.columns.get_loc("signal")] = -1
            return df

        def plot_indicators(self, ax, df, colors):
            pass

    n = 120
    idx = pd.bdate_range("2024-01-01", periods=n)
    px = pd.Series(np.linspace(10.0, 13.0, n), index=idx)
    df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                       "close": px, "volume": 1e6}, index=idx)

    result = run_backtest("600000", "20240101", "20241231",
                          strategy=_Scripted(), df=df, limit_move_check=False)

    row = result_to_row(_cand("600000", "测试"), result, index_df=_index(n=n))
    assert set(row) == set(SUMMARY_COLUMNS)
    assert row["成本占比%"] > 0, "真实回测必然产生交易成本"
    # 在场比例必须由引擎真的产出：脚本策略第 10 根发买入信号、第 60 根发卖出，
    # 都在次日开盘成交 → 收盘时持仓的是第 11~60 根，共 50 天 / 120 天
    assert row["在场比例%"] == pytest.approx(50 / 120 * 100, abs=0.1)
    assert row["平均持仓天数"] == pytest.approx(50.0, abs=0.5)
    assert row["选股alpha%"] is not None, "传了 index_df 就必须算得出选股 alpha"
    # 两列各自四舍五入到 2 位，只能在舍入误差内相等
    assert row["日均超额bp"] == pytest.approx(
        row["超额收益%"] * 100 / row["统计交易日数"], abs=0.01)

    curves = {"600000 测试": normalized_equity(result)}
    assert curves["600000 测试"].iloc[0] == pytest.approx(1.0)

    summary = write_batch_report([row], curves, str(tmp_path))
    assert len(summary) == 1
    assert (tmp_path / "summary.csv").exists()


# ── 评级对照（正向池 vs 看空池） ──────────────────────────────────────────────

def _pool(alphas, start="2024-01-01"):
    """按给定的选股 alpha 造一批行（其余指标不影响对照表）。"""
    rows = []
    for i, a in enumerate(alphas):
        # 指数 +10%，令 基准收益% = 10 + a，则 选股alpha% = a
        r = _fake_result(5.0, 10.0 + a, n=100, start=start)
        rows.append(result_to_row(_cand(f"60000{i}", f"S{i}"), r,
                                  index_df=_index(start=start)))
    return build_summary(rows)


def test_compare_rating_pools_detects_a_working_rating(tmp_path, capsys):
    """看多池选股 alpha 明显高于看空池 → 评级方向正确。"""
    long_df = _pool([12.0, 8.0, 15.0])
    ctrl_df = _pool([-6.0, -9.0, -3.0])

    out = compare_rating_pools(long_df, ctrl_df, str(tmp_path))
    pick = out[out["指标"] == "选股alpha中位%"].iloc[0]

    assert pick["差值"] > 0
    assert (tmp_path / "summary_rating_compare.csv").exists()
    assert "评级方向正确" in capsys.readouterr().out


def test_compare_rating_pools_calls_out_a_useless_rating(tmp_path, capsys):
    """
    看多池并不优于看空池时必须直说没有区分度——只跑看多池会把
    "这段行情什么都涨"读成"评级有效"，对照组存在的意义就在这里。
    """
    long_df = _pool([2.0, -1.0, 0.5])
    ctrl_df = _pool([9.0, 11.0, 7.0])

    out = compare_rating_pools(long_df, ctrl_df, str(tmp_path))
    assert out[out["指标"] == "选股alpha中位%"].iloc[0]["差值"] < 0
    assert "没有区分度" in capsys.readouterr().out


def test_compare_warns_that_it_is_not_a_significance_test(tmp_path, capsys):
    compare_rating_pools(_pool([5.0, 6.0]), _pool([1.0, 2.0]), str(tmp_path))
    assert "不是显著性检验" in capsys.readouterr().out
