"""批量汇总层测试：横截面表与等权组合曲线（不联网、不绘图落盘到项目目录）。"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from batch_report import (
    SUMMARY_COLUMNS, build_portfolio_curve, build_summary,
    normalized_equity, result_to_row, write_batch_report,
)


def _fake_result(total, bench, *, n=100, capital=100_000.0, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n)
    equity = pd.Series(np.linspace(capital, capital * (1 + total / 100), n),
                       index=idx)
    return {
        "initial_capital": capital,
        "equity_base": capital,
        "equity_curve": pd.DataFrame({"equity": equity}, index=idx),
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


def _cand(code, name, date="20240102"):
    return {"code": code, "name": name, "date": date, "reason": "x"}


def test_result_to_row_has_all_columns_and_excess():
    row = result_to_row(_cand("600000", "浦发"), _fake_result(20.0, 8.0, n=100))
    assert set(row) == set(SUMMARY_COLUMNS)
    assert row["超额收益%"] == pytest.approx(12.0)
    assert row["日均超额bp"] == pytest.approx(12.0 * 100 / 100)
    assert row["成本占比%"] == pytest.approx(0.42)


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
    from engine import run_backtest
    from strategies.base import BaseStrategy

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

    row = result_to_row(_cand("600000", "测试"), result)
    assert set(row) == set(SUMMARY_COLUMNS)
    assert row["成本占比%"] > 0, "真实回测必然产生交易成本"
    # 两列各自四舍五入到 2 位，只能在舍入误差内相等
    assert row["日均超额bp"] == pytest.approx(
        row["超额收益%"] * 100 / row["统计交易日数"], abs=0.01)

    curves = {"600000 测试": normalized_equity(result)}
    assert curves["600000 测试"].iloc[0] == pytest.approx(1.0)

    summary = write_batch_report([row], curves, str(tmp_path))
    assert len(summary) == 1
    assert (tmp_path / "summary.csv").exists()
