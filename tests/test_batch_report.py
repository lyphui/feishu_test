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
        "equity_curve": pd.DataFrame({"equity": equity}, index=idx),
        "total_return": total,
        "benchmark_return": bench,
        "max_drawdown": -8.0,
        "sharpe_ratio": 1.1,
        "total_trades": 3,
        "win_rate": 66.7,
        "profit_factor": 1.8,
        "blocked_trades": pd.DataFrame(),
    }


def _cand(code, name, date="20240102"):
    return {"code": code, "name": name, "date": date, "reason": "x"}


def test_result_to_row_has_all_columns_and_excess():
    row = result_to_row(_cand("600000", "浦发"), _fake_result(20.0, 8.0))
    assert set(row) == set(SUMMARY_COLUMNS)
    assert row["超额收益%"] == pytest.approx(12.0)


def test_build_summary_sorted_by_excess_desc():
    rows = [
        result_to_row(_cand("600000", "A"), _fake_result(5.0, 10.0)),   # -5
        result_to_row(_cand("600001", "B"), _fake_result(30.0, 10.0)),  # +20
        result_to_row(_cand("600002", "C"), _fake_result(12.0, 10.0)),  # +2
    ]
    df = build_summary(rows)
    assert list(df["超额收益%"]) == [20.0, 2.0, -5.0]


def test_normalized_equity_starts_at_one():
    r = _fake_result(25.0, 5.0)
    norm = normalized_equity(r)
    assert norm.iloc[0] == pytest.approx(1.0)
    assert norm.iloc[-1] == pytest.approx(1.25)


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
