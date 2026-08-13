"""
参数扫描层测试：样本内/外切分与网格矩阵（不联网、不绘图到项目目录）。

重点守住两件事：
  1. 样本外集合必须真的没参与选参数——同一天推荐的标的不得跨界
  2. None（"不启用止盈/止损"）是合法的参数取值，不能在网格里被静默丢掉
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from backtest.scripts.sweep_params import (AXES, DEFAULTS, METRICS, build_matrix, fmt_value,
                         parse_args, resolve_universe, split_candidates)


def _cands(dates):
    return [{"code": f"60000{i}", "name": f"S{i}", "date": d, "reason": ""}
            for i, d in enumerate(dates)]


# ── 股票池注入（扫描机制与"票从哪来"之间的唯一接口）────────────────────────────

def test_explicit_codes_bypass_the_jcy_json(monkeypatch):
    """给了 --codes 就不该去碰 jcy_insights.json——这正是解绑的意义。"""
    import backtest.scripts.sweep_params as param_sweep
    monkeypatch.setattr(param_sweep, "load_candidates",
                        lambda *a, **k: pytest.fail("显式池不该读 JCY JSON"))
    monkeypatch.setattr(param_sweep, "JSON_PATH", "/nonexistent/nope.json")

    args = parse_args_for(["--codes", "601857", "600938",
                           "--codes-start", "20180101"])
    cands, label = resolve_universe(args)
    assert [c["code"] for c in cands] == ["601857", "600938"]
    assert all(c["date"] == "20180101" for c in cands)
    assert "显式" in label


def test_jcy_pool_is_still_the_default(monkeypatch):
    import backtest.scripts.sweep_params as param_sweep
    monkeypatch.setattr(param_sweep, "load_candidates",
                        lambda path, ratings: _cands(["20240101", "20240202"]))
    monkeypatch.setattr(param_sweep.os.path, "exists", lambda p: True)
    cands, label = resolve_universe(parse_args_for([]))
    assert len(cands) == 2
    assert "JCY" in label


def test_universe_entries_carry_only_what_evaluate_combo_needs():
    """下游只用 code / date 两个键；接口契约别悄悄变宽。"""
    cands, _ = resolve_universe(parse_args_for(["--codes", "601857"]))
    assert {"code", "date"} <= set(cands[0])


def parse_args_for(argv):
    import sys
    old = sys.argv
    sys.argv = ["sweep_params.py", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = old


# ── 样本内 / 样本外切分 ───────────────────────────────────────────────────────

def test_split_is_disabled_by_default():
    c = _cands(["20240101", "20240201", "20240301", "20240401"])
    is_, oos = split_candidates(c, 0.0)
    assert is_ == c and oos == []


def test_split_puts_later_recommendations_out_of_sample():
    c = _cands(["20240101", "20240201", "20240301", "20240401",
                "20240501", "20240601"])
    is_, oos = split_candidates(c, 0.34)

    assert len(is_) + len(oos) == len(c)
    assert oos, "应切出样本外集合"
    assert max(x["date"] for x in is_) < min(x["date"] for x in oos)


def test_split_never_straddles_a_recommendation_date():
    """
    同一天推荐的标的必须整组落在同一侧：否则同一篇研报的股票会同时出现在
    选参集和验证集里，样本外就被污染了。
    """
    c = _cands(["20240101", "20240201", "20240201", "20240201", "20240301"])
    is_, oos = split_candidates(c, 0.5)

    is_dates = {x["date"] for x in is_}
    oos_dates = {x["date"] for x in oos}
    assert is_dates.isdisjoint(oos_dates)


def test_split_degrades_gracefully_on_tiny_or_uniform_samples():
    assert split_candidates(_cands(["20240101"] * 3), 0.3)[1] == []      # 太少
    assert split_candidates(_cands(["20240101"] * 8), 0.3)[1] == []      # 全同日
    assert split_candidates(_cands(["20240101", "20240201"]), 0.99)[1] == []


# ── 网格矩阵 ──────────────────────────────────────────────────────────────────

def test_build_matrix_keeps_none_valued_axis():
    """None = 不设止盈，恰恰是默认值；pivot_table 会把它当 NaN 丢掉。"""
    cells = {(None, 2): 1.0, (0.10, 2): -2.0, (None, 3): 3.0, (0.10, 3): 0.5}
    matrix, xs, ys = build_matrix(cells, "take_profit", "expand_bars")

    assert None in xs, "不设止盈这一列不能消失"
    assert matrix.shape == (len(ys), len(xs))
    assert matrix[ys.index(3), xs.index(None)] == pytest.approx(3.0)


def test_build_matrix_follows_declared_axis_order():
    """轴顺序取自 AXES 声明，而不是按值排序（True/False 排序没有意义）。"""
    cells = {(v, True): 1.0 for v in AXES["stop_loss"]["values"]}
    cells.update({(v, False): 0.0 for v in AXES["stop_loss"]["values"]})
    _, xs, ys = build_matrix(cells, "stop_loss", "shrink_exit")

    assert xs == AXES["stop_loss"]["values"]
    assert ys == AXES["shrink_exit"]["values"]


def test_build_matrix_marks_missing_cells_as_nan():
    cells = {(None, 2): 1.0}
    matrix, xs, ys = build_matrix(cells, "take_profit", "expand_bars")
    assert np.isfinite(matrix).sum() == 1


# ── 默认值一致性 ──────────────────────────────────────────────────────────────

def test_defaults_are_valid_points_on_every_axis():
    """默认值必须落在各轴的候选值上，否则热力图标不出"默认格"。"""
    for axis, spec in AXES.items():
        assert axis in DEFAULTS, f"{axis} 缺少默认值"
        assert DEFAULTS[axis] in spec["values"], (
            f"{axis} 的默认值 {DEFAULTS[axis]} 不在候选值 {spec['values']} 中"
        )


def test_defaults_match_batch_cli(monkeypatch):
    """
    扫描的默认值必须与批量回测 CLI 的真实默认值一致，否则热力图上金框标出的
    "默认格"不是实际在跑的那组参数。这里直接调用 batch.parse_args() 取值，
    不复制一份常量——复制出来的测试永远通过，也就永远发现不了漂移。
    """
    import backtest.scripts.backtest_jcy_pool as batch

    monkeypatch.setattr("sys.argv", ["backtest_jcy_pool.py"])
    cli = vars(batch.parse_args())

    for key in ("stop_loss", "take_profit", "shrink_exit"):
        assert DEFAULTS[key] == cli[key], (
            f"{key}: 扫描默认 {DEFAULTS[key]} != 批量回测默认 {cli[key]}"
        )


@pytest.mark.parametrize("raw,expected", [
    ("0.15", 0.15), ("none", None), ("NONE", None),
    ("off", None), ("", None), ("  ", None),
])
def test_ratio_or_none_parses_disable_keywords(raw, expected):
    import backtest.scripts.backtest_jcy_pool as batch
    assert batch._ratio_or_none(raw) == expected


def test_every_metric_is_actually_produced_by_evaluate_combo(monkeypatch):
    """
    METRICS 是 --metric 的 choices，也是 main() 用来取值的键。它必须与
    evaluate_combo 的真实产出对齐：少一个则合法指标被 argparse 拒掉，
    多一个则跑完整个网格才在最后 KeyError。这里打桩掉行情与回测，只验契约。
    """
    import backtest.scripts.sweep_params as param_sweep

    monkeypatch.setattr(param_sweep, "_cached_stock",
                        lambda *a: pd.DataFrame({"close": range(100)}))
    monkeypatch.setattr(param_sweep, "BullStrategyAdapter",
                        lambda *a, **kw: None)
    monkeypatch.setattr(param_sweep, "run_backtest",
                        lambda **kw: {"total_return": 12.0, "benchmark_return": 5.0,
                                      "max_drawdown": -8.0, "total_trades": 3,
                                      "exposure_pct": 35.0,
                                      "equity_curve": pd.DataFrame(index=range(70))})

    row = param_sweep.evaluate_combo({}, _cands(["20240101", "20240201"]),
                                     None, "20241231", 100_000.0)

    assert row["样本数"] == 2
    for m in METRICS:
        assert m in row, f"METRICS 收录了 evaluate_combo 并不产出的键：{m}"
    # 日均口径必须真的按窗口长度归一：+7% 超额 / 70 个交易日 = 10bp/日
    assert row["日均超额中位bp"] == pytest.approx(10.0)


def test_daily_metric_is_the_default_not_total_excess():
    """
    选参数的默认指标必须是对窗口长度归一的那个。用总超额%选参，选出来的是
    "哪组参数恰好被长窗口标的占了多数"——batch_report 已经论证过它不可比。
    """
    import sys
    from unittest import mock

    with mock.patch.object(sys, "argv", ["sweep_params.py"]):
        assert parse_args().metric == "日均超额中位bp"


def test_bad_metric_is_rejected_before_the_grid_runs(monkeypatch):
    """拼错指标名要在 argparse 阶段就退出，而不是跑完几十分钟的网格再崩。"""
    monkeypatch.setattr("sys.argv", ["sweep_params.py", "--metric", "不存在的指标"])
    with pytest.raises(SystemExit):
        parse_args()


def test_fmt_value_renders_none_and_bool_readably():
    assert fmt_value(None) == "关闭"          # 不能显示成空白
    assert fmt_value(True) == "是"
    assert fmt_value(0.10) == "0.1"
    assert fmt_value(12) == "12"
