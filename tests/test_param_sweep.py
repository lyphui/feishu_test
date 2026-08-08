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

from param_sweep import (AXES, DEFAULTS, METRICS, build_matrix, fmt_value,
                         parse_args, split_candidates)


def _cands(dates):
    return [{"code": f"60000{i}", "name": f"S{i}", "date": d, "reason": ""}
            for i, d in enumerate(dates)]


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
    import jcy_macd_bull_batch as batch

    monkeypatch.setattr("sys.argv", ["jcy_macd_bull_batch.py"])
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
    import jcy_macd_bull_batch as batch
    assert batch._ratio_or_none(raw) == expected


def test_every_metric_is_actually_produced_by_evaluate_combo(monkeypatch):
    """
    METRICS 是 --metric 的 choices，也是 main() 用来取值的键。它必须与
    evaluate_combo 的真实产出对齐：少一个则合法指标被 argparse 拒掉，
    多一个则跑完整个网格才在最后 KeyError。这里打桩掉行情与回测，只验契约。
    """
    import param_sweep

    monkeypatch.setattr(param_sweep, "_cached_stock",
                        lambda *a: pd.DataFrame({"close": range(100)}))
    monkeypatch.setattr(param_sweep, "BullStrategyAdapter",
                        lambda *a, **kw: None)
    monkeypatch.setattr(param_sweep, "run_backtest",
                        lambda **kw: {"total_return": 12.0, "benchmark_return": 5.0,
                                      "max_drawdown": -8.0, "total_trades": 3})

    row = param_sweep.evaluate_combo({}, _cands(["20240101", "20240201"]),
                                     None, "20241231", 100_000.0)

    assert row["样本数"] == 2
    for m in METRICS:
        assert m in row, f"METRICS 收录了 evaluate_combo 并不产出的键：{m}"


def test_bad_metric_is_rejected_before_the_grid_runs(monkeypatch):
    """拼错指标名要在 argparse 阶段就退出，而不是跑完几十分钟的网格再崩。"""
    monkeypatch.setattr("sys.argv", ["param_sweep.py", "--metric", "不存在的指标"])
    with pytest.raises(SystemExit):
        parse_args()


def test_fmt_value_renders_none_and_bool_readably():
    assert fmt_value(None) == "关闭"          # 不能显示成空白
    assert fmt_value(True) == "是"
    assert fmt_value(0.10) == "0.1"
    assert fmt_value(12) == "12"
