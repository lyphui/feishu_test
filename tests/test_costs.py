"""
成本口径的单一真值源守护测试。

这些断言存在的理由：`engine.py`（二元仓位）与 `lib/ladder.py`（连续仓位）是两套
独立的撮合骨架，但成本假设必须逐位相同——否则"梯度加仓 vs 满仓持有""网格 vs
静态同敞口"这类横向比较比的是费率不是策略。以前靠 docstring 写一句"与 engine.py
保持一致"人肉同步，这里改成机器守。
"""

import inspect

import pytest

import engine
from lib import costs, ladder


def test_engine_defaults_come_from_the_shared_source():
    """引擎的四个成本默认值必须就是 lib/costs.py 的值，不能另写字面量。"""
    sig = inspect.signature(engine.run_backtest).parameters
    assert sig["commission_rate"].default == costs.COMMISSION_RATE
    assert sig["min_commission"].default == costs.MIN_COMMISSION
    assert sig["stamp_duty"].default == costs.STAMP_DUTY
    assert sig["slippage"].default == costs.SLIPPAGE


def test_ladder_uses_the_same_constants_as_engine():
    """分批建仓模拟器与引擎同源——两者漂开会让敞口对齐口径失效。"""
    assert ladder.COMMISSION_RATE == costs.COMMISSION_RATE
    assert ladder.MIN_COMMISSION == costs.MIN_COMMISSION
    assert ladder.STAMP_DUTY == costs.STAMP_DUTY
    assert ladder.SLIPPAGE == costs.SLIPPAGE
    assert ladder.LOT == costs.LOT


def test_fatfinger_inherits_the_same_constants():
    """乌龙指模拟器经 lib.ladder 转引，也必须在同一口径内。"""
    from lib import fatfinger
    assert fatfinger.SLIPPAGE == costs.SLIPPAGE
    assert fatfinger.STAMP_DUTY == costs.STAMP_DUTY
    assert fatfinger.LOT == costs.LOT
    assert fatfinger.COMMISSION_RATE == costs.COMMISSION_RATE


def test_config_ini_defaults_match_the_shared_source():
    """`.ini` 留空时落到的缺省值也必须同源。"""
    from config import BacktestConfig, execution_kwargs
    # extra 为空 = .ini 里那几行全部留空
    cfg = BacktestConfig(symbol="600519", name="x",
                         start_date="20200101", end_date="20241231")
    kw = execution_kwargs(cfg)
    assert kw["commission_rate"] == costs.COMMISSION_RATE
    assert kw["min_commission"] == costs.MIN_COMMISSION
    assert kw["stamp_duty"] == costs.STAMP_DUTY
    assert kw["slippage"] == costs.SLIPPAGE


def test_commission_respects_the_floor():
    assert costs.commission(1_000) == costs.MIN_COMMISSION      # 万三 = 0.3 元 < 5
    assert costs.commission(100_000) == pytest.approx(30.0)     # 万三 = 30 元 > 5
    assert costs.commission(1_000, rate=0.001, minimum=0) == pytest.approx(1.0)


@pytest.mark.parametrize("symbol,expected", [
    ("600519", 0.10), ("000001", 0.10),
    ("688111", 0.20), ("300750", 0.20), ("301236", 0.20),
    ("430047", 0.30), ("830799", 0.30),
])
def test_infer_limit_pct(symbol, expected):
    assert costs.infer_limit_pct(symbol) == expected


def test_engine_reexports_infer_limit_pct():
    """历史导入 `from engine import infer_limit_pct` 必须继续可用。"""
    assert engine.infer_limit_pct is costs.infer_limit_pct
