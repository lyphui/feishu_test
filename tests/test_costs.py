"""
成本口径的单一真值源守护测试。

这些断言存在的理由：`engine.py`（二元仓位）与 `lib/ladder.py`（连续仓位）是两套
独立的撮合骨架，但成本假设必须逐位相同——否则"梯度加仓 vs 满仓持有""网格 vs
静态同敞口"这类横向比较比的是费率不是策略。以前靠 docstring 写一句"与 engine.py
保持一致"人肉同步，这里改成机器守。
"""

import inspect

import pytest

import backtest.engine as engine
from backtest.lib import costs, ladder


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
    from backtest.lib import fatfinger
    assert fatfinger.SLIPPAGE == costs.SLIPPAGE
    assert fatfinger.STAMP_DUTY == costs.STAMP_DUTY
    assert fatfinger.LOT == costs.LOT
    assert fatfinger.COMMISSION_RATE == costs.COMMISSION_RATE


def test_config_ini_defaults_match_the_shared_source():
    """`.ini` 留空时落到的缺省值也必须同源。"""
    from backtest.config import BacktestConfig, execution_kwargs
    # extra 为空 = .ini 里那几行全部留空
    cfg = BacktestConfig(symbol="600519", name="x",
                         start_date="20200101", end_date="20241231")
    kw = execution_kwargs(cfg)
    assert kw["commission_rate"] == costs.COMMISSION_RATE
    assert kw["min_commission"] == costs.MIN_COMMISSION
    assert kw["stamp_duty"] == costs.STAMP_DUTY
    assert kw["slippage"] == costs.SLIPPAGE


# ── 无风险利率 / 现金利率 ──────────────────────────────────────────────────────
#
# 费率有上面的同源守护，rf 与 cash_rate 曾是唯一没有测试守的漂移面：
# 三处夏普各写一份 0.02 字面量，哪天改了一处，同一张对比表的排序就变成
# 按公式排序。收编进 costs.py 后由这里钉住。

def test_sharpe_rf_defaults_come_from_the_shared_source():
    """engine / compare_playbooks 的夏普默认 rf 必须同源。"""
    sig = inspect.signature(engine._calc_sharpe).parameters
    assert sig["rf"].default == costs.RISK_FREE_RATE

    from backtest.scripts import compare_playbooks
    sig = inspect.signature(compare_playbooks._sharpe).parameters
    assert sig["rf"].default == costs.RISK_FREE_RATE


def test_ladder_cash_rate_comes_from_the_shared_source():
    """ladder 空仓计息利率必须同源——它是分批打法净值的一部分。"""
    sig = inspect.signature(ladder._run).parameters
    assert sig["cash_rate"].default == costs.CASH_RATE


# ── 港股成本（评审项 6：自 trend_stop 收编） ──────────────────────────────────

def test_hk_costs_have_exactly_one_implementation():
    """trend_stop 必须是 costs 的消费者而非定义者（保留同名 re-export）。"""
    from backtest.lib import trend_stop
    assert trend_stop.hk_trade_cost is costs.hk_trade_cost
    assert trend_stop.hk_fee_rate is costs.hk_fee_rate


def test_for_market_aggregates_the_same_constants():
    """market-aware 入口返回的费率组必须与散常量逐位一致。"""
    a = costs.for_market("A")
    assert (a.commission_rate, a.min_commission, a.stamp_duty, a.slippage, a.lot) == \
        (costs.COMMISSION_RATE, costs.MIN_COMMISSION, costs.STAMP_DUTY,
         costs.SLIPPAGE, costs.LOT)
    hk = costs.for_market("HK")
    assert (hk.commission_rate, hk.min_commission, hk.stamp_duty,
            hk.platform_fee) == (costs.HK_COMMISSION_RATE, costs.HK_MIN_COMMISSION,
                                 costs.HK_STAMP_DUTY, costs.HK_PLATFORM_FEE)
    assert hk.etf_stamp_exempt and hk.lot is None
    with pytest.raises(ValueError):
        costs.for_market("US")


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


@pytest.mark.parametrize("name,expected", [
    ("ST张三", 0.05), ("*ST李四", 0.05), ("st某某", 0.05), ("SST前锋", 0.05),
    ("  *ST带空格", 0.05),
    ("贵州茅台", 0.10), ("", 0.10), (None, 0.10),
    # 只认前缀：风险警示标识只出现在名称开头，含 ST 二字的普通名称不该被误判
    # ——误判会让 ±10% 的票按 ±5% 撮合，成交被 tradability 静默拦掉
    ("某某ST控股", 0.10), ("EAST科技", 0.10), ("BEST集团", 0.10),
])
def test_infer_limit_pct_with_name(name, expected):
    """主板 ST 靠名称判定；无名称时维持代码前缀推断（评审项 7）。"""
    assert costs.infer_limit_pct("600519", name) == expected


@pytest.mark.parametrize("symbol,expected", [
    ("300750", 0.20), ("301236", 0.20),     # 创业板注册制：ST 也是 20%
    ("688111", 0.20), ("689009", 0.20),     # 科创板：ST 也是 20%
    ("430047", 0.30), ("830799", 0.30),     # 北交所：ST 也是 30%
])
def test_board_takes_precedence_over_st(symbol, expected):
    """
    板块必须先于 ST 判定：±5% 的风险警示幅度**只适用于主板**。

    曾经 ST 分支排在板块前缀之前，`300xxx` 的 ST 票被判成 ±5%——
    任何超过 5% 的开盘都会被 `tradability` 当成封板，成交静默拦掉、
    「受阻次数」虚高。
    """
    assert costs.infer_limit_pct(symbol, "*ST某某") == expected


@pytest.mark.parametrize("name,expected", [
    ("ST路桥", True), ("*ST海投", True), ("sst前锋", True), (" ST带空格", True),
    ("某某ST", False), ("EAST科技", False), ("", False), (None, False),
])
def test_is_st_name(name, expected):
    assert costs.is_st_name(name) is expected


def test_engine_reexports_infer_limit_pct():
    """历史导入 `from backtest.engine import infer_limit_pct` 必须继续可用。"""
    assert engine.infer_limit_pct is costs.infer_limit_pct


# ── 成交约束（涨跌停 / 停牌） ──────────────────────────────────────────────────
#
# 与上面的成本常量同理：engine 与 ladder 曾各存一份成交判定且**行为不同**
# （ladder 用 0.999 的松 10 倍容差、`volume == 0` 漏判负值与 NaN），
# 两套骨架的回测因此不可比。合并之后必须有测试钉住，否则谁再图省事写回
# 一份本地实现，全套测试照样绿。

def test_tradability_has_exactly_one_implementation():
    """engine 与 ladder 必须指向同一个函数对象，且两侧都不许再有本地实现。"""
    assert engine._tradability is costs.tradability
    assert ladder.tradability is costs.tradability
    assert not hasattr(ladder, "_tradable"), "ladder 不应再自存一份成交判定"


@pytest.mark.parametrize("volume", [0, -1, float("nan")])
def test_tradability_blocks_halted_bars(volume):
    """
    停牌：0 / 负值 / NaN 一律不可成交。

    NaN 必须单独钉住——`float("nan") <= 0` 恒为 False，只写 `<= 0` 会把停牌日
    当成可成交。按交易日历 reindex 出来的空行正是 NaN 成交量。
    """
    row = {"open": 10.0, "volume": volume}
    assert costs.tradability(row, 10.0, 0.10) == (False, False)


def test_tradability_uses_the_strict_tolerance():
    """
    容差是 1e-4，不是 ladder 旧实现的 1e-3。

    钉的是差异带 [涨停价×0.999, 涨停价×0.9999) 的归属：这一带旧 ladder 判
    "买不进"，统一后判"买得进"。方向别记反了——容差变窄 = 判死的日子更少。
    """
    prev, pct = 10.0, 0.10
    up = prev * (1 + pct)
    assert costs.tradability({"open": up * 0.9995, "volume": 1}, prev, pct)[0] is True
    assert costs.tradability({"open": up * 0.99995, "volume": 1}, prev, pct)[0] is False


def test_tradability_limit_moves_block_one_side_only():
    """涨停只挡买、跌停只挡卖——挡错边会让回测凭空多出/少掉一半交易。"""
    prev, pct = 10.0, 0.10
    up, down = prev * (1 + pct), prev * (1 - pct)
    assert costs.tradability({"open": up, "volume": 1}, prev, pct) == (False, True)
    assert costs.tradability({"open": down, "volume": 1}, prev, pct) == (True, False)


@pytest.mark.parametrize("prev_close", [None, float("nan"), 0.0, -1.0])
def test_tradability_passes_the_first_bar(prev_close):
    """首根 K 线没有前收：一律放行，且**不能抛异常**（ladder 旧实现遇 None 会 TypeError）。"""
    assert costs.tradability({"open": 10.0, "volume": 1}, prev_close, 0.10) == (True, True)
