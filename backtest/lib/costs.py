"""
A 股成交成本与交易约束的**唯一真值源**。

为什么值得单独一个模块
----------------------
本项目有两套互不相容的撮合骨架，谁也塞不进谁（原因见各自 docstring）：

* `engine.py` —— 二元仓位（signal=1 全仓 / -1 空仓），成本走**函数参数**，
  可以被 `.ini` 预设逐项覆盖；
* `lib/ladder.py` —— 连续仓位（分批建仓/网格/自适应），成本走**模块常量**；
  `lib/fatfinger.py` 同样直取本模块。

骨架可以不同，**成本假设必须逐位相同**——否则"梯度加仓 vs 满仓持有""网格 vs
静态同敞口"这类横向比较，比的就不是策略而是费率，整个 `docs/tracking-and-ladder.md`
的敞口对齐口径都会失效。

以前这四个数字在 `engine.py` 与 `lib/ladder.py` 里各写一遍字面量，靠 ladder 的
docstring 写一句"与 engine.py 保持一致"人肉同步。现在收成一处：改一个数，两套
引擎同时生效，也不会出现"改了一处忘了另一处、两组回测数字从此不可比"。

涨跌停 / 停牌的成交判定（`tradability`）同样曾有两份实现：engine 用相对容差
1e-4 且检查 volume<=0 / prev_close 非法，ladder 用 0.999 的松 10 倍容差且漏判
负值与 NaN。两者**行为真的不同**，导致两套撮合骨架的回测不可比。统一收在
本模块的 `tradability`（采用 engine 的严格口径）。

口径
----
默认值取 A 股零售常见水平。佣金**双边**收取、印花税**仅卖出**、滑点按**单边**
计（买价上浮、卖价下压）。要试"零成本"对照组，改调用方的传参，别改这里。
"""

import numpy as np

#: 券商佣金费率，双边收取（万三）
COMMISSION_RATE = 0.0003
#: 单笔最低佣金（元）——A 股券商普遍 5 元。小额单的费用率由它主导
MIN_COMMISSION = 5.0
#: 印花税，仅卖出单边收取（千一）
STAMP_DUTY = 0.001
#: 单边滑点（千一）：买价上浮、卖价下压
SLIPPAGE = 0.001

#: A 股最小交易单位（一手 = 100 股）
LOT = 100

#: 涨跌停幅度：主板 / 创业板科创板 / 北交所
LIMIT_PCT_MAIN = 0.10
LIMIT_PCT_GROWTH = 0.20
LIMIT_PCT_BSE = 0.30


def commission(amount: float,
               rate: float = COMMISSION_RATE,
               minimum: float = MIN_COMMISSION) -> float:
    """券商佣金：按成交额比例收取，但不低于单笔最低佣金。"""
    return max(amount * rate, minimum)


def tradability(row, prev_close: float, limit_pct: float) -> tuple[bool, bool]:
    """
    判断当日开盘能否买入 / 卖出，返回 (can_buy, can_sell)。

    `engine` 与 `lib/ladder` 两套撮合骨架共用这一份实现（engine 以
    `_tradability` 旧名 re-export，历史导入不受影响）。**只有这一份**——
    曾有两份且行为不同：ladder 用 0.999 的松容差、`volume == 0` 漏判负值/NaN、
    `prev_close <= 0` 遇 None 直接 TypeError，与 engine 不可比。这里统一采用
    engine 的严格口径。

    - 停牌（volume <= 0，含 NaN / 负值）：买卖都不可能成交
    - 开盘即涨停：买单排在队尾，成交不了；卖出不受影响
    - 开盘即跌停：卖不掉；买入不受影响
    - `prev_close` 非法（None / 非有限 / <= 0）：首根 K 线没有前收，放行

    价格用相对容差（1e-4）比较而不是 round(_, 2)：复权后的价格已不是真实报价，
    绝对分位对不上。

    volume 缺列（`None`）与 volume 为 NaN 是两回事：前者是"这份数据没带成交量"，
    放行；后者多来自按交易日历 reindex 出的空行，就是停牌，拦下。
    """
    vol = row.get("volume", 1)
    if vol is not None:
        vol = float(vol)
        # NaN 必须显式判：`float("nan") <= 0` 恒为 False，只写 <= 0 会把停牌日
        # 当成可成交（这里曾经就漏了，docstring 却写着"含 NaN"）。
        if not np.isfinite(vol) or vol <= 0:
            return False, False
    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        return True, True

    tol = 1e-4
    open_ = row["open"]
    limit_up   = prev_close * (1 + limit_pct)
    limit_down = prev_close * (1 - limit_pct)
    can_buy  = open_ < limit_up * (1 - tol)
    can_sell = open_ > limit_down * (1 + tol)
    return bool(can_buy), bool(can_sell)


def infer_limit_pct(symbol: str) -> float:
    """
    按代码前缀推断涨跌停幅度。

    688/689 科创板、300/301 创业板 → 20%
    4xx/8xx 北交所                 → 30%
    其余主板                        → 10%

    注意：ST / *ST 为 5%，但从代码看不出来，也不随时间回溯，这里按主板 10%
    处理，对 ST 股会**低估**涨跌停的封单概率（结果偏乐观），需要时显式传
    limit_pct=0.05。
    """
    if symbol.startswith(("688", "689", "300", "301")):
        return LIMIT_PCT_GROWTH
    if symbol.startswith(("4", "8")):
        return LIMIT_PCT_BSE
    return LIMIT_PCT_MAIN
