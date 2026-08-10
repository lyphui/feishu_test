"""
A 股成交成本与交易约束的**唯一真值源**。

为什么值得单独一个模块
----------------------
本项目有两套互不相容的撮合骨架，谁也塞不进谁（原因见各自 docstring）：

* `engine.py` —— 二元仓位（signal=1 全仓 / -1 空仓），成本走**函数参数**，
  可以被 `.ini` 预设逐项覆盖；
* `lib/ladder.py` —— 连续仓位（分批建仓/网格/自适应），成本走**模块常量**，
  `lib/fatfinger.py` 再从它转引。

骨架可以不同，**成本假设必须逐位相同**——否则"梯度加仓 vs 满仓持有""网格 vs
静态同敞口"这类横向比较，比的就不是策略而是费率，整个 `docs/tracking-and-ladder.md`
的敞口对齐口径都会失效。

以前这四个数字在 `engine.py` 与 `lib/ladder.py` 里各写一遍字面量，靠 ladder 的
docstring 写一句"与 engine.py 保持一致"人肉同步。现在收成一处：改一个数，两套
引擎同时生效，也不会出现"改了一处忘了另一处、两组回测数字从此不可比"。

口径
----
默认值取 A 股零售常见水平。佣金**双边**收取、印花税**仅卖出**、滑点按**单边**
计（买价上浮、卖价下压）。要试"零成本"对照组，改调用方的传参，别改这里。
"""

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
