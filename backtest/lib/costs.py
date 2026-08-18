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

import re
from dataclasses import dataclass

import numpy as np

#: 券商佣金费率，双边收取（万三）
COMMISSION_RATE = 0.0003
#: 单笔最低佣金（元）——A 股券商普遍 5 元。小额单的费用率由它主导
MIN_COMMISSION = 5.0
#: 印花税，仅卖出单边收取（千一）
STAMP_DUTY = 0.001
#: 单边滑点（千一）：买价上浮、卖价下压
SLIPPAGE = 0.001

#: 无风险利率（年化），夏普比率的折减项。engine / ladder / compare_playbooks
#: 三处夏普若各写一份字面量，同一张对比表的排序就变成按公式排序。
RISK_FREE_RATE = 0.02
#: 空仓现金利率（年化），ladder 系模拟器里闲置资金的计息
CASH_RATE = 0.015

#: A 股最小交易单位（一手 = 100 股）
LOT = 100

#: 涨跌停幅度：主板 / 创业板科创板 / 北交所
LIMIT_PCT_MAIN = 0.10
LIMIT_PCT_GROWTH = 0.20
LIMIT_PCT_BSE = 0.30
#: ST / *ST（主板）——只能靠名称判定，代码前缀看不出来
LIMIT_PCT_ST = 0.05


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


def is_st_name(name: str | None) -> bool:
    """
    名称是否表示 ST / *ST。

    用**前缀**匹配而不是 `"ST" in name`：后者会被任何恰好含这两个字母的名称
    命中（英文名、拼音简称、带 "ST" 的产品名），而交易所的风险警示标识
    只出现在名称开头，形如 `ST路桥` / `*ST海投` / `SST前锋`。
    误判的代价不对称——把非 ST 判成 ST 会让 ±10% 的票按 ±5% 撮合，
    `tradability` 于是把一切超过 5% 的开盘当成封板、静默拦掉成交。
    """
    if not name:
        return False
    return re.match(r"^\s*\*?S?ST", name.strip().upper()) is not None


def infer_limit_pct(symbol: str, name: str | None = None) -> float:
    """
    按代码前缀（+ 可选名称）推断涨跌停幅度。

    688/689 科创板、300/301 创业板     → 20%（ST 也是 20%）
    4xx/8xx 北交所                     → 30%（ST 也是 30%）
    主板且名称以 ST/*ST 开头            → 5%
    其余主板                            → 10%

    **板块先于 ST**：风险警示的 ±5% 只适用于**主板**。创业板/科创板注册制下
    ST 股仍是 ±20%、北交所 ST 仍是 ±30%，若先判 ST 再判板块，`300xxx` 的
    ST 票会被按 ±5% 撮合——任何超过 5% 的开盘都被当成封板，成交被静默拦掉、
    「受阻次数」虚高。

    ST 状态从代码看不出来，也**不随时间回溯**：这里用的是调用方给的当期名称，
    会被套用到整个回测窗口。历史上戴过帽、现在摘了（或反之）的票口径会偏，
    方向偏保守，可接受。`name=None` 时按非 ST 处理，对 ST 股**低估**封单概率
    （结果偏乐观），故 jcy 候选池这类自带名称的调用方应当传 name。
    """
    if symbol.startswith(("688", "689", "300", "301")):
        return LIMIT_PCT_GROWTH
    if symbol.startswith(("4", "8")):
        return LIMIT_PCT_BSE
    if is_st_name(name):
        return LIMIT_PCT_ST
    return LIMIT_PCT_MAIN


# ── 港股成本（自 lib/trend_stop.py 收编，评审项 6；数值逐字未动） ──────────────
#
# 此前这套费率定义在 trend_stop 里、与月频决策耦合。港股标的不止 3175.HK
# 一只，成本层应是公共的：trend_stop 改为消费者（保留同名 re-export）。

#: 港股券商佣金（中银香港零售网上渠道）：0.25%，最低 HK$100。
#: 最低佣金是月频策略全部设计的起点——成交额低于 4 万港币时费用率随金额爆炸。
HK_COMMISSION_RATE = 0.0025
HK_MIN_COMMISSION = 100.0
#: 平台费（银行自定固定服务费，各家不同，按成交单实际列示调整）
HK_PLATFORM_FEE = 30.0
#: 港股印花税（千一）；ETF 豁免
HK_STAMP_DUTY = 0.001

#: 港股法定/结算费率（与盘型无关，只跟成交金额挂钩）
SFC_LEVY = 0.000027           # 证监会交易征费
AFRC_LEVY = 0.0000015         # 财汇局交易征费
HKEX_TRADING_FEE = 0.0000565  # 联交所交易费
CCASS_RATE = 0.00002          # 中央结算费
CCASS_MIN, CCASS_MAX = 2.0, 100.0


def hk_trade_cost(
    value: float,
    *,
    commission_rate: float = HK_COMMISSION_RATE,
    commission_min: float = HK_MIN_COMMISSION,
    platform_fee: float = HK_PLATFORM_FEE,
    is_etf: bool = True,
) -> float:
    """
    单边交易总成本（港币）。默认参数取自中银香港零售网上渠道。

    `commission_min` 是这套策略全部设计的起点：成交额低于
    `commission_min / commission_rate`（默认 4 万港币）时，佣金按最低收，
    费用率随金额下降而爆炸——600 股 ≈ 5,790 港币的单子费用率高达 2.29%。

    `platform_fee` 是银行自定的固定服务费，各家不同，按成交单实际列示调整。
    ETF 在香港豁免印花税（`is_etf=True`），个股要另加 0.1%。
    """
    if value <= 0:
        return 0.0
    commission = max(value * commission_rate, commission_min)
    levies = value * (SFC_LEVY + AFRC_LEVY + HKEX_TRADING_FEE)
    ccass = min(max(value * CCASS_RATE, CCASS_MIN), CCASS_MAX)
    stamp = 0.0 if is_etf else value * HK_STAMP_DUTY
    return commission + platform_fee + levies + ccass + stamp


def hk_fee_rate(value: float, **kw) -> float:
    """单边成本占成交额的比例。回测里的 `fee` 参数就用它算。"""
    return hk_trade_cost(value, **kw) / value if value > 0 else 0.0


# ── 分市场费率组 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MarketCosts:
    """一个市场的费率组与撮合约束。`for_market("A"|"HK")` 取用。"""
    market: str
    commission_rate: float
    min_commission: float
    stamp_duty: float
    slippage: float
    lot: int | None            # 港股每手股数因票而异，故允许 None
    platform_fee: float = 0.0
    etf_stamp_exempt: bool = False


def for_market(market: str) -> MarketCosts:
    """按市场返回费率组：A 股聚合本模块上部常量，港股聚合港股常量。"""
    m = market.upper()
    if m in ("A", "CN", "ASHARE"):
        return MarketCosts("A", COMMISSION_RATE, MIN_COMMISSION,
                           STAMP_DUTY, SLIPPAGE, LOT)
    if m == "HK":
        # 港股现有回测（trend_stop）不计滑点，如实记 0
        return MarketCosts("HK", HK_COMMISSION_RATE, HK_MIN_COMMISSION,
                           HK_STAMP_DUTY, 0.0, None,
                           platform_fee=HK_PLATFORM_FEE, etf_stamp_exempt=True)
    raise ValueError(f"未知市场：{market}（可选 A / HK）")
