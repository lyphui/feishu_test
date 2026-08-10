"""
「乌龙指捕捉」策略模拟器：永久保持 50% 持仓 / 50% 现金，两侧各挂一张远离市价的
限价单，等别人敲错价格把单子吃掉，成交后再平衡回 50/50 继续挂。

    持仓腿：在 锚价×(1+k_up) 挂卖，等有人错手高价买
    现金腿：在 锚价×(1−k_dn) 挂买，等有人错手低价卖
    任一腿成交 → 回到 50/50（锚价重置为再平衡成交价）→ 继续挂

为什么单独写而不复用 `lib.ladder`
---------------------------------
`ladder._run` 只在**开盘**撮合昨日决策，够用是因为那些策略的触发条件都是收盘价。
本策略的触发条件是**日内是否触及某个限价**，必须读 high/low，且成交价是自己挂的
那个价而不是开盘价——撮合骨架不一样，硬塞进 `_run` 会把两套语义搅在一起。
成本常量与统计口径直接复用 `lib.ladder`，保证和梯度/网格/自适应那几套数字可比。

三条决定结论的建模约定
----------------------
1. **涨跌停限制了挂单价本身。** A 股主板每日有效申报区间是前收 ±10%，超出这个
   范围的限价单交易所直接拒收——不是"挂着没成交"，是**根本挂不出去**。所以
   `k=0.30` 这种挂法在 A 股不存在；模拟器逐日检查可挂性，挂不出去的天数会计入
   `n_rejected_days`。
2. **限价单成交不吃滑点，再平衡腿吃。** 挂单方是被动的一侧，成交价就是自己报的
   价；跳空穿过挂单价时按开盘价成交（价格改善，算给策略）。回补/再平衡是主动
   下单，按 `ladder` 的千一滑点计。这两条都**偏袒**策略，是故意的：让它在最有利
   的假设下跑，输了才说明问题出在逻辑而不是摩擦成本。
3. **日线 OHLC 分不出乌龙指和真实行情。** 一笔瞬间被打回的错价冲高，和一段真涨
   到那个位置的行情，在日线上都只是一个 high。模拟器把**每一次触及都算成成交**，
   等于假设"所有触及都是可捡的乌龙指"——这同样是给策略的上限。真实世界里成交的
   那些天大多是后者，`fill_edge()` 就是用来量这件事的。

`fill_edge()`：成交后到底捡到没有
---------------------------------
每笔成交记 `edge = 成交价 / 再平衡回补价 − 1`（买单反向）。乌龙指假说成立的话
它应该显著为正——尖峰被打回，你卖在高处、按正常价买回来。若它围绕 0 甚至为负，
说明成交是被趋势带过去的，不是被人敲错送过来的。

**edge 是「毛」价差**：回补价取的是不含滑点的裸价，佣金印花税也不在里面。
这是故意的——摩擦成本已经完整体现在净值曲线里，再扣一遍就成了双重计费。
代价是它天然偏乐观，所以**判断"捡没捡到"不能拿 edge > 0 当门槛，要跟单边
摩擦成本比**（本模块口径下约 26bp，`ROUND_TRIP_BP` 给出估算）。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 成本模型与统计口径与分批建仓模拟器共用，否则两边数字没法横向比
from lib.ladder import (COMMISSION_RATE, LOT, SLIPPAGE, STAMP_DUTY,
                        LadderResult, _commission, _summarize)

#: 判定"挂单价是否落在涨跌停带内"的浮点容差（纯数值误差，不是报价最小变动单位）
_EPS = 1e-9

#: 一个来回（限价单成交 + 主动回补）的单边摩擦估算，bp。
#: 回补腿吃滑点千一 + 双边佣金万三 + 卖出印花税千一，摊到单边约 26bp。
#: `edge` 是毛价差，必须跨过这条线才谈得上"捡到"。
ROUND_TRIP_BP = (SLIPPAGE + COMMISSION_RATE * 2 + STAMP_DUTY) * 1e4 / 2


class _Book:
    """持仓账本：整手撮合 + `lib.ladder` 的成本模型。两个模拟器共用同一套记账。"""

    def __init__(self, capital: float):
        self.cash = float(capital)
        self.shares = 0
        self.trades: list = []

    def buy(self, px: float, amount: float, date, kind: str):
        """按金额买入，向下取整到手。返回成交价，没买成返回 None。"""
        qty = int(min(amount, self.cash) / (px * LOT)) * LOT
        return self._buy_qty(px, qty, date, kind)

    def _buy_qty(self, px: float, qty: int, date, kind: str):
        while qty > 0:
            amt = qty * px
            fee = _commission(amt)
            if amt + fee <= self.cash:
                self.cash -= amt + fee
                self.shares += qty
                self.trades.append({"date": date, "action": "buy", "kind": kind,
                                    "price": px, "shares": qty, "amount": amt,
                                    "fee": fee})
                return px
            qty -= LOT                      # 差一点钱就少买一手，别把现金买成负数
        return None

    def sell(self, px: float, qty: float, date, kind: str):
        qty = min(int(qty / LOT) * LOT, self.shares)
        if qty <= 0:
            return None
        amt = qty * px
        fee = _commission(amt) + amt * STAMP_DUTY
        self.cash += amt - fee
        self.shares -= qty
        self.trades.append({"date": date, "action": "sell", "kind": kind,
                            "price": px, "shares": qty, "amount": amt, "fee": fee})
        return px

    def rebalance_to(self, target: float, px_raw: float, date, kind="rebalance"):
        """
        把持仓拉回 `target` 比例。主动下单，吃滑点。

        目标手数**四舍五入**而不是向下取整：一手在 10 万本金里就是好几个百分点，
        一律向下截断会让每次再平衡都少买一手，把常态仓位系统性压到 target 之下
        （实测 50% 的设定跑出 40% 的平均仓位，等于偷偷换了一个策略）。
        整手买不动时才逐手退让，退让由 `_buy_qty` / `sell` 负责。
        """
        eq = self.cash + self.shares * px_raw
        px_buy, px_sell = px_raw * (1 + SLIPPAGE), px_raw * (1 - SLIPPAGE)
        want = int(round(eq * target / (px_raw * LOT))) * LOT
        delta = want - self.shares
        if delta >= LOT:
            self._buy_qty(px_buy, delta, date, kind)
        elif -delta >= LOT:
            self.sell(px_sell, -delta, date, kind)
        return px_raw

    def equity(self, px: float) -> float:
        return self.cash + self.shares * px

    def accrue(self, daily_rate: float) -> None:
        self.cash *= 1 + daily_rate


@dataclass
class FatFingerResult:
    """`LadderResult` 之外再带一份成交明细，用于判断"捡到"是不是真的。"""
    result: LadderResult
    fills: pd.DataFrame                    # 每笔限价单成交的明细与事后回补价
    diag: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.result.name

    @property
    def stats(self) -> dict:
        return self.result.stats


def simulate_fatfinger(
    df: pd.DataFrame,
    capital: float = 100_000,
    *,
    k_up: float = 0.05,
    k_dn: float = 0.05,
    target: float = 0.5,
    limit_pct: float = 0.10,
    fast_sell_rebalance: bool = False,
    cash_rate: float = 0.015,
    name: str = None,
) -> FatFingerResult:
    """
    k_up / k_dn        卖单、买单相对锚价的偏离幅度
    target             常态持仓比例（0.5 = 一半股票一半现金）
    limit_pct          每日涨跌停幅度，决定限价单挂不挂得出去
    fast_sell_rebalance
        True  = 卖单成交后**当日收盘**就买回（A 股卖出资金当日可用，合法）；
                买单成交后仍须等 T+1（当日买入不可卖出）。
        False = 两侧都等 T+1 开盘再平衡（默认，保守且对称）。
    """
    book = _Book(capital)
    anchor = None
    # 欠着一笔 T+1 开盘再平衡的那条成交记录（None = 没有欠账）。
    # 存记录本身而不是布尔量，是为了在**真正执行**的那一天把回补价写回去——
    # 若改成事后按"成交次日开盘"去猜，碰上次日停牌就会记一个没发生过的价。
    pending_fill = None
    equity_s, exposure_s, fills = [], [], []
    daily_cash_rate = cash_rate / 252
    n_rejected_sell = n_rejected_buy = n_both_sides = 0

    prev_close = float(df["close"].iloc[0])

    for i, (date, row) in enumerate(df.iterrows()):
        book.accrue(daily_cash_rate)
        o, h, l, c = (float(row["open"]), float(row["high"]),
                      float(row["low"]), float(row["close"]))
        halted = row.get("volume", 1) == 0

        # ── ① 建初始仓位 ──
        # 第 0 根只算"决定要建仓"，成交排到下一根开盘：与 `ladder._run` 的 T+1
        # 口径一致。不这样做的话，首根 K 线的开盘价会被当成可成交价——600938 的
        # 首根就是上市当日（开 12.96、次日开 14.51），等于白拿 12% 的头，
        # 把这个基准抬高之后，任何策略的"超额"都是负的。
        if anchor is None:
            if i > 0 and not halted:
                anchor = book.rebalance_to(target, o, date, "open")
            eq = book.equity(c)
            equity_s.append(eq)
            exposure_s.append(book.shares * c / eq if eq > 0 else 0.0)
            prev_close = c
            continue

        # ── ② 昨日成交后欠的再平衡，今日开盘补上 ──
        if pending_fill is not None and not halted:
            anchor = book.rebalance_to(target, o, date)
            pending_fill["rebalance"] = o
            pending_fill = None

        # ── ③ 挂单与撮合 ──
        if not halted and pending_fill is None:
            up_limit = prev_close * (1 + limit_pct)
            dn_limit = prev_close * (1 - limit_pct)
            sell_px = anchor * (1 + k_up)
            buy_px = anchor * (1 - k_dn)
            can_place_sell = sell_px <= up_limit + _EPS and book.shares > 0
            can_place_buy = buy_px >= dn_limit - _EPS and book.cash > 0
            if book.shares > 0 and sell_px > up_limit + _EPS:
                n_rejected_sell += 1
            if book.cash > 0 and buy_px < dn_limit - _EPS:
                n_rejected_buy += 1

            # 跳空穿过挂单价 → 按开盘价成交（价格改善，算给策略）
            sell_fill = (max(o, sell_px) if can_place_sell and h >= sell_px else None)
            buy_fill = (min(o, buy_px) if can_place_buy and l <= buy_px else None)

            # 两侧同日触发：日线看不出先后。只认离开盘更近的那一侧，另一侧视为撤单
            if sell_fill is not None and buy_fill is not None:
                n_both_sides += 1
                if abs(o - sell_fill) <= abs(o - buy_fill):
                    buy_fill = None
                else:
                    sell_fill = None

            if sell_fill is not None:
                book.sell(sell_fill, book.shares, date, "limit")
                fills.append({"date": date, "side": "sell", "anchor": anchor,
                              "fill": sell_fill, "i": i})
                pending_fill = fills[-1]
                if fast_sell_rebalance:
                    # 卖出资金当日可用，收盘立刻买回；锚价随之重置
                    anchor = book.rebalance_to(target, c, date)
                    pending_fill["rebalance"] = c
                    pending_fill = None
            elif buy_fill is not None:
                book.buy(buy_fill, book.cash, date, "limit")
                fills.append({"date": date, "side": "buy", "anchor": anchor,
                              "fill": buy_fill, "i": i})
                pending_fill = fills[-1]   # 当日买入不可卖出，必须等 T+1

        eq = book.equity(c)
        equity_s.append(eq)
        exposure_s.append(book.shares * c / eq if eq > 0 else 0.0)
        prev_close = c

    label = name or f"乌龙指 ±{k_up:.1%}/{k_dn:.1%}（{target:.0%}仓）"
    res = _summarize(label, pd.Series(equity_s, index=df.index),
                     pd.Series(exposure_s, index=df.index), book.trades, capital)

    f = pd.DataFrame(fills)
    if not f.empty:
        # 回补价在真正执行的那一天就地写入了。样本末尾那笔可能还没回补 → NaN，
        # `fill_edge` 会把它 dropna 掉，不去猜一个没发生的价。
        if "rebalance" not in f.columns:
            f["rebalance"] = np.nan
        f = _attach_forward(f, df)

    years = max(len(df) / 252, 1e-9)
    diag = {
        "n_fills": len(f),
        "n_sell": int((f["side"] == "sell").sum()) if not f.empty else 0,
        "n_buy": int((f["side"] == "buy").sum()) if not f.empty else 0,
        "fills_per_year": len(f) / years,
        "n_rejected_sell_days": n_rejected_sell,
        "n_rejected_buy_days": n_rejected_buy,
        "n_both_sides_days": n_both_sides,
        "k_up": k_up, "k_dn": k_dn,
    }
    return FatFingerResult(res, f, diag)


def _attach_forward(f: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """给每笔成交补上事后走势：回补价差（edge）与成交后 N 日收益。"""
    close = df["close"].to_numpy()
    n = len(close)
    for h in (1, 5, 20):
        f[f"fwd{h}"] = [close[min(i + h, n - 1)] / close[i] - 1 for i in f["i"]]
    # edge > 0 = 这一趟真的捡到了：卖在高处、按更低的价格补回来
    f["edge"] = np.where(f["side"] == "sell",
                         f["fill"] / f["rebalance"] - 1,
                         f["rebalance"] / f["fill"] - 1)
    return f


def fill_edge(f: pd.DataFrame) -> pd.DataFrame:
    """
    按方向汇总成交质量。**edge 要显著超过 `ROUND_TRIP_BP`（≈26bp）**，
    「捡乌龙指」这件事才算真实存在——它是毛价差，不含滑点与佣金（见模块 docstring）。

    `n` 只计入回补价已知的成交：样本末尾那笔可能还欠着再平衡，算不出 edge。
    """
    if f.empty:
        return pd.DataFrame()
    rows = []
    for side, sub in f.groupby("side"):
        e = sub["edge"].dropna()
        if e.empty:
            continue
        t = (e.mean() / (e.std() / np.sqrt(len(e)))) if len(e) > 1 and e.std() > 0 else np.nan
        rows.append({
            "side": side, "n": len(e),
            "edge_mean_bp": e.mean() * 1e4, "edge_median_bp": e.median() * 1e4,
            "t": t, "win_rate": float((e > 0).mean()),
            "fwd1": sub["fwd1"].mean(), "fwd5": sub["fwd5"].mean(),
            "fwd20": sub["fwd20"].mean(),
        })
    return pd.DataFrame(rows)


def simulate_static_mix(
    df: pd.DataFrame,
    capital: float = 100_000,
    *,
    target: float = 0.5,
    rebalance_days: int = None,
    cash_rate: float = 0.015,
    name: str = None,
) -> LadderResult:
    """
    对照组：同样 50/50，但**不挂任何单**。`rebalance_days=None` 即买完躺平。

    这是判断乌龙指策略的正确基准——它和乌龙指版本承担同样的市场敞口，
    差别只有"多做的那些交易"。跑不赢它，说明那些交易是负收益的。
    """
    book = _Book(capital)
    equity_s, exposure_s = [], []
    daily_cash_rate = cash_rate / 252
    pending = True

    for i, (date, row) in enumerate(df.iterrows()):
        book.accrue(daily_cash_rate)
        o, c = float(row["open"]), float(row["close"])
        halted = row.get("volume", 1) == 0

        # i > 0：首根只做决策，成交排到下一根开盘（T+1 口径，同 `ladder._run`）
        if pending and not halted and i > 0:
            book.rebalance_to(target, o, date)
            pending = False

        if rebalance_days and i > 0 and i % rebalance_days == 0:
            pending = True

        eq = book.equity(c)
        equity_s.append(eq)
        exposure_s.append(book.shares * c / eq if eq > 0 else 0.0)

    label = name or (f"静态{target:.0%}仓"
                     + (f"·{rebalance_days}日再平衡" if rebalance_days else "·躺平"))
    return _summarize(label, pd.Series(equity_s, index=df.index),
                      pd.Series(exposure_s, index=df.index), book.trades, capital)
