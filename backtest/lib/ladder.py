"""
分批建仓（梯度加仓）模拟器。

为什么不用 `engine.run_backtest`
-------------------------------
引擎是**二元仓位**：signal=1 全仓、signal=-1 空仓。而"分阶段买入"的收益结构
恰恰来自中间状态——什么时候只有三成仓、什么时候满仓、闲置现金占了多久。用
二元引擎模拟会把这些全部抹平，得到的收益率没有意义。

口径约定（与 engine.py 保持一致，便于横向比较）
----------------------------------------------
* T 日收盘触发 → **T+1 开盘**成交
* 100 股整数手；佣金双边万三、单笔最低 5 元；印花税卖出单边千一；双边滑点千一
* 停牌（volume<=0）、开盘涨停买不进 / 开盘跌停卖不掉时，当日不成交。
  成交判定唯一实现在 `lib/costs.tradability`（与 engine.py 共用），
  采用 engine 的严格口径：相对容差 1e-4、`volume <= 0`、prev_close 非法放行
* 行情用**后复权（hfq）**，即含股息再投的全收益口径，各策略同口径可比

闲置现金按 `cash_rate` 计息。不计息的话，分批策略会被平白扣掉一块收益——
它天然长期持有现金，和满仓策略比就不公平了。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 成本假设与 `engine.py` 共用同一份定义（`lib/costs.py`），不在这里另写字面量——
# 两套撮合骨架的数字一旦漂开，梯度/网格与满仓持有的横向比较就失去意义。
from backtest.lib.costs import (COMMISSION_RATE, LIMIT_PCT_MAIN, LOT,
                       MIN_COMMISSION, SLIPPAGE, STAMP_DUTY, commission,
                       tradability)


# ── 成交约束 ──────────────────────────────────────────────────────────────────
# 涨跌停 / 停牌成交判定唯一实现在 `lib/costs.tradability`（与 engine.py 共用）。
# 曾在此另存一份 `_tradable`：容差 0.999（松 10 倍）、`volume == 0` 漏判负值/NaN、
# `prev_close <= 0` 遇 None 直接 TypeError——与引擎不可比，已删除。


# ── 结果容器 ──────────────────────────────────────────────────────────────────

@dataclass
class LadderResult:
    name: str
    equity: pd.Series                      # 总权益（持仓市值 + 现金）
    exposure: pd.Series                    # 仓位占比（持仓市值 / 总权益）
    trades: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        s = self.stats
        return (f"{self.name:<22} 总收益 {s['total_return']:>7.1%} | "
                f"年化 {s['annual_return']:>6.1%} | 最大回撤 {s['max_drawdown']:>7.1%} | "
                f"平均仓位 {s['avg_exposure']:>5.0%} | 投入资金收益 {s['deployed_return']:>7.1%} | "
                f"交易 {s['n_trades']:>2d} 笔")


def summarize(name, equity, exposure, trades, capital) -> LadderResult:
    eq = equity.astype(float)
    years = max(len(eq) / 252, 1e-9)
    total = eq.iloc[-1] / capital - 1
    dd = (eq / eq.cummax() - 1).min()
    ret = eq.pct_change().dropna()
    avg_exp = float(exposure.mean())
    stats = {
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(total),
        "annual_return": float((1 + total) ** (1 / years) - 1),
        "max_drawdown": float(dd),
        "avg_exposure": avg_exp,
        # 只按实际压上去的钱算收益：分批策略手里长期有现金，
        # 用总资金算会低估选股/择时本身的效果
        "deployed_return": float(total / avg_exp) if avg_exp > 1e-6 else 0.0,
        "sharpe": float((ret.mean() * 252 - 0.02) / (ret.std() * np.sqrt(252)))
                  if ret.std() > 0 else float("nan"),
        "n_trades": len(trades),
        "days": len(eq),
    }
    return LadderResult(name, eq, exposure, trades, stats)


# ── 通用回测骨架 ──────────────────────────────────────────────────────────────

def _run(df: pd.DataFrame, capital: float, decide, name: str,
         cash_rate: float = 0.015,
         limit_pct: float = LIMIT_PCT_MAIN) -> LadderResult:
    """
    `decide(ctx) -> list[order]`：在 T 日收盘调用，返回 T+1 开盘要执行的指令。
    order = ("buy", 金额) / ("sell", 持仓比例) / ("sell_shares", 股数)

    网格类策略必须用 sell_shares：每格买入的股数不同（价格不同），
    按"比例"折算会让卖出的股数和当初买入那一格对不上。
    """
    cash, shares, cost_basis = float(capital), 0, 0.0
    pending: list = []
    equity, exposure, trades = [], [], []
    daily_cash_rate = cash_rate / 252
    prev_close = float(df["close"].iloc[0])

    for i, (date, row) in enumerate(df.iterrows()):
        cash *= 1 + daily_cash_rate
        can_buy, can_sell = tradability(row, prev_close, limit_pct)

        # ① 开盘执行昨日挂单
        for side, size in pending:
            if side == "buy" and can_buy and size > 0:
                px = float(row["open"]) * (1 + SLIPPAGE)
                lots = int(min(size, cash) / (px * LOT))
                if lots > 0:
                    qty = lots * LOT
                    amt = qty * px
                    fee = commission(amt)
                    if amt + fee <= cash:
                        cash -= amt + fee
                        cost_basis += amt + fee
                        shares += qty
                        trades.append({"date": date, "action": "buy", "price": px,
                                       "shares": qty, "amount": amt, "fee": fee})
            elif side in ("sell", "sell_shares") and can_sell and shares > 0:
                want = shares * size if side == "sell" else size
                qty = min(int(want / LOT) * LOT, shares)
                if qty > 0:
                    px = float(row["open"]) * (1 - SLIPPAGE)
                    amt = qty * px
                    fee = commission(amt) + amt * STAMP_DUTY
                    cash += amt - fee
                    cost_basis *= 1 - qty / shares
                    shares -= qty
                    trades.append({"date": date, "action": "sell", "price": px,
                                   "shares": qty, "amount": amt, "fee": fee})
        pending = []

        close = float(row["close"])
        eq = cash + shares * close
        equity.append(eq)
        exposure.append(shares * close / eq if eq > 0 else 0.0)

        # ② 收盘决策 → 明日开盘执行
        pending = decide({
            "i": i, "date": date, "df": df, "close": close,
            "cash": cash, "shares": shares, "equity": eq,
            "cost_basis": cost_basis,
            "unrealized": (shares * close / cost_basis - 1) if shares and cost_basis > 0 else 0.0,
        }) or []
        prev_close = close

    idx = df.index
    return summarize(name, pd.Series(equity, index=idx),
                     pd.Series(exposure, index=idx), trades, capital)


# ── 策略一：一次性买入并持有 ──────────────────────────────────────────────────

def simulate_buy_hold(df, capital=100_000, **kw) -> LadderResult:
    def decide(ctx):
        return [("buy", ctx["cash"])] if ctx["i"] == 0 else []
    return _run(df, capital, decide, "一次性满仓持有", **kw)


# ── 策略二：定投 ──────────────────────────────────────────────────────────────

def simulate_dca(df, capital=100_000, n_tranches=10, every_days=21, **kw) -> LadderResult:
    per = capital / n_tranches
    state = {"done": 0}

    def decide(ctx):
        if state["done"] >= n_tranches or ctx["i"] % every_days != 0:
            return []
        state["done"] += 1
        return [("buy", per)]
    return _run(df, capital, decide, f"定投{n_tranches}期/{every_days}日", **kw)


# ── 策略三：回撤梯度建仓 ──────────────────────────────────────────────────────

def simulate_ladder(
    df: pd.DataFrame,
    capital: float = 100_000,
    *,
    n_tranches: int = 5,
    step: float = 0.07,
    base_immediate: bool = True,
    ratchet: bool = True,
    lookback: int = 120,
    take_profit: float = None,
    tp_fraction: float = 1.0,
    trail_stop: float = None,
    ma_exit: int = None,
    force_buy_days: int = None,
    name: str = None,
    **kw,
) -> LadderResult:
    """
    从"近期高点"往下铺梯子，每跌一档买一档；触发卖出条件后重置梯子。

    n_tranches     总档数，资金等分
    step           档间距（相对参考高点的回撤幅度）
    base_immediate 第一档立即建底仓（False = 第一档也要等回撤 step）
    ratchet        参考高点随新高上移（追随趋势买回调）；False = 锚定初始高点
    lookback       参考高点的回溯窗口（交易日）
    take_profit    持仓浮盈达标止盈；tp_fraction 为卖出比例
    trail_stop     从持仓期最高价回撤该幅度清仓
    ma_exit        收盘跌破 N 日均线清仓
    force_buy_days 距上次买入超过该天数仍有余额未投则强制买一档（防踏空）

    止损型离场（破均线 / 移动止损）后会**锁住再入场**，直到趋势重新转好
    （破均线：收盘重回均线上方；移动止损：自低点反弹一个档距）。
    不加这道锁的话，清仓当天梯子立刻重新铺、次日买回底仓、当晚又触发止损，
    在一段下跌里能刷出几百笔来回交易——那是实现缺陷，不是策略表现。
    """
    per = capital / n_tranches
    peak_ma = df["close"].rolling(lookback, min_periods=1).max()
    ma = df["close"].rolling(ma_exit, min_periods=ma_exit).mean() if ma_exit else None

    st = {"used": 0, "ref": None, "hold_peak": 0.0, "last_buy_i": -10**9,
          "locked": False, "low_since_exit": np.inf}

    def decide(ctx):
        i, close = ctx["i"], ctx["close"]
        if st["ref"] is None:
            st["ref"] = max(float(peak_ma.iloc[i]), close)
        if ratchet:
            st["ref"] = max(st["ref"], close)
        if ctx["shares"] > 0:
            st["hold_peak"] = max(st["hold_peak"], close)

        # ── 止损后的再入场锁：趋势重新转好才解锁 ──
        if st["locked"]:
            st["low_since_exit"] = min(st["low_since_exit"], close)
            ma_ok = ma is not None and not np.isnan(ma.iloc[i]) and close > ma.iloc[i]
            bounce_ok = ma is None and close >= st["low_since_exit"] * (1 + step)
            if ma_ok or bounce_ok:
                st["locked"] = False
                st["ref"] = close
                st["low_since_exit"] = np.inf
            else:
                return []

        # ── 卖出条件（任一触发即离场，梯子重置）──
        if ctx["shares"] > 0:
            hit = None
            if take_profit is not None and ctx["unrealized"] >= take_profit:
                hit = ("止盈", tp_fraction, False)
            elif trail_stop is not None and st["hold_peak"] > 0 \
                    and close <= st["hold_peak"] * (1 - trail_stop):
                hit = ("移动止损", 1.0, True)
            elif ma is not None and not np.isnan(ma.iloc[i]) and close < ma.iloc[i]:
                hit = ("破均线", 1.0, True)
            if hit:
                _, frac, lock = hit
                freed = int(round(st["used"] * frac))
                st["used"] = max(st["used"] - freed, 0)
                st["ref"] = close                    # 从当前价重新铺梯子
                st["hold_peak"] = 0.0
                if lock:
                    st["locked"] = True
                    st["low_since_exit"] = close
                return [("sell", frac)]

        # ── 买入：当前价跌破第 used 档的触发线 ──
        if st["used"] >= n_tranches or ctx["cash"] < per * 0.5:
            return []
        k = st["used"] + (0 if base_immediate else 1)
        trigger = st["ref"] * (1 - step * k)
        due = force_buy_days is not None and (i - st["last_buy_i"]) >= force_buy_days
        if close <= trigger or due:
            st["used"] += 1
            st["last_buy_i"] = i
            return [("buy", per)]
        return []

    label = name or (f"梯度{n_tranches}档×{step:.0%}"
                     + (f"·止盈{take_profit:.0%}" if take_profit else "·长持"))
    return _run(df, capital, decide, label, **kw)


# ── 策略四：固定网格 ──────────────────────────────────────────────────────────

def simulate_grid(
    df: pd.DataFrame,
    capital: float = 100_000,
    *,
    base_position: float = 0.5,
    grid_step: float = 0.07,
    n_grids: int = 5,
    ratchet: bool = False,
    name: str = None,
    **kw,
) -> LadderResult:
    """
    底仓 + 网格：跌一格买一份、涨一格卖一份，底仓始终不动。

    底仓保证长期上涨吃得到，网格部分赚波动的钱。适合"看好但认为要震荡很久"。

    ratchet
        锚价是否随新高上移（**只在 level==0、即手上没有网格仓位时**才移；
        持有网格仓位时上移锚价会让卖出触发价对不上当初的买入格）。

        默认 False 保持原行为：锚价钉在首根 K 线的收盘价，一辈子不动。
        这在长期上涨的标的上会让网格**从未装上膛**——600938 自 2022-04-21
        上市起从没跌破首日锚 1.2% 以上，36 组参数全都只有 1 笔交易，
        策略实际退化成"买 base_position 然后躺平"，参数怎么调都没区别。
        判断网格好不好用之前，先看 `n_trades` 是不是 1。

        已知局限：持仓期间锚不动，因为卖出触发价是从 `anchor − level×step` 推的，
        抬锚就会和当初买入的那一格对不上。要让锚随时能动，得把 `stack` 从「只存
        股数」改成「存 (股数, 该格的买入价)」，卖出按各格自己的买入价 ×(1+step)
        触发——买卖就对称了，锚怎么动都不影响已持有的格子。
        **没做**：实测网格在这两只票上没有稳定的敞口对齐超额（54 组中位数 ≈ −3pp），
        为一个不赚钱的策略重构撮合逻辑不划算。真要用网格再回来改。
    """
    grid_cash = capital * (1 - base_position)
    per = grid_cash / n_grids
    # 每格实际买到多少股要等成交后才知道，用持仓变化回填这个栈，
    # 卖出时按栈顶那一格的真实股数卖（LIFO），底仓永远压在栈底不动
    st = {"anchor": None, "level": 0, "based": False,
          "stack": [], "pending_buy": False, "last_shares": 0}

    def decide(ctx):
        close, shares = ctx["close"], ctx["shares"]
        if st["pending_buy"]:
            st["pending_buy"] = False
            got = shares - st["last_shares"]
            if got > 0:
                st["stack"].append(got)
            else:
                st["level"] = max(st["level"] - 1, 0)   # 没买成，档位退回
        st["last_shares"] = shares

        if not st["based"]:
            st["based"] = True
            st["anchor"] = close
            return [("buy", capital * base_position)]

        if ratchet and st["level"] == 0:
            st["anchor"] = max(st["anchor"], close)

        lvl_price = st["anchor"] * (1 - grid_step * st["level"])
        if close <= lvl_price * (1 - grid_step) and st["level"] < n_grids \
                and ctx["cash"] >= per:
            st["level"] += 1
            st["pending_buy"] = True
            return [("buy", per)]
        if st["level"] > 0 and close >= lvl_price * (1 + grid_step) and st["stack"]:
            st["level"] -= 1
            return [("sell_shares", st["stack"].pop())]
        return []

    label = name or (f"底仓{base_position:.0%}+网格{n_grids}×{grid_step:.0%}"
                     + ("·锚随高点" if ratchet else ""))
    return _run(df, capital, decide, label, **kw)


# ── 策略五：按市场状态切换打法 ────────────────────────────────────────────────

#: 每个状态一套打法。含义见 lib.regime。
#: max_pos 是该状态允许的最高仓位——趋势下行档设 0 等于清仓离场。
PLAYBOOK = {
    "趋势上行": dict(n_tranches=3, step=0.05, take_profit=None, tp_fraction=1.0,
                     force_buy_days=20, max_pos=1.0),
    "宽幅震荡": dict(n_tranches=4, step=0.08, take_profit=0.30, tp_fraction=0.5,
                     force_buy_days=None, max_pos=1.0),
    # 降到底仓而不是清空：这两只高股息票的下行段后面常常直接跟一段反弹，
    # 清仓的版本在回测里反复踏空，收益反而低于留 30% 底仓的版本。
    "趋势下行": dict(n_tranches=4, step=0.10, take_profit=None, tp_fraction=1.0,
                     force_buy_days=None, max_pos=0.30),
}


def simulate_adaptive(
    df: pd.DataFrame,
    regimes: pd.Series,
    capital: float = 100_000,
    *,
    playbook: dict = None,
    lookback: int = 120,
    name: str = "自适应（按状态切换）",
    **kw,
) -> LadderResult:
    """
    按 `regimes`（lib.regime.classify 的 regime 列）逐日切换打法。

    切换状态时不清零重来：已经压上去的钱按新打法的档数折算成"已用档位"，
    否则每次换挡都会把仓位当成 0、立刻再买一轮，凭空放大仓位和交易次数。
    """
    pb = playbook or PLAYBOOK
    peak_ma = df["close"].rolling(lookback, min_periods=1).max()
    reg = regimes.reindex(df.index).ffill()

    st = {"cur": None, "used": 0, "ref": None, "last_buy_i": -10**9}

    def decide(ctx):
        i, close = ctx["i"], ctx["close"]
        r = reg.iloc[i]
        p = pb.get(r, pb["宽幅震荡"])
        per = capital / p["n_tranches"]

        if r != st["cur"]:                       # 换挡：按新档数折算已用档位
            deployed = min(ctx["cost_basis"] / capital, 1.0)
            st["used"] = int(round(deployed * p["n_tranches"]))
            st["ref"] = max(float(peak_ma.iloc[i]), close)
            st["cur"] = r

        st["ref"] = max(st["ref"], close)

        # 仓位超过该状态的上限 → 减到上限（max_pos=0 即清仓离场）
        if ctx["shares"] > 0 and ctx["equity"] > 0:
            exposure = ctx["shares"] * close / ctx["equity"]
            if exposure > p["max_pos"] + 1e-9:
                frac = 1.0 if p["max_pos"] <= 0 else 1 - p["max_pos"] / exposure
                st["used"] = max(st["used"] - int(round(st["used"] * frac)), 0)
                st["ref"] = close
                return [("sell", min(frac, 1.0))]

        # 止盈
        if ctx["shares"] > 0 and p["take_profit"] is not None \
                and ctx["unrealized"] >= p["take_profit"]:
            frac = p["tp_fraction"]
            st["used"] = max(st["used"] - int(round(st["used"] * frac)), 0)
            st["ref"] = close
            return [("sell", frac)]

        # 加仓
        if p["max_pos"] <= 0 or st["used"] >= p["n_tranches"] or ctx["cash"] < per * 0.5:
            return []
        if ctx["equity"] > 0 and ctx["shares"] * close / ctx["equity"] >= p["max_pos"]:
            return []
        trigger = st["ref"] * (1 - p["step"] * st["used"])
        due = (p["force_buy_days"] is not None
               and (i - st["last_buy_i"]) >= p["force_buy_days"])
        if close <= trigger or due:
            st["used"] += 1
            st["last_buy_i"] = i
            return [("buy", per)]
        return []

    return _run(df, capital, decide, name, **kw)
