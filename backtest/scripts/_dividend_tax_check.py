"""
中长持仓打法的红利税实算（评审 `docs/backtest-review.md` 项 4 / 3.1 的后续实测）。

评审 3.1 已实测：高换手打法（LuMACDBull，平均持仓 3.5 天）在 601857 上
**没有一笔持仓跨越除息日**，税后修正 ≈ 0。它指出真正的落点是持仓
1 个月–1 年的打法（ladder 波段、trend_stop 月频）——适用 10% 税档。
本脚本把这个推断变成实测。

口径与近似
----------
- A 股红利税按持有期分档：≤1 个月 20% / 1 个月–1 年 10% / >1 年 0%，
  卖出时按「税前现金分红 × 股数 × 税率」补扣。
- hfq 回测里的股数 = 金额 / 后复权价，与真实股数有口径差（后复权价
  高于盘面价时股数被低估），税额是同方向的**估计值**，看量级不看分毫。
- ladder 是分批进出的部分成交，按 FIFO 配对成持仓段；期末未平仓段
  单独报告（税在卖出时才按当时持有期定档，这里不计）。
- 数据纯离线：load_daily / load_dividends 均 auto_update=False。

用法：
    .venv/Scripts/python.exe -m backtest.scripts._dividend_tax_check
"""

import pandas as pd

from backtest.lib.console import use_utf8
from backtest.lib.ladder import simulate_buy_hold, simulate_ladder
from backtest.lib.price_store import load_daily, load_dividends
from backtest.lib.trend_stop import simulate as trend_stop_simulate

CAPITAL = 100_000
START = "20180101"

# 本地已有分红数据的高股息标的（data/market/dividend/）
SYMBOLS = ["601857", "600028"]


def fifo_segments(trades: list[dict], last_date) -> tuple[list, list]:
    """ladder 成交流水 → FIFO 配对的持仓段。返回（已平仓段， 未平仓段）。"""
    lots, closed = [], []
    for t in trades:
        if t["action"] == "buy":
            lots.append([t["date"], t["shares"]])
            continue
        qty = t["shares"]
        while qty > 0 and lots:
            take = min(qty, lots[0][1])
            lots[0][1] -= take
            qty -= take
            closed.append((lots[0][0], t["date"], take))
            if lots[0][1] == 0:
                lots.pop(0)
    open_segs = [(d, last_date, q) for d, q in lots]
    return closed, open_segs


def tax_of(segs: list, div: pd.DataFrame) -> tuple[float, int]:
    """逐段匹配除息日，按持有期分档计税。返回（税额, 跨越除息次数）。"""
    total, crosses = 0.0, 0
    for b, s, qty in segs:
        days = (s - b).days
        rate = 0.20 if days <= 30 else (0.10 if days <= 365 else 0.0)
        hit = div[(div.ex_date > b) & (div.ex_date <= s)]
        crosses += len(hit)
        total += float((hit["cash_before_tax"] * qty * rate).sum())
    return total, crosses


def report(name: str, symbol: str, segs: list, open_segs: list,
           div: pd.DataFrame) -> dict:
    tax, crosses = tax_of(segs, div)
    hold_days = sum((s - b).days for b, s, _ in segs)
    row = {
        "标的": symbol, "打法": name,
        "平仓段数": len(segs), "未平仓段": len(open_segs),
        "合计持仓天": hold_days, "跨越除息": crosses,
        "税额¥": round(tax, 0), "占本金%": round(tax / CAPITAL * 100, 3),
    }
    print(f"  {symbol} {name:<14s} 段 {row['平仓段数']:>3d}（未平 {row['未平仓段']}） "
          f"持仓 {hold_days:>5d} 天  跨越除息 {crosses:>2d} 次  "
          f"税 ¥{tax:>8.0f}（{tax / CAPITAL:+.3%} 本金）")
    return row


def main():
    use_utf8()
    rows = []
    print("\n中长持仓打法 × 高股息标的：红利税实算（纯离线）")
    print("─" * 78)
    for symbol in SYMBOLS:
        df = load_daily(symbol, START, auto_update=False, verbose=False)
        div = load_dividends(symbol, auto_update=False)
        if div.empty:
            print(f"  {symbol} 无本地分红数据，跳过")
            continue

        # ladder 波段（默认档距，PLAYBOOK 主力参数族）
        lad = simulate_ladder(df, CAPITAL)
        closed, open_segs = fifo_segments(lad.trades, df.index[-1])
        rows.append(report("ladder 波段", symbol, closed, open_segs, div))

        # ladder 带止盈：才会产生完整往返，落在 1 个月–1 年的 10% 档
        lad_tp = simulate_ladder(df, CAPITAL, take_profit=0.30)
        closed, open_segs = fifo_segments(lad_tp.trades, df.index[-1])
        rows.append(report("ladder+止盈30%", symbol, closed, open_segs, div))

        # 买入持有对照（>1 年 0% 档，税额应为 0）
        bh = simulate_buy_hold(df, CAPITAL)
        closed, open_segs = fifo_segments(bh.trades, df.index[-1])
        rows.append(report("买入持有", symbol, closed, open_segs, div))

        # trend_stop 月频（净值归一没有股数，按 本金/入场价 近似）
        ts = trend_stop_simulate(df, ma_len=150, stop=0.15)
        segs = [(r.entry_date, r.exit_date, CAPITAL / r.entry_px)
                for r in ts.trades.itertuples()]
        rows.append(report("trend_stop 月频", symbol, segs, [], div))

    print("─" * 78)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
