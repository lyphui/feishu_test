"""
港股原油 ETF 月频信号台：每月底跑一次，告诉你该买、该卖、还是不动。

主用标的 3175.HK（三星标普高盛原油期货 ETF）。规则见 `lib/trend_stop.py`：
月末收盘价 vs MA150 决定次日进出，持仓期间从入场后最高收盘价回落 15% 立即离场。

用法
----
    python -m backtest.scripts.track_hk_oil_etf                  # 更新行情 + 当前信号 + 回测
    python -m backtest.scripts.track_hk_oil_etf --offline        # 不联网，只读本地缓存
    python -m backtest.scripts.track_hk_oil_etf --sweep          # 附参数敏感性网格
    python -m backtest.scripts.track_hk_oil_etf --ma 120 --stop 0.12
    python -m backtest.scripts.track_hk_oil_etf --symbol 3097.HK --capital 50000

费用口径
--------
换仓成本按 `lib.trend_stop.hk_trade_cost` 由本金实算（中银香港零售档：佣金
0.25% 最低 HK$100，另加固定服务费 30，ETF 免印花税），而不是拍一个固定比例。
本金越小费用率越高，回测结果会随 `--capital` 变化——这是真实的，不是 bug。
"""

import argparse

import pandas as pd

from backtest.lib.price_store import load_daily
from backtest.lib.trend_stop import (buy_hold, hk_fee_rate, hk_trade_cost,
                            next_decision_date, simulate, sweep)
from backtest.lib.console import use_utf8
from backtest.lib.cli import base_parser

DEFAULT_SYMBOL = "3175.HK"
HISTORY_START = "20160101"
NAMES = {"3175.HK": "三星标普高盛原油期货 ETF",
         "3097.HK": "GlobalX 标普原油期货 ETF"}
SWEEP_MAS = [80, 100, 120, 150, 180, 200, 250]
SWEEP_STOPS = [None, 0.10, 0.12, 0.15, 0.20, 0.25]


def print_state(res, capital: float, fee_rate: float) -> None:
    st = res.state
    print(f"\n{'='*78}")
    print("▶ 当前信号")
    print(f"{'='*78}")
    print(f"  数据截至    {st['last_date']:%Y-%m-%d}　收盘 {st['last_close']:.3f}")
    if st.get("ma") is None:
        print(f"  MA{st['ma_len']} 尚未成形，样本不足，不给结论")
        return
    gap = st["ma_gap"]
    # 这行是「今天若是月末会怎么判」的预览，不是当前仓位——真正的换仓只在月末发生
    print(f"  MA{st['ma_len']}       {st['ma']:.3f}　"
          f"现价{'高于' if gap > 0 else '低于'} {abs(gap):.1%}　"
          f"若今天是月末则：{'做多' if st['signal_long'] else '离场'}")

    if st["holding"]:
        print(f"  持仓中      入场 {st['entry_date']:%Y-%m-%d} @ {st['entry_px']:.3f}"
              f"　浮动 {st['last_close']/st['entry_px']-1:+.1%}")
        print(f"  期间最高    {st['peak']:.3f}　当前距高点 "
              f"{st['last_close']/st['peak']-1:+.1%}")
        buf = st["last_close"] / st["stop_px"] - 1
        print(f"  移动止损线  {st['stop_px']:.3f}　"
              + (f"⚠ 已跌破 → 立即清仓" if buf < 0 else f"尚有 {buf:.1%} 缓冲"))
    else:
        print("  空仓中      等下一个月末信号转多再进场")

    nd = next_decision_date(st["last_date"])
    print(f"  下次决策日  {nd:%Y-%m-%d}（当月最后一个工作日，遇港交所假期前移）")

    print("\n  今天该做什么：")
    if st["holding"]:
        if st["last_close"] < st["stop_px"]:
            print("    ▸ 触发移动止损 —— 清仓，等下个月末信号仍为多才重入")
        else:
            print(f"    ▸ 继续持有。跌破 {st['stop_px']:.3f} 立刻走；"
                  f"否则等 {nd:%m-%d} 收盘再看均线")
    else:
        if st["signal_long"]:
            print(f"    ▸ 空仓但信号为多 —— 等 {nd:%m-%d} 月末确认后次日买入")
        else:
            print("    ▸ 继续空仓")

    shares = int(capital / st["last_close"])
    cost = hk_trade_cost(capital)
    print(f"\n  按本金 {capital:,.0f} 港币：约 {shares:,} 股，"
          f"单边费用 {cost:.0f} 港币（{fee_rate:.2%}）")
    if fee_rate > 0.005:
        print(f"    ⚠ 费用率超过 0.5%，本金偏小。最低佣金 HK$100 意味着"
              f"单笔低于 4 万港币就在交冤枉钱")


def print_backtest(res, bh, df) -> None:
    print(f"\n{'='*78}")
    print(f"▶ 回测　{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}"
          f"（{res.stats['years']:.1f} 年，{len(df)} 个交易日）")
    print(f"{'='*78}")
    rows = [{"策略": r.name,
             "年化": f"{r.stats['annual_return']:.1%}",
             "总收益": f"{r.stats['total_return']:.0%}",
             "波动": f"{r.stats['volatility']:.0%}",
             "最大回撤": f"{r.stats['max_drawdown']:.1%}",
             "夏普": f"{r.stats['sharpe']:.2f}",
             "交易/年": f"{r.stats['trades_per_year']:.1f}",
             "在场": f"{r.stats['exposure']:.0%}"} for r in (res, bh)]
    print("  " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n  "))

    t = res.trades
    if len(t):
        won = (t["ret"] > 0).sum()
        print(f"\n  逐段持仓：{len(t)} 段，胜率 {won}/{len(t)} = {won/len(t):.0%}，"
              f"平均收益 {t['ret'].mean():+.1%}，平均持有 {t['days'].mean():.0f} 天")
        print("  ⚠ 低胜率是趋势跟踪的常态：靠少数大赢家覆盖多次小亏，"
              "扛不住连亏就别用这套")
        show = t.tail(6).assign(
            entry_date=lambda x: x["entry_date"].dt.strftime("%Y-%m-%d"),
            exit_date=lambda x: x["exit_date"].dt.strftime("%Y-%m-%d"),
            ret=lambda x: x["ret"].map("{:+.1%}".format),
            entry_px=lambda x: x["entry_px"].map("{:.3f}".format),
            exit_px=lambda x: x["exit_px"].map("{:.3f}".format))
        print("\n  最近 6 段：")
        print("    " + show.to_string(index=False).replace("\n", "\n    "))


def print_sweep(df, fee_rate) -> None:
    print(f"\n{'='*78}")
    print("▶ 参数敏感性（年化收益 %）—— 看最优点是不是一整片高原")
    print(f"{'='*78}")
    g = sweep(df, SWEEP_MAS, SWEEP_STOPS, fee=fee_rate)
    piv = (g.pivot(index="ma", columns="stop", values="年化") * 100).round(1)
    piv.columns = ["无止损" if c == 0 else f"-{c:.0%}" for c in piv.columns]
    piv.index = [f"MA{m}" for m in piv.index]
    print("  " + piv.to_string().replace("\n", "\n  "))
    print("\n  ▶ 最大回撤 %")
    piv2 = (g.pivot(index="ma", columns="stop", values="最大回撤") * 100).round(1)
    piv2.columns = ["无止损" if c == 0 else f"-{c:.0%}" for c in piv2.columns]
    piv2.index = [f"MA{m}" for m in piv2.index]
    print("  " + piv2.to_string().replace("\n", "\n  "))
    print("\n  相邻格子塌陷 = 拟合噪音；连成一片才敢用。选高原中部，别选峰顶。")


def main():
    use_utf8()
    ap = argparse.ArgumentParser(description="港股原油 ETF 月频均线+移动止损信号",
                                 parents=[base_parser()])
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL, help="港交所代码，如 3175.HK")
    ap.add_argument("--ma", type=int, default=150, help="均线长度（交易日）")
    ap.add_argument("--stop", type=float, default=0.15,
                    help="移动止损幅度，0 表示不设")
    ap.add_argument("--freq", default="month", choices=["day", "month"],
                    help="决策频率，默认月频（日频会被手续费吃光，别用）")
    ap.add_argument("--sweep", action="store_true", help="附参数敏感性网格")
    ap.set_defaults(capital=100_000, start=HISTORY_START)
    args = ap.parse_args()

    # 港股走 yfinance（前复权口径），price_store 对 qfq 强制整表重建
    df = load_daily(args.symbol, args.start, adjust="qfq", kind="hk",
                    auto_update=not args.offline, verbose=not args.offline)
    if df.empty or len(df) < args.ma + 2:
        raise SystemExit(f"{args.symbol} 数据不足（{len(df)} 行），"
                         f"MA{args.ma} 至少需要 {args.ma + 2} 行")

    fee_rate = hk_fee_rate(args.capital)
    stop = args.stop if args.stop and args.stop > 0 else None

    name = NAMES.get(args.symbol, args.symbol)
    freq_cn = {"day": "日", "month": "月"}[args.freq]
    print(f"\n{name}　{args.symbol}　"
          f"规则：MA{args.ma} {freq_cn}频 + 移动止损 "
          f"{f'{stop:.0%}' if stop else '无'}")

    res = simulate(df, ma_len=args.ma, stop=stop, fee=fee_rate, freq=args.freq)
    print_state(res, args.capital, fee_rate)
    print_backtest(res, buy_hold(df, fee_rate), df)
    if args.sweep:
        print_sweep(df, fee_rate)

    print(f"\n{'='*78}")
    print("  口径提醒：月末信号次日按收盘价成交、止损按触发当日收盘离场，"
          "两条都偏乐观；")
    print("  参数是在同一段样本上选的，实盘请打折看。")


if __name__ == "__main__":
    main()
