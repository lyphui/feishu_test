"""
「乌龙指捕捉」策略实测台：50% 持仓挂高价卖、50% 现金挂低价买，成交后回到 50/50。

这套打法的假设是：偶尔有人敲错价格，把单子送到远离市价的地方，你挂在那里就能
捡到，而且捡完价格会回到正常水平。本脚本用 601857 / 600938 的全样本日线检验
两件事——**挂得出去吗**（涨跌停限制）、**捡到的是乌龙指还是趋势**（fill edge）。

用法
----
    python backtest/fatfinger_bench.py                     # 默认两只票、五档 k
    python backtest/fatfinger_bench.py --offline           # 不联网，只读本地缓存
    python backtest/fatfinger_bench.py --k 0.02 0.05 0.08
    python backtest/fatfinger_bench.py --fast              # 卖单当日收盘就回补
    python backtest/fatfinger_bench.py --symbols 601857

口径
----
后复权（hfq）行情、`lib.ladder` 的成本模型（佣金万三最低 5 元、印花税千一、
主动单滑点千一），闲置现金按 1.5% 计息。限价单成交不吃滑点、跳空按开盘价成交、
日内每次触及都算成交——三条都偏袒策略，详见 `lib/fatfinger.py` 的 docstring。
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.fatfinger import (ROUND_TRIP_BP, fill_edge, simulate_fatfinger,
                           simulate_static_mix)
from lib.ladder import simulate_buy_hold, simulate_grid
from lib.price_store import load_daily
from lib.console import use_utf8

DEFAULT_SYMBOLS = ["601857", "600938"]
DEFAULT_KS = [0.02, 0.03, 0.05, 0.08, 0.095, 0.30]
NAMES = {"601857": "中国石油", "600938": "中国海油"}
HISTORY_START = "20180101"
LIMIT_PCT = 0.10


def touch_stats(df: pd.DataFrame, ks) -> pd.DataFrame:
    """
    先看物理上限：以**昨收**为锚，日内高/低点够得着 ±k 的天数占比。

    「当日可挂」判的也是相对昨收——超出 ±10% 交易所拒收。注意这**不等于**这张单
    永远挂不出去：锚价固定在上次再平衡的成交价，股价慢慢漂开之后，同一个 k 相对
    新的前收就落回涨跌停带内了。所以 k=30% 的单子不是永远挂不出，而是要等股价
    自己走出 20%+ 的趋势才挂得上——那时候成交的显然不是谁敲错了。
    """
    pc = df["close"].shift(1)
    up, dn = df["high"] / pc - 1, df["low"] / pc - 1
    return pd.DataFrame([{
        "k": f"±{k:.1%}",
        "冲高触及": f"{(up >= k).sum()} 天 ({(up >= k).mean():.2%})",
        "下探触及": f"{(dn <= -k).sum()} 天 ({(dn <= -k).mean():.2%})",
        "当日可挂": "否·超涨跌停" if k > LIMIT_PCT else "可",
    } for k in ks])


def run_symbol(symbol: str, df: pd.DataFrame, ks, capital: float, fast: bool) -> None:
    name = NAMES.get(symbol, symbol)
    print(f"\n{'='*110}")
    print(f"{name} {symbol}　{df.index[0].date()}~{df.index[-1].date()}"
          f"（{len(df)} 个交易日，本金 {capital:,.0f}，后复权口径）")
    print(f"{'='*110}")

    print("\n▶ 物理可行性：日内够不够得着（锚 = 前收，涨跌停 ±10%）")
    print("  " + touch_stats(df, ks).to_string(index=False).replace("\n", "\n  "))

    baselines = [
        simulate_buy_hold(df, capital),
        simulate_static_mix(df, capital, target=0.5),
        simulate_static_mix(df, capital, target=0.5, rebalance_days=21),
        # 同一个"低买高卖来回做"的想法，但底仓不动、只拿一半资金分格滚动，
        # 且卖出按 LIFO 真实股数——这是这套思路被正确实现之后的样子
        simulate_grid(df, capital, base_position=0.5, n_grids=5, grid_step=0.07),
    ]
    runs = [simulate_fatfinger(df, capital, k_up=k, k_dn=k,
                               limit_pct=LIMIT_PCT, fast_sell_rebalance=fast)
            for k in ks]

    print("\n▶ 净值对比（连续全样本，不分段、不重置资金）")
    rows = [{"策略": r.name, "总收益": f"{r.stats['total_return']:.1%}",
             "年化": f"{r.stats['annual_return']:.1%}",
             "最大回撤": f"{r.stats['max_drawdown']:.1%}",
             "平均仓位": f"{r.stats['avg_exposure']:.0%}",
             "夏普": f"{r.stats['sharpe']:.2f}",
             "笔数": r.stats["n_trades"], "限价成交": "—"} for r in baselines]
    for r in runs:
        rows.append({"策略": r.name, "总收益": f"{r.stats['total_return']:.1%}",
                     "年化": f"{r.stats['annual_return']:.1%}",
                     "最大回撤": f"{r.stats['max_drawdown']:.1%}",
                     "平均仓位": f"{r.stats['avg_exposure']:.0%}",
                     "夏普": f"{r.stats['sharpe']:.2f}",
                     "笔数": r.stats["n_trades"],
                     "限价成交": f"{r.diag['n_fills']}"})
    print("  " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n  "))

    ref = baselines[1].stats["total_return"]
    print(f"\n▶ 相对「静态50%仓·躺平」的超额（该基准总收益 {ref:.1%}）")
    print("  " + "   ".join(
        f"±{r.diag['k_up']:.1%}: {r.stats['total_return'] - ref:+.1%}" for r in runs))

    print("\n▶ 成交质量：捡到的是乌龙指还是趋势？")
    print(f"  edge = 成交价相对回补价的**毛**价差（bp，不含滑点佣金）；")
    print(f"  门槛不是 0 而是单边摩擦 {ROUND_TRIP_BP:.0f}bp —— 跨不过去就等于没捡到；")
    print("  fwdN = 成交后 N 日股价平均涨跌（卖单成交后仍上涨 = 卖飞了）")
    for r in runs:
        d = r.diag
        if d["n_fills"] == 0:
            reason = ("挂单价超出涨跌停，交易所拒收"
                      if d["n_rejected_sell_days"] + d["n_rejected_buy_days"] > 0
                      else "全样本从未触及")
            print(f"\n  ±{d['k_up']:.1%}：0 笔成交（{reason}）"
                  f"　拒收天数 卖{d['n_rejected_sell_days']}/买{d['n_rejected_buy_days']}")
            continue
        e = fill_edge(r.fills)
        e = e.assign(edge_mean_bp=e["edge_mean_bp"].round(1),
                     edge_median_bp=e["edge_median_bp"].round(1),
                     t=e["t"].round(2), win_rate=e["win_rate"].map("{:.0%}".format),
                     fwd1=e["fwd1"].map("{:+.2%}".format),
                     fwd5=e["fwd5"].map("{:+.2%}".format),
                     fwd20=e["fwd20"].map("{:+.2%}".format))
        print(f"\n  ±{d['k_up']:.1%}：共 {d['n_fills']} 笔"
              f"（卖 {d['n_sell']} / 买 {d['n_buy']}），"
              f"平均每年 {d['fills_per_year']:.1f} 笔"
              + (f"，两侧同日触发 {d['n_both_sides_days']} 天" if d["n_both_sides_days"] else ""))
        print("    " + e.to_string(index=False).replace("\n", "\n    "))


def main():
    use_utf8()
    ap = argparse.ArgumentParser(description="乌龙指捕捉策略实测")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--k", nargs="+", type=float, default=DEFAULT_KS,
                    help="挂单相对锚价的偏离幅度，可给多个")
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--offline", action="store_true", help="不联网，只读本地缓存")
    ap.add_argument("--fast", action="store_true",
                    help="卖单成交后当日收盘即回补（买单仍受 T+1 约束）")
    args = ap.parse_args()

    if args.fast:
        print("※ 快速回补模式：卖单成交当日收盘就买回（A 股卖出资金当日可用）；"
              "买单仍须等 T+1")

    for s in args.symbols:
        df = load_daily(s, HISTORY_START, adjust="hfq",
                        auto_update=not args.offline, verbose=False)
        run_symbol(s, df, args.k, args.capital, args.fast)


if __name__ == "__main__":
    main()
