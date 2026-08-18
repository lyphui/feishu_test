"""
单票打法对比台：给定一只票，回答「这只票该用什么打法」
====================================================

    python -m backtest.scripts.compare_playbooks --code 688256 --name 寒武纪
    python -m backtest.scripts.compare_playbooks --code 601899 --start 20180101 --offline

为什么单独一个入口
------------------
`compare_ma_cross.py` 比的是**同一个策略家族在不同品类上**的表现；这里反过来，
比的是**同一只票上不同打法**——择时、分批、趋势跟踪三条路，各自的模拟器在
仓库里本来就有（`engine` / `lib.ladder` / `lib.trend_stop`），缺的只是把它们
摆到同一张表、同一段数据、同一套成本口径上。散在三个脚本里各跑各的，
数字就没法直接比大小。

对比的三类打法
--------------
  1. **满仓类**：一次性满仓持有（基准）、定投分批
  2. **分批/仓位管理**（`lib.ladder`）：回撤梯度加仓、网格、按市场状态自适应
  3. **趋势跟踪**（`lib.trend_stop` / `engine`）：月频均线+移动止损、日频 MA 交叉

口径
----
  * 行情一律 hfq（后复权，含股息再投），三套模拟器共用 `lib/costs.py` 的费率
  * 涨跌停幅度按代码前缀推断（科创板/创业板 20%），**不能用主板 10% 硬编码**：
    对 688/300 的票会把本可成交的单子判成封死
  * `trend_stop` 是收盘价撮合、按比例扣费的简化模型，与前两套的整手撮合略有
    出入，**不要拿它的小数点后一位和 ladder 比**，看的是量级与排序

输出的四张表
------------
  ① 全区间打法对比
  ② 阶段划分：连续 ≥20 个交易日的段落，含起止日期与区间涨跌，便于对着 K 线核对
  ③④ 分阶段年化% / 累计%：把每个打法的日收益按**当日**市场状态分组
     （`lib/regime.classify`，全 rolling + 滞回，标签当天可知，不是未来函数）

  ③④ 是**归因**不是可交易策略：同一状态的日子在时间上并不连续，连乘等于假设
  能在状态切换瞬间无成本进出。真按状态切换的结果看「自适应」那一行——它带
  成本、带滞回，是能落地的版本。表末的 `regime_stats` 用来检验标签本身有没有
  信息量：三档的未来 60 日收益若长得一样，按它切换只会白付摩擦成本。

怎么读这张表
------------
高波动个股上，"总收益最高"几乎总是满仓持有——它天然吃满了全部涨幅，
代价是吃满全部回撤。所以必须同时看三列：
  * **最大回撤**：这只票真正的问题往往在这里，−70% 级别的回撤拿不住就没有然后
  * **平均仓位**：分批策略手里长期有现金，收益本来就该低一截
  * **投入资金收益**（`deployed_return`）：按实际压上去的钱算，才是"这笔钱值不值"
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.lib import costs, trend_stop
from backtest.lib.cli import base_parser
from backtest.lib.console import print_wide, use_utf8
from backtest.lib.manifest import write_run_manifest
from backtest.lib.ladder import (simulate_adaptive, simulate_buy_hold, simulate_dca,
                        simulate_grid, simulate_ladder)
from backtest.lib.price_store import load_daily
from backtest.lib.regime import (BEAR, CHOP, LABELS, TREND_UP, classify,
                        regime_episodes, regime_stats)
from backtest.lib.swings import drawdown_profile
from backtest.strategies import MACrossStrategy

TRADING_DAYS = 252
DEFAULT_START = "20180101"

# trend_stop 是按比例扣费的简化模型，只能给一个**单边**费率，这里取买卖两侧的
# 平均：买入 0.13%（佣金万三 + 滑点千一）、卖出 0.23%（再加印花千一），平均 0.18%。
ASHARE_FEE = costs.COMMISSION_RATE + costs.SLIPPAGE + costs.STAMP_DUTY / 2

# 日频 MA 交叉对照组的参数；最长的慢线决定了统计窗口起点（见 eval_start_of）
MA_CROSS_VARIANTS = ((5, 8), (20, 60))


def profile(df: pd.DataFrame) -> dict:
    ret = df["close"].pct_change().dropna()
    return {
        "交易日": len(df),
        "年化波动%": float(ret.std() * np.sqrt(TRADING_DAYS) * 100),
        "日收益ρ1": float(ret.autocorr(1)),
        "最大单日%": float(ret.max() * 100),
        "最小单日%": float(ret.min() * 100),
    }


def eval_start_of(df: pd.DataFrame) -> pd.Timestamp:
    """
    统计窗口起点：**所有打法都已经有净值**的第一天。

    `ladder` / `trend_stop` 从数据首日就有净值（预热期是空仓，净值恒定），
    但引擎那两行要等 `MACrossStrategy.prepare()` 的 dropna 走完才有第一条记录，
    MA20/60 就是 59 根。不对齐的话，"同一段数据"这句话是假的——688256 上
    三个窗口的买入持有分别是 +703% / +606% / +834%，表底那句「几个打法跑赢
    满仓持有」比的其实是起跑线。同 `ma_cross_bench.common_eval_start()`。
    """
    warm = max(slow for _, slow in MA_CROSS_VARIANTS) - 1
    return df.index[min(warm, len(df) - 1)]


def _sharpe(ret: pd.Series, rf: float = costs.RISK_FREE_RATE, min_obs: int = 20):
    """年化夏普，公式与 `engine._calc_sharpe` 一致（含无风险利率项）。

    三套模拟器自带的 `sharpe` 口径并不相同（`trend_stop` 不减无风险利率、
    年化天数也各按各的），混在一张表里排序等于按公式排序，所以这里一律重算。
    """
    excess = (ret - rf / TRADING_DAYS).dropna()
    if len(excess) < min_obs:
        return float("nan")
    std = float(excess.std())
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def window_stats(name: str, equity: pd.Series, exposure: pd.Series,
                 fills: list, start: pd.Timestamp) -> dict:
    """
    统一从「净值 + 仓位序列」算指标，三类模拟器共用一套公式。

    以前是各用各自 `stats` 里的字段，于是同一张表里的「年化」有的按交易日、
    有的按自然日折算，「投入资金收益」对 ladder 是按投入资金、对择时类是按
    在场比例——列名一样，含义不同，排序就没有意义。

    收益分母取**窗口前一日**的权益：窗口起点之前的持仓盈亏不该算进窗口收益。
    引擎那两行在窗口起点之前没有净值，退化为用起点当日权益（= 初始资金）。
    """
    eq_all = equity.astype(float).dropna()
    eq = eq_all.loc[start:]
    if len(eq) < 2:
        return {"打法": name}
    prior = eq_all.loc[eq_all.index < start]
    base = float(prior.iloc[-1]) if len(prior) else float(eq.iloc[0])

    total = eq.iloc[-1] / base - 1
    ann = (1 + total) ** (TRADING_DAYS / len(eq)) - 1
    # 回撤的历史高点要把窗口起点的本金算进去，否则起点即下跌的那段回撤会被漏掉
    dd = float((eq / eq.cummax().clip(lower=base) - 1).min())
    exp = exposure.reindex(eq_all.index).astype(float).loc[start:]
    avg_exp = float(exp.mean()) if exp.notna().any() else float("nan")
    ret = eq_all.pct_change().loc[start:]

    return {
        "打法": name,
        "总收益%": total * 100,
        "年化%": ann * 100,
        "最大回撤%": dd * 100,
        "平均仓位%": avg_exp * 100,
        # 只按实际压上去的钱算：空仓/留现金的打法不该因为"没满仓"被扣分
        "投入资金收益%": (total / avg_exp * 100
                          if avg_exp and np.isfinite(avg_exp) and avg_exp > 1e-6
                          else float("nan")),
        "夏普": _sharpe(ret),
        # 一律数**统计窗口内的成交笔数**（买卖各算一笔）。引擎的 `total_trades`
        # 只数买入，混在一起会让择时行看起来只有实际的一半活跃。
        # 注意：满仓持有/梯度长持的底仓建在预热期，窗口内笔数会是 0——
        # 那是「窗口内没再动手」，不是「从没建仓」，展示层需注明。
        "窗口内笔数": sum(1 for d in fills if d >= start),
    }


def engine_exposure(r: dict) -> pd.Series:
    """引擎结果 → 逐日仓位占比（持仓市值 / 总权益）。取全量曲线，窗口在外层裁。"""
    eq = r["equity_curve_full"]
    return (eq["shares"] * eq["close"] / eq["equity"]).astype(float)


def trend_exposure(r) -> pd.Series:
    """趋势跟踪结果 → 逐日仓位。净值用的是滞后一天的仓位，敞口口径须一致。"""
    return r.position.shift(1).fillna(0.0).astype(float)


def trend_fills(r) -> list:
    """趋势跟踪的成交日：仓位发生变化的那些天（进、出各算一笔）。"""
    pos = r.position
    change = pos.diff().fillna(pos.iloc[0]).abs()
    return list(change.index[change > 0])


def sized_capital(df: pd.DataFrame, capital: float, min_lots: int = 30) -> float:
    """后复权价可能远高于盘面报价，本金不够整手会让引擎一次都不成交。"""
    return max(capital, float(df["open"].iloc[0]) * costs.LOT * min_lots)


def build_rows(df: pd.DataFrame, symbol: str, capital: float, *,
               start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """跑完全部打法，返回（汇总表，各打法的净值曲线）。

    净值曲线一并带出来，是为了在**同一批回测**上做分阶段拆解——重跑一遍
    再拆，两张表的数字就可能对不上。全部指标经 `window_stats` 统一重算，
    统计窗口对齐到 `eval_start_of(df)`。
    """
    limit_pct = costs.infer_limit_pct(symbol)       # 科创板/创业板 20%，别用主板 10%
    reg = classify(df)["regime"]
    lad = dict(cash_rate=costs.CASH_RATE, limit_pct=limit_pct)
    eval_start = eval_start_of(df)

    rows, curves = [], {}

    for r in [
        simulate_buy_hold(df, capital, **lad),
        simulate_dca(df, capital, n_tranches=10, every_days=21, **lad),
        simulate_ladder(df, capital, n_tranches=4, step=0.08,
                        name="梯度4×8% 长持", **lad),
        simulate_ladder(df, capital, n_tranches=4, step=0.08, take_profit=0.30,
                        tp_fraction=0.5, name="梯度4×8% 半止盈30", **lad),
        simulate_ladder(df, capital, n_tranches=4, step=0.08, trail_stop=0.25,
                        name="梯度4×8% 移动止损25", **lad),
        simulate_grid(df, capital, base_position=0.5, n_grids=5, grid_step=0.07,
                      **lad),
        simulate_adaptive(df, reg, capital, **lad),
    ]:
        rows.append(window_stats(r.name, r.equity, r.exposure,
                                 [t["date"] for t in r.trades], eval_start))
        curves[r.name] = r.equity

    for ma_len in (100, 150, 200):
        for stop in (None, 0.25):
            r = trend_stop.simulate(df, ma_len=ma_len, stop=stop,
                                    fee=ASHARE_FEE, freq="month")
            name = f"月频MA{ma_len}" + (f"+止损{stop:.0%}" if stop else "（无止损）")
            rows.append(window_stats(name, r.equity, trend_exposure(r),
                                     trend_fills(r), eval_start))
            curves[name] = r.equity

    for fast, slow in MA_CROSS_VARIANTS:
        r = run_backtest(symbol, start, end, df=df,
                         strategy=MACrossStrategy(fast=fast, slow=slow),
                         initial_capital=capital, limit_pct=limit_pct,
                         commission_rate=costs.COMMISSION_RATE,
                         min_commission=costs.MIN_COMMISSION,
                         stamp_duty=costs.STAMP_DUTY, slippage=costs.SLIPPAGE)
        name = f"日频MA{fast}/{slow} 交叉"
        eq = r["equity_curve_full"]["equity"]
        fills = list(r["trades"]["date"]) if not r["trades"].empty else []
        rows.append(window_stats(name, eq, engine_exposure(r), fills, eval_start))
        curves[name] = eq

    return pd.DataFrame(rows), curves


# ── 分阶段拆解 ────────────────────────────────────────────────────────────────

def regime_breakdown(curves: dict, reg: pd.Series,
                     index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    把每个打法的日收益按当日市场状态分组，返回（分阶段年化%，分阶段累计%）。

    口径与三条注意事项
    ------------------
    * 状态标签是**当日可知**的（`lib/regime.classify` 全是 rolling，带滞回），
      所以按它分组不构成未来函数。
    * 但这是**归因**，不是可交易策略：同一状态的日子在时间上并不连续，
      把它们的日收益连乘等于假设你能在状态切换的瞬间无成本进出。真要按状态
      切换请看「自适应」那一行——它是带成本、带滞回的真实模拟。
    * 各打法的净值起点不同（引擎那两行要等均线预热完才有净值），因此按各自
      **实际有净值的天数**折年化，天数一并列出。
    """
    rates, totals = {}, {}
    for name, eq in curves.items():
        ret = eq.reindex(index).astype(float).pct_change()
        g = pd.DataFrame({"r": ret, "regime": reg}).dropna()
        row_ann, row_tot = {}, {}
        for label in LABELS:
            seg = g.loc[g["regime"] == label, "r"]
            if len(seg) < 10:                  # 样本太少，年化没有意义
                row_ann[label] = float("nan")
                row_tot[label] = float("nan")
                continue
            cum = float((1 + seg).prod() - 1)
            row_tot[label] = cum * 100
            row_ann[label] = ((1 + cum) ** (TRADING_DAYS / len(seg)) - 1) * 100
        rates[name] = row_ann
        totals[name] = row_tot

    ann = pd.DataFrame(rates).T[[l for l in LABELS]]
    tot = pd.DataFrame(totals).T[[l for l in LABELS]]
    return ann, tot


def episode_table(df: pd.DataFrame, reg_df: pd.DataFrame, min_days: int = 20,
                  since: pd.Timestamp = None) -> pd.DataFrame:
    """连续 ≥min_days 个交易日的阶段区间表（含区间涨跌），供逐段拆解复用。

    `since` = 统计窗口起点。**跨在起点上的那一段截断到起点**，而不是整段丢掉：
    直接丢会连带扔掉一整段行情（688256 上就是上市后那 273 天的下跌，恰恰是
    分批建仓优势最明显的一段）；不截断又会出现半截净值的行。被截断的段落
    标 `*`，展示层注明。完全落在起点之前的段才丢弃。
    """
    eps = regime_episodes(reg_df)
    if since is not None:
        eps = eps[eps["end"] > since].copy()
        eps["truncated"] = eps["start"] < since
        eps.loc[eps["truncated"], "start"] = since
        # 截断后重算天数，再按 min_days 过滤——剩不到 min_days 的残段不单列
        eps["days"] = [len(reg_df.loc[s:e]) for s, e in zip(eps["start"], eps["end"])]
    else:
        eps["truncated"] = False
    eps = eps[eps["days"] >= min_days].reset_index(drop=True)
    if eps.empty:
        return eps
    # 与 ⑤ 的打法列同口径：都用**段内日收益连乘**。
    # 用 close[e]/close[s] 会漏掉「进入该阶段当天」那根 K 线的涨跌，
    # 而打法列是含的——±20% 的票上，这一天就能差出一个涨停板。
    ret = df["close"].pct_change()
    eps["区间涨跌%"] = [
        float((1 + ret.loc[s:e].dropna()).prod() - 1) * 100
        for s, e in zip(eps["start"], eps["end"])
    ]
    # 行标签带序号与起止日：分阶段表要能一眼对上 ② 里的那一段
    eps["标签"] = [
        f"{i + 1:02d} {r} {s:%y-%m-%d}→{e:%y-%m-%d}" + ("*" if t else "")
        for i, (r, s, e, t) in enumerate(
            zip(eps["regime"], eps["start"], eps["end"], eps["truncated"]))
    ]
    return eps


def episode_breakdown(curves: dict, eps: pd.DataFrame,
                      index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    **逐段**收益表：行 = 具体阶段（按时间顺序），列 = 打法，值 = 该段区间收益%。

    与 `regime_breakdown` 的区别：那张表把同一状态的所有日子合并成一个数，
    看不出「同样叫趋势下行，2021 那次和 2024 那次差多少」。这张按 ② 里划出的
    每一段单独算，段内日收益连乘——段内时间是连续的，所以这个数**是**该段
    真实可实现的收益（不像合并表那样需要瞬间无成本切换的假设）。

    净值不完整的段留空，不填数——半截净值算出来的区间收益会被直接误读成
    该打法的表现（曾经出现过「标的 −57.0% 对 MA20/60 −39.8%」，而那 273 天里
    引擎只覆盖了 214 天，同期标的实为 −50.0%）。段落本身已按统计窗口起点
    过滤过，这里只是最后一道保险：覆盖率不足 99% 就不出数。
    """
    out = {}
    for name, eq in curves.items():
        ret = eq.reindex(index).astype(float).pct_change()
        col = []
        for s, e in zip(eps["start"], eps["end"]):
            span = ret.loc[s:e]
            seg = span.dropna()
            col.append(float((1 + seg).prod() - 1) * 100
                       if len(span) and len(seg) >= len(span) * 0.99
                       else float("nan"))
        out[name] = col
    return pd.DataFrame(out, index=eps["标签"].tolist())


def print_regime_periods(eps: pd.DataFrame, min_days: int = 20) -> None:
    """「阶段」到底是哪几段时间，要能对照 K 线核对。"""
    if eps.empty:
        print(f"  （没有长于 {min_days} 个交易日的阶段）")
        return
    out = pd.DataFrame({
        "序": [f"{i + 1:02d}" for i in range(len(eps))],
        "阶段": eps["regime"],
        "起": eps["start"].dt.date, "止": eps["end"].dt.date,
        "交易日": eps["days"], "区间涨跌%": eps["区间涨跌%"],
        "": ["*" if t else "" for t in eps["truncated"]],
    })
    print(out.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))
    if eps["truncated"].any():
        print("  * 该段起点早于统计窗口，已截断到窗口起点"
              "（窗口之前有的打法还没有净值）")


def print_regime_mix(reg: pd.Series) -> None:
    """状态占比。标签直接用 `lib/regime` 的常量原文——自己另起一套叫法
    （"趋势走坏" vs "趋势下行"）会让这一行和后面几张表对不上号。"""
    mix = reg.value_counts(normalize=True)
    parts = [f"{k} {v:.0%}" for k in LABELS if k in mix.index
             for v in [mix[k]]]
    print("  市场状态占比（只用当日及以前数据判定）：" + "　".join(parts))


def main():
    use_utf8()
    t0 = time.time()
    ap = argparse.ArgumentParser(description="单票打法对比",
                                 parents=[base_parser()])
    ap.set_defaults(start=DEFAULT_START, capital=100_000.0,
                    output="output/stock_playbook")
    ap.add_argument("--code", required=True, help="股票代码，如 688256")
    ap.add_argument("--name", default="", help="显示用名称")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-days", type=int, default=20,
                    help="阶段最短交易日数，短于它的段落不单列（默认 20）")
    args = ap.parse_args()

    df = load_daily(args.code, args.start, args.end, adjust="hfq",
                    auto_update=not args.offline, verbose=False)
    if len(df) < 250:
        print(f"  ❌ {args.code} 数据仅 {len(df)} 行，样本太短")
        sys.exit(1)

    capital = sized_capital(df, args.capital)
    eval_start = eval_start_of(df)
    reg_df = classify(df)
    reg = reg_df["regime"]
    # 画像与状态占比都只看统计窗口内：窗口之前的日子不进任何一张表
    p = profile(df.loc[eval_start:])
    title = f"{args.code} {args.name}".strip()

    print("\n" + "═" * 92)
    print(f"  单票打法对比  {title}  "
          f"{eval_start.date()} → {df.index[-1].date()}（{p['交易日']} 个交易日）")
    print(f"  本金 ¥{capital:,.0f}"
          + ("（后复权价过高，已按首日 30 手放大）" if capital > args.capital else "")
          + f"　涨跌停 ±{costs.infer_limit_pct(args.code):.0%}")
    print(f"  数据自 {df.index[0].date()} 起（前 "
          f"{df.index.get_loc(eval_start)} 根为均线预热，不计入统计）")
    print("═" * 92)

    print(f"\n  标的画像：年化波动 {p['年化波动%']:.1f}%　"
          f"日收益ρ1 {p['日收益ρ1']:+.3f}　"
          f"单日最大 {p['最大单日%']:+.1f}% / 最小 {p['最小单日%']:+.1f}%")
    print_regime_mix(reg.loc[eval_start:])

    prof = drawdown_profile(df["close"].loc[eval_start:])
    if not prof.empty:
        print("\n  回撤修复耗时（历史上跌这么多之后，平均多久才回到前高）：")
        print("  " + prof.to_string().replace("\n", "\n  "))

    table, curves = build_rows(df, args.code, capital,
                               start=args.start,
                               end=args.end or df.index[-1].strftime("%Y%m%d"))
    print("\n" + "═" * 92)
    print("  ① 打法对比·全区间（同一统计窗口、同一成本口径，指标统一重算）")
    print("═" * 92)
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    bh = table.iloc[0]
    print("\n  「窗口内笔数」只数统计窗口内的成交；满仓持有与梯度长持的底仓建在"
          "预热期，\n  显示 0 是「窗口内没再动手」，不是「从没建仓」。")
    print(f"\n  基准（一次性满仓持有）：总收益 {bh['总收益%']:+.1f}%　"
          f"年化 {bh['年化%']:+.1f}%　最大回撤 {bh['最大回撤%']:.1f}%")
    beat = table[table["总收益%"] > bh["总收益%"]]
    softer = table[table["最大回撤%"] > bh["最大回撤%"]]
    print(f"  收益跑赢满仓持有的打法：{len(beat)}/{len(table) - 1}"
          + ("（" + "、".join(beat["打法"]) + "）" if len(beat) else ""))
    print(f"  回撤浅于满仓持有的打法：{len(softer)}/{len(table) - 1}")

    # ── 分阶段 ──
    print("\n" + "═" * 92)
    print(f"  ② 阶段划分（连续 ≥{args.min_days} 个交易日的段落；"
          f"标签只用当日及以前数据判定）")
    print("═" * 92)
    eps = episode_table(df, reg_df, min_days=args.min_days, since=eval_start)
    print_regime_periods(eps, args.min_days)

    reg_win = reg.loc[eval_start:]
    counts = reg_win.value_counts()
    print("\n  各阶段合计交易日：" + "　".join(
        f"{l} {int(counts.get(l, 0))} 天（{counts.get(l, 0) / len(reg_win):.0%}）"
        for l in LABELS if l in counts.index))

    ann, tot = regime_breakdown(curves, reg_win, df.loc[eval_start:].index)
    print("\n" + "═" * 92)
    print("  ③ 分阶段年化%（同状态的日子拼起来折年，跨阶段可比）")
    print("═" * 92)
    print(ann.to_string(float_format=lambda v: f"{v:10.2f}"))

    print("\n" + "═" * 92)
    print("  ④ 分阶段累计%（该状态所有日子的收益连乘，未折年）")
    print("═" * 92)
    print(tot.to_string(float_format=lambda v: f"{v:10.2f}"))

    print("\n  ⚠ ③④ 是**归因**不是可交易策略：同一状态的日子在时间上并不连续，"
          "\n    连乘等于假设能在状态切换瞬间无成本进出。真按状态切换的结果见"
          "「自适应」那一行（带成本、带滞回）。")

    # ⑤ 逐段明细：③ 把同状态合并了，这里按 ② 划出的每一段单独出数
    epi = (episode_breakdown(curves, eps, df.loc[eval_start:].index)
           if not eps.empty else pd.DataFrame())
    if not epi.empty:
        print("\n" + "═" * 92)
        print("  ⑤ 逐阶段区间收益%（② 的每一段单独算；段内时间连续，是真实可实现的收益）")
        print("═" * 92)
        ref = eps.set_index("标签")["区间涨跌%"]
        epi_show = epi.copy()
        epi_show.insert(0, "标的涨跌%", ref)
        print_wide(epi_show, chunk=5)
        print("\n  第一列是标的自身该段的涨跌，其余列为各打法在该段的区间收益。"
              "\n  空白 = 该段净值缺失（引擎那两行要等均线预热完才有净值）。")

    st = regime_stats(df, reg_df, horizon=60)
    if not st.empty:
        f = st.copy()
        for col in ["未来均值", "未来中位", "胜率", "最差", "最好"]:
            f[col] = f[col].map("{:.1%}".format)
        print("\n  状态分类器有没有信息量（各状态之后 60 日的收益分布）：")
        print("  " + f.to_string().replace("\n", "\n  "))
        print("  三档若长得一样，说明标签是噪声，按它切换只会白付摩擦成本。")

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f"{args.code}_playbook.csv")
    ann_path = os.path.join(args.output, f"{args.code}_by_regime.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    by_reg = ann.add_suffix("·年化%").join(tot.add_suffix("·累计%"))
    by_reg.index.name = "打法"
    by_reg.to_csv(ann_path, encoding="utf-8-sig")
    print(f"\n  → {path}\n  → {ann_path}")

    if not epi.empty:
        epi_path = os.path.join(args.output, f"{args.code}_by_episode.csv")
        # CSV 里把阶段元信息补齐：只有标签的话，事后没法按天数/涨跌重新排序筛选
        meta = eps.set_index("标签")[["regime", "start", "end", "days", "区间涨跌%"]]
        meta.columns = ["阶段", "起", "止", "交易日", "标的涨跌%"]
        meta.join(epi).to_csv(epi_path, encoding="utf-8-sig", index_label="阶段标签")
        print(f"  → {epi_path}")

    # 可复现清单（评审项 2）
    write_run_manifest(args.output, symbols=[args.code], started_at=t0)


if __name__ == "__main__":
    main()
