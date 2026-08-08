"""
批量回测汇总报告
================
把 N 只股票的独立回测结果聚合成可以直接下结论的东西：

  - summary.csv        每股一行的横截面表（收益/超额/回撤/夏普/胜率…）
  - 控制台汇总表        跑赢基准比例、超额收益的均值与中位数、最好/最差各 5 只
  - summary_portfolio.png
                       等权组合净值 vs 大盘指数

为什么要单独做组合曲线：逐股各看一张图无法回答"这批推荐整体到底赚不赚钱"。
单股回测赢在选择性偏差上——只要有一只翻倍就显得策略很行。等权组合把每只票
的权重摊平，才是这批推荐作为一个整体的真实表现。

组合口径说明
------------
每只股票的推荐日不同，无法凑出一条严格意义上的组合净值曲线。这里采用
"平均在场净值"：每只票以自己的统计起点归一为 1.0，按日历对齐后，对当天
**已经开始**的股票取算术平均。因此曲线早期样本少、噪声大，越往后越可信；
它衡量的是"随机挑一只被推荐的票、从推荐日起持有该策略"的平均结果，
而不是一个固定资金池的真实组合收益——每只票都按满仓 10 万独立回测，
N 只票同时满仓在现实中不可能，所以这条曲线**不是**可投资的组合净值。

为什么主排序用"日均超额"而不是"超额收益"
----------------------------------------
各股推荐日不同，统计窗口可能从 30 个交易日到 500 个交易日不等。把持有
两年的 +40% 和持有两个月的 +40% 放进同一列取均值/中位数，比较的是
"谁的窗口更长"而不是"谁的策略更好"。
日均超额 bp = 超额收益% × 100 ÷ 统计交易日数，对窗口长度线性归一，
跨标的可比。不用年化是因为 (1+r)^(252/n) 在 n 很小时会几何放大，
短窗口标的会被放大到失去意义（引擎的 annual_return 在 n<252 时直接返回
None 就是这个原因）。
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_MUTED, setup_matplotlib, style_ax,
)


SUMMARY_COLUMNS = [
    "代码", "名称", "推荐日", "统计交易日数",
    "策略收益%", "基准收益%", "超额收益%", "日均超额bp",
    "最大回撤%", "夏普", "交易次数", "胜率%", "盈亏比",
    "成本占比%", "受阻次数",
]

# 窗口长度分组：把"跑得久"和"跑得好"分开看
WINDOW_BUCKETS = [
    ("< 3 月",    0,   60),
    ("3-6 月",    60,  120),
    ("6-12 月",   120, 252),
    ("≥ 1 年",    252, 10**9),
]


# ── 单股结果 → 一行 ───────────────────────────────────────────────────────────

def result_to_row(candidate: dict, result: dict) -> dict:
    blocked = result.get("blocked_trades")
    sharpe = result["sharpe_ratio"]
    n_days = len(result["equity_curve"])
    excess = result["total_return"] - result["benchmark_return"]
    costs = result.get("costs") or {}
    return {
        "代码":         candidate["code"],
        "名称":         candidate["name"],
        "推荐日":       candidate["date"],
        "统计交易日数": n_days,
        "策略收益%":    round(result["total_return"], 2),
        "基准收益%":    round(result["benchmark_return"], 2),
        "超额收益%":    round(excess, 2),
        # 对窗口长度线性归一，跨标的唯一可比的收益口径
        "日均超额bp":   round(excess * 100 / n_days, 2) if n_days else None,
        "最大回撤%":    round(result["max_drawdown"], 2),
        "夏普":         None if sharpe is None else round(sharpe, 2),
        "交易次数":     result["total_trades"],
        "胜率%":        round(result["win_rate"], 1),
        "盈亏比":       round(result["profit_factor"], 2),
        "成本占比%":    (None if costs.get("cost_drag_pct") is None
                         else round(costs["cost_drag_pct"], 2)),
        "受阻次数":     0 if blocked is None or blocked.empty else len(blocked),
    }


def normalized_equity(result: dict) -> pd.Series:
    """
    统计窗口内的净值曲线，起点归一为 1.0。

    基数用 equity_base（窗口起点权益）而不是 initial_capital：预热期不交易
    时两者相等，但调用方若在窗口前就有成交，只有 equity_base 能让曲线真正
    从 1.0 起步。
    """
    eq = result["equity_curve"]["equity"]
    base = result.get("equity_base") or result["initial_capital"]
    return eq / base


# ── 汇总表 ────────────────────────────────────────────────────────────────────

def build_summary(rows: list[dict]) -> pd.DataFrame:
    """按日均超额降序排。用总超额排序会让窗口长的标的天然靠前（见模块 docstring）。"""
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return df.sort_values("日均超额bp", ascending=False).reset_index(drop=True)


def print_summary_table(df: pd.DataFrame, top_n: int = 5) -> None:
    if df.empty:
        print("\n  无可汇总的回测结果")
        return

    n = len(df)
    beat = (df["超额收益%"] > 0).sum()
    positive = (df["策略收益%"] > 0).sum()
    days = df["统计交易日数"]

    print("\n" + "═" * 78)
    print(f"  横截面汇总（{n} 只）")
    print("═" * 78)
    print(f"  跑赢买入持有 : {beat}/{n}  ({beat / n:.1%})")
    print(f"  绝对正收益   : {positive}/{n}  ({positive / n:.1%})")
    print(f"  统计窗口     : {int(days.min())} ~ {int(days.max())} 个交易日"
          f"（中位数 {int(days.median())}）")
    print(f"  日均超额     : 均值 {df['日均超额bp'].mean():+.2f}bp   "
          f"中位数 {df['日均超额bp'].median():+.2f}bp   ← 唯一跨窗口可比的口径")
    print(f"  策略收益     : 均值 {df['策略收益%'].mean():+.2f}%   "
          f"中位数 {df['策略收益%'].median():+.2f}%")
    print(f"  超额收益     : 均值 {df['超额收益%'].mean():+.2f}%   "
          f"中位数 {df['超额收益%'].median():+.2f}%")
    print(f"  最大回撤     : 均值 {df['最大回撤%'].mean():.2f}%   "
          f"最深 {df['最大回撤%'].min():.2f}%")
    if df["夏普"].notna().any():
        print(f"  夏普         : 中位数 {df['夏普'].median():.2f}")
    print(f"  平均交易次数 : {df['交易次数'].mean():.1f}")
    if df["成本占比%"].notna().any():
        print(f"  交易成本     : 中位 {df['成本占比%'].median():.2f}%   "
              f"最高 {df['成本占比%'].max():.2f}%（占各自起点资金）")
    blocked_total = df["受阻次数"].sum()
    if blocked_total:
        print(f"  涨跌停受阻   : 合计 {blocked_total} 次（这些信号在实盘中根本下不进去）")

    # 中位数才是这类右偏分布的重心：均值容易被个别翻倍股拉高
    if df["超额收益%"].mean() > 0 > df["超额收益%"].median():
        print("\n  ⚠️ 均值为正但中位数为负：整体收益集中在少数几只上，"
              "多数标的其实跑输基准。")
    if len(days) > 1 and days.min() > 0 and days.max() / days.min() >= 3:
        print(f"\n  ⚠️ 各股统计窗口相差 {days.max() / days.min():.1f} 倍，"
              "「超额收益%」列不可直接横向比较，请看日均超额与下方分组。")

    _print_window_buckets(df)

    def _block(title, sub):
        print(f"\n  {title}")
        print(f"    {'代码':<8}{'名称':<10}{'日数':>6}{'策略%':>9}{'基准%':>9}"
              f"{'超额%':>9}{'日均bp':>9}{'回撤%':>9}")
        for _, r in sub.iterrows():
            print(f"    {r['代码']:<8}{str(r['名称']):<10}"
                  f"{int(r['统计交易日数']):>6}"
                  f"{r['策略收益%']:>9.2f}{r['基准收益%']:>9.2f}"
                  f"{r['超额收益%']:>9.2f}{r['日均超额bp']:>9.2f}"
                  f"{r['最大回撤%']:>9.2f}")

    _block(f"日均超额 Top {top_n}", df.head(top_n))
    _block(f"日均超额 Bottom {top_n}", df.tail(top_n).iloc[::-1])
    print("═" * 78)


def _print_window_buckets(df: pd.DataFrame) -> None:
    """
    按统计窗口长度分组展示。

    只看整体中位数会掩盖一件事：策略可能只在短窗口（推荐后一两个月）有效，
    时间拉长就被磨平。分组后如果日均超额随窗口拉长单调衰减，说明的是
    "信号衰减"而不是"策略赚钱"。
    """
    rows = []
    for label, lo, hi in WINDOW_BUCKETS:
        sub = df[(df["统计交易日数"] >= lo) & (df["统计交易日数"] < hi)]
        if sub.empty:
            continue
        rows.append((label, len(sub),
                     sub["日均超额bp"].median(),
                     sub["超额收益%"].median(),
                     (sub["超额收益%"] > 0).mean()))
    if len(rows) < 2:
        return          # 只有一个分组时，分组展示没有信息量

    print(f"\n  按统计窗口长度分组")
    print(f"    {'窗口':<10}{'只数':>6}{'日均超额bp':>13}{'超额%中位':>12}{'跑赢比例':>10}")
    for label, cnt, dbp, exc, win in rows:
        print(f"    {label:<10}{cnt:>6}{dbp:>13.2f}{exc:>12.2f}{win:>10.1%}")


# ── 等权组合曲线 ──────────────────────────────────────────────────────────────

def build_portfolio_curve(curves: dict[str, pd.Series]) -> pd.DataFrame:
    """
    把各股归一净值按日历对齐，取当日"已在场"标的的算术平均。

    返回 DataFrame，含 portfolio（等权净值）与 n_active（当日在场只数）。
    """
    if not curves:
        return pd.DataFrame(columns=["portfolio", "n_active"])

    wide = pd.concat(curves, axis=1).sort_index()
    # 各股统计起点不同：起点之前保持 NaN（未在场），起点之后停牌的按前值延续
    wide = wide.ffill()
    portfolio = wide.mean(axis=1, skipna=True)
    n_active = wide.notna().sum(axis=1)
    return pd.DataFrame({"portfolio": portfolio, "n_active": n_active}).dropna()


def plot_portfolio(portfolio: pd.DataFrame,
                   index_df: pd.DataFrame | None,
                   save_path: str,
                   index_name: str = "沪深300") -> None:
    if portfolio.empty:
        print("  组合曲线为空，跳过绘图")
        return

    setup_matplotlib()      # 独立调用时也能正确渲染中文，不依赖入口脚本

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), facecolor=C_BG,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True,
    )

    port = portfolio["portfolio"]
    ax1.plot(port.index, port * 100, color=C_GREEN, lw=1.8,
             label="平均单股净值（等权，非可投资组合）")

    if index_df is not None and not index_df.empty:
        idx = index_df["close"].reindex(
            index_df.index.union(port.index)).ffill().reindex(port.index)
        if idx.notna().any():
            idx = idx / idx.dropna().iloc[0] * 100
            ax1.plot(idx.index, idx, color=C_BLUE, lw=1.2, linestyle="--",
                     label=f"{index_name}（同期）")

    ax1.axhline(100, color=C_MUTED, lw=0.6, linestyle=":")
    final = port.iloc[-1] * 100
    ax1.set_title(
        f"JCY 增持股 —— 平均单股净值  |  期末 {final:.1f}（起点=100）  "
        f"|  {len(portfolio)} 个交易日  |  最多 {int(portfolio['n_active'].max())} 只在场\n"
        f"口径：每只票各自满仓独立回测后按日历取算术平均，"
        f"不是一个固定资金池的真实组合收益",
        color=C_FG, fontsize=11, pad=10,
    )
    ax1.set_ylabel("平均净值（各自起点=100）", color=C_FG, fontsize=9)
    ax1.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=9)
    style_ax(ax1)

    ax2.fill_between(portfolio.index, portfolio["n_active"], 0,
                     color=C_RED, alpha=0.35, step="post")
    ax2.set_ylabel("在场只数", color=C_FG, fontsize=9)
    style_ax(ax2)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right",
             color=C_FG, fontsize=8)

    # 不调 tight_layout：它会覆盖上面显式设定的 hspace，而且与 colorbar/共享轴
    # 不兼容会告警。savefig 的 bbox_inches="tight" 已经负责裁掉多余白边。
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  组合净值图已保存至：{save_path}")


# ── 对外总入口 ────────────────────────────────────────────────────────────────

def write_batch_report(rows: list[dict],
                       curves: dict[str, pd.Series],
                       output_dir: str,
                       index_df: pd.DataFrame | None = None,
                       index_name: str = "沪深300") -> pd.DataFrame:
    """写出 summary.csv + 组合净值图，并在控制台打印汇总表。返回汇总 DataFrame。"""
    summary = build_summary(rows)
    if summary.empty:
        print("\n  没有成功的回测，跳过汇总")
        return summary

    csv_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  横截面汇总已保存至：{csv_path}")

    print_summary_table(summary)

    portfolio = build_portfolio_curve(curves)
    if not portfolio.empty:
        portfolio_csv = os.path.join(output_dir, "summary_portfolio.csv")
        portfolio.to_csv(portfolio_csv, encoding="utf-8-sig")
        print(f"  组合净值序列已保存至：{portfolio_csv}")
        plot_portfolio(portfolio, index_df,
                       os.path.join(output_dir, "summary_portfolio.png"),
                       index_name=index_name)

    return summary
