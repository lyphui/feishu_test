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
而不是一个固定资金池的真实组合收益。
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
    "策略收益%", "基准收益%", "超额收益%",
    "最大回撤%", "夏普", "交易次数", "胜率%", "盈亏比", "受阻次数",
]


# ── 单股结果 → 一行 ───────────────────────────────────────────────────────────

def result_to_row(candidate: dict, result: dict) -> dict:
    blocked = result.get("blocked_trades")
    sharpe = result["sharpe_ratio"]
    return {
        "代码":         candidate["code"],
        "名称":         candidate["name"],
        "推荐日":       candidate["date"],
        "统计交易日数": len(result["equity_curve"]),
        "策略收益%":    round(result["total_return"], 2),
        "基准收益%":    round(result["benchmark_return"], 2),
        "超额收益%":    round(result["total_return"] - result["benchmark_return"], 2),
        "最大回撤%":    round(result["max_drawdown"], 2),
        "夏普":         None if sharpe is None else round(sharpe, 2),
        "交易次数":     result["total_trades"],
        "胜率%":        round(result["win_rate"], 1),
        "盈亏比":       round(result["profit_factor"], 2),
        "受阻次数":     0 if blocked is None or blocked.empty else len(blocked),
    }


def normalized_equity(result: dict) -> pd.Series:
    """统计窗口内的净值曲线，起点归一为 1.0。"""
    eq = result["equity_curve"]["equity"]
    return eq / result["initial_capital"]


# ── 汇总表 ────────────────────────────────────────────────────────────────────

def build_summary(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return df.sort_values("超额收益%", ascending=False).reset_index(drop=True)


def print_summary_table(df: pd.DataFrame, top_n: int = 5) -> None:
    if df.empty:
        print("\n  无可汇总的回测结果")
        return

    n = len(df)
    beat = (df["超额收益%"] > 0).sum()
    positive = (df["策略收益%"] > 0).sum()

    print("\n" + "═" * 78)
    print(f"  横截面汇总（{n} 只）")
    print("═" * 78)
    print(f"  跑赢买入持有 : {beat}/{n}  ({beat / n:.1%})")
    print(f"  绝对正收益   : {positive}/{n}  ({positive / n:.1%})")
    print(f"  策略收益     : 均值 {df['策略收益%'].mean():+.2f}%   "
          f"中位数 {df['策略收益%'].median():+.2f}%")
    print(f"  超额收益     : 均值 {df['超额收益%'].mean():+.2f}%   "
          f"中位数 {df['超额收益%'].median():+.2f}%")
    print(f"  最大回撤     : 均值 {df['最大回撤%'].mean():.2f}%   "
          f"最深 {df['最大回撤%'].min():.2f}%")
    if df["夏普"].notna().any():
        print(f"  夏普         : 中位数 {df['夏普'].median():.2f}")
    print(f"  平均交易次数 : {df['交易次数'].mean():.1f}")
    blocked_total = df["受阻次数"].sum()
    if blocked_total:
        print(f"  涨跌停受阻   : 合计 {blocked_total} 次（这些信号在实盘中根本下不进去）")

    # 中位数才是这类右偏分布的重心：均值容易被个别翻倍股拉高
    if df["超额收益%"].mean() > 0 > df["超额收益%"].median():
        print("\n  ⚠️ 均值为正但中位数为负：整体收益集中在少数几只上，"
              "多数标的其实跑输基准。")

    def _block(title, sub):
        print(f"\n  {title}")
        print(f"    {'代码':<8}{'名称':<10}{'策略%':>9}{'基准%':>9}{'超额%':>9}{'回撤%':>9}")
        for _, r in sub.iterrows():
            print(f"    {r['代码']:<8}{str(r['名称']):<10}"
                  f"{r['策略收益%']:>9.2f}{r['基准收益%']:>9.2f}"
                  f"{r['超额收益%']:>9.2f}{r['最大回撤%']:>9.2f}")

    _block(f"超额收益 Top {top_n}", df.head(top_n))
    _block(f"超额收益 Bottom {top_n}", df.tail(top_n).iloc[::-1])
    print("═" * 78)


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
    ax1.plot(port.index, port * 100, color=C_GREEN, lw=1.8, label="等权组合净值")

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
        f"JCY 增持组合 —— 等权净值曲线  |  期末 {final:.1f}（起点=100）  "
        f"|  {len(portfolio)} 个交易日  |  最多 {int(portfolio['n_active'].max())} 只在场",
        color=C_FG, fontsize=12, pad=10,
    )
    ax1.set_ylabel("净值（起点=100）", color=C_FG, fontsize=9)
    ax1.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=9)
    style_ax(ax1)

    ax2.fill_between(portfolio.index, portfolio["n_active"], 0,
                     color=C_RED, alpha=0.35, step="post")
    ax2.set_ylabel("在场只数", color=C_FG, fontsize=9)
    style_ax(ax2)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right",
             color=C_FG, fontsize=8)

    plt.tight_layout()
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
