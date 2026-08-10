"""
单股回测的**图表层**：`run_backtest` 结果 → 标准 4 面板图。

为什么从 engine.py 拆出来
-------------------------
`engine.py` 是纯计算：给它行情和策略，返回一个 dict。绘图属于展示，二者的变更
理由完全不同（改成本口径 vs 改配色/面板），而把 `plot_backtest` 留在引擎里会让
**任何** `import engine` 都连带拖进 matplotlib——批量回测、参数扫描、pytest
这些根本不出图的场景全都要付这个代价，无 GUI 环境还得先操心 backend。

项目里 `bull_report.py` / `batch_report.py` 早就是这么分的，engine 只是没跟上。

    from engine import run_backtest
    from report import plot_backtest          # 只有真要出图才 import

`macd_analysis.py` 仍 re-export `plot_backtest`，历史导入路径不受影响。
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from engine import fmt_sharpe
from lib.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_MUTED, COLORS,
    setup_matplotlib, style_ax,
)

# 模块级调用：与拆分前 `import engine` 的副作用等价，保证中文标签正常渲染
setup_matplotlib()


def plot_backtest(result: dict, save_path: str = None):
    df       = result["df"]
    eq_df    = result["equity_curve"]
    trades   = result["trades"]
    symbol   = result["symbol"]
    strategy = result["strategy"]

    fig = plt.figure(figsize=(16, 12), facecolor=C_BG)
    gs  = GridSpec(4, 1, figure=fig, hspace=0.08,
                   height_ratios=[3, 1.5, 1.5, 1.5])

    ax_kwargs = dict(facecolor=C_BG)

    # ── 子图1：K线 + 买卖点 ──
    ax1 = fig.add_subplot(gs[0], **ax_kwargs)
    ax1.plot(df.index, df["close"], color=C_BLUE, lw=1.2, label="收盘价")

    if not trades.empty:
        buys  = trades[trades["action"] == "买入"]
        sells = trades[trades["action"].isin(["卖出", "止损卖出", "止盈卖出", "期末清仓"])]
        ax1.scatter(buys["date"],  buys["price"],  marker="^", color=C_GREEN,
                    s=80, zorder=5, label="买入")
        ax1.scatter(sells["date"], sells["price"], marker="v", color=C_RED,
                    s=80, zorder=5, label="卖出")

    ax1.set_title(f"A股策略回测 [{strategy.name}]  |  {symbol}  |  "
                  f"总收益 {result['total_return']:+.2f}%  "
                  f"基准 {result['benchmark_return']:+.2f}%  "
                  f"夏普 {fmt_sharpe(result['sharpe_ratio'])}",
                  color=C_FG, fontsize=12, pad=10)
    ax1.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=9)
    style_ax(ax1)

    # ── 子图2：策略指标（由策略对象自行绘制） ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1, **ax_kwargs)
    strategy.plot_indicators(ax2, df, COLORS)
    style_ax(ax2)

    # ── 子图3：资产曲线 vs 基准 ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1, **ax_kwargs)
    # 基准以统计窗口首日**开盘价**为基数，与策略的建仓口径一致
    norm_eq    = eq_df["equity"] / result["equity_base"] * 100
    norm_bench = eq_df["close"]  / result["benchmark_base"]  * 100
    ax3.plot(eq_df.index, norm_eq,    color=C_GREEN, lw=1.5, label="策略净值")
    ax3.plot(eq_df.index, norm_bench, color=C_MUTED, lw=1,   label="基准(买入持有)", linestyle="--")
    ax3.axhline(100, color=C_MUTED, lw=0.5, linestyle=":")
    ax3.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    ax3.set_ylabel("净值（基准=100）", color=C_FG, fontsize=9)
    style_ax(ax3)

    # ── 子图4：回撤 ──
    ax4 = fig.add_subplot(gs[3], sharex=ax1, **ax_kwargs)
    ax4.fill_between(eq_df.index, eq_df["drawdown"] * 100, 0,
                     color=C_RED, alpha=0.4, label="策略回撤")
    ax4.set_ylabel("回撤 (%)", color=C_FG, fontsize=9)
    ax4.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    style_ax(ax4)

    # ── 关键日期：每笔交易日期画垂直虚线并在价格图顶部标注日期 ──
    if not trades.empty:
        price_max = df["close"].max()
        price_min = df["close"].min()
        label_y   = price_max + (price_max - price_min) * 0.01
        for _, trade in trades.iterrows():
            t_date   = trade["date"]
            t_action = trade["action"]
            t_color  = C_GREEN if t_action == "买入" else C_RED
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=t_date, color=t_color, lw=0.7, alpha=0.45, linestyle=":")
            ax1.text(
                t_date, label_y,
                t_date.strftime("%Y-%m-%d"),
                color=t_color, fontsize=6, rotation=90,
                va="bottom", ha="center",
            )

    # 隐藏x轴刻度（除最后一张）
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), visible=False)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", color=C_FG, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"\n  图表已保存至：{save_path}")
    else:
        plt.show()

    return fig
