"""
分时择时 3 面板图：价格 + GO 窗口 | MACD | 成交量。

从 `backtest_jcy_intraday`（原 jcy_intraday_timing）拆出。报告层判据：
**一切 import matplotlib 的模块都在 `backtest/reports/`**；计算与数据在
`lib/`（分时取数在 `lib/intraday_store`，MACD/GO/计价在 `lib/execution`）。

图只做展示，不含任何决策逻辑；`TimingSummary` 来自 `lib/execution`。
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from backtest.lib.execution import TimingSummary
from backtest.reports.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED, setup_matplotlib,
    style_ax,
)

setup_matplotlib()

# 分时图表显示执行日前多少天的上下文
CHART_CONTEXT_DAYS = 28


def _highlight_exec_day(ax, day_bars: pd.DataFrame, color: str, period: int):
    """在指定子图上为执行日添加半透明背景高亮。"""
    if not day_bars.empty:
        ax.axvspan(
            day_bars.index[0] - timedelta(minutes=period // 2),
            day_bars.index[-1] + timedelta(minutes=period // 2),
            alpha=0.10, color=color, lw=0,
        )


def _plot_price_panel(ax, plot_df: pd.DataFrame, day_bars: pd.DataFrame,
                      exec_date: pd.Timestamp, signal_date: pd.Timestamp,
                      summary: TimingSummary, action: str,
                      color_dir: str, period: int):
    """子图1：分时收盘价 + GO 窗口标注 + 信号日/执行日高亮。"""
    ax.plot(plot_df.index, plot_df["close"], color=C_BLUE, lw=1.0)
    _highlight_exec_day(ax, day_bars, color_dir, period)

    # 信号日金色背景（若与执行日不同）
    if signal_date != exec_date:
        sig_bars = plot_df[plot_df.index.normalize() == signal_date]
        if not sig_bars.empty:
            ax.axvspan(
                sig_bars.index[0] - timedelta(minutes=period // 2),
                sig_bars.index[-1] + timedelta(minutes=period // 2),
                alpha=0.07, color=C_GOLD, lw=0,
            )
            ax.axvline(x=sig_bars.index[0], color=C_GOLD, lw=1.2,
                        linestyle="--", alpha=0.7,
                        label=f"日线信号日 {signal_date.strftime('%Y-%m-%d')}")

    # GO 窗口竖线标注
    price_max = plot_df["close"].max()
    price_min = plot_df["close"].min()
    label_y   = price_max + (price_max - price_min) * 0.012
    for t in summary.go_times:
        ax.axvline(x=t, color=color_dir, lw=1.5, alpha=0.8, linestyle="--")
        ax.text(t, label_y, t.strftime("%H:%M"),
                color=color_dir, fontsize=8, rotation=90,
                va="bottom", ha="center", weight="bold")

    # 标题
    action_cn = "买入" if action == "buy" else "卖出"
    title_status = (f"✅ {summary.go_count} 个 GO 窗口，首选 "
                    f"{summary.first_go.strftime('%H:%M')}"
                    if summary.has_go else "⚠️ 无明确 GO 窗口，建议观望")
    ax.set_title(
        f"分时择时（{period}min）  |  "
        f"{action_cn}信号 {signal_date.strftime('%Y-%m-%d')}  |  "
        f"执行日 {exec_date.strftime('%Y-%m-%d')}  |  {title_status}",
        color=C_FG, fontsize=11, pad=8,
    )
    ax.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    style_ax(ax)


def _plot_macd_panel(ax, plot_df: pd.DataFrame, day_bars: pd.DataFrame,
                     summary: TimingSummary, action: str,
                     color_dir: str, period: int):
    """子图2：分时 MACD 柱 + DIF/DEA + GO 标记。"""
    _highlight_exec_day(ax, day_bars, color_dir, period)

    bar_width = timedelta(minutes=period - 2)
    bar_colors = np.where(plot_df["MACD"].values >= 0, C_RED, C_GREEN)
    expanding  = plot_df["hist_expanding"].values

    ax.bar(plot_df.index[~expanding], plot_df["MACD"].values[~expanding],
           color=bar_colors[~expanding], alpha=0.4, width=bar_width)
    ax.bar(plot_df.index[expanding], plot_df["MACD"].values[expanding],
           color=bar_colors[expanding], alpha=0.9, width=bar_width)

    ax.plot(plot_df.index, plot_df["DIF"], color=C_BLUE, lw=1.0, label="DIF")
    ax.plot(plot_df.index, plot_df["DEA"], color=C_GOLD, lw=1.0, label="DEA")
    ax.axhline(0, color=C_MUTED, lw=0.5, linestyle="--")

    # GO 窗口的 DIF 点位标注
    marker = "^" if action == "buy" else "v"
    for t in summary.go_times:
        if t in plot_df.index:
            y = plot_df.loc[t, "DIF"]
            ax.scatter([t], [y], marker=marker, color=color_dir,
                       s=100, zorder=8, edgecolors="white", linewidths=0.5)

    ax.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    ax.set_ylabel(f"分时 MACD ({period}min)", color=C_FG, fontsize=9)
    style_ax(ax)


def _plot_volume_panel(ax, plot_df: pd.DataFrame, day_bars: pd.DataFrame,
                       exec_mask, color_dir: str, period: int):
    """子图3：成交量（执行日按方向上色，其余灰色）。"""
    bar_width = timedelta(minutes=period - 2)
    non_exec = plot_df.index[~exec_mask]
    ax.bar(non_exec, plot_df.loc[non_exec, "volume"].values,
           color=C_MUTED, alpha=0.5, width=bar_width)
    if not day_bars.empty:
        ax.bar(day_bars.index, day_bars["volume"].values,
               color=color_dir, alpha=0.75, width=bar_width)

    ax.set_ylabel("成交量", color=C_FG, fontsize=9)
    style_ax(ax)


def plot_intraday_chart(
    intraday_df: pd.DataFrame,
    exec_date: pd.Timestamp,
    symbol: str,
    name: str,
    action: str,
    signal_date: pd.Timestamp,
    summary: TimingSummary,
    period: int,
    save_path: str | None = None,
):
    """
    3 面板分时图：价格 + GO 窗口 | MACD | 成交量。
    只显示执行日前 CHART_CONTEXT_DAYS 天的数据。
    """
    context_start = exec_date - timedelta(days=CHART_CONTEXT_DAYS)
    plot_df   = intraday_df[intraday_df.index.normalize() >= context_start].copy()
    exec_mask = plot_df.index.normalize() == exec_date
    day_bars  = plot_df[exec_mask]
    color_dir = C_GREEN if action == "buy" else C_RED

    fig = plt.figure(figsize=(16, 10), facecolor=C_BG)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.06, height_ratios=[3, 2, 1])

    ax1 = fig.add_subplot(gs[0], facecolor=C_BG)
    _plot_price_panel(ax1, plot_df, day_bars, exec_date, signal_date,
                      summary, action, color_dir, period)

    ax2 = fig.add_subplot(gs[1], sharex=ax1, facecolor=C_BG)
    _plot_macd_panel(ax2, plot_df, day_bars, summary, action, color_dir, period)

    ax3 = fig.add_subplot(gs[2], sharex=ax1, facecolor=C_BG)
    _plot_volume_panel(ax3, plot_df, day_bars, exec_mask, color_dir, period)

    # X 轴格式
    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right",
             color=C_FG, fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"    图表已保存至：{save_path}")
    else:
        plt.show()
    plt.close(fig)
