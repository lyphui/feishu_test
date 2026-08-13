"""
卢麒元 MACD 三级确认策略的报告层：6 面板图 + 每日状态 CSV。

从 `backtest_lu_macd`（原 lu_macd_analysis，471 行 CLI）拆出——绘图与 CSV
导出与 `reports/report.py` / `reports/bull_report.py` 同构，收进报告层后
脚本只留编排。报告层判据：**一切 import matplotlib 的模块都在
`backtest/reports/`**。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from backtest.engine import fmt_sharpe
from backtest.reports.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED,
    setup_matplotlib, style_ax,
)

setup_matplotlib()


def _draw_macd_panel(ax, df, dif_col, dea_col, macd_col, bar_width,
                     label_prefix, trades=None, signal_col="signal"):
    """
    在单个 Axes 上绘制 MACD 柱 + DIF + DEA + 买卖标记。

    买卖标记来源（双保险）：
      1. df[signal_col]：策略信号（买 =1 / 卖 =-1）
      2. trades：实际成交记录的垂直线，确保买入点可见
    """
    bar_colors = np.where(df[macd_col] >= 0, C_GREEN, C_RED)
    ax.bar(df.index, df[macd_col], color=bar_colors, alpha=0.55,
           width=bar_width, label=f"MACD柱({label_prefix})")
    ax.plot(df.index, df[dif_col], color=C_BLUE, lw=1.2, label=f"DIF({label_prefix})")
    ax.plot(df.index, df[dea_col], color=C_GOLD, lw=1.2, label=f"DEA({label_prefix})")
    ax.axhline(0, color=C_MUTED, lw=0.6, linestyle="--")

    # ── 策略信号标记（df["signal"]）────────────────────────────────────────────
    buy_idx  = df.index[df[signal_col] == 1]
    sell_idx = df.index[df[signal_col] == -1]
    if len(buy_idx):
        ax.scatter(buy_idx, df.loc[buy_idx, dif_col],
                   marker="^", color=C_GREEN, s=80, zorder=7, label="买入信号")
    if len(sell_idx):
        ax.scatter(sell_idx, df.loc[sell_idx, dif_col],
                   marker="v", color=C_RED, s=60, zorder=6, label="卖出信号")

    # ── 实际成交垂直线（保证买入点肉眼可见）──────────────────────────────────
    if trades is not None and not trades.empty:
        sell_actions = {"卖出", "止损卖出", "止盈卖出", "期末清仓"}
        for _, t in trades.iterrows():
            is_buy = t["action"] == "买入"
            c = C_GREEN if is_buy else C_RED
            lw = 1.2 if is_buy else 0.8
            ax.axvline(x=t["date"], color=c, lw=lw, alpha=0.55,
                       linestyle="--" if is_buy else ":")
            # 在 DIF 曲线的实际成交位置再画一个大号标记
            if t["date"] in df.index:
                y = df.loc[t["date"], dif_col]
                marker = "^" if is_buy else "v"
                size   = 120 if is_buy else 90
                label  = "买入执行" if is_buy else None
                ax.scatter([t["date"]], [y], marker=marker, color=c,
                           s=size, zorder=8, edgecolors="white", linewidths=0.5,
                           label=label)

    ax.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED,
              fontsize=7, ncol=4, loc="upper left")
    ax.set_ylabel(label_prefix, color=C_FG, fontsize=9)
    style_ax(ax)


def plot_lu_backtest(result: dict, save_path: "str | None" = None):
    """
    专属绘图：6 个子图
      1. 日线价格 + 买卖执行点
      2. 月线 MACD（DIF_M / DEA_M / MACD_M）
      3. 周线 MACD（DIF_W / DEA_W / MACD_W）
      4. 日线 MACD（DIF / DEA / MACD）
      5. 资产曲线 vs 基准
      6. 回撤
    """
    df     = result["df"]
    eq_df  = result["equity_curve"]
    trades = result["trades"]
    symbol = result["symbol"]

    fig = plt.figure(figsize=(18, 20), facecolor=C_BG)
    gs  = GridSpec(6, 1, figure=fig, hspace=0.06,
                   height_ratios=[3, 1.8, 1.8, 1.8, 1.5, 1])

    # ── 子图1：日线价格 + 买卖执行点 ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["close"], color=C_BLUE, lw=1.2, label="收盘价")

    if not trades.empty:
        buys  = trades[trades["action"] == "买入"]
        sells = trades[trades["action"].isin(["卖出", "止损卖出", "止盈卖出", "期末清仓"])]
        if not buys.empty:
            ax1.scatter(buys["date"], buys["price"],
                        marker="^", color=C_GREEN, s=90, zorder=5, label="买入执行")
        if not sells.empty:
            ax1.scatter(sells["date"], sells["price"],
                        marker="v", color=C_RED, s=90, zorder=5, label="卖出执行")

        # 交易日期垂直线 + 顶部日期标注
        price_max = df["close"].max()
        price_min = df["close"].min()
        label_y   = price_max + (price_max - price_min) * 0.012
        for _, t in trades.iterrows():
            color = C_GREEN if t["action"] == "买入" else C_RED
            ax1.axvline(x=t["date"], color=color, lw=0.7, alpha=0.4, linestyle=":")
            ax1.text(t["date"], label_y, t["date"].strftime("%m-%d"),
                     color=color, fontsize=6, rotation=90, va="bottom", ha="center")

    ax1.set_title(
        f"卢麒元三级 MACD  |  {symbol}  |  "
        f"总收益 {result['total_return']:+.2f}%  "
        f"基准 {result['benchmark_return']:+.2f}%  "
        f"夏普 {fmt_sharpe(result['sharpe_ratio'])}",
        color=C_FG, fontsize=12, pad=8,
    )
    ax1.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=9)
    style_ax(ax1)

    # ── 子图2：月线 MACD ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _draw_macd_panel(ax2, df, "DIF_M", "DEA_M", "MACD_M",
                     bar_width=20, label_prefix="月线", trades=trades)

    # ── 子图3：周线 MACD ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _draw_macd_panel(ax3, df, "DIF_W", "DEA_W", "MACD_W",
                     bar_width=5, label_prefix="周线", trades=trades)

    # ── 子图4：日线 MACD ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    _draw_macd_panel(ax4, df, "DIF", "DEA", "MACD",
                     bar_width=1, label_prefix="日线", trades=trades)

    # ── 子图5：资产曲线 vs 基准 ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    norm_eq    = eq_df["equity"] / result["equity_base"] * 100
    # 基准以统计窗口首日**开盘价**为基数，与策略的建仓口径一致
    norm_bench = eq_df["close"]  / result["benchmark_base"]  * 100
    ax5.plot(eq_df.index, norm_eq,    color=C_GREEN, lw=1.5, label="策略净值")
    ax5.plot(eq_df.index, norm_bench, color=C_MUTED, lw=1,
             linestyle="--", label="基准(买入持有)")
    ax5.axhline(100, color=C_MUTED, lw=0.5, linestyle=":")
    ax5.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    ax5.set_ylabel("净值（基准=100）", color=C_FG, fontsize=9)
    style_ax(ax5)

    # ── 子图6：回撤 ──────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.fill_between(eq_df.index, eq_df["drawdown"] * 100, 0,
                     color=C_RED, alpha=0.4, label="策略回撤")
    ax6.set_ylabel("回撤 (%)", color=C_FG, fontsize=9)
    ax6.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    style_ax(ax6)

    # ── X 轴：只显示最后一张 ─────────────────────────────────────────────────
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax6.get_xticklabels(), rotation=30, ha="right", color=C_FG, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"\n  图表已保存至：{save_path}")
    else:
        plt.show()

    return fig


def export_daily_status(result: dict, save_path: str) -> None:
    """
    将每日指标状态导出为 CSV，方便排查三级确认条件为何未触发。

    输出列说明
    ----------
    月线_DIF/DEA        月线 MACD 指标值（前向填充到日线）
    月线_0轴下          DIF_M < 0 且 DEA_M < 0
    月线_DIF上穿DEA     DIF_M > DEA_M（当前位置关系，非金叉事件）
    月线_绿柱           MACD_M < 0
    月线_已确认         monthly_confirmed（一旦金叉触发后持续为 True）
    周线_DIF/DEA        周线 MACD 指标值
    周线_0轴下          DIF_W < 0 且 DEA_W < 0
    周线_DIF上穿DEA     DIF_W > DEA_W
    周线_量能放大       vol_expanding
    周线_已确认         weekly_confirmed
    日线_DIF/DEA        日线 MACD 指标值
    日线_DIF上穿DEA     DIF > DEA
    价格阶段            phase（相对底部涨幅分级）
    信号                1=买入 / -1=卖出 / 0=观望
    未达条件            当日阻断买入信号的原因描述
    """
    df = result["df"].copy()

    rows = []
    for date, r in df.iterrows():
        m_dif   = r.get("DIF_M", float("nan"))
        m_dea   = r.get("DEA_M", float("nan"))
        m_macd  = r.get("MACD_M", float("nan"))
        w_dif   = r.get("DIF_W", float("nan"))
        w_dea   = r.get("DEA_W", float("nan"))
        d_dif   = r.get("DIF",  float("nan"))
        d_dea   = r.get("DEA",  float("nan"))
        vol_exp = bool(r.get("vol_expanding", False))
        m_conf  = bool(r.get("monthly_confirmed", False))
        w_conf  = bool(r.get("weekly_confirmed",  False))
        phase   = r.get("phase", "—")
        signal  = int(r.get("signal", 0))

        # ── 各级子条件 ──
        m_below0  = (m_dif < 0) and (m_dea < 0)
        m_dif_up  = m_dif > m_dea
        m_green   = m_macd < 0
        w_below0  = (w_dif < 0) and (w_dea < 0)
        w_dif_up  = w_dif > w_dea
        d_dif_up  = d_dif > d_dea

        # ── 未达条件描述 ──
        if signal == 1:
            blocking = "✅ 买入信号触发"
        elif signal == -1:
            blocking = "⚡ 日线死叉，卖出信号"
        elif m_conf and w_conf:
            blocking = "三级已全部确认，持仓中 / 等待下一周线金叉"
        elif not m_conf:
            reasons = []
            if not m_below0:
                reasons.append(f"月线未在0轴下(DIF_M={m_dif:.3f},DEA_M={m_dea:.3f})")
            if not m_dif_up:
                reasons.append(f"月线DIF未上穿DEA(DIF_M={m_dif:.3f}<DEA_M={m_dea:.3f})")
            if not m_green:
                reasons.append(f"月线柱非绿柱(MACD_M={m_macd:.3f})")
            if not reasons:
                reasons.append("月线本月尚未形成金叉事件")
            blocking = "❌ L1未达: " + " / ".join(reasons)
        else:  # m_conf=True, w_conf=False
            reasons = []
            if not w_below0:
                reasons.append(f"周线未在0轴下(DIF_W={w_dif:.3f},DEA_W={w_dea:.3f})")
            if not w_dif_up:
                reasons.append(f"周线DIF未上穿DEA(DIF_W={w_dif:.3f}<DEA_W={w_dea:.3f})")
            if not vol_exp:
                reasons.append("周线量能未放大")
            if not reasons:
                reasons.append("本周尚未形成满足条件的周线金叉事件")
            blocking = "⚠️ L2未达: " + " / ".join(reasons)

        rows.append({
            "日期":        date.strftime("%Y-%m-%d"),
            "收盘价":      round(r["close"], 2),
            "月线_DIF":    round(m_dif,  4),
            "月线_DEA":    round(m_dea,  4),
            "月线_0轴下":  m_below0,
            "月线_DIF上穿DEA": m_dif_up,
            "月线_绿柱":   m_green,
            "月线_已确认": m_conf,
            "周线_DIF":    round(w_dif, 4),
            "周线_DEA":    round(w_dea, 4),
            "周线_0轴下":  w_below0,
            "周线_DIF上穿DEA": w_dif_up,
            "周线_量能放大":   vol_exp,
            "周线_已确认": w_conf,
            "日线_DIF":    round(d_dif, 4),
            "日线_DEA":    round(d_dea, 4),
            "日线_DIF上穿DEA": d_dif_up,
            "价格阶段":    phase,
            "信号":        signal,
            "未达条件":    blocking,
        })

    status_df = pd.DataFrame(rows)
    status_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  每日状态表已保存至：{save_path}")
