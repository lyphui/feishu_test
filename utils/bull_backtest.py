"""
卢麒元 MACD 牛市动能截取策略 —— 共享组件
========================================
- BullStrategyAdapter: 将双参数 prepare(df, index_df) 适配为单参数接口
- export_bull_daily_status: 每日指标状态 CSV 导出
- plot_bull_backtest: 5 面板回测图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from strategies import LuMACDBullStrategy
from strategies.base import BaseStrategy
from utils.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED, style_ax,
)


# ── 策略适配器 ────────────────────────────────────────────────────────────────

class BullStrategyAdapter(BaseStrategy):
    """
    将 LuMACDBullStrategy.prepare(df, index_df) 适配为 run_backtest 所需的
    单参数 prepare(df) 接口。

    trade_start_date : str "YYYYMMDD" or None
        若指定，该日期之前的买入 / 卖出信号全部清零（用于 MACD 预热期）。
    """

    def __init__(self, inner: LuMACDBullStrategy, index_df,
                 trade_start_date: str | None = None):
        self._inner            = inner
        self._index_df         = index_df
        self._trade_start_date = trade_start_date

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def params(self) -> dict:
        return self._inner.params

    def prepare(self, df):
        result = self._inner.prepare(df, self._index_df)
        if self._trade_start_date:
            cutoff = pd.to_datetime(self._trade_start_date, format="%Y%m%d")
            result.loc[result.index < cutoff, "signal"] = 0

        # 第一次操作必须是买入：清除首次买入信号之前的所有卖出信号
        buy_indices = result.index[result["signal"] == 1]
        if len(buy_indices) > 0:
            first_buy = buy_indices[0]
            result.loc[(result.index < first_buy) & (result["signal"] == -1), "signal"] = 0
        else:
            result.loc[result["signal"] == -1, "signal"] = 0

        return result

    def plot_indicators(self, ax, df, colors):
        return self._inner.plot_indicators(ax, df, colors)


# ── 每日状态导出 ──────────────────────────────────────────────────────────────

def export_bull_daily_status(result: dict, save_path: str) -> None:
    """将每日指标状态导出为 CSV，方便排查牛市动能截取策略的买卖判断依据。"""
    df = result["df"].copy()
    rows = []
    for date, r in df.iterrows():
        dif_idx   = r.get("DIF_IDX", float("nan"))
        dea_idx   = r.get("DEA_IDX", float("nan"))
        bull      = bool(r.get("bull_market", False))
        dif       = r.get("DIF",  float("nan"))
        dea       = r.get("DEA",  float("nan"))
        macd      = r.get("MACD", float("nan"))
        expanding = bool(r.get("hist_expanding", False))
        shrinking = bool(r.get("hist_shrinking", False))
        signal    = int(r.get("signal", 0))

        idx_above0    = dif_idx > 0
        idx_dif_up    = dif_idx > dea_idx
        dif_above_dea = dif > dea

        if signal == 1:
            blocking = "✅ 买入信号触发（金叉 + 牛市 + 红柱拉长）"
        elif signal == -1:
            if not bull:
                blocking = "⚡ 卖出：熊市保护强制清仓"
            elif shrinking:
                blocking = "⚡ 卖出：红柱开始缩短（动能衰减）"
            else:
                blocking = "⚡ 卖出：死叉（DIF 下穿 DEA）"
        else:
            reasons = []
            if not bull:
                sub = []
                if not idx_above0:
                    sub.append(f"大盘DIF_M={dif_idx:.3f}<0")
                if not idx_dif_up:
                    sub.append(f"大盘DIF_M({dif_idx:.3f})<DEA_M({dea_idx:.3f})")
                reasons.append("❌ 牛市未确认: " + " / ".join(sub) if sub else "❌ 牛市未确认")
            else:
                if not dif_above_dea:
                    reasons.append(f"⚠️ 个股DIF({dif:.4f})<DEA({dea:.4f})，未金叉")
                if not expanding:
                    if macd <= 0:
                        reasons.append(f"⚠️ MACD柱({macd:.4f})≤0，红柱未出现")
                    else:
                        reasons.append(f"⚠️ 红柱未拉长（MACD={macd:.4f}，等待加速）")
                if not reasons:
                    reasons.append("持仓中 / 等待下一金叉机会")
            blocking = " | ".join(reasons)

        rows.append({
            "日期":         date.strftime("%Y-%m-%d"),
            "收盘价":       round(r["close"], 2),
            "大盘_DIF":     round(dif_idx, 4),
            "大盘_DEA":     round(dea_idx, 4),
            "大盘_DIF>0":   idx_above0,
            "大盘_DIF>DEA": idx_dif_up,
            "牛市确认":     bull,
            "个股_DIF":     round(dif,  4),
            "个股_DEA":     round(dea,  4),
            "个股_MACD柱":  round(macd, 4),
            "红柱拉长":     expanding,
            "红柱缩短":     shrinking,
            "信号":         signal,
            "判断依据":     blocking,
        })

    status_df = pd.DataFrame(rows)
    status_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  每日状态表已保存至：{save_path}")


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_bull_backtest(result: dict, save_path: "str | None" = None,
                       trade_start_date: str | None = None):
    """
    5 面板回测图表：价格、MACD、大盘月线、资产曲线、回撤。

    trade_start_date : str "YYYYMMDD"
        若指定，在图表上用金色竖线标注推荐日期。
    """
    df     = result["df"]
    eq_df  = result["equity_curve"]
    trades = result["trades"]
    symbol = result["symbol"]

    fig = plt.figure(figsize=(18, 18), facecolor=C_BG)
    gs  = GridSpec(5, 1, figure=fig, hspace=0.06,
                   height_ratios=[3, 2, 1.5, 1.5, 1])

    # ── 子图1：价格 + 买卖点 ──────────────────────────────────────────────────
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

        price_max = df["close"].max()
        price_min = df["close"].min()
        label_y   = price_max + (price_max - price_min) * 0.012
        for _, t in trades.iterrows():
            color = C_GREEN if t["action"] == "买入" else C_RED
            ax1.axvline(x=t["date"], color=color, lw=0.7, alpha=0.4, linestyle=":")
            ax1.text(t["date"], label_y, t["date"].strftime("%m-%d"),
                     color=color, fontsize=6, rotation=90, va="bottom", ha="center")

    if trade_start_date:
        tsd = pd.to_datetime(trade_start_date, format="%Y%m%d")
        ax1.axvline(x=tsd, color=C_GOLD, lw=1.4, linestyle="--", alpha=0.85,
                    label=f"推荐日 {tsd.strftime('%Y-%m-%d')}")

    ax1.set_title(
        f"卢麒元牛市动能截取策略  |  {symbol}  |  "
        f"总收益 {result['total_return']:+.2f}%  "
        f"基准 {result['benchmark_return']:+.2f}%  "
        f"夏普 {result['sharpe_ratio']:.2f}",
        color=C_FG, fontsize=12, pad=8,
    )
    ax1.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=9)
    style_ax(ax1)

    # ── 子图2：MACD + 牛市背景 ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if "bull_market" in df.columns:
        bull_on = False; bull_start = None
        for date, row in df.iterrows():
            if row["bull_market"] and not bull_on:
                bull_start = date; bull_on = True
            elif not row["bull_market"] and bull_on:
                ax2.axvspan(bull_start, date, alpha=0.07, color=C_GREEN, lw=0)
                bull_on = False
        if bull_on and bull_start is not None:
            ax2.axvspan(bull_start, df.index[-1], alpha=0.07, color=C_GREEN, lw=0)

    expanding_mask = df.get("hist_expanding", pd.Series(False, index=df.index))
    bar_colors = np.where(df["MACD"] >= 0, C_RED, C_GREEN)
    ax2.bar(df.index[~expanding_mask], df["MACD"][~expanding_mask],
            color=bar_colors[~expanding_mask.values], alpha=0.4, width=1, label="MACD柱")
    ax2.bar(df.index[expanding_mask], df["MACD"][expanding_mask],
            color=bar_colors[expanding_mask.values], alpha=0.9, width=1, label="MACD柱(动能↑)")
    ax2.plot(df.index, df["DIF"], color=C_BLUE, lw=1.2, label="DIF")
    ax2.plot(df.index, df["DEA"], color=C_GOLD, lw=1.2, label="DEA")
    ax2.axhline(0, color=C_MUTED, lw=0.6, linestyle="--")

    buy_idx  = df.index[df["signal"] == 1]
    sell_idx = df.index[df["signal"] == -1]
    if len(buy_idx):
        ax2.scatter(buy_idx, df.loc[buy_idx, "DIF"],
                    marker="^", color=C_GREEN, s=80, zorder=7, label="买入(动能起点)")
    if len(sell_idx):
        ax2.scatter(sell_idx, df.loc[sell_idx, "DIF"],
                    marker="v", color=C_RED, s=60, zorder=6, label="卖出(动能衰减)")

    if not trades.empty:
        for _, t in trades.iterrows():
            is_buy = t["action"] == "买入"
            c  = C_GREEN if is_buy else C_RED
            lw = 1.2 if is_buy else 0.8
            ax2.axvline(x=t["date"], color=c, lw=lw, alpha=0.55,
                        linestyle="--" if is_buy else ":")
            if t["date"] in df.index:
                y = df.loc[t["date"], "DIF"]
                ax2.scatter([t["date"]], [y],
                            marker="^" if is_buy else "v",
                            color=c, s=120 if is_buy else 90,
                            zorder=8, edgecolors="white", linewidths=0.5)

    ax2.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED,
               fontsize=7, ncol=4, loc="upper left")
    ax2.set_ylabel("日线 MACD", color=C_FG, fontsize=9)
    style_ax(ax2)

    # ── 子图3：大盘月线 DIF/DEA ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if "DIF_IDX" in df.columns and df["DIF_IDX"].notna().any():
        ax3.plot(df.index, df["DIF_IDX"], color=C_BLUE, lw=1.2, label="大盘月线 DIF")
        ax3.plot(df.index, df["DEA_IDX"], color=C_GOLD, lw=1.2, label="大盘月线 DEA")
        ax3.axhline(0, color=C_MUTED, lw=0.6, linestyle="--")
        ax3.fill_between(df.index,
                         df["DIF_IDX"], df["DEA_IDX"],
                         where=(df["DIF_IDX"] > df["DEA_IDX"]) & (df["DIF_IDX"] > 0),
                         interpolate=True, alpha=0.20, color=C_GREEN, label="牛市区间")
    ax3.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED,
               fontsize=7, ncol=3, loc="upper left")
    ax3.set_ylabel("大盘月线 MACD", color=C_FG, fontsize=9)
    style_ax(ax3)

    # ── 子图4：资产曲线 ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    norm_eq    = eq_df["equity"] / result["initial_capital"] * 100
    norm_bench = eq_df["close"]  / eq_df["close"].iloc[0]  * 100
    ax4.plot(eq_df.index, norm_eq,    color=C_GREEN, lw=1.5, label="策略净值")
    ax4.plot(eq_df.index, norm_bench, color=C_MUTED, lw=1,
             linestyle="--", label="基准(买入持有)")
    ax4.axhline(100, color=C_MUTED, lw=0.5, linestyle=":")
    ax4.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    ax4.set_ylabel("净值（基准=100）", color=C_FG, fontsize=9)
    style_ax(ax4)

    # ── 子图5：回撤 ──────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.fill_between(eq_df.index, eq_df["drawdown"] * 100, 0,
                     color=C_RED, alpha=0.4, label="策略回撤")
    ax5.set_ylabel("回撤 (%)", color=C_FG, fontsize=9)
    ax5.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    style_ax(ax5)

    for ax in [ax1, ax2, ax3, ax4]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax5.get_xticklabels(), rotation=30, ha="right", color=C_FG, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"  图表已保存至：{save_path}")
    else:
        plt.show()

    plt.close(fig)
