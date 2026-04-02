"""
卢麒元 MACD 三级确认策略回测入口
==================================
复用 macd_analysis.py 中的数据获取和回测引擎，
使用专属绘图函数将月线/周线/日线 MACD 分三个子图展示。

使用方法：
    python lu_macd_analysis.py

配置文件：config/lu_macd_config.ini
"""

import configparser
import os
import sys
from datetime import date as _date

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 复用数据获取和回测引擎
from macd_analysis import fetch_stock_data, run_backtest
from strategies import LuMACDStrategy
from utils.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED, COLORS,
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
        f"夏普 {result['sharpe_ratio']:.2f}",
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
    norm_eq    = eq_df["equity"] / result["initial_capital"] * 100
    norm_bench = eq_df["close"]  / eq_df["close"].iloc[0]  * 100
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


def _write_default_config(path: str) -> None:
    content = """\
[backtest]
# 股票代码（沪市6开头，深市0/3开头）
symbol     = 600519

# 股票名称（用于文件名，建议拼音或英文）
name       = maotai

# 回测区间（YYYYMMDD）
# 注意：月线+周线需要足够热身数据，start_date 建议比实际分析起点早 3 年以上
start_date = 20180101
# end_date 留空则默认使用当天日期
end_date   =

# 初始资金（元）
capital    = 100000

# 止损比例（如 0.10 表示 10%），留空则不设置
stop_loss  =

# 止盈比例（如 0.30 表示 30%），留空则不设置
take_profit =

# 图表和CSV保存目录（留空则弹窗显示，不保存CSV）
save_chart_dir = output/

# HTTP 代理（如 http://127.0.0.1:7890），留空则直连
proxy =

# ── LuMACD 策略专属参数 ──────────────────────────────────────────────────────

# 量能放大判断窗口（周线根数），与前 N 周均量比较
vol_window = 4

# True = 缺少 volume 数据时抛出异常；False = 降级运行（跳过量能验证）
require_volume = false
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


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

    import pandas as pd
    status_df = pd.DataFrame(rows)
    status_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  每日状态表已保存至：{save_path}")


def _enrich_trades(trades, df):
    """
    在 trades DataFrame 中追加「参考依据」列，
    说明每笔操作当日的三级确认状态、MACD 数值和价格阶段。
    """
    if trades.empty:
        return trades

    records = []
    for _, t in trades.iterrows():
        action = t["action"]
        date   = t["date"]

        # 从 df 取当日数据（日期可能不完全对齐，取最近）
        row = df.loc[date] if date in df.index else df.iloc[df.index.get_indexer([date], method="nearest")[0]]

        dif  = row.get("DIF",  float("nan"))
        dea  = row.get("DEA",  float("nan"))
        difw = row.get("DIF_W", float("nan"))
        deaw = row.get("DEA_W", float("nan"))
        difm = row.get("DIF_M", float("nan"))
        deam = row.get("DEA_M", float("nan"))
        monthly = row.get("monthly_confirmed", False)
        weekly  = row.get("weekly_confirmed",  False)
        phase   = row.get("phase", "—")

        if action == "买入":
            basis = (
                f"三级确认触发 | "
                f"月线确认={monthly} DIF_M={difm:.4f}>DEA_M={deam:.4f} | "
                f"周线确认={weekly} DIF_W={difw:.4f}>DEA_W={deaw:.4f} | "
                f"日线DIF={dif:.4f} DEA={dea:.4f} | "
                f"价格阶段={phase}"
            )
        elif action == "卖出":
            basis = (
                f"日线死叉 DIF({dif:.4f}) < DEA({dea:.4f}) | "
                f"价格阶段={phase}"
            )
        elif action == "止损卖出":
            pct = t.get("return_pct", float("nan"))
            basis = f"止损触发 收益={pct:.2f}% | 日线DIF={dif:.4f} DEA={dea:.4f} | 价格阶段={phase}"
        elif action == "止盈卖出":
            pct = t.get("return_pct", float("nan"))
            basis = f"止盈触发 收益={pct:.2f}% | 日线DIF={dif:.4f} DEA={dea:.4f} | 价格阶段={phase}"
        elif action == "期末清仓":
            basis = f"回测结束强制清仓 | 价格阶段={phase}"
        else:
            basis = "—"

        records.append(basis)

    result = trades.copy()
    result.insert(result.columns.get_loc("action") + 1, "参考依据", records)
    return result


def main():
    print("\n" + "─" * 55)
    print("  卢麒元 MACD 三级确认策略回测")
    print("  数据来源：akshare（前复权）")
    print("─" * 55)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "lu_macd_config.ini"
    )
    if not os.path.exists(config_path):
        print(f"  配置文件不存在，已生成默认配置：{config_path}")
        _write_default_config(config_path)

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    s = cfg["backtest"]

    symbol     = s.get("symbol", "600519").strip()
    name       = s.get("name", "stock").strip()
    start_date = s.get("start_date", "20180101").strip()
    end_date   = s.get("end_date", "").strip()
    if not end_date:
        end_date = _date.today().strftime("%Y%m%d")
        print(f"  end_date 未设置，默认使用今日：{end_date}")

    capital         = float(s.get("capital", "100000"))
    stop_loss_raw   = s.get("stop_loss", "").strip()
    stop_loss       = float(stop_loss_raw) if stop_loss_raw else None
    take_profit_raw = s.get("take_profit", "").strip()
    take_profit     = float(take_profit_raw) if take_profit_raw else None
    save_dir        = s.get("save_chart_dir", "").strip()
    proxy           = s.get("proxy", "").strip()

    # LuMACD 专属参数
    vol_window        = int(s.get("vol_window", "4"))
    require_volume    = s.get("require_volume",    "false").strip().lower() == "true"
    require_green_bar = s.get("require_green_bar", "true").strip().lower() == "true"

    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"  代理：{proxy}")

    file_stem = f"lu_macd_{name}_{symbol}_{end_date}"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_chart = os.path.join(save_dir, file_stem + ".png")
        save_csv   = os.path.join(save_dir, file_stem + ".csv")
    else:
        save_chart = None
        save_csv   = None

    print(f"  股票代码：{symbol}  |  {start_date} → {end_date}")
    print(f"  初始资金：{capital:,.0f}  |  止损：{stop_loss}  |  止盈：{take_profit}")
    print(f"  vol_window：{vol_window}  |  require_volume：{require_volume}  |  require_green_bar：{require_green_bar}")

    try:
        strategy = LuMACDStrategy(
            vol_window=vol_window,
            require_volume=require_volume,
            require_green_bar=require_green_bar,
        )
        result = run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            initial_capital=capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        plot_lu_backtest(result, save_path=save_chart)

        # 每日状态诊断表（无论是否有交易都保存）
        if save_dir:
            status_csv = os.path.join(save_dir, file_stem + "_daily_status.csv")
            export_daily_status(result, status_csv)

        # 交易记录（有交易时才保存）
        if save_csv and not result["trades"].empty:
            enriched = _enrich_trades(result["trades"], result["df"])
            enriched.to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"  交易记录已保存至：{save_csv}")
        elif save_csv:
            print("  本次回测无成交记录，不生成交易 CSV")

    except Exception as e:
        print(f"\n  ❌ 回测失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
