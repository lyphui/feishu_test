"""
JCY 日线信号 + 分时择时（多周期共振）
======================================
当日线 LuMACDBull 策略触发买/卖信号时，
获取执行日的分时行情（默认 30 分钟 K 线），
在分时级别应用 MACD 动能分析，推荐最佳操作时间窗口。

设计思路（多周期共振）：
  日线 → 确认方向（买 / 卖）
  分时 → 确认时机（几点操作）

  日线金叉 + 红柱拉长 → 确认买入方向
  分时金叉 + 红柱拉长 → 确认具体入场时机
  两者同向共振 = 最优操作点

常见问题：日线满足条件但分时找不到好时机吗？
──────────────────────────────────────────────
  会，常见场景：
  1. 高开低走：日线金叉收盘，次日跳空高开后立刻回落，分时全天无拉长窗口
  2. 全天震荡：分时 MACD 反复穿越零轴，无持续方向，找不到共振点
  3. 信号滞后：日线信号 T 日收盘确认，T+1 开盘已涨 3-5%，无安全价位可进
  4. 卖出跌停：一字跌停，分时无成交，"好的卖出时机"已不存在
  5. 小盘流动性不足：分时 MACD 因稀疏成交而失真

  应对建议（本脚本会自动提示）：
    买入无 GO 窗口 → 挂限价单，不追高，等次日再观察
    卖出无 GO 窗口 → 次日集合竞价直接挂单卖出，不等盘中机会

用法：
    python backtest/jcy_intraday_timing.py
    python backtest/jcy_intraday_timing.py --lookback 15   # 只看最近 15 天的信号
    python backtest/jcy_intraday_timing.py --period 60     # 用 60min K 线
    python backtest/jcy_intraday_timing.py --code 600519   # 只分析指定股票
    python backtest/jcy_intraday_timing.py --exec_day same # 信号当日分析（盘中发现信号时）
"""

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from macd_analysis import fetch_stock_data
from strategies import LuMACDBullStrategy
from utils.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED,
    setup_matplotlib, style_ax,
)
from utils.market_data import fetch_index_data
from utils.bull_backtest import BullStrategyAdapter
from utils.jcy_common import JSON_PATH, load_candidates

setup_matplotlib()


# ── 配置常量 ─────────────────────────────────────────────────────────────────

WARMUP_DAYS = 600               # 日线 MACD 预热所需自然日
INTRADAY_WARMUP_DAYS = 55       # 分时 MACD 预热所需自然日（需覆盖 slow=26 周期）
CHART_CONTEXT_DAYS = 28         # 分时图表显示执行日前多少天的上下文
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

TIMING_LABEL = {
    "GO":    "✅ GO   ",
    "WAIT":  "⏳ WAIT ",
    "AVOID": "🚫 AVOID",
}


# ── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class TimingSummary:
    """单个执行日的分时择时汇总。"""
    has_go: bool
    go_times: list = field(default_factory=list)
    first_go: pd.Timestamp | None = None
    second_go: pd.Timestamp | None = None
    go_count: int = 0
    total_bars: int = 0


@dataclass
class SignalTimingResult:
    """单个信号的完整择时分析结果。"""
    code: str
    name: str
    action: str
    signal_date: pd.Timestamp
    exec_date: pd.Timestamp
    has_go: bool
    first_go: pd.Timestamp | None
    go_count: int


# ── 分时数据获取 ──────────────────────────────────────────────────────────────

def fetch_intraday(symbol: str, start_date: str, end_date: str,
                   period: int = 30) -> pd.DataFrame:
    """
    获取分时 K 线（前复权）。
    start_date / end_date: YYYYMMDD 格式
    period: 分钟数，5 / 15 / 30 / 60
    """
    try:
        import akshare as ak
        s = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 09:30:00"
        e = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]} 15:00:00"
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            period=str(period),
            start_date=s,
            end_date=e,
            adjust="qfq",
        )
        df = df.rename(columns={
            "时间":  "datetime",
            "开盘":  "open",
            "收盘":  "close",
            "最高":  "high",
            "最低":  "low",
            "成交量": "volume",
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols]
    except Exception as e:
        print(f"    ⚠ 分时数据获取失败：{e}")
        return pd.DataFrame()


# ── 分时 MACD 计算 ────────────────────────────────────────────────────────────

def add_macd(df: pd.DataFrame,
             fast: int = MACD_FAST,
             slow: int = MACD_SLOW,
             sig: int = MACD_SIGNAL) -> pd.DataFrame:
    """在 DataFrame 上追加 DIF / DEA / MACD / hist_expanding / hist_shrinking 列。"""
    df = df.copy()
    ema_f = df["close"].ewm(span=fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=slow, adjust=False).mean()
    dif   = ema_f - ema_s
    dea   = dif.ewm(span=sig, adjust=False).mean()
    hist  = (dif - dea) * 2

    df["DIF"]  = dif
    df["DEA"]  = dea
    df["MACD"] = hist
    df["hist_expanding"] = (hist > 0) & (hist > hist.shift(1))
    df["hist_shrinking"] = (hist > 0) & (hist < hist.shift(1))
    return df


# ── 分时择时核心逻辑 ──────────────────────────────────────────────────────────

def classify_timing(day_slice: pd.DataFrame, action: str) -> pd.DataFrame:
    """
    对执行日的每根分时 K 线打时机标签：GO / WAIT / AVOID。

    action: 'buy' or 'sell'

    买入判断：
      GO    → DIF > DEA 且红柱正在拉长（与日线信号完全共振）
      WAIT  → DIF > DEA 但红柱未拉长（方向对，等动能起步）
      AVOID → DIF <= DEA（方向相反，开盘混沌期或回调中）

    卖出判断：
      GO    → 红柱开始缩短（动能衰减）或 DIF 下穿 DEA（死叉）
      WAIT  → 动能仍在高位（暂时持有）
    """
    df = day_slice.copy()

    if action == "buy":
        conditions = [
            df["hist_expanding"] & (df["DIF"] > df["DEA"]),
            df["DIF"] > df["DEA"],
        ]
        df["timing"] = np.select(conditions, ["GO", "WAIT"], default="AVOID")
    else:
        death_cross = (df["DIF"] < df["DEA"]) & (df["DIF"].shift(1) >= df["DEA"].shift(1))
        df["timing"] = np.select(
            [df["hist_shrinking"] | death_cross],
            ["GO"],
            default="WAIT",
        )

    return df


def summarize_timing(day_df: pd.DataFrame) -> TimingSummary:
    """从打好标签的执行日切片中汇总 GO 窗口信息。"""
    go_bars = day_df[day_df["timing"] == "GO"]
    return TimingSummary(
        has_go=len(go_bars) > 0,
        go_times=list(go_bars.index),
        first_go=go_bars.index[0] if len(go_bars) > 0 else None,
        second_go=go_bars.index[1] if len(go_bars) > 1 else None,
        go_count=len(go_bars),
        total_bars=len(day_df),
    )


# ── 绘图辅助 ─────────────────────────────────────────────────────────────────

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


# ── 打印择时表格 ──────────────────────────────────────────────────────────────

def _print_timing_advice(summary: TimingSummary, action_cn: str):
    """打印操作建议（与表格解耦，便于独立维护）。"""
    print(f"\n    📌 操作建议：")
    if summary.has_go:
        if summary.first_go:
            print(f"      首选：{summary.first_go.strftime('%H:%M')}"
                  f"  ← {action_cn}，30min MACD 与日线方向共振")
        if summary.second_go:
            print(f"      次选：{summary.second_go.strftime('%H:%M')}"
                  f"  ← 动能加速确认后{action_cn}")
    elif action_cn == "买入":
        print(f"      ⚠️  全天无明确 GO 窗口（分时条件未共振）")
        print(f"         建议：挂限价单，不追高；或等次日再观察")
    else:
        print(f"      ⚠️  全天无明确缩量 / 死叉信号")
        print(f"         建议：若持仓，在次日集合竞价挂单卖出，不等盘中机会")


def print_timing_table(exec_bars: pd.DataFrame, summary: TimingSummary,
                       action_cn: str, code: str, name: str,
                       signal_date: pd.Timestamp, exec_date: pd.Timestamp):
    """打印分时择时明细表 + 操作建议。"""
    print(f"\n    ── {code} {name}  {action_cn}信号 "
          f"{signal_date.strftime('%Y-%m-%d')}  →  执行日 "
          f"{exec_date.strftime('%Y-%m-%d')} ──")
    print(f"    {'时间':6s}  {'收盘':>8s}  {'DIF':>8s}  {'DEA':>8s}  "
          f"{'MACD柱':>9s}  {'拉长':4s}  {'建议'}")
    print(f"    {'─' * 68}")

    for dt, row in exec_bars.iterrows():
        expanding = "✓" if row.get("hist_expanding") else " "
        label     = TIMING_LABEL.get(row.get("timing", ""), "")
        print(f"    {dt.strftime('%H:%M'):6s}  "
              f"{row['close']:>8.2f}  "
              f"{row.get('DIF',  float('nan')):>8.4f}  "
              f"{row.get('DEA',  float('nan')):>8.4f}  "
              f"{row.get('MACD', float('nan')):>+9.4f}  "
              f"{expanding:4s}  {label}")

    _print_timing_advice(summary, action_cn)


# ── 单股分析（拆分后的子函数） ───────────────────────────────────────────────

def _fetch_daily_signals(
    code: str, name: str, trade_start: str,
    index_symbol: str, lookback_days: int,
    warmup_days: int = WARMUP_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    获取日线+大盘数据，运行 LuMACDBull 策略，返回 lookback 窗口内的信号。
    返回 (df_sig, signal_days) 或 None（失败时打印原因）。
    """
    today_str  = _date.today().strftime("%Y%m%d")
    trade_dt   = _date(int(trade_start[:4]),
                       int(trade_start[4:6]),
                       int(trade_start[6:]))
    data_start = (trade_dt - timedelta(days=warmup_days)).strftime("%Y%m%d")

    daily_df = fetch_stock_data(code, data_start, today_str)
    if daily_df.empty or len(daily_df) < 50:
        print(f"    日线数据不足，跳过")
        return None

    index_df = fetch_index_data(index_symbol, data_start, today_str)
    if index_df.empty:
        print(f"    大盘数据为空，跳过")
        return None

    inner   = LuMACDBullStrategy(shrink_exit=True)
    adapter = BullStrategyAdapter(inner, index_df, trade_start_date=trade_start)
    df_sig  = adapter.prepare(daily_df)

    cutoff      = pd.Timestamp.today() - timedelta(days=lookback_days)
    signal_days = df_sig[(df_sig.index >= cutoff) & (df_sig["signal"] != 0)]

    if signal_days.empty:
        print(f"    最近 {lookback_days} 天无买/卖信号")
        return None

    return df_sig, signal_days


def _determine_exec_date(df_sig: pd.DataFrame,
                         sig_date: pd.Timestamp,
                         mode: str) -> pd.Timestamp:
    """根据 exec_day_mode 确定执行日（same=信号当日，next=信号次日）。"""
    if mode == "same":
        return sig_date
    future = df_sig.index[df_sig.index > sig_date]
    return future[0] if len(future) > 0 else sig_date


def _analyze_single_signal(
    code: str, exec_date: pd.Timestamp, action: str, period: int,
) -> tuple[pd.DataFrame, pd.DataFrame, TimingSummary] | None:
    """
    获取分时数据 → 计算 MACD → 择时分类。
    返回 (intra_df, exec_bars, summary) 或 None（数据缺失时打印原因）。
    """
    intra_start = (exec_date - timedelta(days=INTRADAY_WARMUP_DAYS)).strftime("%Y%m%d")
    intra_end   = exec_date.strftime("%Y%m%d")
    intra_df    = fetch_intraday(code, intra_start, intra_end, period)

    if intra_df.empty:
        print(f"    分时数据为空，跳过")
        return None

    intra_df  = add_macd(intra_df)
    exec_mask = intra_df.index.normalize() == exec_date
    exec_bars = intra_df[exec_mask].copy()

    if exec_bars.empty:
        print(f"    执行日 {exec_date.strftime('%Y-%m-%d')} 无分时数据"
              f"（可能是非交易日或数据缺失）")
        return None

    exec_bars = classify_timing(exec_bars, action)
    summary   = summarize_timing(exec_bars)
    return intra_df, exec_bars, summary


def _save_signal_chart(intra_df: pd.DataFrame, exec_date: pd.Timestamp,
                       code: str, name: str, action: str,
                       sig_date: pd.Timestamp, summary: TimingSummary,
                       period: int, save_dir: str):
    """生成安全文件名并调用绘图。"""
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", name)
    fname = (f"intraday_{code}_{safe_name}_{action}"
             f"_{sig_date.strftime('%Y%m%d')}"
             f"_{exec_date.strftime('%Y%m%d')}.png")
    plot_intraday_chart(
        intraday_df=intra_df,
        exec_date=exec_date,
        symbol=code,
        name=name,
        action=action,
        signal_date=sig_date,
        summary=summary,
        period=period,
        save_path=os.path.join(save_dir, fname),
    )


# ── 单股分析（编排函数） ─────────────────────────────────────────────────────

def analyze_candidate(
    candidate: dict,
    lookback_days: int,
    index_symbol: str,
    period: int,
    exec_day_mode: str,
    save_dir: str,
) -> list[SignalTimingResult]:
    """
    对单只股票：
      1. 获取日线数据，运行 LuMACDBull 策略
      2. 找出 lookback_days 内的买 / 卖信号日
      3. 逐个信号日：获取分时数据 → MACD → 择时分析 → 绘图
    """
    code = candidate["code"]
    name = candidate["name"]
    print(f"\n  [{code}] {name}")

    try:
        fetched = _fetch_daily_signals(
            code, name, candidate["date"], index_symbol, lookback_days)
        if fetched is None:
            return []
        df_sig, signal_days = fetched

        results: list[SignalTimingResult] = []
        for sig_date, sig_row in signal_days.iterrows():
            action    = "buy" if sig_row["signal"] == 1 else "sell"
            action_cn = "买入" if action == "buy" else "卖出"
            exec_date = _determine_exec_date(df_sig, sig_date, exec_day_mode)

            print(f"    {action_cn}信号：{sig_date.strftime('%Y-%m-%d')}  "
                  f"执行日：{exec_date.strftime('%Y-%m-%d')}")

            analysis = _analyze_single_signal(code, exec_date, action, period)
            if analysis is None:
                continue
            intra_df, exec_bars, summary = analysis

            print_timing_table(exec_bars, summary, action_cn, code, name,
                               sig_date, exec_date)
            _save_signal_chart(intra_df, exec_date, code, name, action,
                               sig_date, summary, period, save_dir)

            results.append(SignalTimingResult(
                code=code, name=name, action=action_cn,
                signal_date=sig_date, exec_date=exec_date,
                has_go=summary.has_go, first_go=summary.first_go,
                go_count=summary.go_count,
            ))

        return results

    except Exception as e:
        print(f"    ❌ 分析失败：{e}")
        import traceback
        traceback.print_exc()
        return []


# ── 主入口 ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="JCY 日线信号 + 分时择时（多周期共振）"
    )
    parser.add_argument("--lookback",  type=int, default=30,
                        help="向前查找信号的天数，默认 30 天")
    parser.add_argument("--period",    type=int, default=30,
                        choices=[5, 15, 30, 60],
                        help="分时 K 线周期（分钟），默认 30")
    parser.add_argument("--index",     type=str, default="000300",
                        help="大盘指数代码，默认 000300（沪深300）")
    parser.add_argument("--exec_day",  type=str, default="next",
                        choices=["next", "same"],
                        help="执行日：next=信号次日（默认），same=信号当日")
    parser.add_argument("--code",      type=str, default=None,
                        help="只分析指定股票代码，留空则分析全部")
    parser.add_argument("--output",    type=str, default="output/intraday",
                        help="输出目录，默认 output/intraday/")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "─" * 65)
    print("  JCY 日线信号 + 分时择时（多周期共振）")
    print("─" * 65)
    print(f"  信号查找窗口  ：最近 {args.lookback} 天")
    print(f"  分时 K 线周期 ：{args.period} 分钟")
    print(f"  执行日模式    ：{args.exec_day}"
          f"（{'信号次日' if args.exec_day == 'next' else '信号当日'}）")
    print(f"  大盘指数      ：{args.index}")
    print(f"  输出目录      ：{args.output}/")
    print("─" * 65)

    if not os.path.exists(JSON_PATH):
        print(f"  ❌ 找不到 JSON 文件：{JSON_PATH}")
        sys.exit(1)

    candidates = load_candidates(JSON_PATH)
    if not candidates:
        print("  ❌ 未找到增持 A 股，请检查 JSON 数据")
        sys.exit(1)

    if args.code:
        candidates = [c for c in candidates if c["code"] == args.code]
        if not candidates:
            print(f"  ❌ 未找到代码 {args.code}，请检查 JSON 数据")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"  共 {len(candidates)} 只候选股票\n")

    all_results: list[SignalTimingResult] = []
    for candidate in candidates:
        results = analyze_candidate(
            candidate=candidate,
            lookback_days=args.lookback,
            index_symbol=args.index,
            period=args.period,
            exec_day_mode=args.exec_day,
            save_dir=args.output,
        )
        all_results.extend(results)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  分时择时汇总")
    print("═" * 65)

    if all_results:
        print(f"  {'代码':8s}  {'名称':8s}  {'操作':4s}  "
              f"{'信号日':12s}  {'执行日':12s}  {'首选时间':8s}  {'状态'}")
        print(f"  {'─' * 65}")
        for r in all_results:
            first_go_str = (r.first_go.strftime("%H:%M")
                            if r.first_go else "  —  ")
            status = (f"✅ {r.go_count} 个 GO 窗口"
                      if r.has_go else "⚠️  无 GO 窗口，建议等待")
            print(f"  {r.code:8s}  {r.name:8s}  {r.action:4s}  "
                  f"{r.signal_date.strftime('%Y-%m-%d'):12s}  "
                  f"{r.exec_date.strftime('%Y-%m-%d'):12s}  "
                  f"{first_go_str:8s}  {status}")
    else:
        print(f"  最近 {args.lookback} 天内无买/卖信号")
        print(f"  → 可通过 --lookback 扩大查找窗口，例如 --lookback 60")

    print("\n" + "─" * 65)
    print(f"  完成。结果已保存至：{os.path.abspath(args.output)}/")
    print("─" * 65)


if __name__ == "__main__":
    main()
