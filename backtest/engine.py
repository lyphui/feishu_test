"""
回测引擎（核心，无 CLI）
========================
被所有回测入口脚本复用：run_backtest 执行回测，plot_backtest 绘制标准 4 面板图。

直接调用：
    from engine import run_backtest, plot_backtest
    result = run_backtest("600519", "20200101", "20241231")
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from utils.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED, COLORS,
    setup_matplotlib, style_ax,
)
from utils.market_data import fetch_stock_data   # noqa: F401 — re-export for backward compat

warnings.filterwarnings("ignore")
setup_matplotlib()


# ─────────────────────────────────────────
# 回测引擎
# ─────────────────────────────────────────

def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy=None,                      # BaseStrategy 实例，默认使用 MACDStrategy
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.0003,    # 佣金：万三
    stamp_duty: float = 0.001,          # 印花税：千一（仅卖出收取）
    position_size: float = 1.0,         # 每次建仓比例（1.0 = 全仓）
    stop_loss: float = None,            # 止损比例，如 0.08 = 8%，None = 不止损
    take_profit: float = None,          # 止盈比例，如 0.20 = 20%，None = 不止盈
) -> dict:
    """
    核心回测函数，返回回测统计结果 dict
    """

    from strategies import MACDStrategy
    if strategy is None:
        strategy = MACDStrategy()

    print(f"\n{'='*55}")
    print(f"  A股策略回测  [{strategy.name}]")
    print(f"  股票代码：{symbol}  周期：{start_date} → {end_date}")
    print(f"{'='*55}")

    # ── 获取数据 ──
    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty or len(df) < 50:
        raise ValueError("数据不足，请检查股票代码或延长时间范围")

    print(f"  获取到 {len(df)} 个交易日数据")

    # ── 计算指标 + 生成信号（委托给策略） ──
    df = strategy.prepare(df)

    # ── 模拟交易 ──
    cash     = initial_capital
    shares   = 0
    position = 0        # 0=空仓，1=持仓
    cost_price = 0.0    # 买入成本价

    trades    = []      # 交易记录
    equity    = []      # 每日资产

    for date, row in df.iterrows():
        price = row["close"]

        # 止损 / 止盈检查（持仓中）
        if position == 1 and shares > 0:
            pnl_pct = (price - cost_price) / cost_price
            if stop_loss is not None and pnl_pct <= -stop_loss:
                # 触发止损，强制卖出
                proceeds = shares * price
                fee = proceeds * (commission_rate + stamp_duty)
                cash += proceeds - fee
                trades.append({
                    "date": date, "action": "止损卖出", "price": price,
                    "shares": shares, "cash": cash,
                    "return_pct": pnl_pct * 100
                })
                shares = 0; position = 0; cost_price = 0.0
            elif take_profit is not None and pnl_pct >= take_profit:
                # 触发止盈，强制卖出
                proceeds = shares * price
                fee = proceeds * (commission_rate + stamp_duty)
                cash += proceeds - fee
                trades.append({
                    "date": date, "action": "止盈卖出", "price": price,
                    "shares": shares, "cash": cash,
                    "return_pct": pnl_pct * 100
                })
                shares = 0; position = 0; cost_price = 0.0

        # MACD 信号交易
        if row["signal"] == 1 and position == 0:
            # 买入
            invest = cash * position_size
            raw_shares = int(invest / price / 100) * 100  # A股最小单位100股
            if raw_shares >= 100:
                cost = raw_shares * price
                fee  = cost * commission_rate
                cash -= (cost + fee)
                shares = raw_shares
                position = 1
                cost_price = price
                trades.append({
                    "date": date, "action": "买入", "price": price,
                    "shares": shares, "cash": cash, "return_pct": None
                })

        elif row["signal"] == -1 and position == 1 and shares > 0:
            # 卖出
            proceeds = shares * price
            fee = proceeds * (commission_rate + stamp_duty)
            cash += proceeds - fee
            pnl_pct = (price - cost_price) / cost_price * 100
            trades.append({
                "date": date, "action": "卖出", "price": price,
                "shares": shares, "cash": cash,
                "return_pct": pnl_pct
            })
            shares = 0; position = 0; cost_price = 0.0

        # 记录当日资产
        total = cash + shares * price
        equity.append({"date": date, "equity": total, "close": price})

    # 如果结束时仍持仓，按最后收盘价清算
    if shares > 0:
        last_price = df["close"].iloc[-1]
        last_date  = df.index[-1]
        proceeds = shares * last_price
        fee = proceeds * (commission_rate + stamp_duty)
        cash += proceeds - fee
        pnl_pct = (last_price - cost_price) / cost_price * 100
        trades.append({
            "date": last_date, "action": "期末清仓", "price": last_price,
            "shares": shares, "cash": cash, "return_pct": pnl_pct
        })
        equity[-1]["equity"] = cash

    # ── 统计指标 ──
    eq_df = pd.DataFrame(equity).set_index("date")
    eq_df["returns"]  = eq_df["equity"].pct_change()
    eq_df["drawdown"] = eq_df["equity"] / eq_df["equity"].cummax() - 1

    total_return    = (eq_df["equity"].iloc[-1] / initial_capital - 1) * 100
    annual_trading_days = 252
    n_days          = len(eq_df)
    annual_return   = ((1 + total_return / 100) ** (annual_trading_days / n_days) - 1) * 100
    max_drawdown    = eq_df["drawdown"].min() * 100
    sharpe          = _calc_sharpe(eq_df["returns"], annual_trading_days)
    win_rate, avg_win, avg_loss, profit_factor = _calc_trade_stats(trades)

    # 基准（持有不动）
    bench_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    result = {
        "symbol": symbol, "start": start_date, "end": end_date,
        "initial_capital": initial_capital,
        "final_equity": eq_df["equity"].iloc[-1],
        "total_return": total_return,
        "annual_return": annual_return,
        "benchmark_return": bench_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "total_trades": len([t for t in trades if t["action"] == "买入"]),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "trades": pd.DataFrame(trades),
        "equity_curve": eq_df,
        "df": df,
        "strategy": strategy,
    }

    _print_summary(result)
    return result


def _calc_sharpe(returns: pd.Series, annual_days: int = 252, rf: float = 0.02) -> float:
    daily_rf = rf / annual_days
    excess   = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(annual_days))


def _calc_trade_stats(trades: list):
    closed = [t for t in trades if t.get("return_pct") is not None]
    if not closed:
        return 0.0, 0.0, 0.0, 0.0
    wins   = [t["return_pct"] for t in closed if t["return_pct"] > 0]
    losses = [t["return_pct"] for t in closed if t["return_pct"] <= 0]
    win_rate     = len(wins) / len(closed) * 100 if closed else 0
    avg_win      = np.mean(wins)   if wins   else 0
    avg_loss     = np.mean(losses) if losses else 0
    total_win    = sum(wins)
    total_loss   = abs(sum(losses))
    profit_factor = total_win / total_loss if total_loss > 0 else 0.0
    return win_rate, avg_win, avg_loss, profit_factor


def _print_summary(r: dict):
    print(f"\n  ── 回测结果 ──────────────────────────────")
    print(f"  初始资金      : ¥{r['initial_capital']:>12,.2f}")
    print(f"  期末资产      : ¥{r['final_equity']:>12,.2f}")
    print(f"  策略总收益    : {r['total_return']:>+8.2f}%")
    print(f"  策略年化收益  : {r['annual_return']:>+8.2f}%")
    print(f"  基准收益(持有): {r['benchmark_return']:>+8.2f}%")
    print(f"  超额收益      : {r['total_return'] - r['benchmark_return']:>+8.2f}%")
    print(f"  最大回撤      : {r['max_drawdown']:>8.2f}%")
    print(f"  夏普比率      : {r['sharpe_ratio']:>8.2f}")
    print(f"  交易次数      : {r['total_trades']:>8}  次")
    print(f"  胜率          : {r['win_rate']:>8.1f}%")
    print(f"  平均盈利      : {r['avg_win']:>+8.2f}%")
    print(f"  平均亏损      : {r['avg_loss']:>+8.2f}%")
    pf = r['profit_factor']
    print(f"  盈亏比        : {'    N/A' if pf == 0 else f'{pf:>8.2f}'}")
    print(f"  ──────────────────────────────────────────")


# ─────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────

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
                  f"夏普 {result['sharpe_ratio']:.2f}",
                  color=C_FG, fontsize=12, pad=10)
    ax1.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=9)
    style_ax(ax1)

    # ── 子图2：策略指标（由策略对象自行绘制） ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1, **ax_kwargs)
    strategy.plot_indicators(ax2, df, COLORS)
    style_ax(ax2)

    # ── 子图3：资产曲线 vs 基准 ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1, **ax_kwargs)
    norm_eq    = eq_df["equity"] / result["initial_capital"] * 100
    norm_bench = eq_df["close"]  / eq_df["close"].iloc[0]  * 100
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
