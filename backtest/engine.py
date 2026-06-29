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

from lib.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED, COLORS,
    setup_matplotlib, style_ax,
)
from lib.market_data import fetch_stock_data   # noqa: F401 — re-export for backward compat

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
    verbose: bool = False,              # True 时打印表头/数据量/结果汇总（库默认静默）
) -> dict:
    """
    核心回测函数，返回回测统计结果 dict。

    verbose=False（默认）时引擎不打印任何内容，由调用方决定如何展示结果，
    避免批量回测刷屏；需要完整汇总表时传 verbose=True 或自行调用 print_summary。
    """

    from strategies import MACDStrategy
    if strategy is None:
        strategy = MACDStrategy()

    if verbose:
        print(f"\n{'='*55}")
        print(f"  A股策略回测  [{strategy.name}]")
        print(f"  股票代码：{symbol}  周期：{start_date} → {end_date}")
        print(f"{'='*55}")

    # ── 获取数据 ──
    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty or len(df) < 50:
        raise ValueError("数据不足，请检查股票代码或延长时间范围")

    if verbose:
        print(f"  获取到 {len(df)} 个交易日数据")

    # ── 计算指标 + 生成信号（委托给策略） ──
    df = strategy.prepare(df)

    # ── 执行信号：T 日产生的信号在 T+1 日开盘成交，消除前视偏差 ──
    # signal 用 T 日收盘价算出，不可能在 T 日收盘价成交；shift(1) 后
    # 第 T 日看到的 signal_exec 实为 T-1 日信号，配合 T 日 open 成交。
    df["signal_exec"] = df["signal"].shift(1).fillna(0)

    # ── 模拟交易 ──
    cash     = initial_capital
    shares   = 0
    position = 0        # 0=空仓，1=持仓
    cost_price = 0.0    # 买入成本价

    trades    = []      # 交易记录
    equity    = []      # 每日资产

    for date, row in df.iterrows():
        price = row["close"]          # 估值用收盘价
        open_ = row["open"]           # 成交用开盘价（T+1 开盘）
        low   = row["low"]
        high  = row["high"]

        # 止损 / 止盈检查（持仓中）——盘中触及即成交，更贴近实盘
        if position == 1 and shares > 0:
            stop_price = cost_price * (1 - stop_loss)   if stop_loss   is not None else None
            tp_price   = cost_price * (1 + take_profit) if take_profit is not None else None
            exit_price = None
            exit_action = None
            # 同一根 K 线内若同时触及，保守起见优先止损
            if stop_price is not None and low <= stop_price:
                exit_price  = min(open_, stop_price)    # 跳空低开则按开盘价
                exit_action = "止损卖出"
            elif tp_price is not None and high >= tp_price:
                exit_price  = max(open_, tp_price)      # 跳空高开则按开盘价
                exit_action = "止盈卖出"

            if exit_price is not None:
                proceeds = shares * exit_price
                fee = proceeds * (commission_rate + stamp_duty)
                cash += proceeds - fee
                pnl_pct = (exit_price - cost_price) / cost_price
                trades.append({
                    "date": date, "action": exit_action, "price": exit_price,
                    "shares": shares, "cash": cash,
                    "return_pct": pnl_pct * 100
                })
                shares = 0; position = 0; cost_price = 0.0

        # 策略信号交易（按 T+1 开盘价成交）
        if row["signal_exec"] == 1 and position == 0:
            # 买入
            invest = cash * position_size
            raw_shares = int(invest / open_ / 100) * 100  # A股最小单位100股
            if raw_shares >= 100:
                cost = raw_shares * open_
                fee  = cost * commission_rate
                cash -= (cost + fee)
                shares = raw_shares
                position = 1
                cost_price = open_
                trades.append({
                    "date": date, "action": "买入", "price": open_,
                    "shares": shares, "cash": cash, "return_pct": None
                })

        elif row["signal_exec"] == -1 and position == 1 and shares > 0:
            # 卖出
            proceeds = shares * open_
            fee = proceeds * (commission_rate + stamp_duty)
            cash += proceeds - fee
            pnl_pct = (open_ - cost_price) / cost_price * 100
            trades.append({
                "date": date, "action": "卖出", "price": open_,
                "shares": shares, "cash": cash,
                "return_pct": pnl_pct
            })
            shares = 0; position = 0; cost_price = 0.0

        # 记录当日资产（按收盘价估值）
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
    # 年化收益：样本不足一年时按日历外推会几何放大（如 10 天 +8% → 年化 ~600%），
    # 失去统计意义，因此样本 < 一年时返回 None，由展示层标注 N/A。
    if n_days >= annual_trading_days:
        annual_return = ((1 + total_return / 100) ** (annual_trading_days / n_days) - 1) * 100
    else:
        annual_return = None
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

    if verbose:
        _print_summary(result)
    return result


def _calc_sharpe(returns: pd.Series, annual_days: int = 252, rf: float = 0.02,
                 min_obs: int = 20):
    """
    年化夏普比率。样本过少时不可靠，返回 None（展示层标注 N/A）。

    min_obs : 有效日收益的最小样本数；少于此值时 std 极不稳定，
              可能算出 18 这种无意义高值或 NaN，故直接弃算。
    """
    daily_rf = rf / annual_days
    excess   = (returns - daily_rf).dropna()      # 去掉 pct_change 首行 NaN
    if len(excess) < min_obs:
        return None
    std = excess.std()
    if not np.isfinite(std) or std == 0:
        return None
    return float(excess.mean() / std * np.sqrt(annual_days))


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


def fmt_sharpe(v) -> str:
    """夏普比率展示格式：None（样本不足）显示 N/A，否则两位小数。"""
    return "N/A" if v is None else f"{v:.2f}"


def print_summary(r: dict):
    """打印回测结果汇总表（公开接口，供单股入口脚本在静默引擎后手动调用）。"""
    _print_summary(r)


def _print_summary(r: dict):
    print(f"\n  ── 回测结果 ──────────────────────────────")
    print(f"  初始资金      : ¥{r['initial_capital']:>12,.2f}")
    print(f"  期末资产      : ¥{r['final_equity']:>12,.2f}")
    print(f"  策略总收益    : {r['total_return']:>+8.2f}%")
    ann = r['annual_return']
    print(f"  策略年化收益  : {'  N/A(样本<1年)' if ann is None else f'{ann:>+8.2f}%'}")
    print(f"  基准收益(持有): {r['benchmark_return']:>+8.2f}%")
    print(f"  超额收益      : {r['total_return'] - r['benchmark_return']:>+8.2f}%")
    print(f"  最大回撤      : {r['max_drawdown']:>8.2f}%")
    shp = r['sharpe_ratio']
    print(f"  夏普比率      : {'  N/A(样本不足)' if shp is None else f'{shp:>8.2f}'}")
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
