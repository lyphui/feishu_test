"""
A股 MACD 策略回测工具
======================
依赖安装：
    pip install akshare pandas numpy matplotlib

使用方法：
    python macd_backtest.py

或者直接调用函数：
    from macd_backtest import run_backtest
    result = run_backtest("600519", "20200101", "20241231")
"""

import configparser
import os
from datetime import date as _date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
import sys

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'STHeiti', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────
# 1. 数据获取
# ─────────────────────────────────────────

def fetch_stock_data(symbol: str, start_date: str, end_date: str,
                     proxy: str = "") -> pd.DataFrame:
    """
    获取A股历史行情数据（前复权）

    symbol: 股票代码，如 "600519"（茅台）、"000858"（五粮液）
    start_date / end_date: 格式 "YYYYMMDD"
    proxy: HTTP 代理地址，如 "http://127.0.0.1:7890"，留空则不使用
    """
    # akshare 底层用 requests，设置环境变量即可让它走代理
    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy

    try:
        import akshare as ak
        print(f"  正在从 akshare 获取 {symbol} 数据...")
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    except ImportError:
        print("  未找到 akshare，尝试使用 yfinance 作为备用...")
        return _fetch_via_yfinance(symbol, start_date, end_date)

    except Exception as e:
        print(f"  akshare 获取失败：{e}，尝试 yfinance 备用...")
        return _fetch_via_yfinance(symbol, start_date, end_date)


def _fetch_via_yfinance(symbol: str, start_date: str, end_date: str,
                        max_retries: int = 3, retry_delay: int = 10) -> pd.DataFrame:
    """yfinance 备用，自动判断沪/深，兼容新版 MultiIndex 列，支持限流重试"""
    import time
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("未安装 yfinance，请运行：pip install yfinance")

    suffix = ".SS" if symbol.startswith("6") else ".SZ"
    ticker = symbol + suffix
    start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end   = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  正在从 yfinance 获取 {ticker} 数据（第 {attempt}/{max_retries} 次）...")
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raise ValueError(f"未返回数据，ticker={ticker}")
            # 新版 yfinance 返回 MultiIndex(field, ticker)，需要降级
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            return raw[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "RateLimit" in type(e).__name__ or "Too Many Requests" in err_str or "429" in err_str:
                if attempt < max_retries:
                    wait = retry_delay * attempt
                    print(f"  yfinance 触发限流，等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
            break

    raise RuntimeError(
        f"数据获取失败：{last_err}\n"
        "建议：\n"
        "  1. 稍等几分钟后再运行（yfinance 有访问频率限制）\n"
        "  2. 确认已安装 akshare：pip install akshare\n"
        "  3. 在配置文件中填写 proxy（如 http://127.0.0.1:7890）"
    )


# ─────────────────────────────────────────
# 2. 回测引擎
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
    profit_factor = total_win / total_loss if total_loss > 0 else float("inf")
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
    print(f"  盈亏比        : {r['profit_factor']:>8.2f}")
    print(f"  ──────────────────────────────────────────")


# ─────────────────────────────────────────
# 4. 可视化
# ─────────────────────────────────────────

def plot_backtest(result: dict, save_path: str = None):
    df       = result["df"]
    eq_df    = result["equity_curve"]
    trades   = result["trades"]
    symbol   = result["symbol"]
    strategy = result["strategy"]

    fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
    gs  = GridSpec(4, 1, figure=fig, hspace=0.08,
                   height_ratios=[3, 1.5, 1.5, 1.5])

    c_bg    = "#0d1117"
    c_fg    = "#e6edf3"
    c_green = "#39d353"
    c_red   = "#f85149"
    c_blue  = "#58a6ff"
    c_gold  = "#e3b341"
    c_muted = "#484f58"

    ax_kwargs = dict(facecolor=c_bg)

    # ── 子图1：K线 + 买卖点 ──
    ax1 = fig.add_subplot(gs[0], **ax_kwargs)
    ax1.plot(df.index, df["close"], color=c_blue, lw=1.2, label="收盘价")

    if not trades.empty:
        buys  = trades[trades["action"] == "买入"]
        sells = trades[trades["action"].isin(["卖出", "止损卖出", "止盈卖出", "期末清仓"])]
        ax1.scatter(buys["date"],  buys["price"],  marker="^", color=c_green,
                    s=80, zorder=5, label="买入")
        ax1.scatter(sells["date"], sells["price"], marker="v", color=c_red,
                    s=80, zorder=5, label="卖出")

    ax1.set_title(f"A股策略回测 [{strategy.name}]  |  {symbol}  |  "
                  f"总收益 {result['total_return']:+.2f}%  "
                  f"基准 {result['benchmark_return']:+.2f}%  "
                  f"夏普 {result['sharpe_ratio']:.2f}",
                  color=c_fg, fontsize=12, pad=10)
    ax1.legend(facecolor=c_bg, labelcolor=c_fg, edgecolor=c_muted, fontsize=9)
    _style_ax(ax1, c_fg, c_muted)

    # ── 子图2：策略指标（由策略对象自行绘制） ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1, **ax_kwargs)
    _colors = dict(bg=c_bg, fg=c_fg, green=c_green, red=c_red,
                   blue=c_blue, gold=c_gold, muted=c_muted)
    strategy.plot_indicators(ax2, df, _colors)
    _style_ax(ax2, c_fg, c_muted)

    # ── 子图3：资产曲线 vs 基准 ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1, **ax_kwargs)
    norm_eq    = eq_df["equity"] / result["initial_capital"] * 100
    norm_bench = eq_df["close"]  / eq_df["close"].iloc[0]  * 100
    ax3.plot(eq_df.index, norm_eq,    color=c_green, lw=1.5, label="策略净值")
    ax3.plot(eq_df.index, norm_bench, color=c_muted, lw=1,   label="基准(买入持有)", linestyle="--")
    ax3.axhline(100, color=c_muted, lw=0.5, linestyle=":")
    ax3.legend(facecolor=c_bg, labelcolor=c_fg, edgecolor=c_muted, fontsize=8)
    ax3.set_ylabel("净值（基准=100）", color=c_fg, fontsize=9)
    _style_ax(ax3, c_fg, c_muted)

    # ── 子图4：回撤 ──
    ax4 = fig.add_subplot(gs[3], sharex=ax1, **ax_kwargs)
    ax4.fill_between(eq_df.index, eq_df["drawdown"] * 100, 0,
                     color=c_red, alpha=0.4, label="策略回撤")
    ax4.set_ylabel("回撤 (%)", color=c_fg, fontsize=9)
    ax4.legend(facecolor=c_bg, labelcolor=c_fg, edgecolor=c_muted, fontsize=8)
    _style_ax(ax4, c_fg, c_muted)

    # ── 关键日期：每笔交易日期画垂直虚线并在价格图顶部标注日期 ──
    if not trades.empty:
        price_max = df["close"].max()
        price_min = df["close"].min()
        label_y   = price_max + (price_max - price_min) * 0.01
        for _, trade in trades.iterrows():
            t_date   = trade["date"]
            t_action = trade["action"]
            t_color  = c_green if t_action == "买入" else c_red
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
    plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", color=c_fg, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=c_bg)
        print(f"\n  图表已保存至：{save_path}")
    else:
        plt.show()

    return fig


def _style_ax(ax, fg, muted):
    ax.tick_params(colors=fg, labelsize=8)
    ax.spines[:].set_color(muted)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.yaxis.label.set_color(fg)
    ax.grid(color=muted, linewidth=0.3, alpha=0.5)


# ─────────────────────────────────────────
# 5. 命令行入口
# ─────────────────────────────────────────

def _write_default_config(path: str) -> None:
    """生成默认配置文件。"""
    content = """\
[backtest]
# 股票代码（沪市6开头，深市0/3开头）
symbol     = 600519

# 股票名称（用于文件名，建议拼音或英文）
name       = maotai

# 回测区间（YYYYMMDD）
start_date = 20200101
# end_date 留空则默认使用当天日期
end_date   =

# 初始资金（元）
capital    = 100000

# 止损比例（如 0.08 表示 8%），留空则不设置
stop_loss  =

# 止盈比例（如 0.20 表示 20%），留空则不设置
take_profit =

# 图表和CSV保存目录（留空则弹窗显示，不保存CSV）
save_chart_dir = output/

# HTTP 代理（如 http://127.0.0.1:7890），留空则直连
proxy =
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    from strategies import MACDStrategy

    print("\n" + "─"*55)
    print("  A股策略回测工具")
    print("  数据来源：akshare（前复权）")
    print("─"*55)

    # 读取配置文件
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "rjgd_syr_260130.ini")
    if not os.path.exists(config_path):
        print(f"  配置文件不存在，已生成默认配置：{config_path}")
        _write_default_config(config_path)

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    s = cfg["backtest"]

    symbol     = s.get("symbol", "600519").strip()
    name       = s.get("name", "stock").strip()
    start_date = s.get("start_date", "20200101").strip()
    end_date   = s.get("end_date", "").strip()
    if not end_date:
        end_date = _date.today().strftime("%Y%m%d")
        print(f"  end_date 未设置，默认使用今日：{end_date}")
    capital    = float(s.get("capital", "100000"))
    stop_loss_raw  = s.get("stop_loss", "").strip()
    stop_loss      = float(stop_loss_raw) if stop_loss_raw else None
    take_profit_raw = s.get("take_profit", "").strip()
    take_profit     = float(take_profit_raw) if take_profit_raw else None
    save_dir   = s.get("save_chart_dir", "").strip()
    proxy      = s.get("proxy", "").strip()
    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"  代理：{proxy}")

    # 构建保存路径
    file_stem = f"{name}_{symbol}_{end_date}"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_chart = os.path.join(save_dir, file_stem + ".png")
        save_csv   = os.path.join(save_dir, file_stem + ".csv")
    else:
        save_chart = None
        save_csv   = None

    print(f"  股票代码：{symbol}  |  {start_date} → {end_date}")
    print(f"  初始资金：{capital:,.0f}  |  止损：{stop_loss}  |  止盈：{take_profit}")

    try:
        strategy = MACDStrategy()
        result = run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            initial_capital=capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        plot_backtest(result, save_path=save_chart)

        # 保存交易记录
        if save_csv and not result["trades"].empty:
            result["trades"].to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"  交易记录已保存至：{save_csv}")

    except Exception as e:
        print(f"\n  ❌ 回测失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()