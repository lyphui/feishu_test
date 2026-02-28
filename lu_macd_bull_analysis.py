"""
卢麒元 MACD 牛市动能截取策略回测入口
=====================================
复用 macd_analysis.py 中的数据获取和回测引擎，
使用专属绘图函数展示：日线价格、MACD指标（含牛市背景）、资产曲线、回撤。

使用方法：
    python lu_macd_bull_analysis.py

配置文件：config/lu_macd_bull_config.ini
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
from strategies import LuMACDBullStrategy
from strategies.base import BaseStrategy


def fetch_index_data(symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
    """
    专用指数数据获取函数，支持 akshare 和 yfinance 备用。

    symbol: 指数代码，如 "000300"（沪深300）、"000001"（上证）、"399001"（深证成指）
    """
    import pandas as pd

    # ── akshare 优先 ─────────────────────────────────────────────────────────
    try:
        import akshare as ak
        # 判断 sh / sz 前缀：399xxx 为深证，其余默认沪市
        prefix = "sz" if symbol.startswith("399") else "sh"
        ak_symbol = prefix + symbol
        print(f"  正在从 akshare 获取指数 {ak_symbol} 数据...")
        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        df = df.rename(columns={"date": "date", "close": "close",
                                 "open": "open", "high": "high",
                                 "low": "low", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # 按日期范围裁剪
        start = pd.to_datetime(start_date, format="%Y%m%d")
        end   = pd.to_datetime(end_date,   format="%Y%m%d")
        df = df.loc[start:end]
        if not df.empty:
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols]
    except ImportError:
        pass
    except Exception as e:
        print(f"  akshare 指数获取失败：{e}，尝试 yfinance 备用...")

    # ── yfinance 备用 ─────────────────────────────────────────────────────────
    import yfinance as yf
    # 399xxx 深证 → .SZ；其他（000xxx）沪证 → .SS
    suffix = ".SZ" if symbol.startswith("399") else ".SS"
    ticker = symbol + suffix
    start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_str   = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    print(f"  正在从 yfinance 获取指数 {ticker} 数据...")
    raw = yf.download(ticker, start=start_str, end=end_str,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"指数数据获取失败，ticker={ticker}")
    if isinstance(raw.columns, __import__("pandas").MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = __import__("pandas").to_datetime(raw.index)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    return raw[cols]


plt.rcParams["font.sans-serif"] = ["SimHei", "STHeiti", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

C_BG    = "#0d1117"
C_FG    = "#e6edf3"
C_GREEN = "#39d353"
C_RED   = "#f85149"
C_BLUE  = "#58a6ff"
C_GOLD  = "#e3b341"
C_MUTED = "#484f58"
COLORS  = dict(bg=C_BG, fg=C_FG, green=C_GREEN, red=C_RED,
               blue=C_BLUE, gold=C_GOLD, muted=C_MUTED)


def _style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.tick_params(colors=C_FG, labelsize=8)
    ax.spines[:].set_color(C_MUTED)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.yaxis.label.set_color(C_FG)
    ax.grid(color=C_MUTED, linewidth=0.3, alpha=0.5)


class _BullStrategyAdapter(BaseStrategy):
    """
    将 LuMACDBullStrategy.prepare(df, index_df) 的双参数接口
    适配为 run_backtest 期望的单参数 prepare(df) 接口。
    """

    def __init__(self, inner: LuMACDBullStrategy, index_df):
        self._inner    = inner
        self._index_df = index_df

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def params(self) -> dict:
        return self._inner.params

    def prepare(self, df):
        return self._inner.prepare(df, self._index_df)

    def plot_indicators(self, ax, df, colors):
        return self._inner.plot_indicators(ax, df, colors)


def export_daily_status(result: dict, save_path: str) -> None:
    """
    将每日指标状态导出为 CSV，方便排查牛市动能截取策略的买卖判断依据。

    输出列说明
    ----------
    日期              交易日
    收盘价            当日收盘价
    大盘_DIF          大盘月线 DIF（对齐到日线，ffill）
    大盘_DEA          大盘月线 DEA
    大盘_DIF>0        大盘月线 DIF 是否在 0 轴上方
    大盘_DIF>DEA      大盘月线 DIF 是否上穿 DEA
    牛市确认          bull_market：两条件同时满足
    个股_DIF          个股日线 DIF
    个股_DEA          个股日线 DEA
    个股_MACD柱       个股日线 MACD 柱（2*(DIF-DEA)）
    金叉              DIF 上穿 DEA（当根）
    红柱拉长          hist_expanding：MACD>0 且本根>上根
    红柱缩短          hist_shrinking：MACD>0 且本根<上根
    信号              1=买入 / -1=卖出 / 0=观望
    未达条件          当日阻断买入/触发卖出的原因描述
    """
    import pandas as pd

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

        idx_above0   = dif_idx > 0
        idx_dif_up   = dif_idx > dea_idx

        # 金叉判断（DIF 前一根需在 df 里取，这里用近似：DIF>DEA 且 MACD 柱从负转正或已转）
        # 精确金叉已在策略内计算为 signal==1 的触发条件之一，此处用 DIF 与 DEA 相对位置作参考
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
            # 观望：逐层拆解阻断原因
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
            "日期":           date.strftime("%Y-%m-%d"),
            "收盘价":         round(r["close"], 2),
            "大盘_DIF":       round(dif_idx, 4),
            "大盘_DEA":       round(dea_idx, 4),
            "大盘_DIF>0":     idx_above0,
            "大盘_DIF>DEA":   idx_dif_up,
            "牛市确认":       bull,
            "个股_DIF":       round(dif,  4),
            "个股_DEA":       round(dea,  4),
            "个股_MACD柱":    round(macd, 4),
            "红柱拉长":       expanding,
            "红柱缩短":       shrinking,
            "信号":           signal,
            "判断依据":       blocking,
        })

    status_df = pd.DataFrame(rows)
    status_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  每日状态表已保存至：{save_path}")


def plot_bull_backtest(result: dict, save_path: "str | None" = None):
    """
    专属绘图：5 个子图
      1. 日线价格 + 买卖执行点
      2. 日线 MACD（含牛市背景色、动能拉长/缩短高亮）
      3. 大盘月线 DIF/DEA 参考线（牛市判断依据）
      4. 资产曲线 vs 基准
      5. 回撤
    """
    df     = result["df"]
    eq_df  = result["equity_curve"]
    trades = result["trades"]
    symbol = result["symbol"]

    fig = plt.figure(figsize=(18, 18), facecolor=C_BG)
    gs  = GridSpec(5, 1, figure=fig, hspace=0.06,
                   height_ratios=[3, 2, 1.5, 1.5, 1])

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
        f"卢麒元牛市动能截取策略  |  {symbol}  |  "
        f"总收益 {result['total_return']:+.2f}%  "
        f"基准 {result['benchmark_return']:+.2f}%  "
        f"夏普 {result['sharpe_ratio']:.2f}",
        color=C_FG, fontsize=12, pad=8,
    )
    ax1.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=9)
    _style_ax(ax1)

    # ── 子图2：日线 MACD + 牛市背景 + 动能高亮 ──────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # 牛市背景色
    if "bull_market" in df.columns:
        bull_on = False
        bull_start = None
        for date, row in df.iterrows():
            if row["bull_market"] and not bull_on:
                bull_start = date; bull_on = True
            elif not row["bull_market"] and bull_on:
                ax2.axvspan(bull_start, date, alpha=0.07, color=C_GREEN, lw=0)
                bull_on = False
        if bull_on and bull_start is not None:
            ax2.axvspan(bull_start, df.index[-1], alpha=0.07, color=C_GREEN, lw=0)

    # MACD 柱：拉长段加深显示
    expanding_mask = df.get("hist_expanding", __import__("pandas").Series(False, index=df.index))
    bar_colors = np.where(df["MACD"] >= 0, C_RED, C_GREEN)
    ax2.bar(df.index[~expanding_mask], df["MACD"][~expanding_mask],
            color=bar_colors[~expanding_mask.values], alpha=0.4, width=1, label="MACD柱")
    ax2.bar(df.index[expanding_mask], df["MACD"][expanding_mask],
            color=bar_colors[expanding_mask.values], alpha=0.9, width=1, label="MACD柱(动能↑)")

    ax2.plot(df.index, df["DIF"], color=C_BLUE, lw=1.2, label="DIF")
    ax2.plot(df.index, df["DEA"], color=C_GOLD, lw=1.2, label="DEA")
    ax2.axhline(0, color=C_MUTED, lw=0.6, linestyle="--")

    # 信号标记
    buy_idx  = df.index[df["signal"] == 1]
    sell_idx = df.index[df["signal"] == -1]
    if len(buy_idx):
        ax2.scatter(buy_idx, df.loc[buy_idx, "DIF"],
                    marker="^", color=C_GREEN, s=80, zorder=7, label="买入(动能起点)")
    if len(sell_idx):
        ax2.scatter(sell_idx, df.loc[sell_idx, "DIF"],
                    marker="v", color=C_RED, s=60, zorder=6, label="卖出(动能衰减)")

    # 实际成交垂直线
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
    _style_ax(ax2)

    # ── 子图3：大盘月线 DIF/DEA（牛市判断依据）────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if "DIF_IDX" in df.columns and df["DIF_IDX"].notna().any():
        ax3.plot(df.index, df["DIF_IDX"], color=C_BLUE, lw=1.2, label="大盘月线 DIF")
        ax3.plot(df.index, df["DEA_IDX"], color=C_GOLD, lw=1.2, label="大盘月线 DEA")
        ax3.axhline(0, color=C_MUTED, lw=0.6, linestyle="--")
        # 标出 DIF>0 且 DIF>DEA 的牛市区域
        ax3.fill_between(df.index,
                         df["DIF_IDX"], df["DEA_IDX"],
                         where=(df["DIF_IDX"] > df["DEA_IDX"]) & (df["DIF_IDX"] > 0),
                         interpolate=True, alpha=0.20, color=C_GREEN, label="牛市区间")
    ax3.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED,
               fontsize=7, ncol=3, loc="upper left")
    ax3.set_ylabel("大盘月线 MACD", color=C_FG, fontsize=9)
    _style_ax(ax3)

    # ── 子图4：资产曲线 vs 基准 ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    norm_eq    = eq_df["equity"] / result["initial_capital"] * 100
    norm_bench = eq_df["close"]  / eq_df["close"].iloc[0]  * 100
    ax4.plot(eq_df.index, norm_eq,    color=C_GREEN, lw=1.5, label="策略净值")
    ax4.plot(eq_df.index, norm_bench, color=C_MUTED, lw=1,
             linestyle="--", label="基准(买入持有)")
    ax4.axhline(100, color=C_MUTED, lw=0.5, linestyle=":")
    ax4.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    ax4.set_ylabel("净值（基准=100）", color=C_FG, fontsize=9)
    _style_ax(ax4)

    # ── 子图5：回撤 ──────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.fill_between(eq_df.index, eq_df["drawdown"] * 100, 0,
                     color=C_RED, alpha=0.4, label="策略回撤")
    ax5.set_ylabel("回撤 (%)", color=C_FG, fontsize=9)
    ax5.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    _style_ax(ax5)

    # ── X 轴：只显示最后一张 ─────────────────────────────────────────────────
    for ax in [ax1, ax2, ax3, ax4]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax5.get_xticklabels(), rotation=30, ha="right", color=C_FG, fontsize=8)

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
start_date = 20180101
# end_date 留空则默认使用当天日期
end_date   =

# 大盘指数代码（用于牛市判断，默认 000300 沪深300）
index_symbol = 000300

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

# ── LuMACDBull 策略专属参数 ────────────────────────────────────────────────────

# True  = 红柱缩短即卖出（截陡坡，高手模式）
# False = 等死叉再卖（保守模式）
shrink_exit = true
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("\n" + "─" * 55)
    print("  卢麒元 MACD 牛市动能截取策略回测")
    print("  数据来源：akshare（前复权）")
    print("─" * 55)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "lu_macd_bull_config.ini"
    )
    if not os.path.exists(config_path):
        print(f"  配置文件不存在，已生成默认配置：{config_path}")
        _write_default_config(config_path)

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    s = cfg["backtest"]

    symbol       = s.get("symbol", "600519").strip()
    name         = s.get("name", "stock").strip()
    index_symbol = s.get("index_symbol", "000300").strip()
    start_date   = s.get("start_date", "20180101").strip()
    end_date     = s.get("end_date", "").strip()
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
    shrink_exit     = s.get("shrink_exit", "true").strip().lower() == "true"

    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"  代理：{proxy}")

    file_stem = f"lu_bull_{name}_{symbol}_{end_date}"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_chart = os.path.join(save_dir, file_stem + ".png")
        save_csv   = os.path.join(save_dir, file_stem + ".csv")
    else:
        save_chart = None
        save_csv   = None

    print(f"  股票代码：{symbol}  大盘指数：{index_symbol}  |  {start_date} → {end_date}")
    print(f"  初始资金：{capital:,.0f}  |  止损：{stop_loss}  |  止盈：{take_profit}")
    print(f"  shrink_exit：{shrink_exit}（{'红柱缩短即离场' if shrink_exit else '等死叉再离场'}）")

    try:
        # 获取大盘指数数据（用于月线牛市判断）
        print(f"\n  正在获取大盘指数数据（{index_symbol}）...")
        index_df = fetch_index_data(index_symbol, start_date, end_date)
        if index_df.empty:
            raise ValueError(f"大盘指数 {index_symbol} 数据为空，请检查代码")
        print(f"  大盘指数获取到 {len(index_df)} 个交易日数据")

        inner_strategy = LuMACDBullStrategy(shrink_exit=shrink_exit)
        strategy = _BullStrategyAdapter(inner_strategy, index_df)

        result = run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            initial_capital=capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        plot_bull_backtest(result, save_path=save_chart)

        # 每日状态诊断表（无论是否有交易都保存）
        if save_dir:
            status_csv = os.path.join(save_dir, file_stem + "_daily_status.csv")
            export_daily_status(result, status_csv)

        # 交易记录（有交易时才保存）
        if save_csv and not result["trades"].empty:
            result["trades"].to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"  交易记录已保存至：{save_csv}")
        elif save_csv:
            print("  本次回测无成交记录，不生成交易 CSV")

    except Exception as e:
        print(f"\n  ❌ 回测失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
