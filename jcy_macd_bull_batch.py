"""
JCY 增持股票批量回测 —— 卢麒元 MACD 牛市动能截取策略
=======================================================
从 data/jcy/jcy_insights.json 读取研报数据，筛选满足以下条件的股票：
  - rating == "增持"
  - code 为 6 位纯数字的 A 股代码

同一股票多次出现时，保留日期最早的那条记录。

回测参数
--------
  --stop_loss   止损比例，默认 0.20
  --take_profit 止盈比例，默认 0.10
  --capital     初始资金，默认 100000
  --index       大盘指数代码，默认 000300（沪深300）
  --shrink_exit 红柱缩短即离场，默认 True

数据与买入逻辑
--------------
  - 数据起始 = JSON 推荐日期往前推 365 天（让 MACD 充分预热，避免初始失真）
  - JSON 推荐日期之前：所有买入和卖出信号全部清零，不发生任何操作
  - JSON 推荐日期当天及之后：买入、卖出、止损、止盈均正常执行

输出目录结构
------------
  output/
    jcy_{股票代码}_{股票名称}_{推荐日期}/
      lu_bull_{股票名称}_{股票代码}_{结束日期}.png
      lu_bull_{股票名称}_{股票代码}_{结束日期}.csv          # 交易记录
      lu_bull_{股票名称}_{股票代码}_{结束日期}_daily_status.csv

用法示例
--------
    python jcy_macd_bull_batch.py
    python jcy_macd_bull_batch.py --stop_loss 0.15 --take_profit 0.12
"""

import argparse
import json
import os
import re
import sys
from datetime import date as _date, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 复用数据获取和回测引擎
from macd_analysis import fetch_stock_data, run_backtest
from strategies import LuMACDBullStrategy
from strategies.base import BaseStrategy

# ── 字体 & 颜色 ──────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "STHeiti", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

C_BG    = "#0d1117"
C_FG    = "#e6edf3"
C_GREEN = "#39d353"
C_RED   = "#f85149"
C_BLUE  = "#58a6ff"
C_GOLD  = "#e3b341"
C_MUTED = "#484f58"

JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "jcy", "jcy_insights.json",
)


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def is_ashare_code(code) -> bool:
    """判断是否为 6 位纯数字的 A 股代码。"""
    return bool(code and re.fullmatch(r"\d{6}", str(code)))


def load_candidates(json_path: str) -> list[dict]:
    """
    从 JSON 文件中筛选增持 A 股，去重后返回候选列表。

    返回格式：[{"code": ..., "name": ..., "date": "YYYYMMDD", "reason": ...}, ...]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 按日期升序排列文章，保证 earliest 记录能被正确保留
    articles = sorted(data.get("articles", []), key=lambda a: a.get("date", ""))

    seen: dict[str, dict] = {}  # code -> first record
    for article in articles:
        article_date = article.get("date", "")
        for company in article.get("companies", []):
            code   = company.get("code")
            name   = company.get("name", "")
            rating = company.get("rating", "")
            reason = company.get("rating_reason", "")

            if rating != "增持":
                continue
            if not is_ashare_code(code):
                continue

            # 日期转 YYYYMMDD
            date_str = article_date.replace("-", "")

            if code not in seen:
                seen[code] = {
                    "code":   code,
                    "name":   name,
                    "date":   date_str,
                    "reason": reason,
                }

    return list(seen.values())


# ── 指数数据 ─────────────────────────────────────────────────────────────────

def fetch_index_data(symbol: str, start_date: str, end_date: str):
    """获取大盘指数日线数据，akshare 优先，yfinance 备用。"""
    import pandas as pd

    try:
        import akshare as ak
        prefix    = "sz" if symbol.startswith("399") else "sh"
        ak_symbol = prefix + symbol
        print(f"    正在从 akshare 获取指数 {ak_symbol} 数据...")
        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        df = df.rename(columns={"date": "date", "close": "close",
                                 "open": "open", "high": "high",
                                 "low": "low", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        start = pd.to_datetime(start_date, format="%Y%m%d")
        end   = pd.to_datetime(end_date,   format="%Y%m%d")
        df = df.loc[start:end]
        if not df.empty:
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols]
    except Exception as e:
        print(f"    akshare 指数获取失败：{e}，尝试 yfinance 备用...")

    import yfinance as yf
    suffix    = ".SZ" if symbol.startswith("399") else ".SS"
    ticker    = symbol + suffix
    start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_str   = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    print(f"    正在从 yfinance 获取指数 {ticker} 数据...")
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


# ── 策略适配器 ────────────────────────────────────────────────────────────────

class _BullStrategyAdapter(BaseStrategy):
    """
    将 LuMACDBullStrategy 的双参数 prepare 适配为单参数接口。

    trade_start_date : str, "YYYYMMDD"
        仅在该日期（含）之后才允许任何交易信号生效。
        之前的买入和卖出信号均被清零，不发生任何操作。
        用途：让 MACD 在推荐日期前有足够的历史数据充分预热。
    """

    def __init__(self, inner: LuMACDBullStrategy, index_df,
                 trade_start_date: str | None = None):
        self._inner            = inner
        self._index_df         = index_df
        self._trade_start_date = trade_start_date  # "YYYYMMDD" or None

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def params(self) -> dict:
        return self._inner.params

    def prepare(self, df):
        import pandas as pd
        result = self._inner.prepare(df, self._index_df)
        if self._trade_start_date:
            cutoff = pd.to_datetime(self._trade_start_date, format="%Y%m%d")
            # 推荐日期之前一律不交易（买入和卖出信号全部清零）
            result.loc[result.index < cutoff, "signal"] = 0
        return result

    def plot_indicators(self, ax, df, colors):
        return self._inner.plot_indicators(ax, df, colors)


# ── 每日状态导出 ──────────────────────────────────────────────────────────────

def export_daily_status(result: dict, save_path: str) -> None:
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

        idx_above0 = dif_idx > 0
        idx_dif_up = dif_idx > dea_idx
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
            "日期":        date.strftime("%Y-%m-%d"),
            "收盘价":      round(r["close"], 2),
            "大盘_DIF":    round(dif_idx, 4),
            "大盘_DEA":    round(dea_idx, 4),
            "大盘_DIF>0":  idx_above0,
            "大盘_DIF>DEA": idx_dif_up,
            "牛市确认":    bull,
            "个股_DIF":    round(dif,  4),
            "个股_DEA":    round(dea,  4),
            "个股_MACD柱": round(macd, 4),
            "红柱拉长":    expanding,
            "红柱缩短":    shrinking,
            "信号":        signal,
            "判断依据":    blocking,
        })

    status_df = __import__("pandas").DataFrame(rows)
    status_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"    每日状态表已保存至：{save_path}")


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.tick_params(colors=C_FG, labelsize=8)
    ax.spines[:].set_color(C_MUTED)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.yaxis.label.set_color(C_FG)
    ax.grid(color=C_MUTED, linewidth=0.3, alpha=0.5)


def plot_bull_backtest(result: dict, save_path: "str | None" = None,
                       trade_start_date: str | None = None):
    """
    trade_start_date : str "YYYYMMDD"
        JSON 推荐日期，在图表上用金色竖线标注"买入开放线"。
    """
    import pandas as pd
    df     = result["df"]
    eq_df  = result["equity_curve"]
    trades = result["trades"]
    symbol = result["symbol"]

    fig = plt.figure(figsize=(18, 18), facecolor=C_BG)
    gs  = GridSpec(5, 1, figure=fig, hspace=0.06,
                   height_ratios=[3, 2, 1.5, 1.5, 1])

    # 子图1：价格 + 买卖点
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

    # 推荐日期竖线（金色）标注买入开放时间
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
    _style_ax(ax1)

    # 子图2：MACD + 牛市背景
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

    import pandas as _pd
    expanding_mask = df.get("hist_expanding", _pd.Series(False, index=df.index))
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
    _style_ax(ax2)

    # 子图3：大盘月线 DIF/DEA
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
    _style_ax(ax3)

    # 子图4：资产曲线
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

    # 子图5：回撤
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.fill_between(eq_df.index, eq_df["drawdown"] * 100, 0,
                     color=C_RED, alpha=0.4, label="策略回撤")
    ax5.set_ylabel("回撤 (%)", color=C_FG, fontsize=9)
    ax5.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    _style_ax(ax5)

    for ax in [ax1, ax2, ax3, ax4]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax5.get_xticklabels(), rotation=30, ha="right", color=C_FG, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"    图表已保存至：{save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── 单只股票回测 ──────────────────────────────────────────────────────────────

def backtest_one(candidate: dict, end_date: str, index_symbol: str,
                 capital: float, stop_loss: float, take_profit: float,
                 shrink_exit: bool, base_output_dir: str,
                 warmup_days: int = 365) -> bool:
    """
    对单只股票执行回测并保存结果。
    返回 True 表示成功，False 表示失败。

    warmup_days : int
        在 JSON 推荐日期前额外取多少天的历史数据，用于 MACD 预热。
        默认 365 天（约 250 个交易日，足以让 EMA-26 充分稳定）。
    """
    code             = candidate["code"]
    name             = candidate["name"]
    trade_start_date = candidate["date"]   # JSON 推荐日期，YYYYMMDD
    reason           = candidate["reason"]

    # 数据起始往前推 warmup_days 天，保证 MACD 稳定
    trade_dt   = _date(int(trade_start_date[:4]),
                       int(trade_start_date[4:6]),
                       int(trade_start_date[6:]))
    data_start = (trade_dt - timedelta(days=warmup_days)).strftime("%Y%m%d")

    # 安全化名称（用于文件/目录名）
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", name)

    # 子目录：jcy_{code}_{name}_{推荐日期}
    sub_dir   = f"jcy_{code}_{safe_name}_{trade_start_date}"
    save_dir  = os.path.join(base_output_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    file_stem  = f"lu_bull_{safe_name}_{code}_{end_date}"
    save_chart = os.path.join(save_dir, file_stem + ".png")
    save_csv   = os.path.join(save_dir, file_stem + ".csv")
    status_csv = os.path.join(save_dir, file_stem + "_daily_status.csv")

    print(f"\n  [{code}] {name}  |  推荐日期：{trade_start_date}  "
          f"数据起始：{data_start}  止损：{stop_loss}  止盈：{take_profit}")
    print(f"    推荐原因：{reason}")

    try:
        print(f"    获取大盘指数 {index_symbol} 数据（{data_start} → {end_date}）...")
        index_df = fetch_index_data(index_symbol, data_start, end_date)
        if index_df.empty:
            raise ValueError(f"大盘指数数据为空")

        inner_strategy = LuMACDBullStrategy(shrink_exit=shrink_exit)
        strategy       = _BullStrategyAdapter(inner_strategy, index_df,
                                              trade_start_date=trade_start_date)

        result = run_backtest(
            symbol=code,
            start_date=data_start,        # 含预热期，保证 MACD 稳定
            end_date=end_date,
            strategy=strategy,
            initial_capital=capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        print(f"    总收益：{result['total_return']:+.2f}%  "
              f"基准：{result['benchmark_return']:+.2f}%  "
              f"夏普：{result['sharpe_ratio']:.2f}  "
              f"最大回撤：{result['max_drawdown']:.2f}%")

        plot_bull_backtest(result, save_path=save_chart,
                           trade_start_date=trade_start_date)
        export_daily_status(result, status_csv)

        if not result["trades"].empty:
            result["trades"].to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"    交易记录已保存至：{save_csv}")
        else:
            print("    本次回测无成交记录")

        return True

    except Exception as e:
        print(f"    ❌ 回测失败：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── 主入口 ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="JCY 增持股票批量回测（卢麒元 MACD 牛市动能截取策略）"
    )
    parser.add_argument("--stop_loss",   type=float, default=0.20,
                        help="止损比例，默认 0.20")
    parser.add_argument("--take_profit", type=float, default=0.10,
                        help="止盈比例，默认 0.10")
    parser.add_argument("--capital",     type=float, default=100000,
                        help="初始资金，默认 100000")
    parser.add_argument("--index",       type=str,   default="000300",
                        help="大盘指数代码，默认 000300（沪深300）")
    parser.add_argument("--shrink_exit", type=lambda x: x.lower() != "false",
                        default=True,
                        help="红柱缩短即离场，默认 True；传 false 则等死叉")
    parser.add_argument("--output",      type=str,   default="output",
                        help="输出根目录，默认 output/")
    return parser.parse_args()


def main():
    args = parse_args()

    end_date = _date.today().strftime("%Y%m%d")

    print("\n" + "─" * 60)
    print("  JCY 增持股票批量回测 —— 卢麒元 MACD 牛市动能截取策略")
    print("─" * 60)
    print(f"  数据来源：{JSON_PATH}")
    print(f"  止损：{args.stop_loss}  止盈：{args.take_profit}  "
          f"资金：{args.capital:,.0f}  大盘：{args.index}")
    print(f"  结束日期：{end_date}  输出目录：{args.output}/")
    print("─" * 60)

    # 加载候选股票
    if not os.path.exists(JSON_PATH):
        print(f"  ❌ 找不到 JSON 文件：{JSON_PATH}")
        sys.exit(1)

    candidates = load_candidates(JSON_PATH)
    if not candidates:
        print("  ❌ 未找到满足条件的增持 A 股，请检查 JSON 数据")
        sys.exit(1)

    print(f"\n  共找到 {len(candidates)} 只增持 A 股（已去重，保留最早记录）：")
    for c in candidates:
        print(f"    {c['code']}  {c['name']:8s}  起始日期：{c['date'][:4]}-{c['date'][4:6]}-{c['date'][6:]}")

    os.makedirs(args.output, exist_ok=True)

    # 逐只回测
    success_count = 0
    fail_count    = 0
    for candidate in candidates:
        ok = backtest_one(
            candidate     = candidate,
            end_date      = end_date,
            index_symbol  = args.index,
            capital       = args.capital,
            stop_loss     = args.stop_loss,
            take_profit   = args.take_profit,
            shrink_exit   = args.shrink_exit,
            base_output_dir = args.output,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "─" * 60)
    print(f"  批量回测完成：成功 {success_count} 只，失败 {fail_count} 只")
    print(f"  结果已保存至：{os.path.abspath(args.output)}/")
    print("─" * 60)


if __name__ == "__main__":
    main()
