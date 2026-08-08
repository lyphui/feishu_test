"""
参数敏感性扫描
==============
在 JCY 推荐股票池上网格遍历策略参数，回答一个问题：
**当前这组参数的表现，是策略本身有效，还是恰好挑中的一个幸运点？**

判读方式（重点看稳健性，不是看最大值）
--------------------------------------
  - 网格上多数格子都为正 → 结论稳健，参数怎么调都不太亏
  - 只有个别格子亮眼、邻居全是负的 → 典型过拟合，换个市场环境就失效
  - 默认参数远离最优区 → 值得调整；默认参数在高原中央 → 保持不动

输出
----
  output/sweep/sweep_results.csv    每个参数组合一行（横截面聚合后的指标）
  output/sweep/sweep_heatmap.png    两个主参数的超额收益热力图

用法
----
    python backtest/param_sweep.py                       # 默认网格
    python backtest/param_sweep.py --limit 20            # 只跑前 20 只，快速试
    python backtest/param_sweep.py --axis expand_bars cross_window
    python backtest/param_sweep.py --axis stop_loss take_profit

注意：网格大小 = 轴1 × 轴2 × 股票数，每格都要跑一次完整回测。
行情数据在进程内缓存，但首次拉取仍然耗时，建议先用 --limit 试跑。
"""

import argparse
import itertools
import os
import sys
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine import run_backtest
from strategies import LuMACDBullStrategy
from lib.plotting import (
    C_BG, C_FG, C_MUTED, setup_matplotlib, style_ax,
)
from lib.market_data import fetch_stock_data, fetch_index_data
from lib.bull_backtest import BullStrategyAdapter
from jcy.lib.common import JSON_PATH, load_candidates

setup_matplotlib()


# ── 可扫描的参数轴 ────────────────────────────────────────────────────────────
# 每个轴声明：候选值列表 + 该参数是喂给策略还是喂给引擎
AXES = {
    "expand_bars":  {"values": [1, 2, 3, 4],              "target": "strategy"},
    "cross_window": {"values": [1, 2, 3, 5, 8],           "target": "strategy"},
    "fast":         {"values": [8, 10, 12, 16],           "target": "strategy"},
    "slow":         {"values": [20, 26, 34],              "target": "strategy"},
    "signal_period": {"values": [6, 9, 12],               "target": "strategy"},
    "shrink_exit":  {"values": [True, False],             "target": "strategy"},
    "stop_loss":    {"values": [0.08, 0.12, 0.20, 0.30],  "target": "engine"},
    "take_profit":  {"values": [0.10, 0.20, 0.30, 0.50],  "target": "engine"},
}

DEFAULTS = {
    "expand_bars": 2, "cross_window": 3,
    "fast": 12, "slow": 26, "signal_period": 9, "shrink_exit": True,
    "stop_loss": 0.20, "take_profit": 0.10,
}


# ── 行情缓存 ──────────────────────────────────────────────────────────────────

_price_cache: dict[tuple, pd.DataFrame] = {}


def _cached_stock(symbol: str, start: str, end: str) -> pd.DataFrame:
    """同一只股票在整个网格里只拉一次行情。"""
    key = (symbol, start, end)
    if key not in _price_cache:
        _price_cache[key] = fetch_stock_data(symbol, start, end)
    return _price_cache[key]


# ── 单个参数组合的评估 ────────────────────────────────────────────────────────

def evaluate_combo(combo: dict, candidates: list[dict], index_df,
                   end_date: str, capital: float,
                   warmup_days: int = 600) -> dict:
    """
    用一组参数把整个股票池跑一遍，返回横截面聚合指标。

    看中位数而不是均值：这类收益分布右偏严重，个别翻倍股会把均值拉得
    很好看，中位数才代表"随手挑一只"的典型结果。
    """
    params = {**DEFAULTS, **combo}
    strategy_kw = {k: v for k, v in params.items() if AXES[k]["target"] == "strategy"}
    engine_kw = {k: v for k, v in params.items() if AXES[k]["target"] == "engine"}

    excess, total, drawdown, trades = [], [], [], []
    failures = 0

    for c in candidates:
        trade_dt = _date(int(c["date"][:4]), int(c["date"][4:6]), int(c["date"][6:]))
        data_start = (trade_dt - timedelta(days=warmup_days)).strftime("%Y%m%d")
        try:
            df = _cached_stock(c["code"], data_start, end_date)
            if df.empty or len(df) < 50:
                failures += 1
                continue
            strategy = BullStrategyAdapter(
                LuMACDBullStrategy(**strategy_kw), index_df,
                trade_start_date=c["date"],
            )
            r = run_backtest(
                symbol=c["code"], start_date=data_start, end_date=end_date,
                strategy=strategy, initial_capital=capital,
                eval_start=c["date"], df=df, **engine_kw,
            )
        except Exception:
            failures += 1
            continue

        excess.append(r["total_return"] - r["benchmark_return"])
        total.append(r["total_return"])
        drawdown.append(r["max_drawdown"])
        trades.append(r["total_trades"])

    if not excess:
        return {**params, "样本数": 0}

    excess_arr = np.array(excess)
    return {
        **params,
        "样本数":        len(excess),
        "失败数":        failures,
        "超额中位数%":   round(float(np.median(excess_arr)), 2),
        "超额均值%":     round(float(excess_arr.mean()), 2),
        "跑赢基准比例":  round(float((excess_arr > 0).mean()), 3),
        "收益中位数%":   round(float(np.median(total)), 2),
        "正收益比例":    round(float((np.array(total) > 0).mean()), 3),
        "平均最大回撤%": round(float(np.mean(drawdown)), 2),
        "平均交易次数":  round(float(np.mean(trades)), 1),
    }


# ── 热力图 ────────────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, axis_x: str, axis_y: str,
                 metric: str, save_path: str) -> None:
    pivot = df.pivot_table(index=axis_y, columns=axis_x, values=metric)
    if pivot.empty:
        print("  网格为空，跳过热力图")
        return

    fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 4,
                                    1.0 * len(pivot.index) + 3),
                           facecolor=C_BG)
    vmax = float(np.nanmax(np.abs(pivot.values))) or 1.0
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    ax.set_xlabel(axis_x, color=C_FG)
    ax.set_ylabel(axis_y, color=C_FG)
    ax.set_title(f"参数敏感性：{metric}\n（整片偏绿=稳健，孤立亮点=过拟合）",
                 color=C_FG, fontsize=12, pad=12)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="#111111", fontsize=9)

    # 默认参数所在格用金框标出，方便判断"当前位置在不在高原上"
    if DEFAULTS.get(axis_x) in list(pivot.columns) and \
       DEFAULTS.get(axis_y) in list(pivot.index):
        jx = list(pivot.columns).index(DEFAULTS[axis_x])
        iy = list(pivot.index).index(DEFAULTS[axis_y])
        ax.add_patch(plt.Rectangle((jx - 0.5, iy - 0.5), 1, 1, fill=False,
                                   edgecolor="#e3b341", lw=2.5))

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=C_FG)
    ax.tick_params(colors=C_FG)
    for sp in ax.spines.values():
        sp.set_color(C_MUTED)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  热力图已保存至：{save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="策略参数敏感性扫描")
    p.add_argument("--axis", nargs=2, default=["expand_bars", "cross_window"],
                   metavar=("X", "Y"),
                   help=f"扫描的两个参数轴，可选：{list(AXES)}")
    p.add_argument("--metric", default="超额中位数%",
                   help="热力图展示的指标，默认 超额中位数%%")
    p.add_argument("--limit", type=int, default=0,
                   help="只取前 N 只候选股（快速试跑），0=全部")
    p.add_argument("--capital", type=float, default=100000)
    p.add_argument("--index", type=str, default="000300")
    p.add_argument("--output", type=str, default="output/sweep")
    return p.parse_args()


def main():
    args = parse_args()
    axis_x, axis_y = args.axis
    for a in (axis_x, axis_y):
        if a not in AXES:
            print(f"  ❌ 未知参数轴：{a}，可选 {list(AXES)}")
            sys.exit(1)

    if not os.path.exists(JSON_PATH):
        print(f"  ❌ 找不到 JSON 文件：{JSON_PATH}")
        sys.exit(1)
    candidates = load_candidates(JSON_PATH)
    if args.limit:
        candidates = candidates[:args.limit]
    if not candidates:
        print("  ❌ 未找到候选股")
        sys.exit(1)

    end_date = _date.today().strftime("%Y%m%d")
    earliest = min(c["date"] for c in candidates)
    index_start = (_date(int(earliest[:4]), int(earliest[4:6]), int(earliest[6:]))
                   - timedelta(days=600)).strftime("%Y%m%d")

    grid = list(itertools.product(AXES[axis_x]["values"], AXES[axis_y]["values"]))
    print("\n" + "─" * 66)
    print(f"  参数敏感性扫描：{axis_x} × {axis_y}")
    print(f"  候选股 {len(candidates)} 只 × 网格 {len(grid)} 格 "
          f"= {len(candidates) * len(grid)} 次回测")
    print(f"  其余参数固定为默认值：{DEFAULTS}")
    print("─" * 66)

    print(f"\n  获取大盘指数 {args.index}（{index_start} → {end_date}）...")
    index_df = fetch_index_data(args.index, index_start, end_date)

    os.makedirs(args.output, exist_ok=True)
    rows = []
    for n, (vx, vy) in enumerate(grid, 1):
        combo = {axis_x: vx, axis_y: vy}
        print(f"\n  [{n}/{len(grid)}] {axis_x}={vx}  {axis_y}={vy}")
        row = evaluate_combo(combo, candidates, index_df, end_date, args.capital)
        rows.append(row)
        if row.get("样本数"):
            print(f"      超额中位数 {row['超额中位数%']:+.2f}%  "
                  f"跑赢基准 {row['跑赢基准比例']:.1%}  "
                  f"平均回撤 {row['平均最大回撤%']:.2f}%")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, "sweep_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  扫描结果已保存至：{csv_path}")

    if args.metric in df.columns and df["样本数"].gt(0).any():
        valid = df[df["样本数"] > 0]
        best = valid.loc[valid[args.metric].idxmax()]
        default_row = valid[
            (valid[axis_x] == DEFAULTS[axis_x]) & (valid[axis_y] == DEFAULTS[axis_y])
        ]
        print(f"\n  最优格：{axis_x}={best[axis_x]}  {axis_y}={best[axis_y]}  "
              f"{args.metric}={best[args.metric]:+.2f}")
        if not default_row.empty:
            d = default_row.iloc[0]
            print(f"  默认格：{axis_x}={d[axis_x]}  {axis_y}={d[axis_y]}  "
                  f"{args.metric}={d[args.metric]:+.2f}")
        positive = (valid[args.metric] > 0).mean()
        print(f"  网格中 {positive:.0%} 的格子为正 —— "
              f"{'结论较稳健' if positive > 0.6 else '稳健性存疑，警惕过拟合'}")

        plot_heatmap(valid, axis_x, axis_y, args.metric,
                     os.path.join(args.output, "sweep_heatmap.png"))


if __name__ == "__main__":
    main()
