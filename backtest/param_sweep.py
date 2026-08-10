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

样本外验证（--oos-frac）
------------------------
上面的网格是**纯样本内**的：在同一批股票、同一段时间上遍历参数再挑最好的，
只能说明"参数高原平不平坦"，无法排除整个策略在这段数据上过拟合。

--oos-frac 0.3 会按推荐日把候选股切成两段：较早的 70% 用来选参数（IS），
最近的 30% 只用来验证（OOS）。选参数时完全看不到 OOS 那批票，因此
"IS 最优参数在 OOS 上的表现"才是这个策略能不能上实盘的真实估计。

判读：
  - IS 最优在 OOS 上仍为正、且不明显差于默认参数 → 参数选择是可迁移的
  - IS 最优在 OOS 上转负 → 网格搜索选中的是噪声，别用这组参数
  - 默认参数在 OOS 上比 IS 最优还好 → 更说明最优格是拟合出来的

注意这是**按标的**的时间切分（较晚被推荐的票留作验证），不是单只票内部的
时间切分——每只票的窗口本来就各不相同，只有按推荐日切才是真正的时序留出。

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
    python backtest/param_sweep.py --oos-frac 0.3        # 留最近 30% 做样本外

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
from config import index_history_start
from strategies import LuMACDBullStrategy
from lib.plotting import (
    C_BG, C_FG, C_MUTED, setup_matplotlib, style_ax,
)
from lib.market_data import fetch_stock_data, fetch_index_data
from lib.bull_backtest import BullStrategyAdapter
from jcy.lib.common import (JSON_PATH, LONG_RATINGS, load_candidates,
                            parse_ratings)

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
    "stop_loss":    {"values": [None, 0.08, 0.10, 0.15, 0.20],  "target": "engine"},
    # None = 不设止盈，完全由 shrink_exit 的动能衰减规则离场。
    # 这是默认值：本策略要"截取最陡峭的部分"，固定比例止盈会把陡坡提前切断。
    "take_profit":  {"values": [None, 0.10, 0.20, 0.30, 0.50], "target": "engine"},
}

# evaluate_combo 在有样本时产出的全部聚合指标，也是 --metric 的合法取值。
#
# 主指标是**日均超额 bp**，不是总超额%：各股统计窗口从几十到几百个交易日不等
# （batch_report.py 的 docstring 论证过这一点），把持有两年的 +40% 和持有两个月的
# +40% 放进同一个中位数，比的是"谁的窗口更长"。用一个自己判定为不可比的指标
# 去选参数，选出来的就是"哪组参数恰好被长窗口标的占了多数"。
METRICS = [
    "日均超额中位bp", "日均超额均值bp", "跑赢基准比例",
    "日均收益中位bp", "正收益比例", "平均最大回撤%", "平均交易次数",
    "平均在场比例%",
    # 保留总超额仅供对照，不建议用来选参数（窗口长度不可比）
    "超额中位数%",
]

# 必须与 backtest/jcy_macd_bull_batch.py 的 CLI 默认值保持一致，
# 否则热力图上标出的"默认格"不是实际在跑的那组参数。
DEFAULTS = {
    "expand_bars": 2, "cross_window": 3,
    "fast": 12, "slow": 26, "signal_period": 9, "shrink_exit": True,
    "stop_loss": 0.10, "take_profit": None,
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
    daily_excess, daily_total, exposure = [], [], []
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

        exc = r["total_return"] - r["benchmark_return"]
        excess.append(exc)
        total.append(r["total_return"])
        drawdown.append(r["max_drawdown"])
        trades.append(r["total_trades"])

        # 按统计窗口长度线性归一（与 batch_report 的「日均超额bp」同口径）
        n_days = len(r["equity_curve"])
        if n_days:
            daily_excess.append(exc * 100 / n_days)
            daily_total.append(r["total_return"] * 100 / n_days)
        if r.get("exposure_pct") is not None:
            exposure.append(r["exposure_pct"])

    if not excess:
        return {**params, "样本数": 0}

    excess_arr = np.array(excess)
    d_exc = np.array(daily_excess) if daily_excess else np.array([np.nan])
    return {
        **params,
        "样本数":        len(excess),
        "失败数":        failures,
        "日均超额中位bp": round(float(np.median(d_exc)), 2),
        "日均超额均值bp": round(float(np.mean(d_exc)), 2),
        "跑赢基准比例":  round(float((excess_arr > 0).mean()), 3),
        "日均收益中位bp": round(float(np.median(daily_total)), 2) if daily_total else None,
        "正收益比例":    round(float((np.array(total) > 0).mean()), 3),
        "平均最大回撤%": round(float(np.mean(drawdown)), 2),
        "平均交易次数":  round(float(np.mean(trades)), 1),
        "平均在场比例%": round(float(np.mean(exposure)), 1) if exposure else None,
        # 仅供对照：窗口长度不可比，不要用它选参数
        "超额中位数%":   round(float(np.median(excess_arr)), 2),
    }


# ── 热力图 ────────────────────────────────────────────────────────────────────

def fmt_value(v) -> str:
    """参数取值的显示文本。None 表示"该项不启用"，不能显示成空白。"""
    if v is None:
        return "关闭"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_matrix(cells: dict, axis_x: str, axis_y: str) -> tuple:
    """
    按 AXES 的**声明顺序**排列网格，返回 (matrix, x_values, y_values)。

    不走 df.pivot_table：pivot 会把 None 当成 NaN 直接丢掉整行整列
    （"不设止盈"恰恰是默认值，丢了就看不到默认格），而且它按值排序，
    True/False 这类轴的顺序会变得没有意义。
    """
    xs = [v for v in AXES[axis_x]["values"] if any(k[0] == v for k in cells)]
    ys = [v for v in AXES[axis_y]["values"] if any(k[1] == v for k in cells)]
    matrix = np.full((len(ys), len(xs)), np.nan)
    for i, vy in enumerate(ys):
        for j, vx in enumerate(xs):
            val = cells.get((vx, vy))
            if val is not None:
                matrix[i, j] = val
    return matrix, xs, ys


def plot_heatmap(cells: dict, axis_x: str, axis_y: str,
                 metric: str, save_path: str) -> None:
    """cells: {(x取值, y取值): 指标值}"""
    matrix, xs, ys = build_matrix(cells, axis_x, axis_y)
    if matrix.size == 0 or not np.isfinite(matrix).any():
        print("  网格为空，跳过热力图")
        return

    fig, ax = plt.subplots(figsize=(1.6 * len(xs) + 4, 1.0 * len(ys) + 3),
                           facecolor=C_BG)
    vmax = float(np.nanmax(np.abs(matrix))) or 1.0
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(xs)), [fmt_value(v) for v in xs])
    ax.set_yticks(range(len(ys)), [fmt_value(v) for v in ys])
    ax.set_xlabel(axis_x, color=C_FG)
    ax.set_ylabel(axis_y, color=C_FG)
    ax.set_title(f"参数敏感性：{metric}\n（整片偏绿=稳健，孤立亮点=过拟合）",
                 color=C_FG, fontsize=12, pad=12)

    for i in range(len(ys)):
        for j in range(len(xs)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="#111111", fontsize=9)

    # 默认参数所在格用金框标出，方便判断"当前位置在不在高原上"
    if DEFAULTS.get(axis_x) in xs and DEFAULTS.get(axis_y) in ys:
        jx, iy = xs.index(DEFAULTS[axis_x]), ys.index(DEFAULTS[axis_y])
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

def split_candidates(candidates: list[dict],
                     oos_frac: float) -> tuple[list[dict], list[dict]]:
    """
    按推荐日切成 (样本内, 样本外)。candidates 必须已按 date 升序排好。

    切分点落在日期边界上：同一天被推荐的股票不会被拆到两边，否则同一篇研报
    的标的会同时出现在选参数集和验证集里，样本外就不再干净了。
    """
    if oos_frac <= 0 or len(candidates) < 4:
        return candidates, []

    n_is = int(len(candidates) * (1 - oos_frac))
    if n_is <= 0 or n_is >= len(candidates):
        return candidates, []

    # 把切点推到下一个不同日期，保证同日标的不跨界
    split_date = candidates[n_is]["date"]
    is_cands = [c for c in candidates if c["date"] < split_date]
    oos_cands = [c for c in candidates if c["date"] >= split_date]
    if not is_cands or not oos_cands:
        return candidates, []
    return is_cands, oos_cands


def parse_args():
    p = argparse.ArgumentParser(description="策略参数敏感性扫描")
    p.add_argument("--axis", nargs=2, default=["expand_bars", "cross_window"],
                   metavar=("X", "Y"),
                   help=f"扫描的两个参数轴，可选：{list(AXES)}")
    # 用 choices 卡死：跑完整个网格才因为拼错指标名 KeyError，代价是几十分钟。
    p.add_argument("--metric", default="日均超额中位bp", choices=METRICS,
                   help="热力图展示的指标，默认 日均超额中位bp"
                        "（对窗口长度归一，跨标的可比）")
    p.add_argument("--ratings", type=parse_ratings, default=LONG_RATINGS,
                   help=f"进入扫描的评级，逗号分隔，默认 {','.join(LONG_RATINGS)}")
    p.add_argument("--limit", type=int, default=0,
                   help="只取前 N 只候选股（快速试跑），0=全部")
    p.add_argument("--oos-frac", type=float, default=0.0, metavar="FRAC",
                   help="按推荐日留出最近 FRAC 比例的候选股做样本外验证，"
                        "如 0.3；0=关闭（纯样本内，结论不足以支撑实盘）")
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
    candidates = load_candidates(JSON_PATH, ratings=args.ratings)
    if args.limit:
        candidates = candidates[:args.limit]
    if not candidates:
        print("  ❌ 未找到候选股")
        sys.exit(1)

    # 按推荐日排序后切分：较早的选参数，较晚的只用于验证。
    candidates = sorted(candidates, key=lambda c: c["date"])
    is_cands, oos_cands = split_candidates(candidates, args.oos_frac)
    if args.oos_frac > 0 and not oos_cands:
        print(f"  ⚠️ 候选股仅 {len(candidates)} 只，按 {args.oos_frac:.0%} 切不出"
              "样本外集合，本次退化为纯样本内扫描")

    end_date = _date.today().strftime("%Y%m%d")
    # 指数起点为绝对日期，不随候选池变化（月线 MACD 预热，见 config.py）
    index_start = index_history_start()

    grid = list(itertools.product(AXES[axis_x]["values"], AXES[axis_y]["values"]))
    print("\n" + "─" * 66)
    print(f"  参数敏感性扫描：{axis_x} × {axis_y}")
    if oos_cands:
        print(f"  样本内 {len(is_cands)} 只（推荐日 ≤ {is_cands[-1]['date']}）"
              f"  |  样本外 {len(oos_cands)} 只（≥ {oos_cands[0]['date']}）")
        print(f"  网格只在样本内跑：{len(is_cands)} × {len(grid)} "
              f"= {len(is_cands) * len(grid)} 次回测")
    else:
        print(f"  候选股 {len(is_cands)} 只 × 网格 {len(grid)} 格 "
              f"= {len(is_cands) * len(grid)} 次回测（纯样本内）")
    print(f"  其余参数固定为默认值：{DEFAULTS}")
    print("─" * 66)

    print(f"\n  获取大盘指数 {args.index}（{index_start} → {end_date}）...")
    index_df = fetch_index_data(args.index, index_start, end_date)

    os.makedirs(args.output, exist_ok=True)
    rows, cells = [], {}
    for n, (vx, vy) in enumerate(grid, 1):
        combo = {axis_x: vx, axis_y: vy}
        print(f"\n  [{n}/{len(grid)}] {axis_x}={fmt_value(vx)}  "
              f"{axis_y}={fmt_value(vy)}")
        row = evaluate_combo(combo, is_cands, index_df, end_date, args.capital)
        rows.append(row)
        if row.get("样本数"):
            cells[(vx, vy)] = row[args.metric]
            print(f"      日均超额中位 {row['日均超额中位bp']:+.2f}bp  "
                  f"跑赢基准 {row['跑赢基准比例']:.1%}  "
                  f"平均回撤 {row['平均最大回撤%']:.2f}%  "
                  f"在场 {row['平均在场比例%'] or 0:.0f}%")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, "sweep_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  扫描结果已保存至：{csv_path}")

    if not cells:
        print("  所有参数组合都没有有效样本，无法给出结论")
        return

    best_key = max(cells, key=lambda k: cells[k])
    default_key = (DEFAULTS[axis_x], DEFAULTS[axis_y])
    print(f"\n  最优格：{axis_x}={fmt_value(best_key[0])}  "
          f"{axis_y}={fmt_value(best_key[1])}  "
          f"{args.metric}={cells[best_key]:+.2f}")
    if default_key in cells:
        print(f"  默认格：{axis_x}={fmt_value(default_key[0])}  "
              f"{axis_y}={fmt_value(default_key[1])}  "
              f"{args.metric}={cells[default_key]:+.2f}")
    positive = float(np.mean([v > 0 for v in cells.values()]))
    print(f"  网格中 {positive:.0%} 的格子为正 —— "
          f"{'结论较稳健' if positive > 0.6 else '稳健性存疑，警惕过拟合'}")

    plot_heatmap(cells, axis_x, axis_y, args.metric,
                 os.path.join(args.output, "sweep_heatmap.png"))

    if oos_cands:
        run_oos_check(best_key, default_key, axis_x, axis_y, oos_cands,
                      index_df, end_date, args.capital, args.metric,
                      cells.get(best_key), cells.get(default_key))
    else:
        print("\n  ⚠️ 本次为纯样本内扫描：最优格是在同一批数据上挑出来的，"
              "不能作为实盘依据。\n     加 --oos-frac 0.3 做时序留出验证。")


def run_oos_check(best_key, default_key, axis_x: str, axis_y: str,
                  oos_cands: list[dict], index_df, end_date: str,
                  capital: float, metric: str,
                  best_is, default_is) -> None:
    """
    把样本内选出的最优参数，拿到从未参与选参的样本外标的上重跑一遍。

    这是整个扫描里唯一有外部效度的一步：网格上的最大值必然是正的（挑出来的），
    只有它在没见过的数据上仍然成立，才说明参数选择可迁移而不是拟合噪声。
    """
    print("\n" + "═" * 66)
    print(f"  样本外验证（{len(oos_cands)} 只，推荐日 ≥ {oos_cands[0]['date']}）")
    print("═" * 66)

    def _run(key, label):
        combo = {axis_x: key[0], axis_y: key[1]}
        row = evaluate_combo(combo, oos_cands, index_df, end_date, capital)
        if not row.get("样本数"):
            print(f"  {label}：无有效样本")
            return None
        print(f"  {label}：{axis_x}={fmt_value(key[0])} "
              f"{axis_y}={fmt_value(key[1])}  "
              f"{metric}={row[metric]:+.2f}  "
              f"跑赢基准 {row['跑赢基准比例']:.1%}  "
              f"样本 {row['样本数']} 只")
        return row[metric]

    best_oos = _run(best_key, "样本内最优参数")
    default_oos = (best_oos if best_key == default_key
                   else _run(default_key, "默认参数    "))

    if best_oos is None:
        return

    if best_is is not None:
        decay = best_is - best_oos
        print(f"\n  最优参数：样本内 {best_is:+.2f} → 样本外 {best_oos:+.2f}"
              f"（衰减 {decay:+.2f}）")

    if best_oos <= 0:
        print("  ❌ 样本内最优参数在样本外转负 —— 网格搜索选中的是噪声，"
              "不要采用这组参数。")
    elif default_oos is not None and default_oos > best_oos:
        print("  ⚠️ 默认参数在样本外反而更好 —— 更说明最优格是拟合出来的，"
              "保持默认值。")
    else:
        print("  ✅ 样本内最优参数在样本外仍为正且不劣于默认值 —— "
              "参数选择具备一定可迁移性。")
    print("  提醒：样本外只有一次，通过不等于策略有效；反复调参后重跑，"
          "这个集合也就不再是样本外了。")
    print("═" * 66)


if __name__ == "__main__":
    main()
