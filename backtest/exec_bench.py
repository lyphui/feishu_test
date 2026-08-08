"""
日内下单方案实测台（买/卖两侧，按股票池分别出数）
================================================
把 `lib/execution.py` 的度量层跑在真实分时数据上，产出「哪种下单方式成交价更好」
的实测表。**度量衡共享，读数不共享**——每个池子各跑各的，结论各写各的文档。

    python backtest/exec_bench.py --universe oil --side both
    python backtest/exec_bench.py --universe jcy --side sell --limit 45
    python backtest/exec_bench.py --universe oil --offline        # 只读本地缓存

为什么要有这个脚本而不是临时脚本
--------------------------------
这些 bp 数字会被写进 `jcy_intraday_timing.py` 的 docstring、CLAUDE.md 和 changelog。
写进文档的数字必须随时能重跑验证，否则过几个月没人说得清它们是怎么来的。
分时行情缓存在 `data/market/intraday/`（不复权，见 `lib/intraday_store.py`）。

买卖两侧的关系
--------------
`open` / `close` 这类**固定时点**方案，两侧的原始 bp 是同一个数——某天开盘价
比 VWAP 低 18bp 就是低 18bp，与你买还是卖无关。变的只是好坏：买入便宜是省钱，
卖出便宜是亏钱。所以卖侧那两行的「优势bp」正好是买侧的相反数，不是新信息。

真正需要卖侧单独测的只有两类**依赖侧向定义**的方案：
  * GO 窗口：买侧看红柱拉长 + DIF>DEA；卖侧看红柱缩短 **或** 死叉，条件宽得多
  * 限价单：买侧挂低价等回调（`rest_low` 触发）；卖侧挂高价等冲高（`rest_high` 触发）
"""

import argparse
import os
import random
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):      # backtest/ 与仓库根都要在 path 上
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.execution import (add_limit_plan, benchmark, daily_panel, intraday_macd,
                           split_by_go, wait_value)
from lib.intraday_store import load_intraday
from jcy.lib.common import is_ashare_code, load_candidates

# JCY 池抽样的固定种子——换了种子就是换了样本，文档里的数字会对不上
SAMPLE_SEED = 20260809
OIL_SYMBOLS = {"601857": "中国石油", "600938": "中国海油"}

DEFAULT_START = "20220101"
LIMIT_OFFSET = 0.005            # 限价单相对开盘价的偏移幅度（买侧取负）

LABELS = {
    "open": "A 开盘集合竞价",
    "close": "B 尾盘 15:00",
    "go_price": "C GO 窗口",
}


def jcy_universe(limit: int | None) -> dict[str, str]:
    """从 jcy_insights.json 取 A 股推荐票，去重后**定种子**抽样。"""
    pool: dict[str, str] = {}
    for c in load_candidates():
        code = str(c.get("code", "")).strip()
        if is_ashare_code(code) and code not in pool:
            pool[code] = c.get("name", "")
    codes = sorted(pool)
    if limit and limit < len(codes):
        codes = sorted(random.Random(SAMPLE_SEED).sample(codes, limit))
    return {c: pool[c] for c in codes}


def build_universe(name: str, limit: int | None) -> dict[str, str]:
    if name == "oil":
        return dict(OIL_SYMBOLS)
    if name == "jcy":
        return jcy_universe(limit)
    raise ValueError(f"未知股票池 {name!r}")


def panel_for(symbol: str, start: str, end: str | None, period: int,
              side: str, offline: bool) -> pd.DataFrame:
    """取一只票的分时 → MACD → 每日一行面板，并挂上限价单方案。"""
    bars = load_intraday(symbol, start, end, period=period,
                         auto_update=not offline, verbose=False)
    if bars.empty:
        return pd.DataFrame()
    panel = daily_panel(intraday_macd(bars), side=side)
    if panel.empty:
        return panel
    offset = LIMIT_OFFSET if side == "sell" else -LIMIT_OFFSET
    panel = add_limit_plan(panel, offset=offset, fallback="close", side=side)
    panel["symbol"] = symbol
    return panel


def run_side(universe: dict[str, str], side: str, *, start: str, end: str | None,
             period: int, offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """跑一侧，返回（汇总表，逐股面板）。"""
    panels, failed = [], []
    for i, (code, name) in enumerate(universe.items(), 1):
        try:
            p = panel_for(code, start, end, period, side, offline)
        except Exception as e:                  # noqa: BLE001 — 退市/停牌/源异常
            failed.append((code, str(e)[:60]))
            continue
        if p.empty:
            failed.append((code, "无数据"))
            continue
        panels.append(p)
        print(f"  [{i}/{len(universe)}] {code} {name} — {len(p)} 天", flush=True)

    if failed:
        print(f"  ⚠ 跳过 {len(failed)} 只：" +
              "，".join(f"{c}({m})" for c, m in failed[:5]) +
              ("…" if len(failed) > 5 else ""))
    if not panels:
        return pd.DataFrame(), pd.DataFrame()

    pooled = pd.concat(panels, ignore_index=True)
    limit_col = next(c for c in pooled.columns
                     if c.startswith("limit_") and not c.endswith("_filled"))
    cols = ["open", "close", "go_price", limit_col]
    labels = dict(LABELS)
    labels[limit_col] = (f"D 限价挂开盘{'+' if side == 'sell' else '−'}"
                         f"{LIMIT_OFFSET:.1%}")
    return benchmark(pooled, cols, labels, side=side), pooled


def per_stock_winner(pooled: pd.DataFrame, side: str) -> pd.Series:
    """逐股看哪个方案最优——池子层面的均值可能被少数票主导。"""
    sign = -1.0 if side == "buy" else 1.0
    limit_col = next(c for c in pooled.columns
                     if c.startswith("limit_") and not c.endswith("_filled"))
    cols = ["open", "close", "go_price", limit_col]
    edge = pooled.groupby("symbol").apply(
        lambda g: pd.Series({c: sign * ((g[c] / g["vwap"] - 1) * 1e4).mean()
                             for c in cols}),
        include_groups=False)
    return edge.idxmax(axis=1).value_counts()


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:7.1f}")


def main():
    ap = argparse.ArgumentParser(description="日内下单方案实测（按池分别出数）")
    ap.add_argument("--universe", choices=["jcy", "oil"], default="oil")
    ap.add_argument("--side", choices=["buy", "sell", "both"], default="both")
    ap.add_argument("--limit", type=int, default=45, help="JCY 池抽样只数（定种子）")
    ap.add_argument("--period", type=int, default=30, choices=[5, 15, 30, 60])
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--offline", action="store_true", help="不联网，只读本地缓存")
    ap.add_argument("--output", default="output/exec_bench")
    args = ap.parse_args()

    universe = build_universe(args.universe, args.limit)
    sides = ["buy", "sell"] if args.side == "both" else [args.side]
    os.makedirs(args.output, exist_ok=True)

    print(f"\n股票池：{args.universe}（{len(universe)} 只）  "
          f"周期：{args.period}min  区间：{args.start}~{args.end or '今日'}")

    for side in sides:
        cn = "买入" if side == "buy" else "卖出"
        print(f"\n{'=' * 72}\n{cn}侧（正 = 优于 VWAP）\n{'=' * 72}")
        table, pooled = run_side(universe, side, start=args.start, end=args.end,
                                 period=args.period, offline=args.offline)
        if table.empty:
            print("  无有效数据")
            continue

        n_days, n_sym = len(pooled), pooled["symbol"].nunique()
        print(f"\n样本：{n_sym} 只 × 合计 {n_days} 个股票-交易日  "
              f"GO 天占比 {pooled['has_go'].mean():.0%}\n")
        print(_fmt(table))

        print(f"\n逐股最优方案计票（共 {n_sym} 只）：")
        for plan, cnt in per_stock_winner(pooled, side).items():
            print(f"  {LABELS.get(plan, plan):<16} {cnt} 只")

        wv = wait_value(pooled, side=side)
        if not wv.empty:
            print(f"\nGO 成交后再等到收盘（因果可执行的比较）：")
            print(_fmt(wv))

        print(f"\n按当天有无 GO 拆开（归因用，不可作决策）：")
        print(split_by_go(pooled, side=side).to_string(
            float_format=lambda v: f"{v:8.1f}"))

        path = os.path.join(args.output, f"{args.universe}_{side}_{args.period}m.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n→ {path}")


if __name__ == "__main__":
    main()
