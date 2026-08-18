"""
取数路径 A/B 对比：本地仓库**存量**缓存 vs 直连 fetch_*
========================================================

评审 `docs/backtest-review.md` 项 1 的验证：把批量任务从 `market_data.fetch_*`
统一切到 `price_store.load_daily`，会不会静默改数？

风险在哪
--------
`market_data` 的三源回退是**静默**的（akshare 挂了退 baostock，再挂退 yfinance），
而三者的 hfq 基准并不相同。仓库里某只票若是回退源写入的，切到仓库取数就等于
换了一套历史价格。

第一版这个脚本测不出这件事，两个原因（都已修）
------------------------------------------------
1. **样本取的是 jcy 池前 10 只**，而那 9 只的缓存正是跑 A/B 时刚建的——
   比的是「直连 akshare」vs「刚由同一次 akshare 调用写进仓库的文件」，
   必然一致。现在改为按 `meta.json` 的 `updated_at` 取**最老的 N 份存量缓存**，
   那才是可能由回退源写入、可能与今天的直连结果不一致的那批。
2. **`load_daily` 默认 `auto_update=True`**，比对前会先联网把缓存刷新一遍
   （甚至因重叠对账不符而整表重建），于是又变成新数据比新数据。
   现在一律 `auto_update=False`，读的就是**当前躺在盘上的那份**。

`source` 列来自 `meta.json`（评审项 6 新增）。值为空 = 该文件写入时还没有
源记录，属于最不可考、最该被抽中的一批，脚本会优先取它们。

判定：按日期 inner join 后逐列比较 OHLCV，容差复用 `price_store.PRICE_RTOL`。
输出 `output/ab_price_source/diff.csv`，并打印总结论。

用法：
    .venv/Scripts/python.exe -m backtest.scripts._ab_price_source
    .venv/Scripts/python.exe -m backtest.scripts._ab_price_source --limit 20
"""

import argparse
import glob
import os
from datetime import date as _date

import pandas as pd

from backtest.lib.console import use_utf8
from backtest.lib.market_data import fetch_index_data, fetch_stock_data
from backtest.lib.price_store import (DAILY_DIR, PRICE_RTOL, read_daily,
                                      read_meta, slice_range)

START = "20180101"
END = _date.today().strftime("%Y%m%d")
OUT_DIR = os.path.join("output", "ab_price_source")

OHLCV = ["open", "high", "low", "close", "volume"]


def stale_hfq_symbols(limit: int) -> list[tuple[str, dict]]:
    """
    仓库里**最老**的 N 份 hfq 缓存（按 meta.updated_at 升序），连同各自 meta。

    没有 `source` 记录的排在最前：那批写入时还没有源追踪，最不可考。
    今天刚写的排最后——它们与直连结果一致是同义反复，没有验证价值。
    """
    out = []
    for path in glob.glob(os.path.join(DAILY_DIR, "*_hfq.meta.json")):
        symbol = os.path.basename(path).rsplit("_hfq.meta.json", 1)[0]
        meta = read_meta(symbol, "hfq")
        if meta.get("kind", "stock") != "stock":
            continue
        out.append((symbol, meta))
    # 排序键：先按「有没有 source」，再按更新时间
    out.sort(key=lambda kv: (bool(kv[1].get("source")),
                             kv[1].get("updated_at") or ""))
    return out[:limit]


def compare(symbol: str, live: pd.DataFrame, cached: pd.DataFrame,
            source: str | None = None) -> dict:
    """直连结果 vs 盘上缓存的逐列对比。返回一行汇总。"""
    row = {"symbol": symbol, "source": source or "(未记录)",
           "live_rows": len(live), "cached_rows": len(cached)}
    if live.empty or cached.empty:
        row.update(status="EMPTY", mismatched="", max_rel_err="")
        return row
    shared = live.index.intersection(cached.index)
    row["shared_rows"] = len(shared)
    row["live_only"] = len(live.index.difference(cached.index))
    row["cached_only"] = len(cached.index.difference(shared))
    if len(shared) == 0:
        row.update(status="NO_OVERLAP", mismatched="", max_rel_err="")
        return row
    bad_cols, max_err = [], 0.0
    for col in OHLCV:
        if col not in live.columns or col not in cached.columns:
            continue
        a = live.loc[shared, col].astype(float)
        b = cached.loc[shared, col].astype(float)
        # volume 口径在不同数据源间常有手/股差异，只记相对误差不做硬判
        denom = b.abs()
        rel = (a - b).abs() / denom.where(denom > 1e-8, 1.0)
        col_err = float(rel.max()) if len(rel) else 0.0
        max_err = max(max_err, col_err)
        if col != "volume" and (rel > PRICE_RTOL).any():
            bad_cols.append(f"{col}({int((rel > PRICE_RTOL).sum())})")
    row["status"] = "OK" if not bad_cols else "DIFF"
    row["mismatched"] = ",".join(bad_cols)
    row["max_rel_err"] = f"{max_err:.2e}"
    return row


def _print_row(row: dict) -> None:
    print(f"  {row['status']:<10} 源={row['source']:<12} "
          f"共享行 {row.get('shared_rows', '-')}  "
          f"最大相对误差 {row.get('max_rel_err', '-')}  "
          f"{row.get('mismatched', '')}")


def main():
    use_utf8()
    ap = argparse.ArgumentParser(description="取数路径 A/B（存量缓存 vs 直连）")
    ap.add_argument("--limit", type=int, default=15,
                    help="抽取最老的 N 份 hfq 存量缓存，默认 15")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sample = stale_hfq_symbols(args.limit)
    if not sample:
        print("  仓库里没有 hfq 缓存，先在线跑一次任何批量脚本再来")
        return

    print(f"  抽样：仓库中最老的 {len(sample)} 份 hfq 缓存"
          f"（无 source 记录的优先）")
    print(f"  比对：直连 fetch_stock_data  vs  read_daily（**不刷新**）\n")

    rows = []
    for symbol, meta in sample:
        print(f"[A/B] {symbol}  缓存更新于 {meta.get('updated_at')}  "
              f"源={meta.get('source') or '(未记录)'}")
        try:
            live = fetch_stock_data(symbol, START, END)
        except Exception as e:  # noqa: BLE001
            print(f"  直连失败：{e}")
            live = pd.DataFrame()
        # 关键：读盘上那份，不 auto_update——否则先刷新再比，等于自证
        cached = slice_range(read_daily(symbol, "hfq"), START, END)
        row = compare(symbol, live, cached, meta.get("source"))
        rows.append(row)
        _print_row(row)

    # 指数：牛市过滤器输入，必须留在价格指数一侧（评审 3.2 陷阱框）
    print("\n[A/B] 000300 指数（kind=index, adjust=none）")
    try:
        live = fetch_index_data("000300", START, END)
    except Exception as e:  # noqa: BLE001
        print(f"  直连失败：{e}")
        live = pd.DataFrame()
    cached = slice_range(read_daily("000300", "none"), START, END)
    row = compare("000300(index)", live, cached,
                  read_meta("000300", "none").get("source"))
    rows.append(row)
    _print_row(row)

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "diff.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    diffs = df[df["status"] == "DIFF"]
    print("\n" + "─" * 66)
    if diffs.empty and (df["status"] == "OK").all():
        print(f"结论：{len(df)} 份存量缓存与直连结果全部一致"
              f"（容差 {PRICE_RTOL}）→ 切换不改数")
    else:
        print(f"结论：{len(diffs)}/{len(df)} 份不一致 → 切换会改数，"
              f"须查明来源并记入 changelog：")
        for _, r in diffs.iterrows():
            print(f"  {r['symbol']}（源={r['source']}）: {r['mismatched']}  "
                  f"max_rel_err={r['max_rel_err']}")
    print(f"明细：{out}")


if __name__ == "__main__":
    main()
