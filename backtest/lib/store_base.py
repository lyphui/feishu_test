"""
两个行情仓库共享的增量更新骨架。

`price_store`（日线）与 `intraday_store`（分时）过去各抄了一份逐函数同构的
实现：「头缺口 / 尾缺口 + OVERLAP_DAYS 重叠对账 + PRICE_RTOL 容差不符则
整表重建」。这里抽成唯一实现，两个 store 以参数化方式转调——怎么抓、文件
放哪、列集合、按索引还是按 dt 列合并、重叠几天，全部由调用方注入。

`oil_price.py` 是同一模式的第三个简化变体（每次整表覆盖、无头尾段逻辑），
差异足够大，**有意不复用本模块**，见其 docstring。
"""

from datetime import date, datetime, timedelta

import pandas as pd


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _shift_ymd(d: str, days: int) -> str:
    return (datetime.strptime(d, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


def incremental_update(
    symbol: str,
    start: str,
    end: str | None,
    *,
    columns: list,
    overlap_days: int,
    log_prefix: str,
    force_rebuild: bool = False,
    rebuild: bool = False,
    verbose: bool = True,
    read_local,        # () -> DataFrame；空表 = 无缓存
    write_local,       # (df) -> None
    read_meta,         # () -> dict
    write_meta,        # (req_start, req_end, df) -> None
    fetch_full,        # (start, end) -> DataFrame；首拉失败应抛异常
    fetch_gap,         # (start, end) -> DataFrame；补缺口失败返回空表
    local_bounds,      # (df) -> (first_ymd, last_ymd)
    overlap_check,     # (local, fresh) -> bool
    merge_pieces,      # (pieces: list[df]) -> df
    slice_range,       # (df, start, end) -> df
) -> pd.DataFrame:
    """
    把 [start, end] 补齐到本地仓库并返回该区间数据。

    只抓本地缺的头段和尾段；尾段多抓 `overlap_days` 天与本地对账，
    收盘价不一致（数据源改口径 / 修数）则整表重建。`force_rebuild`
    （如 qfq 无法安全追加）时一律整表重建。
    """
    end = end or _today()
    if force_rebuild:
        rebuild = True

    local = read_local()
    if rebuild or local.empty:
        local = pd.DataFrame(columns=columns)
        # 全量首拉 / 强制重建：确实没有可退的本地数据
        if verbose:
            print(f"[{log_prefix}] {symbol} 全量拉取 {start} → {end}")
        merged = fetch_full(start, end)
        if merged.empty:
            raise RuntimeError(f"{symbol} 未取到任何行情数据（{start}~{end}）")
        write_local(merged)
        write_meta(start, end, merged)
        if verbose:
            print(f"[{log_prefix}] {symbol} 写入 {len(merged)} 行")
        return slice_range(merged, start, end)

    meta = read_meta()
    first_ymd, last_ymd = local_bounds(local)
    covered_start = meta.get("requested_start") or first_ymd
    covered_end = meta.get("requested_end") or last_ymd
    pieces = [local]

    # 头段：请求区间比已覆盖区间更早
    if start < covered_start:
        head_end = _shift_ymd(covered_start, -1)
        if verbose:
            print(f"[{log_prefix}] {symbol} 补头段 {start} → {head_end}")
        head = fetch_gap(start, head_end)
        if not head.empty:
            pieces.append(head)

    # 尾段：多抓 overlap_days 天用于对账。
    # `end >= 今天` 时总是重查一次：早盘跑过一次、收盘后再跑，当天的 K 线才补得上。
    if end > covered_end or end >= _today():
        fetch_from = _shift_ymd(last_ymd, -overlap_days)
        if verbose:
            print(f"[{log_prefix}] {symbol} 补尾段 {fetch_from} → {end}"
                  f"（含 {overlap_days} 天重叠对账）")
        tail = fetch_gap(fetch_from, end)
        if not tail.empty:
            if not overlap_check(local, tail):
                print(f"[{log_prefix}] ⚠ {symbol} 重叠区间收盘价与本地不一致"
                      f"（数据源可能改了复权口径或修了历史数据），整表重建")
                return incremental_update(
                    symbol, start, end, columns=columns,
                    overlap_days=overlap_days, log_prefix=log_prefix,
                    rebuild=True, verbose=verbose,
                    read_local=read_local, write_local=write_local,
                    read_meta=read_meta, write_meta=write_meta,
                    fetch_full=fetch_full, fetch_gap=fetch_gap,
                    local_bounds=local_bounds, overlap_check=overlap_check,
                    merge_pieces=merge_pieces, slice_range=slice_range)
            pieces.append(tail)

    merged = merge_pieces(pieces)
    if len(merged) != len(local):
        write_local(merged)
        if verbose:
            print(f"[{log_prefix}] {symbol} 新增 {len(merged) - len(local)} 行，"
                  f"合计 {len(merged)} 行")
    elif verbose:
        print(f"[{log_prefix}] {symbol} 已是最新（本地 {len(merged)} 行）")
    write_meta(start, end, merged)
    return slice_range(merged, start, end)
