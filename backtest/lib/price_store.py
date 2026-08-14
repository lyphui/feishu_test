"""
本地行情仓库：第一次拉全量，之后每次只补增量。

为什么可以安全地"只追加"
------------------------
仓库固定存 **后复权（hfq）**：基准是上市价，历史价一旦确定就不再变动，
所以新数据只会往尾巴上长，旧行不会被改写。前复权（qfq）以最新价为基准回算，
每次分红送股整段历史都会变，追加就会把两种口径的价格拼在一起 —— 因此
qfq 只允许整表重建（`update_daily(..., adjust="qfq")` 会强制 rebuild）。

即便如此，数据源换口径 / 补数据 / 修正错误都可能悄悄改写历史。所以每次增量
更新都会**重叠取回最近 `overlap_days` 天**与本地对账，收盘价对不上就自动整表
重建，而不是把两段不一致的数据缝起来。

存储布局
--------
    data/market/daily/{symbol}_{adjust}.csv        date,open,high,low,close,volume
    data/market/daily/{symbol}_{adjust}.meta.json  已覆盖的请求区间 + 最后更新时间
    data/market/dividend/{symbol}.csv              ex_date,pay_date,cash_before_tax

meta 记的是**请求过**的区间，而不是数据的首尾日期。两者不同：请求 20180101
起，实际第一根 K 线是 20180102（1 日是假日）。只看数据首日的话，每次更新都会
以为头段缺了一天，反复去抓一个永远为空的区间。

用法
----
    from backtest.lib.price_store import load_daily, update_daily, load_dividends

    df = load_daily("601857", "20240101", "20260808")   # 缺什么补什么，然后返回
    update_daily("601857", rebuild=True)                # 整表重建
"""

import json
import os
from datetime import date, datetime

import pandas as pd

from backtest.lib import store_base
from backtest.lib.market_data import (DEFAULT_ADJUST, fetch_etf_data, fetch_hk_data,
                             fetch_index_data, fetch_stock_data, to_baostock_code)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE_DIR = os.path.join(_BASE_DIR, "data", "market")
DAILY_DIR = os.path.join(STORE_DIR, "daily")
DIVIDEND_DIR = os.path.join(STORE_DIR, "dividend")

OHLCV = ["open", "high", "low", "close", "volume"]

# 增量更新时与本地重叠对账的自然日数
OVERLAP_DAYS = 10
# 收盘价对账容差（相对误差）
PRICE_RTOL = 1e-4


# ── 路径 ──────────────────────────────────────────────────────────────────────

def daily_path(symbol: str, adjust: str = DEFAULT_ADJUST) -> str:
    return os.path.join(DAILY_DIR, f"{symbol}_{adjust}.csv")


def meta_path(symbol: str, adjust: str = DEFAULT_ADJUST) -> str:
    return os.path.join(DAILY_DIR, f"{symbol}_{adjust}.meta.json")


def dividend_path(symbol: str) -> str:
    return os.path.join(DIVIDEND_DIR, f"{symbol}.csv")


# ── 日期工具 ──────────────────────────────────────────────────────────────────

def _to_ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


# ── 读写 ──────────────────────────────────────────────────────────────────────

def read_daily(symbol: str, adjust: str = DEFAULT_ADJUST) -> pd.DataFrame:
    """读本地缓存；文件不存在返回空 DataFrame。"""
    path = daily_path(symbol, adjust)
    if not os.path.exists(path):
        return pd.DataFrame(columns=OHLCV)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df[[c for c in OHLCV if c in df.columns]]


def write_daily(df: pd.DataFrame, symbol: str, adjust: str = DEFAULT_ADJUST) -> str:
    os.makedirs(DAILY_DIR, exist_ok=True)
    path = daily_path(symbol, adjust)
    out = df.sort_index()
    out.index.name = "date"
    out.to_csv(path, float_format="%.6f")
    return path


def read_meta(symbol: str, adjust: str = DEFAULT_ADJUST) -> dict:
    path = meta_path(symbol, adjust)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_meta(symbol: str, adjust: str, *, req_start: str, req_end: str,
               df: pd.DataFrame, kind: str) -> None:
    os.makedirs(DAILY_DIR, exist_ok=True)
    old = read_meta(symbol, adjust)
    meta = {
        "symbol": symbol,
        "adjust": adjust,
        "kind": kind,
        # 请求过的最宽区间：下次判断头段/尾段缺口用它，而不是数据首尾日
        "requested_start": min(filter(None, [old.get("requested_start"), req_start])),
        "requested_end": max(filter(None, [old.get("requested_end"), req_end])),
        "data_start": _to_ymd(df.index[0]) if len(df) else None,
        "data_end": _to_ymd(df.index[-1]) if len(df) else None,
        "rows": int(len(df)),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path(symbol, adjust), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ── 抓取 ──────────────────────────────────────────────────────────────────────

def _fetch(symbol: str, start: str, end: str, kind: str, adjust: str,
           proxy: str = "") -> pd.DataFrame:
    if kind == "index":
        df = fetch_index_data(symbol, start, end)
    elif kind == "hk":
        df = fetch_hk_data(symbol, start, end)
    elif kind == "etf":
        df = fetch_etf_data(symbol, start, end, adjust=adjust, proxy=proxy)
    else:
        df = fetch_stock_data(symbol, start, end, proxy=proxy, adjust=adjust)
    if df.empty:
        return df
    df = df[[c for c in OHLCV if c in df.columns]]
    return df[~df.index.duplicated(keep="last")].sort_index()


# ── 增量更新 ──────────────────────────────────────────────────────────────────

def update_daily(
    symbol: str,
    start: str = "20180101",
    end: str = None,
    *,
    adjust: str = DEFAULT_ADJUST,
    kind: str = "stock",
    proxy: str = "",
    rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    把 [start, end] 补齐到本地仓库并返回该区间数据。

    只抓本地缺的头段和尾段；尾段多抓 OVERLAP_DAYS 天与本地对账，
    收盘价不一致（数据源改口径/修数）则整表重建。
    qfq 口径无法安全追加，一律整表重建。

    增量算法本体在 `store_base.incremental_update`（与 intraday_store 共用）。
    """
    return store_base.incremental_update(
        symbol, start, end,
        columns=OHLCV,
        overlap_days=OVERLAP_DAYS,
        log_prefix="store",
        force_rebuild=adjust == "qfq",
        rebuild=rebuild,
        verbose=verbose,
        read_local=lambda: read_daily(symbol, adjust),
        write_local=lambda df: write_daily(df, symbol, adjust),
        read_meta=lambda: read_meta(symbol, adjust),
        write_meta=lambda req_start, req_end, df: write_meta(
            symbol, adjust, req_start=req_start, req_end=req_end, df=df, kind=kind),
        fetch_full=lambda s, e: _fetch(symbol, s, e, kind, adjust, proxy),
        fetch_gap=lambda s, e: _fetch_safe(symbol, s, e, kind, adjust, proxy),
        local_bounds=lambda df: (_to_ymd(df.index[0]), _to_ymd(df.index[-1])),
        overlap_check=_overlap_matches,
        merge_pieces=_merge_by_index,
        slice_range=slice_range,
    )


def _fetch_safe(symbol: str, start: str, end: str, kind: str, adjust: str,
                proxy: str = "") -> pd.DataFrame:
    """
    补缺口专用：抓不到就返回空表。

    本地已有可用数据时，一次网络故障不该让整个流程崩掉——退化成"用现有缓存
    继续跑"比抛异常有用。全量首拉不走这里，那时确实没有可退的东西。
    """
    try:
        return _fetch(symbol, start, end, kind, adjust, proxy)
    except Exception as e:                      # noqa: BLE001 — 数据源异常五花八门
        print(f"[store] ⚠ {symbol} 补 {start}~{end} 失败（{e}），沿用本地缓存")
        return pd.DataFrame(columns=OHLCV)


def _overlap_matches(local: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    """本地与新抓数据在重叠日期上的收盘价是否一致。无重叠视为一致。"""
    shared = local.index.intersection(fresh.index)
    if len(shared) == 0:
        return True
    a = local.loc[shared, "close"].astype(float)
    b = fresh.loc[shared, "close"].astype(float)
    return bool((a - b).abs().le(b.abs() * PRICE_RTOL + 1e-8).all())


def _merge_by_index(pieces: list) -> pd.DataFrame:
    merged = pd.concat(pieces)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def slice_range(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    lo = pd.to_datetime(start, format="%Y%m%d") if start else None
    hi = pd.to_datetime(end, format="%Y%m%d") if end else None
    return df.loc[lo:hi]


def load_daily(
    symbol: str,
    start: str = "20180101",
    end: str = None,
    *,
    adjust: str = DEFAULT_ADJUST,
    kind: str = "stock",
    auto_update: bool = True,
    proxy: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    取行情。`auto_update=True`（默认）时先把缺口补齐；
    `auto_update=False` 纯离线读本地（跑批 / 无网时用）。
    """
    if auto_update:
        return update_daily(symbol, start, end, adjust=adjust, kind=kind,
                            proxy=proxy, verbose=verbose)
    local = read_daily(symbol, adjust)
    if local.empty:
        raise FileNotFoundError(
            f"本地无 {symbol}（{adjust}）缓存，先跑一次 auto_update=True")
    return slice_range(local, start, end)


# ── 分红 ──────────────────────────────────────────────────────────────────────

def update_dividends(symbol: str, start_year: int = 2015, end_year: int = None,
                     *, verbose: bool = True) -> pd.DataFrame:
    """
    抓派息记录（baostock）并落盘。返回 ex_date / pay_date / cash_before_tax。

    每次全量重抓：记录只有几十行，且历史会被交易所修订，追加没有意义。
    baostock 未安装时打印提示并返回本地已有数据（可能为空）。
    """
    end_year = end_year or date.today().year
    try:
        import baostock as bs
    except ImportError:
        print("[store] 未安装 baostock，跳过分红抓取（pip install baostock）")
        return read_dividends(symbol)

    bs.login()
    try:
        rows, fields = [], []
        for year in range(start_year, end_year + 1):
            rs = bs.query_dividend_data(code=to_baostock_code(symbol), year=str(year),
                                        yearType="report")
            fields = rs.fields or fields
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        if verbose:
            print(f"[store] {symbol} 无分红记录")
        return read_dividends(symbol)

    raw = pd.DataFrame(rows, columns=fields)
    df = pd.DataFrame({
        "ex_date": pd.to_datetime(raw.get("dividOperateDate"), errors="coerce"),
        "pay_date": pd.to_datetime(raw.get("dividPayDate"), errors="coerce"),
        "cash_before_tax": pd.to_numeric(raw.get("dividCashPsBeforeTax"),
                                         errors="coerce"),
    }).dropna(subset=["ex_date"]).drop_duplicates("ex_date").sort_values("ex_date")

    os.makedirs(DIVIDEND_DIR, exist_ok=True)
    df.to_csv(dividend_path(symbol), index=False)
    if verbose:
        print(f"[store] {symbol} 分红 {len(df)} 条 → {dividend_path(symbol)}")
    return df


def read_dividends(symbol: str) -> pd.DataFrame:
    path = dividend_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ex_date", "pay_date", "cash_before_tax"])
    return pd.read_csv(path, parse_dates=["ex_date", "pay_date"])


def load_dividends(symbol: str, *, auto_update: bool = True) -> pd.DataFrame:
    if auto_update:
        return update_dividends(symbol, verbose=False)
    return read_dividends(symbol)
