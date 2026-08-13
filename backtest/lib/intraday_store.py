"""
本地分时行情仓库（**不复权**，带成交额），供 `lib/execution.py` 的下单测算使用。

为什么另起一个仓库，不复用 price_store
--------------------------------------
1. `price_store` 存日线 **hfq**；执行测算要的是**不复权**价，因为基准 VWAP 由
   `amount / volume` 算出，而这两个字段永远是原始口径。复权价与它不是一个尺度，
   混用会得到几百上千 bp 的系统性错位（`execution._check_same_basis` 就是拦这个的）。
2. 不复权价**天然只增不改**：除权除息不回溯改写历史，所以增量追加比 hfq 还安全。
3. 分时数据只有 baostock 一条路——akshare 的分钟接口打 eastmoney 域名，本机被阻断。

存储布局
--------
    data/market/intraday/{symbol}_{period}m_none.csv        dt,date,OHLCV,amount
    data/market/intraday/{symbol}_{period}m_none.meta.json  已请求区间 + 更新时间

列名与 `lib.execution.REQUIRED` 一一对应，取出来可直接喂给 `intraday_macd()`。

这个目录**不入库**（`.gitignore` 里）：日线仓库只有几百 KB 所以提交了，分时
47 只就 32MB，而且随时可以从 baostock 原样重建，属于缓存而不是数据源。

用法
----
    from backtest.lib.intraday_store import load_intraday

    bars = load_intraday("601857", "20220101", "20260808", period=30)
    bars = load_intraday("601857", period=30, auto_update=False)   # 纯离线
"""

import json
import os
from datetime import datetime

import pandas as pd

from backtest.lib import store_base

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTRADAY_DIR = os.path.join(_BASE_DIR, "data", "market", "intraday")

COLUMNS = ["dt", "date", "open", "high", "low", "close", "volume", "amount"]

# 增量时与本地重叠对账的自然日数
OVERLAP_DAYS = 5
PRICE_RTOL = 1e-4


def intraday_path(symbol: str, period: int = 30) -> str:
    return os.path.join(INTRADAY_DIR, f"{symbol}_{period}m_none.csv")


def meta_path(symbol: str, period: int = 30) -> str:
    return os.path.join(INTRADAY_DIR, f"{symbol}_{period}m_none.meta.json")


def _dash(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _bs_code(symbol: str) -> str:
    return f"{'sh' if symbol[0] in '69' else 'sz'}.{symbol}"


# ── 抓取 ──────────────────────────────────────────────────────────────────────

#: baostock 的 adjustflag 取值。**本仓库唯一的分时取数实现**就靠它区分口径：
#: 下单测算要不复权（VWAP 由 amount/volume 算，那两个字段恒为原始值），
#: 分时 MACD 要复权（不复权序列在除权日有假跳空，指标会被它带偏）。
ADJUST_FLAGS = {"none": "3", "qfq": "2", "hfq": "1"}


def fetch_intraday_raw(symbol: str, start: str, end: str,
                       period: int = 30, adjust: str = "none") -> pd.DataFrame:
    """
    从 baostock 拉分时 K 线，带 amount。**全仓库只有这一处分时取数实现。**

    baostock 的 `time` 字段标的是这根 K 线的**结束时刻**（30min 下首根是 10:00），
    因此首根的 `open` 就是当日集合竞价成交价。

    adjust
        `none` 不复权（默认，本仓库入库的就是它）/ `qfq` 前复权 / `hfq` 后复权。

        **非 none 的结果不要入库**：`qfq` 会随分红回溯改写历史价，增量追加会把两种
        口径缝在一起（与 `price_store` 同一个坑）。`load_intraday` 因此只接受 none。

        **非 none 时 `amount` 仍是原始成交额**，与被缩放过的价格不同尺度——拿它算
        VWAP 会得到几百上千 bp 的系统性错位。`execution._check_same_basis()` 会拦，
        但别指望它：复权价只用来算指标，不要用来做 VWAP 基准。
    """
    import baostock as bs

    if adjust not in ADJUST_FLAGS:
        raise ValueError(f"未知复权口径：{adjust}（可选 {list(ADJUST_FLAGS)}）")

    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            _bs_code(symbol),
            "date,time,open,high,low,close,volume,amount",
            start_date=_dash(start), end_date=_dash(end),
            frequency=str(period), adjustflag=ADJUST_FLAGS[adjust],
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        fields = rs.fields
    finally:
        bs.logout()

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    raw = pd.DataFrame(rows, columns=fields)
    out = pd.DataFrame({
        "dt": pd.to_datetime(raw["time"], format="%Y%m%d%H%M%S%f"),
        "date": pd.to_datetime(raw["date"]),
    })
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        out[c] = pd.to_numeric(raw[c], errors="coerce")
    out = out[(out["close"] > 0) & (out["volume"] > 0) & (out["amount"] > 0)]
    return out.dropna().sort_values("dt").reset_index(drop=True)[COLUMNS]


def fetch_intraday_indexed(symbol: str, start: str, end: str,
                           period: int = 30, adjust: str = "qfq") -> pd.DataFrame:
    """
    取分时 K 线并转成 **datetime 索引** 的宽表（open/high/low/close/volume，
    无 amount），供按**连续跨日序列**算 MACD
    （`lib.execution.intraday_macd`）消费。

    `fetch_intraday_raw` 返回扁平表（dt/date 列 + amount，供 VWAP 测算）；
    本函数是它的展示层适配，默认 `adjust="qfq"`——MACD 要复权，否则除权日的
    假跳空会带偏指标。不复权（none）才是 VWAP 基准的正确口径，两个口径都保留、
    别混用（`execution._check_same_basis` 会拦混用）。

    曾有两份独立的分时取数实现（backtest_jcy_intraday 自带一份 akshare/baostock
    回退 + 重试，口径还是 qfq），已删除；全仓库的分时取数现在只有 `fetch_intraday_raw`
    一处，本函数只是它的一种索引形态。
    """
    flat = fetch_intraday_raw(symbol, start, end, period=period, adjust=adjust)
    if flat.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return (flat.set_index("dt")[["open", "high", "low", "close", "volume"]]
                .rename_axis("dt"))


# ── 读写 ──────────────────────────────────────────────────────────────────────

def read_intraday(symbol: str, period: int = 30) -> pd.DataFrame:
    path = intraday_path(symbol, period)
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, parse_dates=["dt", "date"])
    return df.sort_values("dt").reset_index(drop=True)


def write_intraday(df: pd.DataFrame, symbol: str, period: int = 30) -> str:
    os.makedirs(INTRADAY_DIR, exist_ok=True)
    path = intraday_path(symbol, period)
    df.sort_values("dt").to_csv(path, index=False, float_format="%.6f")
    return path


def read_meta(symbol: str, period: int = 30) -> dict:
    path = meta_path(symbol, period)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_meta(symbol: str, period: int, *, req_start: str, req_end: str,
               df: pd.DataFrame) -> None:
    os.makedirs(INTRADAY_DIR, exist_ok=True)
    old = read_meta(symbol, period)
    meta = {
        "symbol": symbol, "period": period, "adjust": "none",
        # 记请求过的区间而非数据首尾：请求起点落在假日时，看数据首日会永远以为缺头段
        "requested_start": min(filter(None, [old.get("requested_start"), req_start])),
        "requested_end": max(filter(None, [old.get("requested_end"), req_end])),
        "bars": int(len(df)),
        "days": int(df["date"].nunique()) if len(df) else 0,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path(symbol, period), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _overlap_matches(local: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    """重叠时段的收盘价是否一致；无重叠视为一致。"""
    a = local.set_index("dt")["close"]
    b = fresh.set_index("dt")["close"]
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return True
    return bool((a[shared] - b[shared]).abs()
                .le(b[shared].abs() * PRICE_RTOL + 1e-8).all())


# ── 增量更新 ──────────────────────────────────────────────────────────────────

def update_intraday(symbol: str, start: str = "20220101", end: str | None = None,
                    *, period: int = 30, rebuild: bool = False,
                    verbose: bool = True) -> pd.DataFrame:
    """把 [start, end] 补齐到本地并返回该区间。只抓缺的头段与尾段。

    增量算法本体在 `store_base.incremental_update`（与 price_store 共用）。
    """
    return store_base.incremental_update(
        symbol, start, end,
        columns=COLUMNS,
        overlap_days=OVERLAP_DAYS,
        log_prefix="intraday",
        rebuild=rebuild,
        verbose=verbose,
        read_local=lambda: read_intraday(symbol, period),
        write_local=lambda df: write_intraday(df, symbol, period),
        read_meta=lambda: read_meta(symbol, period),
        write_meta=lambda req_start, req_end, df: write_meta(
            symbol, period, req_start=req_start, req_end=req_end, df=df),
        fetch_full=lambda s, e: fetch_intraday_raw(symbol, s, e, period),
        fetch_gap=lambda s, e: _fetch_safe(symbol, s, e, period),
        local_bounds=lambda df: (df["date"].iloc[0].strftime("%Y%m%d"),
                                 df["date"].iloc[-1].strftime("%Y%m%d")),
        overlap_check=_overlap_matches,
        merge_pieces=_merge_by_dt,
        slice_range=slice_range,
    )


def _fetch_safe(symbol: str, start: str, end: str, period: int) -> pd.DataFrame:
    """补缺口专用：抓不到就返回空表，沿用本地缓存继续跑。"""
    try:
        return fetch_intraday_raw(symbol, start, end, period)
    except Exception as e:                      # noqa: BLE001 — 数据源异常五花八门
        print(f"[intraday] ⚠ {symbol} 补 {start}~{end} 失败（{e}），沿用本地缓存")
        return pd.DataFrame(columns=COLUMNS)


def _merge_by_dt(pieces: list) -> pd.DataFrame:
    return (pd.concat(pieces).drop_duplicates("dt", keep="last")
            .sort_values("dt").reset_index(drop=True))


def slice_range(df: pd.DataFrame, start: str | None = None,
                end: str | None = None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["date"] >= pd.to_datetime(start, format="%Y%m%d")]
    if end:
        out = out[out["date"] <= pd.to_datetime(end, format="%Y%m%d")]
    return out.reset_index(drop=True)


def load_intraday(symbol: str, start: str = "20220101", end: str | None = None,
                  *, period: int = 30, auto_update: bool = True,
                  verbose: bool = True, adjust: str = "none") -> pd.DataFrame:
    """
    取分时行情。`auto_update=False` 时纯离线读本地（跑批 / 无网时用）。

    只支持 `adjust="none"`：仓库存的就是不复权，复权口径既不能安全增量追加，
    缓存了也没用（每次都得整表重建）。要复权分时直接调 `fetch_intraday_raw`。
    """
    if adjust != "none":
        raise ValueError(
            f"仓库只存不复权分时，不接受 adjust={adjust!r}。"
            f"复权价请直接用 fetch_intraday_raw(..., adjust={adjust!r})——"
            f"它会随分红回溯改写历史，不能增量追加，缓存也没有意义")
    if auto_update:
        return update_intraday(symbol, start, end, period=period, verbose=verbose)
    local = read_intraday(symbol, period)
    if local.empty:
        raise FileNotFoundError(
            f"本地无 {symbol} 的 {period}min 分时缓存，先跑一次 auto_update=True")
    return slice_range(local, start, end)
