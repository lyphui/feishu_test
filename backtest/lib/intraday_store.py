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
    from lib.intraday_store import load_intraday

    bars = load_intraday("601857", "20220101", "20260808", period=30)
    bars = load_intraday("601857", period=30, auto_update=False)   # 纯离线
"""

import json
import os
from datetime import date, datetime, timedelta

import pandas as pd

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


def _today() -> str:
    return date.today().strftime("%Y%m%d")


def _dash(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _shift(d: str, days: int) -> str:
    return (datetime.strptime(d, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


def _bs_code(symbol: str) -> str:
    return f"{'sh' if symbol[0] in '69' else 'sz'}.{symbol}"


# ── 抓取 ──────────────────────────────────────────────────────────────────────

def fetch_intraday_raw(symbol: str, start: str, end: str,
                       period: int = 30) -> pd.DataFrame:
    """
    从 baostock 拉**不复权**分时 K 线（`adjustflag="3"`），带 amount。

    baostock 的 `time` 字段标的是这根 K 线的**结束时刻**（30min 下首根是 10:00），
    因此首根的 `open` 就是当日集合竞价成交价。
    """
    import baostock as bs

    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            _bs_code(symbol),
            "date,time,open,high,low,close,volume,amount",
            start_date=_dash(start), end_date=_dash(end),
            frequency=str(period), adjustflag="3",
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
    """把 [start, end] 补齐到本地并返回该区间。只抓缺的头段与尾段。"""
    end = end or _today()
    local = pd.DataFrame(columns=COLUMNS) if rebuild else read_intraday(symbol, period)

    if local.empty:
        if verbose:
            print(f"[intraday] {symbol} 全量拉取 {start} → {end}（{period}min，不复权）")
        merged = fetch_intraday_raw(symbol, start, end, period)
        if merged.empty:
            raise RuntimeError(f"{symbol} 未取到任何分时数据（{start}~{end}）")
        write_intraday(merged, symbol, period)
        write_meta(symbol, period, req_start=start, req_end=end, df=merged)
        if verbose:
            print(f"[intraday] {symbol} 写入 {len(merged)} 根 / "
                  f"{merged['date'].nunique()} 天")
        return slice_range(merged, start, end)

    meta = read_meta(symbol, period)
    covered_start = meta.get("requested_start") or local["date"].iloc[0].strftime("%Y%m%d")
    covered_end = meta.get("requested_end") or local["date"].iloc[-1].strftime("%Y%m%d")
    pieces, added = [local], 0

    if start < covered_start:
        head_end = _shift(covered_start, -1)
        if verbose:
            print(f"[intraday] {symbol} 补头段 {start} → {head_end}")
        head = _fetch_safe(symbol, start, head_end, period)
        if not head.empty:
            pieces.append(head)
            added += len(head)

    if end > covered_end or end >= _today():
        fetch_from = _shift(local["date"].iloc[-1].strftime("%Y%m%d"), -OVERLAP_DAYS)
        if verbose:
            print(f"[intraday] {symbol} 补尾段 {fetch_from} → {end}（含重叠对账）")
        tail = _fetch_safe(symbol, fetch_from, end, period)
        if not tail.empty:
            if not _overlap_matches(local, tail):
                print(f"[intraday] ⚠ {symbol} 重叠时段收盘价与本地不一致，整表重建")
                return update_intraday(symbol, start, end, period=period,
                                       rebuild=True, verbose=verbose)
            pieces.append(tail)
            added += len(tail)

    merged = (pd.concat(pieces).drop_duplicates("dt", keep="last")
              .sort_values("dt").reset_index(drop=True))
    if len(merged) != len(local):
        write_intraday(merged, symbol, period)
        if verbose:
            print(f"[intraday] {symbol} 新增 {len(merged) - len(local)} 根，"
                  f"合计 {len(merged)} 根 / {merged['date'].nunique()} 天")
    elif verbose:
        print(f"[intraday] {symbol} 已是最新（{len(merged)} 根）")
    write_meta(symbol, period, req_start=start, req_end=end, df=merged)
    return slice_range(merged, start, end)


def _fetch_safe(symbol: str, start: str, end: str, period: int) -> pd.DataFrame:
    """补缺口专用：抓不到就返回空表，沿用本地缓存继续跑。"""
    try:
        return fetch_intraday_raw(symbol, start, end, period)
    except Exception as e:                      # noqa: BLE001 — 数据源异常五花八门
        print(f"[intraday] ⚠ {symbol} 补 {start}~{end} 失败（{e}），沿用本地缓存")
        return pd.DataFrame(columns=COLUMNS)


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
                  verbose: bool = True) -> pd.DataFrame:
    """取分时行情。`auto_update=False` 时纯离线读本地（跑批 / 无网时用）。"""
    if auto_update:
        return update_intraday(symbol, start, end, period=period, verbose=verbose)
    local = read_intraday(symbol, period)
    if local.empty:
        raise FileNotFoundError(
            f"本地无 {symbol} 的 {period}min 分时缓存，先跑一次 auto_update=True")
    return slice_range(local, start, end)
