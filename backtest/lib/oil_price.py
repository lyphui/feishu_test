"""
原油价格序列：Brent / WTI / SC（上海原油期货，人民币计价）。

为什么单独走一条路，而不是复用 market_data.py 的 akshare→baostock→yfinance
------------------------------------------------------------------------
那三源里，个股/指数用的 akshare 接口大多打 eastmoney 域名；本机 eastmoney
系接口被主动阻断（TLS 握手成功、请求发出后 empty reply，是 DPI 重置的特征，
不是普通超时），yfinance 商品期货长期 429 限流。两条路都拿不到油价。

akshare 的 `futures_foreign_hist`（外盘 Brent/WTI）和 `futures_main_sina`
（SC 原油主力连续）底层打的是 `stock2.finance.sina.com.cn`，本机实测连通，
所以油价改走这两个函数。baostock 不提供商品期货数据，没有备用可回退——
一旦新浪这条线也被墙，这里会抛异常，调用方按 backtest/scripts/track_oil.py 的做法
捕获后跳过传导分析即可，不必让整个流程崩掉。

口径提醒：WTI/Brent 是美元计价连续合约，SC 是人民币计价的上海国际能源
交易中心主力连续合约——三者币种、连续合约换月处理都不同，**不要直接比
价格水平**，只用于算收益率/相关性。

三个序列的已知特性（实测，用之前先知道）
----------------------------------------
* **都是拼接连续合约，不是单一合约的可交易价格。**最直接的证据：2020-04-20
  WTI 前月合约结算 −37.63 美元，本序列当天是 21.22（走的是次月合约）。
  好处是不会出现负价把 `pct_change` 算爆；代价是**换月当天有拼接跳空**，
  那根收益率不对应任何真实持仓的盈亏。做相关性够用，拿去回测原油头寸不行。
* **WTI 的 `volume` 整列为 0**（新浪该接口不返回外盘成交量），Brent/SC 正常。
  任何按 `volume > 0` 过滤交易日的逻辑用在 WTI 上会把整段数据清空。
* 覆盖区间差异很大：WTI 自 1996-08、Brent 仅自 2016-08、SC 自 2018-03
  （上市日）。跨品种比较相关系数时注意样本区间并不一致。

存储布局
--------
    data/market/oil/{symbol}.csv        date,open,high,low,close,volume
    data/market/oil/{symbol}.meta.json  数据区间 + 最后更新时间

跟 price_store.py / intraday_store.py 不同：那两个仓库共用
`lib/store_base.incremental_update`（头尾段增量 + 重叠对账 + 容差不符则重建）；
新浪这两个接口不支持"只要某段区间"的增量拉取——每次都是整段全量吐给你，
start/end 只是拿到手之后的本地切片，多传参数不省流量。数据量也小（几千行、
几十 KB），所以这里**有意不复用 store_base**，每次更新就是整表覆盖，
省一套用不上的对账代码。
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE_DIR = os.path.join(_BASE_DIR, "data", "market")
OIL_DIR = os.path.join(STORE_DIR, "oil")

OHLCV = ["open", "high", "low", "close", "volume"]

# symbol → (akshare 函数名, 传给该函数的 symbol 参数)
_SOURCES = {
    "WTI": ("futures_foreign_hist", "CL"),
    "BRENT": ("futures_foreign_hist", "OIL"),
    "SC": ("futures_main_sina", "SC0"),
}
OIL_SYMBOLS = list(_SOURCES)

# A 股油气股票代码 → 名称。track_oil / compare_exec_plans / backtest_fatfinger
# 共用（曾各写一遍，且与上面的 `OIL_SYMBOLS`（商品代码 WTI/BRENT/SC）**同名不同义**，
# 已改名区分）。注意这里的代码是 A 股，不是上面那个商品品种列表。
OIL_STOCKS = {"601857": "中国石油", "600938": "中国海油"}

_COLUMN_MAP_CN = {
    "日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low",
    "收盘价": "close", "成交量": "volume",
}


# ── 路径 ──────────────────────────────────────────────────────────────────────

def oil_path(symbol: str) -> str:
    return os.path.join(OIL_DIR, f"{symbol}.csv")


def meta_path(symbol: str) -> str:
    return os.path.join(OIL_DIR, f"{symbol}.meta.json")


# ── 抓取 ──────────────────────────────────────────────────────────────────────

def fetch_oil_price(symbol: str) -> pd.DataFrame:
    """
    抓取原油期货历史日线（新浪源）。

    symbol : "WTI" / "BRENT" / "SC"
    """
    if symbol not in _SOURCES:
        raise ValueError(f"未知原油品种：{symbol}，可选 {OIL_SYMBOLS}")

    import akshare as ak

    func_name, ak_symbol = _SOURCES[symbol]
    func = getattr(ak, func_name)
    print(f"  正在从新浪财经获取 {symbol}（{ak_symbol}）历史行情...")
    raw = func(symbol=ak_symbol)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV)

    raw = raw.rename(columns=_COLUMN_MAP_CN)

    # 必需列缺失要当场报错，不能补 NA 混过去。
    # 三个接口的 schema 并不统一：futures_foreign_hist 直接返回英文列名，
    # futures_main_sina 返回中文列名靠 _COLUMN_MAP_CN 翻译。akshare 上游改任何
    # 一个列名，补 NA 的写法都会让 close 变成整列 NaN，一路流到 dropna() 里
    # 被清空，最终表现为"传导分析悄悄没有了"而不是报错——这是最难查的那种坏。
    missing = [c for c in ("date", "close") if c not in raw.columns]
    if missing:
        raise RuntimeError(
            f"{symbol}（{func_name}/{ak_symbol}）返回的列不含 {missing}；"
            f"实际列为 {list(raw.columns)}。akshare 上游 schema 可能变了，"
            f"请更新 lib/oil_price.py 的 _COLUMN_MAP_CN")

    raw["date"] = pd.to_datetime(raw["date"])
    df = raw.set_index("date").sort_index()
    for col in OHLCV:                       # 只有 OHLV 允许缺（如外盘无成交量）
        if col not in df.columns:
            df[col] = pd.NA
    df = df[OHLCV]
    df = df[~df.index.duplicated(keep="last")]

    if df["close"].isna().all():
        raise RuntimeError(f"{symbol} 取回 {len(df)} 行但 close 全为空值")
    return df


# ── 读写 ──────────────────────────────────────────────────────────────────────

def read_oil(symbol: str) -> pd.DataFrame:
    path = oil_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame(columns=OHLCV)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df[[c for c in OHLCV if c in df.columns]]


def read_meta(symbol: str) -> dict:
    path = meta_path(symbol)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def update_oil(symbol: str, *, force: bool = False,
               verbose: bool = True) -> pd.DataFrame:
    """
    整表重建（新浪接口不支持增量，每次全量覆盖）。

    覆盖前先和本地对一次账。整表覆盖没有 price_store 的头尾段增量逻辑，
    也就没了它那层重叠对账保护：数据源半残（只吐回最近几百行）时，
    一次 `update_oil` 就能把本地几千行历史抹掉，而且不会报任何错。
    所以这里加两道闸——行数不得明显缩水、重叠区间收盘价必须对得上。
    `force=True` 可强行覆盖（确认上游确实重构了历史时用）。
    """
    df = fetch_oil_price(symbol)
    if df.empty:
        raise RuntimeError(f"{symbol} 未取到任何行情数据")

    local = read_oil(symbol)
    if not local.empty and not force:
        if len(df) < len(local) * 0.95:
            raise RuntimeError(
                f"{symbol} 新取回 {len(df)} 行，明显少于本地已有的 {len(local)} 行，"
                f"疑似数据源残缺，已拒绝覆盖。确认上游确实重构了历史再用 force=True")
        shared = local.index.intersection(df.index)
        if len(shared) >= 20:
            a = local.loc[shared, "close"].astype(float)
            b = df.loc[shared, "close"].astype(float)
            bad = (a - b).abs() > b.abs() * 1e-3 + 1e-8
            if bad.mean() > 0.01:               # 容忍个别修正，成片对不上就是换口径了
                raise RuntimeError(
                    f"{symbol} 与本地重叠的 {len(shared)} 个交易日里有 "
                    f"{bad.sum()} 天收盘价对不上（首个分歧日 "
                    f"{bad[bad].index[0].date()}），疑似上游换了合约拼接口径，"
                    f"已拒绝覆盖。核对后用 force=True 重建")

    os.makedirs(OIL_DIR, exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(oil_path(symbol), float_format="%.4f")

    meta = {
        "symbol": symbol,
        "data_start": df.index[0].strftime("%Y%m%d"),
        "data_end": df.index[-1].strftime("%Y%m%d"),
        "rows": int(len(df)),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path(symbol), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[oil] {symbol} 写入 {len(df)} 行"
              f"（{meta['data_start']}~{meta['data_end']}）→ {oil_path(symbol)}")
    return df


def load_oil(symbol: str, *, auto_update: bool = True, verbose: bool = True) -> pd.DataFrame:
    """取原油价格。`auto_update=False` 纯离线读本地缓存（跑批/无网时用）。"""
    if auto_update:
        return update_oil(symbol, verbose=verbose)
    local = read_oil(symbol)
    if local.empty:
        raise FileNotFoundError(f"本地无 {symbol} 原油缓存，先跑一次 auto_update=True")
    return local


# ── 传导性分析 ────────────────────────────────────────────────────────────────

def transmission_table(oil: pd.DataFrame, stock: pd.DataFrame,
                       lags=(0, 1, 2, 3, 5, 10, 20),
                       max_staleness_days: int = 7) -> pd.DataFrame:
    """
    油价 → 股价传导性：油价日收益率领先股价日收益率 `lag` 个**股票交易日**的
    皮尔逊相关系数。纯描述性统计，不是回测、不产生交易信号。

    两者交易日历不同（WTI/Brent 是美盘时区连续报价，SC 是上期能源夜盘+日盘，
    都不等于 A 股交易日历），所以先用 `merge_asof` 把油价对齐到股票的每个
    交易日——取该日或之前最新一个已知的油价收盘价，再分别算收益率。

    `max_staleness_days` 是这次对齐允许回溯多远。**必须设**：不设容差时
    merge_asof 会把最后一个已知油价一路前向填充，油价缓存过期半年的话，
    那半年的"油价收益率"全是 0，而 `n` 一点不掉——相关系数被悄悄稀释，
    表面上还是一张正常的表。实测缓存停更半年会让 lag=1 从 0.262 掉到 0.225。
    超出容差的日期油价置为缺失、直接从样本里剔除，`n` 会跟着掉下来，
    读表的人能看见。

    lag=0 是"同一（股票）交易日"相关，其实隐含的是隔夜/亚盘时段油价变动
    已经被计入的信息，不能解读成"当天以后才发生的传导"；lag>0 才是严格意义
    上"油价变动出现在先、股价变动在后"的传导关系。

    输出的 `ci95` 是 r=0 假设下的噪声阈（±1.96/√n），`signif` 为 |corr| 是否
    超过它。不给这一列的话，一张表里 7 个 lag 有 5 个是纯噪声，很容易被当成
    "还有微弱的传导"来解读。
    """
    cols = ["lag_days", "n", "corr", "ci95", "signif"]
    oil_c = oil["close"].dropna().sort_index().astype(float)
    stock_c = stock["close"].dropna().sort_index().astype(float)
    if oil_c.empty or stock_c.empty:
        return pd.DataFrame(columns=cols)

    aligned = pd.merge_asof(
        stock_c.rename("stock").to_frame(),
        oil_c.rename("oil").to_frame(),
        left_index=True, right_index=True,
        tolerance=pd.Timedelta(days=max_staleness_days),
    )
    stock_ret = aligned["stock"].pct_change()
    # 油价收益率必须在"油价自己的相邻观测"之间算。先 dropna 再 pct_change，
    # 否则中断处会拿断点两侧的价格相除，凭空造出一根巨大的假收益。
    oil_ret = aligned["oil"].dropna().pct_change().reindex(aligned.index)

    rows = []
    for lag in lags:
        pair = pd.DataFrame({"oil": oil_ret.shift(lag), "stock": stock_ret}).dropna()
        if len(pair) < 30:
            continue
        n = len(pair)
        r = float(pair["oil"].corr(pair["stock"]))
        ci = 1.96 / np.sqrt(n)
        rows.append({"lag_days": lag, "n": n, "corr": r,
                     "ci95": ci, "signif": abs(r) > ci})
    return pd.DataFrame(rows, columns=cols)
