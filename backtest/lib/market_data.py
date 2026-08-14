"""
共享行情数据获取：个股 + 指数，akshare → baostock → yfinance 三源。

复权口径
--------
回测默认取 **后复权（hfq）**，不要用前复权：

  前复权(qfq) 以"最新价"为基准回算历史价，每逢新的分红送股，**整段历史价格
  都会变**。同一个脚本今天跑和下个月跑会得到不同的买入价、不同的整手股数、
  不同的收益——回测结果不可复现，也无法和实盘对账。
  后复权(hfq) 以"上市价"为基准，历史价格一旦确定就不再变动。

ADJUST 常量集中管理三个数据源的口径映射，避免回退到备用源时口径悄悄变掉。
yfinance 的 auto_adjust=True 实为前复权口径（最新价不变、历史价下修），
与 hfq 不等价，仅作为最后兜底并会打印告警。
"""

import os
import time
import warnings

import pandas as pd

# 复权方式 → 各数据源的参数值
ADJUST = {
    "hfq":  {"akshare": "hfq", "baostock": "1"},   # 后复权（回测默认）
    "qfq":  {"akshare": "qfq", "baostock": "2"},   # 前复权（看盘习惯，不适合回测）
    "none": {"akshare": "",    "baostock": "3"},   # 不复权
}
DEFAULT_ADJUST = "hfq"


def _adjust_params(adjust: str) -> dict:
    if adjust not in ADJUST:
        raise ValueError(f"未知复权方式：{adjust}，可选 {list(ADJUST)}")
    return ADJUST[adjust]


# ── baostock 辅助 ──────────────────────────────────────────────────────────────

def to_baostock_code(symbol: str) -> str:
    """
    A 股代码 → baostock 格式（sh.600519 / sz.002202）。

    公开名：`price_store`（查派息）与 `intraday_store`（查分时）各自也要拼这个
    格式，曾各抄一份逐字节相同的 `_bs_code`，三份并存。这里是唯一实现。
    """
    prefix = "sh" if symbol.startswith("6") or symbol.startswith("9") else "sz"
    return f"{prefix}.{symbol}"


def _to_baostock_index(symbol: str) -> str:
    """指数代码 → baostock 格式（sh.000300 / sz.399006）。"""
    prefix = "sz" if symbol.startswith("399") else "sh"
    return f"{prefix}.{symbol}"


def _baostock_query(code: str, start: str, end: str,
                    frequency: str = "d",
                    fields: str = "date,open,high,low,close,volume",
                    adjustflag: str = "2") -> pd.DataFrame:
    """
    通用 baostock 查询，返回标准 DataFrame。

    code       : baostock 格式 "sh.600519"
    start/end  : "YYYY-MM-DD"
    frequency  : "d"=日线, "5"/"15"/"30"/"60"=分钟线
    adjustflag : "2"=前复权, "1"=后复权, "3"=不复权
    """
    import baostock as bs

    lg = bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            code, fields,
            start_date=start, end_date=end,
            frequency=frequency, adjustflag=adjustflag,
        )
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rs.fields)
        # 数值列转换
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    finally:
        bs.logout()


# ── 共享 yfinance 下载（含限流重试） ─────────────────────────────────────────

def _yfinance_download(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_delay: int = 10,
) -> pd.DataFrame:
    """
    通用 yfinance 下载，自动处理 MultiIndex 列名和限流重试。

    ticker     : yfinance 格式，如 "600519.SS"、"000300.SS"
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("未安装 yfinance，请运行：pip install yfinance")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  正在从 yfinance 获取 {ticker} 数据（第 {attempt}/{max_retries} 次）...")
            raw = yf.download(ticker, start=start_date, end=end_date,
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raise ValueError(f"未返回数据，ticker={ticker}")
            # 新版 yfinance 返回 MultiIndex(field, ticker)，需要降级
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            return raw
        except (ValueError, KeyError, OSError, RuntimeError) as e:
            last_err = e
            err_str = str(e)
            if "RateLimit" in type(e).__name__ or "Too Many Requests" in err_str or "429" in err_str:
                if attempt < max_retries:
                    wait = retry_delay * attempt
                    print(f"  yfinance 触发限流，等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
            break

    raise RuntimeError(
        f"yfinance 数据获取失败：{last_err}\n"
        "建议：\n"
        "  1. 稍等几分钟后再运行（yfinance 有访问频率限制）\n"
        "  2. 确认已安装 akshare：pip install akshare\n"
        "  3. 在配置文件中填写 proxy（如 http://127.0.0.1:7890）"
    )


def _warn_yfinance_adjust(adjust: str) -> None:
    """yfinance 只能给出前复权口径，与 hfq 不一致时明确告警而不是悄悄换口径。"""
    if adjust != "qfq":
        warnings.warn(
            f"已回退到 yfinance，其 auto_adjust 为**前复权**口径，"
            f"与请求的 adjust='{adjust}' 不一致：历史价会随未来分红变动，"
            "该股回测结果不可复现。建议装好 akshare 或 baostock 后重跑。",
            UserWarning, stacklevel=3,
        )


def _to_yfinance_ticker(symbol: str, is_index: bool = False,
                        is_fund: bool = False) -> str:
    """A 股代码 → yfinance ticker（沪/深自动判断）。

    场内基金（is_fund）的号段与个股不同：沪市 ETF/LOF 是 5 开头（510300、
    518880），深市是 1 开头（159915、161226）。按个股规则判断会把 510300
    错判成深市，取回的要么是空数据要么是另一只标的。
    """
    if is_index:
        suffix = ".SZ" if symbol.startswith("399") else ".SS"
    elif is_fund:
        suffix = ".SS" if symbol.startswith("5") else ".SZ"
    else:
        suffix = ".SS" if symbol.startswith("6") else ".SZ"
    return symbol + suffix


def _date_yyyymmdd_to_dash(d: str) -> str:
    """'20200101' → '2020-01-01'"""
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


# ── 个股日线数据 ─────────────────────────────────────────────────────────────

def fetch_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    proxy: str = "",
    adjust: str = DEFAULT_ADJUST,
) -> pd.DataFrame:
    """
    获取 A 股个股历史行情数据。

    symbol     : 股票代码，如 "600519"
    start_date : "YYYYMMDD"
    end_date   : "YYYYMMDD"
    proxy      : HTTP 代理地址，留空则不使用
    adjust     : 复权方式，"hfq"（默认，后复权，回测唯一可复现的口径）
                 / "qfq" / "none"，见模块 docstring
    """
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    params = _adjust_params(adjust)

    try:
        import akshare as ak
        print(f"  正在从 akshare 获取 {symbol} 数据（{adjust}）...")
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date, adjust=params["akshare"],
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except ImportError:
        print("  未找到 akshare，尝试 baostock 备用...")
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"  akshare 获取失败：{e}，尝试 baostock 备用...")

    # ── baostock 备用 ──
    try:
        bs_code = to_baostock_code(symbol)
        start_dash = _date_yyyymmdd_to_dash(start_date)
        end_dash = _date_yyyymmdd_to_dash(end_date)
        print(f"  正在从 baostock 获取 {bs_code} 日线数据（{adjust}）...")
        df = _baostock_query(bs_code, start_dash, end_dash, frequency="d",
                             adjustflag=params["baostock"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            return df[["open", "high", "low", "close", "volume"]]
        print("  baostock 返回空数据，尝试 yfinance 备用...")
    except Exception as e:
        print(f"  baostock 获取失败：{e}，尝试 yfinance 备用...")

    # ── yfinance 备用 ──
    _warn_yfinance_adjust(adjust)
    ticker = _to_yfinance_ticker(symbol, is_index=False)
    raw = _yfinance_download(
        ticker,
        _date_yyyymmdd_to_dash(start_date),
        _date_yyyymmdd_to_dash(end_date),
    )
    return raw[["open", "high", "low", "close", "volume"]]


# ── 场内基金（ETF / LOF）日线数据 ────────────────────────────────────────────

def fetch_etf_data(
    symbol: str,
    start_date: str,
    end_date: str,
    proxy: str = "",
    adjust: str = DEFAULT_ADJUST,
) -> pd.DataFrame:
    """
    获取 A 股场内基金（ETF / LOF）日线。

    symbol : 基金代码，如 "510300"（沪深300ETF）、"159915"（创业板ETF）

    为什么不能复用 `fetch_stock_data`
    --------------------------------
    东财的个股接口（`stock_zh_a_hist`）不含场内基金，baostock 干脆不覆盖
    基金——`sh.510300` 返回空表。所以这里走 akshare 的 `fund_etf_hist_em`，
    失败才退到 yfinance（口径退化为前复权，会告警，见 `_warn_yfinance_adjust`）。
    """
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    params = _adjust_params(adjust)

    try:
        import akshare as ak
        print(f"  正在从 akshare 获取场内基金 {symbol} 数据（{adjust}）...")
        df = ak.fund_etf_hist_em(
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date, adjust=params["akshare"],
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        })
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            return df[["open", "high", "low", "close", "volume"]]
        print("  akshare 基金接口返回空数据，尝试 yfinance 备用...")
    except ImportError:
        print("  未找到 akshare，尝试 yfinance 备用...")
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"  akshare 基金获取失败：{e}，尝试 yfinance 备用...")

    # ── yfinance 备用 ──
    _warn_yfinance_adjust(adjust)
    ticker = _to_yfinance_ticker(symbol, is_fund=True)
    raw = _yfinance_download(
        ticker,
        _date_yyyymmdd_to_dash(start_date),
        _date_yyyymmdd_to_dash(end_date),
    )
    return raw[["open", "high", "low", "close", "volume"]]


# ── 大盘指数日线数据 ─────────────────────────────────────────────────────────

def fetch_hk_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    获取港股 / 港股 ETF 日线（yfinance 单源，akshare 与 baostock 都不覆盖港股 ETF）。

    symbol     : 港交所代码，带不带 .HK 后缀都行，如 "3175" / "3175.HK"
    start_date : "YYYYMMDD"
    end_date   : "YYYYMMDD"

    口径警告：yfinance 的 auto_adjust=True 是**前复权**，历史价会随未来分红回算。
    因此港股必须以 `adjust="qfq"` 存进 price_store —— 那边对 qfq 强制整表重建，
    不做增量追加，否则会把两种口径的价格缝在一起。
    """
    ticker = symbol if symbol.upper().endswith(".HK") else f"{symbol}.HK"
    # yfinance 的 end 是**开区间**，直接传当天会把当天的 K 线丢掉 —— 实盘信号
    # 就差这最后一根，所以往后多要一天
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    raw = _yfinance_download(
        ticker,
        _date_yyyymmdd_to_dash(start_date),
        _date_yyyymmdd_to_dash(end_exclusive),
    )
    # yfinance 有时返回带时区的索引，price_store 按 naive 日期对账，这里统一剥掉
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    cols = [c for c in ["open", "high", "low", "close", "volume"]
            if c in raw.columns]
    return raw[cols]


def fetch_index_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    获取大盘指数日线数据。

    symbol     : 指数代码，如 "000300"（沪深300）
    start_date : "YYYYMMDD"
    end_date   : "YYYYMMDD"
    """
    try:
        import akshare as ak
        prefix = "sz" if symbol.startswith("399") else "sh"
        ak_symbol = prefix + symbol
        print(f"  正在从 akshare 获取指数 {ak_symbol} 数据...")
        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        start = pd.to_datetime(start_date, format="%Y%m%d")
        end = pd.to_datetime(end_date, format="%Y%m%d")
        df = df.loc[start:end]
        if not df.empty:
            cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df.columns]
            return df[cols]
    except ImportError:
        print("  未找到 akshare，尝试 baostock 备用...")
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"  akshare 指数获取失败：{e}，尝试 baostock 备用...")

    # ── baostock 备用 ──
    try:
        bs_code = _to_baostock_index(symbol)
        start_dash = _date_yyyymmdd_to_dash(start_date)
        end_dash = _date_yyyymmdd_to_dash(end_date)
        print(f"  正在从 baostock 获取指数 {bs_code} 日线数据...")
        df = _baostock_query(bs_code, start_dash, end_dash, frequency="d")
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in df.columns]
            return df[cols]
        print("  baostock 指数返回空数据，尝试 yfinance 备用...")
    except Exception as e:
        print(f"  baostock 指数获取失败：{e}，尝试 yfinance 备用...")

    # ── yfinance 备用 ──（指数无复权概念，不需要口径告警）
    ticker = _to_yfinance_ticker(symbol, is_index=True)
    raw = _yfinance_download(
        ticker,
        _date_yyyymmdd_to_dash(start_date),
        _date_yyyymmdd_to_dash(end_date),
    )
    cols = [c for c in ["open", "high", "low", "close", "volume"]
            if c in raw.columns]
    return raw[cols]
