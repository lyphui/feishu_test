"""共享行情数据获取：个股 + 指数，akshare 优先，yfinance 备用。"""

import os
import time

import pandas as pd


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


def _to_yfinance_ticker(symbol: str, is_index: bool = False) -> str:
    """A 股代码 → yfinance ticker（沪/深自动判断）。"""
    if is_index:
        suffix = ".SZ" if symbol.startswith("399") else ".SS"
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
) -> pd.DataFrame:
    """
    获取 A 股个股历史行情数据（前复权）。

    symbol     : 股票代码，如 "600519"
    start_date : "YYYYMMDD"
    end_date   : "YYYYMMDD"
    proxy      : HTTP 代理地址，留空则不使用
    """
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    try:
        import akshare as ak
        print(f"  正在从 akshare 获取 {symbol} 数据...")
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date, adjust="qfq",
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
        print("  未找到 akshare，尝试使用 yfinance 作为备用...")
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"  akshare 获取失败：{e}，尝试 yfinance 备用...")

    ticker = _to_yfinance_ticker(symbol, is_index=False)
    raw = _yfinance_download(
        ticker,
        _date_yyyymmdd_to_dash(start_date),
        _date_yyyymmdd_to_dash(end_date),
    )
    return raw[["open", "high", "low", "close", "volume"]]


# ── 大盘指数日线数据 ─────────────────────────────────────────────────────────

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
        pass
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"  akshare 指数获取失败：{e}，尝试 yfinance 备用...")

    ticker = _to_yfinance_ticker(symbol, is_index=True)
    raw = _yfinance_download(
        ticker,
        _date_yyyymmdd_to_dash(start_date),
        _date_yyyymmdd_to_dash(end_date),
    )
    cols = [c for c in ["open", "high", "low", "close", "volume"]
            if c in raw.columns]
    return raw[cols]
