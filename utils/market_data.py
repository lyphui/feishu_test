"""共享指数数据获取：akshare 优先，yfinance 备用。"""

import pandas as pd


def fetch_index_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取大盘指数日线数据。

    symbol: 指数代码，如 "000300"（沪深300）、"000001"（上证）、"399001"（深证成指）
    start_date / end_date: "YYYYMMDD"
    """
    # ── akshare 优先 ─────────────────────────────────────────────────────────
    try:
        import akshare as ak
        prefix = "sz" if symbol.startswith("399") else "sh"
        ak_symbol = prefix + symbol
        print(f"  正在从 akshare 获取指数 {ak_symbol} 数据...")
        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        df = df.rename(columns={"date": "date", "close": "close",
                                 "open": "open", "high": "high",
                                 "low": "low", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        start = pd.to_datetime(start_date, format="%Y%m%d")
        end   = pd.to_datetime(end_date,   format="%Y%m%d")
        df = df.loc[start:end]
        if not df.empty:
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols]
    except ImportError:
        pass
    except Exception as e:
        print(f"  akshare 指数获取失败：{e}，尝试 yfinance 备用...")

    # ── yfinance 备用 ─────────────────────────────────────────────────────────
    import yfinance as yf
    suffix = ".SZ" if symbol.startswith("399") else ".SS"
    ticker = symbol + suffix
    start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_str   = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    print(f"  正在从 yfinance 获取指数 {ticker} 数据...")
    raw = yf.download(ticker, start=start_str, end=end_str,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"指数数据获取失败，ticker={ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    return raw[cols]
