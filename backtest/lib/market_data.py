"""
共享行情数据获取：个股 + 指数 + ETF + 港股，多源带回退。

源顺序按品种而定，不是全局统一的
--------------------------------
  个股 `fetch_stock_data`  baostock → akshare → yfinance
  指数 `fetch_index_data`  akshare(新浪源) → baostock → yfinance
  ETF  `fetch_etf_data`    akshare(东财) → yfinance      （baostock 不覆盖基金）
  港股 `fetch_hk_data`     yfinance 单源                  （另两家都不覆盖）

个股为什么把 baostock 提到首位见 `fetch_stock_data` 的 docstring：**东财的 hfq
不是全收益口径**，高股息标的上年化能差 4pp。指数不受影响（无复权概念，且
akshare 指数接口走新浪不经东财，两源实测逐位一致），故维持原顺序。

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

#: 各 fetch_* 返回的标准列集合（`price_store.OHLCV` 与之一致）
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _adjust_params(adjust: str) -> dict:
    if adjust not in ADJUST:
        raise ValueError(f"未知复权方式：{adjust}，可选 {list(ADJUST)}")
    return ADJUST[adjust]


def _tagged(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    在返回帧上记下**是哪个源给的数据**（`df.attrs["source"]`）。

    为什么必须记：三源回退是静默的（akshare 挂了就退 baostock，再挂退
    yfinance），而三者的 hfq 基准并不相同——同一只票换个源，整条历史价格
    就是另一套数字。`price_store` 把它写进 `meta.json`，否则事后无从判断
    仓库里哪些文件是回退源写的，"换取数路径会不会改数"这类问题也就没法查。
    """
    df = df.copy()
    df.attrs["source"] = source
    return df


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


def _warn_akshare_hfq(symbol: str, adjust: str) -> None:
    """
    回退到 akshare 取 hfq 时告警：它与首选源 baostock 的口径**不同**。

    东财 hfq 的分红再投累积方式与 baostock 不一致——601225 实测 8.6 年年化
    差 4.1pp（详见 `fetch_stock_data` docstring）。落到仓库里就是"这一只票
    与其余标的不同口径"，横截面比较会被污染，所以必须响一声，并靠
    `meta.json` 的 `source` 字段留痕。
    """
    if adjust == "hfq":
        warnings.warn(
            f"{symbol} 已回退到 akshare(东财) 取 hfq：其分红再投累积与首选源 "
            "baostock 不同（高股息标的年化可差约 4pp），该票与仓库其余标的"
            "**不同口径**，横截面比较需谨慎。meta.json 的 source 会记为 akshare。",
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
    获取 A 股个股历史行情数据。源顺序：**baostock → akshare → yfinance**。

    symbol     : 股票代码，如 "600519"
    start_date : "YYYYMMDD"
    end_date   : "YYYYMMDD"
    proxy      : HTTP 代理地址，留空则不使用
    adjust     : 复权方式，"hfq"（默认，后复权，回测唯一可复现的口径）
                 / "qfq" / "none"，见模块 docstring

    为什么首选 baostock 而不是 akshare（2026-08 换的顺序）
    ------------------------------------------------------
    **东财的 hfq 不是全收益口径。** 用 601225 陕西煤业实测（纯现金分红、
    无送转，可从第一性原理重建）：不复权价 + 税前现金分红重建出的全收益
    年化 +22.27%；baostock 的 hfq 年化 +22.18%，与重建的日收益**中位差为 0**，
    8.6 年里只有 9 天超过 1bp（都在除息日附近）；而东财那份年化只有 +18.17%，
    日收益中位差 15bp、2086 天里 2013 天超 1bp，累计 +298% vs +425%——
    **年化差 4.1pp**。除息日单看两边都对，差的是整段的分红再投累积方式。

    对一个把 hfq 当作"含股息再投的全收益口径"来用的回测仓库，这是口径错误，
    不是精度差异。另外两条次要理由：本机 eastmoney 被 DPI 阻断（akshare 的
    个股接口打 `push2his.eastmoney.com`，长期不通）；东财只给 2 位小数，
    在低价股上会给日收益注入 ~15bp 的舍入噪声，baostock 给 6 位。

    仅限**个股**。指数（`fetch_index_data`）走的是新浪接口、不经东财，本机可达
    且两源实测逐位一致，故维持 akshare 优先；ETF 与港股 baostock 根本不覆盖。
    """
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    params = _adjust_params(adjust)

    # ── baostock 首选（hfq 为真全收益口径，见 docstring） ──
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
            return _tagged(df[["open", "high", "low", "close", "volume"]], "baostock")
        print("  baostock 返回空数据，尝试 akshare 备用...")
    except ImportError:
        print("  未找到 baostock，尝试 akshare 备用...")
    except Exception as e:                          # noqa: BLE001 — 源异常五花八门
        print(f"  baostock 获取失败：{e}，尝试 akshare 备用...")

    # ── akshare 备用 ──
    # 口径告警：东财 hfq 的分红再投累积与 baostock 不同（高股息票年化可差 4pp），
    # 回退到这里的标的与仓库其余部分**不同口径**，meta.json 的 source 会记下来。
    try:
        import akshare as ak
        _warn_akshare_hfq(symbol, adjust)
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
        return _tagged(df[["open", "high", "low", "close", "volume"]], "akshare")
    except ImportError:
        print("  未找到 akshare，尝试 yfinance 备用...")
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"  akshare 获取失败：{e}，尝试 yfinance 备用...")

    # ── yfinance 备用 ──
    _warn_yfinance_adjust(adjust)
    ticker = _to_yfinance_ticker(symbol, is_index=False)
    raw = _yfinance_download(
        ticker,
        _date_yyyymmdd_to_dash(start_date),
        _date_yyyymmdd_to_dash(end_date),
    )
    return _tagged(raw[["open", "high", "low", "close", "volume"]], "yfinance")


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
            return _tagged(df[["open", "high", "low", "close", "volume"]], "akshare")
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
    return _tagged(raw[["open", "high", "low", "close", "volume"]], "yfinance")


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
    return _tagged(raw[cols], "yfinance")


def fetch_index_tr_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    获取中证全收益指数日线（含股息再投），如 "H00300"（沪深300全收益）。

    用途只有一个：**选股 alpha 的分母**（评审 `docs/backtest-review.md` 3.2）。
    个股 hfq 是全收益口径（含股息再投），拿它减不含股息的价格指数会把
    「选股alpha%」系统性抬高一个股息率的量级——2018-01→2026-08 沪深300
    实测价格指数年化 +1.62% vs 全收益 +4.07%，差 2.45%/年。

    牛市过滤器**不要**用它：「大盘在不在牛市」讲的是点位不是全收益，
    那一侧应留在价格指数（`fetch_index_data`）。两者混用会把一次口径修正
    变成一次策略变更。

    这个源有两个坑，都在下面处理掉，不要把原始返回直接入库
    ------------------------------------------------------
    1. **只发布收盘价**：开/高/低多数交易日为 NaN（实测 2018-01→2026-08 的
       2824 行里 87.8% 缺 OHL，且 2021-09 之前整段全空）。直接入库会让
       `data/market/daily/` 里出现一份「只有 close 是真的」的日线，而
       `price_store._overlap_matches` 只对账 close，永远发现不了。
       这里把 OHL 一律填成 close：全收益指数没有盘中报价，一根
       open=high=low=close 的平坦 K 线是它唯一自洽的表示，下游
       （`tradability` 读 `row["open"]`、任何策略）至少不会拿到 NaN。
    2. **非交易日补行**：休市日会重复前一交易日的收盘价与成交量
       （实测 2026-08-01 周六与 08-03 的 close/涨跌/成交量逐位相同，
       2015-01-01 元旦同理）。真实平盘日的「涨跌」是 0 而不是重复的非零值，
       所以按 (close, 涨跌, 成交量) 三元组与上一行完全相同来识别补行并丢弃。

    列名按**中文列名** rename，不按位置赋值：原实现写死 16 个位置名，
    akshare 换列序就会把「涨跌幅」静默映射成 `close`，而数值仍是合理量级，
    没有任何地方会报错。
    """
    import akshare as ak
    print(f"  正在从中证指数获取全收益指数 {symbol} 数据...")
    raw = ak.stock_zh_index_hist_csindex(symbol=symbol, start_date=start_date,
                                         end_date=end_date)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    colmap = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
              "收盘": "close", "涨跌": "chg", "成交量": "volume"}
    missing = [cn for cn in ("日期", "收盘") if cn not in raw.columns]
    if missing:
        raise ValueError(
            f"中证指数返回的列不含 {missing}，实际列={list(raw.columns)}；"
            "接口口径可能变了，先核对再改 colmap")
    df = raw.rename(columns=colmap)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    close = pd.to_numeric(df["close"], errors="coerce")
    chg = pd.to_numeric(df.get("chg"), errors="coerce")
    volume = pd.to_numeric(df.get("volume"), errors="coerce")

    # ② 丢掉休市补行：三元组与上一行逐位相同 → 后者是补行，保留前者
    dup = (close.eq(close.shift())
           & chg.eq(chg.shift())
           & volume.eq(volume.shift()))
    keep = ~dup.fillna(False)

    # 头部特例：请求区间的第一天若恰是休市日（如 INDEX_HISTORY_START=20150101
    # 撞上元旦），第 0 行本身就是对区间之前那根 K 线的补行——它没有"上一行"
    # 可比，`dup` 漏掉它，反而把紧随其后的**真实交易日**当成重复丢掉。
    # 所以领头的这一段重复里保留最后一根而不是第一根。
    # 已知残留：区间中段的**非连续**补行（休市日的值恰好与更早某天相同但与
    # 前一行不同）识别不了；这类只影响日期标签，数值仍是前值延续。
    if len(dup) > 1 and bool(dup.iloc[1]):
        k = 1
        while k < len(dup) and bool(dup.iloc[k]):
            k += 1
        keep.iloc[0] = False
        keep.iloc[k - 1] = True

    out = pd.DataFrame({
        # ① 只有 close 是真的，OHL 填成 close（见 docstring）
        "open": close, "high": close, "low": close, "close": close,
        "volume": volume,
    }).loc[keep]
    return _tagged(out, "csindex")


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
            return _tagged(df[cols], "akshare")
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
            return _tagged(df[cols], "baostock")
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
    return _tagged(raw[cols], "yfinance")
