"""
akshare 取数稳定性探针（独立脚本，不依赖本项目任何模块）
=======================================================

按品类各打一个 akshare 接口，记录成功/失败、耗时、行数，用来复现"时好时坏"。

    python probe_akshare.py                 # 五个品类各跑一轮
    python probe_akshare.py --repeat 5      # 跑 5 轮，看失败率
    python probe_akshare.py --only 港股 商品 # 只跑指定品类
    python probe_akshare.py --timeout 30    # 单次调用超过 30 秒判为挂死

失败原因原样打印（异常类名 + 消息），不做重试、不做回退——这里要的就是
未经掩盖的原始表现。项目里的 `fetch_*` 会静默退到 baostock/yfinance，
所以正常跑流水线时看不出 akshare 到底挂了多少次。
"""

import argparse
import sys
import threading
import time
from datetime import date, timedelta

END = date.today().strftime("%Y%m%d")
START = (date.today() - timedelta(days=180)).strftime("%Y%m%d")


def a_stock():
    """A 股个股日线（东财 push2his）"""
    import akshare as ak
    return ak.stock_zh_a_hist(symbol="600519", period="daily",
                              start_date=START, end_date=END, adjust="hfq")


def etf():
    """场内 ETF 日线（东财）"""
    import akshare as ak
    return ak.fund_etf_hist_em(symbol="510300", period="daily",
                               start_date=START, end_date=END, adjust="hfq")


def index():
    """指数日线（新浪，全历史一次性返回）"""
    import akshare as ak
    return ak.stock_zh_index_daily(symbol="sh000300")


def hk():
    """港股日线（东财）"""
    import akshare as ak
    return ak.stock_hk_hist(symbol="00700", period="daily",
                            start_date=START, end_date=END, adjust="")


def commodity():
    """外盘商品：Brent 原油（新浪）"""
    import akshare as ak
    return ak.futures_foreign_hist(symbol="OIL")


PROBES = {
    "A股": a_stock,
    "ETF": etf,
    "指数": index,
    "港股": hk,
    "商品": commodity,
}


def call(fn, timeout):
    """跑一次，返回 (状态, 耗时, 信息)。超时不等它，直接判挂死继续下一个。"""
    box = {}

    def work():
        t0 = time.perf_counter()
        try:
            df = fn()
            box["ok"] = True
            box["info"] = f"{len(df)} 行  列={list(df.columns)[:6]}"
        except BaseException as e:  # noqa: BLE001 - 什么都要如实记下来
            box["ok"] = False
            box["info"] = f"{type(e).__name__}: {e}"
        box["sec"] = time.perf_counter() - t0

    t = threading.Thread(target=work, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout)
    if t.is_alive():
        return "挂死", time.perf_counter() - t0, f"超过 {timeout}s 未返回（线程仍在跑）"
    return ("成功" if box["ok"] else "失败"), box["sec"], box["info"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="akshare 各品类取数稳定性探针")
    ap.add_argument("--repeat", type=int, default=1, help="每个品类跑几轮，默认 1")
    ap.add_argument("--only", nargs="*", choices=list(PROBES), help="只跑指定品类")
    ap.add_argument("--timeout", type=float, default=60, help="单次调用超时秒数，默认 60")
    args = ap.parse_args()

    try:
        import akshare as ak
        print(f"akshare {ak.__version__}   区间 {START}~{END}\n")
    except ImportError:
        sys.exit("未安装 akshare：pip install akshare")

    names = args.only or list(PROBES)
    stats = {n: [] for n in names}

    for r in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"── 第 {r}/{args.repeat} 轮 " + "─" * 40)
        for name in names:
            status, sec, info = call(PROBES[name], args.timeout)
            stats[name].append(status)
            print(f"{name:<4} {status}  {sec:6.2f}s  {info}")
        print()

    print("═" * 60)
    for name in names:
        got = stats[name]
        ok = got.count("成功")
        bad = [s for s in got if s != "成功"]
        tail = f"  失败/挂死 {len(bad)} 次" if bad else ""
        print(f"{name:<4} 成功 {ok}/{len(got)}{tail}")


if __name__ == "__main__":
    main()
