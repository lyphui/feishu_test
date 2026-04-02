"""
卢麒元 MACD 牛市动能截取策略回测入口
=====================================
复用 macd_analysis.py 中的数据获取和回测引擎，
使用专属绘图函数展示：日线价格、MACD指标（含牛市背景）、资产曲线、回撤。

使用方法：
    python lu_macd_bull_analysis.py

配置文件：config/lu_macd_bull_config.ini
"""

import configparser
import os
import sys
from datetime import date as _date

# 复用数据获取和回测引擎
from macd_analysis import fetch_stock_data, run_backtest
from strategies import LuMACDBullStrategy
from utils.plotting import setup_matplotlib
from utils.market_data import fetch_index_data
from utils.bull_backtest import export_bull_daily_status, plot_bull_backtest

setup_matplotlib()


def _write_default_config(path: str) -> None:
    content = """\
[backtest]
# 股票代码（沪市6开头，深市0/3开头）
symbol     = 600519

# 股票名称（用于文件名，建议拼音或英文）
name       = maotai

# 回测区间（YYYYMMDD）
start_date = 20180101
# end_date 留空则默认使用当天日期
end_date   =

# 大盘指数代码（用于牛市判断，默认 000300 沪深300）
index_symbol = 000300

# 初始资金（元）
capital    = 100000

# 止损比例（如 0.10 表示 10%），留空则不设置
stop_loss  =

# 止盈比例（如 0.30 表示 30%），留空则不设置
take_profit =

# 图表和CSV保存目录（留空则弹窗显示，不保存CSV）
save_chart_dir = output/

# HTTP 代理（如 http://127.0.0.1:7890），留空则直连
proxy =

# ── LuMACDBull 策略专属参数 ────────────────────────────────────────────────────

# True  = 红柱缩短即卖出（截陡坡，高手模式）
# False = 等死叉再卖（保守模式）
shrink_exit = true
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("\n" + "─" * 55)
    print("  卢麒元 MACD 牛市动能截取策略回测")
    print("  数据来源：akshare（前复权）")
    print("─" * 55)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "lu_macd_bull_config.ini"
    )
    if not os.path.exists(config_path):
        print(f"  配置文件不存在，已生成默认配置：{config_path}")
        _write_default_config(config_path)

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    s = cfg["backtest"]

    symbol       = s.get("symbol", "600519").strip()
    name         = s.get("name", "stock").strip()
    index_symbol = s.get("index_symbol", "000300").strip()
    start_date   = s.get("start_date", "20180101").strip()
    end_date     = s.get("end_date", "").strip()
    if not end_date:
        end_date = _date.today().strftime("%Y%m%d")
        print(f"  end_date 未设置，默认使用今日：{end_date}")

    capital         = float(s.get("capital", "100000"))
    stop_loss_raw   = s.get("stop_loss", "").strip()
    stop_loss       = float(stop_loss_raw) if stop_loss_raw else None
    take_profit_raw = s.get("take_profit", "").strip()
    take_profit     = float(take_profit_raw) if take_profit_raw else None
    save_dir        = s.get("save_chart_dir", "").strip()
    proxy           = s.get("proxy", "").strip()
    shrink_exit     = s.get("shrink_exit", "true").strip().lower() == "true"

    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"  代理：{proxy}")

    file_stem = f"lu_bull_{name}_{symbol}_{end_date}"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_chart = os.path.join(save_dir, file_stem + ".png")
        save_csv   = os.path.join(save_dir, file_stem + ".csv")
    else:
        save_chart = None
        save_csv   = None

    print(f"  股票代码：{symbol}  大盘指数：{index_symbol}  |  {start_date} → {end_date}")
    print(f"  初始资金：{capital:,.0f}  |  止损：{stop_loss}  |  止盈：{take_profit}")
    print(f"  shrink_exit：{shrink_exit}（{'红柱缩短即离场' if shrink_exit else '等死叉再离场'}）")

    try:
        # 获取大盘指数数据（用于月线牛市判断）
        print(f"\n  正在获取大盘指数数据（{index_symbol}）...")
        index_df = fetch_index_data(index_symbol, start_date, end_date)
        if index_df.empty:
            raise ValueError(f"大盘指数 {index_symbol} 数据为空，请检查代码")
        print(f"  大盘指数获取到 {len(index_df)} 个交易日数据")

        strategy = LuMACDBullStrategy(shrink_exit=shrink_exit, index_df=index_df)

        result = run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            initial_capital=capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        plot_bull_backtest(result, save_path=save_chart)

        # 每日状态诊断表（无论是否有交易都保存）
        if save_dir:
            status_csv = os.path.join(save_dir, file_stem + "_daily_status.csv")
            export_bull_daily_status(result, status_csv)

        # 交易记录（有交易时才保存）
        if save_csv and not result["trades"].empty:
            result["trades"].to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"  交易记录已保存至：{save_csv}")
        elif save_csv:
            print("  本次回测无成交记录，不生成交易 CSV")

    except Exception as e:
        print(f"\n  ❌ 回测失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
