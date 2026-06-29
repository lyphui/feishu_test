"""
A股 MACD 策略回测工具（CLI 入口）
==================================
回测引擎已抽取至 engine.py；本文件仅保留 MACDStrategy 的命令行入口，
并 re-export 引擎符号以兼容历史导入（from macd_analysis import run_backtest, ...）。

使用方法：
    python backtest/macd_analysis.py --config jxty_jcy_260104.ini

或直接调用引擎：
    from engine import run_backtest
    result = run_backtest("600519", "20200101", "20241231")
"""

import sys

# 向后兼容 re-export：历史代码 from macd_analysis import run_backtest / fetch_stock_data
from engine import run_backtest, plot_backtest, fetch_stock_data  # noqa: F401

from config import load_backtest_config, OutputPaths


_DEFAULT_INI = """\
[backtest]
# 股票代码（沪市6开头，深市0/3开头）
symbol     = 600519

# 股票名称（用于文件名，建议拼音或英文）
name       = maotai

# 回测区间（YYYYMMDD）
start_date = 20200101
# end_date 留空则默认使用当天日期
end_date   =

# 初始资金（元）
capital    = 100000

# 止损比例（如 0.08 表示 8%），留空则不设置
stop_loss  =

# 止盈比例（如 0.20 表示 20%），留空则不设置
take_profit =

# 图表和CSV保存目录（留空则弹窗显示，不保存CSV）
save_chart_dir = output/

# HTTP 代理（如 http://127.0.0.1:7890），留空则直连
proxy =
"""


def main():
    import argparse
    from strategies import MACDStrategy

    parser = argparse.ArgumentParser(description="A股 MACD 策略回测工具")
    parser.add_argument("--config", type=str, default="jxty_jcy_260104.ini",
                        help="配置文件名（位于 backtest/presets/ 目录下），默认 jxty_jcy_260104.ini")
    args = parser.parse_args()

    print("\n" + "─"*55)
    print("  A股策略回测工具")
    print("  数据来源：akshare（前复权）")
    print("─"*55)

    cfg   = load_backtest_config(args.config, defaults=_DEFAULT_INI)
    paths = OutputPaths(cfg.save_dir, "macd", cfg.name, cfg.symbol, cfg.end_date)

    print(f"  股票代码：{cfg.symbol}  |  {cfg.start_date} → {cfg.end_date}")
    print(f"  初始资金：{cfg.capital:,.0f}  |  止损：{cfg.stop_loss}  |  止盈：{cfg.take_profit}")

    try:
        result = run_backtest(
            symbol=cfg.symbol,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            strategy=MACDStrategy(),
            initial_capital=cfg.capital,
            stop_loss=cfg.stop_loss,
            take_profit=cfg.take_profit,
            verbose=True,
        )

        plot_backtest(result, save_path=paths.chart)

        # 保存交易记录
        if paths.csv and not result["trades"].empty:
            result["trades"].to_csv(paths.csv, index=False, encoding="utf-8-sig")
            print(f"  交易记录已保存至：{paths.csv}")

    except Exception as e:
        print(f"\n  ❌ 回测失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
