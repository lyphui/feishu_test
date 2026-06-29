"""
卢麒元 MACD 牛市动能截取策略回测入口
=====================================
复用 macd_analysis.py 中的数据获取和回测引擎，
使用专属绘图函数展示：日线价格、MACD指标（含牛市背景）、资产曲线、回撤。

使用方法：
    python lu_macd_bull_analysis.py

配置文件：backtest/presets/lu_macd_bull_config.ini
"""

import sys

# 复用回测引擎、共享配置层与报告输出
from engine import run_backtest
from config import load_backtest_config, OutputPaths
from bull_report import export_bull_daily_status, plot_bull_backtest
from strategies import LuMACDBullStrategy
from lib.plotting import setup_matplotlib
from lib.market_data import fetch_index_data

setup_matplotlib()


_DEFAULT_INI = """\
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


def main():
    print("\n" + "─" * 55)
    print("  卢麒元 MACD 牛市动能截取策略回测")
    print("  数据来源：akshare（前复权）")
    print("─" * 55)

    cfg   = load_backtest_config("lu_macd_bull_config.ini", defaults=_DEFAULT_INI)
    paths = OutputPaths(cfg.save_dir, "lu_bull", cfg.name, cfg.symbol, cfg.end_date)
    shrink_exit = cfg.get_bool("shrink_exit", True)

    print(f"  股票代码：{cfg.symbol}  大盘指数：{cfg.index_symbol}  |  {cfg.start_date} → {cfg.end_date}")
    print(f"  初始资金：{cfg.capital:,.0f}  |  止损：{cfg.stop_loss}  |  止盈：{cfg.take_profit}")
    print(f"  shrink_exit：{shrink_exit}（{'红柱缩短即离场' if shrink_exit else '等死叉再离场'}）")

    try:
        # 获取大盘指数数据（用于月线牛市判断）
        print(f"\n  正在获取大盘指数数据（{cfg.index_symbol}）...")
        index_df = fetch_index_data(cfg.index_symbol, cfg.start_date, cfg.end_date)
        if index_df.empty:
            raise ValueError(f"大盘指数 {cfg.index_symbol} 数据为空，请检查代码")
        print(f"  大盘指数获取到 {len(index_df)} 个交易日数据")

        strategy = LuMACDBullStrategy(shrink_exit=shrink_exit, index_df=index_df)

        result = run_backtest(
            symbol=cfg.symbol,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            strategy=strategy,
            initial_capital=cfg.capital,
            stop_loss=cfg.stop_loss,
            take_profit=cfg.take_profit,
            verbose=True,
        )

        plot_bull_backtest(result, save_path=paths.chart)

        # 每日状态诊断表（无论是否有交易都保存）
        if paths.status:
            export_bull_daily_status(result, paths.status)

        # 交易记录（有交易时才保存）
        if paths.csv and not result["trades"].empty:
            result["trades"].to_csv(paths.csv, index=False, encoding="utf-8-sig")
            print(f"  交易记录已保存至：{paths.csv}")
        elif paths.csv:
            print("  本次回测无成交记录，不生成交易 CSV")

    except Exception as e:
        print(f"\n  ❌ 回测失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
