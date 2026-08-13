"""
卢麒元 MACD 三级确认策略回测入口
==================================
复用 macd_analysis.py 中的数据获取和回测引擎，
使用专属绘图函数将月线/周线/日线 MACD 分三个子图展示。

使用方法：
    python -m backtest.scripts.backtest_lu_macd

配置文件：backtest/presets/lu_macd_config.ini
"""

import sys

# 复用回测引擎与共享配置层；绘图与 CSV 导出已拆到 reports/lu_macd_report.py
from backtest.engine import run_backtest
from backtest.config import load_backtest_config, execution_kwargs, OutputPaths

from backtest.strategies import LuMACDStrategy
from backtest.reports.lu_macd_report import export_daily_status, plot_lu_backtest
from backtest.lib.console import use_utf8


_DEFAULT_INI = """\
[backtest]
# 股票代码（沪市6开头，深市0/3开头）
symbol     = 600519

# 股票名称（用于文件名，建议拼音或英文）
name       = maotai

# 回测区间（YYYYMMDD）
# 注意：月线+周线需要足够热身数据，start_date 建议比实际分析起点早 3 年以上
start_date = 20180101
# end_date 留空则默认使用当天日期
end_date   =

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

# ── 成交成本与交易约束（留空即用默认值，见 backtest/config.py）────────────────
# 券商佣金费率（双边），默认 0.0003
commission_rate =
# 单笔最低佣金（元），默认 5
min_commission =
# 印花税（仅卖出），默认 0.001
stamp_duty =
# 单边滑点，默认 0.001；设 0 可与无滑点结果对比
slippage =
# 是否模拟涨跌停/停牌无法成交，默认 true
limit_move_check =
# 信号因涨跌停未成交时最多顺延几个交易日，默认 3
max_pending_days =

# ── LuMACD 策略专属参数 ──────────────────────────────────────────────────────

# 量能放大判断窗口（周线根数），与前 N 周均量比较
vol_window = 4

# True = 缺少 volume 数据时抛出异常；False = 降级运行（跳过量能验证）
require_volume = false
"""


def _enrich_trades(trades, df):
    """
    在 trades DataFrame 中追加「参考依据」列，
    说明每笔操作当日的三级确认状态、MACD 数值和价格阶段。
    """
    if trades.empty:
        return trades

    records = []
    for _, t in trades.iterrows():
        action = t["action"]
        date   = t["date"]

        # 从 df 取当日数据（日期可能不完全对齐，取最近）
        row = df.loc[date] if date in df.index else df.iloc[df.index.get_indexer([date], method="nearest")[0]]

        dif  = row.get("DIF",  float("nan"))
        dea  = row.get("DEA",  float("nan"))
        difw = row.get("DIF_W", float("nan"))
        deaw = row.get("DEA_W", float("nan"))
        difm = row.get("DIF_M", float("nan"))
        deam = row.get("DEA_M", float("nan"))
        monthly = row.get("monthly_confirmed", False)
        weekly  = row.get("weekly_confirmed",  False)
        phase   = row.get("phase", "—")

        if action == "买入":
            basis = (
                f"三级确认触发 | "
                f"月线确认={monthly} DIF_M={difm:.4f}>DEA_M={deam:.4f} | "
                f"周线确认={weekly} DIF_W={difw:.4f}>DEA_W={deaw:.4f} | "
                f"日线DIF={dif:.4f} DEA={dea:.4f} | "
                f"价格阶段={phase}"
            )
        elif action == "卖出":
            basis = (
                f"日线死叉 DIF({dif:.4f}) < DEA({dea:.4f}) | "
                f"价格阶段={phase}"
            )
        elif action == "止损卖出":
            pct = t.get("return_pct", float("nan"))
            basis = f"止损触发 收益={pct:.2f}% | 日线DIF={dif:.4f} DEA={dea:.4f} | 价格阶段={phase}"
        elif action == "止盈卖出":
            pct = t.get("return_pct", float("nan"))
            basis = f"止盈触发 收益={pct:.2f}% | 日线DIF={dif:.4f} DEA={dea:.4f} | 价格阶段={phase}"
        elif action == "期末清仓":
            basis = f"回测结束强制清仓 | 价格阶段={phase}"
        else:
            basis = "—"

        records.append(basis)

    result = trades.copy()
    result.insert(result.columns.get_loc("action") + 1, "参考依据", records)
    return result


def main():
    use_utf8()
    print("\n" + "─" * 55)
    print("  卢麒元 MACD 三级确认策略回测")
    print("  数据来源：akshare（前复权）")
    print("─" * 55)

    cfg   = load_backtest_config("lu_macd_config.ini", defaults=_DEFAULT_INI)
    paths = OutputPaths(cfg.save_dir, "lu_macd", cfg.name, cfg.symbol, cfg.end_date)

    # LuMACD 专属参数
    vol_window        = cfg.get_int("vol_window", 4)
    require_volume    = cfg.get_bool("require_volume", False)
    require_green_bar = cfg.get_bool("require_green_bar", True)

    print(f"  股票代码：{cfg.symbol}  |  {cfg.start_date} → {cfg.end_date}")
    print(f"  初始资金：{cfg.capital:,.0f}  |  止损：{cfg.stop_loss}  |  止盈：{cfg.take_profit}")
    print(f"  vol_window：{vol_window}  |  require_volume：{require_volume}  |  require_green_bar：{require_green_bar}")

    try:
        strategy = LuMACDStrategy(
            vol_window=vol_window,
            require_volume=require_volume,
            require_green_bar=require_green_bar,
        )
        result = run_backtest(
            symbol=cfg.symbol,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            strategy=strategy,
            initial_capital=cfg.capital,
            stop_loss=cfg.stop_loss,
            take_profit=cfg.take_profit,
            **execution_kwargs(cfg),
            verbose=True,
        )

        plot_lu_backtest(result, save_path=paths.chart)

        # 每日状态诊断表（无论是否有交易都保存）
        if paths.status:
            export_daily_status(result, paths.status)

        # 交易记录（有交易时才保存）
        if paths.csv and not result["trades"].empty:
            enriched = _enrich_trades(result["trades"], result["df"])
            enriched.to_csv(paths.csv, index=False, encoding="utf-8-sig")
            print(f"  交易记录已保存至：{paths.csv}")
        elif paths.csv:
            print("  本次回测无成交记录，不生成交易 CSV")

    except Exception as e:
        print(f"\n  ❌ 回测失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
