"""
JCY 增持股票批量回测 —— 卢麒元 MACD 牛市动能截取策略
=======================================================
从 data/jcy/jcy_insights.json 读取研报数据，筛选满足以下条件的股票：
  - rating == "增持"
  - code 为 6 位纯数字的 A 股代码

同一股票多次出现时，保留 rating=增持 的最早记录。

回测参数
--------
  --stop_loss   止损比例，默认 0.20
  --take_profit 止盈比例，默认 0.10
  --capital     初始资金，默认 100000
  --index       大盘指数代码，默认 000300（沪深300）
  --shrink_exit 红柱缩短即离场，默认 True

数据与买入逻辑
--------------
  - 数据起始 = JSON 推荐日期往前推 365 天（让 MACD 充分预热，避免初始失真）
  - JSON 推荐日期之前：所有买入和卖出信号全部清零，不发生任何操作
  - JSON 推荐日期当天及之后：买入、卖出、止损、止盈均正常执行

输出目录结构
------------
  output/
    jcy_{股票代码}_{股票名称}_{推荐日期}/
      lu_bull_{股票名称}_{股票代码}_{结束日期}.png
      lu_bull_{股票名称}_{股票代码}_{结束日期}.csv          # 交易记录
      lu_bull_{股票名称}_{股票代码}_{结束日期}_daily_status.csv

用法示例
--------
    python jcy_macd_bull_batch.py
    python jcy_macd_bull_batch.py --stop_loss 0.15 --take_profit 0.12
"""

import argparse
import os
import re
import sys
from datetime import date as _date, timedelta

from macd_analysis import fetch_stock_data, run_backtest
from strategies import LuMACDBullStrategy
from utils.plotting import setup_matplotlib
from utils.market_data import fetch_index_data
from utils.bull_backtest import (
    BullStrategyAdapter,
    export_bull_daily_status,
    plot_bull_backtest,
)
from utils.jcy_common import JSON_PATH, load_candidates

setup_matplotlib()


# ── 单只股票回测 ──────────────────────────────────────────────────────────────

def backtest_one(candidate: dict, end_date: str, index_symbol: str,
                 capital: float, stop_loss: float, take_profit: float,
                 shrink_exit: bool, base_output_dir: str,
                 warmup_days: int = 600) -> bool:
    """
    对单只股票执行回测并保存结果。
    返回 True 表示成功，False 表示失败。

    warmup_days : int
        在 JSON 推荐日期前额外取多少天的历史数据，用于 MACD 预热。
        默认 365 天（约 250 个交易日，足以让 EMA-26 充分稳定）。
    """
    code             = candidate["code"]
    name             = candidate["name"]
    trade_start_date = candidate["date"]   # JSON 推荐日期，YYYYMMDD
    reason           = candidate["reason"]

    # 数据起始往前推 warmup_days 天，保证 MACD 稳定
    trade_dt   = _date(int(trade_start_date[:4]),
                       int(trade_start_date[4:6]),
                       int(trade_start_date[6:]))
    data_start = (trade_dt - timedelta(days=warmup_days)).strftime("%Y%m%d")

    # 安全化名称（用于文件/目录名）
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", name)

    # 子目录：jcy_{code}_{name}_{推荐日期}
    sub_dir   = f"jcy_{code}_{safe_name}_{trade_start_date}"
    save_dir  = os.path.join(base_output_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    file_stem  = f"lu_bull_{safe_name}_{code}_{end_date}"
    save_chart = os.path.join(save_dir, file_stem + ".png")
    save_csv   = os.path.join(save_dir, file_stem + ".csv")
    status_csv = os.path.join(save_dir, file_stem + "_daily_status.csv")

    print(f"\n  [{code}] {name}  |  推荐日期：{trade_start_date}  "
          f"数据起始：{data_start}  止损：{stop_loss}  止盈：{take_profit}")
    print(f"    推荐原因：{reason}")

    try:
        print(f"    获取大盘指数 {index_symbol} 数据（{data_start} → {end_date}）...")
        index_df = fetch_index_data(index_symbol, data_start, end_date)
        if index_df.empty:
            raise ValueError(f"大盘指数数据为空")

        inner_strategy = LuMACDBullStrategy(shrink_exit=shrink_exit)
        strategy       = BullStrategyAdapter(inner_strategy, index_df,
                                             trade_start_date=trade_start_date)

        result = run_backtest(
            symbol=code,
            start_date=data_start,        # 含预热期，保证 MACD 稳定
            end_date=end_date,
            strategy=strategy,
            initial_capital=capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        print(f"    总收益：{result['total_return']:+.2f}%  "
              f"基准：{result['benchmark_return']:+.2f}%  "
              f"夏普：{result['sharpe_ratio']:.2f}  "
              f"最大回撤：{result['max_drawdown']:.2f}%")

        plot_bull_backtest(result, save_path=save_chart,
                           trade_start_date=trade_start_date)
        export_bull_daily_status(result, status_csv)

        if not result["trades"].empty:
            result["trades"].to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"    交易记录已保存至：{save_csv}")
        else:
            print("    本次回测无成交记录")

        return True

    except Exception as e:
        print(f"    ❌ 回测失败：{e}")
        import traceback
        traceback.print_exc()
        return False


# ── 主入口 ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="JCY 增持股票批量回测（卢麒元 MACD 牛市动能截取策略）"
    )
    parser.add_argument("--stop_loss",   type=float, default=0.20,
                        help="止损比例，默认 0.20")
    parser.add_argument("--take_profit", type=float, default=0.10,
                        help="止盈比例，默认 0.10")
    parser.add_argument("--capital",     type=float, default=100000,
                        help="初始资金，默认 100000")
    parser.add_argument("--index",       type=str,   default="000300",
                        help="大盘指数代码，默认 000300（沪深300）")
    parser.add_argument("--shrink_exit", type=lambda x: x.lower() != "false",
                        default=True,
                        help="红柱缩短即离场，默认 True；传 false 则等死叉")
    parser.add_argument("--output",      type=str,   default="output",
                        help="输出根目录，默认 output/")
    return parser.parse_args()


def main():
    args = parse_args()

    end_date = _date.today().strftime("%Y%m%d")

    print("\n" + "─" * 60)
    print("  JCY 增持股票批量回测 —— 卢麒元 MACD 牛市动能截取策略")
    print("─" * 60)
    print(f"  数据来源：{JSON_PATH}")
    print(f"  止损：{args.stop_loss}  止盈：{args.take_profit}  "
          f"资金：{args.capital:,.0f}  大盘：{args.index}")
    print(f"  结束日期：{end_date}  输出目录：{args.output}/")
    print("─" * 60)

    # 加载候选股票
    if not os.path.exists(JSON_PATH):
        print(f"  ❌ 找不到 JSON 文件：{JSON_PATH}")
        sys.exit(1)

    candidates = load_candidates(JSON_PATH)
    if not candidates:
        print("  ❌ 未找到满足条件的增持 A 股，请检查 JSON 数据")
        sys.exit(1)

    print(f"\n  共找到 {len(candidates)} 只增持 A 股（已去重，保留 rating=增持 的最早记录）：")
    for c in candidates:
        print(f"    {c['code']}  {c['name']:8s}  起始日期：{c['date'][:4]}-{c['date'][4:6]}-{c['date'][6:]}")

    os.makedirs(args.output, exist_ok=True)

    # 逐只回测
    success_count = 0
    fail_count    = 0
    for candidate in candidates:
        ok = backtest_one(
            candidate     = candidate,
            end_date      = end_date,
            index_symbol  = args.index,
            capital       = args.capital,
            stop_loss     = args.stop_loss,
            take_profit   = args.take_profit,
            shrink_exit   = args.shrink_exit,
            base_output_dir = args.output,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "─" * 60)
    print(f"  批量回测完成：成功 {success_count} 只，失败 {fail_count} 只")
    print(f"  结果已保存至：{os.path.abspath(args.output)}/")
    print("─" * 60)


if __name__ == "__main__":
    main()
