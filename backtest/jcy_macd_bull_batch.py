"""
JCY 推荐股票批量回测 —— 卢麒元 MACD 牛市动能截取策略
=======================================================
从 data/jcy/jcy_insights.json 读取研报数据，筛选满足以下条件的股票：
  - rating 落在 --ratings（默认「买入,增持」）
  - code 为 6 位纯数字的 A 股代码

同一股票多次出现时，保留首次落入该评级集合的最早记录。

为什么默认收「买入 + 增持」而不只收「增持」
------------------------------------------
数据里有 51 只 A 股是「买入」（Strong Buy）、239 只是「增持」。旧实现只收
增持，等于用中间档定义股票池、把最强的那档整个丢在回测之外。

对照池（--control）
-------------------
只跑看多池，最多能说明"这批票在这段行情里涨没涨"——牛市随便挑一篮子都涨，
回答不了"这套评级有没有区分度"。传 --control 减持,回避 会把看空池用**完全
相同的策略与参数**再跑一遍，并输出两池的同口径对照表。看多池明显更好，
"买入/增持"这个标签才带信息。

回测参数
--------
  --ratings     进入回测的评级，逗号分隔，默认 买入,增持
  --control     看空对照池评级，逗号分隔，如 减持,回避；留空不跑对照
  --stop_loss   止损比例，默认 0.10；传 none 关闭
  --take_profit 止盈比例，默认关闭（none）；传数值则启用
  --capital     初始资金，默认 100000
  --index       大盘指数代码，默认 000300（沪深300）
  --shrink_exit 红柱缩短即离场，默认 True

为什么默认不设止盈
------------------
本策略的立意是"截取 MACD 最陡峭的部分"，而 shrink_exit=True 本身已经是
一套动能衰减离场规则。再叠加一个固定比例止盈，等于在陡坡刚起来时把它切断：
策略最该赚到的那段被系统性砍掉，留下一堆小赢加少数大亏。
早先的默认值是 stop_loss=0.20 / take_profit=0.10——1:2 的反向盈亏比，
与策略前提直接冲突。现改为止损 0.10、止盈交给 shrink_exit。
要验证这个选择，跑：
    python backtest/param_sweep.py --axis stop_loss take_profit

数据与买入逻辑
--------------
  - 个股数据起始 = JSON 推荐日期往前推 600 天（让日线 MACD 充分预热）
  - 大盘指数从 config.INDEX_HISTORY_START 起取，与候选池无关（月线 MACD 预热）
  - JSON 推荐日期之前：所有买入和卖出信号全部清零，不发生任何操作
  - JSON 推荐日期当天及之后：买入、卖出、止损、止盈均正常执行

输出目录结构
------------
  output/
    summary.csv                    # 横截面汇总：每股一行，按日均超额排序
    summary_portfolio.csv / .png   # 等权组合净值 vs 大盘
    summary_rating_compare.csv     # 评级区分度对照表（仅 --control 时）
    jcy_{股票代码}_{股票名称}_{推荐日期}/
      lu_bull_{股票名称}_{股票代码}_{结束日期}.png
      lu_bull_{股票名称}_{股票代码}_{结束日期}.csv          # 交易记录
      lu_bull_{股票名称}_{股票代码}_{结束日期}_daily_status.csv
    control/                       # 对照池（仅 --control 时），结构同上

用法示例
--------
    python jcy_macd_bull_batch.py
    python jcy_macd_bull_batch.py --control 减持,回避       # 带看空对照组
    python jcy_macd_bull_batch.py --ratings 买入            # 只跑最强评级
    python jcy_macd_bull_batch.py --stop_loss 0.15 --take_profit 0.12
"""

import argparse
import os
import sys
from datetime import date as _date, timedelta

from engine import run_backtest, fmt_sharpe
from config import OutputPaths, index_history_start
from bull_report import export_bull_daily_status, plot_bull_backtest
from batch_report import (result_to_row, normalized_equity, write_batch_report,
                          compare_rating_pools)
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):      # backtest/ 与仓库根都要在 path 上
    if _p not in sys.path:
        sys.path.insert(0, _p)

from strategies import LuMACDBullStrategy
from lib.plotting import setup_matplotlib
from lib.market_data import fetch_index_data
from lib.bull_backtest import BullStrategyAdapter
from lib.console import use_utf8
from jcy.lib.common import (JSON_PATH, LONG_RATINGS, CONTROL_RATINGS,
                            load_candidates, parse_ratings)

setup_matplotlib()


# ── 止损/止盈参数处理 ─────────────────────────────────────────────────────────

def _fmt_ratio(v: float | None) -> str:
    return "关闭" if v is None else f"{v:.0%}"


def _ratio_or_none(value: str) -> float | None:
    """止损/止盈参数：接受比例数值，或 none/off/空字符串表示不启用。"""
    if value is None or value.strip().lower() in ("", "none", "off", "no"):
        return None
    return float(value)


# ── 单只股票回测 ──────────────────────────────────────────────────────────────

def backtest_one(candidate: dict, end_date: str, index_df,
                 capital: float,
                 stop_loss: float | None, take_profit: float | None,
                 shrink_exit: bool, base_output_dir: str,
                 warmup_days: int = 600) -> dict | None:
    """
    对单只股票执行回测并保存结果。
    返回 run_backtest 的结果 dict；失败返回 None。

    index_df : 大盘指数日线（由调用方统一取一次后复用，避免 N 次重复请求）

    warmup_days : int
        在 JSON 推荐日期前额外取多少个**自然日**的个股历史数据，用于指标预热。
        默认 600 天（约 400 个交易日），日线 EMA-26 几十根就稳定，足够有余。

        只管个股。牛市过滤器用的大盘**月线** MACD 另有一套预热——指数一律从
        `config.INDEX_HISTORY_START` 起取，与本参数、与候选池都无关（月线
        EMA-26 要几十根月线才收敛，靠这 600 天远远不够）。

    预热期只喂数据不交易（BullStrategyAdapter 清零信号），且通过
    run_backtest(eval_start=...) 排除在收益/回撤/夏普/基准统计之外。
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
    safe_name = OutputPaths.safe(name)

    # 子目录：jcy_{code}_{name}_{推荐日期}，文件名 stem：lu_bull_{name}_{code}_{结束日期}
    sub_dir = f"jcy_{code}_{safe_name}_{trade_start_date}"
    paths   = OutputPaths(os.path.join(base_output_dir, sub_dir),
                          "lu_bull", safe_name, code, end_date)
    save_chart = paths.chart
    save_csv   = paths.csv
    status_csv = paths.status

    print(f"\n  [{code}] {name}  |  推荐日期：{trade_start_date}  "
          f"数据起始：{data_start}  止损：{_fmt_ratio(stop_loss)}  "
          f"止盈：{_fmt_ratio(take_profit)}")
    print(f"    推荐原因：{reason}")

    try:
        if index_df is None or index_df.empty:
            raise ValueError("大盘指数数据为空")

        inner_strategy = LuMACDBullStrategy(shrink_exit=shrink_exit)
        strategy       = BullStrategyAdapter(inner_strategy, index_df,
                                             trade_start_date=trade_start_date)

        result = run_backtest(
            symbol=code,
            start_date=data_start,        # 含预热期，保证指标稳定
            end_date=end_date,
            strategy=strategy,
            initial_capital=capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
            eval_start=trade_start_date,  # 预热期不计入收益/回撤/夏普/基准
        )

        blocked = result.get("blocked_trades")
        blocked_txt = ("" if blocked is None or blocked.empty
                       else f"  受阻：{len(blocked)}次")
        print(f"    总收益：{result['total_return']:+.2f}%  "
              f"基准：{result['benchmark_return']:+.2f}%  "
              f"夏普：{fmt_sharpe(result['sharpe_ratio'])}  "
              f"最大回撤：{result['max_drawdown']:.2f}%{blocked_txt}")

        plot_bull_backtest(result, save_path=save_chart,
                           trade_start_date=trade_start_date)
        export_bull_daily_status(result, status_csv)

        if not result["trades"].empty:
            result["trades"].to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"    交易记录已保存至：{save_csv}")
        else:
            print("    本次回测无成交记录")

        return result

    except Exception as e:
        print(f"    ❌ 回测失败：{e}")
        import traceback
        traceback.print_exc()
        return None


# ── 主入口 ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="JCY 推荐股票批量回测（卢麒元 MACD 牛市动能截取策略）"
    )
    parser.add_argument("--ratings",     type=parse_ratings,
                        default=LONG_RATINGS,
                        help=f"进入回测的评级，逗号分隔，默认 {','.join(LONG_RATINGS)}")
    parser.add_argument("--control",     type=parse_ratings, default=(),
                        help=f"看空对照池评级，逗号分隔（建议 "
                             f"{','.join(CONTROL_RATINGS)}）；留空则不跑对照")
    parser.add_argument("--stop_loss",   type=_ratio_or_none, default=0.10,
                        help="止损比例，默认 0.10；传 none/空 关闭")
    parser.add_argument("--take_profit", type=_ratio_or_none, default=None,
                        help="止盈比例，默认关闭（由 shrink_exit 决定离场）")
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


WARMUP_DAYS = 600      # 个股行情在推荐日前多取的自然日数（指标预热）


def run_pool(candidates: list[dict], label: str, output_dir: str,
             end_date: str, index_df, args) -> "tuple[object, dict]":
    """把一个候选池整体跑完，写出该池的汇总，返回 (summary_df, curves)。

    正向池与对照池共用这个函数——策略、参数、成本、指数完全相同，
    两池的差异才只来自评级本身。
    """
    print("\n" + "─" * 60)
    print(f"  {label}：{len(candidates)} 只  →  {output_dir}/")
    print("─" * 60)
    for c in candidates:
        print(f"    {c['code']}  {c['name']:8s}  [{c.get('rating', '')}]  "
              f"起始日期：{c['date'][:4]}-{c['date'][4:6]}-{c['date'][6:]}")

    rows: list[dict] = []
    curves: dict[str, "object"] = {}
    fail_count = 0
    for candidate in candidates:
        result = backtest_one(
            candidate     = candidate,
            end_date      = end_date,
            index_df      = index_df,
            capital       = args.capital,
            stop_loss     = args.stop_loss,
            take_profit   = args.take_profit,
            shrink_exit   = args.shrink_exit,
            base_output_dir = output_dir,
            warmup_days   = WARMUP_DAYS,
        )
        if result is None:
            fail_count += 1
            continue
        rows.append(result_to_row(candidate, result, index_df=index_df))
        curves[f"{candidate['code']} {candidate['name']}"] = normalized_equity(result)

    print("\n" + "─" * 60)
    print(f"  {label} 回测完成：成功 {len(rows)} 只，失败 {fail_count} 只")
    if fail_count:
        print(f"  ⚠️ 失败的 {fail_count} 只（多为退市/长期停牌/取数失败）不进汇总，"
              "剩下的样本带幸存者偏差")
    print("─" * 60)

    summary = write_batch_report(rows, curves, output_dir,
                                 index_df=index_df, index_name=args.index,
                                 label=label)
    return summary, curves


def main():
    use_utf8()
    args = parse_args()

    if not args.ratings:
        print("  ❌ --ratings 不能为空")
        sys.exit(1)

    end_date = _date.today().strftime("%Y%m%d")
    long_label    = f"看多池 {'/'.join(args.ratings)}"
    control_label = f"对照池 {'/'.join(args.control)}" if args.control else ""

    print("\n" + "─" * 60)
    print("  JCY 推荐股票批量回测 —— 卢麒元 MACD 牛市动能截取策略")
    print("─" * 60)
    print(f"  数据来源：{JSON_PATH}")
    print(f"  评级：{','.join(args.ratings)}"
          f"{'   对照：' + ','.join(args.control) if args.control else ''}")
    print(f"  止损：{_fmt_ratio(args.stop_loss)}  "
          f"止盈：{_fmt_ratio(args.take_profit)}  "
          f"资金：{args.capital:,.0f}  大盘：{args.index}")
    print(f"  结束日期：{end_date}  输出目录：{args.output}/")
    print("─" * 60)

    # 加载候选股票
    if not os.path.exists(JSON_PATH):
        print(f"  ❌ 找不到 JSON 文件：{JSON_PATH}")
        sys.exit(1)

    candidates = load_candidates(JSON_PATH, ratings=args.ratings)
    if not candidates:
        print(f"  ❌ 未找到评级为 {','.join(args.ratings)} 的 A 股，请检查 JSON 数据")
        sys.exit(1)

    control_candidates = (load_candidates(JSON_PATH, ratings=args.control)
                          if args.control else [])
    if args.control and not control_candidates:
        print(f"  ⚠️ 未找到评级为 {','.join(args.control)} 的 A 股，跳过对照池")

    os.makedirs(args.output, exist_ok=True)

    # 大盘指数只取一次：所有个股、两个池子共用。起点是**绝对日期**，不随候选池
    # 变化——月线 EMA(26) 需要几十根月线才收敛，若按「最早推荐日 − 600 天」取，
    # 加一篇更早的文章就会改写全部个股的 bull_market 历史（见 config.py）。
    index_start = index_history_start()
    print(f"\n  获取大盘指数 {args.index} 数据（{index_start} → {end_date}）...")
    try:
        index_df = fetch_index_data(args.index, index_start, end_date)
    except Exception as e:
        print(f"  ❌ 大盘指数获取失败：{e}")
        sys.exit(1)

    summary, _ = run_pool(candidates, long_label, args.output,
                          end_date, index_df, args)

    if control_candidates:
        control_dir = os.path.join(args.output, "control")
        control_summary, _ = run_pool(control_candidates, control_label,
                                      control_dir, end_date, index_df, args)
        if not summary.empty and not control_summary.empty:
            compare_rating_pools(summary, control_summary, args.output,
                                 long_label=long_label,
                                 control_label=control_label)

    print(f"\n  全部结果已保存至：{os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
