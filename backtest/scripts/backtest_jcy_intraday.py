"""
JCY 日线信号 + 分时择时（多周期共振）
======================================
当日线 LuMACDBull 策略触发买/卖信号时，
获取执行日的分时行情（默认 30 分钟 K 线），
在分时级别应用 MACD 动能分析，推荐最佳操作时间窗口。

设计思路（多周期共振）：
  日线 → 确认方向（买 / 卖）
  分时 → 确认时机（几点操作）

  日线金叉 + 红柱拉长 → 确认买入方向
  分时金叉 + 红柱拉长 → 确认具体入场时机
  两者同向共振 = 最优操作点

常见问题：日线满足条件但分时找不到好时机吗？
──────────────────────────────────────────────
  会，常见场景：
  1. 高开低走：日线金叉收盘，次日跳空高开后立刻回落，分时全天无拉长窗口
  2. 全天震荡：分时 MACD 反复穿越零轴，无持续方向，找不到共振点
  3. 信号滞后：日线信号 T 日收盘确认，T+1 开盘已涨 3-5%，无安全价位可进
  4. 卖出跌停：一字跌停，分时无成交，"好的卖出时机"已不存在
  5. 小盘流动性不足：分时 MACD 因稀疏成交而失真

  应对建议（本脚本会自动提示，两条旧建议都已被实测推翻，见下方「实测」）：
    买入无 GO 窗口 → 次日开盘集合竞价直接买，不要挂低价等回调
    卖出无 GO 窗口 → 尾盘 15:00 前卖掉，不要留到次日集合竞价

买入信号的完整链路（代码走向）
──────────────────────────────
  ① 日线定方向  `_fetch_daily_signals()`
       fetch_stock_data(code, trade_start − WARMUP_DAYS, today)  → 日线（hfq）
       fetch_index_data(index_symbol, ...)                       → 大盘
       LuMACDBullStrategy(shrink_exit=True) 包进 BullStrategyAdapter(index_df,
       trade_start_date)，adapter 把 trade_start 之前的信号清零（防未来数据）
       → df_sig["signal"]==1 即买入信号；再按 lookback_days 截出最近的信号日
  ② 定执行日    `_determine_exec_date()`   next=信号次日（默认）/ same=信号当日
  ③ 分时定时机  `_analyze_single_signal()`
       fetch_intraday_indexed(code, exec_date − INTRADAY_WARMUP_DAYS, exec_date,
       period)（lib/intraday_store，默认 qfq）
       lib/execution.intraday_macd() 在**连续跨日**序列上算 30min MACD(12/26/9)
       （唯一真值源，go_buy/go_sell 也在这里）
       切出执行日 8 根 → lib/execution.classify_timing(action="buy")
         GO    = go_buy（hist_expanding 且 DIF > DEA）
         WAIT  = DIF > DEA 但红柱未拉长
         AVOID = DIF <= DEA
       exec_price（lib/execution.executable_price()，买卖同口径）：
         有 GO → 首个 GO 柱的**下一根 K 线开盘价**（GO 在最后一根时退化为收盘价）
         无 GO → 当日最后一柱收盘价
  ④ 落仓位      `PositionTracker.run(df_sig, intraday_map)`（lib/position_tracker）
       exec_date == 信号日      → 当日建仓
       否则                    → 挂 _pending[exec_date]，到日执行
       exec_price 为 None（分时数据缺失）→ 兜底用日线收盘价，仍然建仓
       信号日不在 intraday_map（历史信号）→ 回退用日线收盘价当日执行
  ⑤ 建仓规模    `_buy_initial()`  DIF<0 → 1/3 可用资金；DIF≥0 → 1/2
                `_buy_add()`      次日红柱续拉长 → 补满仓
                初仓阶段遇 红柱缩短 / DIF<DEA / DIF<0 → 全退，不等满仓

  两条必须守住的口径（2026-08-09 修正，详见 `_analyze_single_signal` docstring）：
    * **可成交价不是信号柱的收盘价。**要等这根 K 线走完才能判定它是 GO，
      那一刻收盘价已成历史。能下的第一个单成交在**下一根的开盘**。
    * **分时只决定「几点买」，不决定「买不买」。**旧实现在"买入且无 GO"时
      直接跳过建仓，等于让执行层条件充当隐式策略过滤器——无 GO 占四成日子，
      影响远超它该有的分量。现在一律建仓，只是成交价不同。

卖出信号的链路（与买入共用 ①③，仓位规则不同）
──────────────────────────────────────────────
  三级递进离场，只有**第一级**走分时择时，后两级直接用日线收盘价：

    一级  红柱缩短      → `_sell_portion(level=1)`  减半仓   ← 有分时择时
    二级  DIF < DEA 死叉 → `_sell_portion(level=2)`  再减    ← 无，日线收盘价
    三级  DIF < 0        → `_sell_remaining()`       清空    ← 无，日线收盘价
    另：初仓阶段遇 红柱缩短 / DIF<DEA / DIF<0 直接全退（不分级）

  分时侧 `classify_timing(action="sell")` 的 GO 定义与买入**不对称**：
    GO   = hist_shrinking（红柱缩短）**或** death_cross（DIF 下穿 DEA）
    WAIT = 其余；没有 AVOID 这一档
  买侧要求方向与动能同时成立，卖侧只要动能转弱就放行，死叉那一路根本不看
  红柱正负。所以卖侧 GO 天数占比天然更高（本池 64% vs 买侧 57%），
  两侧的 bp 不能横向比较。`lib.execution.intraday_macd()` 的 `go_sell` 同式。

执行时点：统一 T+1（2026-08-09 修正）
────────────────────────────────────
  日线的 signal / 红柱缩短 / 死叉 / DIF<0 **全都要等当日收盘才算得出来**，
  所以每一条都排到下一个交易日成交，买卖两侧、`lookback` 窗口内外一视同仁。
  实现见 `PositionTracker._place()` / `_execute()`。

  修掉的是三套口径并存：窗口内的信号 T+1 成交、窗口外的当日收盘成交、
  二三级卖出一律当日收盘成交。后两者是零延迟假设——在知道结果的同一刻
  按那个价成交；而分界线取决于 `--lookback`，改一个只该影响打印的参数
  就能改变历史收益。现在 `intraday_map` **只换成交价、不换成交日**。

  唯一保留的当日成交是 `--exec_day same`：那是使用者明确选的盘中实时口径
  （信号当天就在盘中发现并操作），不是回测默认路径。

实测：这套择时到底有没有用（JCY 抽样池 45 只 × 46728 个股票-交易日，2022-01~2026-08）
──────────────────────────────────────────────────────────────────────────
复现：`python -m backtest.scripts.compare_exec_plans --universe jcy --side both --limit 45`
（定种子抽样，度量层在 `lib/execution.py`，分时缓存在 `data/market/intraday/`）
下表一律**优势bp**：正 = 优于当日 VWAP（买得更便宜 / 卖得更贵）。

                          买入          卖出
    A 开盘集合竞价        +16.7 ★最优   −16.7 ✗最差
    B 尾盘 15:00           +5.4         −5.4
    C GO 窗口              +9.4         −3.2 ★最优
    D 限价挂开盘∓0.5%     −46.6 ✗最差  −72.9 ✗最差（成交率均 66%）
    逐股最优计票        开盘 27/45     GO 29/45（尾盘 12、开盘 4）

  **两侧不对称，别把买入结论翻过来用。**A、B 两行买卖数值互为相反数，因为它们
  测的是同一个日内形状：开盘价比 VWAP 低 16.7bp——买入正因这个"便宜"而首选开盘，
  卖出遇上同一个便宜就是净亏。所以开盘集合竞价是**买入最优、卖出最差**。

  **买入侧：GO 窗口不如直接开盘买**（差约 7bp，逐股 27/45 支持开盘）。
  **卖出侧：GO 窗口反而是最优**（−3.2bp，逐股 29/45 支持），且 GO 卖出后
  再等到收盘平均 −2.0bp / 中位 −12.4bp（t=−1.7 不显著）——等不出钱来，
  说明 GO 给出的离场时点在本池上是对的，没有卖早。

  **两侧最贵的都是限价单**，卖侧尤甚（−72.9bp）。逆向选择方向相反但都是亏：
  买侧成交在没涨的日子、被迫追高在涨的日子；卖侧成交在冲高的日子、
  被迫砸盘在跌的日子。看着漂亮的都是成交样本，账单在未成交的那 34% 里。

  GO 标签本身**有信息量但只是同期的**：买侧有 GO 的日子开盘→收盘 +166bp、
  无 GO 的日子 −177bp，区分很干净；可它要等当天走完才知道，用于**归因**成立，
  用于**决策**不成立（`lib.execution.split_by_go` 就是干这个归因用的）。
  唯一可照着做决策的是 `lib.execution.wait_value()`：它比较的两个动作
  （GO 一出现就做掉 / 放着等收盘）在同一时刻都摆在面前，不需要预知有没有 GO。

  ⚠️ 这些数字只属于 JCY 池（中小市值动量股）。同一套测算在油气蓝筹
  （601857/600938）上跑出来的日内形状**不一样**，卖出侧连排序都反了：
  那边最优是尾盘（+8.6bp）而非 GO 窗口（+5.3bp），且 GO 卖完再等到收盘还能
  多拿 6.2bp（t=2.1，卖早了）。方向可外推，数值与排序都不可外推，换池必须重测。

用法：
    python -m backtest.scripts.backtest_jcy_intraday
    python -m backtest.scripts.backtest_jcy_intraday --lookback 15   # 只看最近 15 天的信号
    python -m backtest.scripts.backtest_jcy_intraday --period 60     # 用 60min K 线
    python -m backtest.scripts.backtest_jcy_intraday --code 600519   # 只分析指定股票
    python -m backtest.scripts.backtest_jcy_intraday --exec_day same # 信号当日分析（盘中发现信号时）
"""

import argparse
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import date as _date, timedelta

import pandas as pd

warnings.filterwarnings("ignore")

from backtest.strategies import LuMACDBullStrategy
from backtest.config import index_history_start
from backtest.lib.cli import base_parser
from backtest.lib.manifest import write_run_manifest
from backtest.lib.price_store import load_daily
from backtest.strategies.bull_backtest import BullStrategyAdapter
from backtest.lib.console import use_utf8
# 仓位管理 / 分时取数 / MACD·GO·计价已拆进 lib，绘图已拆进 reports，本脚本只做编排
from backtest.lib.position_tracker import PositionTracker
from backtest.lib.execution import (TimingSummary, classify_timing,
                                    executable_price, intraday_macd,
                                    summarize_timing)
from backtest.lib.intraday_store import fetch_intraday_indexed
from backtest.reports.intraday_report import plot_intraday_chart
from jcy.lib.common import JSON_PATH, load_candidates


# ── 配置常量 ─────────────────────────────────────────────────────────────────

WARMUP_DAYS = 600               # 日线 MACD 预热所需自然日
INTRADAY_WARMUP_DAYS = 55       # 分时 MACD 预热所需自然日（需覆盖 slow=26 周期）

TIMING_LABEL = {
    "GO":    "✅ GO   ",
    "WAIT":  "⏳ WAIT ",
    "AVOID": "🚫 AVOID",
}


# ── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class SignalTimingResult:
    """单个信号的完整择时分析结果。"""
    code: str
    name: str
    action: str
    signal_date: pd.Timestamp
    exec_date: pd.Timestamp
    has_go: bool
    first_go: pd.Timestamp | None
    go_count: int


# ── 打印择时表格 ──────────────────────────────────────────────────────────────

def _print_timing_advice(summary: TimingSummary, action_cn: str):
    """打印操作建议（与表格解耦，便于独立维护）。"""
    print(f"\n    📌 操作建议：")
    if summary.has_go:
        if summary.first_go:
            print(f"      首选：{summary.first_go.strftime('%H:%M')}"
                  f"  ← {action_cn}，30min MACD 与日线方向共振")
        if summary.second_go:
            print(f"      次选：{summary.second_go.strftime('%H:%M')}"
                  f"  ← 动能加速确认后{action_cn}")
    elif action_cn == "买入":
        print(f"      ⚠️  全天无明确 GO 窗口（分时条件未共振）")
        # 旧建议是「挂限价单，不追高；或等次日再观察」，实测是所有做法里最贵的一种：
        # 挂开盘−0.5% 的限价单在 JCY 抽样池上比 VWAP 贵 46bp（t=63），
        # 因为没成交的那 34% 恰是股票当天一路走高的日子，最后被迫在更高处补。
        print(f"         建议：次日开盘集合竞价直接买（实测最优），不要挂低价等回调")
    else:
        print(f"      ⚠️  全天无明确缩量 / 死叉信号")
        # 旧建议是「次日集合竞价挂单卖出」，实测是所有固定时点里最差的一个：
        # 开盘价在 JCY 池比 VWAP 低 16.7bp、在油气池低 12.3bp——买入侧正因为
        # 这个"便宜"而首选开盘，卖出侧同一个便宜就是净亏。两池都是尾盘更好。
        print(f"         建议：尾盘 15:00 前卖出，不要留到次日集合竞价")


def print_timing_table(exec_bars: pd.DataFrame, summary: TimingSummary,
                       action_cn: str, code: str, name: str,
                       signal_date: pd.Timestamp, exec_date: pd.Timestamp):
    """打印分时择时明细表 + 操作建议。"""
    print(f"\n    ── {code} {name}  {action_cn}信号 "
          f"{signal_date.strftime('%Y-%m-%d')}  →  执行日 "
          f"{exec_date.strftime('%Y-%m-%d')} ──")
    print(f"    {'时间':6s}  {'收盘':>8s}  {'DIF':>8s}  {'DEA':>8s}  "
          f"{'MACD柱':>9s}  {'拉长':4s}  {'建议'}")
    print(f"    {'─' * 68}")

    for dt, row in exec_bars.iterrows():
        expanding = "✓" if row.get("hist_expanding") else " "
        label     = TIMING_LABEL.get(row.get("timing", ""), "")
        print(f"    {dt.strftime('%H:%M'):6s}  "
              f"{row['close']:>8.2f}  "
              f"{row.get('DIF',  float('nan')):>8.4f}  "
              f"{row.get('DEA',  float('nan')):>8.4f}  "
              f"{row.get('MACD', float('nan')):>+9.4f}  "
              f"{expanding:4s}  {label}")

    _print_timing_advice(summary, action_cn)


# ── 单股分析（拆分后的子函数） ───────────────────────────────────────────────

def _fetch_daily_signals(
    code: str, name: str, trade_start: str,
    index_symbol: str, lookback_days: int,
    warmup_days: int = WARMUP_DAYS, offline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    获取日线+大盘数据，运行 LuMACDBull 策略，返回 lookback 窗口内的信号。
    返回 (df_sig, signal_days) 或 None（失败时打印原因）。
    """
    today_str  = _date.today().strftime("%Y%m%d")
    trade_dt   = _date(int(trade_start[:4]),
                       int(trade_start[4:6]),
                       int(trade_start[6:]))
    data_start = (trade_dt - timedelta(days=warmup_days)).strftime("%Y%m%d")

    # 统一走本地仓库（评审项 1）：默认增量补齐；--offline 纯读缓存
    daily_df = load_daily(code, data_start, today_str,
                          auto_update=not offline, verbose=False)
    if daily_df.empty or len(daily_df) < 50:
        print(f"    日线数据不足，跳过")
        return None

    # 指数起点用绝对日期，不跟着个股的 data_start 走：否则每只票看到的月线
    # MACD 预热长度都不一样，同一天的 bull_market 可能因股而异（见 config.py）
    index_df = load_daily(index_symbol, index_history_start(), today_str,
                          adjust="none", kind="index",
                          auto_update=not offline, verbose=False)
    if index_df.empty:
        print(f"    大盘数据为空，跳过")
        return None

    inner   = LuMACDBullStrategy(shrink_exit=True)
    adapter = BullStrategyAdapter(inner, index_df, trade_start_date=trade_start)
    df_sig  = adapter.prepare(daily_df)

    cutoff      = pd.Timestamp.today() - timedelta(days=lookback_days)
    signal_days = df_sig[(df_sig.index >= cutoff) & (df_sig["signal"] != 0)]

    if signal_days.empty:
        print(f"    最近 {lookback_days} 天无买/卖信号")
        return None

    return df_sig, signal_days


def _determine_exec_date(df_sig: pd.DataFrame,
                         sig_date: pd.Timestamp,
                         mode: str) -> pd.Timestamp:
    """根据 exec_day_mode 确定执行日（same=信号当日，next=信号次日）。"""
    if mode == "same":
        return sig_date
    future = df_sig.index[df_sig.index > sig_date]
    return future[0] if len(future) > 0 else sig_date


def _analyze_single_signal(
    code: str, exec_date: pd.Timestamp, action: str, period: int,
) -> tuple[pd.DataFrame, pd.DataFrame, TimingSummary, float | None] | None:
    """
    获取分时数据 → 计算 MACD → 择时分类。
    返回 (intra_df, exec_bars, summary, exec_price) 或 None（数据缺失时打印原因）。

    exec_price（买入卖出同一套口径，恒为 float，不再有 None）：
      有 GO → 首个 GO 柱的**下一根 K 线开盘价**
      无 GO → 当日最后一柱收盘价（= 收盘集合竞价）

    为什么是"下一根的开盘价"而不是"GO 柱的收盘价"：
      要等这根 K 线走完才能判定它是不是 GO，那一刻它的收盘价已经成为历史，
      挂不进去。你看到 GO 之后能下的第一个单，成交在下一根的开盘。
      GO 出现在当日最后一根时，后面没有 K 线了，只能按收盘价成交。

    为什么无 GO 的兜底是"收盘价"而不是"开盘价"：
      "今天没出 GO" 这件事本身要等收盘才知道。若在无 GO 分支里用当日开盘价，
      等于用当天的结果去挑当天的入场点，是前视。既然规则是"等 GO"，
      等不到就只能在最后一刻买。
      （实测**不等 GO、直接在执行日开盘买**才是最优解——那是一条不依赖任何
        分时信息的无条件规则，因此完全因果。见模块 docstring 的「实测」一节。）
    """
    intra_start = (exec_date - timedelta(days=INTRADAY_WARMUP_DAYS)).strftime("%Y%m%d")
    intra_end   = exec_date.strftime("%Y%m%d")
    intra_df    = fetch_intraday_indexed(code, intra_start, intra_end, period)

    if intra_df.empty:
        print(f"    分时数据为空，跳过")
        return None

    # lib/execution.intraday_macd 在连续跨日序列上算 MACD 与 go_buy/go_sell
    # （唯一真值源）。索引形态 → 扁平形态 → 再还原。
    flat = intraday_macd(intra_df.reset_index())
    intra_df = flat.set_index("dt")

    exec_mask = intra_df.index.normalize() == exec_date
    exec_bars = intra_df[exec_mask].copy()

    if exec_bars.empty:
        print(f"    执行日 {exec_date.strftime('%Y-%m-%d')} 无分时数据"
              f"（可能是非交易日或数据缺失）")
        return None

    exec_bars = classify_timing(exec_bars, action)
    summary   = summarize_timing(exec_bars)

    exec_price = executable_price(exec_bars, summary)
    return intra_df, exec_bars, summary, exec_price


def _save_signal_chart(intra_df: pd.DataFrame, exec_date: pd.Timestamp,
                       code: str, name: str, action: str,
                       sig_date: pd.Timestamp, summary: TimingSummary,
                       period: int, save_dir: str):
    """生成安全文件名并调用绘图。"""
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", name)
    fname = (f"intraday_{code}_{safe_name}_{action}"
             f"_{sig_date.strftime('%Y%m%d')}"
             f"_{exec_date.strftime('%Y%m%d')}.png")
    plot_intraday_chart(
        intraday_df=intra_df,
        exec_date=exec_date,
        symbol=code,
        name=name,
        action=action,
        signal_date=sig_date,
        summary=summary,
        period=period,
        save_path=os.path.join(save_dir, fname),
    )


# ── 单股分析（编排函数） ─────────────────────────────────────────────────────

def _print_trade_log(tracker: PositionTracker, code: str, name: str):
    """打印单只股票的交易记录。"""
    if not tracker.trades:
        return
    print(f"\n    ── {code} {name} 交易记录（三级递进卖出） ──")
    print(f"    {'日期':12s}  {'操作':4s}  {'原因':12s}  "
          f"{'价格':>8s}  {'数量':>6s}  {'金额':>12s}  "
          f"{'剩余仓位':>8s}  {'本笔盈亏':>10s}")
    print(f"    {'─' * 88}")
    for t in tracker.trades:
        print(f"    {t.date.strftime('%Y-%m-%d'):12s}  "
              f"{t.action:4s}  {t.reason:12s}  "
              f"{t.price:>8.2f}  {t.shares:>6d}  "
              f"{t.amount:>+12.2f}  "
              f"{t.position_pct:>7.0f}%  "
              f"{t.realized_pnl:>+10.2f}")
    print(f"    {'─' * 88}")
    print(f"    累计盈亏：{tracker.total_pnl:>+.2f}  "
          f"收益率：{tracker.total_return_pct:>+.2f}%  "
          f"期末资金：{tracker.final_capital:>.2f}")


def _tracker_to_rows(code: str, name: str, tracker: PositionTracker) -> list[dict]:
    """将单只股票的 PositionTracker 转为 CSV 行列表。"""
    rows = []
    cum_pnl = 0.0
    for t in tracker.trades:
        cum_pnl += t.realized_pnl
        rows.append({
            "代码":        code,
            "名称":        name,
            "日期":        t.date.strftime("%Y-%m-%d"),
            "操作":        t.action,
            "原因":        t.reason,
            "价格":        round(t.price, 3),
            "数量(股)":    t.shares,
            "金额":        round(t.amount, 2),
            "仓位%":       round(t.position_pct, 1),
            "本笔盈亏":    round(t.realized_pnl, 2),
            "累计盈亏":    round(cum_pnl, 2),
            "累计收益率%": round(cum_pnl / tracker.initial_capital * 100, 2),
        })
    return rows


def save_stock_trades_csv(
    code: str, name: str, tracker: PositionTracker, save_dir: str
) -> str:
    """单只股票交易完成后立即保存其 CSV，返回保存路径（无交易则返回空字符串）。"""
    rows = _tracker_to_rows(code, name, tracker)
    if not rows:
        return ""
    safe_name  = re.sub(r'[\\/:*?"<>|]', "_", name)
    stock_path = os.path.join(save_dir, f"trades_{code}_{safe_name}.csv")
    pd.DataFrame(rows).to_csv(stock_path, index=False, encoding="utf-8-sig")
    print(f"    [{code}] {name} 交易记录已保存 → {stock_path}")
    return stock_path


def export_trades_csv(
    all_trackers: list[tuple[str, str, PositionTracker]],
    save_dir: str,
) -> str:
    """汇总所有股票交易记录，保存 trades_summary_{today}.csv。"""
    from datetime import date as _d
    today = _d.today().strftime("%Y%m%d")

    all_rows = []
    for code, name, tracker in all_trackers:
        rows = _tracker_to_rows(code, name, tracker)
        all_rows.extend(rows)

    if not all_rows:
        print("  ⚠ 无交易记录，跳过汇总 CSV 导出")
        return ""

    summary_path = os.path.join(save_dir, f"trades_summary_{today}.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n  汇总交易记录已导出至：{summary_path}")
    return summary_path


def analyze_candidate(
    candidate: dict,
    lookback_days: int,
    index_symbol: str,
    period: int,
    exec_day_mode: str,
    save_dir: str,
    offline: bool = False,
    capital: float = 100_000.0,
) -> tuple[list[SignalTimingResult], PositionTracker | None]:
    """
    对单只股票：
      1. 获取日线数据，运行 LuMACDBull 策略
      2. 运行三级递进仓位管理器，计算收益
      3. 找出 lookback_days 内的买 / 卖信号日
      4. 逐个信号日：获取分时数据 → MACD → 择时分析 → 绘图
    """
    code = candidate["code"]
    name = candidate["name"]
    print(f"\n  [{code}] {name}")

    try:
        fetched = _fetch_daily_signals(
            code, name, candidate["date"], index_symbol, lookback_days,
            offline=offline)
        if fetched is None:
            return [], None
        df_sig, signal_days = fetched

        # ── 第一阶段：分时择时分析，收集执行价格 ──────────────────────────────
        # 必须先于 tracker.run()，只有分时有 GO 的信号才计入实际交易
        results: list[SignalTimingResult] = []
        intraday_map: dict[pd.Timestamp, dict] = {}   # sig_date → 执行信息

        for sig_date, sig_row in signal_days.iterrows():
            action    = "buy" if sig_row["signal"] == 1 else "sell"
            action_cn = "买入" if action == "buy" else "卖出"
            exec_date = _determine_exec_date(df_sig, sig_date, exec_day_mode)

            print(f"    {action_cn}信号：{sig_date.strftime('%Y-%m-%d')}  "
                  f"执行日：{exec_date.strftime('%Y-%m-%d')}")

            sig_dif = float(df_sig.loc[sig_date, "DIF"]) if sig_date in df_sig.index else 0.0  # type: ignore[arg-type]
            analysis = _analyze_single_signal(code, exec_date, action, period)

            if analysis is None:
                # 无分时数据：买卖都兜底用日线收盘价。取不到分时数据是数据问题，
                # 不该让它决定这一笔到底建不建仓。
                daily_close = (float(df_sig.loc[exec_date, "close"])  # type: ignore[arg-type]
                               if exec_date in df_sig.index else None)
                intraday_map[sig_date] = {
                    "exec_date":  exec_date,
                    "exec_price": daily_close,
                    "action":     action,
                    "dif":        sig_dif,
                }
                continue

            intra_df, exec_bars, summary, exec_price = analysis
            intraday_map[sig_date] = {
                "exec_date":  exec_date,
                "exec_price": exec_price,
                "action":     action,
                "dif":        sig_dif,
            }

            print_timing_table(exec_bars, summary, action_cn, code, name,
                               sig_date, exec_date)
            _save_signal_chart(intra_df, exec_date, code, name, action,
                               sig_date, summary, period, save_dir)

            results.append(SignalTimingResult(
                code=code, name=name, action=action_cn,
                signal_date=sig_date, exec_date=exec_date,
                has_go=summary.has_go, first_go=summary.first_go,
                go_count=summary.go_count,
            ))

        # ── 第二阶段：仓位管理，使用分时确认后的价格计算收益 ─────────────────
        # 本金走 --capital：此前这里硬编码 100_000，而 base_parser 又把
        # --capital 挂在了命令行上——传了不报错也不生效，是最坏的一种默默无效
        tracker = PositionTracker(capital=capital)
        tracker.run(df_sig, intraday_map)
        _print_trade_log(tracker, code, name)
        if tracker.trades:
            save_stock_trades_csv(code, name, tracker, save_dir)

        return results, tracker

    except Exception as e:
        print(f"    ❌ 分析失败：{e}")
        import traceback
        traceback.print_exc()
        return [], None


# ── 主入口 ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="JCY 日线信号 + 分时择时（多周期共振）",
        # 不继承 --start：起点由各票推荐日与 --lookback 决定
        parents=[base_parser(start=False)],
    )
    parser.set_defaults(output="output/intraday", capital=100_000.0)
    parser.add_argument("--lookback",  type=int, default=None,
                        help="向前查找信号的天数；不指定时，按各股首次增持日起算")
    parser.add_argument("--period",    type=int, default=5,
                        choices=[5, 15, 30, 60],
                        help="分时 K 线周期（分钟），默认 30")
    parser.add_argument("--index",     type=str, default="000300",
                        help="大盘指数代码，默认 000300（沪深300）")
    parser.add_argument("--exec_day",  type=str, default="next",
                        choices=["next", "same"],
                        help="执行日：next=信号次日（默认），same=信号当日")
    parser.add_argument("--code",      type=str, default="300274",
                        help="只分析指定股票代码，留空则分析全部")
    return parser.parse_args()


def main():
    use_utf8()
    args = parse_args()
    t0 = time.time()

    print("\n" + "─" * 65)
    print("  JCY 日线信号 + 分时择时（多周期共振）")
    print("─" * 65)
    print(f"  信号查找窗口  ：{'最近 ' + str(args.lookback) + ' 天' if args.lookback else '首次增持日起（各股独立）'}")
    print(f"  分时 K 线周期 ：{args.period} 分钟")
    print(f"  执行日模式    ：{args.exec_day}"
          f"（{'信号次日' if args.exec_day == 'next' else '信号当日'}）")
    print(f"  大盘指数      ：{args.index}")
    print(f"  输出目录      ：{args.output}/")
    print("─" * 65)

    if not os.path.exists(JSON_PATH):
        print(f"  ❌ 找不到 JSON 文件：{JSON_PATH}")
        sys.exit(1)

    candidates = load_candidates(JSON_PATH)
    if not candidates:
        print("  ❌ 未找到增持 A 股，请检查 JSON 数据")
        sys.exit(1)

    if args.code:
        candidates = [c for c in candidates if c["code"] == args.code]
        if not candidates:
            print(f"  ❌ 未找到代码 {args.code}，请检查 JSON 数据")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"  共 {len(candidates)} 只候选股票\n")

    all_results: list[SignalTimingResult] = []
    all_trackers: list[tuple[str, str, PositionTracker]] = []  # (code, name, tracker)

    for candidate in candidates:
        if args.lookback is not None:
            lookback_days = args.lookback
        else:
            first_rating_date = _date.fromisoformat(
                f"{candidate['date'][:4]}-{candidate['date'][4:6]}-{candidate['date'][6:]}"
            )
            lookback_days = (_date.today() - first_rating_date).days + 1
        results, tracker = analyze_candidate(
            candidate=candidate,
            lookback_days=lookback_days,
            index_symbol=args.index,
            period=args.period,
            exec_day_mode=args.exec_day,
            save_dir=args.output,
            offline=args.offline,
            capital=args.capital,
        )
        all_results.extend(results)
        if tracker and tracker.trades:
            all_trackers.append((candidate["code"], candidate["name"], tracker))

    # ── 分时择时汇总 ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  分时择时汇总")
    print("═" * 65)

    if all_results:
        print(f"  {'代码':8s}  {'名称':8s}  {'操作':4s}  "
              f"{'信号日':12s}  {'执行日':12s}  {'首选时间':8s}  {'状态'}")
        print(f"  {'─' * 65}")
        for r in all_results:
            first_go_str = (r.first_go.strftime("%H:%M")
                            if r.first_go else "  —  ")
            status = (f"✅ {r.go_count} 个 GO 窗口"
                      if r.has_go else "⚠️  无 GO 窗口，建议等待")
            print(f"  {r.code:8s}  {r.name:8s}  {r.action:4s}  "
                  f"{r.signal_date.strftime('%Y-%m-%d'):12s}  "
                  f"{r.exec_date.strftime('%Y-%m-%d'):12s}  "
                  f"{first_go_str:8s}  {status}")
    else:
        if args.lookback:
            print(f"  最近 {args.lookback} 天内无买/卖信号")
            print(f"  → 可通过 --lookback 扩大查找窗口，例如 --lookback 60")
        else:
            print(f"  首次增持日起至今无买/卖信号")

    # ── 仓位管理收益汇总 ─────────────────────────────────────────────────────
    if all_trackers:
        print("\n" + "═" * 65)
        print("  三级递进卖出 — 收益汇总")
        print("═" * 65)
        print(f"  {'代码':8s}  {'名称':8s}  {'交易数':>6s}  "
              f"{'累计盈亏':>12s}  {'收益率':>8s}  {'期末资金':>12s}")
        print(f"  {'─' * 65}")

        total_pnl = 0.0
        total_capital = 0.0
        total_initial = 0.0
        winners = 0

        for code, name, tracker in all_trackers:
            pnl     = tracker.total_pnl
            ret_pct = tracker.total_return_pct
            final   = tracker.final_capital
            n_trades = len(tracker.trades)
            total_pnl     += pnl
            total_capital  += final
            total_initial  += tracker.initial_capital
            if pnl > 0:
                winners += 1

            flag = "✅" if pnl > 0 else ("⚠️ " if pnl == 0 else "❌")
            print(f"  {code:8s}  {name:8s}  {n_trades:>6d}  "
                  f"{pnl:>+12.2f}  {ret_pct:>+7.2f}%  {final:>12.2f}  {flag}")

        n_stocks = len(all_trackers)
        avg_ret  = total_pnl / total_initial * 100 if total_initial > 0 else 0
        win_rate = winners / n_stocks * 100 if n_stocks > 0 else 0

        print(f"  {'─' * 65}")
        print(f"  合计  {n_stocks} 只股票  |  "
              f"总盈亏 {total_pnl:>+.2f}  |  "
              f"平均收益率 {avg_ret:>+.2f}%  |  "
              f"胜率 {win_rate:.0f}%（{winners}/{n_stocks}）")

    # ── 交易记录导出 CSV ─────────────────────────────────────────────────────
    if all_trackers:
        export_trades_csv(all_trackers, args.output)

    # 可复现清单（评审项 2）
    write_run_manifest(args.output,
                       symbols=[c["code"] for c in candidates]
                               + [(args.index, "none")],
                       started_at=t0)

    print("\n" + "─" * 65)
    print(f"  完成。结果已保存至：{os.path.abspath(args.output)}/")
    print("─" * 65)


if __name__ == "__main__":
    main()
