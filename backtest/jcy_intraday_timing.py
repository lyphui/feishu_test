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
       fetch_intraday(code, exec_date − INTRADAY_WARMUP_DAYS, exec_date, period)
       add_macd() 在**连续跨日**序列上算 30min MACD(12/26/9)
       切出执行日 8 根 → classify_timing(action="buy")
         GO    = hist_expanding 且 DIF > DEA
         WAIT  = DIF > DEA 但红柱未拉长
         AVOID = DIF <= DEA
       exec_price（`_executable_price()`，买卖同口径）：
         有 GO → 首个 GO 柱的**下一根 K 线开盘价**（GO 在最后一根时退化为收盘价）
         无 GO → 当日最后一柱收盘价
  ④ 落仓位      `PositionTracker.run(df_sig, intraday_map)`
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
复现：`python backtest/exec_bench.py --universe jcy --side both --limit 45`
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
    python backtest/jcy_intraday_timing.py
    python backtest/jcy_intraday_timing.py --lookback 15   # 只看最近 15 天的信号
    python backtest/jcy_intraday_timing.py --period 60     # 用 60min K 线
    python backtest/jcy_intraday_timing.py --code 600519   # 只分析指定股票
    python backtest/jcy_intraday_timing.py --exec_day same # 信号当日分析（盘中发现信号时）
"""

import argparse
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

from strategies import LuMACDBullStrategy
from config import index_history_start
from lib.plotting import (
    C_BG, C_FG, C_GREEN, C_RED, C_BLUE, C_GOLD, C_MUTED,
    setup_matplotlib, style_ax,
)
from lib.market_data import fetch_stock_data, fetch_index_data
from lib.bull_backtest import BullStrategyAdapter
from jcy.lib.common import JSON_PATH, load_candidates

setup_matplotlib()


# ── 配置常量 ─────────────────────────────────────────────────────────────────

WARMUP_DAYS = 600               # 日线 MACD 预热所需自然日
INTRADAY_WARMUP_DAYS = 55       # 分时 MACD 预热所需自然日（需覆盖 slow=26 周期）
CHART_CONTEXT_DAYS = 28         # 分时图表显示执行日前多少天的上下文
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

TIMING_LABEL = {
    "GO":    "✅ GO   ",
    "WAIT":  "⏳ WAIT ",
    "AVOID": "🚫 AVOID",
}


# ── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class TimingSummary:
    """单个执行日的分时择时汇总。"""
    has_go: bool
    go_times: list = field(default_factory=list)
    first_go: pd.Timestamp | None = None
    second_go: pd.Timestamp | None = None
    go_count: int = 0
    total_bars: int = 0


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


# ── 仓位管理 ────────────────────────────────────────────────────────────────

def _next_trading_day(index: pd.Index, date) -> pd.Timestamp | None:
    """下一个交易日；`date` 已是最后一根 K 线时返回 None。"""
    future = index[index > date]
    return future[0] if len(future) > 0 else None


@dataclass
class TradeRecord:
    """单笔交易记录。"""
    date: pd.Timestamp
    action: str          # 买入 / 减仓 / 清仓
    reason: str          # 金叉+红柱拉长 / 红柱缩短 / 死叉 / DIF<0 / 期末清算
    price: float
    shares: int
    amount: float        # 正=收入，负=支出（含手续费）
    position_pct: float  # 操作后剩余仓位占比（%）
    realized_pnl: float  # 本笔已实现盈亏


class PositionTracker:
    """
    分级买入 + 三级递进卖出仓位管理器。

    买入 ── 分两阶段进场（根据 DIF 位置调整初仓比例）：
      初仓：DIF < 0（零轴下方弱信号）→ 买入 1/3 可用资金
            DIF ≥ 0（零轴上方标准信号）→ 买入 1/2 可用资金
      加仓：初仓次日起，若红柱持续拉长 → 买入剩余资金补满仓
      退出：初仓阶段遇红柱缩短/死叉/DIF<0 → 直接全退，不等满仓

    卖出 ── 三级递进（满仓后）：
      Level 1: 红柱缩短      → 卖出 1/3 满仓股数
      Level 2: 死叉(DIF<DEA) → 卖出 1/3 满仓股数
      Level 3: DIF < 0       → 清仓剩余

    费用：佣金 0.03% 双边 + 印花税 0.1% 卖方单边（A 股标准）
    """

    COMMISSION_RATE = 0.0003   # 佣金（买卖各收）
    STAMP_TAX_RATE  = 0.001    # 印花税（仅卖方）

    def __init__(self, capital: float = 100_000):
        self.initial_capital = capital
        self.cash = capital
        self.shares = 0
        self._buy_shares = 0       # 满仓时的股数（用于计算 1/3）
        self._avg_cost = 0.0
        self._sell_level = 0       # 0=满仓/空仓, 1=已卖1/3, 2=已卖2/3
        self._buy_level = 0        # 0=空仓, 1=初仓(半仓), 2=满仓
        self._buy_date = None      # 初仓日期（防止当日重复操作）
        self.trades: list[TradeRecord] = []

    # ── 核心入口 ─────────────────────────────────────────────────────────────

    def run(self, df_sig: pd.DataFrame,
            intraday_map: "dict | None" = None) -> list[TradeRecord]:
        """
        遍历日线数据，按分级买入 + 三级递进卖出规则模拟交易。

        **统一 T+1 执行。**日线的 signal / 红柱缩短 / 死叉 / DIF<0 都要等当日收盘
        才算得出来，所以每一条都排到**下一个交易日**成交，没有例外。信号落在
        最后一根 K 线时没有 T+1，该操作作废（仓位交给期末清算）。

        成交价按这个优先级取：
          ① `intraday_map` 给的分时价（`_executable_price()`，见其 docstring）
          ② 执行日的日线收盘价——注意是**执行日**不是信号日，两者差一天，
             拿信号日的价成交次日的单是把已知价错配到另一天

        intraday_map: {sig_date: {"exec_date", "exec_price", "action", "dif"}}
          - 只覆盖 lookback 窗口内的信号日，用来把成交价换成分时价
          - `exec_price` 为 None（分时数据缺失）→ 走 ②，仍然照常成交
          - `exec_date == sig_date`（`--exec_day same` 的盘中实时模式）→ 当日成交；
            这是使用者明确选的实盘口径，回测默认的 next 模式不会走到

        两条铁律：
          * **分时只决定「几点做」，不决定「做不做」。**早先在"买入且分时无 GO"
            时直接跳过建仓，等于让执行层条件充当隐式策略过滤器——无 GO 占四成日子。
          * **执行时点不许依赖 `--lookback`。**早先窗口内的信号 T+1 成交、窗口外的
            当日收盘成交，等于同一次回测前后两段用两套规则，改个打印参数就能改变
            历史收益。现在两段同为 T+1，`intraday_map` 只换价、不换日。
        """
        intraday_map = intraday_map or {}

        # exec_date → op；每个遍历日最多排一条，故 key 不会冲突
        _pending: dict[pd.Timestamp, dict] = {}

        for date, row in df_sig.iterrows():
            dif = row.get("DIF", 0)
            dea = row.get("DEA", 0)

            # ── 先执行昨日排定的操作，再用更新后的状态判断今天 ────────────────
            if date in _pending:
                self._execute(_pending.pop(date), date, df_sig)  # type: ignore[call-overload]

            nxt = _next_trading_day(df_sig.index, date)

            # ── 空仓：等买入信号 ─────────────────────────────────────────────
            if self.shares == 0:
                if row.get("signal") == 1:
                    info = intraday_map.get(date)
                    self._place(_pending, df_sig, date, nxt, info, {
                        "action": "buy_initial",
                        "dif": info["dif"] if info else dif,
                        "reason": "金叉+红柱拉长",
                    })

            # ── 初仓阶段：等候次日确认加仓，或提前退出 ──────────────────────
            elif self._buy_level == 1:
                if date == self._buy_date:
                    pass  # 初仓当日不重复操作
                elif dif < 0:
                    self._place(_pending, df_sig, date, nxt, None,
                                {"action": "sell_all", "reason": "DIF<0"})
                elif row.get("hist_expanding", False):
                    self._place(_pending, df_sig, date, nxt, None,
                                {"action": "buy_add", "reason": "红柱续拉长"})
                elif row.get("hist_shrinking", False) or dif < dea:
                    self._place(_pending, df_sig, date, nxt, None,
                                {"action": "sell_all", "reason": "初仓退出"})

            # ── 满仓阶段：三级递进卖出 ───────────────────────────────────────
            else:
                if dif < 0:
                    self._place(_pending, df_sig, date, nxt, None,
                                {"action": "sell_all", "reason": "DIF<0"})
                elif self._sell_level == 0 and row.get("hist_shrinking", False):
                    self._place(_pending, df_sig, date, nxt, intraday_map.get(date),
                                {"action": "sell_1", "reason": "红柱缩短"})
                elif self._sell_level == 1 and dif < dea:
                    self._place(_pending, df_sig, date, nxt, None,
                                {"action": "sell_2", "reason": "死叉"})
                elif self._sell_level == 2 and dif < 0:
                    self._place(_pending, df_sig, date, nxt, None,
                                {"action": "sell_all", "reason": "DIF<0"})

        # 期末仍有持仓 → 按最新价清算
        if self.shares > 0:
            last_date  = df_sig.index[-1]
            last_price = df_sig["close"].iloc[-1]
            self._sell_remaining(last_date, last_price, "期末清算")

        return self.trades

    # ── 排单与成交 ───────────────────────────────────────────────────────────

    def _place(self, pending: dict, df_sig: pd.DataFrame, cur_date,
               nxt, info: dict | None, op: dict) -> None:
        """
        把一个操作排到执行日。**执行日恒为下一个交易日**，`info` 只用来换成交价。

        唯一的例外是 `--exec_day same`：`info["exec_date"] == cur_date`，
        使用者明确要求按盘中实时口径当日成交。
        """
        exec_date = info["exec_date"] if info else nxt
        if exec_date is None:
            return                      # 信号在最后一根 K 线，没有 T+1 可成交
        op = {**op, "price": info.get("exec_price") if info else None}
        if exec_date == cur_date:
            self._execute(op, cur_date, df_sig)
        else:
            pending[exec_date] = op

    def _execute(self, op: dict, date, df_sig: pd.DataFrame) -> None:
        """按 op 成交。价缺失时兜底取**执行日**（= `date`）的日线收盘价。"""
        px = op.get("price")
        price = float(px) if px is not None else float(df_sig.loc[date, "close"])
        action = op["action"]

        if action == "buy_initial" and self.shares == 0:
            self._buy_initial(date, price, op["dif"])
        elif action == "buy_add" and self.shares > 0:
            self._buy_add(date, price)
        elif action == "sell_1" and self.shares > 0:
            self._sell_portion(date, price, 1, op["reason"])
        elif action == "sell_2" and self.shares > 0:
            self._sell_portion(date, price, 2, op["reason"])
        elif action == "sell_all" and self.shares > 0:
            self._sell_remaining(date, price, op["reason"])

    # ── 买入 ─────────────────────────────────────────────────────────────────

    def _buy_initial(self, date, price, dif: float):
        """
        初仓：根据 DIF 位置决定仓位比例。
          DIF < 0 → 1/3 可用资金（零轴下方，弱信号，保守）
          DIF ≥ 0 → 1/2 可用资金（零轴上方，标准信号）
        """
        fraction    = 1 / 3 if dif < 0 else 1 / 2
        target_cash = self.cash * fraction
        shares      = int(target_cash / price / 100) * 100
        if shares <= 0:
            return
        reason = f"金叉+红柱拉长（DIF{'<0，保守1/3' if dif < 0 else '≥0，标准1/2'}）"
        self._do_buy(date, price, shares, "初仓", reason)
        self._buy_level = 1
        self._buy_date  = date

    def _buy_add(self, date, price):
        """加仓：用剩余可用资金买入，补至满仓。"""
        shares = int(self.cash / price / 100) * 100
        if shares <= 0:
            self._buy_level = 2   # 资金不足，仍视为满仓
            return
        self._do_buy(date, price, shares, "加仓", "红柱持续拉长")
        self._buy_level  = 2
        self._buy_shares = self.shares   # 更新满仓基准

    def _do_buy(self, date, price, buy_shares: int, action: str, reason: str):
        cost       = buy_shares * price
        commission = cost * self.COMMISSION_RATE
        self.cash -= (cost + commission)
        prev_shares     = self.shares
        self.shares    += buy_shares
        # 加权均价
        if prev_shares > 0:
            self._avg_cost = (self._avg_cost * prev_shares + cost) / self.shares
        else:
            self._avg_cost = price
        self._buy_shares = self.shares
        self._sell_level = 0
        pos_pct = self._pos_pct(price)
        self.trades.append(TradeRecord(
            date=date, action=action, reason=reason,
            price=price, shares=buy_shares,
            amount=-(cost + commission),
            position_pct=pos_pct, realized_pnl=0.0,
        ))

    # ── 卖出（按级别） ───────────────────────────────────────────────────────

    def _sell_portion(self, date, price, level: int, reason: str):
        """卖出 1/3 满仓股数。level: 1 或 2。"""
        sell_shares = int(self._buy_shares / 3 / 100) * 100
        if sell_shares <= 0 or sell_shares > self.shares:
            sell_shares = self.shares
        self._do_sell(date, price, sell_shares, "减仓", reason)
        self._sell_level = level

    def _sell_remaining(self, date, price, reason: str):
        """清仓所有剩余持仓。"""
        self._do_sell(date, price, self.shares, "清仓", reason)
        self._sell_level = 0
        self._buy_level  = 0
        self._buy_shares = 0

    def _do_sell(self, date, price, sell_shares: int, action: str, reason: str):
        if sell_shares <= 0:
            return
        revenue    = sell_shares * price
        commission = revenue * self.COMMISSION_RATE
        stamp_tax  = revenue * self.STAMP_TAX_RATE
        net        = revenue - commission - stamp_tax
        pnl        = (price - self._avg_cost) * sell_shares - commission - stamp_tax
        self.cash   += net
        self.shares -= sell_shares
        self.trades.append(TradeRecord(
            date=date, action=action, reason=reason,
            price=price, shares=sell_shares,
            amount=net, position_pct=self._pos_pct(price), realized_pnl=pnl,
        ))

    # ── 辅助 ─────────────────────────────────────────────────────────────────

    def _pos_pct(self, price: float) -> float:
        """当前股票市值占总资产的百分比。"""
        total = self.cash + self.shares * price
        return (self.shares * price / total * 100) if total > 0 else 0.0

    # ── 汇总 ─────────────────────────────────────────────────────────────────

    @property
    def total_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.trades)

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / self.initial_capital * 100

    @property
    def final_capital(self) -> float:
        return self.cash


# ── 分时数据获取 ──────────────────────────────────────────────────────────────

def _fetch_intraday_akshare(symbol: str, start_date: str, end_date: str,
                            period: int, max_retries: int = 1,
                            retry_delay: float = 5) -> pd.DataFrame | None:
    """akshare 分时数据获取，含限流重试。成功返回 DataFrame，失败返回 None。"""
    try:
        import akshare as ak
    except ImportError:
        return None

    s = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 09:30:00"
    e = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]} 15:00:00"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                wait = retry_delay * attempt
                print(f"    akshare 等待 {wait:.0f} 秒后重试（第 {attempt}/{max_retries} 次）...")
                time.sleep(wait)
            else:
                time.sleep(1)

            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol, period=str(period),
                start_date=s, end_date=e, adjust="qfq",
            )
            df = df.rename(columns={
                "时间": "datetime", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            })
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols]
        except (ValueError, KeyError, OSError, RuntimeError) as err:
            last_err = err
            err_str = str(err)
            if any(kw in err_str for kw in ("RemoteDisconnected", "Connection",
                                             "reset", "Timeout", "timed out")):
                if attempt < max_retries:
                    continue
            break

    print(f"    ⚠ akshare 分时获取失败：{last_err}")
    return None


def _fetch_intraday_baostock(symbol: str, start_date: str, end_date: str,
                             period: int) -> pd.DataFrame | None:
    """baostock 分时数据获取。成功返回 DataFrame，失败返回 None。"""
    try:
        import baostock as bs
    except ImportError:
        return None

    prefix = "sh" if symbol.startswith("6") or symbol.startswith("9") else "sz"
    bs_code = f"{prefix}.{symbol}"
    start_dash = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_dash = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    print(f"    正在从 baostock 获取 {bs_code} 分时({period}min)数据...")
    try:
        lg = bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            "time,open,high,low,close,volume",
            start_date=start_dash, end_date=end_dash,
            frequency=str(period), adjustflag="2",
        )
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        bs.logout()

        if not rows:
            print(f"    baostock 分时返回空数据")
            return None

        df = pd.DataFrame(rows, columns=rs.fields)
        # baostock time 格式：20260305093000000 → datetime
        df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
        df = df.drop(columns=["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("datetime").sort_index()
        # 过滤掉全零行（baostock 偶尔返回空行）
        df = df[(df["close"] > 0) & (df["volume"] > 0)]
        if df.empty:
            return None
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"    ⚠ baostock 分时获取失败：{e}")
        try:
            bs.logout()
        except Exception:
            pass
        return None


def fetch_intraday(symbol: str, start_date: str, end_date: str,
                   period: int = 5) -> pd.DataFrame:
    """
    获取分时 K 线（前复权），akshare → baostock 双源。
    start_date / end_date: YYYYMMDD 格式
    period: 分钟数，5 / 15 / 30 / 60
    """
    # 优先 akshare
    result = _fetch_intraday_akshare(symbol, start_date, end_date, period)
    if result is not None and not result.empty:
        return result

    # baostock 备用
    print(f"    尝试 baostock 备用...")
    result = _fetch_intraday_baostock(symbol, start_date, end_date, period)
    if result is not None and not result.empty:
        return result

    return pd.DataFrame()


# ── 分时 MACD 计算 ────────────────────────────────────────────────────────────

def add_macd(df: pd.DataFrame,
             fast: int = MACD_FAST,
             slow: int = MACD_SLOW,
             sig: int = MACD_SIGNAL) -> pd.DataFrame:
    """
    在 DataFrame 上追加 DIF / DEA / MACD / hist_expanding / hist_shrinking / death_cross 列。

    三个布尔列都要在**连续跨日**序列上算，因为它们都用 `shift(1)` 看前一根：
    切成日切片再算，每天第一根的 shift 是 NaN，当天开盘那根就永远判不出
    「红柱开始缩短」或「隔夜死叉」——而隔夜跳空恰恰是这类翻转最集中的地方。
    `classify_timing()` 因此只读这些列，不自己重算。
    """
    df = df.copy()
    ema_f = df["close"].ewm(span=fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=slow, adjust=False).mean()
    dif   = ema_f - ema_s
    dea   = dif.ewm(span=sig, adjust=False).mean()
    hist  = (dif - dea) * 2

    df["DIF"]  = dif
    df["DEA"]  = dea
    df["MACD"] = hist
    df["hist_expanding"] = (hist > 0) & (hist > hist.shift(1))
    df["hist_shrinking"] = (hist > 0) & (hist < hist.shift(1))
    df["death_cross"] = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    return df


# ── 分时择时核心逻辑 ──────────────────────────────────────────────────────────

def classify_timing(day_slice: pd.DataFrame, action: str) -> pd.DataFrame:
    """
    对执行日的每根分时 K 线打时机标签：GO / WAIT / AVOID。

    action: 'buy' or 'sell'

    买入判断：
      GO    → DIF > DEA 且红柱正在拉长（与日线信号完全共振）
      WAIT  → DIF > DEA 但红柱未拉长（方向对，等动能起步）
      AVOID → DIF <= DEA（方向相反，开盘混沌期或回调中）

    卖出判断：
      GO    → 红柱开始缩短（动能衰减）或 DIF 下穿 DEA（死叉）
      WAIT  → 动能仍在高位（暂时持有）

    两侧**不对称**：买侧要求方向与动能同时成立，还留了 AVOID 一档；
    卖侧只要动能转弱就放行，死叉那一路根本不看红柱正负，也没有 AVOID。
    所以卖侧的 GO 天数占比天然高得多，两边的 bp 不可横向比较。

    所有依赖前一根的布尔列都由 `add_macd()` 在连续跨日序列上算好，这里只读不算——
    在日切片上重算会丢掉每天第一根的判定（见 `add_macd` docstring）。
    与 `lib.execution.intraday_macd()` 的 `go_buy`/`go_sell` 同口径。
    """
    df = day_slice.copy()

    if action == "buy":
        conditions = [
            df["hist_expanding"] & (df["DIF"] > df["DEA"]),
            df["DIF"] > df["DEA"],
        ]
        df["timing"] = np.select(conditions, ["GO", "WAIT"], default="AVOID")
    else:
        df["timing"] = np.select(
            [df["hist_shrinking"] | df["death_cross"]],
            ["GO"],
            default="WAIT",
        )

    return df


def summarize_timing(day_df: pd.DataFrame) -> TimingSummary:
    """从打好标签的执行日切片中汇总 GO 窗口信息。"""
    go_bars = day_df[day_df["timing"] == "GO"]
    return TimingSummary(
        has_go=len(go_bars) > 0,
        go_times=list(go_bars.index),
        first_go=go_bars.index[0] if len(go_bars) > 0 else None,
        second_go=go_bars.index[1] if len(go_bars) > 1 else None,
        go_count=len(go_bars),
        total_bars=len(day_df),
    )


# ── 绘图辅助 ─────────────────────────────────────────────────────────────────

def _highlight_exec_day(ax, day_bars: pd.DataFrame, color: str, period: int):
    """在指定子图上为执行日添加半透明背景高亮。"""
    if not day_bars.empty:
        ax.axvspan(
            day_bars.index[0] - timedelta(minutes=period // 2),
            day_bars.index[-1] + timedelta(minutes=period // 2),
            alpha=0.10, color=color, lw=0,
        )


def _plot_price_panel(ax, plot_df: pd.DataFrame, day_bars: pd.DataFrame,
                      exec_date: pd.Timestamp, signal_date: pd.Timestamp,
                      summary: TimingSummary, action: str,
                      color_dir: str, period: int):
    """子图1：分时收盘价 + GO 窗口标注 + 信号日/执行日高亮。"""
    ax.plot(plot_df.index, plot_df["close"], color=C_BLUE, lw=1.0)
    _highlight_exec_day(ax, day_bars, color_dir, period)

    # 信号日金色背景（若与执行日不同）
    if signal_date != exec_date:
        sig_bars = plot_df[plot_df.index.normalize() == signal_date]
        if not sig_bars.empty:
            ax.axvspan(
                sig_bars.index[0] - timedelta(minutes=period // 2),
                sig_bars.index[-1] + timedelta(minutes=period // 2),
                alpha=0.07, color=C_GOLD, lw=0,
            )
            ax.axvline(x=sig_bars.index[0], color=C_GOLD, lw=1.2,
                        linestyle="--", alpha=0.7,
                        label=f"日线信号日 {signal_date.strftime('%Y-%m-%d')}")

    # GO 窗口竖线标注
    price_max = plot_df["close"].max()
    price_min = plot_df["close"].min()
    label_y   = price_max + (price_max - price_min) * 0.012
    for t in summary.go_times:
        ax.axvline(x=t, color=color_dir, lw=1.5, alpha=0.8, linestyle="--")
        ax.text(t, label_y, t.strftime("%H:%M"),
                color=color_dir, fontsize=8, rotation=90,
                va="bottom", ha="center", weight="bold")

    # 标题
    action_cn = "买入" if action == "buy" else "卖出"
    title_status = (f"✅ {summary.go_count} 个 GO 窗口，首选 "
                    f"{summary.first_go.strftime('%H:%M')}"
                    if summary.has_go else "⚠️ 无明确 GO 窗口，建议观望")
    ax.set_title(
        f"分时择时（{period}min）  |  "
        f"{action_cn}信号 {signal_date.strftime('%Y-%m-%d')}  |  "
        f"执行日 {exec_date.strftime('%Y-%m-%d')}  |  {title_status}",
        color=C_FG, fontsize=11, pad=8,
    )
    ax.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    style_ax(ax)


def _plot_macd_panel(ax, plot_df: pd.DataFrame, day_bars: pd.DataFrame,
                     summary: TimingSummary, action: str,
                     color_dir: str, period: int):
    """子图2：分时 MACD 柱 + DIF/DEA + GO 标记。"""
    _highlight_exec_day(ax, day_bars, color_dir, period)

    bar_width = timedelta(minutes=period - 2)
    bar_colors = np.where(plot_df["MACD"].values >= 0, C_RED, C_GREEN)
    expanding  = plot_df["hist_expanding"].values

    ax.bar(plot_df.index[~expanding], plot_df["MACD"].values[~expanding],
           color=bar_colors[~expanding], alpha=0.4, width=bar_width)
    ax.bar(plot_df.index[expanding], plot_df["MACD"].values[expanding],
           color=bar_colors[expanding], alpha=0.9, width=bar_width)

    ax.plot(plot_df.index, plot_df["DIF"], color=C_BLUE, lw=1.0, label="DIF")
    ax.plot(plot_df.index, plot_df["DEA"], color=C_GOLD, lw=1.0, label="DEA")
    ax.axhline(0, color=C_MUTED, lw=0.5, linestyle="--")

    # GO 窗口的 DIF 点位标注
    marker = "^" if action == "buy" else "v"
    for t in summary.go_times:
        if t in plot_df.index:
            y = plot_df.loc[t, "DIF"]
            ax.scatter([t], [y], marker=marker, color=color_dir,
                       s=100, zorder=8, edgecolors="white", linewidths=0.5)

    ax.legend(facecolor=C_BG, labelcolor=C_FG, edgecolor=C_MUTED, fontsize=8)
    ax.set_ylabel(f"分时 MACD ({period}min)", color=C_FG, fontsize=9)
    style_ax(ax)


def _plot_volume_panel(ax, plot_df: pd.DataFrame, day_bars: pd.DataFrame,
                       exec_mask, color_dir: str, period: int):
    """子图3：成交量（执行日按方向上色，其余灰色）。"""
    bar_width = timedelta(minutes=period - 2)
    non_exec = plot_df.index[~exec_mask]
    ax.bar(non_exec, plot_df.loc[non_exec, "volume"].values,
           color=C_MUTED, alpha=0.5, width=bar_width)
    if not day_bars.empty:
        ax.bar(day_bars.index, day_bars["volume"].values,
               color=color_dir, alpha=0.75, width=bar_width)

    ax.set_ylabel("成交量", color=C_FG, fontsize=9)
    style_ax(ax)


def plot_intraday_chart(
    intraday_df: pd.DataFrame,
    exec_date: pd.Timestamp,
    symbol: str,
    name: str,
    action: str,
    signal_date: pd.Timestamp,
    summary: TimingSummary,
    period: int,
    save_path: str | None = None,
):
    """
    3 面板分时图：价格 + GO 窗口 | MACD | 成交量。
    只显示执行日前 CHART_CONTEXT_DAYS 天的数据。
    """
    context_start = exec_date - timedelta(days=CHART_CONTEXT_DAYS)
    plot_df   = intraday_df[intraday_df.index.normalize() >= context_start].copy()
    exec_mask = plot_df.index.normalize() == exec_date
    day_bars  = plot_df[exec_mask]
    color_dir = C_GREEN if action == "buy" else C_RED

    fig = plt.figure(figsize=(16, 10), facecolor=C_BG)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.06, height_ratios=[3, 2, 1])

    ax1 = fig.add_subplot(gs[0], facecolor=C_BG)
    _plot_price_panel(ax1, plot_df, day_bars, exec_date, signal_date,
                      summary, action, color_dir, period)

    ax2 = fig.add_subplot(gs[1], sharex=ax1, facecolor=C_BG)
    _plot_macd_panel(ax2, plot_df, day_bars, summary, action, color_dir, period)

    ax3 = fig.add_subplot(gs[2], sharex=ax1, facecolor=C_BG)
    _plot_volume_panel(ax3, plot_df, day_bars, exec_mask, color_dir, period)

    # X 轴格式
    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right",
             color=C_FG, fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"    图表已保存至：{save_path}")
    else:
        plt.show()
    plt.close(fig)


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
    warmup_days: int = WARMUP_DAYS,
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

    daily_df = fetch_stock_data(code, data_start, today_str)
    if daily_df.empty or len(daily_df) < 50:
        print(f"    日线数据不足，跳过")
        return None

    # 指数起点用绝对日期，不跟着个股的 data_start 走：否则每只票看到的月线
    # MACD 预热长度都不一样，同一天的 bull_market 可能因股而异（见 config.py）
    index_df = fetch_index_data(index_symbol, index_history_start(), today_str)
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
    intra_df    = fetch_intraday(code, intra_start, intra_end, period)

    if intra_df.empty:
        print(f"    分时数据为空，跳过")
        return None

    intra_df  = add_macd(intra_df)
    exec_mask = intra_df.index.normalize() == exec_date
    exec_bars = intra_df[exec_mask].copy()

    if exec_bars.empty:
        print(f"    执行日 {exec_date.strftime('%Y-%m-%d')} 无分时数据"
              f"（可能是非交易日或数据缺失）")
        return None

    exec_bars = classify_timing(exec_bars, action)
    summary   = summarize_timing(exec_bars)

    exec_price = _executable_price(exec_bars, summary)
    return intra_df, exec_bars, summary, exec_price


def _executable_price(exec_bars: pd.DataFrame, summary: TimingSummary) -> float:
    """
    把择时结论换算成**可成交价**（口径见 `_analyze_single_signal` docstring）。

    与 `lib.execution.daily_panel()` 的 `go_price` 同口径，两处若要改动请一起改；
    `tests/test_intraday_exec_price.py` 守住这条一致性。
    """
    last_close = float(exec_bars["close"].iloc[-1])
    if not (summary.has_go and summary.first_go is not None):
        return last_close
    pos = exec_bars.index.get_loc(summary.first_go)
    if pos + 1 >= len(exec_bars):
        return last_close                      # GO 在最后一根，身后没有可成交的 K 线
    return float(exec_bars["open"].iloc[pos + 1])


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
            code, name, candidate["date"], index_symbol, lookback_days)
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
        tracker = PositionTracker(capital=100_000)
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
        description="JCY 日线信号 + 分时择时（多周期共振）"
    )
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
    parser.add_argument("--output",    type=str, default="output/intraday",
                        help="输出目录，默认 output/intraday/")
    return parser.parse_args()


def main():
    args = parse_args()

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

    print("\n" + "─" * 65)
    print(f"  完成。结果已保存至：{os.path.abspath(args.output)}/")
    print("─" * 65)


if __name__ == "__main__":
    main()
