"""
JCY 分级买入 + 三级递进卖出仓位管理器。

从 `backtest_jcy_intraday`（原 jcy_intraday_timing，1349 行 CLI）拆出——
`PositionTracker` 与 `TradeRecord` 是纯计算逻辑（只依赖 dataclass 与 pandas），
不碰网络、不碰 matplotlib，因此单独成库让测试直接导入，不必 `importorskip`
进那个大 CLI 脚本。

执行时点口径（2026-08-09 修正，见 `PositionTracker.run` docstring）：
**统一 T+1。** 日线的 signal / 红柱缩短 / 死叉 / DIF<0 都要等当日收盘才算得
出来，所以每一条都排到下一个交易日成交。`intraday_map` 只换成交价、不换成交日。
"""

from dataclasses import dataclass, field

import pandas as pd


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
          ① `intraday_map` 给的分时价（`executable_price`，见 lib/execution）
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
