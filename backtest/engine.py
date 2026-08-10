"""
回测引擎（核心，无 CLI，**不含绘图**）
======================================
被所有回测入口脚本复用：`run_backtest` 执行回测并返回结果 dict，
`print_summary` 打印文本汇总。

    from engine import run_backtest
    result = run_backtest("600519", "20200101", "20241231")

图表在 `report.py`（`from report import plot_backtest`）——引擎保持纯计算，
批量回测/参数扫描/pytest 这些不出图的场景就不必拖进 matplotlib。
"""

import warnings

import numpy as np
import pandas as pd

from lib.market_data import fetch_stock_data   # noqa: F401 — re-export for backward compat
# 成本假设与 `lib/ladder.py` 共用同一份定义，不在这里另写字面量——两套撮合骨架的
# 数字一旦漂开，二元引擎与分批建仓模拟器的结果就不可横向比较。
# `_commission` / `infer_limit_pct` 保留旧名 re-export，历史导入不受影响。
from lib.costs import (COMMISSION_RATE, LOT, MIN_COMMISSION, SLIPPAGE,
                       STAMP_DUTY, infer_limit_pct)   # noqa: F401
from lib.costs import commission as _commission

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# A 股交易规则
# ─────────────────────────────────────────

def _tradability(row, prev_close: float, limit_pct: float) -> tuple[bool, bool]:
    """
    判断当日开盘能否买入 / 卖出，返回 (can_buy, can_sell)。

    - 停牌（无成交量）：买卖都不可能成交
    - 开盘即涨停：买单排在队尾，成交不了；卖出不受影响
    - 开盘即跌停：卖不掉；买入不受影响

    价格用相对容差比较而不是 round(_, 2)：复权后的价格已不是真实报价，
    绝对分位对不上。
    """
    if row.get("volume", 1) is not None and float(row.get("volume", 1)) <= 0:
        return False, False
    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        return True, True

    tol = 1e-4
    open_ = row["open"]
    limit_up   = prev_close * (1 + limit_pct)
    limit_down = prev_close * (1 - limit_pct)
    can_buy  = open_ < limit_up   * (1 - tol)
    can_sell = open_ > limit_down * (1 + tol)
    return bool(can_buy), bool(can_sell)




# ─────────────────────────────────────────
# 回测引擎
# ─────────────────────────────────────────

def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy=None,                      # BaseStrategy 实例，默认使用 MACDStrategy
    initial_capital: float = 100_000.0,
    # 四个成本默认值来自 lib/costs.py（与 lib/ladder.py 同源），可被 .ini 逐项覆盖
    commission_rate: float = COMMISSION_RATE,
    min_commission: float = MIN_COMMISSION,
    stamp_duty: float = STAMP_DUTY,
    slippage: float = SLIPPAGE,
    position_size: float = 1.0,         # 每次建仓比例（1.0 = 全仓）
    stop_loss: float = None,            # 止损比例，如 0.08 = 8%，None = 不止损
    take_profit: float = None,          # 止盈比例，如 0.20 = 20%，None = 不止盈
    limit_move_check: bool = True,      # 是否模拟涨跌停/停牌无法成交
    limit_pct: float = None,            # 涨跌停幅度，None = 按代码前缀推断
    max_pending_days: int = 3,          # 信号因涨跌停未成交时最多顺延几个交易日
    eval_start: str = None,             # 统计口径起点 "YYYYMMDD"，None = 全区间
    df: pd.DataFrame = None,            # 直接注入行情（测试/复用，跳过网络请求）
    verbose: bool = False,              # True 时打印表头/数据量/结果汇总（库默认静默）
) -> dict:
    """
    核心回测函数，返回回测统计结果 dict。

    verbose=False（默认）时引擎不打印任何内容，由调用方决定如何展示结果，
    避免批量回测刷屏；需要完整汇总表时传 verbose=True 或自行调用 print_summary。

    单根 K 线内的事件顺序（与实盘一致）
    ------------------------------------
      1. 开盘：执行上一日产生的信号（受涨跌停/停牌约束，成交不了则顺延挂单）
      2. 盘中：检查止损/止盈；**当日刚建仓的不检查**（A 股 T+1，当天买当天卖不掉）
      3. 收盘：按收盘价给持仓估值，记入权益曲线

    旧实现把第 2 步放在第 1 步之前，会出现"盘中最低价止损出场后，又用当天
    开盘价重新买入"——买在了比出场更早的时点上。

    eval_start
    ----------
      批量回测常在推荐日前多取一年数据给 MACD 预热，这段时间策略不交易、
      权益恒为初始资金。若把它算进统计，年化被摊薄、夏普被稀释、基准还多算
      了一年买入持有收益。传 eval_start 后，收益/回撤/夏普/基准全部只在
      该日之后的区间上计算，图表仍显示完整区间（保留指标上下文）。
    """

    from strategies import MACDStrategy
    if strategy is None:
        strategy = MACDStrategy()
    if limit_pct is None:
        limit_pct = infer_limit_pct(symbol)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  A股策略回测  [{strategy.name}]")
        print(f"  股票代码：{symbol}  周期：{start_date} → {end_date}")
        print(f"{'='*55}")

    # ── 获取数据 ──
    if df is None:
        df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty or len(df) < 50:
        raise ValueError("数据不足，请检查股票代码或延长时间范围")

    if verbose:
        print(f"  获取到 {len(df)} 个交易日数据")

    # ── 计算指标 + 生成信号（委托给策略） ──
    df = strategy.prepare(df)

    # ── 执行信号：T 日产生的信号在 T+1 日开盘成交，消除前视偏差 ──
    # signal 用 T 日收盘价算出，不可能在 T 日收盘价成交；shift(1) 后
    # 第 T 日看到的 signal_exec 实为 T-1 日信号，配合 T 日 open 成交。
    df["signal_exec"] = df["signal"].shift(1).fillna(0)
    prev_close_series = df["close"].shift(1)

    # ── 模拟交易 ──
    cash       = initial_capital
    shares     = 0
    position   = 0          # 0=空仓，1=持仓
    cost_price = 0.0        # 买入成本价（已含买入滑点，即真实持仓成本）
    entry_date = None       # 建仓日；T+1 规则下当日不得卖出

    entry_fee  = 0.0        # 建仓时付出的佣金；平仓算净收益要把它摊进成本

    pending     = 0         # 未成交的挂单方向：1=待买 / -1=待卖 / 0=无
    pending_age = 0         # 挂单已顺延的交易日数
    abandoned   = 0         # 已因超时作废的挂单方向；信号消失前不再重新挂单
    blocked     = []        # 因涨跌停/停牌未能成交的记录

    trades = []             # 交易记录
    equity = []             # 每日资产

    def _sell(date, raw_price, action):
        """
        按 raw_price 卖出全部持仓，返回成交价（已含滑点）。

        return_pct 为**净收益**：买入佣金 + 卖出佣金 + 印花税全部计入。
        毛收益另记在 gross_return_pct，便于对照成本吃掉了多少。
        对一个平均持仓只有几天的高频策略，往返成本约占 0.2%，用毛收益算
        胜率会把一批实际亏损的交易记成盈利。
        """
        nonlocal cash, shares, position, cost_price, entry_date, entry_fee
        exec_price = raw_price * (1 - slippage)
        proceeds   = shares * exec_price
        commission = _commission(proceeds, commission_rate, min_commission)
        duty       = proceeds * stamp_duty
        cash += proceeds - commission - duty

        outlay = shares * cost_price + entry_fee     # 建仓总支出（含买入佣金）
        net    = proceeds - commission - duty        # 平仓净回款
        trades.append({
            "date": date, "action": action, "price": exec_price,
            "shares": shares, "cash": cash,
            "return_pct": (net - outlay) / outlay * 100,
            "gross_return_pct": (exec_price - cost_price) / cost_price * 100,
            "commission": commission,
            "stamp_duty": duty,
            # 滑点成本：相对未滑点的理论价少收到的部分
            "slippage_cost": shares * raw_price * slippage,
        })
        shares = 0; position = 0; cost_price = 0.0
        entry_date = None; entry_fee = 0.0
        return exec_price

    for date, row in df.iterrows():
        price = row["close"]          # 估值用收盘价
        open_ = row["open"]           # 成交用开盘价（T+1 开盘）
        low   = row["low"]
        high  = row["high"]
        prev_close = prev_close_series.get(date, float("nan"))

        if limit_move_check:
            can_buy, can_sell = _tradability(row, prev_close, limit_pct)
        else:
            can_buy = can_sell = True

        # ── 挂单簿更新：新信号覆盖旧挂单 ──
        # 动能策略在持续期会**连日发出同向信号**。若每天都把 pending_age 归零、
        # 或在超时作废后立刻用当天的同向信号重新挂单，max_pending_days 就形同
        # 虚设——一字连板期间挂单被无限续期，最终追在最高点上。
        # 因此：同向信号重复出现不重置年龄；已作废的方向要等信号真正消失
        # （出现 0 或反向信号）才允许重新挂单。
        sig = int(row["signal_exec"])
        if sig != abandoned:
            abandoned = 0
        if sig != 0 and sig != abandoned and pending != sig:
            pending, pending_age = sig, 0
        # 方向与当前仓位矛盾的挂单直接作废（已持仓还挂买 / 空仓还挂卖）
        if (pending == 1 and position == 1) or (pending == -1 and position == 0):
            pending, pending_age = 0, 0

        # ── 步骤 1：开盘执行挂单 ──
        if pending == 1 and position == 0:
            if can_buy:
                exec_price = open_ * (1 + slippage)
                budget     = cash * position_size
                lots       = int(budget / exec_price / LOT)
                # 佣金可能让总支出超出可用现金，逐手回退直到付得起
                while lots >= 1:
                    cost = lots * LOT * exec_price
                    fee  = _commission(cost, commission_rate, min_commission)
                    if cost + fee <= cash:
                        break
                    lots -= 1
                if lots >= 1:
                    shares = lots * LOT
                    cost   = shares * exec_price
                    fee    = _commission(cost, commission_rate, min_commission)
                    cash  -= (cost + fee)
                    position   = 1
                    cost_price = exec_price
                    entry_date = date
                    entry_fee  = fee
                    pending, pending_age = 0, 0
                    trades.append({
                        "date": date, "action": "买入", "price": exec_price,
                        "shares": shares, "cash": cash,
                        "return_pct": None, "gross_return_pct": None,
                        "commission": fee, "stamp_duty": 0.0,
                        "slippage_cost": shares * open_ * slippage,
                    })
                else:
                    pending, pending_age = 0, 0     # 钱不够一手，放弃
            else:
                blocked.append({"date": date, "action": "买入受阻",
                                "reason": "开盘涨停或停牌"})

        elif pending == -1 and position == 1:
            if can_sell:
                _sell(date, open_, "卖出")
                pending, pending_age = 0, 0
            else:
                blocked.append({"date": date, "action": "卖出受阻",
                                "reason": "开盘跌停或停牌"})

        # 未成交的挂单顺延，超过上限则放弃（避免一字板连板时追到天上）
        if pending != 0:
            pending_age += 1
            if pending_age > max_pending_days:
                abandoned = pending
                pending, pending_age = 0, 0

        # ── 步骤 2：盘中止损 / 止盈 ──
        # T+1：当日建仓的不检查，当天买入无法当天卖出
        if position == 1 and shares > 0 and entry_date != date and can_sell:
            stop_price = cost_price * (1 - stop_loss)   if stop_loss   is not None else None
            tp_price   = cost_price * (1 + take_profit) if take_profit is not None else None
            exit_price = exit_action = None
            # 同一根 K 线内若同时触及，保守起见优先止损
            if stop_price is not None and low <= stop_price:
                exit_price  = min(open_, stop_price)    # 跳空低开则按开盘价
                exit_action = "止损卖出"
            elif tp_price is not None and high >= tp_price:
                exit_price  = max(open_, tp_price)      # 跳空高开则按开盘价
                exit_action = "止盈卖出"
            if exit_price is not None:
                _sell(date, exit_price, exit_action)

        # ── 步骤 3：收盘估值 ──
        # 记下当日收盘时的持股数：汇总层要靠它算**在场比例**。空仓日的权益不动，
        # 回撤和夏普都天然好看，不知道在场比例就没法判断那是"控制住了风险"
        # 还是"根本没参与"。
        equity.append({"date": date, "equity": cash + shares * price,
                       "close": price, "shares": shares})

    # 如果结束时仍持仓，按最后收盘价清算
    if shares > 0:
        _sell(df.index[-1], df["close"].iloc[-1], "期末清仓")
        equity[-1]["equity"] = cash

    # ── 统计指标 ──
    eq_full = pd.DataFrame(equity).set_index("date")

    # 统计窗口：预热期权益恒等于初始资金，计入会摊薄年化并稀释夏普
    trades_full = pd.DataFrame(trades)
    if eval_start:
        cut = pd.to_datetime(eval_start, format="%Y%m%d")
        eq_df = eq_full.loc[eq_full.index >= cut]
        px    = df.loc[df.index >= cut]
        if eq_df.empty:
            raise ValueError(f"eval_start={eval_start} 之后没有任何交易日")
        # 交易级统计同样只看窗口内：预热期通常无成交，但调用方若在预热期放了
        # 信号，这些成交不该混进胜率与成本统计。
        # 已知边界：跨窗口边界的那笔（窗口前买入、窗口内卖出）按卖出日归属。
        win_trades = (trades_full[trades_full["date"] >= cut]
                      if not trades_full.empty else trades_full)
    else:
        eq_df, px = eq_full, df
        win_trades = trades_full

    eq_df = eq_df.copy()
    # 窗口起点的基准资金：取窗口前最后一天的权益。
    # 正常用法下预热期不交易，它就等于 initial_capital；但若调用方在
    # eval_start 之前也放了信号，这里才不会把预热期的盈亏算进窗口收益。
    prior = eq_full.loc[eq_full.index < eq_df.index[0], "equity"]
    base_equity = float(prior.iloc[-1]) if len(prior) else initial_capital
    eq_df["returns"]  = eq_df["equity"].pct_change()
    eq_df["drawdown"] = eq_df["equity"] / eq_df["equity"].cummax() - 1

    total_return    = (eq_df["equity"].iloc[-1] / base_equity - 1) * 100
    annual_trading_days = 252
    n_days          = len(eq_df)
    # 年化收益：样本不足一年时按日历外推会几何放大（如 10 天 +8% → 年化 ~600%），
    # 失去统计意义，因此样本 < 一年时返回 None，由展示层标注 N/A。
    if n_days >= annual_trading_days:
        annual_return = ((1 + total_return / 100) ** (annual_trading_days / n_days) - 1) * 100
    else:
        annual_return = None
    max_drawdown    = eq_df["drawdown"].min() * 100
    sharpe          = _calc_sharpe(eq_df["returns"], annual_trading_days)
    win_rate, avg_win, avg_loss, profit_factor = _calc_trade_stats(
        win_trades.to_dict("records") if not win_trades.empty else []
    )

    # ── 交易成本：窗口内实际付出的钱 ──
    # 只报费率假设是不够的——高频策略的成本吃掉多少收益，必须直接看金额。
    # cost_drag_pct 的含义：这些成本相当于窗口起点资金的百分之几。
    def _cost_sum(col: str) -> float:
        return float(win_trades[col].sum()) if not win_trades.empty else 0.0

    total_commission = _cost_sum("commission")
    total_stamp_duty = _cost_sum("stamp_duty")
    total_slippage   = _cost_sum("slippage_cost")
    total_cost       = total_commission + total_stamp_duty + total_slippage

    # ── 暴露度：窗口内有多少比例的交易日真的持有仓位 ──
    # 空仓日既不赚也不亏，回撤与夏普都被"没参与"美化过。不给出在场比例，
    # 就没法判断一个 -6% 的最大回撤是控制住了风险，还是全程躺在现金里。
    n_buys       = (0 if win_trades.empty
                    else int((win_trades["action"] == "买入").sum()))
    in_market    = eq_df["shares"] > 0
    exposure_pct = float(in_market.mean()) * 100
    # 平均持仓天数：在场天数 ÷ 建仓次数。跨窗口边界的持仓会让它略偏大。
    avg_holding_days = (float(in_market.sum()) / n_buys) if n_buys else None

    # 基准（买入持有）：与策略同口径 —— 同一个统计窗口、同样开盘买入收盘估值
    bench_base   = px["open"].iloc[0]
    bench_return = (px["close"].iloc[-1] / bench_base - 1) * 100

    result = {
        "symbol": symbol, "start": start_date, "end": end_date,
        "eval_start": eval_start,
        "initial_capital": initial_capital,
        "equity_base": base_equity,     # 统计窗口起点权益（无 eval_start 时 == initial_capital）
        "final_equity": eq_df["equity"].iloc[-1],
        "total_return": total_return,
        "annual_return": annual_return,
        "benchmark_return": bench_return,
        "benchmark_base": bench_base,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "total_trades": n_buys,
        "exposure_pct": exposure_pct,           # 在场交易日占比（%）
        "avg_holding_days": avg_holding_days,   # 平均持仓交易日数（无成交时 None）
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "trades": trades_full,
        "blocked_trades": pd.DataFrame(blocked),
        "equity_curve": eq_df,
        "equity_curve_full": eq_full,
        "df": df,
        "strategy": strategy,
        "costs": {
            "commission_rate": commission_rate,
            "min_commission": min_commission,
            "stamp_duty": stamp_duty,
            "slippage": slippage,
            "limit_pct": limit_pct if limit_move_check else None,
            # 实际发生额（仅统计窗口内）
            "total_commission": total_commission,
            "total_stamp_duty": total_stamp_duty,
            "total_slippage": total_slippage,
            "total_cost": total_cost,
            "cost_drag_pct": total_cost / base_equity * 100,
        },
    }

    if verbose:
        _print_summary(result)
    return result


def _calc_sharpe(returns: pd.Series, annual_days: int = 252, rf: float = 0.02,
                 min_obs: int = 20):
    """
    年化夏普比率。样本过少时不可靠，返回 None（展示层标注 N/A）。

    min_obs : 有效日收益的最小样本数；少于此值时 std 极不稳定，
              可能算出 18 这种无意义高值或 NaN，故直接弃算。
    """
    daily_rf = rf / annual_days
    excess   = (returns - daily_rf).dropna()      # 去掉 pct_change 首行 NaN
    if len(excess) < min_obs:
        return None
    std = excess.std()
    if not np.isfinite(std) or std == 0:
        return None
    return float(excess.mean() / std * np.sqrt(annual_days))


def _calc_trade_stats(trades: list):
    # 必须同时排除 None 和 NaN：调用方传的是 DataFrame.to_dict("records")，
    # pandas 已把建仓行的 return_pct=None 转成 float NaN。只判 `is not None`
    # 会把建仓记录算进胜率分母（NaN 既不 >0 也不 <=0，于是分子不变、分母翻倍，
    # 一笔盈利的交易会被报成 50% 胜率）。
    closed = [t for t in trades
              if t.get("return_pct") is not None and not pd.isna(t["return_pct"])]
    if not closed:
        return 0.0, 0.0, 0.0, 0.0
    wins   = [t["return_pct"] for t in closed if t["return_pct"] > 0]
    losses = [t["return_pct"] for t in closed if t["return_pct"] <= 0]
    win_rate     = len(wins) / len(closed) * 100 if closed else 0
    avg_win      = np.mean(wins)   if wins   else 0
    avg_loss     = np.mean(losses) if losses else 0
    total_win    = sum(wins)
    total_loss   = abs(sum(losses))
    profit_factor = total_win / total_loss if total_loss > 0 else 0.0
    return win_rate, avg_win, avg_loss, profit_factor


def fmt_sharpe(v) -> str:
    """夏普比率展示格式：None（样本不足）显示 N/A，否则两位小数。"""
    return "N/A" if v is None else f"{v:.2f}"


def print_summary(r: dict):
    """打印回测结果汇总表（公开接口，供单股入口脚本在静默引擎后手动调用）。"""
    _print_summary(r)


def _print_summary(r: dict):
    print(f"\n  ── 回测结果 ──────────────────────────────")
    c = r.get("costs", {})
    if c:
        limit_txt = "不限" if c.get("limit_pct") is None else f"±{c['limit_pct']:.0%}"
        print(f"  成本假设      : 佣金{c['commission_rate']:.2%}(最低{c['min_commission']:.0f}元) "
              f"印花税{c['stamp_duty']:.2%} 滑点{c['slippage']:.2%} 涨跌停{limit_txt}")
    if r.get("eval_start"):
        print(f"  统计起点      : {r['eval_start']}（此前为指标预热期，不计入）")
    print(f"  初始资金      : ¥{r['initial_capital']:>12,.2f}")
    print(f"  期末资产      : ¥{r['final_equity']:>12,.2f}")
    print(f"  策略总收益    : {r['total_return']:>+8.2f}%")
    ann = r['annual_return']
    print(f"  策略年化收益  : {'  N/A(样本<1年)' if ann is None else f'{ann:>+8.2f}%'}")
    print(f"  基准收益(持有): {r['benchmark_return']:>+8.2f}%")
    print(f"  超额收益      : {r['total_return'] - r['benchmark_return']:>+8.2f}%")
    print(f"  最大回撤      : {r['max_drawdown']:>8.2f}%")
    shp = r['sharpe_ratio']
    print(f"  夏普比率      : {'  N/A(样本不足)' if shp is None else f'{shp:>8.2f}'}")
    print(f"  交易次数      : {r['total_trades']:>8}  次")
    hold = r.get("avg_holding_days")
    print(f"  在场比例      : {r['exposure_pct']:>8.1f}%  "
          f"（平均持仓 {'N/A' if hold is None else f'{hold:.1f}'} 个交易日）"
          f"  ← 回撤/夏普未按暴露度调整")
    if c.get("total_cost") is not None:
        print(f"  交易成本合计  : ¥{c['total_cost']:>12,.2f}  "
              f"（占起点资金 {c['cost_drag_pct']:.2f}%）")
        print(f"    其中 佣金 ¥{c['total_commission']:,.2f}  "
              f"印花税 ¥{c['total_stamp_duty']:,.2f}  "
              f"滑点 ¥{c['total_slippage']:,.2f}")
    print(f"  胜率(净额)    : {r['win_rate']:>8.1f}%")
    print(f"  平均盈利      : {r['avg_win']:>+8.2f}%")
    print(f"  平均亏损      : {r['avg_loss']:>+8.2f}%")
    pf = r['profit_factor']
    print(f"  盈亏比        : {'    N/A' if pf == 0 else f'{pf:>8.2f}'}")
    blocked = r.get("blocked_trades")
    if blocked is not None and not blocked.empty:
        print(f"  受阻未成交    : {len(blocked):>8}  次（涨跌停/停牌）")
    print(f"  ──────────────────────────────────────────")
