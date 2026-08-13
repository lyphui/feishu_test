"""
月频均线 + 移动止损：给**高波动、高单笔成本**的商品 ETF 用的低频趋势跟踪。

为什么是「月频」而不是日频
--------------------------
在 3175.HK（三星标普高盛原油期货 ETF）上做过参数扫描：同一条均线规则，
决策频率从日频换到月频，八个均线长度（MA80~MA250）的年化收益从
「一半为负」变成「全部为正」，最大回撤从 -70% 收窄到 -34%，交易次数从
每年 8~13 次降到 1.8~3.1 次。**决策频率比均线长度重要得多。**

原因有两条，缺一不可：
  1. 中银香港这类零售渠道的最低佣金是每笔 HK$100 起（见 `hk_trade_cost`）。
     10 万港币本金下每次换仓约 0.29%，日频每年 10 次以上直接吃掉全部超额。
  2. 原油期货 ETF 年化波动 40%+，日频均线在噪音里反复穿越，假信号本身
     就是负收益，跟手续费叠加是双杀。

为什么必须有卖出规则
--------------------
这类 ETF 的收益 = 油价涨跌 + 展期收益，而展期收益在 contango 期间是**持续
失血**。3175 上市十年买入持有年化 -2.3%、最大回撤 -89.6%；仅 2020 一年，
WTI 近月跌 20.5%，ETF 跌 75.4%——差出来的 55 个百分点全是 contango。
所以这里的止损不是锦上添花，是这个品种能不能拿住的前提。

一条走过的弯路（别再试了）
--------------------------
"用已实现展期收益（ETF 收益 - 近月涨跌）择时"这条路测过，不通：该指标的
自相关只有 0.029，与未来 60 日收益相关性 -0.044，分组看方向还是反的
（最 contango 组未来 60 日 +5.4%，最 backwardation 组 -1.9%）。
当下的期限结构是**状态描述**，不是预测。真正的远期曲线斜率拿不到历史数据，
所以无法证伪"真曲线有没有用"——但这个代理没有预测力，不该拿它下注。

时序铁律
--------
月末收盘价与均线比较 → **次一交易日**才建/平仓（`decide[i-1]` 决定 `pos[i]`）。
唯一的日内动作是移动止损：持仓期间收盘价从**入场后最高收盘价**回落超过
`stop`，当日收盘离场，不等月末。止损后必须等下一个月末、信号仍为多才重入。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 港股法定/结算费率（与盘型无关，只跟成交金额挂钩）
SFC_LEVY = 0.000027          # 证监会交易征费
AFRC_LEVY = 0.0000015        # 财汇局交易征费
HKEX_TRADING_FEE = 0.0000565  # 联交所交易费
CCASS_RATE = 0.00002          # 中央结算费
CCASS_MIN, CCASS_MAX = 2.0, 100.0
STAMP_DUTY = 0.001            # 印花税，ETF 豁免


def hk_trade_cost(
    value: float,
    *,
    commission_rate: float = 0.0025,
    commission_min: float = 100.0,
    platform_fee: float = 30.0,
    is_etf: bool = True,
) -> float:
    """
    单边交易总成本（港币）。默认参数取自中银香港零售网上渠道。

    `commission_min` 是这套策略全部设计的起点：成交额低于
    `commission_min / commission_rate`（默认 4 万港币）时，佣金按最低收，
    费用率随金额下降而爆炸——600 股 ≈ 5,790 港币的单子费用率高达 2.29%。

    `platform_fee` 是银行自定的固定服务费，各家不同，按成交单实际列示调整。
    ETF 在香港豁免印花税（`is_etf=True`），个股要另加 0.1%。
    """
    if value <= 0:
        return 0.0
    commission = max(value * commission_rate, commission_min)
    levies = value * (SFC_LEVY + AFRC_LEVY + HKEX_TRADING_FEE)
    ccass = min(max(value * CCASS_RATE, CCASS_MIN), CCASS_MAX)
    stamp = 0.0 if is_etf else value * STAMP_DUTY
    return commission + platform_fee + levies + ccass + stamp


def hk_fee_rate(value: float, **kw) -> float:
    """单边成本占成交额的比例。回测里的 `fee` 参数就用它算。"""
    return hk_trade_cost(value, **kw) / value if value > 0 else 0.0


def month_end_flags(index) -> pd.Series:
    """每月最后一个**交易日**为 True（按实际存在的行判定，不是自然月末）。"""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("max") == idx


def _decision_flags(index, freq: str) -> np.ndarray:
    if freq == "month":
        return month_end_flags(index).to_numpy()
    if freq == "day":
        return np.ones(len(index), dtype=bool)
    raise ValueError(f"未知决策频率：{freq}（可选 day / month）")


@dataclass
class TrendStopResult:
    name: str
    position: pd.Series            # 0/1，第 i 天收盘成交后建立的仓位（净值里再滞后一天）
    equity: pd.Series              # 净值曲线（起点 1.0，已扣费）
    trades: pd.DataFrame           # 每段持仓一行
    stats: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)   # 末日的持仓状态，供实盘用


def simulate(
    df: pd.DataFrame,
    *,
    ma_len: int = 150,
    stop: float = 0.15,
    fee: float = 0.0029,
    freq: str = "month",
    name: str = None,
) -> TrendStopResult:
    """
    跑一遍规则，返回净值、逐段交易与末日状态。

    df   : 含 `close` 列、按日期升序的 DataFrame
    stop : 移动止损幅度（0.15 = 从入场后最高收盘价回落 15%）；None / 0 表示不设

    成交假设（都偏乐观，看结果时打个折）：
      * 月末信号次日按**收盘价**成交，不含冲击成本，只扣 `fee`
      * 止损在**触发当日收盘**离场——真实操作要等看到收盘价才能动手
    """
    close = df["close"].astype(float)
    if close.isna().any():
        close = close.ffill()
    ma = close.rolling(ma_len).mean()
    above = (close > ma).to_numpy()
    valid = ma.notna().to_numpy()
    decide = _decision_flags(close.index, freq)
    px = close.to_numpy()
    dates = close.index

    n = len(close)
    pos = np.zeros(n)
    holding, peak = False, None
    entry_i = None
    rows = []

    for i in range(1, n):
        # ① 月末（或周末/每日）看信号，次日执行
        if decide[i - 1] and valid[i - 1]:
            if above[i - 1] and not holding:
                holding, peak, entry_i = True, px[i], i
            elif not above[i - 1] and holding:
                rows.append((dates[entry_i], px[entry_i], dates[i], px[i], "信号转空"))
                holding, peak, entry_i = False, None, None
        # ② 持仓期间盯移动止损
        if holding:
            peak = px[i] if peak is None else max(peak, px[i])
            if stop and px[i] / peak - 1 < -stop:
                rows.append((dates[entry_i], px[entry_i], dates[i], px[i], "移动止损"))
                holding, peak, entry_i = False, None, None
        pos[i] = 1.0 if holding else 0.0

    position = pd.Series(pos, index=dates)
    ret = close.pct_change().fillna(0.0)
    # 净值必须用**滞后一天**的仓位：`pos[i]` 是按第 i 天的收盘价成交才拿到的，
    # 它赚的是第 i+1 天的涨跌。直接 `position * ret` 等于按第 i-1 天收盘价成交，
    # 而止损恰恰是**看见第 i 天的收盘价**才触发的——不滞后就把触发当天的那根
    # 阴线整段躲掉了，是彻头彻尾的未来函数。3175 上实测：年化被虚增到 9.5%
    # （实为 4.9%）、最大回撤被美化到 -37.6%（实为 -52.0%）。
    # 校验方法：单段持仓的净值涨幅应当等于 `exit_px / entry_px`（见测试）。
    held = position.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.iloc[0])   # 费用记在成交当天
    net = held * ret - turnover * fee
    equity = (1.0 + net).cumprod()

    trades = pd.DataFrame(rows, columns=["entry_date", "entry_px",
                                         "exit_date", "exit_px", "reason"])
    if len(trades):
        trades["ret"] = trades["exit_px"] / trades["entry_px"] - 1
        trades["days"] = (trades["exit_date"] - trades["entry_date"]).dt.days

    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    vol = float(net.std() * np.sqrt(252))
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    n_switch = int((turnover > 0).sum())
    stats = {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": cagr,
        "volatility": vol,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "sharpe": cagr / vol if vol else float("nan"),
        "n_trades": n_switch,
        "trades_per_year": n_switch / years,
        "exposure": float(held.mean()),
        "win_rate": float((trades["ret"] > 0).mean()) if len(trades) else float("nan"),
        "years": years,
    }

    state = {
        "holding": holding,
        "last_date": dates[-1],
        "last_close": float(px[-1]),
        "ma_len": ma_len,
        "ma": float(ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else None,
        "stop": stop,
        "entry_date": dates[entry_i] if holding else None,
        "entry_px": float(px[entry_i]) if holding else None,
        "peak": float(peak) if holding and peak is not None else None,
        "stop_px": float(peak * (1 - stop)) if holding and peak and stop else None,
    }
    if state["ma"]:
        state["ma_gap"] = state["last_close"] / state["ma"] - 1
        state["signal_long"] = state["last_close"] > state["ma"]
    freq_cn = {"day": "日", "week": "周", "month": "月"}.get(freq, freq)
    label = name or (f"MA{ma_len} {freq_cn}频"
                     + (f" 止损{stop:.0%}" if stop else " 无止损"))
    return TrendStopResult(label, position, equity, trades, stats, state)


def buy_hold(df: pd.DataFrame, fee: float = 0.0029) -> TrendStopResult:
    """满仓不动的对照组，口径与 `simulate` 完全一致，便于直接比。"""
    close = df["close"].astype(float).ffill()
    position = pd.Series(1.0, index=close.index)
    ret = close.pct_change().fillna(0.0)
    net = position * ret
    net.iloc[0] -= fee
    equity = (1.0 + net).cumprod()
    years = max((close.index[-1] - close.index[0]).days / 365.25, 1e-9)
    vol = float(net.std() * np.sqrt(252))
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    return TrendStopResult(
        "买入持有", position, equity, pd.DataFrame(),
        {"total_return": float(equity.iloc[-1] - 1), "annual_return": cagr,
         "volatility": vol,
         "max_drawdown": float((equity / equity.cummax() - 1).min()),
         "sharpe": cagr / vol if vol else float("nan"), "n_trades": 1,
         "trades_per_year": 1 / years, "exposure": 1.0,
         "win_rate": float("nan"), "years": years}, {})


def next_decision_date(last_date) -> pd.Timestamp:
    """
    下一个决策日 = 当月最后一个工作日；若已过则顺延到下月。

    只按工作日推算，**不含港交所假期**——真赶上假期，实际决策日会往前挪，
    以交易所日历为准。
    """
    d = pd.Timestamp(last_date).normalize()
    eom = d + pd.offsets.BMonthEnd(0)
    return eom if eom >= d else d + pd.offsets.BMonthEnd(1)


def sweep(df: pd.DataFrame, ma_lens, stops, *, freq: str = "month",
          fee: float = 0.0029) -> pd.DataFrame:
    """
    参数敏感性网格：看最优格子是不是一整片高原。

    单点最优毫无意义——相邻参数一掉下去就说明是拟合噪音。
    """
    rows = []
    for m in ma_lens:
        for s in stops:
            r = simulate(df, ma_len=m, stop=s, fee=fee, freq=freq)
            rows.append({"ma": m, "stop": s if s else 0.0,
                         "年化": r.stats["annual_return"],
                         "最大回撤": r.stats["max_drawdown"],
                         "夏普": r.stats["sharpe"],
                         "交易每年": r.stats["trades_per_year"],
                         "在场": r.stats["exposure"]})
    return pd.DataFrame(rows)
