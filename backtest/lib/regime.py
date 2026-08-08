"""
市场状态识别：把一只票当下处在什么"风格"里，用**当天就能算出来**的量判定。

为什么需要它
------------
同一套买卖规则在不同阶段的排名会翻转：趋势上行期，分批建仓因为一直踏空而
跑输满仓；宽幅震荡期，分批 + 部分止盈明显占优；趋势下行期，只有带趋势止损
的那一档能活下来。所以"选哪个策略"本质上是"现在是什么状态"。

铁律：只能用过去的数据
----------------------
所有指标都是 rolling / expanding，当天的标签只依赖当天及之前的收盘价。
用全样本分位数、或者事后画出来的"牛市区间"来打标签，回测会好看得不像话，
实盘一天都用不了 —— 那不是策略，是answer key。

状态还带滞回（`confirm_days`）：条件要连续成立若干天才换挡。否则价格在
均线上下反复穿越时，状态天天翻，策略跟着天天换，全是摩擦成本。
"""

import numpy as np
import pandas as pd

TREND_UP = "趋势上行"
CHOP = "宽幅震荡"
BEAR = "趋势下行"

LABELS = [TREND_UP, CHOP, BEAR]


def classify(
    df: pd.DataFrame,
    *,
    ma_long: int = 250,
    slope_win: int = 20,
    dd_win: int = 250,
    dd_limit: float = 0.12,
    confirm_days: int = 5,
) -> pd.DataFrame:
    """
    返回逐日状态表：regime / ma / slope / dd / vol，索引与 df 对齐。

    判定（长均线定方向，回撤定"还算不算在趋势里"）：
      趋势下行  收盘 < MA250 且 MA250 在走平下行
      趋势上行  收盘 > MA250 且 MA250 向上 且 距 250 日高点回撤浅于 dd_limit
      宽幅震荡  其余（含均线上方但深度回调、均线走平反复穿越）

    预热期（MA250 尚未成形）标为宽幅震荡：信息不足时按最中性的那档处理，
    既不满仓押趋势，也不空仓踏空。
    """
    close = df["close"].astype(float)
    ma = close.rolling(ma_long, min_periods=ma_long).mean()
    slope = ma.pct_change(slope_win)
    dd = close / close.rolling(dd_win, min_periods=1).max() - 1
    vol = close.pct_change().rolling(60, min_periods=20).std() * np.sqrt(252)

    raw = pd.Series(CHOP, index=close.index, dtype=object)
    valid = ma.notna() & slope.notna()
    raw[valid & (close < ma) & (slope < 0)] = BEAR
    raw[valid & (close > ma) & (slope > 0) & (dd > -dd_limit)] = TREND_UP

    # 滞回：新状态连续 confirm_days 天成立才真正切换
    out, cur, run = [], CHOP, 0
    prev = None
    for v in raw:
        run = run + 1 if v == prev else 1
        prev = v
        if v != cur and run >= confirm_days:
            cur = v
        out.append(cur)

    return pd.DataFrame({"regime": out, "ma": ma, "slope": slope,
                         "dd": dd, "vol": vol}, index=close.index)


def regime_stats(df: pd.DataFrame, reg: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """
    各状态下的**未来** `horizon` 日收益分布 —— 用来检验标签是否真的有信息量。

    如果三个状态的未来收益长得一模一样，那这个分类器就是噪声，
    基于它做策略切换只会白白增加摩擦。
    """
    close = df["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1
    g = pd.DataFrame({"regime": reg["regime"], "fwd": fwd}).dropna()
    out = g.groupby("regime")["fwd"].agg(
        天数="size", 未来均值="mean", 未来中位="median",
        胜率=lambda x: (x > 0).mean(), 最差="min", 最好="max")
    return out.reindex([l for l in LABELS if l in out.index])


def regime_episodes(reg: pd.DataFrame) -> pd.DataFrame:
    """把逐日标签压缩成连续区间，便于人工核对状态划得对不对。"""
    r = reg["regime"]
    grp = (r != r.shift()).cumsum()
    rows = []
    for _, seg in r.groupby(grp):
        rows.append({"regime": seg.iloc[0], "start": seg.index[0],
                     "end": seg.index[-1], "days": len(seg)})
    return pd.DataFrame(rows)
