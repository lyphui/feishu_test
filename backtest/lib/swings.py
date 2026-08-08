"""
波段与回撤剖面：把一段价格序列拆成"涨了多少、跌了多少、跌完多久回来"。

这是给**分批建仓**定参数用的：网格间距不该拍脑袋定 5% 或 10%，而该来自这只
票自己的回撤分布——历史上一次典型回调有多深、多久修复，决定了梯子该铺多宽、
铺几档、以及在最坏情况下要压多久的时间成本。
"""

import pandas as pd


# ── ZigZag 波段分解 ───────────────────────────────────────────────────────────

def zigzag(close: pd.Series, pct: float = 0.08) -> list[tuple]:
    """
    ZigZag 拐点分解：返回 [(date, price, kind), ...]，kind ∈ start/top/bottom/end。

    从上一个已确认拐点起，同时跟踪running high 与 running low；先被击穿 `pct`
    的那一侧确认方向。方向未定时两侧都要独立跟踪 —— 若共用一个"极值"变量，
    上涨时被抬高、下跌时被压低，极值恒等于当前价，除非出现单日 ≥pct 的跳空，
    否则永远确认不了第一个拐点，序列前半段的波段会整段丢失。
    """
    if len(close) < 2:
        return []

    piv = [(close.index[0], float(close.iloc[0]), "start")]
    hi_i = lo_i = close.index[0]
    hi_p = lo_p = float(close.iloc[0])
    direction = 0                                  # 0=未定, 1=上行, -1=下行

    for t, p in close.items():
        p = float(p)
        if p > hi_p:
            hi_i, hi_p = t, p
        if p < lo_p:
            lo_i, lo_p = t, p

        if direction >= 0 and p <= hi_p * (1 - pct):
            piv.append((hi_i, hi_p, "top"))
            direction = -1
            # 新的下行段从这个顶开始找底，低点要在 [顶, 当前] 窗口内重新取
            seg = close.loc[hi_i:t]
            lo_i, lo_p = seg.idxmin(), float(seg.min())
            hi_i, hi_p = t, p
        elif direction <= 0 and p >= lo_p * (1 + pct):
            piv.append((lo_i, lo_p, "bottom"))
            direction = 1
            seg = close.loc[lo_i:t]
            hi_i, hi_p = seg.idxmax(), float(seg.max())
            lo_i, lo_p = t, p

    piv.append((close.index[-1], float(close.iloc[-1]), "end"))
    return piv


def swing_table(close: pd.Series, pct: float = 0.08) -> pd.DataFrame:
    """ZigZag 段落表：起止日期、自然日历时、涨跌幅。"""
    piv = zigzag(close, pct)
    rows = [{"start": t0, "end": t1, "days": (t1 - t0).days, "pct": p1 / p0 - 1}
            for (t0, p0, _), (t1, p1, _) in zip(piv, piv[1:]) if t1 > t0]
    return pd.DataFrame(rows)


# ── 回撤事件 ──────────────────────────────────────────────────────────────────

def drawdown_episodes(close: pd.Series) -> pd.DataFrame:
    """
    以"创新高 → 再创新高"切分独立回撤事件。

    不要用"收盘价低于前高 X%"逐日计数：价格在阈值上下反复穿越时，同一轮回调
    会被记成几十次，频次统计完全失真。一次回撤 = 一段从前高跌下去、直到重新
    收复前高为止的完整历程。

    返回 peak_date / trough_date / recover_date / depth / fall_days / recover_days，
    最后一次尚未收复前高时 recover_date 为 NaT。
    """
    close = close.dropna()
    if close.empty:
        return pd.DataFrame(columns=["peak_date", "trough_date", "recover_date",
                                     "depth", "fall_days", "recover_days"])

    eps = []
    peak_p, peak_t = float(close.iloc[0]), close.index[0]
    in_dd = False
    trough_p = trough_t = None

    for t, p in close.items():
        p = float(p)
        if p >= peak_p:
            if in_dd:
                eps.append({"peak_date": peak_t, "trough_date": trough_t,
                            "recover_date": t, "depth": trough_p / peak_p - 1,
                            "fall_days": (trough_t - peak_t).days,
                            "recover_days": (t - trough_t).days})
                in_dd = False
            peak_p, peak_t = p, t
        else:
            if not in_dd:
                in_dd, trough_p, trough_t = True, p, t
            elif p < trough_p:
                trough_p, trough_t = p, t

    if in_dd:
        eps.append({"peak_date": peak_t, "trough_date": trough_t,
                    "recover_date": pd.NaT, "depth": trough_p / peak_p - 1,
                    "fall_days": (trough_t - peak_t).days, "recover_days": float("nan")})

    return pd.DataFrame(eps)


def drawdown_profile(close: pd.Series,
                     thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)) -> pd.DataFrame:
    """各深度档位的回撤事件数与修复耗时中位/最长（自然日）。"""
    ep = drawdown_episodes(close)
    rows = []
    for th in thresholds:
        sub = ep[ep["depth"] <= -th]
        rec = sub["recover_days"].dropna()
        rows.append({
            "threshold": th,
            "episodes": len(sub),
            "recovered": len(rec),
            "median_recover_days": rec.median() if len(rec) else float("nan"),
            "max_recover_days": rec.max() if len(rec) else float("nan"),
        })
    return pd.DataFrame(rows)
