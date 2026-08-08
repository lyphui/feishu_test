"""
日内下单方案的成交价测算（与标的、与策略无关的度量层）。

回答的问题只有一个：**已经决定要买了，用哪种下单方式成交价更好。**
这跟"买不买""买多少"无关——那些是策略的事，在 strategies/ 和 lib/ladder.py 里。

基准取当日 VWAP（`sum(amount) / sum(volume)`，全天成交的加权均价）。
为什么是 VWAP 而不是当日最低价：最低价只有事后才知道、谁也挂不到，
拿它当标准会让所有方案都是"失败"，比不出高下。VWAP 是**可达成的中性结果**
（把单子摊到全天慢慢买就大致能买到），因此"跑赢 VWAP"才是有意义的说法。
输出单位一律 bp（万分之一），负 = 买得比当天典型成交价便宜。

三条踩过的坑，写在这里免得再踩
------------------------------
1. **VWAP 与价格必须同一复权口径。**`amount`/`volume` 永远是原始成交额与股数，
   而 `close` 若取了前/后复权就跟它们不是一个尺度，算出来的偏离会是几百上千 bp
   的系统性错位（实测某次是 −1500bp，看着像"天天抄到底"）。
   拉分时数据请用**不复权**（baostock `adjustflag="3"`）。
   `daily_panel()` 会做一致性校验，口径不匹配直接抛错。

2. **"信号柱的收盘价"不是可成交价。**任何"这根 K 线满足条件"的判断，
   都要等这根 K 线走完才成立，那时它的收盘价已经成为历史。可成交的是
   **下一根 K 线的开盘价**。本模块统一按后者计价（`next_open`）。

3. **允许"不成交"的方案必须配一个强制兜底。**否则"没等到好价就不买"
   等于偷偷给策略加了一个免费的择时期权，回测会凭空变好。
   `add_limit_plan()` 强制要求 `fallback`，没得选。

用法
----
    from lib.execution import intraday_macd, daily_panel, add_limit_plan, benchmark

    bars = pd.read_csv(...)                     # 30min 不复权 K 线
    bars = intraday_macd(bars)                  # 追加 DIF/DEA/hist/go_buy
    panel = daily_panel(bars)                   # 压成每日一行
    panel = add_limit_plan(panel, offset=-0.005, fallback="close")
    print(benchmark(panel, ["open", "close", "go_price", "limit_-50bp"]))
"""

import numpy as np
import pandas as pd

REQUIRED = ["dt", "date", "open", "high", "low", "close", "volume", "amount"]

# close/vwap 的中位比值偏离 1 超过这个幅度，判定为复权口径不一致
_BASIS_TOL = 0.02


def intraday_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                  signal: int = 9) -> pd.DataFrame:
    """
    在**连续跨日**的分时序列上算 MACD，并标出买入侧的 GO 柱。

    与 `jcy_intraday_timing.add_macd` / `classify_timing(action="buy")` 同式：
        go_buy = 红柱为正且正在拉长 且 DIF > DEA

    必须在跨日连续序列上算——每天单独重算会让每日头几根柱子处在 EMA 预热期，
    结果由预热噪声主导。调用方自行丢弃开头若干根（建议 ≥ 2×slow）。
    """
    out = df.sort_values("dt").reset_index(drop=True).copy()
    ema_f = out["close"].ewm(span=fast, adjust=False).mean()
    ema_s = out["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    out["DIF"], out["DEA"], out["MACD"] = dif, dea, hist
    out["hist_expanding"] = (hist > 0) & (hist > hist.shift(1))
    out["go_buy"] = out["hist_expanding"] & (dif > dea)
    return out


def daily_panel(df: pd.DataFrame, *, go_col: str = "go_buy",
                warmup_bars: int = 60) -> pd.DataFrame:
    """
    把分时 K 线压成每日一行，产出各下单方案的**可成交价**。

    列：
      vwap        当日成交量加权均价（基准）
      open/close  当日开盘价（= 集合竞价成交价）/ 收盘价
      day_low     当日最低价
      rest_low    首根 K 线之后的全天最低价（判断限价单能否成交用）
      has_go      当天是否出现过 GO 柱（**事后信息**，只可用于归因，不可用于决策）
      go_time     首个 GO 柱的时刻
      go_price    按首个 GO 柱的**下一根开盘价**计的可成交价；
                  全天无 GO、或 GO 出现在最后一根 → 退化为收盘价

    `warmup_bars` 丢弃开头若干根（MACD 预热），默认 60（30min 下约 7.5 个交易日）。
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"分时数据缺列 {missing}；需要 {REQUIRED}")

    d = df.sort_values("dt").reset_index(drop=True).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["_nxt_open"] = d["open"].shift(-1)
    d["_nxt_date"] = d["date"].shift(-1)
    if warmup_bars:
        d = d.iloc[warmup_bars:]

    rows = []
    for day, g in d.groupby("date", sort=True):
        vol = g["volume"].sum()
        if len(g) < 2 or vol <= 0:
            continue
        vwap = g["amount"].sum() / vol
        if not np.isfinite(vwap) or vwap <= 0:
            continue
        go = g[g[go_col]] if go_col in g.columns else g.iloc[0:0]
        if len(go):
            f = go.iloc[0]
            go_px = (f["_nxt_open"] if (f["_nxt_date"] == day
                                        and pd.notna(f["_nxt_open"]))
                     else g.iloc[-1]["close"])
            go_time = pd.Timestamp(f["dt"]).strftime("%H:%M")
        else:
            go_px, go_time = g.iloc[-1]["close"], None
        rows.append({
            "date": day, "vwap": vwap,
            "open": g.iloc[0]["open"], "close": g.iloc[-1]["close"],
            "day_low": g["low"].min(), "rest_low": g["low"].iloc[1:].min(),
            "has_go": len(go) > 0, "go_time": go_time, "go_price": go_px,
        })
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    _check_same_basis(panel)
    return panel.reset_index(drop=True)


def _check_same_basis(panel: pd.DataFrame) -> None:
    """价格与 VWAP 必须同一复权口径，否则整张表的 bp 都是错位的。"""
    ratio = float((panel["close"] / panel["vwap"]).median())
    if abs(ratio - 1.0) > _BASIS_TOL:
        raise ValueError(
            f"close/vwap 中位比值 {ratio:.4f} 明显偏离 1。"
            f"amount/volume 恒为原始口径，而价格看起来做过复权——"
            f"两者不同尺度时所有 bp 都是系统性错位的。"
            f"请用不复权分时数据（baostock adjustflag='3'）")


def add_limit_plan(panel: pd.DataFrame, *, offset: float,
                   fallback: str = "close", name: str | None = None,
                   ref: str = "open") -> pd.DataFrame:
    """
    追加一个"挂限价单等回调"的方案，成交价写进新列。

    offset   相对 `ref` 的偏移，-0.005 = 挂在开盘价下方 0.5%
    fallback 全天未触及限价时的兜底成交列（**必填，不许省**）。
             允许"没成交就不买"等于给策略白送一个择时期权，回测会凭空变好。

    成交判定用 `rest_low`（首根之后的最低价）：开盘那一刻你还没挂上单，
    用 day_low 会把开盘瞬间的下影线也算成成交，高估成交率。
    """
    if fallback not in panel.columns:
        raise ValueError(f"fallback 列 {fallback!r} 不存在；必须给一个强制成交的兜底价")
    col = name or f"limit_{int(offset * 1e4):+d}bp"
    limit = panel[ref] * (1 + offset)
    filled = panel["rest_low"] <= limit
    out = panel.copy()
    out[col] = np.where(filled, limit, panel[fallback])
    out[col + "_filled"] = filled
    return out


def benchmark(panel: pd.DataFrame, cols: list[str],
              labels: dict[str, str] | None = None) -> pd.DataFrame:
    """
    各方案成交价相对当日 VWAP 的偏离（bp），带 t 值与显著性。

    负 = 买得比当天典型成交价便宜。`signif` 是 |t| > 1.96（均值为 0 的双尾检验）。
    同时给中位数：均值容易被少数暴涨暴跌日主导，两者同号才算稳。
    """
    labels = labels or {}
    rows = []
    for c in cols:
        s = ((panel[c] / panel["vwap"] - 1) * 1e4).dropna()
        if len(s) < 30:
            continue
        se = s.std(ddof=1) / np.sqrt(len(s))
        t = 0.0 if se == 0 else s.mean() / se       # 恒等于基准时方差为 0
        row = {"方案": labels.get(c, c), "n": len(s), "均值bp": s.mean(),
               "中位bp": s.median(), "t值": t, "signif": abs(t) > 1.96}
        fc = c + "_filled"
        row["成交率"] = f"{panel[fc].mean():.0%}" if fc in panel.columns else "100%"
        rows.append(row)
    return pd.DataFrame(rows)


def split_by_go(panel: pd.DataFrame, price_col: str = "go_price") -> pd.DataFrame:
    """
    按"当天有无 GO"拆开看方案表现——用于**归因**，不是可执行的分组。

    `has_go` 要等当天走完才知道，拿它做决策是拿未来信息筛样本。
    这张表的用途是看清一个规则的钱是在哪一侧赚到 / 亏掉的。
    """
    def agg(s):
        return pd.Series({
            "天数": len(s),
            "开盘vsVWAP bp": ((s["open"] / s["vwap"] - 1) * 1e4).mean(),
            "方案vsVWAP bp": ((s[price_col] / s["vwap"] - 1) * 1e4).mean(),
            "开盘→收盘 bp": ((s["close"] / s["open"] - 1) * 1e4).mean(),
        })
    return panel.groupby("has_go").apply(agg, include_groups=False)
