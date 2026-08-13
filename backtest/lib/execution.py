"""
日内下单方案的成交价测算（与标的、与策略无关的度量层）。

回答的问题只有一个：**已经决定要买（卖）了，用哪种下单方式成交价更好。**
这跟"买不买""买多少"无关——那些是策略的事，在 backtest/strategies/ 和 lib/ladder.py 里。

基准取当日 VWAP（`sum(amount) / sum(volume)`，全天成交的加权均价）。
为什么是 VWAP 而不是当日最低价：最低价只有事后才知道、谁也挂不到，
拿它当标准会让所有方案都是"失败"，比不出高下。VWAP 是**可达成的中性结果**
（把单子摊到全天慢慢做就大致能拿到），因此"跑赢 VWAP"才是有意义的说法。

买卖两侧共用这一套基准，只有方向相反
------------------------------------
原始偏离 `bp = (成交价 / VWAP − 1) × 1e4` 是**中性的日内形状描述**，
与你是买是卖无关：某天开盘价低于 VWAP 18bp，就是低 18bp。
好坏才分侧——买入低于基准是省钱，卖出低于基准是亏钱。所以：

    优势bp = −原始bp （买）   |   +原始bp （卖）      正 = 优于 VWAP

**直接推论：固定时点方案（开盘 / 尾盘）的卖出结论可以从买入表直接翻符号得到，
不需要重测。**开盘价相对 VWAP 便宜多少，买入就省多少、卖出就亏多少。
真正需要另测的只有两类**依赖侧向定义**的方案：
  * GO 窗口——买侧看红柱拉长，卖侧看红柱缩短/死叉，是两个不同的条件
  * 限价单——买侧挂低价等回调（`rest_low` 触发），卖侧挂高价等冲高（`rest_high` 触发），
    连"成交与否"的判定都不一样

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
   `add_limit_plan()` 强制要求 `fallback`，没得选。卖侧更要命：
   "没卖掉就继续拿着"是把一次择时失败变成了一个未平仓头寸，代价不在这张表里。

4. **这张表是「所有交易日」的无条件形状，不是「你真的会下单那些天」的形状。**
   买侧影响不大：结论是"开盘就买"，无条件规则用无条件样本刚好对得上。
   **卖侧要当心**——两个固定时点的差额（尾盘 − 开盘）本质上就是全样本的
   开盘→收盘平均漂移。若这段漂移为正，表会告诉你"拖到尾盘再卖更划算"，
   可你之所以要卖，往往正是因为动能已经衰减，那天未必还随大流上漂。
   要判断"再等等"值不值，看 `split_by_go()` 的 `方案→收盘 bp`：
   它比较的两个动作在同一时刻都可选，是因果的（见该函数 docstring）。

用法
----
    from backtest.lib.execution import intraday_macd, daily_panel, add_limit_plan, benchmark

    bars = pd.read_csv(...)                     # 30min 不复权 K 线
    bars = intraday_macd(bars)                  # 追加 DIF/DEA/hist/go_buy/go_sell

    panel = daily_panel(bars)                            # 买侧
    panel = add_limit_plan(panel, offset=-0.005, fallback="close")
    print(benchmark(panel, ["open", "close", "go_price", "limit_-50bp"]))

    panel = daily_panel(bars, side="sell")               # 卖侧
    panel = add_limit_plan(panel, offset=+0.005, fallback="close", side="sell")
    print(benchmark(panel, ["open", "close", "go_price", "limit_+50bp"], side="sell"))
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

REQUIRED = ["dt", "date", "open", "high", "low", "close", "volume", "amount"]

# close/vwap 的中位比值偏离 1 超过这个幅度，判定为复权口径不一致
_BASIS_TOL = 0.02

SIDES = ("buy", "sell")
# 每侧的 GO 列与限价单的成交判定参考列
_GO_COL = {"buy": "go_buy", "sell": "go_sell"}
_LIMIT_REF = {"buy": "rest_low", "sell": "rest_high"}


def _check_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side 必须是 {SIDES} 之一，收到 {side!r}")
    return side


def intraday_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                  signal: int = 9) -> pd.DataFrame:
    """
    在**连续跨日**的分时序列上算 MACD，并标出买/卖两侧的 GO 柱。

    与 `backtest_jcy_intraday.add_macd` / `classify_timing()` 同式：
        go_buy  = 红柱为正且正在拉长 且 DIF > DEA          （动能起步，进场）
        go_sell = 红柱为正但正在缩短 或 DIF 下穿 DEA        （动能衰减，离场）

    两侧**不是互补关系**：go_buy 要求方向与动能同时成立，条件严；
    go_sell 只要动能转弱就成立，且死叉那一路完全不看红柱正负，条件宽得多。
    所以卖侧的 has_go 天数占比天然远高于买侧，两边的 bp 不能横向比较。

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
    out["hist_shrinking"] = (hist > 0) & (hist < hist.shift(1))
    death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    out["go_buy"] = out["hist_expanding"] & (dif > dea)
    out["go_sell"] = out["hist_shrinking"] | death_cross
    return out


def daily_panel(df: pd.DataFrame, *, side: str = "buy", go_col: str | None = None,
                warmup_bars: int = 60) -> pd.DataFrame:
    """
    把分时 K 线压成每日一行，产出各下单方案的**可成交价**。

    `side` 只决定用哪个 GO 列（buy→`go_buy`，sell→`go_sell`）；其余各列
    是中性的日内形状，两侧共用。`go_col` 可显式覆盖。

    列：
      vwap        当日成交量加权均价（基准）
      open/close  当日开盘价（= 集合竞价成交价）/ 收盘价
      day_low     当日最低价        rest_low   首根之后的全天最低价
      day_high    当日最高价        rest_high  首根之后的全天最高价
                  （`rest_*` 用于判断限价单能否成交：买侧看 low，卖侧看 high）
      has_go      当天是否出现过 GO 柱（**事后信息**，只可用于归因，不可用于决策）
      go_time     首个 GO 柱的时刻
      go_price    按首个 GO 柱的**下一根开盘价**计的可成交价；
                  全天无 GO、或 GO 出现在最后一根 → 退化为收盘价

    无 GO 兜底成收盘价，买卖两侧同理：规则是"等 GO"，等不到就只能最后一刻做掉。
    卖侧尤其不许兜底成"不卖"——那是把择时失败变成留仓，代价落在这张表之外。

    `warmup_bars` 丢弃开头若干根（MACD 预热），默认 60（30min 下约 7.5 个交易日）。
    """
    _check_side(side)
    go_col = go_col or _GO_COL[side]
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
            "day_high": g["high"].max(), "rest_high": g["high"].iloc[1:].max(),
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
                   ref: str = "open", side: str = "buy") -> pd.DataFrame:
    """
    追加一个"挂限价单等一个更好的价"的方案，成交价写进新列。

    offset   相对 `ref` 的偏移。买侧用负数（挂低价等回调，-0.005 = 开盘价下方 0.5%），
             卖侧用正数（挂高价等冲高）。方向与 side 不符直接报错——
             挂在错误一侧会立刻成交，测的就不是"等价格"这件事了。
    fallback 全天未触及限价时的兜底成交列（**必填，不许省**）。
             允许"没成交就不做"等于给策略白送一个择时期权，回测会凭空变好。

    成交判定用首根之后的极值（买 `rest_low` / 卖 `rest_high`）：开盘那一刻
    你还没挂上单，用 day_low/day_high 会把开盘瞬间的影线也算成成交，高估成交率。

    ⚠️ 两侧的逆向选择方向相反，但**都是亏**：买侧成交在没涨的日子、被迫追高在
    涨的日子；卖侧成交在冲高的日子、被迫砸盘在跌的日子。成交的样本看着都很漂亮，
    未成交的那部分才是账单——所以只看 `_filled` 那些行的均价一定会得出错误结论。
    """
    side = _check_side(side)
    if fallback not in panel.columns:
        raise ValueError(f"fallback 列 {fallback!r} 不存在；必须给一个强制成交的兜底价")
    if side == "buy" and offset > 0:
        raise ValueError(f"买侧限价单的 offset 应为负（挂低价等回调），收到 {offset:+}")
    if side == "sell" and offset < 0:
        raise ValueError(f"卖侧限价单的 offset 应为正（挂高价等冲高），收到 {offset:+}")

    col = name or f"limit_{int(round(offset * 1e4)):+d}bp"
    limit = panel[ref] * (1 + offset)
    ref_col = _LIMIT_REF[side]
    filled = (panel[ref_col] <= limit) if side == "buy" else (panel[ref_col] >= limit)
    out = panel.copy()
    out[col] = np.where(filled, limit, panel[fallback])
    out[col + "_filled"] = filled
    return out


def benchmark(panel: pd.DataFrame, cols: list[str],
              labels: dict[str, str] | None = None,
              side: str = "buy") -> pd.DataFrame:
    """
    各方案成交价相对当日 VWAP 的偏离（bp），带 t 值与显著性。

    `均值bp` / `中位bp` 是**中性的原始偏离**（负 = 成交价低于 VWAP），与买卖无关；
    `优势bp` 才分侧：正 = 优于 VWAP（买得更便宜 / 卖得更贵），买侧即原始偏离取反。

    `signif` 是 |t| > 1.96（均值为 0 的双尾检验）。同时给中位数：
    均值容易被少数暴涨暴跌日主导，两者同号才算稳。
    """
    side = _check_side(side)
    sign = -1.0 if side == "buy" else 1.0
    labels = labels or {}
    rows = []
    for c in cols:
        s = ((panel[c] / panel["vwap"] - 1) * 1e4).dropna()
        if len(s) < 30:
            continue
        se = s.std(ddof=1) / np.sqrt(len(s))
        t = 0.0 if se == 0 else s.mean() / se       # 恒等于基准时方差为 0
        row = {"方案": labels.get(c, c), "n": len(s), "均值bp": s.mean(),
               "中位bp": s.median(), "优势bp": sign * s.mean(),
               "t值": t, "signif": abs(t) > 1.96}
        fc = c + "_filled"
        row["成交率"] = f"{panel[fc].mean():.0%}" if fc in panel.columns else "100%"
        rows.append(row)
    return pd.DataFrame(rows)


def wait_value(panel: pd.DataFrame, price_col: str = "go_price",
               side: str = "buy") -> pd.DataFrame:
    """
    「按方案价成交」之后**继续等到收盘**值多少 bp——卖出侧最该看的一张表。

    只统计方案价真的不等于收盘价的那些行（无 GO 的日子兜底就是收盘价，
    留着会掺进一堆恒等于 0 的样本，把均值和 t 值一起稀释掉）。

    这是**因果可执行**的比较，与 `split_by_go` 的分组不同：GO 柱一出现，
    "立刻做掉"和"放着等收盘"两个动作在那一刻都摆在你面前，不需要预知任何未来。
    正 = 等一等更划算（卖侧卖得更贵 / 买侧买完还在涨）。
    """
    side = _check_side(side)
    d = panel[panel[price_col] != panel["close"]]
    s = ((d["close"] / d[price_col] - 1) * 1e4).dropna()
    if len(s) < 30:
        return pd.DataFrame()
    se = s.std(ddof=1) / np.sqrt(len(s))
    t = 0.0 if se == 0 else s.mean() / se
    note = "等到收盘更贵→GO 卖早了" if side == "sell" else "买完还在涨→GO 买对了"
    return pd.DataFrame([{
        "比较": f"{price_col} 成交后等到收盘",
        "n": len(s), "均值bp": s.mean(), "中位bp": s.median(),
        "t值": t, "signif": abs(t) > 1.96, "正的含义": note,
    }])


def split_by_go(panel: pd.DataFrame, price_col: str = "go_price",
                side: str = "buy") -> pd.DataFrame:
    """
    按"当天有无 GO"拆开看方案表现——用于**归因**，不是可执行的分组。

    `has_go` 要等当天走完才知道，拿它做决策是拿未来信息筛样本。
    这张表的用途是看清一个规则的钱是在哪一侧赚到 / 亏掉的：
    `方案优势bp` 已按 side 转成"正 = 好"，`开盘→收盘 bp` 保持中性（是当天的走势）。

    `方案→收盘 bp`（= 成交后价格还走了多少）是**唯一可以照着做决策**的一列。
    它比较的两个动作在同一时刻都是可选的：GO 柱一出现，你既可以立刻做掉，
    也可以放着等收盘——不需要预知当天有没有 GO。因此在 has_go=True 那一行里，
    它直接回答"看到 GO 之后继续等，值不值"：
        卖侧为正 = 等到收盘卖得更贵，GO 卖早了
        买侧为正 = 买完还在涨，GO 买对了
    """
    side = _check_side(side)
    sign = -1.0 if side == "buy" else 1.0

    def agg(s):
        return pd.Series({
            "天数": len(s),
            "开盘vsVWAP bp": ((s["open"] / s["vwap"] - 1) * 1e4).mean(),
            "方案vsVWAP bp": ((s[price_col] / s["vwap"] - 1) * 1e4).mean(),
            "方案优势bp": sign * ((s[price_col] / s["vwap"] - 1) * 1e4).mean(),
            "方案→收盘 bp": ((s["close"] / s[price_col] - 1) * 1e4).mean(),
            "开盘→收盘 bp": ((s["close"] / s["open"] - 1) * 1e4).mean(),
        })
    return panel.groupby("has_go").apply(agg, include_groups=False)


# ── 分时择时的展示层辅助（供 backtest_jcy_intraday 使用）───────────────────────
#
# `intraday_macd` 在跨日序列上算好 `go_buy`/`go_sell`（唯一真值源）。下面的
# 打标签 / 汇总 / 计价只读这些列，不再重复推导条件——从
# `backtest_jcy_intraday` 的 `add_macd`/`classify_timing` 合并过来后，
# GO 定义只剩 `intraday_macd` 一处。

@dataclass
class TimingSummary:
    """单个执行日的分时择时汇总。"""
    has_go: bool
    go_times: list = field(default_factory=list)
    first_go: pd.Timestamp | None = None
    second_go: pd.Timestamp | None = None
    go_count: int = 0
    total_bars: int = 0


def classify_timing(day_slice: pd.DataFrame, action: str) -> pd.DataFrame:
    """
    对执行日的每根分时 K 线打时机标签：GO / WAIT / AVOID。

    action: 'buy' or 'sell'

    买入判断：
      GO    → `go_buy`（DIF > DEA 且红柱正在拉长，与日线信号完全共振）
      WAIT  → DIF > DEA 但红柱未拉长（方向对，等动能起步）
      AVOID → DIF <= DEA（方向相反，开盘混沌期或回调中）

    卖出判断：
      GO    → `go_sell`（红柱开始缩短，或 DIF 下穿 DEA）
      WAIT  → 动能仍在高位（暂时持有）

    两侧**不对称**：买侧要求方向与动能同时成立，还留了 AVOID 一档；
    卖侧只要动能转弱就放行，死叉那一路根本不看红柱正负，也没有 AVOID。
    所以卖侧的 GO 天数占比天然高得多，两边的 bp 不可横向比较。

    前置条件：入参必须来自 `intraday_macd` 的输出（含 `go_buy`/`go_sell`
    与 DIF/DEA 列），且是**单个交易日**的切片。
    """
    df = day_slice.copy()
    if action == "buy":
        conditions = [
            df["go_buy"],
            df["DIF"] > df["DEA"],
        ]
        df["timing"] = np.select(conditions, ["GO", "WAIT"], default="AVOID")
    else:
        df["timing"] = np.select([df["go_sell"]], ["GO"], default="WAIT")
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


def executable_price(exec_bars: pd.DataFrame, summary: TimingSummary) -> float:
    """
    把择时结论换算成**可成交价**（口径见 `daily_panel.go_price`）。

    有 GO → 首个 GO 柱的**下一根 K 线开盘价**；GO 出现在当日最后一根、
    或全天无 GO → 当日最后一柱收盘价。

    为什么是"下一根的开盘价"而不是"GO 柱的收盘价"：要等这根 K 线走完才能判定
    它是不是 GO，那一刻它的收盘价已经成为历史，挂不进去。你看到 GO 之后能下的
    第一个单，成交在下一根的开盘。与 `daily_panel()` 的 `go_price` 同口径，
    两处若要改动请一起改；`tests/test_intraday_exec_price.py` 守住这条一致性。
    """
    last_close = float(exec_bars["close"].iloc[-1])
    if not (summary.has_go and summary.first_go is not None):
        return last_close
    pos = exec_bars.index.get_loc(summary.first_go)
    if pos + 1 >= len(exec_bars):
        return last_close                      # GO 在最后一根，身后没有可成交的 K 线
    return float(exec_bars["open"].iloc[pos + 1])
