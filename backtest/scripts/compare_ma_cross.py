"""
MA5/MA8 金叉死叉：分品类实测台
==============================
回答一个具体问题：**「5 日线上穿 8 日线买、下穿卖」这套超短线纪律，
放在哪类标的上还能剩下正的超额？**

    python -m backtest.scripts.compare_ma_cross                 # 全部品类，全部变体
    python -m backtest.scripts.compare_ma_cross --quick         # 只跑核心 4 个变体
    python -m backtest.scripts.compare_ma_cross --offline       # 不联网，只读本地缓存
    python -m backtest.scripts.compare_ma_cross --buckets 宽基ETF 高波成长股

为什么要按品类分桶，而不是跑一个大池子取平均
--------------------------------------------
「适合哪种投资品类」这个问题，本身就要求结果**按品类分开报**。把宽基 ETF、
周期股、成长股混进一个池子取中位数，得到的数字谁也用不上：它既不是你买
沪深300ETF 时的预期，也不是你炒宁德时代时的预期。所以这里固定六个桶，
每桶 5 个标的，各报各的，并在同一张表上给出该桶的**品类特征**
（年化波动、日收益一阶自相关），让「哪类适合」这句话有可检验的依据。

三个必须同时出现在表里的对照
----------------------------
  1. **买入持有**：不给基准，任何绝对收益都没有意义。这里报的主指标是
     `年化超额 = 策略年化 − 同窗口买入持有年化`。
  2. **零成本对照**：同一组信号在佣金/印花税/滑点全为 0 时重跑一遍。
     两者之差就是成本吃掉的部分——短均线策略每年交易几十次，
     不把成本单独拆出来，就分不清「信号没用」和「信号有用但被费用吃光」。
  3. **参数邻域**：MA3/8、MA5/10、MA5/20、MA10/20、MA20/60。
     单点最优毫无意义（同 `sweep_params.py` 的判读方式）；只有邻域整体
     同号，结论才不是噪音。

成交口径
--------
  * ETF 桶用**真实场内基金**（不是指数），因此含跟踪误差与真实成交价；
    计费上 ETF **免印花税**，其余同 A 股零售，涨跌停约束照做（±10%）。
  * 个股桶走引擎默认 A 股口径：佣金万三（最低 5 元）、印花税千一（卖出）、
    滑点千一、T+1、涨跌停无法成交则顺延。
  * ETF 行情用 **qfq（前复权）**：东财源在本机时通时断，稳定可得的只有
    yfinance，而它只有前复权口径（见 `lib/market_data` 的口径说明）。
    后果是分红后整段历史价会回算，数字不像 hfq 那样长期可复现；对**比值型**
    的均线交叉信号影响很小，但换个月重跑，末位数字会有出入。
    个股桶仍是 hfq，完全可复现。
  * 港股 ETF（3175.HK）单列一节：中银香港零售口径，**每笔最低佣金 HK$100**，
    这条最低佣金会把日频策略直接判死刑，见 `lib/trend_stop.py` 的论证。

已知局限
--------
  * 每桶只有 5 个标的，桶内中位数的置信度有限，看的是**桶间差异的方向**，
    不是精确数值。
  * 区间固定 2018-01 至今，覆盖 2018 熊市、2019-21 结构牛、2022-24 调整，
    但仍是一段特定历史；换区间数字会变，方向性结论（波动越大、交易越频繁，
    成本占比越高）不依赖具体区间。
  * 策略是**二元满仓**的：金叉全仓、死叉空仓，不含分批与仓位管理。
"""

import argparse
import os
import sys
from datetime import date as _date, datetime, timedelta

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.lib.console import fmt_table, use_utf8
from backtest.lib.price_store import load_daily
from backtest.lib import costs, trend_stop
from backtest.strategies import MACrossStrategy

# ── 标的池：五个品类桶 ────────────────────────────────────────────────────────
#
# kind/adjust 决定取数通道（price_store），cost 决定计费口径。
# 指数桶用指数代理 ETF，见模块 docstring 的「成交口径」。

BUCKETS = {
    "宽基ETF": {
        "kind": "etf", "adjust": "qfq", "cost": "etf",
        "note": "低波、分散，最常被拿来做「傻瓜定投」的品种",
        "members": {"510300": "沪深300ETF", "510500": "中证500ETF",
                    "512100": "中证1000ETF", "159915": "创业板ETF",
                    "510050": "上证50ETF"},
    },
    "行业ETF": {
        "kind": "etf", "adjust": "qfq", "cost": "etf",
        "note": "单一行业，波动介于宽基与个股之间",
        "members": {"512880": "证券ETF", "512800": "银行ETF", "512010": "医药ETF",
                    "159928": "消费ETF", "512400": "有色ETF"},
    },
    "商品跨境ETF": {
        "kind": "etf", "adjust": "qfq", "cost": "etf",
        "note": "黄金/海外/商品，趋势段最长的一类（豆粕ETF 2019-09 才上市）",
        "members": {"518880": "黄金ETF", "513100": "纳指ETF", "513500": "标普500ETF",
                    "513050": "中概互联ETF", "159985": "豆粕ETF"},
    },
    "蓝筹低波股": {
        "kind": "stock", "adjust": "hfq", "cost": "ashare",
        "note": "大市值、低换手",
        "members": {"600519": "贵州茅台", "601318": "中国平安", "600036": "招商银行",
                    "600900": "长江电力", "601288": "农业银行"},
    },
    "周期资源股": {
        "kind": "stock", "adjust": "hfq", "cost": "ashare",
        "note": "商品价格驱动、趋势段长",
        "members": {"601857": "中国石油", "601088": "中国神华", "601899": "紫金矿业",
                    "603993": "洛阳钼业", "600028": "中国石化"},
    },
    "高波成长股": {
        "kind": "stock", "adjust": "hfq", "cost": "ashare",
        "note": "高波动、题材驱动，最贴近「超短线」的使用场景",
        "members": {"300750": "宁德时代", "002594": "比亚迪", "300760": "迈瑞医疗",
                    "300059": "东方财富", "002230": "科大讯飞"},
    },
}

# 港股 ETF 单列（成本模型完全不同，不能和 A 股放同一张表比）
HK_SYMBOL = "3175.HK"
HK_NAME = "三星原油期货ETF"


def custom_bucket(codes: list[str]) -> dict:
    """
    `--codes 688256:寒武纪 510300` → 一个临时「自选」桶。

    六个固定桶是为了回答「哪类品类适合」，抽样与分组都不能随手改；但「某只
    票跑出来什么样」是另一个问题，不该逼人去改源码。取数通道按代码前缀推断：
    5/1 开头是场内基金（沪 5 / 深 1），其余按个股走 hfq。
    """
    members, kinds = {}, set()
    for item in codes:
        code, _, name = item.partition(":")
        code = code.strip()
        members[code] = name.strip() or code
        kinds.add("etf" if code[:1] in ("5", "1") else "stock")
    if len(kinds) > 1:
        raise ValueError("--codes 里不要混放个股与场内基金：两者复权口径与"
                         "印花税都不同，混在一个桶里的中位数没有意义")
    kind = kinds.pop()
    return {
        "kind": kind,
        "adjust": "qfq" if kind == "etf" else "hfq",
        "cost": "etf" if kind == "etf" else "ashare",
        "note": f"--codes 指定的 {len(members)} 个标的",
        "members": members,
    }

# ── 策略变体 ──────────────────────────────────────────────────────────────────
# (标签, MACrossStrategy 参数, 是否零成本)
VARIANTS = [
    ("MA5/8",        {"fast": 5, "slow": 8}, False),
    ("MA5/8+量能",   {"fast": 5, "slow": 8, "vol_window": 5, "vol_ratio": 1.0}, False),
    ("MA5/8 零成本", {"fast": 5, "slow": 8}, True),
    ("MA3/8",        {"fast": 3, "slow": 8}, False),
    ("MA5/10",       {"fast": 5, "slow": 10}, False),
    ("MA5/20",       {"fast": 5, "slow": 20}, False),
    ("MA10/20",      {"fast": 10, "slow": 20}, False),
    ("MA20/60",      {"fast": 20, "slow": 60}, False),
]
QUICK_VARIANTS = {"MA5/8", "MA5/8+量能", "MA5/8 零成本", "MA20/60"}

DEFAULT_START = "20180102"      # 统计窗口起点（更早的数据只用于均线预热）
WARMUP_DAYS = 200               # 预热自然日：够 MA60 收敛
TRADING_DAYS = 252

# 每只标的至少要买得起这么多手，否则按窗口首日价把本金放大到够为止。
#
# 为什么必须有这一条：回测用**后复权**价，它是"从上市价滚上来的虚拟价"，
# 早就和盘面实际报价脱节——贵州茅台 hfq 末期已过 1 万元/股，一手 100 万，
# 10 万本金连一手都买不起，引擎会一次都不成交，跑出「年化 0.00%、交易 0 次」
# 的假数据混进中位数。放大本金不改变任何信号，只消除整手粒度带来的失真；
# A 股万三佣金在十万级本金上已不受最低 5 元约束，成本口径也不因此改变。
# 用**窗口首日**价标定（不是最高价），避免引入未来信息。
MIN_LOTS = 30


# ── 计费口径 ──────────────────────────────────────────────────────────────────

def cost_kwargs(profile: str, zero: bool = False) -> dict:
    """按品类给出引擎的成本与交易约束参数。"""
    if zero:
        # 零成本对照：只关掉费用，涨跌停/T+1 等成交约束保持不变，
        # 否则差额里会混进「成交可行性」的变化，就不再是纯成本了。
        base = {"commission_rate": 0.0, "min_commission": 0.0,
                "stamp_duty": 0.0, "slippage": 0.0}
    elif profile == "etf":
        # ETF 免印花税；其余同 A 股零售
        base = {"commission_rate": costs.COMMISSION_RATE,
                "min_commission": costs.MIN_COMMISSION,
                "stamp_duty": 0.0, "slippage": costs.SLIPPAGE}
    elif profile == "hk":
        # 中银香港零售：最低佣金 HK$100 是这一行唯一重要的数字
        base = {"commission_rate": 0.0025, "min_commission": 100.0,
                "stamp_duty": 0.0, "slippage": 0.002}
    else:
        base = {"commission_rate": costs.COMMISSION_RATE,
                "min_commission": costs.MIN_COMMISSION,
                "stamp_duty": costs.STAMP_DUTY, "slippage": costs.SLIPPAGE}
    # 港股无涨跌停；A 股个股与场内 ETF 有，幅度由引擎按代码前缀推断。
    # 已知不精确处：ETF 的涨跌停跟随**标的指数**，而代码前缀看不出来——
    # 159915 创业板ETF 自 2020-08 起是 ±20%，引擎按 1 开头判成 ±10%。
    # 实测该误判对本文件的结论零影响（159915 逐日复跑，10% 与 20% 两档
    # 收益/回撤/交易次数逐位相同：唯一被卡的 2024-10-09 那天并无待成交挂单）。
    # 真要精确到单只标的，只能显式传 limit_pct，不能靠前缀。
    base["limit_move_check"] = profile in ("ashare", "etf")
    return base


# ── 取数 ──────────────────────────────────────────────────────────────────────

def warmup_start(start: str) -> str:
    return (datetime.strptime(start, "%Y%m%d")
            - timedelta(days=WARMUP_DAYS)).strftime("%Y%m%d")


def load_prices(symbol: str, kind: str, adjust: str, start: str, end: str,
                offline: bool) -> pd.DataFrame:
    return load_daily(symbol, warmup_start(start), end, adjust=adjust, kind=kind,
                      auto_update=not offline, verbose=False)


# ── 单次回测 → 一行指标 ───────────────────────────────────────────────────────

def annualize(total_pct: float, n_days: int) -> float:
    """总收益（%）按交易日数折年化（%）。窗口不足 10 天返回 nan。"""
    if n_days < 10:
        return float("nan")
    return ((1 + total_pct / 100) ** (TRADING_DAYS / n_days) - 1) * 100


def buy_hold_drawdown(df: pd.DataFrame, start: str) -> float:
    """同窗口买入持有的最大回撤（%），用于和策略回撤对照。"""
    px = df.loc[df.index >= pd.to_datetime(start, format="%Y%m%d"), "close"]
    if px.empty:
        return float("nan")
    return float((px / px.cummax() - 1).min() * 100)


def sized_capital(df: pd.DataFrame, start: str, capital: float) -> float:
    """本金：默认值与「窗口首日买得起 MIN_LOTS 手」两者取大（见 MIN_LOTS 注释）。"""
    px = df.loc[df.index >= pd.to_datetime(start, format="%Y%m%d"), "open"]
    if px.empty:
        return capital
    return max(capital, float(px.iloc[0]) * costs.LOT * MIN_LOTS)


def run_one(symbol: str, df: pd.DataFrame, variant: tuple, profile: str,
            start: str, end: str, capital: float) -> dict:
    label, strat_kw, zero = variant
    cap = sized_capital(df, start, capital)
    r = run_backtest(
        symbol=symbol, start_date=warmup_start(start), end_date=end,
        strategy=MACrossStrategy(**strat_kw), initial_capital=cap,
        eval_start=start, df=df, **cost_kwargs(profile, zero),
    )
    n_days = len(r["equity_curve"])
    years = n_days / TRADING_DAYS
    ann = annualize(r["total_return"], n_days)
    bench_ann = annualize(r["benchmark_return"], n_days)
    # 年均成本率：不用引擎的 cost_drag_pct（它以**窗口起点**权益为分母，
    # 八年下来后期的手续费是按增值后的仓位付的，除以起点会低估）。
    # 这里按窗口内**平均权益**折算，量纲才和「年化收益%」可比。
    eq_mean = float(r["equity_curve"]["equity"].mean())
    cost_pct_yr = (r["costs"]["total_cost"] / eq_mean / years * 100
                   if eq_mean and years else float("nan"))
    return {
        "变体": label,
        "代码": symbol,
        "交易日": n_days,
        "本金": cap,
        # 区间原值与年化值都留着：窗口短于一年时年化是**外推**出来的
        # （半年 +10% 会被放大成年化 +21%），这时只有总收益/绝对次数可读。
        "总收益%": r["total_return"],
        "持有总收益%": r["benchmark_return"],
        "超额%": r["total_return"] - r["benchmark_return"],
        "成交次数": r["total_trades"],
        "年化%": ann,
        "持有年化%": bench_ann,
        "年化超额%": ann - bench_ann,
        "最大回撤%": r["max_drawdown"],
        "持有回撤%": buy_hold_drawdown(df, start),
        "交易次数/年": r["total_trades"] / years if years else float("nan"),
        "成本%/年": cost_pct_yr,
        "成本金额": r["costs"]["total_cost"],
        "胜率%": r["win_rate"],
        "在场%": r["exposure_pct"],
        "平均持仓日": r["avg_holding_days"],
    }


def profile_of(df: pd.DataFrame, start: str) -> dict:
    """品类特征：年化波动率与日收益一阶自相关（趋势跟踪能否成立的前提）。"""
    px = df.loc[df.index >= pd.to_datetime(start, format="%Y%m%d"), "close"]
    ret = px.pct_change().dropna()
    return {
        "年化波动%": float(ret.std() * np.sqrt(TRADING_DAYS) * 100),
        "日收益ρ1": float(ret.autocorr(1)),
    }


# ── 跑一个桶 ──────────────────────────────────────────────────────────────────

def common_eval_start(df: pd.DataFrame, start: str, variants: list) -> str:
    """
    所有变体共用的统计起点：`start` 与「最长慢线预热完毕的那天」取晚者。

    为什么不能各变体各算各的：策略 `prepare()` 会 dropna 掉均线预热期，
    MA60 吃掉的根数远多于 MA8。对**窗口起点之前就上市**的标的没有影响
    （预热在 `start` 之前就消化完了），但对 2018 年之后才上市的标的
    （688256 上市 2020-07、159985 上市 2019-09），各变体的窗口起点会不一样，
    连**买入持有基准**都跟着漂——688256 上曾出现 MA5/8 行的基准年化 40.1%、
    MA20/60 行 49.9%，差的不是策略而是起跑线。统一到最长的那根，所有变体
    与基准才落在同一段区间上。
    """
    max_slow = max(v[1].get("slow", 0) for v in variants)
    if len(df) <= max_slow:
        return start
    warm_done = _to_ymd_str(df.index[max_slow])
    return max(start, warm_done)


def _to_ymd_str(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def run_bucket(bucket: str, spec: dict, variants: list, *, start: str, end: str,
               capital: float, offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, feats = [], []
    for code, name in spec["members"].items():
        try:
            df = load_prices(code, spec["kind"], spec["adjust"], start, end, offline)
        except Exception as e:                  # noqa: BLE001 — 源异常五花八门
            print(f"    ⚠ {code} {name} 取数失败：{str(e)[:70]}")
            continue
        if len(df) < 120:
            print(f"    ⚠ {code} {name} 数据仅 {len(df)} 行，跳过")
            continue

        eval_from = common_eval_start(df, start, variants)
        feats.append({"品类": bucket, "代码": code, "名称": name,
                      **profile_of(df, eval_from)})
        for v in variants:
            row = run_one(code, df, v, spec["cost"], eval_from, end, capital)
            rows.append({"品类": bucket, "名称": name, **row})
        base = next(r for r in rows[-len(variants):] if r["变体"] == variants[0][0])
        # 一次都没成交说明本金买不起一手（后复权价过高），指标全是假的，
        # 不能混进中位数——这里显式报出来而不是让它悄悄拉低桶内数字。
        if base["成交次数"] == 0:
            print(f"    ⚠ {code} {name} 本金 ¥{base['本金']:,.0f} 未产生任何成交，"
                  f"已剔除（一手 ≈ ¥{df['close'].max() * costs.LOT:,.0f}）")
            del rows[-len(variants):]
            feats.pop()
            continue
        # 逐股这行一律报**区间原值**：窗口短于一年时年化是外推的，混着看会误读
        print(f"    {code} {name:<6} {len(df):>5} 行  本金 ¥{base['本金']:>10,.0f}  "
              f"{variants[0][0]} 区间超额 {base['超额%']:+7.2f}%  "
              f"成交 {int(base['成交次数']):>3} 次", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(feats)


# ── 汇总与打印 ────────────────────────────────────────────────────────────────

AGG_COLS = ["总收益%", "持有总收益%", "超额%", "成交次数",
            "年化%", "持有年化%", "年化超额%", "最大回撤%", "持有回撤%",
            "交易次数/年", "成本%/年", "胜率%", "在场%", "平均持仓日"]


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """桶 × 变体的中位数汇总。用中位数：5 个标的里一只翻倍就能带偏均值。"""
    g = detail.groupby(["品类", "变体"], sort=False)[AGG_COLS].median()
    return g.reset_index()


def print_variant_table(summary: pd.DataFrame, buckets: list[str],
                        metric: str) -> None:
    """一个指标一张交叉表：行=品类，列=变体。"""
    piv = summary.pivot(index="品类", columns="变体", values=metric)
    piv = piv.reindex([b for b in buckets if b in piv.index])
    piv = piv[[v for v in summary["变体"].unique() if v in piv.columns]]
    print(piv.to_string(float_format=lambda v: f"{v:8.2f}"))


def main():
    use_utf8()
    ap = argparse.ArgumentParser(description="MA5/MA8 金叉死叉分品类实测")
    ap.add_argument("--start", default=DEFAULT_START, help="统计窗口起点 YYYYMMDD")
    ap.add_argument("--end", default=_date.today().strftime("%Y%m%d"))
    ap.add_argument("--capital", type=float, default=100_000.0,
                    help="A 股/ETF 本金下限，高价股按 MIN_LOTS 手自动放大")
    ap.add_argument("--hk-capital", type=float, default=100_000.0,
                    help="港股一节的本金（港币）；最低佣金 HK$100 的影响与它直接相关")
    ap.add_argument("--buckets", nargs="+", default=list(BUCKETS),
                    help=f"只跑指定品类，可选：{list(BUCKETS)}")
    ap.add_argument("--codes", nargs="+", default=None, metavar="CODE[:名称]",
                    help="改跑自选标的（跳过六个固定桶），"
                         "如 --codes 688256:寒武纪 300308:中际旭创")
    ap.add_argument("--quick", action="store_true",
                    help="只跑核心 4 个变体（MA5/8、+量能、零成本、MA20/60）")
    ap.add_argument("--no-hk", action="store_true", help="跳过港股 ETF 一节")
    ap.add_argument("--offline", action="store_true", help="不联网，只读本地缓存")
    ap.add_argument("--output", default="output/ma_cross_bench")
    args = ap.parse_args()

    if args.codes:
        try:
            buckets = {"自选": custom_bucket(args.codes)}
        except ValueError as e:
            print(f"  ❌ {e}")
            sys.exit(1)
    else:
        unknown = [b for b in args.buckets if b not in BUCKETS]
        if unknown:
            print(f"  ❌ 未知品类：{unknown}，可选 {list(BUCKETS)}")
            sys.exit(1)
        buckets = {b: BUCKETS[b] for b in args.buckets}

    variants = [v for v in VARIANTS if not args.quick or v[0] in QUICK_VARIANTS]
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "═" * 78)
    print(f"  MA 交叉分品类实测  窗口 {args.start} → {args.end}  "
          f"本金下限 ¥{args.capital:,.0f}（高价股按首日 {MIN_LOTS} 手放大）")
    print(f"  变体 {len(variants)} 个：{'、'.join(v[0] for v in variants)}")
    print("═" * 78)

    details, features = [], []
    for b, spec in buckets.items():
        print(f"\n  【{b}】{spec['note']}")
        d, f = run_bucket(b, spec, variants, start=args.start, end=args.end,
                          capital=args.capital, offline=args.offline)
        if not d.empty:
            details.append(d)
            features.append(f)

    if not details:
        print("\n  ❌ 没有任何有效样本")
        return

    detail = pd.concat(details, ignore_index=True)
    feature = pd.concat(features, ignore_index=True)
    summary = summarize(detail)

    # 窗口不足一年时，年化是把区间收益按 252/n 外推出来的（半年 +10% → 年化 +21%），
    # 主表改报**区间原值**：短窗口本来就只能读原值，外推只会放大噪音。
    short_window = int(detail["交易日"].max()) < TRADING_DAYS
    if short_window:
        exc_col, freq_col = "超额%", "成交次数"
        exc_title = "② 区间超额%（策略 − 同窗口买入持有，桶内中位数）"
        freq_title = "③ 区间成交次数（上）与 成本拖累%/年（下，仍为年化口径）"
        print(f"\n  ⚠ 统计窗口仅 {int(detail['交易日'].max())} 个交易日（<1 年），"
              f"年化为外推值、参考意义有限，下表主指标改用**区间原值**。")
    else:
        exc_col, freq_col = "年化超额%", "交易次数/年"
        exc_title = "② 年化超额%（策略年化 − 同窗口买入持有年化，桶内中位数）"
        freq_title = "③ 交易次数/年（上）与 成本拖累%/年（下）"

    # ① 品类特征：解释「为什么这类适合 / 不适合」的前提
    print("\n" + "═" * 78)
    print("  ① 品类特征（桶内中位数）")
    print("═" * 78)
    feat_agg = feature.groupby("品类", sort=False)[["年化波动%", "日收益ρ1"]].median()
    hold = summary[summary["变体"] == variants[0][0]].set_index("品类")
    if short_window:
        feat_agg["买入持有区间%"] = hold["持有总收益%"]
    else:
        feat_agg["买入持有年化%"] = hold["持有年化%"]
    feat_agg["买入持有回撤%"] = hold["持有回撤%"]
    print(feat_agg.to_string(float_format=lambda v: f"{v:8.2f}"))
    print("\n  ρ1 = 日收益一阶自相关。趋势跟踪要靠正的自相关（涨了还涨）吃饭；"
          "\n  ρ1 ≤ 0 说明日线层面是均值回归的，短均线交叉天然逆风。")

    # ② 主表：超额
    print("\n" + "═" * 78)
    print(f"  {exc_title}")
    print("═" * 78)
    print_variant_table(summary, list(buckets), exc_col)

    # ③ 交易频率与成本
    print("\n" + "═" * 78)
    print(f"  {freq_title}")
    print("═" * 78)
    print_variant_table(summary, list(buckets), freq_col)
    print()
    print_variant_table(summary, list(buckets), "成本%/年")

    # ④ 成本 vs 信号：零成本对照
    base_lbl, zero_lbl = "MA5/8", "MA5/8 零成本"
    if {base_lbl, zero_lbl} <= set(summary["变体"].unique()):
        piv = summary.pivot(index="品类", columns="变体", values=exc_col)
        cmp = pd.DataFrame({
            "实盘成本下超额%": piv[base_lbl],
            "零成本下超额%": piv[zero_lbl],
            "成本吃掉%": piv[zero_lbl] - piv[base_lbl],
        }).reindex([b for b in buckets if b in piv.index])
        print("\n" + "═" * 78)
        print("  ④ 信号本身有没有用？（零成本对照）")
        print("═" * 78)
        print(cmp.to_string(float_format=lambda v: f"{v:8.2f}"))
        print("\n  零成本仍为负 → 信号本身无效，省手续费也救不回来；"
              "\n  零成本为正、实盘为负 → 信号有一点用，但被交易成本吃光。")

    # ⑤ 回撤与在场比例
    print("\n" + "═" * 78)
    print("  ⑤ 最大回撤%（策略 vs 买入持有）与 在场比例%")
    print("═" * 78)
    print_variant_table(summary, list(buckets), "最大回撤%")
    print()
    print_variant_table(summary, list(buckets), "在场%")

    # --codes 是临时问答，不能覆盖六个固定桶的产物（文档里的表格出自那三个文件）
    prefix = "custom_" if args.codes else ""
    d_path = os.path.join(args.output, f"{prefix}detail.csv")
    s_path = os.path.join(args.output, f"{prefix}summary.csv")
    f_path = os.path.join(args.output, f"{prefix}features.csv")
    detail.to_csv(d_path, index=False, encoding="utf-8-sig")
    summary.to_csv(s_path, index=False, encoding="utf-8-sig")
    feature.to_csv(f_path, index=False, encoding="utf-8-sig")
    print(f"\n  → {d_path}\n  → {s_path}\n  → {f_path}")

    if not args.no_hk:
        run_hk_section(variants, args)


# ── 港股 ETF：最低佣金 HK$100 的现实 ─────────────────────────────────────────

def run_hk_section(variants: list, args) -> None:
    """
    同一套规则放到港股 ETF 上，用中银香港零售口径计费。

    单列一节而不是并进主表：港股每笔最低佣金 HK$100，与 A 股的 5 元不是
    同一个量级，混在一张表里比的就不是策略了。
    """
    print("\n" + "═" * 78)
    print(f"  ⑥ 港股 ETF {HK_SYMBOL}（{HK_NAME}）— 中银香港口径，"
          f"最低佣金 HK${cost_kwargs('hk')['min_commission']:.0f}/笔")
    print("═" * 78)
    try:
        df = load_daily(HK_SYMBOL, warmup_start(args.start), args.end,
                        adjust="qfq", kind="hk",
                        auto_update=not args.offline, verbose=False)
    except Exception as e:                      # noqa: BLE001
        print(f"  ⚠ 取数失败，跳过本节：{str(e)[:80]}")
        return
    if len(df) < 120:
        print(f"  ⚠ 数据仅 {len(df)} 行，跳过本节")
        return

    # 这一节要展示的正是「小资金 + 最低佣金 HK$100」的现实，本金必须保持
    # 10 万港币量级。3175 单价十几港币，一手不过千余元，MIN_LOTS 的下限
    # 远远够不着，不会被放大——真换成高价港股时要留意这一点。
    hk_start = common_eval_start(df, args.start, variants)
    rows = [run_one(HK_SYMBOL, df, v, "hk", hk_start, args.end, args.hk_capital)
            for v in variants]
    out = pd.DataFrame(rows)[
        ["变体", "年化%", "持有年化%", "年化超额%", "最大回撤%",
         "交易次数/年", "成本%/年", "在场%"]]
    print(fmt_table(out))

    # 参照：已在 docs/hk-oil-etf-trend-stop.md 定稿的月频规则，同一段数据重跑。
    # 费率必须按 --hk-capital 算（最低佣金 HK$100 的费率完全由本金决定），
    # 窗口也必须与上表的 hk_start 一致，否则这两行参照和日频变体就不可比了。
    fee = trend_stop.hk_fee_rate(args.hk_capital)
    px = df.loc[df.index >= pd.to_datetime(hk_start, format="%Y%m%d")]
    ref = trend_stop.simulate(px, ma_len=150, stop=0.15, fee=fee, freq="month")
    bh = trend_stop.buy_hold(px, fee=fee)
    print(f"\n  参照（月频 MA150+15%移动止损，费率按 {fee:.2%}/边）："
          f"年化 {ref.stats['annual_return']:+.2%}  "
          f"回撤 {ref.stats['max_drawdown']:.1%}  "
          f"交易 {ref.stats['trades_per_year']:.1f} 次/年")
    print(f"  参照（买入持有）：年化 {bh.stats['annual_return']:+.2%}  "
          f"回撤 {bh.stats['max_drawdown']:.1%}")
    print("\n  注：日频变体的「成本%/年」是每年被手续费吃掉的本金比例。"
          "\n  10 万港币本金下每笔最低佣金 HK$100 ≈ 单边 0.1%，"
          "一年 20 个来回就是 4% 起步。")

    path = os.path.join(args.output, "hk_3175.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
