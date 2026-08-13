"""
油气双雄（中国石油 601857 / 中国海油 600938）长期跟踪脚本。

做四件事：
  1. 增量更新本地行情仓库（首次全量，之后只补新交易日）
  2. 判定每只票**当前**处在什么市场状态，给出该状态对应的打法和具体触发价
  3. （可选）跑跨周期策略回测，检查这套打法在历史上站不站得住
  4. 拉 Brent/WTI/SC 原油价格，算油价对股价的传导相关性（纯描述性统计）

用法
----
    python -m backtest.scripts.track_oil                 # 增量更新 + 输出跟踪报告
    python -m backtest.scripts.track_oil --offline       # 不联网，只用本地缓存
    python -m backtest.scripts.track_oil --backtest      # 附带连续全样本策略对比
    python -m backtest.scripts.track_oil --chart         # 出图到 output/oil/
    python -m backtest.scripts.track_oil --capital 200000
    python -m backtest.scripts.track_oil --symbols 601857 600938 600028

价格口径
--------
回测与状态判定一律用**后复权（hfq）**，含股息再投、历史不被改写；
而报告里给你看的触发价一律换算成**不复权**的盘面实际价，因为那才是
你在交易软件里挂单要输入的数字。两者用最近一日的比值换算——每次除息后
比值会变，所以除息日之后请重新跑一次本脚本再挂单。

油价数据源
----------
个股/指数走 `lib.market_data`（akshare→baostock→yfinance）；油价单独走
`lib.oil_price`（新浪财经），因为 akshare 里油价相关接口大多打 eastmoney
域名，本机 eastmoney 被墙、yfinance 商品期货限流，只有新浪这条线通。
详见该模块 docstring。新浪这条线如果哪天也不通，抓取失败会打印告警并跳过
传导分析，不影响其余报告内容。
"""

import argparse
import os

import numpy as np
import pandas as pd

from backtest.lib.ladder import (PLAYBOOK, simulate_adaptive, simulate_buy_hold,
                        simulate_dca, simulate_grid, simulate_ladder)
from backtest.lib.oil_price import (OIL_STOCKS, OIL_SYMBOLS, load_oil,
                                    transmission_table, update_oil)
from backtest.lib.oil_price import read_meta as read_oil_meta
from backtest.lib.price_store import (load_daily, load_dividends, read_meta,
                             update_daily, update_dividends)
from backtest.lib.regime import BEAR, CHOP, TREND_UP, classify, regime_episodes, regime_stats
from backtest.lib.swings import drawdown_profile, swing_table
from backtest.lib.console import use_utf8
from backtest.lib.cli import base_parser

# A 股油气股票统一取 lib/oil_price.OIL_STOCKS（商品代码 OIL_SYMBOLS 与之不同义）
DEFAULT_SYMBOLS = list(OIL_STOCKS)
NAMES = {**OIL_STOCKS, "600028": "中国石化", "000300": "沪深300"}
INDEX_SYMBOL = "000300"
HISTORY_START = "20180101"

#: 每个状态给人看的执行说明（回测口径见 lib.ladder.PLAYBOOK）
PLAYBOOK_NOTES = {
    TREND_UP: "趋势确立：快速建仓，每回调 5% 加一档共 3 档，20 个交易日没买到就补一档，不止盈、拿住",
    CHOP: "宽幅震荡：每回调 8% 加一档共 4 档，浮盈 30% 减半仓、剩下继续拿",
    BEAR: "趋势走坏：停止加仓并把仓位降到 30% 底仓，等重新站上年线再启动梯子",
}


def name_of(symbol: str) -> str:
    return NAMES.get(symbol, symbol)


# ── 数据更新 ──────────────────────────────────────────────────────────────────

def refresh(symbols, offline: bool) -> None:
    if offline:
        print("── 离线模式：只读本地缓存 ──")
        for s in symbols + [INDEX_SYMBOL]:
            m = read_meta(s, "hfq" if s != INDEX_SYMBOL else "none")
            if m:
                print(f"  {name_of(s):<8} {m.get('data_start')}~{m.get('data_end')} "
                      f"{m.get('rows')} 行，最后更新 {m.get('updated_at')}")
        for o in OIL_SYMBOLS:
            m = read_oil_meta(o)
            if m:
                print(f"  {o:<8} {m.get('data_start')}~{m.get('data_end')} "
                      f"{m.get('rows')} 行，最后更新 {m.get('updated_at')}")
        return

    print("── 增量更新本地行情仓库 ──")
    for s in symbols:
        for adj in ("hfq", "none"):
            update_daily(s, HISTORY_START, adjust=adj)
        update_dividends(s)
    update_daily(INDEX_SYMBOL, HISTORY_START, kind="index", adjust="none")

    print("── 更新原油价格（新浪源，油价→股价传导分析用）──")
    for o in OIL_SYMBOLS:
        try:
            update_oil(o)
        except Exception as e:                      # noqa: BLE001 — 数据源异常五花八门
            print(f"  ⚠ {o} 更新失败（{e}），传导分析会跳过该品种")


def load_all(symbols, offline: bool) -> dict:
    out = {}
    for s in symbols:
        out[s] = {
            "hfq": load_daily(s, HISTORY_START, adjust="hfq",
                              auto_update=False, verbose=False),
            "raw": load_daily(s, HISTORY_START, adjust="none",
                              auto_update=False, verbose=False),
            "div": load_dividends(s, auto_update=False),
        }
    return out


def load_all_oil() -> dict:
    """读本地原油缓存（refresh() 已经拉过一遍，这里只读，缺文件的品种跳过）。"""
    out = {}
    for o in OIL_SYMBOLS:
        try:
            out[o] = load_oil(o, auto_update=False, verbose=False)
        except FileNotFoundError:
            pass
    return out


# ── 报告：当前状态与打法 ──────────────────────────────────────────────────────

def ttm_dividend(div: pd.DataFrame, asof: pd.Timestamp) -> float:
    """过去 12 个月已除息的每股现金分红合计。"""
    if div.empty:
        return 0.0
    win = div[(div["ex_date"] <= asof) & (div["ex_date"] > asof - pd.Timedelta(days=365))]
    return float(win["cash_before_tax"].sum())


def report_symbol(symbol: str, d: dict, capital: float) -> dict:
    hfq, raw, div = d["hfq"], d["raw"], d["div"]
    reg = classify(hfq)
    last = hfq.index[-1]
    px = float(raw["close"].iloc[-1])
    # hfq → 盘面实际价的换算比例（最近一日实测，除息后会变）
    ratio = px / float(hfq["close"].iloc[-1])

    state = reg["regime"].iloc[-1]
    ma250 = float(reg["ma"].iloc[-1]) * ratio if not np.isnan(reg["ma"].iloc[-1]) else float("nan")
    dd250 = float(reg["dd"].iloc[-1])
    vol = float(reg["vol"].iloc[-1])
    ttm = ttm_dividend(div, last)

    print(f"\n{'='*88}")
    print(f"{name_of(symbol)}  {symbol}    数据截至 {last.date()}")
    print(f"{'='*88}")

    c = raw["close"]
    print(f"  现价(不复权) {px:.2f}   "
          f"近1年 {c.iloc[-250:].min():.2f}~{c.iloc[-250:].max():.2f}"
          f"（分位 {(c.iloc[-250:] <= px).mean():.0%}）   "
          f"近2年 {c.iloc[-500:].min():.2f}~{c.iloc[-500:].max():.2f}"
          f"（分位 {(c.iloc[-500:] <= px).mean():.0%}）")
    mas = "  ".join(f"MA{n}={c.rolling(n).mean().iloc[-1]:.2f}({px/c.rolling(n).mean().iloc[-1]-1:+.1%})"
                    for n in (20, 60, 120, 250))
    print(f"  {mas}")
    print(f"  距 250 日高点 {dd250:.1%}   60 日年化波动 {vol:.1%}   "
          f"TTM 每股分红 {ttm:.4f} 元 → 股息率 {ttm/px:.2%}")

    print(f"\n  ▶ 当前状态：【{state}】")
    print(f"    {PLAYBOOK_NOTES[state]}")
    if not np.isnan(ma250):
        print(f"    年线(MA250) {ma250:.2f}，现价{'在其上方' if px > ma250 else '已跌破'} "
              f"{abs(px/ma250-1):.1%}；MA250 斜率 {reg['slope'].iloc[-1]:+.2%}（20日）")

    # ── 具体触发价 ──
    p = PLAYBOOK[state]
    per = capital / p["n_tranches"]
    ref_hfq = max(float(hfq["close"].rolling(120, min_periods=1).max().iloc[-1]),
                  float(hfq["close"].iloc[-1]))
    ref = ref_hfq * ratio
    print(f"\n  ▶ 建仓阶梯（参考高点 {ref:.2f} = 近 120 日最高，每档 {per:,.0f} 元）")
    print(f"    {'档':<4}{'触发价':>9}{'较现价':>9}{'较参考高点':>11}   {'金额':>10}")
    for k in range(p["n_tranches"]):
        trig = ref * (1 - p["step"] * k)
        flag = " ← 现价已触发" if px <= trig else ""
        print(f"    {k+1:<4}{trig:>9.2f}{px/trig-1:>+9.1%}{-p['step']*k:>11.0%}   "
              f"{per:>10,.0f}{flag}")
    fired = sum(1 for k in range(p["n_tranches"])
                if px <= ref * (1 - p["step"] * k))
    if fired:
        print(f"    → 按规则，从参考高点跌到现价已累计触发 {fired}/{p['n_tranches']} 档"
              f"（回测里这些档是在下跌途中分批买到的，不是今天一次买满）")
    if p["max_pos"] <= 0.999:
        print(f"    仓位上限 {p['max_pos']:.0%}")
    if p["take_profit"]:
        print(f"    止盈：持仓浮盈 ≥ {p['take_profit']:.0%} 时卖出 {p['tp_fraction']:.0%}")
    else:
        print("    止盈：本状态不止盈（趋势里拿住）")
    print(f"    离场保险丝：收盘跌破年线且年线转头向下 → 状态切为「{BEAR}」，降至 30% 底仓")

    # ── 从今天开始铺的梯子 ──
    #
    # 上面那张表以"近 120 日高点"为锚，是给已经在跟踪的持仓用的。
    # 如果是今天新开一笔仓，价格早已跌穿多数档位，照抄那张表等于一次买满，
    # 分批的意义就没了。所以再以今日收盘为锚重铺一遍。
    print(f"\n  ▶ 今日新开仓阶梯（以现价 {px:.2f} 为锚重铺，每档 {per:,.0f} 元）")
    print(f"    {'档':<4}{'挂单价':>9}{'较现价':>9}   {'金额':>10}   说明")
    for k in range(p["n_tranches"]):
        trig = px * (1 - p["step"] * k)
        note = "立即建底仓" if k == 0 else f"再跌 {p['step']*k:.0%} 加一档"
        print(f"    {k+1:<4}{trig:>9.2f}{-p['step']*k:>+9.0%}   {per:>10,.0f}   {note}")
    print(f"    若一路上涨没等到回调：参考高点会随新高上移，按上表的档位追；"
          f"或每 {60} 个交易日强制补一档，避免长期空仓")

    # ── 股息率参照 ──
    if ttm > 0:
        print(f"\n  ▶ 股息率参照价（按 TTM 分红 {ttm:.4f} 元推算，仅作估值锚，未回测）")
        print("    " + "   ".join(f"{y:.1%}→{ttm/y:.2f}" for y in (0.04, 0.045, 0.05, 0.055, 0.06)))

    return {"symbol": symbol, "state": state, "price": px, "ratio": ratio,
            "reg": reg, "ttm": ttm, "dd250": dd250}


def report_structure(symbol: str, hfq: pd.DataFrame) -> None:
    """波段与回撤剖面：解释梯子为什么铺这么宽。"""
    print(f"\n  ▶ 波段结构（决定档距）")
    for lab, sub in [("近两年", hfq.iloc[-484:]), ("全样本", hfq)]:
        t = swing_table(sub["close"], 0.08)
        up, dn = t[t.pct > 0], t[t.pct < 0]
        if len(up) == 0 or len(dn) == 0:
            continue
        print(f"    {lab:<5} 上涨段 {len(up):>2} 个 中位 {up.pct.median():+.0%}/{up.days.median():.0f}天"
              f"   下跌段 {len(dn):>2} 个 中位 {dn.pct.median():+.0%}/{dn.days.median():.0f}天"
              f"   最深 {dn.pct.min():+.0%}")
    prof = drawdown_profile(hfq["close"], (0.08, 0.15, 0.25))
    print("    独立回撤事件（全样本）：" + "  ".join(
        f"跌≥{r.threshold:.0%} {r.episodes} 次"
        + (f"，中位 {r.median_recover_days:.0f} 天修复" if r.recovered else "，未修复")
        for r in prof.itertuples()))


# ── 油价 → 股价传导性 ─────────────────────────────────────────────────────────

def report_oil_transmission(symbols, data, oil_data: dict) -> None:
    """打印油价 vs 各票的领先滞后相关系数表。纯描述性统计，不是回测。"""
    if not oil_data:
        print("\n（原油价格全部品种抓取失败，跳过油价→股价传导分析）")
        return

    print(f"\n{'='*88}")
    print("油价 → 股价传导性（描述性相关系数，不是回测信号）")
    print(f"{'='*88}")
    print("  lag_days = 油价收益率领先股价收益率的（股票）交易日数，corr 为皮尔逊相关系数；")
    print("  ci95 = r=0 假设下的噪声阈 ±1.96/√n，signif=False 的行按「与 0 无异」读；")
    print("  口径与两市场日历对齐方式见 lib.oil_price.transmission_table 的 docstring")

    # 缓存过期会让对齐后的样本被静默剔除，先把新鲜度亮出来
    for o, oil in oil_data.items():
        lag_days = (pd.Timestamp.today().normalize() - oil.index[-1]).days
        if lag_days > 7:
            print(f"  ⚠ {o} 本地缓存最后一根 K 线是 {oil.index[-1].date()}"
                  f"（已过去 {lag_days} 天），超出对齐容差的交易日会被剔除，n 会偏小")

    for s in symbols:
        stock = data[s]["hfq"]
        for o, oil in oil_data.items():
            t = transmission_table(oil, stock)
            if t.empty:
                continue
            t = t.assign(corr=t["corr"].round(3), ci95=t["ci95"].round(3))
            print(f"\n  {name_of(s)}({s}) vs {o}：")
            print("    " + t.to_string(index=False).replace("\n", "\n    "))
            hits = t[t["signif"]]
            if hits.empty:
                print("    → 无一 lag 超过噪声阈，该品种对这只票没有可辨识的传导")
            else:
                best = hits.loc[hits["corr"].abs().idxmax()]
                print(f"    → 最强传导在 lag={int(best['lag_days'])} 天"
                      f"（corr {best['corr']:+.3f}，阈值 ±{best['ci95']:.3f}）")


# ── 回测 ──────────────────────────────────────────────────────────────────────

def run_backtests(symbols, data, capital: float) -> dict:
    """连续全样本对比。不切段、不中途重置资金——这才是长期持有的真实体感。"""
    results = {}
    for s in symbols:
        hfq = data[s]["hfq"]
        reg = classify(hfq)["regime"]
        res = [
            simulate_buy_hold(hfq, capital),
            simulate_dca(hfq, capital, n_tranches=10, every_days=21),
            simulate_ladder(hfq, capital, n_tranches=4, step=0.08, name="固定梯度4×8 长持"),
            simulate_ladder(hfq, capital, n_tranches=4, step=0.08, take_profit=0.30,
                            tp_fraction=0.5, name="固定梯度 半止盈30"),
            simulate_ladder(hfq, capital, n_tranches=4, step=0.08, take_profit=0.30,
                            tp_fraction=0.5, ma_exit=250, name="固定梯度 半止盈+MA250"),
            simulate_grid(hfq, capital, base_position=0.5, n_grids=5, grid_step=0.07),
            simulate_adaptive(hfq, reg, capital),
        ]
        results[s] = res
        print(f"\n{'='*104}")
        print(f"【连续全样本回测】{name_of(s)} {s}  "
              f"{hfq.index[0].date()}~{hfq.index[-1].date()}（{len(hfq)} 个交易日，本金 {capital:,.0f}）")
        print(f"{'='*104}")
        print(pd.DataFrame([{
            "策略": r.name,
            "总收益": f"{r.stats['total_return']:.1%}",
            "年化": f"{r.stats['annual_return']:.1%}",
            "最大回撤": f"{r.stats['max_drawdown']:.1%}",
            "平均仓位": f"{r.stats['avg_exposure']:.0%}",
            "投入资金收益": f"{r.stats['deployed_return']:.1%}",
            "夏普": f"{r.stats['sharpe']:.2f}",
            "笔数": r.stats["n_trades"],
        } for r in res]).to_string(index=False))

        st = regime_stats(hfq, classify(hfq), horizon=60)
        if not st.empty:
            f = st.copy()
            for col in ["未来均值", "未来中位", "胜率", "最差", "最好"]:
                f[col] = f[col].map("{:.1%}".format)
            print(f"\n  状态分类器的信息量（未来 60 日收益，只用当日可得数据判定）：")
            print("  " + f.to_string().replace("\n", "\n  "))
    return results


# ── 绘图 ──────────────────────────────────────────────────────────────────────

REGIME_SHADE = {TREND_UP: "#2ea043", CHOP: "#8b949e", BEAR: "#f85149"}


def plot_symbol(symbol: str, data: dict, info: dict, results, save_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from backtest.reports.plotting import COLORS, setup_matplotlib, style_ax

    setup_matplotlib()
    hfq, raw = data["hfq"], data["raw"]
    reg = info["reg"]
    ratio = info["ratio"]

    n = 3 if results else 2
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True,
                             gridspec_kw={"height_ratios": [3] + [2] * (n - 1)})
    fig.patch.set_facecolor(COLORS["bg"])

    # ① 价格 + 年线 + 状态底色
    ax = axes[0]
    style_ax(ax)
    # 画的是后复权序列按今日盘面价缩放后的曲线：终点等于今天的实际股价，
    # 但历史点位是"含分红再投"的口径，并不等于当年盘面上的挂牌价。
    # 状态判定和回测都用这个口径，图上保持一致才对得上。
    price = hfq["close"] * ratio
    ax.plot(price.index, price, color=COLORS["fg"], lw=1.0,
            label="后复权价（含分红再投，缩放至今日盘面价）")
    ax.plot(reg.index, reg["ma"] * ratio, color=COLORS["gold"], lw=1.0, label="年线 MA250")
    for r in regime_episodes(reg).itertuples():
        ax.axvspan(r.start, r.end, color=REGIME_SHADE[r.regime], alpha=0.10, lw=0)
    ax.set_title(f"{name_of(symbol)} {symbol}　当前状态：{info['state']}　"
                 f"现价 {info['price']:.2f}　距250日高 {info['dd250']:.1%}",
                 color=COLORS["fg"], fontsize=12)
    ax.set_ylabel("价格（元）")
    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["muted"],
              labelcolor=COLORS["fg"], fontsize=8, loc="upper left")

    # ② 距 250 日高点的回撤
    ax = axes[1]
    style_ax(ax)
    ax.fill_between(reg.index, reg["dd"] * 100, 0, color=COLORS["red"], alpha=0.35, lw=0)
    for lvl in (-8, -15, -25):
        ax.axhline(lvl, color=COLORS["muted"], lw=0.5, ls="--")
        ax.text(reg.index[0], lvl, f" {lvl}%", color=COLORS["muted"], fontsize=7, va="bottom")
    ax.set_ylabel("距250日高点 (%)")

    # ③ 策略净值
    if results:
        ax = axes[2]
        style_ax(ax)
        palette = [COLORS["blue"], COLORS["green"], COLORS["gold"],
                   COLORS["red"], COLORS["fg"], COLORS["muted"], "#bc8cff"]
        for r, col in zip(results, palette):
            ax.plot(r.equity.index, r.equity / r.equity.iloc[0],
                    color=col, lw=1.2, label=r.name)
        ax.set_ylabel("策略净值（起点=1）")
        ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["muted"],
                  labelcolor=COLORS["fg"], fontsize=7, ncol=2, loc="upper left")

    axes[-1].tick_params(axis="x", colors=COLORS["fg"])
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{symbol}_{name_of(symbol)}_track.png")
    fig.savefig(path, dpi=130, facecolor=COLORS["bg"])
    plt.close(fig)
    return path


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    use_utf8()
    ap = argparse.ArgumentParser(description="油气双雄长期跟踪",
                                 parents=[base_parser()])
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--backtest", action="store_true", help="附带连续全样本策略对比")
    ap.add_argument("--chart", action="store_true", help="输出图表")
    ap.set_defaults(capital=100_000, output="output/oil")
    args = ap.parse_args()

    refresh(args.symbols, args.offline)
    data = load_all(args.symbols, args.offline)
    oil_data = load_all_oil()

    infos = {}
    for s in args.symbols:
        infos[s] = report_symbol(s, data[s], args.capital)
        report_structure(s, data[s]["hfq"])

    # 两只票的同质性：合计敞口才是真实风险
    if len(args.symbols) >= 2:
        rets = pd.DataFrame({s: data[s]["hfq"]["close"].pct_change()
                             for s in args.symbols}).dropna()
        if len(rets) > 60:
            recent = rets.iloc[-484:]
            print(f"\n{'='*88}")
            print("组合提示")
            print(f"{'='*88}")
            print("  近两年日收益相关系数：")
            print("  " + recent.corr().round(2).to_string().replace("\n", "\n  "))
            eq = recent.mean(axis=1)
            single = recent.std().mean() * np.sqrt(252)
            print(f"  等权组合年化波动 {eq.std()*np.sqrt(252):.1%}　"
                  f"单票平均年化波动 {single:.1%}　"
                  f"分散化效果 {eq.std()*np.sqrt(252)/single-1:+.1%}")

    report_oil_transmission(args.symbols, data, oil_data)

    results = run_backtests(args.symbols, data, args.capital) if args.backtest else {}

    if args.chart:
        print()
        for s in args.symbols:
            print("  图表已保存：" + plot_symbol(s, data[s], infos[s],
                                                 results.get(s), args.output))


if __name__ == "__main__":
    main()
