# 长期跟踪与分批建仓 (`backtest/scripts/track_oil.py` + `lib/price_store|swings|regime|ladder`)

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。也见 [[oil-majors-tracking]] memory（投资论点）与 [[backtest-continuous-vs-segmented]] memory（连续 vs 分段回测教训）。

**职责：** 对固定的几只票做长期迭代跟踪：本地存行情、判定当前市场状态、给出该状态对应的打法与具体挂单价。

- **`lib/price_store.py`** — 本地行情仓库。首次全量、之后只补增量；因为存的是 **hfq（基准为上市价，历史不改写）**才可以安全追加，`adjust="qfq"` 一律强制整表重建。
  每次增量都多抓 `OVERLAP_DAYS` 天与本地**对账收盘价**，不一致就整表重建（数据源改口径/修历史数据时不会把两段缝起来）。
  `*.meta.json` 记的是**请求过的区间**而非数据首尾日 —— 否则请求起点落在节假日时，每次都会去补一个永远为空的头段。
- **`lib/swings.py`** — ZigZag 波段分解 + 独立回撤事件。回撤按"创新高→再创新高"切分，**不要**用"低于前高 X%"逐日计数（价格在阈值附近抖动时同一轮回调会被记成几十次）。
- **`lib/regime.py`** — 市场状态分类（趋势上行 / 宽幅震荡 / 趋势下行）。全部 rolling/expanding，**只用当日及以前的数据**；带 `confirm_days` 滞回防止状态天天翻。
  `regime_stats()` 用未来收益检验标签有没有信息量 —— 三档收益没差别就说明分类器是噪声。
- **`lib/ladder.py`** — 分批建仓模拟器。引擎 `run_backtest` 是二元仓位（全仓/空仓），承载不了"三成仓/满仓/闲置现金"这些中间状态，故另起一套；成本与 T+1 口径与 `engine.py` 对齐，闲置现金按 `cash_rate` 计息。
  止损型离场后**锁住再入场**直到趋势转好，否则下跌途中会刷出成百上千笔来回交易。
  `PLAYBOOK` 定义每个状态的打法，`simulate_adaptive()` 按状态切换。
- **`lib/oil_price.py`** — Brent/WTI/SC 原油价格，走**新浪财经**（`futures_foreign_hist` / `futures_main_sina`）而不是 `market_data.py` 的 akshare→baostock→yfinance 三源：那三源里油价相关接口大多打 eastmoney 域名，本机 eastmoney 被主动阻断（DPI 重置特征）、yfinance 商品期货长期 429，只有新浪这条线通；baostock 不提供商品期货，没有备用可回退。
  新浪这两个接口不支持增量拉取（每次全量返回），所以本地缓存 `data/market/oil/{symbol}.csv` 每次整表覆盖，没有 `price_store.py` 那套头尾段 + 重叠对账逻辑。
  `transmission_table()` 算油价对股价的**领先滞后相关系数**（`merge_asof` 对齐两套不同的交易日历后再算收益率相关性）：纯描述性统计，不是回测信号。实测 601857/600938 上 WTI/Brent 在 **lag=1**（隔夜美盘收盘 → 次日 A 股）相关性明显高于 lag=0，SC（人民币计价、与 A 股同日历）反而在 **lag=0** 最高——原始信号被隔夜时区错位，同日对齐会低估外盘油价的传导。

**关键回测教训：** 分段回测（每段重置资金重铺梯子）会**系统性美化**固定阶梯策略 —— 601857 连续跑 8.5 年，"固定梯度+半止盈+MA250"平均仓位只有 15%，长期空仓。判断长期打法要看**连续全样本**，不要看分段汇总。

**评价口径：必须做敞口对齐。** 底仓 70% 的网格平均仓位 80%+、底仓 30% 的只有 43%，在长期上涨的标的上直接比总收益，比的是"谁买得多"。正确基准是**同平均仓位的静态持有**：把 `fatfinger.simulate_static_mix` 的 target 从 0.1 扫到 1.0 得到一条 (平均仓位 → 总收益) 曲线，再按各策略的实际平均仓位插值。校准检查是 `simulate_buy_hold` 的超额应 ≈ 0。
按这个口径实测（详见 [changelog](../changelog/2026-08-10-limit-order-and-grid-bench.md)）：54 组网格参数**没有稳定超额**，最好的一组两只票各 +6.3pp；唯一量级明显的是 `simulate_adaptive` 的 +46.0pp / +76.0pp —— 起作用的是**按状态调仓位**，不是在固定价格网格里来回蹭。

**`simulate_grid` 的锚价默认钉死在首根收盘价**，一路上涨的标的会让网格从未装上膛：600938 自上市起没跌破首日锚 1.2% 以上，36 组参数全是 1 笔交易、结果完全相同。看网格结论前先看 `n_trades` 是不是 1；需要锚随新高上移就传 `ratchet=True`（只在 `level==0` 时移，持仓期间抬锚会让卖出触发价对不上买入格）。

**CLI：** `python -m backtest.scripts.track_oil [--offline] [--backtest] [--chart] [--symbols ...] [--capital N]`
