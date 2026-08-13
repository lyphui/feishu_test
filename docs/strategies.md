# 策略体系 (`backtest/strategies/`)

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。

| 策略类 | 文件 | 适用场景 |
|--------|------|----------|
| `MACDStrategy` | `macd.py` | 教科书金叉/死叉，无过滤 |
| `LuMACDStrategy` | `lu_macd.py` | 三级底部确认（0 轴上，底背离，金叉），长线建仓 |
| `LuMACDBullStrategy` | `lu_macd_bull.py` | 牛市过滤（大盘月线）+ 截取红柱最陡段，高频战术 |
| `MACrossStrategy` | `ma_cross.py` | 快慢均线交叉（默认 MA5/MA8），可选量能过滤；**实测为负结论**，见 [ma-cross-5-8.md](ma-cross-5-8.md) |

**BaseStrategy 接口（必须实现）：**
```python
prepare(df) -> df          # 计算指标，生成 signal 列（1/-1/0）
plot_indicators(ax, df, colors) -> None
name: str                   # 策略名（图表标题）
params: dict               # 参数字典（展示用）
```

**BaseStrategy 共享方法：**
- `_ema(series, period)` — 静态方法，EMA 计算，所有 MACD 策略子类共用（避免各子类重复定义）
- `_resample_period(df, rule, agg, drop_incomplete=True)` — **跨周期重采样必须走这里**。
  以区间内最后一个**真实交易日**为标签（不是 `"MS"` 月初），并丢掉末尾未走完的那根 K 线。
  直接用 `df.resample("MS")` 会把月末收盘价打上月初标签，构成整月的未来函数
- `_align_to_daily(series, daily_index)` — 低频 → 日线对齐。先在**并集**索引上 ffill 再收敛回日线；
  直接 `series.reindex(daily_index).ffill()` 会把标签不在交易日上的 K 线整根丢掉

> ⚠️ 多周期策略的时序铁律：周/月线信号只在该区间**收盘当天**生效，再由引擎 `shift(1)` 到 T+1 成交。
> `tests/test_strategy_lookahead.py` 用"截断数据重算、历史信号不得改变"的属性测试守住这一点。

**LuMACDBullStrategy 特殊设计：**
- 构造函数接受 `index_df`（大盘数据），`prepare()` 参数中的 `index_df` 优先，否则 fallback 到构造函数传入的值
- 牛市判断：大盘**已收盘**月线 DIF > 0 且 DIF > DEA。
  **指数必须从 `config.INDEX_HISTORY_START` 起取**（走 `index_history_start()`）：
  月线 EMA(26) 要几十根月线才收敛，只喂两年数据算出来的 `bull_market` 会失真，
  且起点若跟着候选池走，加一篇更早的文章就会改写全部历史判定
- 买入：近 `cross_window`(默认3) 根内出现金叉 **且** 红柱连续 `expand_bars`(默认2) 根拉长。
  单根"红柱拉长"在金叉当根恒为真（hist 由 ≤0 翻正），起不到过滤作用，故必须连续确认；
  `expand_bars=1, cross_window=1` 可复现旧口径
- 卖出模式：
  - `shrink_exit=True`（默认）——**持有条件 = 红柱且在拉长**，
    `sell = hist_shrinking | (hist <= 0)`。不能只写 `hist_shrinking`：它自带 `hist > 0` 前提，
    柱子从正值**直接跌破 0**（跳空/急跌）的那根不算缩短，之后整段负柱也不算，
    于是完全没有离场信号，只剩固定止损兜底（实测 600938 有 10/45 次、601857 有 9/70 次
    零轴下穿属于此情形，最长负柱 37 个交易日 / −15.8%）。
    用**状态** `hist <= 0` 而非**事件** `death_cross`：挂单顺延导致建仓时柱子已翻负的情况，
    事件那一根早就过去了。二者本就等价（`hist = (DIF−DEA)×2`），状态版还多兜住了这一类
  - `shrink_exit=False`——等死叉再走（保守版）

**MACrossStrategy（`ma_cross.py`）：**
- `fast/slow`（默认 5/8）、`ma_type`（`sma` / `ema`）：快线上穿慢线 → `signal=1`，
  下穿 → `signal=-1`。均线预热期整行 `dropna`，首根有效 K 线不会凭空报信号
- `vol_window/vol_ratio`（默认关闭）：**量能只过滤进场**——金叉当日量 ≥ N 日均量才买，
  缩量金叉直接放弃、不顺延；**死叉照卖不看量能**。出场纪律一旦附加条件，
  就会退化成「跌了还找理由拿着」，而止损恰恰是这类策略唯一真正的风控
- 均量窗口含当日成交量（收盘后即可知），配合引擎 `shift(1)` 在 T+1 开盘成交，时序自洽
- ⚠️ 这个策略是**为了证伪而写的**：2018-01 至今 30 个 A 股/ETF 标的上，MA5/8 无一跑赢
  买入持有，零成本对照下仍全负。留在仓库里是为了「下次再有人问同一个规则时能直接重跑」，
  不是推荐用法——完整数据见 [ma-cross-5-8.md](ma-cross-5-8.md)
