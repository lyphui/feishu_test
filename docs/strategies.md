# 策略体系 (`strategies/`)

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。

| 策略类 | 文件 | 适用场景 |
|--------|------|----------|
| `MACDStrategy` | `macd.py` | 教科书金叉/死叉，无过滤 |
| `LuMACDStrategy` | `lu_macd.py` | 三级底部确认（0 轴上，底背离，金叉），长线建仓 |
| `LuMACDBullStrategy` | `lu_macd_bull.py` | 牛市过滤（大盘月线）+ 截取红柱最陡段，高频战术 |

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
- 牛市判断：大盘**已收盘**月线 DIF > 0 且 DIF > DEA
- 买入：近 `cross_window`(默认3) 根内出现金叉 **且** 红柱连续 `expand_bars`(默认2) 根拉长。
  单根"红柱拉长"在金叉当根恒为真（hist 由 ≤0 翻正），起不到过滤作用，故必须连续确认；
  `expand_bars=1, cross_window=1` 可复现旧口径
- 卖出模式：`shrink_exit=True`（红柱缩短即走）或 `False`（等死叉）
