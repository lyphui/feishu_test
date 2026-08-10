# [已发布] — 2026-08-10（牛市过滤器指数起点解耦为绝对日期）

## 问题：回测结果依赖候选池里最早的那篇文章

`LuMACDBullStrategy` 的牛市过滤器算的是大盘**月线** MACD，EMA(26) 要几十根
月线才收敛。旧实现给指数定的取数区间是「最早推荐日 − 600 自然日」：

| 调用方 | 旧指数起点 |
|---|---|
| `jcy_macd_bull_batch.py` | `min(候选推荐日) − 600d` |
| `param_sweep.py` | `min(候选推荐日) − 600d` |
| `jcy_intraday_timing.py` | **每只票各自的** `推荐日 − 600d` |
| `lu_macd_bull_analysis.py` | `.ini` 里的 `start_date` |

600 自然日 ≈ 20 根月线。span=26 的 EMA 在第 20 根时，seed（首个收盘价）还占
着 (1−2/27)^20 ≈ 21% 的权重，而 EMA(12) 只剩 3.6%，于是 `DIF = EMA12 − EMA26`
被系统性拉偏。

两个后果，第二个更严重：

1. **判定本身失真。** 沪深300 实测：把指数起点从全历史截断后，`bull_market`
   与全历史口径不一致的月份数为——截到 2023-01：1 个月；2022-01：2 个月；
   2021-01：7 个月；2019-01：14 个月。
2. **回测不可复现。** 起点跟着候选池走，往 `jcy_insights.json` 里加一篇**更早**
   的文章，全部个股的 `bull_market` 历史都会改写，收益随之变化。这与仓库坚持
   后复权（hfq）的理由是同一件事：历史一旦确定就不该再变。

`jcy_intraday_timing.py` 那条最离谱——每只票用自己的 `data_start` 取指数，
同一天的大盘牛熊判定会**因股而异**。

## 改法

`backtest/config.py` 新增绝对起点与取值函数：

```python
INDEX_HISTORY_START = "20150101"

def index_history_start(requested_start=None):
    """不晚于 INDEX_HISTORY_START；请求更早时以请求为准。"""
    return min(requested_start, INDEX_HISTORY_START) if requested_start else INDEX_HISTORY_START
```

四个调用方全部改走它。个股行情不受影响：日线 EMA(26) 几十根就稳，仍按各自
推荐日往前预热 600 天。

`tests/test_backtest_config.py` 新增两条：起点与候选池无关；预热长度 ≥ 96 个月。

## 行为变更

**是。** `bull_market` 序列变了，买卖点与收益随之变化，历史结论需重跑。
方向是**修正**：新口径下的月线 MACD 才是收敛值。

同时消除了「加一篇旧文章 → 所有回测数值变化」这一不可复现性，此后同一份
`jcy_insights.json` 与同一段行情必然给出同一批数字。
