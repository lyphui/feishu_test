# 涨跌停/停牌成交判定统一到 `lib/costs.tradability`

## 背景

`engine._tradability` 与 `lib/ladder._tradable` 是同一条规则的**两份实现**，
且**行为真的不同**，导致二元引擎与分批建仓模拟器的成交判定不可比（docstring
声称"必须可比"，实际不是）：

| | engine | ladder |
|---|---|---|
| 涨停容差 | `open < up * (1-1e-4)`（紧） | `open < up * 0.999`（松 10 倍） |
| 停牌判定 | `volume <= 0`（含 NaN/负值） | `volume == 0`（负值/NaN 漏判） |
| `prev_close` 非法 | 检查 None/非有限，放行 | `prev_close <= 0`，遇 None 直接 TypeError |

## 改动

- **`lib/costs.py` 新增 `tradability(row, prev_close, limit_pct)`** —— 采用
  engine 的严格口径（相对容差 1e-4 + 完整边界检查），成为两套撮合骨架的**唯一
  真值源**（同 `commission` / `infer_limit_pct` 的定位）。
- **`engine.py`** 删除本地实现，`_tradability` 改为 `costs.tradability` 的旧名
  re-export，历史导入（含 `tests/test_engine_execution.py`）不受影响。
- **`lib/ladder.py`** 删除 `_tradable`，`_run` 转调 `costs.tradability`；
  `_commission` 别名改为直接用 `commission`；`_summarize` 提升为公开 `summarize`
  （结果容器 `LadderResult` 本已是公开名）；docstring 的成交口径一节随之更新。
- **`lib/fatfinger.py`** 成本常量改为**直接取自 `costs`**（不再绕道 ladder 的
  私有 `_commission`/`_summarize`）；`LadderResult`/`summarize` 从 ladder 取
  公开名；停牌判定 `volume == 0` 对齐为 `volume <= 0`。

## 行为变更

**是。** ladder / grid / fatfinger 受两处影响，**方向相反**，别记混：

1. **涨跌停容差 0.999 → 0.9999（容差变窄 ⇒ 判死的日子更少 ⇒ 成交变多）。**
   受影响的是开盘价落在 `[涨停价×0.999, 涨停价×0.9999)` 这一窄带内的日子
   （跌停侧对称）。这一带旧 ladder 判「买不进」，统一后判「买得进」——
   也就是 ladder 过去**过度封死**，现在与 engine 一致。
2. **停牌判定 `== 0` → `<= 0` 且显式拦 NaN（判死的日子更多 ⇒ 成交变少）。**
   volume 为负值或 NaN（按交易日历 reindex 出的空行）的日子过去被当成可成交。

**实测差异：零。** 改动前后各跑一遍 `compare_playbooks --code 601857 --offline`
与 `backtest_fatfinger --offline`（601857 / 600938 两只油票，2028 与 1043 个
交易日），输出**逐字一致**——两个样本里没有任何交易日落在差异带内，负值/NaN
成交量日也不存在。当前留档的数字（`docs/stock-playbook.md`、
`docs/execution-bench.md`）无需重跑更新。**注意这是样本结论，不是恒等证明**：
换标的、换区间仍可能出现落在差异带内的日子。

## 补记（同日修）

首版 `costs.tradability` 的 docstring 与本条目都写着「停牌含 NaN」，但代码只写了
`float(...) <= 0`——而 `float("nan") <= 0` 在 IEEE 下**恒为 False**，NaN 成交量
实际被当成可成交，与文字不符。注意这个洞是**从原 engine 实现继承来的**，所以影响
的不只是 ladder 一侧，engine 的回测同样受影响。已改为显式
`not np.isfinite(vol) or vol <= 0`。

**本地缓存里确实存在这种行**：`600900_hfq`（长江电力 2021-11-29~12-10）、
`601088_hfq/none`（中国神华 2025-08-04~08-15），共 3 个文件 31 行，形态都是
OHLC 四价相同 + volume 为 NaN 的真实停牌段。旧口径会在停牌期间按冻结价「成交」。

**实测差异：零。** 跑 `compare_ma_cross --offline --quick --buckets 蓝筹低波股
周期资源股`（这两桶正好覆盖 600900 与 601088），修复前后汇总表与逐股
`detail.csv` **均逐字一致**。原因是停牌段四价冻结、均线不产生交叉，没有任何策略
在那些天尝试下单，闸门没被触发——**是"没被触发"，不是"这条规则不重要"**：
换一个会在停牌段附近发信号的策略就会出现差异。

配套补上钉住它的测试：

- `tests/test_costs.py` —— 函数级：0 / 负值 / NaN 三种停牌、差异带两侧、涨跌停
  各挡一边、首根 K 线放行不抛异常；外加 `engine._tradability is costs.tradability
  is ladder.tradability` 的同一性断言与「ladder 不得再有 `_tradable`」。
- `tests/test_ladder_regime.py` —— 接线级：ladder 撮合循环真的走那个函数
  （停牌日与一字涨停日 `trades` 为空）。

没有这两组测试，谁再在某一侧写回一份本地实现，全套测试照样绿——这正是本次合并
要防的事。
