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

**是。** ladder / grid / fatfinger 的成交判定从 0.999 松容差收紧到 0.9999
（engine 口径），停牌判定从 `== 0` 收紧到 `<= 0`。受影响的是「开盘价落在
[0.9999×涨跌停价, 0.999×涨跌停价) 这一窄带内」的日子，以及 volume 为负/NaN
的日子。

**实测差异：零。** 改动前后各跑一遍 `compare_playbooks --code 601857 --offline`
与 `backtest_fatfinger --offline`（601857 / 600938 两只油票，2028 与 1043 个
交易日），输出**逐字一致**——两个样本里没有任何交易日落在差异带内，负值/NaN
成交量日也不存在。当前留档的数字（`docs/stock-playbook.md`、
`docs/execution-bench.md`）无需重跑更新。

代码层面的语义仍然收紧了：今后若出现开盘价贴着涨跌停价的样本，ladder/grid/
fatfinger 会按 engine 的严格口径判定（更少误判封死），两套骨架的行为才真正一致。
