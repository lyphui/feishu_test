# [已发布] — 2026-08-14（rf/cash_rate 与港股成本收编 costs.py）

评审 `docs/backtest-review.md` 项 3、项 6 落地。**数值逐字未动，只收敛出处。**

## 项 3：无风险利率 / 现金利率

这是此前**唯一没有测试守护的常量漂移面**：`engine._calc_sharpe` 的
`rf=0.02`、`ladder.summarize` 里硬编码的 `- 0.02`、`compare_playbooks._sharpe`
的 `rf=0.02`、`ladder._run` 的 `cash_rate=0.015` 各写一份字面量——同一张
对比表里哪天改了一处，排序就变成按公式排序。

收编进 `lib/costs.py`：`RISK_FREE_RATE = 0.02`、`CASH_RATE = 0.015`，
四处消费点改为引用；`tests/test_costs.py` 新增 `inspect.signature` 同源守护。

## 项 6：港股成本 market-aware

港股成本模型（佣金 0.25% 最低 HK$100、平台费 HK$30、ETF 免印花税、
法定/结算费率）原定义在 `lib/trend_stop.py`、与月频决策耦合。港股标的
不止 3175.HK 一只，成本层应是公共的：

- 常量与 `hk_trade_cost()` / `hk_fee_rate()` 逐字搬进 `lib/costs.py`
  （`STAMP_DUTY` 港股侧改名 `HK_STAMP_DUTY` 以免与 A 股撞名）；
- 新增 `costs.for_market("A"|"HK")` 返回 `MarketCosts` 费率组
  （费率 + 滑点 + 整手 + 印花税豁免标记），为后续账本/组合层备用；
- `trend_stop` 改为消费者，保留同名 re-export，历史导入不受影响；
- `tests/test_costs.py` 守护：`trend_stop.hk_trade_cost is costs.hk_trade_cost`
  及 `for_market` 与散常量逐位一致。

## 行为变更

否。所有费率/利率值与此前逐位相同（港股数值为逐字搬移），回测输出不变。
