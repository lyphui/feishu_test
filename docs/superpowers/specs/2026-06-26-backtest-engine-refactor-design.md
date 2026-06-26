# 回测引擎修复与重构设计

日期：2026-06-26
范围：回测引擎（`run_backtest`）、策略层（`BaseStrategy` + 三子类）、可视化、共享工具

---

## 1. 背景与目标

当前回测平台存在 **三类问题**，按重要性排序：

1. **回测真实性缺陷**（最高优先级，直接决定结论可信度）
   - 成交价用信号当根收盘价 → 系统性高估收益（未来函数）
   - 月线牛市过滤器前视偏差 → 月初就用上整月才确定的月线 MACD
   - 止损止盈仅在收盘价判断 → 盘中穿越不触发，低估实际亏损
2. **抽象边界混乱**
   - `prepare` 双签名 + 构造函数注入 + `BullStrategyAdapter` 三种传参并存
   - 持仓状态逻辑漏到 Adapter 层（本应归引擎）
3. **重复代码**
   - `_calc_macd` 四份拷贝、`BaseStrategy._ema`
   - 两套绘图函数价格/净值/回撤面板逐行重复
   - 统计逻辑（夏普/年化/胜率）与交易模拟混在 160 行大函数里

**目标**：消除未来函数使回测结论可信；统一策略接口；抽离可测的纯函数并补单测；去重。**不改变**任何 CLI 接口、配置文件格式、输出目录结构。

---

## 2. 关键设计决策（已与用户确认）

| 决策点 | 选定方案 |
|---|---|
| 成交时机 | **次日开盘价成交**。信号在 T 日收盘算出，T+1 用 `open` 成交。期末仍按最后收盘价清算。 |
| 月线前视偏差 | **整体滞后一个月**。用上一个已收完的月线 MACD 判断本月牛市。 |
| 止损止盈 | **盘中触发，按阈值价成交**。用当日 `low` 判止损、`high` 判止盈，成交价取阈值价 `cost*(1±pct)`。 |
| 测试策略 | **合成数据单测**。手造小 DataFrame，pytest 驱动，不联网。 |
| 交易成本 | **完整真实成本**：佣金 `max(成交额*万三, 5元)`；沪深过户费万分之0.1（双边）；印花税按日期切换（2023-08-28 前千一、含当日及之后万五，仅卖出）。参数化可调。 |
| 涨跌停 | **硬拦截 + 顺延成交**：次日 `open` 距前收 ±阈值判为一字板则当日不成交，顺延到下一个能成交的交易日。 |
| 夏普口径 | **两个都算**：账户级（含空仓日）+ 策略级（仅持仓日），报告并列。 |
| 范围 | 全部四类（真实性 + 接口 + 去重 + metrics 抽离/单测）。 |

---

## 3. 模块布局（自评后修订）

**不新建 `engine.py`**：`run_backtest` 在 `macd_analysis.py` 内原地重构（现有 4 处 `from macd_analysis import run_backtest` 不受影响，降低风险）。

| 模块 | 变更 |
|---|---|
| `utils/indicators.py` | **新建**。`ema(series, period)`、`calc_macd(close, fast, slow, sig) -> DataFrame[DIF,DEA,MACD]`。删除 4 处 `_calc_macd` 拷贝与 `BaseStrategy._ema`。 |
| `utils/metrics.py` | **新建**。`calc_sharpe`（账户级 + 策略级双口径）、`annualized_return`（含小 n 保护）、`calc_trade_stats`（区分买入次数/已平仓次数）、`max_drawdown`。纯函数，单测覆盖。 |
| `utils/fees.py` | **新建**。`compute_fee(amount, side, trade_date, cfg)` —— 佣金 `max(amount*rate, min_comm)`、过户费、按日期切换的印花税。费率默认值参数化。纯函数，单测覆盖。 |
| `macd_analysis.py` | 原地重构 `run_backtest`：加 `execution="next_open"`、`intraday_stops=True`、`verbose=True` 参数；抽 `_execute_sell` 合并止损/止盈/正常/清仓四段；费用改调 `utils.fees.compute_fee`；买入按含费价反推股数；涨跌停顺延；停牌警告；统计改调 `utils.metrics`；`print` 受 `verbose` 控制。 |
| `strategies/base.py` | `prepare(df)` **单参数**保持。删 `_ema`（移至 indicators）。 |
| `strategies/lu_macd_bull.py` | `prepare` 改单参数（读 `self._index_df`）；月线滞后修复；构造函数加 `trade_start_date`（取代 Adapter 的信号屏蔽）；信号无状态。 |
| `strategies/lu_macd.py` | 月线/周线滞后修复；修 `_align_to_daily` 的 `shift` 缺失 bug。 |
| `strategies/macd.py` | 改用 `utils.indicators.calc_macd`。 |
| `utils/bull_backtest.py` | **删除 `BullStrategyAdapter`**（与引擎状态机冗余）。`export_bull_daily_status`、`plot_bull_backtest` 保留。 |
| `utils/plotting_panels.py` | **新建**。`plot_price_panel`、`plot_equity_panel`、`plot_drawdown_panel`。三套绘图改为薄组合。 |
| `backtest/jcy_macd_bull_batch.py` | 去掉 Adapter，改用 `LuMACDBullStrategy(index_df=..., trade_start_date=...)`。 |
| `tests/` | **新建**。pytest 单测目录。 |

`utils/runner.py`（config 解析去重）**不在本计划**，作为后续独立阶段，避免与正确性修复的验证耦合。

---

## 4. 行为变更细节

### 4.1 次日开盘价成交

引擎主循环改为：T 日产生信号 → 记录"待执行"，T+1 用 `df.loc[T+1, "open"]` 成交。
等价实现：把 `signal` 整体 `shift(1)`（信号延后一日），成交价取当日 `open`。
- 买入价 = T+1 `open`；卖出价 = T+1 `open`。
- 最后一根有信号但无次日 → 该信号作废（无法成交）。
- 期末清仓仍用最后一根 `close`（清算，非信号成交）。

### 4.2 盘中止损/止盈

持仓期间每日：
- 止损：若 `low <= cost*(1-stop_loss)` → 按 `cost*(1-stop_loss)` 成交。
- 止盈：若 `high >= cost*(1+take_profit)` → 按 `cost*(1+take_profit)` 成交。
- 同日同时触发：**止损优先**（保守，假设先到不利价）。
- 止损止盈按当日盘中判断，**早于**当日的信号成交（信号是次日 open，止损是当日盘中，时序上当日盘中在前）。

### 4.3 月线滞后修复

`_bull_market_filter` 返回的月线 bool Series 在对齐回日线前 **`shift(1)`**：本月用上月已收完的月线 MACD。
`_resample_monthly` 标签语义保持 `MS`（月初），但对齐时 `reindex(daily).ffill()` 之前先把月线序列整体后移一个月。
`lu_macd.py` 的周线/月线同样处理；并修 `_align_to_daily` 文档声称 `shift(1)` 而实际未 shift 的 bug。

### 4.4 删除 Adapter

Adapter 的两项逻辑去向：
- `trade_start_date` 信号屏蔽 → `LuMACDBullStrategy.__init__(trade_start_date=...)`，`prepare` 末尾把 `index < cutoff` 的 signal 清零。
- "首次操作必须买入 / 清除买入前卖出" → **删除**。引擎状态机本已保证 `position==0` 时 `-1` 信号无操作，该逻辑仅影响图表标记；图表改从实际 `trades` 还原买卖点，不再依赖预清洗的 signal 列。

---

## 4b. 回测代码缺陷清单（review 补充）

除 §1 三大问题外，代码审查另发现 8 项缺陷，按对结果的扭曲程度排序。
归类到对应 Phase 一并修复。

**交易成本模型（Phase 1，直接改收益数字）**

1. **缺最低佣金 5 元**（`macd_analysis.py:120`）：`fee=cost*0.0003` 未取
   `max(_, 5)`。小额成交被低估固定成本，**系统性抬高高频策略收益**
   （`LuMACDBullStrategy` 首当其冲）。
2. **漏过户费**：沪深均按成交额万分之0.1 双边收取，当前完全没有。
3. **印花税率过时**（`macd_analysis.py:47`）：写死 `0.001`，但 2023-08-28
   起减半为 `0.0005`。2023-08 后回测卖出成本被高估一倍——方向与 1、2 相反，
   故需**按成交日期切换**而非写死。

修复：抽 `compute_fee(amount, side, trade_date, cfg)` 统一计算，
`side∈{buy,sell}`；佣金 `max(amount*rate, min_commission)`；
过户费 `amount*transfer_rate`；印花税仅 `sell` 且按 `trade_date` 取率。
费率全部参数化（默认值见 §2），便于回归测试与未来调整。

**撮合现实性（Phase 1）**

4. **买入现金可能为负**（`macd_analysis.py:117-121`）：股数按 `invest/price`
   取整未给佣金留预算，全仓时扣费后 `cash` 变微小负数。改为按
   `invest/(price*(1+buy_fee_rate))` 反推可买股数。
5. **涨跌停一字板无法成交**：见 §2 决策，硬拦截 + 顺延。需在 OHLC 上判断
   次日 `open` 是否 ≈ 前收 ±涨跌停幅度（主板 10%、创业板/科创板 20%、
   ST 5%——本轮按代码前缀粗略判定，记为可调参数）。
6. **停牌/数据缺口**：`iterrows` 逐行不校验日期连续性；年化用 `len(eq_df)`
   把停牌日算进交易日数，稀释年化。本轮**仅加警告**（检测到日期间隔
   异常时打印），不做插值。

**统计口径（Phase 0，随 metrics 抽离一起修）**

7. **`total_trades` 与胜率分母不一致**（`macd_analysis.py:186`）：前者只数买入，
   后者基于已平仓卖出。期末持仓时两者分母不同。报告改为同时给出
   「买入次数」与「已平仓次数」，胜率明确基于后者。
8. **夏普含大量空仓 0 收益日**（`macd_analysis.py:201-206`）：压低波动率、
   虚高夏普。见 §2 决策，账户级 + 策略级两个都算。
9. **年化小 n 失真**（`macd_analysis.py:169`）：`(1+r)^(252/n)` 在 n 很小时
   指数放大到离谱。`annualized_return` 加保护：n < 阈值（如 20 交易日）时
   不年化，直接返回区间收益并标注「样本不足，未年化」。

**较小项（记录，本轮按注释/文档说明，不强改逻辑）**

10. `position_size<1` 时按 `cash*size` 而非「总资产*size」，多次交易后仓位
    比例漂移。当前仅用 1.0，文档注明该限制。
11. 基准「买入持有」不收费，与策略双边收费对比口径略偏向基准；文档说明。

## 5. 测试计划（合成数据，pytest）

`tests/` 下：

- `test_indicators.py`：`calc_macd` 对已知序列的 DIF/DEA/MACD 数值断言；与旧实现等价性。
- `test_metrics.py`：`annualized_return` 小 n 保护（n=1,2 不爆炸、低于阈值返回区间收益不年化）；`calc_sharpe` 零方差返回 0；账户级 vs 策略级夏普口径；`calc_trade_stats` 胜率/盈亏比边界；买入次数 vs 已平仓次数分母区分。
- `test_fees.py`：`compute_fee` —— 佣金触发最低 5 元（小额成交）；过户费双边；印花税仅卖出且 2023-08-28 前后取不同税率（边界日当天用新税率）；买入不收印花税。
- `test_engine_execution.py`：手造 OHLC，信号在 T 日，断言成交日为 T+1、成交价=T+1 `open`；最后一根信号无次日→不成交；买入扣费后 `cash >= 0`（费用预留）。
- `test_engine_stops.py`：构造当日 `low` 击穿止损/`high` 击穿止盈，断言成交价=阈值价；同日双触发→止损优先。
- `test_limit_halt.py`：构造次日一字涨停→买入顺延；一字跌停→卖出顺延；顺延到下一可成交日成交。
- `test_lookahead.py`：构造月线方向在月中翻转的指数序列，断言日线 `bull_market` 滞后一个月生效。

`pyproject.toml` 加 `pytest` 到 dev 依赖（或 `[project.optional-dependencies]`）。

---

## 6. 实施阶段（可独立验证）

- **Phase 0**：`utils/indicators.py` + `utils/metrics.py`（含小 n 保护、双口径夏普、买入/平仓分母区分，缺陷 #7/#8/#9）+ 单测（纯函数，无行为变更，先立测试台）。
- **Phase 1（正确性）**：次日 open 成交、盘中止损、月线滞后修复，以及交易成本重做（`compute_fee`，缺陷 #1/#2/#3）、买入费用预留（#4）、涨跌停顺延（#5）、停牌警告（#6）+ 对应单测。**重点验证区**。
- **Phase 2（接口）**：`prepare` 单参数、删 Adapter、信号无状态、batch 适配。
- **Phase 3（绘图去重）**：`utils/plotting_panels.py`，三套绘图改组合。
- **Phase 4（可选，独立）**：`utils/runner.py` config 去重。

每个 Phase 跑全量单测 + 至少一次真实回测冒烟（人工核对图表/CSV 合理）。

---

## 7. 非目标（YAGNI）

- 不引入向量化回测框架（backtrader/vectorbt）——保持现有逐日循环风格。
- 不改 CLI 参数、config 格式、输出目录命名。
- 不动数据采集流水线（`prepare_jcy_data.py`）。
- 不做 `runner.py` 之外的入口脚本架构调整。
- 不加滑点参数（用户选了次日 open，非滑点方案）。
