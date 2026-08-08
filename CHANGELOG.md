# 更新日志

本文件记录对回测引擎与报告层有行为影响的改动。格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
「行为变更」一节里的条目会改变已有回测的输出数值，重跑历史结论前请先读。

---

## [未发布] — 2026-08-08

本轮主题：**让回测数字对得起实盘**。此前的成本、胜率、排序口径都存在系统性
高估，参数默认值与策略前提自相矛盾，且整套参数扫描没有任何样本外验证。

### 行为变更（会改变历史回测的输出数值）

- **交易级收益改为净额口径**（`backtest/engine.py`）
  `trades["return_pct"]` 现已扣除买入佣金、卖出佣金与印花税；毛收益保留在新增的
  `gross_return_pct` 列供对照。**胜率与盈亏比随之由净额算出。**
  高频策略一次往返的成本约占 0.2%，用毛收益会把一批实际亏损的交易记成盈利——
  修改后同一份数据的胜率通常下降。
- **交易级统计只看 `eval_start` 之后**（`backtest/engine.py`）
  交易次数、胜率、成本统计不再包含预热期成交；`result["trades"]` 本身仍返回完整
  记录供绘图。已知边界：跨窗口边界的那笔（窗口前买入、窗口内卖出）按卖出日归属。
- **挂单年龄语义收紧**（`backtest/engine.py`）
  同向信号连日重复不再重置 `pending_age`；挂单超时作废后，必须等信号真正消失
  （出现 0 或反向信号）才允许重新挂单。
  此前动能策略在连板期间每天都发买入信号，挂单被无限续期，`max_pending_days`
  形同虚设，最终会追在最高点上。
- **批量回测默认止损止盈调整**（`backtest/jcy_macd_bull_batch.py`）
  `--stop_loss` 由 `0.20` 改为 `0.10`；`--take_profit` 由 `0.10` 改为**默认关闭**。
  本策略靠 `shrink_exit` 的动能衰减离场，固定比例止盈会把「最陡峭的那段」提前切断；
  旧的 0.20/0.10 是 1:2 的反向盈亏比，与策略前提直接冲突。
  两个参数现均接受 `none` / `off` / 空字符串表示不启用。
- **`summary.csv` 主排序键改为「日均超额bp」**（`backtest/batch_report.py`）
  各股统计窗口从几十到几百个交易日不等，按总超额排序等于按「谁跑得久」排。
  `日均超额bp = 超额收益% × 100 ÷ 统计交易日数` 对窗口长度线性归一，跨标的才可比。
  不用年化：`(1+r)^(252/n)` 在 n 小时会几何放大。

### 修复

- **胜率分母混入建仓记录**（`backtest/engine.py::_calc_trade_stats`）
  建仓行的 `return_pct` 是 `None`，经 `DataFrame.to_dict("records")` 后变成
  `float NaN`，而 `is not None` 拦不住 NaN。NaN 既不 `> 0` 也不 `<= 0`，于是
  分子不变、分母翻倍——**一笔全胜的回测被报成 50% 胜率**。现同时排除 `None` 与 `NaN`。
- **`--metric` 拼错要跑完整个网格才报错**（`backtest/param_sweep.py`）
  合法指标名收进 `METRICS` 并作为 argparse `choices`，改为启动时即退出。
- **净值曲线基数用错**（`backtest/batch_report.py::normalized_equity`）
  改用引擎新导出的 `equity_base`（统计窗口起点权益）而非 `initial_capital`。
  预热期不交易时两者相等，但调用方若在窗口前就有成交，只有前者能让曲线从 1.0 起步。
- **热力图丢失 `None` 参数行列**（`backtest/param_sweep.py`）
  改用 `build_matrix()` 替代 `df.pivot_table`：pivot 把 `None` 当 NaN 丢掉整行整列
  （而「不设止盈」恰恰是新的默认值），且按值排序会打乱 `True/False` 这类轴的语义顺序。
- 去掉 `plot_portfolio` 的 `tight_layout()`：它会覆盖显式设定的 `hspace`，
  `savefig(bbox_inches="tight")` 已负责裁白边。

### 新增

- **样本外验证 `--oos-frac`**（`backtest/param_sweep.py`）
  网格搜索本身是纯样本内的，在同一批数据上遍历再挑最好的，只能说明参数高原平不平坦。
  `--oos-frac 0.3` 按推荐日把候选股切成两段（较早 70% 选参数，最近 30% 只做验证），
  切点落在日期边界上，保证同一天推荐的标的不跨界污染样本外。选完参数后自动把
  IS 最优格与默认格拿到 OOS 上重跑并给出判读。未启用时会明确提示「结论不足以支撑实盘」。
- **交易成本实际发生额**（`backtest/engine.py`）
  `result["costs"]` 除费率假设外新增 `total_commission` / `total_stamp_duty` /
  `total_slippage` / `total_cost` / `cost_drag_pct`（占窗口起点资金比例）。
  只报费率假设看不出成本吃掉了多少收益。
  `trades` 同步新增 `commission` / `stamp_duty` / `slippage_cost` 三列。
- **横截面汇总增强**（`backtest/batch_report.py`）
  新增「日均超额bp」「成本占比%」两列；控制台增加统计窗口分布、日均超额均值/中位数、
  交易成本占比，以及**按窗口长度分组**（<3月 / 3-6月 / 6-12月 / ≥1年）的日均超额——
  用来识别「信号只在推荐后一两个月有效」这种衰减。窗口长度相差 ≥3 倍时额外告警。
- `stop_loss` / `take_profit` 扫描轴的候选值加入 `None`（=不启用）。

### 文档

- 组合净值图与模块 docstring 明确标注：每只票各自满仓独立回测后按日历取算术平均，
  **不是可投资的组合净值**（N 只同时满仓在现实中不可能）。图例与标题同步改为
  「平均单股净值」。
- `CLAUDE.md` 同步以上全部口径变更。

### 测试

- 新增 `tests/test_param_sweep.py`：样本内外切分（含同日标的不跨界、小样本优雅退化）、
  `build_matrix` 保留 `None` 且遵循轴声明顺序、`METRICS` 与 `evaluate_combo` 产出的
  契约、`--metric` 非法值拒绝、`fmt_value` 渲染。
- `tests/test_param_sweep.py::test_defaults_match_batch_cli` 直接调用
  `batch.parse_args()` 取真实默认值比对，守住 `DEFAULTS` 与 CLI 的漂移——
  复制一份常量的测试永远通过，也就永远发现不了漂移。
- `tests/test_engine_execution.py` 新增：连续同向信号不续期挂单、信号消失后重新挂单、
  净额收益、只赚手续费的交易算亏损、胜率分母不含建仓记录、成本明细手算比对、
  交易统计与成本遵守 `eval_start`。
- `tests/test_batch_report.py` 新增引擎 → 汇总层的**字段契约测试**：用真实
  `run_backtest` 的输出跑一遍汇总。手写的 `_fake_result` 改了引擎不会报错，这个测试会。

全量测试：149 passed（离线，不联网、不读真实 `data/`）。
