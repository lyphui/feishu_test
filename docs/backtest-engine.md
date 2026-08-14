# MACD 策略回测引擎（`backtest/engine.py` + `backtest/config.py`）

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。

**职责：** 核心回测引擎（纯函数，无 CLI，**不含绘图**），被所有回测入口脚本复用。
图表在 `backtest/reports/report.py`——`import engine` 不会拖进 matplotlib，批量回测/参数扫描/pytest 因此不必付这个代价（`tests/test_engine_no_matplotlib.py` 守住）。
成本假设与涨跌停/停牌成交判定（`tradability()`）都来自 `lib/costs.py`，与 `lib/ladder.py`、`lib/fatfinger.py` 同源（`tests/test_costs.py` 守住）——两套撮合骨架的费率或成交口径一旦漂开，敞口对齐口径就失效。
`backtest/scripts/backtest_macd.py` 为薄入口，re-export `run_backtest`/`plot_backtest`/`fetch_stock_data` 以兼容历史 `from macd_analysis import ...` 导入。

**关键函数：**
- `fetch_stock_data(symbol, start, end)` — 从 `lib.market_data` 再导出（canonical 位置在 `backtest/lib/market_data.py`）
- `run_backtest(symbol, strategy, capital, stop_loss, take_profit, ...)` — 执行回测
  - 按信号买卖，100 股整数手，T 日信号在 **T+1 开盘**成交（`signal.shift(1)`）
  - **单根 K 线内的事件顺序**：① 开盘执行挂单 → ② 盘中止损/止盈 → ③ 收盘估值。
    当日刚建仓的不检查止损（A 股 T+1，当天买当天卖不掉）
  - **A 股成交约束**：开盘涨停买不进、开盘跌停卖不掉、`volume<=0`（含 NaN/负值）停牌不可成交；
    未成交的信号顺延最多 `max_pending_days` 天，之后作废（避免一字连板追到天上）。
    涨跌停幅度由 `infer_limit_pct(symbol)` 按代码前缀推断（主板 10% / 双创 20% / 北交所 30%）
  - **挂单年龄语义**：同向信号连日重复**不重置** `pending_age`；挂单超时作废后，
    必须等信号真正消失（出现 0 或反向信号）才允许重新挂单。
    否则动能策略在连板期间每天都发买入信号，挂单被无限续期，`max_pending_days` 失效
  - **成本**：佣金（双边万三，单笔最低 5 元）+ 印花税（单边千一）+ 双边滑点（默认千一）。
    `result["costs"]` 除费率假设外还含**实际发生额**：`total_commission` / `total_stamp_duty` /
    `total_slippage` / `total_cost` / `cost_drag_pct`（占窗口起点资金比例），只统计 `eval_start` 之后
  - **交易级收益按净额计**：`trades["return_pct"]` 已扣双边佣金与印花税，
    `gross_return_pct` 保留毛收益供对照。胜率/盈亏比由净额算出——高频策略往返成本约 0.2%，
    用毛收益会把一批实际亏损的交易记成盈利。
    建仓行的 `return_pct` 是 `None`，经 DataFrame 会变成 `NaN`，
    `_calc_trade_stats` 必须同时排除两者，否则建仓记录进了胜率分母（全胜报成 50%）
  - `eval_start="YYYYMMDD"` — 统计窗口起点，把指标预热期排除在收益/回撤/夏普/基准之外，
    **交易级统计（次数/胜率/成本）同样只看窗口内**；`trades` 本身仍返回完整记录供绘图
  - `df=` — 直接注入行情，跳过网络请求（测试与参数扫描复用）
  - 基准（买入持有）与策略同口径：窗口首日**开盘价**买入、末日收盘估值
  - **暴露度**：`equity_curve` 带 `shares` 列，`result` 含 `exposure_pct`（在场交易日占比）
    与 `avg_holding_days`。空仓日既不赚也不亏，**回撤与夏普都未按暴露度调整**——
    在场 3% 的时间做出 −24% 的回撤，和满仓做出 −24% 完全是两回事，汇总层与
    `print_summary` 都显式标注这一点
- `plot_backtest(result, save_path)` — 标准 4 面板图（价格+信号、指标、权益、回撤）。**在 `backtest/reports/report.py`**，不在 engine 里

**共享配置层（`backtest/config.py`）：** 三个单股入口共用
- `load_backtest_config(filename, *, defaults)` → `BacktestConfig`：统一解析 `backtest/presets/*.ini` 的 `[backtest]` 段（end_date 默认今日、止损止盈空值转 None、proxy 写环境变量、缺失时按 defaults 写出）；策略专属参数经 `cfg.get_int/get_bool/get_float` 从 `.extra` 读取
- `INDEX_HISTORY_START` / `index_history_start(requested_start=None)`：大盘指数取数的
  **绝对起点**，与候选池无关。牛市过滤器算月线 MACD，EMA(26) 要几十根月线才收敛；
  起点若按「最早推荐日 − 600 天」算，加一篇更早的文章就会改写全部个股的 `bull_market`
  历史（沪深300 实测：截到 2021-01 有 7 个月判定翻转），回测不可复现
- `execution_kwargs(cfg)` → dict：从 `.ini` 读成本与交易约束（commission_rate / min_commission / stamp_duty / slippage / limit_move_check / max_pending_days），直接 `**` 展开给 `run_backtest`，保证三个入口的成本假设一致
- `OutputPaths(save_dir, prefix, name, symbol, end_date)`：统一输出路径（`.chart/.csv/.status`），`OutputPaths.safe()` 清洗文件名

**CLI：** `python -m backtest.scripts.backtest_macd --config jxty_jcy_260104.ini`

## 回测预设格式 (`backtest/presets/*.ini`)

```ini
[backtest]
symbol      = 600519       # 股票代码
name        = maotai       # 名称（用于文件名）
start_date  = 20180101
end_date    =              # 留空 = 今日
index_symbol = 000300      # 大盘指数（牛市判断）
capital     = 100000
stop_loss   =              # 如 0.10 = 10%，留空不设
take_profit =
save_chart_dir = output/
proxy       =              # 如 http://127.0.0.1:7890

# LuMACDBull 专属
shrink_exit = true         # true=红柱缩短即走，false=等死叉
```
