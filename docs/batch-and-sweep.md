# 批量回测与参数敏感性扫描

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。

## 批量回测 (`backtest/jcy_macd_bull_batch.py` + `backtest/batch_report.py`)

**职责：** 读取 `jcy_insights.json` → 筛选 A 股推荐 → 批量执行牛市策略回测 → 横截面汇总

**关键逻辑：**
- `is_ashare_code(code)` — 过滤 A 股代码（沪深交易所）
- `BullStrategyAdapter` — 包装 `LuMACDBullStrategy`，将推荐日期前的信号清零（避免未来数据）
- `backtest_one()` — 单股回测，`eval_start=推荐日`，预热期不计入统计；返回 result dict
- 大盘指数只取一次，全部个股共用（原来每只票各拉一次）
- 预热 600 自然日：牛市过滤器要算大盘**月线** MACD，EMA-26 需要 26 根月线 ≈ 2 年

**默认止损止盈：** `--stop_loss 0.10`、`--take_profit` **默认关闭**（传 `none/off/空` 亦可关闭）。
本策略靠 `shrink_exit` 的动能衰减离场，再叠加固定比例止盈会把"最陡峭的那段"提前切断。
早先的 `stop_loss=0.20 / take_profit=0.10` 是 1:2 的反向盈亏比，与策略前提直接冲突。

**汇总输出（`batch_report.py`）：**
- `output/summary.csv` — 每股一行，按 **日均超额bp** 降序排。
  各股统计窗口从几十到几百个交易日不等，用总超额排序等于按"谁跑得久"排；
  `日均超额bp = 超额收益% × 100 ÷ 统计交易日数` 对窗口长度线性归一，跨标的才可比。
  不用年化：`(1+r)^(252/n)` 在 n 小时会几何放大（引擎的 `annual_return` 在 n<252 时返回 None 同理）
- `output/summary_portfolio.csv|.png` — **平均单股净值** vs 大盘。各股推荐日不同，采用
  "平均在场净值"：每只票从自己的统计起点归一为 1.0，按日历对齐后对当日已在场的取算术平均。
  ⚠️ 每只票都按满仓独立回测，N 只同时满仓在现实中不可能——这条曲线**不是可投资的组合净值**
- 控制台汇总：跑赢基准比例、日均超额与总超额各自的均值/**中位数**、交易成本占比、
  **按窗口长度分组**（<3月 / 3-6月 / 6-12月 / ≥1年）的日均超额，最好/最差各 5 只。
  收益分布右偏，均值会被个别翻倍股拉高；分组则用来识别"信号只在推荐后一两个月有效"这种衰减

**CLI：** `python backtest/jcy_macd_bull_batch.py [--output output/] [--take_profit 0.2]`

## 参数敏感性扫描 (`backtest/param_sweep.py`)

**职责：** 在推荐股票池上网格遍历两个参数轴，判断当前参数是"策略有效"还是"恰好挑中幸运点"。

- 可扫轴见 `AXES`：`expand_bars` / `cross_window` / `fast` / `slow` / `signal_period` / `shrink_exit` / `stop_loss` / `take_profit`。
  `stop_loss` / `take_profit` 的候选值含 `None`（=不启用），`None` 正是 `take_profit` 的默认值
- `DEFAULTS` 必须与 `jcy_macd_bull_batch.py` 的 CLI 默认值一致，否则热力图标出的"默认格"
  不是实际在跑的参数；`tests/test_param_sweep.py::test_defaults_match_batch_cli` 守住这一点
- 判读看**稳健性**不看最大值：整片偏绿=结论稳健；孤立亮点、邻居全负=过拟合
- 输出 `output/sweep/sweep_results.csv` + `sweep_heatmap.png`（默认参数格用金框标出）。
  热力图走 `build_matrix()` 而非 `df.pivot_table`：pivot 会把 `None` 当 NaN 丢掉整行整列，
  且按值排序会打乱 `True/False` 这类轴的语义顺序
- `--metric` 的合法取值收在 `METRICS` 并作为 argparse `choices`：指标名拼错要在启动时
  就退出，而不是跑完几十分钟的网格才在最后一步 KeyError
- 行情在进程内缓存，同一只票整个网格只拉一次

**样本外验证（`--oos-frac`）：** 网格本身是**纯样本内**的——在同一批数据上遍历再挑最好的，
只能说明参数高原平不平坦。`--oos-frac 0.3` 按推荐日把候选股切成两段（较早 70% 选参数，
最近 30% 只用于验证），切点落在日期边界上，保证同一天推荐的标的不跨界污染样本外。
选完参数后自动把 IS 最优格与默认格拿到 OOS 上重跑并给出判读：
IS 最优在 OOS 转负 = 选中的是噪声；默认格在 OOS 反而更好 = 最优格是拟合出来的。

**CLI：** `python backtest/param_sweep.py [--axis stop_loss take_profit] [--limit 20] [--oos-frac 0.3]`
