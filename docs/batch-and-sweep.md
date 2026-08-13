# 批量回测与参数敏感性扫描

> 从 [CLAUDE.md](../CLAUDE.md) 拆出。

## 批量回测 (`backtest/scripts/backtest_jcy_pool.py` + `backtest/reports/batch_report.py`)

**职责：** 读取 `jcy_insights.json` → 筛选 A 股推荐 → 批量执行牛市策略回测 → 横截面汇总

**关键逻辑：**
- `is_ashare_code(code)` — 过滤 A 股代码（沪深交易所）
- `load_candidates(json, ratings=LONG_RATINGS)` — 候选池默认收 **买入 + 增持**
  （`LONG_RATINGS`）。只收「增持」等于用中间档定义股票池、把 51 只 Strong Buy 丢在回测外。
  候选 dict 带 `rating`；去重保留**首次落入所选评级集合**的最早记录
- `BullStrategyAdapter` — 包装 `LuMACDBullStrategy`，将推荐日期前的信号清零（避免未来数据）
- `backtest_one()` — 单股回测，`eval_start=推荐日`，预热期不计入统计；返回 result dict
- 个股预热 600 自然日；**大盘指数只取一次、全池共用，起点是绝对日期
  `config.INDEX_HISTORY_START`**（不是「最早推荐日 − 600 天」）。月线 EMA(26) 要几十根月线
  才收敛，按候选池取会让「加一篇更早的文章 → 所有个股 `bull_market` 改写」，回测不可复现。
  四个调用方（batch / sweep / backtest_jcy_intraday / backtest_lu_macd_bull）统一走 `index_history_start()`

**看空对照组（`--control`）：** `--control 减持,回避` 用**完全相同的策略与参数**把看空池
再跑一遍，输出到 `output/control/` 并写 `summary_rating_compare.csv`。
只跑看多池答不了"这套评级有没有区分度"——牛市里随便挑一篮子都涨。
表里明确标注这是**描述性对比、不是显著性检验**：两池样本量与推荐日分布都不同，
且所有标的共享同一段行情。

**默认止损止盈：** `--stop_loss 0.10`、`--take_profit` **默认关闭**（传 `none/off/空` 亦可关闭）。
本策略靠 `shrink_exit` 的动能衰减离场，再叠加固定比例止盈会把"最陡峭的那段"提前切断。
早先的 `stop_loss=0.20 / take_profit=0.10` 是 1:2 的反向盈亏比，与策略前提直接冲突。

**汇总输出（`batch_report.py`）：**
- `output/summary.csv` — 每股一行，按 **日均超额bp** 降序排。
  各股统计窗口从几十到几百个交易日不等，用总超额排序等于按"谁跑得久"排；
  `日均超额bp = 超额收益% × 100 ÷ 统计交易日数` 对窗口长度线性归一，跨标的才可比。
  不用年化：`(1+r)^(252/n)` 在 n 小时会几何放大（引擎的 `annual_return` 在 n<252 时返回 None 同理）
- **两个 alpha 必须分开看**（同一张表回答的是两个不同问题）：

  | 列 | 算式 | 回答 |
  |---|---|---|
  | `选股alpha%` | `基准收益% − 指数收益%` | 研报**推荐**本身值不值钱 |
  | `超额收益%` | `策略收益% − 基准收益%` | MACD **择时**加不加分 |

  项目宣称目标"验证推荐的实际收益"问的是第一个，而引擎默认的 `benchmark_return` 是第二个。
  只看超额收益，会把"推荐了好票、择时拖后腿"读成策略失败。
  `result_to_row(cand, result, index_df=...)` 传了指数才算得出这两列
- **`在场比例%` / `平均持仓天数`** — 来自引擎的 `exposure_pct` / `avg_holding_days`。
  `shrink_exit=True` 大部分时间空仓（缓存标的实测**仅 2.5%~3.1%**），空仓日既不赚也不亏，
  **表里的回撤与夏普一律未按暴露度调整**，必须对着在场比例读
- `output/summary_portfolio.csv|.png` — **平均单股净值** vs 大盘。各股推荐日不同，采用
  "平均在场净值"：每只票从自己的统计起点归一为 1.0，按日历对齐后对当日已在场的取算术平均。
  ⚠️ 每只票都按满仓独立回测，N 只同时满仓在现实中不可能——这条曲线**不是可投资的组合净值**
- 控制台汇总：跑赢基准比例、日均超额与总超额各自的均值/**中位数**、交易成本占比、
  **按窗口长度分组**（<3月 / 3-6月 / 6-12月 / ≥1年）的日均超额，最好/最差各 5 只。
  收益分布右偏，均值会被个别翻倍股拉高；分组则用来识别"信号只在推荐后一两个月有效"这种衰减

**CLI：** `python -m backtest.scripts.backtest_jcy_pool [--ratings 买入,增持] [--control 减持,回避] [--output output/] [--take_profit 0.2]`

## 参数敏感性扫描 (`backtest/scripts/sweep_params.py`)

**职责：** 在一个股票池上网格遍历两个参数轴，判断当前参数是"策略有效"还是"恰好挑中幸运点"。

**股票池是可注入的**：扫描机制（网格、横截面聚合、样本外切分、热力图）与"票从哪来"之间
只有 `resolve_universe()` 一个接口，它把任何来源归一成 `[{"code", "date"}]`（`date` = 该票的
回测起点，JCY 池里就是推荐日）。默认走 JCY 推荐池，`--codes 601857 600938 --codes-start 20180101`
即可给油气池做同样的过拟合检验。以前 `main()` 直接读 `jcy_insights.json`，这个通用工具因此
只能扫一个池子。注意 `--codes` 池所有票共用同一起点，按推荐日的时序留出无从谈起，
`--oos-frac` 会自动退化为纯样本内并给出提示。

- 可扫轴见 `AXES`：`expand_bars` / `cross_window` / `fast` / `slow` / `signal_period` / `shrink_exit` / `stop_loss` / `take_profit`。
  `stop_loss` / `take_profit` 的候选值含 `None`（=不启用），`None` 正是 `take_profit` 的默认值
- `DEFAULTS` 必须与 `backtest/scripts/backtest_jcy_pool.py` 的 CLI 默认值一致，否则热力图标出的"默认格"
  不是实际在跑的参数；`tests/test_param_sweep.py::test_defaults_match_batch_cli` 守住这一点
- 判读看**稳健性**不看最大值：整片偏绿=结论稳健；孤立亮点、邻居全负=过拟合
- 输出 `output/sweep/sweep_results.csv` + `sweep_heatmap.png`（默认参数格用金框标出）。
  热力图走 `build_matrix()` 而非 `df.pivot_table`：pivot 会把 `None` 当 NaN 丢掉整行整列，
  且按值排序会打乱 `True/False` 这类轴的语义顺序
- `--metric` 的合法取值收在 `METRICS` 并作为 argparse `choices`：指标名拼错要在启动时
  就退出，而不是跑完几十分钟的网格才在最后一步 KeyError
- **主指标是 `日均超额中位bp`，不是 `超额中位数%`。** `batch_report` 已论证总超额% 跨窗口
  不可比，扫描却一度用它选参数——选出来的是"哪组参数恰好被长窗口标的占了多数"。
  总超额% 保留在 `METRICS` 里仅供对照
- `--ratings` 与批量回测同义，扫描的候选池默认同为 买入+增持
- 行情在进程内缓存，同一只票整个网格只拉一次

**样本外验证（`--oos-frac`）：** 网格本身是**纯样本内**的——在同一批数据上遍历再挑最好的，
只能说明参数高原平不平坦。`--oos-frac 0.3` 按推荐日把候选股切成两段（较早 70% 选参数，
最近 30% 只用于验证），切点落在日期边界上，保证同一天推荐的标的不跨界污染样本外。
选完参数后自动把 IS 最优格与默认格拿到 OOS 上重跑并给出判读：
IS 最优在 OOS 转负 = 选中的是噪声；默认格在 OOS 反而更好 = 最优格是拟合出来的。

**CLI：** `python -m backtest.scripts.sweep_params [--axis stop_loss take_profit] [--limit 20] [--oos-frac 0.3] [--codes CODE...] [--codes-start YYYYMMDD]`
