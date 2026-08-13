# backtest 重构收尾：包化、脚本改名、去重、分层

上一轮目录重排只解决了「文件在哪」，本轮把约定做完整：导入约定成立、
真重复消除、验证方式能真正抓到回归。

## 改动

- **包化**：`backtest/` 成为正式包（新增空的 `__init__.py`——文件内注释说明
  为什么必须留空：任何便利 re-export 都会让 `import backtest.engine` 重新拖进
  matplotlib）。全库导入改为 `backtest.*` 绝对导入，删除 12 段 sys.path
  bootstrap 与 `tests/conftest.py` 的 path 注入。运行方式统一为
  `python -m backtest.scripts.x`（放弃 `python backtest/scripts/x.py` 直接
  调用——它的导入顺序脆弱性正是上一轮两个坏脚本的根因）。
- **脚本改名**：12 个 CLI 按动词前缀归类——`backtest_`（跑策略）、
  `compare_`（多方案同表对比）、`sweep_`（参数扫描）、`track_`（增量跟踪），
  `ls backtest/scripts/` 直接按类聚合。用 `git mv` 保留 rename 可追溯。
- **缓存去重**：新增 `lib/store_base.py`，承载「头尾缺口 + 重叠对账 + 容差
  不符则重建」算法；`price_store` / `intraday_store` 两套逐函数同构的实现
  改为转调，公开 API 不变。`lib/oil_price.py` 是有意的简化变体（全表覆盖），
  docstring 已注明，不并。
- **拆大脚本**：`PositionTracker`（买卖状态机、T+1、佣金印花税）从
  `backtest_jcy_intraday` 拆为 `lib/position_tracker.py`；三面板绘图拆为
  `reports/intraday_report.py`，`backtest_lu_macd` 的绘图与 CSV 导出拆为
  `reports/lu_macd_report.py`。`test_position_timing` 等测试改为直接导入库
  模块，去掉 `importorskip` 进 CLI 脚本的写法。
- **分层落地**：一切出图进 `reports/`（report / batch_report / bull_report /
  plotting 移入）；`bull_backtest.py` 因 import 具体策略类从 `lib/` 移入
  `strategies/`。判据写进 CLAUDE.md：import matplotlib 的进 `reports/`，
  import 具体策略类的进 `strategies/`，其余计算与数据工具进 `lib/`。
- **清理与收编**：删除全库零调用的 `EXECUTION_INI_BLOCK`、
  `BacktestConfig.get_str()`、`week_end_flags()` 及 `simulate(freq="week")`
  分支；3 份表格格式化收进 `lib/console.py`；重复的 argparse 选项用
  `lib/cli.py` 的 parent parser 统一（`--offline` 等行为拉齐）；油气代码表
  集中到 `lib/oil_price.py`，A 股代码改名 `OIL_STOCKS` 与商品代码
  `OIL_SYMBOLS` 区分开（此前同名不同义）。
- **验证强化**：仓库根新增 `pytest.ini`（`pythonpath = .`），修掉裸 `pytest`
  无法解析 `jcy` 的问题；新增 `tests/test_scripts_runnable.py`，用子进程
  真正启动 `backtest/scripts/` 下每一个脚本（glob，不写死清单），
  断言无 `ModuleNotFoundError`——`import` 式冒烟测不出 path 顺序 bug。

## 行为变更

**否。** 回测逻辑与输出数值不变；全量测试 343 passed、零意外 skip。
唯一的数值行为变更（涨跌停/停牌成交判定收紧）单列在
[涨跌停/停牌判定统一](2026-08-13-tradability-unified.md)。
