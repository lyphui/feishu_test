# backtest/ 与 strategies/ 目录结构重排

把 `backtest/` 顶层「库模块 + CLI 脚本一锅烩」的混乱整理干净，并将策略体系并入回测目录。

## 目录变化

- **CLI 入口全部移入 `backtest/scripts/`**：`macd_analysis.py`、`lu_macd_analysis.py`、
  `lu_macd_bull_analysis.py`、`exec_bench.py`、`fatfinger_bench.py`、`hk_oil_etf_signal.py`、
  `jcy_intraday_timing.py`、`jcy_macd_bull_batch.py`、`ma_cross_bench.py`、`oil_track.py`、
  `param_sweep.py`、`stock_playbook.py` 共 12 个入口脚本从 `backtest/` 顶层迁入。
- **`strategies/` 整体并入 `backtest/strategies/`**：`__init__.py` / `base.py` / `macd.py` /
  `lu_macd.py` / `lu_macd_bull.py` / `ma_cross.py`，`from strategies import` 路径不变。
- **库层不动**：`backtest/engine.py`、`config.py`、`report.py`、`batch_report.py`、
  `bull_report.py` 仍留在顶层裸导入；`backtest/lib/`、`backtest/presets/` 原样保留。

## 行为变更

运行命令由 `python backtest/x.py` 变为 `python backtest/scripts/x.py`；
库层与 `strategies` / `lib` 的 import 方式不变，回测逻辑与输出数值不变。

## 配套改动

- 12 个入口脚本的 `sys.path` bootstrap 统一为三级（`backtest/scripts/` + `backtest/` + 仓库根）。
- `tests/conftest.py` 追加 `backtest/scripts/` 到 path。
- CLAUDE.md、docs/、.claude/settings.local.json 的路径引用同步更新。
