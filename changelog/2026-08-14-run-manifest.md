# [已发布] — 2026-08-14（run.json 可复现清单）

评审 `docs/backtest-review.md` 项 2 落地。`output/` 下此前只有结果，
没有「谁跑的、用什么参数、数据截止到哪天」——重跑对不上时无从核查。

## 改动

新增 `backtest/lib/manifest.py` 的 `write_run_manifest()`，每个批量脚本
在输出目录落一份 `run.json`：

- git sha + 是否 dirty（git 不可用时降级为 null，不阻断输出）
- 完整 CLI argv
- `costs` 常量快照（含本次收编的 rf / cash_rate）
- 各标的 `price_store` meta 的 `data_end` / `rows`
- 耗时

接线：`backtest_jcy_pool`（逐池）、`sweep_params`、`backtest_jcy_intraday`、
`compare_playbooks`、`compare_ma_cross`、`track_oil`。

测试：`tests/test_manifest.py`（字段齐全、git/meta 缺失降级）。

## 行为变更

否（纯新增输出文件）。
