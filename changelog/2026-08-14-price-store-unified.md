# [已发布] — 2026-08-14（取数统一走 price_store，--offline 全线生效）

评审 `docs/backtest-review.md` 项 1 落地。此前两条取数路径并存：
`price_store.load_daily`（本地仓库 + 重叠对账）与 `market_data.fetch_*`
（直连网络），而**规模最大的两个批量任务在直连那一侧**——jcy 池 248 只
标的每次重下、sweep 是 8 轴网格 × N 只，慢、无 `--offline`、且「同一条
命令两周后重跑得到不同数字」。

## 改动

- `backtest_jcy_pool.py`：个股经 `load_daily` 注入 `df=`、指数走
  `kind="index"`；parser 接入 `cli.base_parser()`，获得 `--offline`。
- `sweep_params.py`：`_cached_stock` 与指数取数改走 `load_daily`；
  接入 `base_parser()` + `--offline`。
- `backtest_jcy_intraday.py`：日线与指数改走 `load_daily`；接入
  `base_parser()` + `--offline`。
- `backtest_lu_macd_bull.py`（.ini 驱动）：指数与个股改走 `load_daily` 注入。
- `compare_playbooks.py` / `compare_ma_cross.py`：手写 `--offline` 等参数
  收编进 `base_parser()`（行为不变，选项定义只留一处）。
- `engine.run_backtest` 默认取数**保持** `fetch_stock_data`——库不该假设
  有本地仓库，由调用方注入 `df=`（评审明确建议）。
- 新增 `backtest/scripts/_ab_price_source.py`：新旧路径 A/B 对比脚本，
  供以后复核。

## A/B 验证（切换前置，评审要求的「会静默改数」检查）

**初版 A/B 的结论（12/12 一致）是无效的**，两个设计缺陷让它测不出目标问题：

1. 样本取 jcy 池前 10 只，而那批缓存正是跑 A/B 时刚建的——比的是
   「直连 akshare」vs「刚由同一次 akshare 调用写进仓库的文件」，必然一致；
2. `load_daily` 默认 `auto_update=True`，比对前先联网把缓存刷了一遍，
   于是又成了新数据比新数据。

重做后（`_ab_price_source.py` 改为按 `meta.updated_at` 抽**最老的 N 份存量
缓存**、且一律 `read_daily` 不刷新）：**16 份里有 2 份不一致**——
`601225`、`600547` 的缓存与直连结果在**收益率序列**上就对不上
（日收益最大绝对差 2.5e-2 / 4.7e-3；两者比值在 0.997~1.338 之间浮动，
不是恒定基准比，所以不是"同一套数据换个基准"）。

根因：仓库里的文件由不同数据源写入，而 akshare 与 baostock 的 hfq
处理并不一致。这两只票的缓存写于 2026-08-10（akshare 可用时段），
其余 14 份写于 08-11（akshare 被代理阻断、回退 baostock）。

## 遗留风险（未在本次解决）

本地仓库目前是**源异构**的：同一个 `data/market/daily/` 里混着 akshare 与
baostock 两种 hfq。这不只影响"切不切取数路径"，它意味着**同一批回测里
不同标的的价格口径可能不同**。

- 已做：`meta.json` 新增 `source` 字段（评审项 6），此后写入的文件都可追溯；
  存量文件的 `source` 为空，`_ab_price_source.py` 会优先抽这一批。
- 未做：按单一首选源重建全仓库。那会重写 106 份缓存并改变已有回测数值，
  属于需要单独决策与单独 changelog 的动作。

## 行为变更

否——**但仅限于本次未触碰数据的前提下**。代码层面只改取数路径，命令行
新增 `--offline`，默认行为（在线增量补齐）不变。上面那 2 只票的差异是
仓库**既有**的源异构，不是本次改动引入的；真正消除它需要另做一次重建。

注意：`--offline` 模式下本地无缓存的标的会抛 `FileNotFoundError`，
在 jcy_pool 里走既有失败计数路径——offline 之前需先在线跑过一次。
