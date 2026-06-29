"""
卢麒元 MACD 牛市动能截取策略 —— 通用策略适配器
================================================
BullStrategyAdapter: 将双参数 prepare(df, index_df) 适配为单参数接口。

报告输出（plot_bull_backtest / export_bull_daily_status）已移至 backtest/bull_report.py。
"""

import pandas as pd

from strategies import LuMACDBullStrategy
from strategies.base import BaseStrategy


# ── 策略适配器 ────────────────────────────────────────────────────────────────

class BullStrategyAdapter(BaseStrategy):
    """
    将 LuMACDBullStrategy.prepare(df, index_df) 适配为 run_backtest 所需的
    单参数 prepare(df) 接口。

    trade_start_date : str "YYYYMMDD" or None
        若指定，该日期之前的买入 / 卖出信号全部清零（用于 MACD 预热期）。
    """

    def __init__(self, inner: LuMACDBullStrategy, index_df,
                 trade_start_date: str | None = None):
        self._inner            = inner
        self._index_df         = index_df
        self._trade_start_date = trade_start_date

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def params(self) -> dict:
        return self._inner.params

    def prepare(self, df):
        result = self._inner.prepare(df, self._index_df)
        if self._trade_start_date:
            cutoff = pd.to_datetime(self._trade_start_date, format="%Y%m%d")
            result.loc[result.index < cutoff, "signal"] = 0

        # 第一次操作必须是买入：清除首次买入信号之前的所有卖出信号
        buy_indices = result.index[result["signal"] == 1]
        if len(buy_indices) > 0:
            first_buy = buy_indices[0]
            result.loc[(result.index < first_buy) & (result["signal"] == -1), "signal"] = 0
        else:
            result.loc[result["signal"] == -1, "signal"] = 0

        return result

    def plot_indicators(self, ax, df, colors):
        return self._inner.plot_indicators(ax, df, colors)
