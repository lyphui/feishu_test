"""
策略基类
========
所有策略必须继承 BaseStrategy 并实现以下接口：

  prepare(df)            → 在 OHLCV df 上计算指标并生成 signal 列，返回新 df
  plot_indicators(ax, df, colors) → 在给定 ax 上绘制策略专属指标图
  name                   → 策略名称字符串（用于图表标题）
  params                 → 策略参数字典（用于展示）
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称，显示在图表标题中。"""

    @property
    @abstractmethod
    def params(self) -> dict:
        """策略参数字典，供展示和日志使用。"""

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在原始行情 df（含 open/high/low/close/volume 列）基础上
        计算技术指标，并追加 signal 列：
            1  → 买入信号
           -1  → 卖出信号
            0  → 观望
        返回处理后的 DataFrame（已 dropna）。
        """

    @abstractmethod
    def plot_indicators(self, ax, df: pd.DataFrame, colors: dict) -> None:
        """
        在给定 Axes 上绘制策略专属指标（图表第二子图）。

        colors 字典包含以下键：
            bg, fg, green, red, blue, gold, muted
        """
