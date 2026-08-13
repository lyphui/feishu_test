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

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        """指数移动平均线（EMA），所有 MACD 策略共用。"""
        return series.ewm(span=period, adjust=False).mean()

    # ── 多周期对齐工具（周线/月线 → 日线） ──────────────────────────────────
    #
    # 这两个方法专门用来消除自制回测里最常见的两类错误，所有需要跨周期的策略
    # 都必须走它们，不要再自己写 resample + reindex：
    #
    #   1. 未来函数：pandas 的 "MS" 规则把 K 线标签打在**月初**，值却由**月末**
    #      收盘算出。直接 ffill 回日线，等于让月初那天就知道了整月的走势。
    #      _resample_period 改用区间内最后一个**真实交易日**作为标签，
    #      标签日 = 该根 K 线收盘日 = 信息可用的第一天。
    #
    #   2. 静默丢数据：Series.reindex(daily_index) 只保留目标索引里存在的日期，
    #      低频标签落在非交易日时整根 K 线直接消失。_resample_period 的标签
    #      取自日线索引本身，天然不会落空；_align_to_daily 再用并集兜底，
    #      即便调用方传入外部序列（如大盘指数，交易日与个股不完全一致）也不丢。

    @staticmethod
    def _resample_period(df: pd.DataFrame, rule: str, agg: dict,
                         drop_incomplete: bool = True) -> pd.DataFrame:
        """
        把日线 df 重采样到低频（"W-FRI" 周线 / "ME" 月线），并以该区间内
        **最后一个真实交易日**作为索引标签。

        标签语义 = "这根 K 线在哪天收盘"，因此对齐回日线时，标签日当天使用
        该值不构成未来函数（当日收盘后即可知），配合引擎的 signal.shift(1)
        在 T+1 开盘成交，时序完全自洽。

        drop_incomplete=True（默认）会丢掉末尾**尚未走完**的那根 K 线：
        数据截止到周三时，本周的周线还没收盘，把它当成一根完整周线会让信号
        随后续数据反复变脸（周三报死叉、周五又撤销）。只用已收盘的 K 线，
        回测与实盘看到的东西才是同一个。

        已知边界：区间最后一个自然日恰逢休市（如月末是周日、周五放假）时，
        最新那根 K 线要等到下一区间开始才会被认定收盘，最多滞后一个区间。
        只影响"当前进行中"的最新一根，历史区间不受影响。
        """
        if df.empty:
            return df.iloc[0:0]
        out = df.resample(rule).agg(agg)
        # 每个区间内最后一个真实交易日（可能为 NaT：该区间没有任何交易日）
        asof = df.index.to_series().resample(rule).last()
        out = out.assign(_asof=asof).dropna(subset=["_asof"])
        out.index = pd.DatetimeIndex(out.pop("_asof"))
        out = out.dropna(how="all")

        if drop_incomplete and len(out):
            last_day = df.index[-1]
            # 包含 last_day 的那个区间在自然日历上的结束时点
            period_end = pd.tseries.frequencies.to_offset(rule).rollforward(last_day)
            if period_end > last_day:
                out = out.iloc[:-1]
        return out

    @staticmethod
    def _align_to_daily(series: pd.Series,
                        daily_index: pd.DatetimeIndex) -> pd.Series:
        """
        把低频序列前向填充对齐到日线索引，不丢任何一根低频 K 线。

        与 `series.reindex(daily_index).ffill()` 的区别：先在**并集**索引上
        ffill，再收敛回日线。这样即使低频标签不在日线索引里（例如大盘指数
        停市而个股交易），它的值也会正确地传播给之后的每一个交易日。

        要求 series 的标签是区间收盘日（用 _resample_period 生成），
        本方法不做任何时间平移。返回值可能含前导 NaN（首根低频 K 线之前）。
        """
        if series.empty:
            return pd.Series(index=daily_index, dtype="float64")
        combined = series.reindex(series.index.union(daily_index)).ffill()
        return combined.reindex(daily_index)

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
