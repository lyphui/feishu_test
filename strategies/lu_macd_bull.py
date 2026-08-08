"""
卢麒元 MACD 牛市动能截取策略
=============================
对应原文："取每一个龙头股MACD最陡峭的部分"

核心逻辑：
  前提     ── 牛市过滤器（大盘月线 MACD 在 0 轴上方，且 DIF > DEA）
               只有牛市确认后，个股信号才生效

  买入     ── 日线金叉（DIF 上穿 DEA，0 轴上下均可）
               + 金叉后 cross_window 根内，红柱**连续** expand_bars 根拉长
               = 动能爆发确认

               注意：金叉当根 hist 必然由 ≤0 翻正，"本根 > 上根"恒成立，
               单看一根等于没有过滤（旧实现即如此，19 个金叉 19 个通过）。
               因此这里要求连续拉长，最早在金叉后第 1 根才可能触发。

  卖出     ── 红柱开始缩短（MACD 柱本根 < 上根，且 MACD > 0）
               即动能衰减，不等死叉，主动离场截取最陡段

  熊市保护 ── 牛市过滤器失效时，强制 signal=0，拒绝一切买入

与其他两个策略的关系：
  MACDStrategy      ── 教科书金叉/死叉，无位置/趋势过滤
  LuMACDStrategy    ── 严格三级底部确认，适合普通投资者长线建仓
  LuMACDBullStrategy── 牛市短炒截陡坡，高手战术，高频进出

输入要求：
  stock_df  ── 日线 DataFrame（DatetimeIndex，含 close）
  index_df  ── 大盘指数日线 DataFrame（DatetimeIndex，含 close）
               用于牛市判断，与 stock_df 索引可以不完全一致

输出（prepare 返回的 df 新增列）：
  DIF / DEA / MACD        个股日线指标
  DIF_IDX / DEA_IDX       大盘月线指标（对齐回日线）
  bull_market             bool，当前是否处于牛市
  hist_expanding          bool，红柱是否正在拉长（动能加速，单根口径）
  hist_expand_run         bool，红柱是否已连续 expand_bars 根拉长
  hist_shrinking          bool，红柱是否开始缩短（动能衰减）
  signal                  int，1=买入 / -1=卖出 / 0=观望
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class LuMACDBullStrategy(BaseStrategy):
    """
    Parameters
    ----------
    fast : int
        DIF 快线 EMA 周期，默认 12
    slow : int
        DIF 慢线 EMA 周期，默认 26
    signal_period : int
        DEA 信号线 EMA 周期，默认 9
    bull_fast : int
        大盘牛市判断用 DIF 快线，默认 12
    bull_slow : int
        大盘牛市判断用 DIF 慢线，默认 26
    bull_signal : int
        大盘牛市判断用 DEA 信号线，默认 9
    shrink_exit : bool
        True  = 红柱缩短即卖出（截陡坡，文中描述的高手做法）
        False = 等待死叉再卖出（保守版，减少假信号）
    expand_bars : int
        买入要求红柱连续拉长的最少根数，默认 2。
        设为 1 等价于无过滤（金叉当根必然"拉长"），仅用于复现旧行为。
    cross_window : int
        金叉后允许多少根 K 线内完成动能确认，默认 3。
        超过这个窗口才出现的连续红柱拉长不再视为该次金叉的买点。
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
        bull_fast: int = 12,
        bull_slow: int = 26,
        bull_signal: int = 9,
        shrink_exit: bool = True,
        expand_bars: int = 2,
        cross_window: int = 3,
        index_df: pd.DataFrame | None = None,
    ):
        if expand_bars < 1:
            raise ValueError("expand_bars 必须 >= 1")
        if cross_window < 1:
            raise ValueError("cross_window 必须 >= 1")
        self.fast          = fast
        self.slow          = slow
        self.signal_period = signal_period
        self.bull_fast     = bull_fast
        self.bull_slow     = bull_slow
        self.bull_signal   = bull_signal
        self.shrink_exit   = shrink_exit
        self.expand_bars   = expand_bars
        self.cross_window  = cross_window
        self._index_df     = index_df

    # ── 接口属性 ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"LuMACDBull({self.fast},{self.slow},{self.signal_period})"

    @property
    def params(self) -> dict:
        return {
            "fast":          self.fast,
            "slow":          self.slow,
            "signal_period": self.signal_period,
            "bull_fast":     self.bull_fast,
            "bull_slow":     self.bull_slow,
            "bull_signal":   self.bull_signal,
            "shrink_exit":   self.shrink_exit,
            "expand_bars":   self.expand_bars,
            "cross_window":  self.cross_window,
        }

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _calc_macd(
        self, close: pd.Series, fast: int, slow: int, sig: int
    ) -> pd.DataFrame:
        ema_fast  = self._ema(close, fast)
        ema_slow  = self._ema(close, slow)
        dif       = ema_fast - ema_slow
        dea       = self._ema(dif, sig)
        histogram = (dif - dea) * 2
        return pd.DataFrame({"DIF": dif, "DEA": dea, "MACD": histogram},
                            index=close.index)

    def _resample_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """月线重采样，索引 = 当月最后一个真实交易日（见 BaseStrategy._resample_period）。"""
        return self._resample_period(df, "ME", {"close": "last"})

    def _monthly_macd(self, index_df: pd.DataFrame) -> pd.DataFrame:
        """大盘月线 MACD，索引 = 月末交易日（该值当天收盘后才可知）。"""
        monthly = self._resample_monthly(index_df)
        return self._calc_macd(
            monthly["close"], self.bull_fast, self.bull_slow, self.bull_signal
        )

    def _bull_market_filter(self, index_df: pd.DataFrame) -> pd.Series:
        """
        大盘牛市判断（月线级别，粗粒度，避免频繁切换）：
          条件：大盘月线 DIF > 0  AND  DIF > DEA
          两者同时满足 → 牛市
          任一不满足   → 非牛市

        返回：bool Series，索引为**月末交易日**（不是月初！月初标签会让
        当月第一天就用上当月收盘价，构成未来函数）
        """
        macd = self._monthly_macd(index_df)
        return (macd["DIF"] > 0) & (macd["DIF"] > macd["DEA"])

    # ── 核心接口 ─────────────────────────────────────────────────────────────

    def prepare(
        self,
        df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df       : 个股日线 DataFrame（DatetimeIndex，含 close）
        index_df : 大盘指数日线 DataFrame（DatetimeIndex，含 close）
                   未传时自动使用构造函数中设置的 _index_df；
                   均为 None 时跳过牛市过滤，等价于永远处于牛市（不推荐）

        Returns
        -------
        附加所有指标列与信号列的 DataFrame
        """
        if index_df is None:
            index_df = self._index_df
        df = df.copy()

        # ── 1. 个股日线 MACD ─────────────────────────────────────────────────
        macd_daily = self._calc_macd(
            df["close"], self.fast, self.slow, self.signal_period
        )
        df[["DIF", "DEA", "MACD"]] = macd_daily

        # ── 2. 大盘牛市过滤器 ────────────────────────────────────────────────
        if index_df is not None:
            # 月线值的标签是月末交易日，_align_to_daily 从该日起前向填充：
            # 当月内的日子只会看到**上个月**已收盘的月线状态，无未来信息。
            bull_monthly = self._bull_market_filter(index_df)
            bull_daily   = self._align_to_daily(bull_monthly, df.index)
            bull_daily   = bull_daily.fillna(False).astype(bool)

            # 大盘月线 DIF/DEA 写入 df 供绘图与状态表参考
            idx_macd = self._monthly_macd(index_df)
            df["DIF_IDX"] = self._align_to_daily(idx_macd["DIF"], df.index)
            df["DEA_IDX"] = self._align_to_daily(idx_macd["DEA"], df.index)
        else:
            # 无大盘数据：降级，默认始终处于牛市（打印警告）
            import warnings
            warnings.warn(
                f"[{self.name}] 未传入 index_df，牛市过滤器已禁用。"
                " 建议传入大盘指数日线数据以避免熊市误操作。",
                UserWarning, stacklevel=2,
            )
            bull_daily = pd.Series(True, index=df.index)
            df["DIF_IDX"] = np.nan
            df["DEA_IDX"] = np.nan

        df["bull_market"] = bull_daily.astype(bool)

        # ── 3. 动能信号计算 ──────────────────────────────────────────────────
        hist = df["MACD"]
        dif  = df["DIF"]
        dea  = df["DEA"]

        # 红柱拉长（单根口径）：MACD > 0 且本根 > 上根（动能加速）
        df["hist_expanding"] = (hist > 0) & (hist > hist.shift(1))

        # 红柱缩短：MACD > 0 且本根 < 上根（动能衰减，离场信号）
        df["hist_shrinking"] = (hist > 0) & (hist < hist.shift(1))

        # 连续拉长：近 expand_bars 根**每一根**都在拉长。
        # 金叉当根 hist 由 ≤0 翻正，单根口径恒为 True，必须连续确认才有过滤力。
        expand_run = df["hist_expanding"].copy()
        for k in range(1, self.expand_bars):
            expand_run &= df["hist_expanding"].shift(k).fillna(False).astype(bool)
        df["hist_expand_run"] = expand_run

        # 金叉：DIF 上穿 DEA（0轴上下均可，牛市战术）
        golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))

        # 金叉是否发生在最近 cross_window 根内（含当根）
        cross_recent = (
            golden_cross.rolling(self.cross_window, min_periods=1)
            .max().fillna(0).astype(bool)
        )

        # 死叉（备用卖出，shrink_exit=False 时使用）
        death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))

        # ── 4. 信号合成 ──────────────────────────────────────────────────────
        #
        # 买入条件：
        #   牛市过滤通过
        #   AND 最近 cross_window 根内出现过日线金叉
        #   AND 红柱已连续 expand_bars 根拉长（动能确认，非金叉当根的恒真条件）
        #
        # 卖出条件（两种模式）：
        #   shrink_exit=True  → 红柱开始缩短即卖（截最陡段，高手模式）
        #   shrink_exit=False → 等死叉再卖（保守模式）
        #
        df["signal"] = 0

        buy_condition = (
            cross_recent
            & df["bull_market"]
            & df["hist_expand_run"]
        )
        df.loc[buy_condition, "signal"] = 1

        # 卖出
        if self.shrink_exit:
            sell_condition = df["hist_shrinking"] & df["bull_market"]
        else:
            sell_condition = death_cross

        # 熊市强制平仓：买入已要求 bull_market，故熊市日不可能是买点，无需再排除
        bear_exit = ~df["bull_market"]

        df.loc[sell_condition, "signal"] = -1
        df.loc[bear_exit,      "signal"] = -1

        return df.dropna(subset=["DIF", "DEA", "MACD"])

    # ── 绘图接口 ─────────────────────────────────────────────────────────────

    def plot_indicators(self, ax, df: pd.DataFrame, colors: dict) -> None:
        """
        绘制：
          - 个股日线 MACD 柱（红柱拉长段高亮）
          - DIF / DEA 曲线
          - 大盘月线 DIF/DEA 参考线
          - 牛市/熊市背景色
          - 买卖信号标注
        """
        c_bg    = colors["bg"]
        c_fg    = colors["fg"]
        c_green = colors["green"]
        c_red   = colors["red"]
        c_blue  = colors["blue"]
        c_gold  = colors["gold"]
        c_muted = colors["muted"]

        # ── 牛市背景 ─────────────────────────────────────────────────────────
        if "bull_market" in df.columns:
            bull_on = False
            bull_start = None
            for date, row in df.iterrows():
                is_bull = row["bull_market"]
                if is_bull and not bull_on:
                    bull_start = date
                    bull_on    = True
                elif not is_bull and bull_on:
                    ax.axvspan(bull_start, date, alpha=0.06,
                               color=c_green, lw=0)
                    bull_on = False
            if bull_on and bull_start is not None:
                ax.axvspan(bull_start, df.index[-1], alpha=0.06,
                           color=c_green, lw=0)

        # ── MACD 柱（红柱拉长段加深显示） ────────────────────────────────────
        # matplotlib bar 不支持逐根 alpha，改用分组绘制
        bar_colors     = np.where(df["MACD"] >= 0, c_red, c_green)
        expanding_mask = df.get("hist_expanding", pd.Series(False, index=df.index))
        ax.bar(df.index[~expanding_mask],
               df["MACD"][~expanding_mask],
               color=bar_colors[~expanding_mask.values], alpha=0.4,
               width=1, label="MACD柱")
        ax.bar(df.index[expanding_mask],
               df["MACD"][expanding_mask],
               color=bar_colors[expanding_mask.values], alpha=0.9,
               width=1, label="MACD柱(动能↑)")

        # ── DIF / DEA ────────────────────────────────────────────────────────
        ax.plot(df.index, df["DIF"], color=c_blue, lw=1.2,
                label=f"DIF({self.fast},{self.slow})")
        ax.plot(df.index, df["DEA"], color=c_gold, lw=1.2,
                label=f"DEA({self.signal_period})")

        # ── 大盘月线 DIF/DEA 参考线 ──────────────────────────────────────────
        if "DIF_IDX" in df.columns and df["DIF_IDX"].notna().any():
            ax.plot(df.index, df["DIF_IDX"], color=c_blue, lw=0.7,
                    linestyle=":", alpha=0.45, label="DIF_大盘月线")
            ax.plot(df.index, df["DEA_IDX"], color=c_gold, lw=0.7,
                    linestyle=":", alpha=0.45, label="DEA_大盘月线")

        # ── 0 轴 ─────────────────────────────────────────────────────────────
        ax.axhline(0, color=c_muted, lw=0.6, linestyle="--")

        # ── 买卖信号标注 ──────────────────────────────────────────────────────
        buy_dates  = df.index[df["signal"] == 1]
        sell_dates = df.index[df["signal"] == -1]
        if len(buy_dates):
            ax.scatter(buy_dates, df.loc[buy_dates, "DIF"],
                       marker="^", color=c_green, s=70, zorder=5,
                       label="买入(动能起点)")
        if len(sell_dates):
            ax.scatter(sell_dates, df.loc[sell_dates, "DIF"],
                       marker="v", color=c_red, s=70, zorder=5,
                       label="卖出(动能衰减)")

        ax.legend(facecolor=c_bg, labelcolor=c_fg, edgecolor=c_muted,
                  fontsize=7, ncol=4, loc="upper left")
        ax.set_ylabel(self.name, color=c_fg, fontsize=9)
