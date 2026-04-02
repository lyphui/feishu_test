"""
卢麒元 MACD 牛市动能截取策略
=============================
对应原文："取每一个龙头股MACD最陡峭的部分"

核心逻辑：
  前提     ── 牛市过滤器（大盘月线 MACD 在 0 轴上方，且 DIF > DEA）
               只有牛市确认后，个股信号才生效

  买入     ── 日线金叉（DIF 上穿 DEA，0 轴上下均可）
               + 红柱开始拉长（MACD 柱本根 > 上根，且 MACD > 0）
               = 动能爆发起点

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
  hist_expanding          bool，红柱是否正在拉长（动能加速）
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
        index_df: pd.DataFrame | None = None,
    ):
        self.fast          = fast
        self.slow          = slow
        self.signal_period = signal_period
        self.bull_fast     = bull_fast
        self.bull_slow     = bull_slow
        self.bull_signal   = bull_signal
        self.shrink_exit   = shrink_exit
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

    @staticmethod
    def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
        return df.resample("MS").agg({"close": "last"}).dropna()

    def _bull_market_filter(self, index_df: pd.DataFrame) -> pd.Series:
        """
        大盘牛市判断（月线级别，粗粒度，避免频繁切换）：
          条件：大盘月线 DIF > 0  AND  DIF > DEA
          两者同时满足 → 牛市
          任一不满足   → 非牛市

        返回：bool Series，索引为月线日期（月初）
        """
        monthly = self._resample_monthly(index_df)
        macd    = self._calc_macd(
            monthly["close"], self.bull_fast, self.bull_slow, self.bull_signal
        )
        bull = (macd["DIF"] > 0) & (macd["DIF"] > macd["DEA"])
        return bull

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
            bull_monthly = self._bull_market_filter(index_df)
            # 月线信号对齐到日线（ffill，月初信号持续到下月初）
            bull_daily = bull_monthly.reindex(df.index).ffill().fillna(False)

            # 大盘月线 DIF/DEA 写入 df 供绘图参考
            idx_monthly = self._resample_monthly(index_df)
            idx_macd    = self._calc_macd(
                idx_monthly["close"], self.bull_fast, self.bull_slow, self.bull_signal
            )
            df["DIF_IDX"] = idx_macd["DIF"].reindex(df.index).ffill()
            df["DEA_IDX"] = idx_macd["DEA"].reindex(df.index).ffill()
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

        # 红柱拉长：MACD > 0 且本根 > 上根（动能加速）
        df["hist_expanding"] = (hist > 0) & (hist > hist.shift(1))

        # 红柱缩短：MACD > 0 且本根 < 上根（动能衰减，离场信号）
        df["hist_shrinking"] = (hist > 0) & (hist < hist.shift(1))

        # 金叉：DIF 上穿 DEA（0轴上下均可，牛市战术）
        golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))

        # 死叉（备用卖出，shrink_exit=False 时使用）
        death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))

        # ── 4. 信号合成 ──────────────────────────────────────────────────────
        #
        # 买入条件：
        #   牛市过滤通过
        #   AND 日线金叉
        #   AND 红柱开始拉长（金叉当根或随后确认）
        #
        # 卖出条件（两种模式）：
        #   shrink_exit=True  → 红柱开始缩短即卖（截最陡段，高手模式）
        #   shrink_exit=False → 等死叉再卖（保守模式）
        #
        df["signal"] = 0

        # 买入：金叉 + 牛市 + 红柱拉长（允许金叉后1-3根内确认）
        buy_condition = (
            golden_cross
            & df["bull_market"]
            & df["hist_expanding"]
        )
        df.loc[buy_condition, "signal"] = 1

        # 卖出
        if self.shrink_exit:
            sell_condition = df["hist_shrinking"] & df["bull_market"]
        else:
            sell_condition = death_cross

        # 熊市强制平仓（无论 shrink_exit 设置）
        bear_exit = ~df["bull_market"] & (df["signal"] != 1)

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
        bar_colors = np.where(df["MACD"] >= 0, c_red, c_green)
        # 拉长段加深
        if "hist_expanding" in df.columns:
            bar_alpha = np.where(df["hist_expanding"], 0.9, 0.4)
        else:
            bar_alpha = 0.6
        # matplotlib bar 不支持逐根 alpha，改用分组绘制
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
