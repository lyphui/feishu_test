"""
卢麒元 MACD 三级确认策略
========================
核心逻辑（严格还原文中规则）：

  Level-1  月线  ── 历史性大底确认
                     白线(DIF) + 黄线(DEA) 均在 0 轴以下
                     + 白线上穿黄线（金叉）
                     + 最好处于绿柱（MACD<0）之下

  Level-2  周线  ── 中期底部确认 + 建仓起点
                     同样条件：0轴下金叉 + 绿柱
                     + 成交量开始温和放大

  Level-3  日线  ── 执行层，记录 phase（鞋底/脚面/小腿/腰部/头部）
                     仅在 Level-1 & Level-2 均已确认后发出 signal=1

  卖出（OR，任一满足即触发）：
    - phase 达到 sell_phase 阈值（默认"腰部"，即底部价格 +60%）
    - 周线死叉（DIF 下穿 DEA，与买入级别对称）

指标公式：
    DIF      = EMA(close, fast) - EMA(close, slow)
    DEA      = EMA(DIF, signal_period)
    MACD柱   = (DIF - DEA) × 2

输入要求：
    df 必须是 **日线** DataFrame，索引为 DatetimeIndex，
    至少包含列：close
    可选列：volume（缺失时降级运行，不做量能验证，会打印警告）

输出（prepare 返回的 df 新增列）：
    DIF / DEA / MACD          日线指标
    DIF_W / DEA_W / MACD_W   周线指标（对齐回日线）
    DIF_M / DEA_M / MACD_M   月线指标（对齐回日线）
    vol_expanding             bool，周线成交量是否较前期放大
    monthly_confirmed         bool，月线底部信号是否已触发（持续为 True）
    weekly_confirmed          bool，周线底部信号是否已触发（持续为 True）
    phase                     str，当前价格所处身体部位
    weekly_death              bool，当日是否触发周线死叉
    phase_exit                bool，当日 phase 是否首次达到 sell_phase
    signal                    int，1=买入 / -1=卖出 / 0=观望
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .base import BaseStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Phase 阈值（基于底部价格的涨幅百分比，可在实例化时覆盖）
# 文中参考：40→45 鞋底，50 脚面，60 小腿，之后腰/头
# 用涨幅倍数表达，便于通用
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PHASE_THRESHOLDS = {
    "鞋底": 0.00,   # 底部价格本身
    "脚面": 0.15,   # +15% 以内
    "小腿": 0.30,   # +30% 以内
    "腰部": 0.60,   # +60% 以内
    "头部": float("inf"),
}


class LuMACDStrategy(BaseStrategy):
    """
    Parameters
    ----------
    fast : int
        DIF 快线 EMA 周期，默认 12
    slow : int
        DIF 慢线 EMA 周期，默认 26
    signal_period : int
        DEA 信号线 EMA 周期，默认 9
    vol_window : int
        量能放大判断窗口（周线根数），默认 4（即与前 4 周均量比较）
    phase_thresholds : dict | None
        鞋底→头部的价格涨幅阈值，None 则使用默认值
    sell_phase : str
        触发卖出的 phase 阈值，默认 "腰部"（+60%）
        可选值与 phase_thresholds 的 key 一致：
        "鞋底" / "脚面" / "小腿" / "腰部" / "头部"
    require_volume : bool
        True = 缺少 volume 列时抛出异常；False = 降级运行并打印警告
    require_green_bar : bool
        True（严格）= 金叉发生时 MACD 柱必须 < 0（绿柱），原文"更好"
        False（宽松）= 忽略绿柱条件，只要 0 轴下金叉即可
        默认 True；设为 False 可放宽以捕捉更多信号。
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
        vol_window: int = 4,
        phase_thresholds: dict | None = None,
        sell_phase: str = "头部",
        require_volume: bool = False,
        require_green_bar: bool = True,
    ):
        self.fast             = fast
        self.slow             = slow
        self.signal_period    = signal_period
        self.vol_window       = vol_window
        self.phase_thresholds = phase_thresholds or DEFAULT_PHASE_THRESHOLDS
        self.sell_phase       = sell_phase
        self.require_volume   = require_volume
        self.require_green_bar = require_green_bar

        # 运行时记录底部价格（用于 phase 计算）
        self._bottom_price: float | None = None

    # ── 接口属性 ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        mode = "严格" if self.require_green_bar else "宽松"
        return f"LuMACD({self.fast},{self.slow},{self.signal_period})[{mode}]"

    @property
    def params(self) -> dict:
        return {
            "fast":          self.fast,
            "slow":          self.slow,
            "signal_period": self.signal_period,
            "vol_window":    self.vol_window,
            "sell_phase":    self.sell_phase,
            "require_green_bar": self.require_green_bar,
        }

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _calc_macd(self, close: pd.Series) -> pd.DataFrame:
        """计算 DIF / DEA / MACD 柱，列名不带后缀。"""
        ema_fast  = self._ema(close, self.fast)
        ema_slow  = self._ema(close, self.slow)
        dif       = ema_fast - ema_slow
        dea       = self._ema(dif, self.signal_period)
        histogram = (dif - dea) * 2
        return pd.DataFrame({"DIF": dif, "DEA": dea, "MACD": histogram},
                            index=close.index)

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """将日线 df resample 到指定频率（'W-FRI' / 'MS'）。"""
        agg: dict = {"close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        return df.resample(rule).agg(agg).dropna()

    def _golden_cross_below_zero(self, macd_df: pd.DataFrame) -> pd.Series:
        """
        判断：当根 K 线是否满足"0轴以下金叉"条件。
          - DIF 和 DEA 均 < 0（必要）
          - DIF 本根 > DEA 且上根 DIF <= DEA（上穿，必要）
          - MACD 柱 < 0（绿柱）：require_green_bar=True 时为必要条件，
            False 时忽略（宽松模式，可捕捉更多信号）
        返回 bool Series。
        """
        dif, dea, hist = macd_df["DIF"], macd_df["DEA"], macd_df["MACD"]
        cross_up   = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        below_zero = (dif < 0) & (dea < 0)
        if self.require_green_bar:
            return cross_up & below_zero & (hist < 0)
        return cross_up & below_zero

    def _death_cross(self, macd_df: pd.DataFrame) -> pd.Series:
        """DIF 下穿 DEA → 死叉。"""
        dif, dea = macd_df["DIF"], macd_df["DEA"]
        return (dif < dea) & (dif.shift(1) >= dea.shift(1))

    def _vol_expanding(self, vol_series: pd.Series) -> pd.Series:
        """
        成交量是否温和放大：当周成交量 > 前 N 周滚动均值。
        返回 bool Series（与 vol_series 同索引）。
        """
        rolling_mean = vol_series.shift(1).rolling(self.vol_window).mean()
        return vol_series > rolling_mean

    @staticmethod
    def _align_to_daily(signal_series: pd.Series,
                        daily_index: pd.DatetimeIndex) -> pd.Series:
        """
        将低频（周/月）bool/int Series 向前填充对齐到日线索引。
        低频信号在下一个交易日生效（shift(1) 后 ffill）。
        """
        # reindex 到日线，只保留低频信号发出后的日期
        aligned = signal_series.reindex(daily_index).ffill()
        return aligned.fillna(False).astype(bool)

    def _calc_phase(self, close: pd.Series, bottom_price: float) -> pd.Series:
        """
        根据相对底部价格的涨幅，判断每根 K 线所处的身体部位。
        """
        pct = (close - bottom_price) / bottom_price
        thresholds = sorted(self.phase_thresholds.items(), key=lambda x: x[1])

        def _map(p: float) -> str:
            for label, upper in thresholds:
                if p <= upper:
                    return label
            return list(thresholds)[-1][0]

        return pct.map(_map)

    # ── 核心接口 ─────────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入：日线 DataFrame（DatetimeIndex，含 close，可选 volume）
        输出：附加所有指标列与信号列的 DataFrame
        """
        df = df.copy()

        # ── 0. volume 兼容处理 ───────────────────────────────────────────────
        has_volume = "volume" in df.columns
        if not has_volume:
            msg = (
                f"[{self.name}] 缺少 volume 列，将跳过成交量验证。"
                " 信号质量会下降，建议补充量能数据。"
            )
            if self.require_volume:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

        # ── 1. 日线 MACD ─────────────────────────────────────────────────────
        macd_daily = self._calc_macd(df["close"])
        df[["DIF", "DEA", "MACD"]] = macd_daily

        # ── 2. 周线 MACD ─────────────────────────────────────────────────────
        weekly_raw      = self._resample_ohlcv(df, "W-FRI")
        macd_weekly     = self._calc_macd(weekly_raw["close"])
        weekly_cross    = self._golden_cross_below_zero(macd_weekly)
        weekly_death    = self._death_cross(macd_weekly)

        # 成交量验证（周线）
        if has_volume:
            vol_exp_weekly = self._vol_expanding(weekly_raw["volume"])
        else:
            vol_exp_weekly = pd.Series(True, index=weekly_raw.index)  # 降级：默认满足

        # 周线综合信号 = 金叉 + 量能
        weekly_signal = weekly_cross & vol_exp_weekly

        # 对齐到日线
        for col, src in [("DIF_W", macd_weekly["DIF"]),
                         ("DEA_W", macd_weekly["DEA"]),
                         ("MACD_W", macd_weekly["MACD"])]:
            df[col] = src.reindex(df.index).ffill()

        df["vol_expanding"]    = self._align_to_daily(vol_exp_weekly,  df.index)
        df["weekly_confirmed"] = False   # 后面逐步标记

        # ── 3. 月线 MACD ─────────────────────────────────────────────────────
        monthly_raw   = self._resample_ohlcv(df, "MS")
        macd_monthly  = self._calc_macd(monthly_raw["close"])
        monthly_cross = self._golden_cross_below_zero(macd_monthly)

        for col, src in [("DIF_M", macd_monthly["DIF"]),
                         ("DEA_M", macd_monthly["DEA"]),
                         ("MACD_M", macd_monthly["MACD"])]:
            df[col] = src.reindex(df.index).ffill()

        df["monthly_confirmed"] = False  # 后面逐步标记

        # ── 4. 三级联动信号生成 ──────────────────────────────────────────────
        #
        # 规则：
        #   monthly_confirmed  一旦月线金叉触发，此后持续为 True，
        #                      直到出现月线死叉才重置为 False
        #   weekly_confirmed   在 monthly_confirmed=True 的前提下，
        #                      周线金叉+量能触发，此后持续为 True，
        #                      直到出现周线死叉才重置
        #   signal=1           weekly_confirmed 首次变为 True 的当天
        #   signal=-1          出现日线死叉（最敏感，止损快）
        #
        df["signal"]    = 0
        df["phase"]     = "未知"
        df["weekly_death"] = False
        df["phase_exit"]   = False

        monthly_on = False
        weekly_on  = False
        bottom_px  = None

        # phase 阈值排序，用于卖出判断
        phase_order = ["鞋底", "脚面", "小腿", "腰部", "头部"]
        sell_phase_rank = phase_order.index(self.sell_phase) \
            if self.sell_phase in phase_order else 3

        # 建立日期→信号的快查字典
        monthly_cross_dates = set(
            monthly_cross[monthly_cross].index.normalize()
        )
        monthly_death_dates = set(
            self._death_cross(macd_monthly)[
                self._death_cross(macd_monthly)
            ].index.normalize()
        )
        weekly_signal_dates = set(
            weekly_signal[weekly_signal].index.normalize()
        )
        weekly_death_dates  = set(
            weekly_death[weekly_death].index.normalize()
        )

        for date, row in df.iterrows():
            d = date.normalize() if hasattr(date, "normalize") else pd.Timestamp(date)

            # ── 月线状态更新 ────────────────────────────────────────────────
            month_start = d.to_period("M").to_timestamp()
            if month_start in monthly_cross_dates:
                if not monthly_on:
                    monthly_on = True
                    bottom_px  = row["close"]
            if month_start in monthly_death_dates:
                monthly_on = False
                weekly_on  = False
                bottom_px  = None

            # ── 周线状态更新 ────────────────────────────────────────────────
            days_to_fri = (4 - d.weekday()) % 7
            week_fri    = d + pd.Timedelta(days=days_to_fri)

            if monthly_on and week_fri in weekly_signal_dates:
                if not weekly_on:
                    weekly_on = True
                    df.at[date, "signal"] = 1

            # 周线死叉判断
            is_weekly_death = week_fri in weekly_death_dates
            if is_weekly_death:
                weekly_on = False
            df.at[date, "weekly_death"] = is_weekly_death

            # ── phase 计算 ───────────────────────────────────────────────────
            current_phase = "未知"
            if bottom_px is not None and bottom_px > 0:
                pct = (row["close"] - bottom_px) / bottom_px
                for label, upper in sorted(
                    self.phase_thresholds.items(), key=lambda x: x[1]
                ):
                    if pct <= upper:
                        current_phase = label
                        break
                else:
                    current_phase = "头部"
            df.at[date, "phase"] = current_phase

            # ── 卖出条件（OR）───────────────────────────────────────────────
            # 条件1：周线死叉
            sell_weekly = is_weekly_death and weekly_on is False

            # 条件2：phase 首次达到 sell_phase 阈值
            current_rank = phase_order.index(current_phase) \
                if current_phase in phase_order else -1
            sell_phase_hit = (
                current_phase != "未知"
                and current_rank >= sell_phase_rank
            )
            df.at[date, "phase_exit"] = sell_phase_hit

            if sell_weekly or sell_phase_hit:
                df.at[date, "signal"] = -1
                # 卖出后重置仓位状态
                if sell_phase_hit:
                    weekly_on = False
                    bottom_px = None

            # ── 更新确认标志 ────────────────────────────────────────────────
            df.at[date, "monthly_confirmed"] = monthly_on
            df.at[date, "weekly_confirmed"]  = weekly_on

        return df.dropna(subset=["DIF", "DEA", "MACD"])

    # ── 绘图接口（保持与原版一致） ───────────────────────────────────────────

    def plot_indicators(self, ax, df: pd.DataFrame, colors: dict) -> None:
        """
        在 ax 上绘制：
          - 日线 MACD 柱状图
          - DIF / DEA 曲线
          - 月线/周线 DIF 参考线（虚线，较细）
          - 买卖信号标注
        """
        c_bg    = colors["bg"]
        c_fg    = colors["fg"]
        c_green = colors["green"]
        c_red   = colors["red"]
        c_blue  = colors["blue"]
        c_gold  = colors["gold"]
        c_muted = colors["muted"]

        # MACD 柱
        bar_colors = np.where(df["MACD"] >= 0, c_green, c_red)
        ax.bar(df.index, df["MACD"], color=bar_colors, alpha=0.6,
               width=1, label="MACD柱(日)")

        # DIF / DEA 日线
        ax.plot(df.index, df["DIF"], color=c_blue, lw=1.2,
                label=f"DIF({self.fast},{self.slow})")
        ax.plot(df.index, df["DEA"], color=c_gold, lw=1.2,
                label=f"DEA({self.signal_period})")

        # 月线 DIF 参考（细虚线，颜色半透明）
        if "DIF_M" in df.columns:
            ax.plot(df.index, df["DIF_M"], color=c_blue, lw=0.7,
                    linestyle=":", alpha=0.5, label="DIF_月线")
            ax.plot(df.index, df["DEA_M"], color=c_gold, lw=0.7,
                    linestyle=":", alpha=0.5, label="DEA_月线")

        # 0 轴
        ax.axhline(0, color=c_muted, lw=0.6, linestyle="--")

        # 买卖信号标注
        buy_dates  = df.index[df["signal"] == 1]
        sell_dates = df.index[df["signal"] == -1]
        if len(buy_dates):
            ax.scatter(buy_dates, df.loc[buy_dates, "DIF"],
                       marker="^", color=c_green, s=60, zorder=5,
                       label="买入信号")
        if len(sell_dates):
            ax.scatter(sell_dates, df.loc[sell_dates, "DIF"],
                       marker="v", color=c_red, s=60, zorder=5,
                       label="卖出信号")

        ax.legend(facecolor=c_bg, labelcolor=c_fg, edgecolor=c_muted,
                  fontsize=7, ncol=4, loc="upper left")
        ax.set_ylabel(self.name, color=c_fg, fontsize=9)
