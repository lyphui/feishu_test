"""
MACD 金叉/死叉策略
==================
DIF 上穿 DEA → 买入（金叉）
DIF 下穿 DEA → 卖出（死叉）

指标公式：
    DIF  = EMA(close, fast) - EMA(close, slow)
    DEA  = EMA(DIF, signal)
    MACD = (DIF - DEA) × 2
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MACDStrategy(BaseStrategy):

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    # ── 接口属性 ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal}

    # ── 指标计算（内部） ─────────────────────────────────────

    def _calc_macd(self, close: pd.Series) -> pd.DataFrame:
        ema_fast  = self._ema(close, self.fast)
        ema_slow  = self._ema(close, self.slow)
        dif       = ema_fast - ema_slow
        dea       = self._ema(dif, self.signal)
        histogram = (dif - dea) * 2
        return pd.DataFrame({"DIF": dif, "DEA": dea, "MACD": histogram})

    # ── BaseStrategy 接口实现 ────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 MACD 指标并生成金叉/死叉信号。"""
        macd_df = self._calc_macd(df["close"])
        df = pd.concat([df, macd_df], axis=1).dropna()

        df["signal"] = 0
        # 金叉：DIF 由下穿上 → 买入
        df.loc[
            (df["DIF"] > df["DEA"]) & (df["DIF"].shift(1) <= df["DEA"].shift(1)),
            "signal"
        ] = 1
        # 死叉：DIF 由上穿下 → 卖出
        df.loc[
            (df["DIF"] < df["DEA"]) & (df["DIF"].shift(1) >= df["DEA"].shift(1)),
            "signal"
        ] = -1

        return df

    def plot_indicators(self, ax, df: pd.DataFrame, colors: dict) -> None:
        """在 ax 上绘制 MACD 柱状图及 DIF/DEA 曲线。"""
        c_bg    = colors["bg"]
        c_fg    = colors["fg"]
        c_green = colors["green"]
        c_red   = colors["red"]
        c_blue  = colors["blue"]
        c_gold  = colors["gold"]
        c_muted = colors["muted"]

        bar_colors = np.where(df["MACD"] >= 0, c_green, c_red)
        ax.bar(df.index, df["MACD"], color=bar_colors, alpha=0.7, width=1, label="MACD柱")
        ax.plot(df.index, df["DIF"], color=c_blue, lw=1,
                label=f"DIF({self.fast},{self.slow})")
        ax.plot(df.index, df["DEA"], color=c_gold, lw=1,
                label=f"DEA({self.signal})")
        ax.axhline(0, color=c_muted, lw=0.5, linestyle="--")
        ax.legend(facecolor=c_bg, labelcolor=c_fg, edgecolor=c_muted,
                  fontsize=8, ncol=3, loc="upper left")
        ax.set_ylabel(self.name, color=c_fg, fontsize=9)
