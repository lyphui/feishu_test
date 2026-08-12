"""
均线交叉策略（快线上穿慢线买入，下穿卖出）
==========================================
默认 MA5 / MA8，即坊间「超短线金叉买、死叉卖」那套纪律：

    金叉：MA5 由下往上穿过 MA8 → 买入 / 继续持有
    死叉：MA5 由上往下穿过 MA8 → 卖出 / 离场

这个类只负责**把规则写成信号**，规则本身好不好由回测回答
（`backtest/ma_cross_bench.py` 做了分品类的实测）。参数完全开放
（fast / slow / ma_type），5-8 只是默认值，方便同一套代码顺手扫
MA10/20、MA20/60 等更慢的组合做对照——短均线交叉最大的问题就是
「换个参数就变天」，不能只测一个点。

量能过滤（vol_window > 0 时启用）
--------------------------------
`vol_window=5, vol_ratio=1.0` 表示：**金叉当日成交量必须 ≥ 最近 5 日
均量**，否则这次金叉不买。死叉照卖，不看量能——出场纪律一旦附加条件，
就会退化成「跌了还找理由拿着」，而这恰恰是短均线策略唯一真正的风控。
量能只用来过滤**进场**，这是把「结合量能看」落成可回测规则时唯一
不引入未来信息、也不破坏止损纪律的落法。

均量窗口含当日成交量（当日收盘后即可知），配合引擎的 signal.shift(1)
在 T+1 开盘成交，时序自洽。

时序铁律
--------
信号一律用**当日收盘**的均线关系判定，由引擎 shift(1) 后在次日开盘成交。
不做任何 `.shift(-1)`、不看未来均线，截断重算测试见
`tests/test_ma_cross.py`。
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MACrossStrategy(BaseStrategy):

    def __init__(self, fast: int = 5, slow: int = 8, ma_type: str = "sma",
                 vol_window: int = 0, vol_ratio: float = 1.0):
        if fast >= slow:
            raise ValueError(f"快线周期必须小于慢线：fast={fast} slow={slow}")
        if ma_type not in ("sma", "ema"):
            raise ValueError(f"未知均线类型：{ma_type}（可选 sma / ema）")
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type
        self.vol_window = vol_window
        self.vol_ratio = vol_ratio

    # ── 接口属性 ────────────────────────────────────────────

    @property
    def name(self) -> str:
        tag = "MA" if self.ma_type == "sma" else "EMA"
        base = f"{tag}{self.fast}/{tag}{self.slow}"
        if self.vol_window:
            base += f"·量≥{self.vol_ratio:g}×{self.vol_window}日均量"
        return base

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "ma_type": self.ma_type,
                "vol_window": self.vol_window, "vol_ratio": self.vol_ratio}

    # ── 指标计算（内部） ─────────────────────────────────────

    def _ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "ema":
            return self._ema(series, period)
        return series.rolling(period).mean()

    # ── BaseStrategy 接口实现 ────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算快慢均线并生成金叉/死叉信号。"""
        fast_ma = self._ma(df["close"], self.fast)
        slow_ma = self._ma(df["close"], self.slow)

        cols = {
            "MA_FAST": fast_ma,
            "MA_SLOW": slow_ma,
            # 快慢线乖离（%）：>0 即快线在上，穿越 0 轴就是金叉/死叉
            "MA_SPREAD": (fast_ma / slow_ma - 1) * 100,
        }
        if self.vol_window:
            vol_ma = df["volume"].rolling(self.vol_window).mean()
            cols["VOL_MA"] = vol_ma
            # 量比：当日量 / 近 N 日均量（均量含当日，收盘后即可知）
            cols["VOL_RATIO"] = df["volume"] / vol_ma.replace(0, np.nan)

        # dropna 掉均线预热期：预热期内 MA 为 NaN，比较恒为 False，
        # 不会产生幽灵信号，但留着会让首根有效 K 线的「前一日」缺失，
        # 索引口径也和其他策略不一致。
        out = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1).dropna()

        above = out["MA_FAST"] > out["MA_SLOW"]
        prev_above = above.shift(1)
        golden = above & (prev_above == False)     # noqa: E712 — NaN 首行须为 False
        death = (~above) & (prev_above == True)    # noqa: E712

        if self.vol_window:
            # 只过滤进场：金叉当日缩量则放弃这次买点，不顺延、不补买。
            golden &= out["VOL_RATIO"] >= self.vol_ratio

        out["signal"] = 0
        out.loc[golden, "signal"] = 1
        out.loc[death, "signal"] = -1
        return out

    def plot_indicators(self, ax, df: pd.DataFrame, colors: dict) -> None:
        """第二子图：快慢线乖离柱（穿 0 轴即交叉）+ 可选的量比线。"""
        c_bg = colors["bg"]
        c_fg = colors["fg"]
        c_green = colors["green"]
        c_red = colors["red"]
        c_gold = colors["gold"]
        c_muted = colors["muted"]

        bar_colors = np.where(df["MA_SPREAD"] >= 0, c_green, c_red)
        ax.bar(df.index, df["MA_SPREAD"], color=bar_colors, alpha=0.75, width=1,
               label=f"乖离 {self.fast}/{self.slow}（%）")
        ax.axhline(0, color=c_muted, lw=0.6, linestyle="--")
        ax.set_ylabel(self.name, color=c_fg, fontsize=9)

        handles, labels = ax.get_legend_handles_labels()
        if self.vol_window and "VOL_RATIO" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df.index, df["VOL_RATIO"], color=c_gold, lw=0.8, alpha=0.9,
                     label=f"量比（{self.vol_window}日）")
            ax2.axhline(self.vol_ratio, color=c_gold, lw=0.5, linestyle=":")
            ax2.tick_params(colors=c_fg, labelsize=7)
            ax2.set_ylim(0, 4)
            h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2

        ax.legend(handles, labels, facecolor=c_bg, labelcolor=c_fg,
                  edgecolor=c_muted, fontsize=8, ncol=2, loc="upper left")
