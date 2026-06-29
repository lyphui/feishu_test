"""
回测共享配置层
==============
- BacktestConfig / load_backtest_config: 统一解析 backtest/presets/*.ini 的 [backtest] 段，
  消除三个单股入口脚本里重复的 ~40 行 .ini 解析 + 默认配置写出 + proxy 环境变量设置。
- OutputPaths: 统一输出文件路径构造（{prefix}_{name}_{symbol}_{end_date} + .png/.csv/.status）。

策略专属参数（vol_window / require_volume / shrink_exit ...）落在 BacktestConfig.extra，
通过 get_int / get_bool / get_float 类型化访问，避免 dataclass 字段爆炸。
"""

import configparser
import os
import re
from dataclasses import dataclass, field
from datetime import date as _date

# 回测预设 .ini 内置于 backtest 包内（backtest/presets/），取本文件所在目录
_PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


@dataclass
class BacktestConfig:
    symbol: str
    name: str
    start_date: str
    end_date: str                       # 已默认为今日（若 .ini 留空）
    capital: float = 100_000.0
    stop_loss: float | None = None
    take_profit: float | None = None
    save_dir: str = ""
    proxy: str = ""
    index_symbol: str = "000300"
    extra: dict = field(default_factory=dict)   # 策略专属原始字符串值

    def get_str(self, key: str, default: str = "") -> str:
        return self.extra.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        raw = self.extra.get(key, "").strip()
        return int(raw) if raw else default

    def get_float(self, key: str, default: float = 0.0) -> float:
        raw = self.extra.get(key, "").strip()
        return float(raw) if raw else default

    def get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.extra.get(key, "").strip().lower()
        if not raw:
            return default
        return raw in ("true", "1", "yes", "on")


# 这些 key 由 BacktestConfig 显式接管，其余 [backtest] 键归入 extra
_KNOWN_KEYS = {
    "symbol", "name", "start_date", "end_date", "capital",
    "stop_loss", "take_profit", "save_chart_dir", "proxy", "index_symbol",
}


def load_backtest_config(filename: str, *, defaults: str | None = None) -> BacktestConfig:
    """读取 backtest/presets/<filename> 的 [backtest] 段为 BacktestConfig。

    - 文件缺失且提供 defaults：写出该模板再解析（保留各策略专属注释）。
    - end_date 留空 → 默认今日（YYYYMMDD）。
    - stop_loss / take_profit 留空 → None，否则 float。
    - proxy 非空 → 写入 HTTP_PROXY / HTTPS_PROXY 环境变量。
    - 未知键 → 原始字符串存入 .extra（供 get_* 访问）。
    """
    config_path = os.path.join(_PRESETS_DIR, filename)
    if not os.path.exists(config_path):
        if defaults is not None:
            os.makedirs(_PRESETS_DIR, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(defaults)
            print(f"  配置文件不存在，已生成默认配置：{config_path}")
        else:
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    s = cfg["backtest"]

    end_date = s.get("end_date", "").strip()
    if not end_date:
        end_date = _date.today().strftime("%Y%m%d")
        print(f"  end_date 未设置，默认使用今日：{end_date}")

    stop_loss_raw   = s.get("stop_loss", "").strip()
    take_profit_raw = s.get("take_profit", "").strip()
    proxy           = s.get("proxy", "").strip()
    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"  代理：{proxy}")

    extra = {k: s.get(k, "") for k in s if k not in _KNOWN_KEYS}

    return BacktestConfig(
        symbol=s.get("symbol", "600519").strip(),
        name=s.get("name", "stock").strip(),
        start_date=s.get("start_date", "20200101").strip(),
        end_date=end_date,
        capital=float(s.get("capital", "100000")),
        stop_loss=float(stop_loss_raw) if stop_loss_raw else None,
        take_profit=float(take_profit_raw) if take_profit_raw else None,
        save_dir=s.get("save_chart_dir", "").strip(),
        proxy=proxy,
        index_symbol=s.get("index_symbol", "000300").strip(),
        extra=extra,
    )


@dataclass
class OutputPaths:
    """统一输出路径：{prefix}_{name}_{symbol}_{end_date} + .png/.csv/.status。

    save_dir 为空时所有路径返回 None（弹窗显示、不落盘）；非空时确保目录存在。
    """
    save_dir: str
    prefix: str
    name: str
    symbol: str
    end_date: str

    def __post_init__(self):
        self._stem = f"{self.prefix}_{self.name}_{self.symbol}_{self.end_date}"
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _path(self, suffix: str) -> str | None:
        if not self.save_dir:
            return None
        return os.path.join(self.save_dir, self._stem + suffix)

    @property
    def chart(self) -> str | None:
        return self._path(".png")

    @property
    def csv(self) -> str | None:
        return self._path(".csv")

    @property
    def status(self) -> str | None:
        return self._path("_daily_status.csv")

    @staticmethod
    def safe(name: str) -> str:
        """清洗用于文件名的字符串（去除路径分隔符等非法字符）。"""
        return re.sub(r'[\\/:*?"<>|]', "_", name)
