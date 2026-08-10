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


# ── 大盘指数取数起点 ──────────────────────────────────────────────────────────
#
# 牛市过滤器算的是大盘**月线** MACD，EMA(26) 要几十根月线才收敛。旧实现把指数
# 起点定为「最早推荐日 − 600 自然日」≈ 20 根月线，seed 权重还剩 ~12%，DIF 被
# 系统性拉向初始值。实测沪深300：把起点从全历史截到 2023-01，bull 判定仍有 1 个
# 月翻转；截到 2021-01 有 7 个月翻转。
#
# 更糟的是那个起点**依赖候选池**：往 jcy_insights.json 里加一篇更早的文章，
# 全部个股的 bull_market 历史都会改写，回测数值随之变化——这与仓库坚持 hfq
# 的理由（结果必须可复现）是同一件事。
#
# 因此指数一律从这个**绝对起点**取起，与候选池无关。个股行情不受影响：日线
# EMA(26) 几十根就稳，仍按各自推荐日往前预热。
INDEX_HISTORY_START = "20150101"


def index_history_start(requested_start: str | None = None) -> str:
    """指数取数起点：不晚于 INDEX_HISTORY_START。

    requested_start 更早时以它为准（回测区间本来就更长），否则一律补到
    INDEX_HISTORY_START，保证月线 MACD 的预热长度不随候选池变化。
    """
    if not requested_start:
        return INDEX_HISTORY_START
    return min(requested_start, INDEX_HISTORY_START)


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


def execution_kwargs(cfg: BacktestConfig) -> dict:
    """
    从 [backtest] 段读取成交成本与交易约束，直接展开给 run_backtest(**kw)。

    三个单股入口共用，保证它们的成本假设一致——否则同一只票用不同脚本
    回测会得出不同收益，却看不出差在哪。缺省值即引擎默认值（A 股常见水平）。
    """
    return {
        "commission_rate":  cfg.get_float("commission_rate", 0.0003),
        "min_commission":   cfg.get_float("min_commission", 5.0),
        "stamp_duty":       cfg.get_float("stamp_duty", 0.001),
        "slippage":         cfg.get_float("slippage", 0.001),
        "limit_move_check": cfg.get_bool("limit_move_check", True),
        "max_pending_days": cfg.get_int("max_pending_days", 3),
    }


# 可粘进任意 preset .ini 的成本参数模板（留空即用默认值）
EXECUTION_INI_BLOCK = """
# ── 成交成本与交易约束（留空即用默认值）──────────────────────────────────────
# 券商佣金费率（双边），默认 0.0003 万三
commission_rate =
# 单笔最低佣金（元），默认 5
min_commission =
# 印花税（仅卖出），默认 0.001 千一
stamp_duty =
# 单边滑点，默认 0.001 千一；设 0 可与旧版无滑点结果对比
slippage =
# 是否模拟涨跌停/停牌无法成交，默认 true
limit_move_check =
# 信号因涨跌停未成交时最多顺延几个交易日，默认 3
max_pending_days =
"""


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
