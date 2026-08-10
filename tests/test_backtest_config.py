"""backtest/config.py 的 load_backtest_config 测试（离线，不读真实 config/）。"""

from datetime import date

import pytest

import config as bt_config
from config import BacktestConfig, load_backtest_config


_INI_FULL = """\
[backtest]
symbol = 600519
name = maotai
start_date = 20200101
end_date = 20240101
capital = 50000
stop_loss = 0.1
take_profit = 0.2
save_chart_dir = out/
proxy =
index_symbol = 000905
vol_window = 6
shrink_exit = true
"""

_INI_BLANKS = """\
[backtest]
symbol = 000001
name = test
start_date = 20210101
end_date =
capital = 100000
stop_loss =
take_profit =
save_chart_dir =
proxy =
"""


def _write_cfg(monkeypatch, tmp_path, filename, text):
    """把 config 模块的 _PRESETS_DIR 指向 tmp_path 并写入一个 .ini。"""
    monkeypatch.setattr(bt_config, "_PRESETS_DIR", str(tmp_path))
    (tmp_path / filename).write_text(text, encoding="utf-8")


def test_parses_full_config(monkeypatch, tmp_path):
    _write_cfg(monkeypatch, tmp_path, "full.ini", _INI_FULL)
    cfg = load_backtest_config("full.ini")
    assert isinstance(cfg, BacktestConfig)
    assert cfg.symbol == "600519"
    assert cfg.name == "maotai"
    assert cfg.start_date == "20200101"
    assert cfg.end_date == "20240101"
    assert cfg.capital == 50000.0
    assert cfg.stop_loss == 0.1
    assert cfg.take_profit == 0.2
    assert cfg.save_dir == "out/"
    assert cfg.index_symbol == "000905"


def test_end_date_defaults_to_today(monkeypatch, tmp_path):
    _write_cfg(monkeypatch, tmp_path, "blank.ini", _INI_BLANKS)
    cfg = load_backtest_config("blank.ini")
    assert cfg.end_date == date.today().strftime("%Y%m%d")


def test_blank_stop_take_become_none(monkeypatch, tmp_path):
    _write_cfg(monkeypatch, tmp_path, "blank.ini", _INI_BLANKS)
    cfg = load_backtest_config("blank.ini")
    assert cfg.stop_loss is None
    assert cfg.take_profit is None


def test_missing_file_writes_defaults_then_parses(monkeypatch, tmp_path):
    monkeypatch.setattr(bt_config, "_PRESETS_DIR", str(tmp_path))
    cfg = load_backtest_config("new.ini", defaults=_INI_FULL)
    assert (tmp_path / "new.ini").exists()
    assert cfg.symbol == "600519"


def test_missing_file_without_defaults_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(bt_config, "_PRESETS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_backtest_config("nope.ini")


def test_proxy_sets_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    text = _INI_BLANKS.replace("proxy =", "proxy = http://127.0.0.1:7890")
    _write_cfg(monkeypatch, tmp_path, "proxy.ini", text)
    cfg = load_backtest_config("proxy.ini")
    assert cfg.proxy == "http://127.0.0.1:7890"
    assert bt_config.os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert bt_config.os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"


def test_extra_holds_strategy_params(monkeypatch, tmp_path):
    _write_cfg(monkeypatch, tmp_path, "full.ini", _INI_FULL)
    cfg = load_backtest_config("full.ini")
    assert cfg.get_int("vol_window", 4) == 6
    assert cfg.get_bool("shrink_exit", False) is True
    assert cfg.get_bool("missing_key", True) is True   # 缺省回退
    assert cfg.get_int("missing_key", 99) == 99


# ── execution_kwargs：三个单股入口共用的成本参数 ──────────────────────────────

def test_execution_kwargs_defaults(monkeypatch, tmp_path):
    """.ini 未写成本参数时，回退到引擎默认值（A 股常见水平）。"""
    _write_cfg(monkeypatch, tmp_path, "full.ini", _INI_FULL)
    kw = bt_config.execution_kwargs(load_backtest_config("full.ini"))
    assert kw == {
        "commission_rate": 0.0003,
        "min_commission": 5.0,
        "stamp_duty": 0.001,
        "slippage": 0.001,
        "limit_move_check": True,
        "max_pending_days": 3,
    }


def test_execution_kwargs_overrides(monkeypatch, tmp_path):
    text = _INI_FULL + "slippage = 0\nlimit_move_check = false\nmin_commission = 0\n"
    _write_cfg(monkeypatch, tmp_path, "cost.ini", text)
    kw = bt_config.execution_kwargs(load_backtest_config("cost.ini"))
    assert kw["slippage"] == 0.0
    assert kw["limit_move_check"] is False
    assert kw["min_commission"] == 0.0


def test_execution_kwargs_feed_run_backtest():
    """返回的键必须全部是 run_backtest 认识的参数名。"""
    import inspect
    from engine import run_backtest
    cfg = BacktestConfig(symbol="600519", name="x", start_date="20200101",
                         end_date="20240101")
    sig = inspect.signature(run_backtest).parameters
    for key in bt_config.execution_kwargs(cfg):
        assert key in sig, f"execution_kwargs 产出了 run_backtest 不接受的参数：{key}"


# ── 指数取数起点（牛市过滤器可复现性） ────────────────────────────────────────

def test_index_history_start_is_absolute_not_pool_dependent():
    """
    月线 EMA(26) 要几十根月线才收敛。旧实现按「最早推荐日 − 600 天」取指数，
    往 jcy_insights.json 里加一篇更早的文章就会改写全部个股的 bull_market
    历史 —— 回测数值随候选池变化，不可复现。起点必须是绝对日期。
    """
    from config import INDEX_HISTORY_START, index_history_start

    # 与候选池无关：不传参数永远是同一个绝对起点
    assert index_history_start() == INDEX_HISTORY_START
    assert index_history_start() == index_history_start()

    # 早于绝对起点的请求以请求为准（回测区间本来就更长）
    assert index_history_start("20100101") == "20100101"
    # 晚于绝对起点的一律补齐，不允许缩短月线预热
    assert index_history_start("20240101") == INDEX_HISTORY_START
    assert index_history_start("") == INDEX_HISTORY_START
    assert index_history_start(None) == INDEX_HISTORY_START


def test_index_history_start_leaves_room_for_monthly_macd():
    """至少要够 EMA(26) 月线收敛（26 根远远不够，这里要求 ≥ 8 年）。"""
    from datetime import date
    from config import INDEX_HISTORY_START

    start = date(int(INDEX_HISTORY_START[:4]), int(INDEX_HISTORY_START[4:6]),
                 int(INDEX_HISTORY_START[6:]))
    months = (date.today().year - start.year) * 12 + date.today().month - start.month
    assert months >= 96, f"指数预热只有 {months} 个月，月线 MACD 无法收敛"
