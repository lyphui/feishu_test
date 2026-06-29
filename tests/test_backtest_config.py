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
