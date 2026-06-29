"""backtest/config.py 的 OutputPaths 测试。"""

import os

from config import OutputPaths


def test_stem_and_suffixes(tmp_path):
    p = OutputPaths(str(tmp_path), "macd", "maotai", "600519", "20240101")
    stem = "macd_maotai_600519_20240101"
    assert p.chart == os.path.join(str(tmp_path), stem + ".png")
    assert p.csv == os.path.join(str(tmp_path), stem + ".csv")
    assert p.status == os.path.join(str(tmp_path), stem + "_daily_status.csv")


def test_makedirs_when_save_dir_set(tmp_path):
    sub = tmp_path / "out" / "nested"
    OutputPaths(str(sub), "lu_bull", "n", "000001", "20240101")
    assert sub.is_dir()


def test_empty_save_dir_all_none(tmp_path, monkeypatch):
    # save_dir 为空 → 路径全 None，且不创建任何目录
    called = {"makedirs": False}
    real_makedirs = os.makedirs

    def _spy(*a, **k):
        called["makedirs"] = True
        return real_makedirs(*a, **k)

    monkeypatch.setattr(os, "makedirs", _spy)
    p = OutputPaths("", "macd", "n", "600519", "20240101")
    assert p.chart is None
    assert p.csv is None
    assert p.status is None
    assert called["makedirs"] is False


def test_safe_sanitizes_illegal_chars():
    assert OutputPaths.safe("A/B:C*?") == "A_B_C__"
    assert OutputPaths.safe("正常名称") == "正常名称"
