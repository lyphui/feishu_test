"""run.json 可复现清单的守护测试（评审项 2）。"""

import json
import time

from backtest.lib import manifest
from backtest.lib.manifest import write_run_manifest


def test_manifest_contains_all_version_dimensions(tmp_path):
    """git / argv / costs / data / 耗时，五个维度缺一不可。"""
    path = write_run_manifest(
        str(tmp_path), argv=["prog", "--offline"],
        symbols=["601857", ("000300", "none")], started_at=time.time(),
    )
    m = json.loads(open(path, encoding="utf-8").read())

    assert m["argv"] == ["prog", "--offline"]
    assert set(m["git"]) == {"sha", "dirty", "dirty_scope"}
    # dirty 只看代码路径：data/market/daily/ 是被跟踪的，每次 auto_update 都会
    # 重写它，全仓库口径下 dirty 恒为 true，就再也表达不了「带未提交代码跑的」
    assert "backtest" in m["git"]["dirty_scope"]
    assert not any(p.startswith("data") for p in m["git"]["dirty_scope"])
    assert m["costs"]["risk_free_rate"] == manifest.costs.RISK_FREE_RATE
    assert m["costs"]["commission_rate"] == manifest.costs.COMMISSION_RATE
    assert m["elapsed_sec"] is not None and m["elapsed_sec"] >= 0
    # 本地有缓存的标的给出 data_end/rows；没有缓存的记 null 而不是报错
    assert "601857_hfq" in m["data"] and "000300_none" in m["data"]


def test_manifest_tolerates_missing_meta_and_git(tmp_path, monkeypatch):
    """无本地缓存 + git 不可用：降级为 null，仍正常落盘。"""
    monkeypatch.setattr(manifest, "_git_info", lambda: {"sha": None, "dirty": None})
    path = write_run_manifest(str(tmp_path / "sub"), symbols=["999999"])
    m = json.loads(open(path, encoding="utf-8").read())
    assert m["git"] == {"sha": None, "dirty": None}
    assert m["data"]["999999_hfq"] is None
    assert m["elapsed_sec"] is None      # 未传 started_at
