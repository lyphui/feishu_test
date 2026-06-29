from utils.jcy_common import title_to_date, title_to_filename


def test_title_to_date_success():
    assert title_to_date("Vol.260626 时代主题") == "2026-06-26"


def test_title_to_date_no_digits_returns_none():
    assert title_to_date("无日期的标题") is None


def test_title_to_filename_with_date():
    assert title_to_filename("Vol.260626 时代主题") == "2026-06-26__Vol.260626 时代主题.md"


def test_title_to_filename_without_date_uses_nodate():
    assert title_to_filename("纯文字标题") == "NODATE__纯文字标题.md"


def test_title_to_filename_sanitizes():
    assert title_to_filename("a/b 260101") == "2026-01-01__a_b 260101.md"
