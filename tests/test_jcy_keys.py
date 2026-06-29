from utils.jcy_common import safe_title, record_key


def test_safe_title_replaces_illegal_chars():
    assert safe_title('a/b:c*d?"e<f>g|h') == "a_b_c_d__e_f_g_h"


def test_safe_title_truncates_long_title():
    long = "x" * 200
    assert len(safe_title(long)) == 80


def test_safe_title_strips_whitespace():
    assert safe_title("  hello  ") == "hello"


def test_record_key_combines_date_and_title():
    assert record_key("2026-06-26", "Vol.260626 时代主题") == "2026-06-26__Vol.260626 时代主题"


def test_record_key_handles_none_date():
    assert record_key(None, "无日期标题") == "NODATE__无日期标题"


def test_record_key_same_date_different_title_distinct():
    a = record_key("2026-06-26", "上午盘评")
    b = record_key("2026-06-26", "下午盘评")
    assert a != b
