

def test_integer_valued_floats_are_not_printed_in_scientific_notation():
    """R-93③/R-108 的同一类反模式：`{:.6g}` 会把 1048576.0 印成 `1.04858e+06`。

    文本视图里整数必须按整数印（人类读的数字不该带指数），非整数仍按 6 位有效数字。
    """
    from kernoscope.render import _fmt_value

    assert _fmt_value(1048576.0, "")[0] == "1048576"
    assert _fmt_value(22524800.0, "")[0] == "22524800"
    assert _fmt_value(4.755, "ms")[0] == "4.755"
    assert _fmt_value(1.5e-7, "")[0] == "1.5e-07"
    # 单位换算路径不受影响
    assert _fmt_value(2.5e6, "MB")[0] == "2.500"
