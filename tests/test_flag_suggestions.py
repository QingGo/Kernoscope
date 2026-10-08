"""R-14：报告里建议的每个 `--ks-*` 开关都必须真的存在。

真机上踩到的形态：`provenance.py` 的"需要插桩"原因里建议了一个不存在的
`--ks-instrumentation`，用户照做直接 `argparse error: unrecognized arguments`。
"失败原因指向一个不存在的补救措施"比不给原因更糟，所以把它变成测试。
"""

from __future__ import annotations

import re
from pathlib import Path

from kernoscope.cli import build_parser

SRC = Path(__file__).resolve().parents[1] / "src" / "kernoscope"
KS_FLAG = re.compile(r"--ks-[a-z0-9][a-z0-9-]*")


def _known_flags() -> set[str]:
    parser = build_parser()
    out: set[str] = set()
    for action in parser._actions:          # noqa: SLF001 - 就是要枚举全部选项
        out.update(action.option_strings)
        out.update(getattr(action, "choices", None) or [])
    # 子命令自己解析的参数
    for sub in ("doctor", "occupancy", "memtrace", "issue"):
        out.add(sub)
    return out


def _mentioned_flags() -> dict[str, list[str]]:
    """源码里出现的 `--ks-*`（含注释与文档字符串，因为它们在用户可见文本里）。"""
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in KS_FLAG.finditer(text):
            found.setdefault(m.group(0), []).append(f"{path.name}:{text[:m.start()].count(chr(10)) + 1}")
    return found


def _exists(flag: str, known: set[str]) -> bool:
    """`--ks-peak-*` 这种通配写法：只要有一族同名开关存在就算存在。"""
    if flag in known:
        return True
    if flag.endswith("-"):
        return any(k.startswith(flag) for k in known)
    return False


def test_every_suggested_ks_flag_exists():
    known = _known_flags()
    missing = {flag: where for flag, where in _mentioned_flags().items()
               if not _exists(flag, known)}
    assert not missing, (
        "以下 --ks-* 开关在源码里被建议/提及，但 argparse 不认："
        + "; ".join(f"{f}（{', '.join(w[:2])}）" for f, w in missing.items()))


def test_unavailable_reasons_point_at_real_switches_only():
    """`UNAVAILABLE_REASONS` 是用户最先读到的补救建议，逐条检查。"""
    from kernoscope.provenance import UNAVAILABLE_REASONS

    known = _known_flags()
    for key, reason in UNAVAILABLE_REASONS.items():
        for flag in KS_FLAG.findall(reason):
            assert _exists(flag, known), \
                f"UNAVAILABLE_REASONS[{key!r}] 建议了不存在的开关 {flag}"


def test_help_text_does_not_crash_and_lists_key_switches(capsys):
    """`--help` 必须能跑（help 文本里的裸 % 会让 argparse 崩过），且列出关键开关。"""
    import pytest

    from kernoscope.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert (e.value.code or 0) == 0
    out = capsys.readouterr().out
    for flag in ("--ks-ledger-mb", "--ks-ledger-total-mb", "--ks-ledger-basis",
                 "--ks-window", "--ks-warmup", "--ks-memtrace", "--ks-instr-hist",
                 "--ks-calibration", "--launch-csv", "--strict"):
        assert flag in out, f"--help 里应该能看到 {flag}"
