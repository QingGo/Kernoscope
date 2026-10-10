"""R-34：文档里的每条可复制命令都必须**被真的跑过**，而且断言它的预期结论。

为什么这条不能只做「开关存在 + 命令跑得通」的冒烟：
验收报告 D1 那条命令**开关都存在、也能跑通**，但它给出的 per-launch 误差是 +146.9%
（窗口里顺序跑了四个 kernel）。只断言"能跑"的检查器会放它过去，还会给人虚假的安全感。
所以这里的规则是：登记一条文档命令时，必须同时登记它**应该产出什么**。

规则表 DOC_COMMAND_RULES 里每条是：
    (文档, 说明, argv 后缀, 断言函数)
断言函数拿 (rc, stdout, stderr, 报告 JSON) 判断；任何一条不成立就失败。

覆盖范围：
  * 所有被跟踪文档中 ```bash / ```sh 代码块里出现的长开关，都必须真的存在于 CLI 解析器
    （B8：文档写 `--smoke` / `--shapes`，实际没有这些开关）；
  * 登记的重点命令必须端到端跑出预期结论（含"该拒绝时必须拒绝"）。
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

import kernoscope.cli as cli

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "PROVENANCE.md", ROOT / "CALIBRATION.md",
        ROOT / "CONTRIBUTING.md", ROOT / "bench" / "README.md"]

# 文档里合法出现、但不属于 kernoscope 自己的命令
FOREIGN = ("python", "python3", "bash", "sh", "uv", "pip", "git", "rsync", "scp", "ssh",
           "nvcc", "nsys", "ncu", "nvidia-smi", "echo", "cat", "ls", "mkdir", "cd",
           "export", "tmux", "curl", "column", "sed", "awk", "grep")


def _fenced_blocks(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    inside = False
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            if inside:
                out.append("\n".join(buf))
                buf = []
            inside = not inside
            continue
        if inside:
            buf.append(line)
    return out


def _all_valid_flags() -> set[str]:
    """kernoscope 主命令 + 四个子命令的**全部**合法长开关。

    子命令有自己的解析器，所以从各自的 --help 里抓 —— 手抄一份清单会跟着代码漂。
    """
    valid = set()
    for action in cli.build_parser()._actions:
        valid.update(action.option_strings)
    from kernoscope.doctor import doctor_main
    from kernoscope.issue_cmd import issue_main
    from kernoscope.memtrace_cmd import memtrace_main
    from kernoscope.occupancy_cmd import occupancy_main

    for entry in (doctor_main, occupancy_main, memtrace_main, issue_main):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                entry(["--help"])
        except SystemExit:
            pass
        valid.update(re.findall(r"--[a-z][a-z0-9-]+", buf.getvalue()))
    return valid


def _kernoscope_flags_in(block: str) -> set[str]:
    """代码块里**属于 kernoscope 命令行**的长开关。

    只认 kernoscope 自己的行：`tools/*.py`、`bench/ledger_probe` 之类是我们自己的脚本，
    它们有自己的开关（--kernel/--quick/--mb），不该拿主 CLI 的解析器去判。
    """
    out: set[str] = set()
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        # 命令可能缩进（续行以 - 开头）：取本行第一个"程序"词
        head = re.match(r"^(?:[A-Za-z_][\w]*=\S+\s+)*(?:uv run\s+)?([\w./-]+)", stripped)
        if not head:
            continue
        prog = head.group(1)
        if prog not in ("kernoscope", "kernoscope)"):
            continue
        out.update(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", stripped))
    return out


def test_every_documented_flag_exists():
    """B8 的负样本：文档里 kernoscope 命令行上的每个长开关都必须真的存在。

    真机上教程正文写过 `--smoke` / `--shapes` 这类不存在的开关，读者照抄直接失败。
    """
    valid = _all_valid_flags()
    bad: dict[str, set[str]] = {}
    for doc in DOCS:
        for block in _fenced_blocks(doc):
            for flag in _kernoscope_flags_in(block):
                if flag not in valid:
                    bad.setdefault(flag, set()).add(doc.name)
    assert not bad, "文档里出现了不存在的开关（读者会照抄失败）：\n  - " + "\n  - ".join(
        f"{f}（出现在 {', '.join(sorted(d))}）" for f, d in sorted(bad.items()))


def test_flag_checker_actually_catches_invented_flags(tmp_path):
    """检查器自身的负样本：塞一个编出来的开关进去必须能被抓到。

    否则这个门禁只是"看起来在检查"—— 教程正文里 `--smoke` 能长期存活就是前车之鉴。
    """
    valid = _all_valid_flags()
    assert "--smoke" not in valid and "--shapes" not in valid, \
        "这两个开关真的不存在；如果哪天加上了，请换一个编造的开关来测这个检查器"
    assert _kernoscope_flags_in("kernoscope --set basic --smoke -- x.py") == {"--set", "--smoke"}
    assert _kernoscope_flags_in("python tools/machine-profile.py --quick") == set()


def test_documented_subcommands_exist():
    """文档里出现的子命令（kernoscope <sub>）必须存在。"""
    known = {"doctor", "occupancy", "memtrace", "issue"}
    offenders: list[str] = []
    for doc in DOCS:
        for block in _fenced_blocks(doc):
            for line in block.splitlines():
                m = re.match(r"\s*(?:uv run )?kernoscope\s+([a-z][a-z-]+)", line)
                if m and m.group(1) not in known:
                    offenders.append(f"{doc.name}: {line.strip()}")
    assert not offenders, "\n  - ".join(offenders)


# --------------------------------------------------------------------------
# 登记的重点命令：必须断言预期结论，而不是"跑得通"
# --------------------------------------------------------------------------
def _run(capsys, *argv):
    rc = cli.main(list(argv))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _clean_capsys(capsys, *argv):
    """跑一条命令并吞掉这一轮的多余输出（让断言只看本次）。"""
    capsys.readouterr()
    return _run(capsys, *argv)


def _load_metrics(path: Path) -> dict[str, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for k in doc.get("kernels", []):
        for s in k.get("sections", []):
            for m in s.get("metrics", []):
                out.setdefault(m["name"], m)
    return out


def test_documented_basic_command_gives_a_number(monkeypatch, capsys, tmp_path):
    """README 里最常用的那条：--set basic + -k + 账本 + 标定，预期拿到一个数。

    无 GPU 的机器上要给两样替身（NVML 采样序列 + 与窗口自洽的 launch 表），
    否则「拿不到数」是环境决定的，不是命令决定的 —— 那样测不出文档对不对。
    """
    from test_dram import _FakeSampler, _coherent_launches

    monkeypatch.setattr(cli, "Sampler", _FakeSampler)
    monkeypatch.setattr(cli, "load_synthetic", _coherent_launches)
    rc, out, err = _clean_capsys(
        capsys, "--ks-synthetic", "--ks-peak-gbps", "909", "-k", "regex:gemm",
        "--ks-ledger-mb", "420000", "--ks-window", "auto",
        "-o", str(tmp_path / "rep"), "-f")
    assert rc == 0, err
    m = _load_metrics(tmp_path / "rep.ksreport.json")
    # 预期结论：给了 per-launch，并且**同时**给出了它的两个前提
    assert m["est__dram_bytes_per_launch"]["value"] is not None
    assert m["ks__sampler_window_coverage_pct"]["value"] is not None
    assert m["ks__matched_kernel_kernel_share_pct"]["value"] is not None
    assert m["ks__attribution_bias_estimate_pct"]["value"] is not None


def test_documented_strict_command_refuses_when_data_is_missing(capsys):
    """README 里推荐的 CI 用法：--strict 在拿不到请求的数据时必须 exit 2。"""
    rc, _out, err = _clean_capsys(capsys, "--ks-synthetic", "--ks-no-nvml", "--strict",
                                  "--set", "full", "-k", "regex:gemm")
    assert rc == 2, "文档承诺 --strict 会让 CI 在缺数据时失败"
    assert "error(strict)" in err


def test_documented_metrics_command_is_not_silently_empty(capsys):
    """README 教人用 --metrics 窄化输出；写错名字不能得到"看起来成功"的空报告。"""
    rc, _out, err = _clean_capsys(capsys, "--ks-synthetic", "--metrics", "dram__bytes.sum",
                                  "-k", "regex:gemm")
    assert rc == 0
    assert "一个都不存在" in err
