"""cubin/ELF 静态后端：`cuobjdump -res-usage` 与 `-sass`（EXACT，且不需要 GPU）。

ncu 的 LaunchStats 里 registers/static smem 就是从 cubin 读的；我们用同一来源做
**交叉校验**：CUPTI activity 给的 registersPerThread 与 cuobjdump 的 REG 应当一致，
不一致时（例如 cubin 与运行的不是同一份代码）在报告里明确标出，而不是二选一。
"""

from __future__ import annotations

import collections
import re
import subprocess

FUNC_RE = re.compile(r"^\s*Function\s+(\S+?)\s*:?\s*$")
KV_RE = re.compile(r"(\w+(?:\[\d+\])?):(\d+)")
SASS_RE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9._]*)")

CATEGORY: list[tuple[str, re.Pattern[str]]] = [
    ("tensor", re.compile(r"^(HMMA|IMMA|BMMA|QMMA|OMMA|MMA|WGMMA|TCGEN)")),
    ("fp", re.compile(r"^(FFMA|FADD|FMUL|FMA|HFMA|DFMA|MUFU|F2F|F2I|I2F|FSETP|FSEL)")),
    ("alu", re.compile(r"^(IADD|IMAD|ISETP|LOP|SHF|LEA|I2I|POPC|FLO|SEL|PRMT|MOV|S2R|CS2R|ULDC|UMOV|USHF|UIADD|UIMAD|UISETP|ULOP|R2UR|S2UR)")),
    ("memory", re.compile(r"^(LDG|STG|LDS|STS|LDL|STL|LD|ST|RED|ATOM|LDC|LDSM|STSM|LDGSTS|CP\.ASYNC)")),
    ("sync", re.compile(r"^(BAR|MEMBAR|FENCE|DEPBAR|ERRBAR|WARPSYNC|BSSY|BSYNC)")),
    ("control", re.compile(r"^(BRA|JMP|JMXU|EXIT|RET|CALL|BRX|JMPA|NOP|BPT|TRAP|YIELD|NANOSLEEP)")),
    ("other", re.compile(r".*")),
]


def find_cuobjdump() -> str | None:
    """多路径发现（PATH → CUDA_HOME → /usr/local/cuda-*/bin → 常见目录）。

    与 doctor 共用 `toolchain.discover`，这样"doctor 说没有、后端却找到了"不可能再发生（B3）。
    """
    from ..toolchain import find

    return find("cuobjdump")


def _run(mode: str, path: str) -> str:
    exe = find_cuobjdump()
    if not exe:
        raise RuntimeError("cuobjdump not found (install the CUDA toolkit)")
    proc = subprocess.run([exe, mode, path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cuobjdump {mode} failed: {proc.stderr[-400:]}")
    return proc.stdout


def parse_res_usage(text: str) -> dict[str, dict[str, int]]:
    """-> {function_name: {REG, SHARED, STACK, LOCAL, ...}}"""
    funcs: dict[str, dict[str, int]] = {}
    cur: str | None = None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, {})
            rest = line.split(":", 1)[1] if line.count(":") > 1 else ""
            for k, v in KV_RE.findall(rest):
                funcs[cur][k] = int(v)
            continue
        if cur and "REG:" in line:
            for k, v in KV_RE.findall(line):
                funcs[cur][k] = int(v)
    return funcs


def res_usage(path: str) -> dict[str, dict[str, int]]:
    return parse_res_usage(_run("-res-usage", path))


def parse_sass_mix(text: str) -> dict[str, object]:
    ops: collections.Counter[str] = collections.Counter()
    for line in text.splitlines():
        if "/*" not in line:
            continue
        m = SASS_RE.search(line)
        if m:
            ops[m.group(1).split(".")[0]] += 1
    total = sum(ops.values()) or 1
    cats: collections.Counter[str] = collections.Counter()
    for op, n in ops.items():
        for name, rx in CATEGORY:
            if rx.match(op):
                cats[name] += n
                break
    return {
        "total": total,
        "top_opcodes": ops.most_common(12),
        "categories": {k: v for k, v in cats.most_common()},
        "categories_pct": {k: 100.0 * v / total for k, v in cats.most_common()},
    }


def sass_mix(path: str) -> dict[str, object]:
    return parse_sass_mix(_run("-sass", path))


def strip_mangled(name: str) -> str:
    """`_Z4addkPfi` -> `addk`（仅用于与 cuobjdump 的函数名做宽松匹配）。"""
    return re.sub(r"^_Z\d+", "", name).split("P")[0] or name
