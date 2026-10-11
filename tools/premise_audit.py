"""逐判词的前提审计（第七轮 X-22）：**某个判词能发出的前提 ⊆ 文档给它列的那套**。

为什么不能只做"全局并集"（旧实现，也是这个文件存在的原因）：
把 `_environment_premises(...)` 挂到 `dram_bytes` 上时，那三个前提 id 当然都出现在标签表里
（别的判词在用）⇒ 全局并集检查全绿，而 `verdicts.md` 给 `dram_bytes` 列的前提是另一套。
于是"代码实际发出的前提"与"文档列出的前提"可以长期分叉，而 `--gates` 9/9。

做法：把 `packet.py` 解析成 AST，对每个 builder 收集
① 它自己能发出的 `Premise("…")` 字面量；
② 它调用到的同模块函数（**传递展开**，所以"经由 helper 发出的前提"也算它的）；
然后要求这堆 id ⊆ `VERDICT_PREMISES[该 builder 会产出的判词 id]`。
判词 id 从 builder 体里的 `id="…"` 字面量取（一个 builder 可能产出多个 id，例如
`_verdict_saving` 的两级 —— 那就每个 id 都要满足）。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Mapping

PACKET_PY = Path(__file__).resolve().parents[1] / "src" / "kernoscope" / "packet.py"


def _collect(source: str) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """源码 → (每个函数发出的前提字面量, 每个函数调用的同模块函数, 每个函数里的判词 id)。"""
    tree = ast.parse(source)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    literals: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    ids: dict[str, set[str]] = {}
    for name, fn in funcs.items():
        lits: set[str] = set()
        cs: set[str] = set()
        vids: set[str] = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not isinstance(f, ast.Name):
                continue
            first = node.args[0] if node.args else None
            if (f.id == "Premise" and isinstance(first, ast.Constant)
                    and isinstance(first.value, str)):
                lits.add(first.value)
            elif f.id in funcs:
                cs.add(f.id)
        # 判词 id：`Verdict(id="…")` 或 `id="…"` 关键字；只认字面量
        for node in ast.walk(fn):
            if isinstance(node, ast.keyword) and node.arg == "id" and isinstance(
                    node.value, ast.Constant) and isinstance(node.value.value, str):
                vids.add(node.value.value)
        literals[name], calls[name], ids[name] = lits, cs, vids
    return literals, calls, ids


def _expand(name: str, literals: Mapping[str, set[str]], calls: Mapping[str, set[str]],
            seen: set[str] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    if name in seen:
        return set()
    seen.add(name)
    out = set(literals.get(name, ()))
    for callee in calls.get(name, ()):
        out |= _expand(callee, literals, calls, seen)
    return out


def audit_source(source: str, verdict_premises: Mapping[str, Iterable[str]],
                 builders: Iterable[str]) -> list[str]:
    """审计一份源码：返回违规说明（空 = 干净）。测试用合成源码证明它会红。"""
    literals, calls, ids = _collect(source)
    declared = {k: set(v) for k, v in verdict_premises.items()}
    problems: list[str] = []
    for b in builders:
        if b not in literals:
            continue
        emitted = _expand(b, literals, calls)
        for vid in sorted(ids.get(b, ())):
            allowed = declared.get(vid)
            if allowed is None:
                problems.append(f"{b} 会产出判词 {vid}，但 VERDICT_PREMISES 里没有它")
                continue
            extra = sorted(emitted - allowed)
            if extra:
                problems.append(
                    f"判词 {vid}（builder {b}）会发出文档没列的前提：{extra}"
                    f"（文档列的是 {sorted(allowed)}）")
    return problems


def audit(packet_py: Path | None = None) -> list[str]:
    """审计真实的 `packet.py`：逐 builder 对 `VERDICT_PREMISES` 检查。"""
    import sys

    sys.path.insert(0, str(PACKET_PY.parents[1]))
    from kernoscope.packet import BUILDERS, VERDICT_PREMISES  # noqa: PLC0415

    path = packet_py or PACKET_PY
    return audit_source(path.read_text(encoding="utf-8"), VERDICT_PREMISES,
                        [b.__name__ for b in BUILDERS])


def main() -> int:
    problems = audit()
    for p in problems:
        print(f"FAIL 前提审计：{p}")
    if problems:
        return 1
    print("ok  逐判词前提 ⊆ 文档列出的前提")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
