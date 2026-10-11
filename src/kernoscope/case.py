"""案件目录（case dir）—— 让一次优化任务**可跨会话恢复、可交接、可审计**。

为什么需要它（对应第 4 轮之后定的目标：让"降本"变成一次便宜的、agent 能自己完成的迭代）：
一次迭代不是"跑一条命令"，而是"测 → 判断 → 改 → 再测 → 比较"。要能**跨会话**继续做，就必须
把"这次任务是什么、用了哪些输入、得到什么证据、下一步该做什么"落在一处，而不是散在
终端历史里。bench/artifacts/ 是我们的**证据产物**，案件目录是**用户的一次任务**。

目录结构（全部相对路径，不含绝对路径与主机名 —— 便于交接与（将来）入库）：

    <out>.case/
      manifest.json    这次任务是什么：工具版本、协议哈希、输入指纹、输出清单、参数
      env.json         这台机器的能力与选中的口径（来自 packet.env / protocol）
      journal.jsonl    追加式日志：每次调用一条（时间、命令、关键结果、拒绝码）
      next.md          「路径」段：下一步该做什么（可直接交给下一个 agent/会话）
      README.md        怎么继续（给人和 agent 的入口，≤ 1 KB）

纪律：**只读 packet 与已落盘的输出**，不重新测量、不写绝对路径、不静默覆盖 journal。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Sequence

CASE_SUFFIX = ".case"
JOURNAL = "journal.jsonl"


def case_dir_for(out_prefix: str) -> pathlib.Path:
    base = out_prefix[:-len(".json")] if out_prefix.endswith(".json") else out_prefix
    return pathlib.Path(base + CASE_SUFFIX)


def _rel(p: str | None, root: pathlib.Path) -> str | None:
    """输出路径一律记成相对案件目录的路径（交接时不会泄漏别人的目录结构）。"""
    if not p:
        return None
    try:
        return os.path.relpath(p, root)
    except ValueError:                      # 不同盘符（Windows）
        return pathlib.Path(p).name


def manifest(packet: dict[str, Any], out_prefix: str, argv: Sequence[str],
             outputs: dict[str, str | None], root: pathlib.Path) -> dict[str, Any]:
    proto = packet.get("protocol") or {}
    env = packet.get("env") or {}
    return {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool_version": packet.get("tool_version"),
        "packet_schema": [packet.get("schema_version"), packet.get("schema_revision")],
        "protocol_hash": proto.get("hash"),
        "protocol_basis": proto.get("basis"),
        "window_mode": proto.get("window_mode"),
        "protocol_inputs": {
            # 输入指纹（标定/轨迹）已经在协议元里：直接带出来，交接时能确认"用的是同一份输入"
            "calibration": (proto.get("calibration") or {}),
            "memtrace": (proto.get("memtrace") or {}),
        },
        "env": {"tier": env.get("tier"), "tier_name": env.get("tier_name"),
                "counter_state": env.get("counter_state"),
                "blocked_by": env.get("blocked_by"), "device": env.get("device")},
        "command": list(argv),
        "outputs": {k: _rel(v, root) for k, v in outputs.items() if v},
        "verdicts": [{"id": v.get("id"), "status": v.get("status")} for v in
                     (packet.get("verdicts") or [])],
        "refusals": sorted((packet.get("refusals") or {}).keys()),
        "next_actions": len(packet.get("next") or []),
    }


def journal_record(packet: dict[str, Any], argv: Sequence[str], rc: int,
                   outputs: dict[str, str | None], root: pathlib.Path) -> dict[str, Any]:
    cost = packet.get("cost") or {}
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": list(argv),
        "rc": rc,
        "packet_bytes": cost.get("packet_bytes"),
        "trimmed": cost.get("trimmed") or [],
        "verdicts": {v.get("id"): v.get("status") for v in (packet.get("verdicts") or [])},
        "refusals": sorted((packet.get("refusals") or {}).keys()),
        "outputs": {k: _rel(v, root) for k, v in outputs.items() if v},
    }


def next_markdown(packet: dict[str, Any], limit: int = 8) -> str:
    """「路径」段：按"挡住决策的程度"排序的下一步（可红的排序规则见 packet.next_ranked）。"""
    from .packet import next_ranked

    items = next_ranked(packet, limit=limit)
    lines = ["# 下一步（按挡住决策的程度排序）", ""]
    if not items:
        lines.append("没有待办：本报告的判词都成立且已给出结论。下一步是改代码，然后再跑一次 `diff`。")
        return "\n".join(lines) + "\n"
    for i, it in enumerate(items, 1):
        head = f"{i}. **{it.get('for')}**"
        if it.get("code"):
            head += f" — 被拒（`{it['code']}`）"
        lines.append(head)
        if it.get("action"):
            lines.append(f"   - 怎么办：{it['action']}")
        if it.get("cmd"):
            lines.append(f"   - 命令：`{it['cmd']}`")
        if it.get("hint"):
            lines.append(f"   - 提示：{it['hint']}")
    lines += ["", "改完之后：`kernoscope diff <改前包> <改后包> --selfcheck <自检>` "
                  "（差值小于 MDD 就是噪声，不要当成改进）。"]
    return "\n".join(lines) + "\n"


_CASE_README = """# 这一次任务（kernoscope 案件目录）

这个目录是**一次优化任务**的证据与进度，供跨会话继续、或交给别人/别的 agent 接手。

| 文件 | 看什么 |
|---|---|
| `manifest.json` | 这次任务是什么：工具版本、协议哈希、**输入指纹**（标定/轨迹）、参数、输出清单 |
| `env.json` | 这台机器能答什么：能力层级、计数器状态、被哪些缺件挡住 |
| `journal.jsonl` | 追加式日志：每次调用一条（命令、rc、判词状态、拒绝码、包大小） |
| `next.md` | **下一步**：按挡住决策的程度排序，带「怎么办」与命令 |
| `../*.ksreport.json` | 完整报告（每个数字带出处）；`../*.packet.json` 是给 agent 的决策包 |

继续的方式：
1. 读 `next.md`，照着改代码或补数据源；
2. 再跑一次同一命令（`-o` 指同一个前缀，日志会**追加**在 `journal.jsonl` 里）；
3. 用 `kernoscope diff <旧包> <新包> --selfcheck <自检>` 判断"真改进 / 真退化 / 分不出来"。

纪律：这里的数字都带出处（exact/measured/estimated/unavailable）；`n/a` 不是失败，是
"前提不成立所以不给数"，每条 `n/a` 都带一个闭集 code 与处置办法。
"""


def write_case(packet: dict[str, Any], out_prefix: str, argv: Sequence[str], rc: int,
               outputs: dict[str, str | None], *, quiet: bool = False) -> pathlib.Path:
    """写（或追加）案件目录；返回目录路径。

    * `manifest.json` / `env.json` / `README.md`：每次调用**覆盖**（它们是"当前状态"，带 schema_version）；
    * `journal.jsonl`：**只追加**（历史不许被静默改写）；
    * `next.md`：每次覆盖（它就是"现在的下一步"）。
    """
    root = case_dir_for(out_prefix)
    root.mkdir(parents=True, exist_ok=True)
    man = manifest(packet, out_prefix, argv, outputs, root)
    (root / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "env.json").write_text(
        json.dumps(man["env"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "next.md").write_text(next_markdown(packet), encoding="utf-8")
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(_CASE_README, encoding="utf-8")
    with (root / JOURNAL).open("a", encoding="utf-8") as f:
        f.write(json.dumps(journal_record(packet, argv, rc, outputs, root),
                           ensure_ascii=False) + "\n")
    if not quiet:
        print(f"==PROF== 案件目录: {root}（manifest/env/journal/next.md）", file=sys.stderr)
    return root
