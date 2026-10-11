"""案件目录（case dir）—— 让一次优化任务**可跨会话恢复、可交接、可审计**。

为什么需要它（对应第 4 轮之后定的目标：让"降本"变成一次便宜的、agent 能自己完成的迭代）：
一次迭代不是"跑一条命令"，而是"测 → 判断 → 改 → 再测 → 比较"。要能**跨会话**继续做，就必须
把"这次任务是什么、用了哪些输入、得到什么证据、下一步该做什么"落在一处，而不是散在
终端历史里。bench/artifacts/ 是我们的**证据产物**，案件目录是**用户的一次任务**。

目录结构（**不含绝对路径与主机名** —— 便于交接与（将来）入库）：

    <out>.case/
      manifest.json    这次任务是什么：工具版本、协议哈希、输入指纹、输出清单、参数
      env.json         这台机器的能力与选中的口径（来自 packet.env / protocol）
      journal.jsonl    追加式日志：每次调用一条（时间、命令、关键结果、拒绝码）
      next.md          「路径」段：下一步该做什么（可直接交给下一个 agent/会话）
      README.md        怎么继续（给人和 agent 的入口，≤ 1 KB）

「不含绝对路径」是**执行出来的**，不是自律：`--ks-calibration /abs/x.json` 这类参数在写进
`manifest.json`/`journal.jsonl` 之前会过一遍 `_sanitize_argv()`（案件目录旁边的一族产物记成
相对路径、其它绝对路径换成 `<abs>/名字` 占位符）—— 见 `_sanitize_argv()` 的说明。

纪律：**只读 packet 与已落盘的输出**，不重新测量、不写绝对路径、不静默覆盖 journal。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Sequence

CASE_SUFFIX = ".case"
JOURNAL = "journal.jsonl"

# `next.md` 与 `manifest.next_actions` 必须数**同一批**「下一步」：这个上限只在这里定一次，
# 两边都从这里取（各写一个数字=两个数字迟早打架，V-26② 就是这么来的）。
NEXT_LIMIT = 8

_ABS_WIN = re.compile(r"^[A-Za-z]:[\\/]")


def _looks_absolute(tok: str) -> bool:
    return tok.startswith("/") or tok.startswith("\\\\") or bool(_ABS_WIN.match(tok))


def _placeholder(tok: str) -> str:
    """外部绝对路径 → `<abs>/名字`：保留"它是个本机绝对路径"这个事实，但不泄漏目录。"""
    name = os.path.basename(tok.rstrip("/\\")) or tok
    return f"<abs>/{name}"


def _sanitize_token(tok: str, root: pathlib.Path) -> str:
    """一个 argv token：绝对路径 → 相对案件目录（就近）或 `<abs>/名字`（在别处）。"""
    if not _looks_absolute(tok):
        return tok
    try:
        rel = os.path.relpath(tok, root)
    except ValueError:                      # 不同盘符（Windows）
        return _placeholder(tok)
    # 允许"案件目录的上一级"以内的相对形式：那一层就是 `-o <前缀>` 一族产物所在处，
    # 记成 `../rep.json` 之后这条命令**在案件目录里是可重跑的**。再往外（../../…）
    # 就换占位符 —— 越往上写越是在泄漏别人的目录结构深度。
    up = 0
    for part in pathlib.PurePosixPath(rel.replace(os.sep, "/")).parts:
        if part == "..":
            up += 1
        else:
            break
    return rel if up <= 1 else _placeholder(tok)


def sanitize_argv(argv: Sequence[str], root: pathlib.Path) -> list[str]:
    """把 argv 里的绝对路径脱敏（幂等：脱敏过的 token 不再是绝对路径，原样返回）。

    `--flag=/abs/path` 这种**同一 token 里带值**的写法也要处理 —— 只按"整个 token 是路径"
    判断会漏掉一半（`--ks-memtrace=`、`--packet=` 都是这个形状）。
    """
    out: list[str] = []
    for tok in argv:
        if tok.startswith("-") and "=" in tok:
            flag, _, value = tok.partition("=")
            out.append(f"{flag}={_sanitize_token(value, root)}")
        else:
            out.append(_sanitize_token(str(tok), root))
    return out


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
             outputs: dict[str, str | None], root: pathlib.Path,
             *, next_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    proto = packet.get("protocol") or {}
    env = packet.get("env") or {}
    items = _next_items(packet) if next_items is None else next_items
    return {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool_version": packet.get("tool_version"),
        "packet_schema": [packet.get("schema_version"), packet.get("schema_revision")],
        "protocol_hash": proto.get("hash"),
        "protocol_basis": proto.get("basis"),
        "window_mode": proto.get("window_mode"),
        # 输入指纹（标定/轨迹）由**管线**写进包的 `protocol` 块（`packet.build()` 与
        # `protocol_hash()` 读的是 meta 里同一批键），这里只是把它带出来 —— 承诺"用的是
        # 同一份输入"必须有指纹可查（V-26①）。形状随管线：标定是 `文件名:内容哈希` 字符串，
        # 轨迹是 `{name, sha256}` 字典。**没有这份输入就给 null，不给空 `{}`** ——
        # `{}` 正是这条缺陷原来的样子（"给了标定却写成空"），不能让它再冒充一个正常值。
        "protocol_inputs": {
            "calibration": proto.get("calibration") or None,
            "memtrace": proto.get("memtrace") or None,
        },
        "env": {"tier": env.get("tier"), "tier_name": env.get("tier_name"),
                "counter_state": env.get("counter_state"),
                "blocked_by": env.get("blocked_by"), "device": env.get("device")},
        "command": sanitize_argv(argv, root),
        "outputs": {k: _rel(v, root) for k, v in outputs.items() if v},
        "verdicts": [{"id": v.get("id"), "status": v.get("status")} for v in
                     (packet.get("verdicts") or [])],
        "refusals": sorted((packet.get("refusals") or {}).keys()),
        # 与 `next.md` **同源**（同一份 items，同一套排序）：以前这里数的是包里按预算裁剪过
        # 的 `packet["next"]`（常常被整段裁掉 ⇒ 0），而 next.md 渲染的是现算出来的清单 ——
        # 同一个案件目录里两个数字打架（V-26②）。
        "next_actions": {
            "count": len(items),
            "items": [{"for": it.get("for"), "code": it.get("code"),
                       "hint": it.get("hint"), "cmd": it.get("cmd")} for it in items],
        },
    }


def journal_record(packet: dict[str, Any], argv: Sequence[str], rc: int,
                   outputs: dict[str, str | None], root: pathlib.Path) -> dict[str, Any]:
    cost = packet.get("cost") or {}
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": sanitize_argv(argv, root),
        "rc": rc,
        "packet_bytes": cost.get("packet_bytes"),
        "trimmed": cost.get("trimmed") or [],
        "verdicts": {v.get("id"): v.get("status") for v in (packet.get("verdicts") or [])},
        "refusals": sorted((packet.get("refusals") or {}).keys()),
        "outputs": {k: _rel(v, root) for k, v in outputs.items() if v},
    }


def _next_items(packet: dict[str, Any], limit: int = NEXT_LIMIT) -> list[dict[str, Any]]:
    """「下一步」的唯一取法：`next.md` 与 `manifest.next_actions` 都从这里取（V-26②）。"""
    from .packet import next_ranked

    return next_ranked(packet, limit=limit)


def next_markdown(packet: dict[str, Any], limit: int = NEXT_LIMIT,
                  items: list[dict[str, Any]] | None = None) -> str:
    """「路径」段：按"挡住决策的程度"排序的下一步（可红的排序规则见 packet.next_ranked）。"""
    if items is None:
        items = _next_items(packet, limit)
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


_CASE_README = """# 任务目录（kernoscope）

一次优化任务的证据与进度：可跨会话继续、可交接。

| 文件 | 看什么 |
|---|---|
| `manifest.json` | 工具版本、协议哈希、**输入指纹**（标定/轨迹）、参数、输出、下一步 |
| `env.json` | 本机层级、计数器状态、被哪些缺件挡住 |
| `journal.jsonl` | 追加日志：命令、rc、判词、拒绝码 |
| `next.md` | **下一步**：按挡住决策的程度排序（带命令） |
| `../*.ksreport.json` | 完整报告（每个数字带出处）；`../*.packet.json` 给 agent |

继续：① 读 `next.md` 照做；② 同前缀重跑（日志在 `journal.jsonl` 追加）；
③ `kernoscope diff <旧> <新> --selfcheck <自检>` 判真改进/真退化/分不出。

纪律：数字带出处；`n/a` = 前提不成立所以不给数（每条带闭集 code 与处置）；
路径不含绝对路径（就近记相对，别处记 `<abs>/名字`），可直接交接入库。
"""


def write_case(packet: dict[str, Any], out_prefix: str, argv: Sequence[str], rc: int,
               outputs: dict[str, str | None], *, quiet: bool = False) -> pathlib.Path:
    """写（或追加）案件目录；返回目录路径。

    * `manifest.json` / `env.json` / `README.md`：每次调用**覆盖**（它们是"当前状态"，带 schema_version）；
    * `journal.jsonl`：**只追加**（历史不许被静默改写）；
    * `next.md`：每次覆盖（它就是"现在的下一步"）。

    `next.md` 与 `manifest.next_actions` 由**同一份** items 生成（`_next_items()`）——
    两个文件说同一件事，不许各数各的（V-26②）。
    """
    root = case_dir_for(out_prefix)
    root.mkdir(parents=True, exist_ok=True)
    next_items = _next_items(packet)
    man = manifest(packet, out_prefix, argv, outputs, root, next_items=next_items)
    (root / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "env.json").write_text(
        json.dumps(man["env"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "next.md").write_text(next_markdown(packet, items=next_items), encoding="utf-8")
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(_CASE_README, encoding="utf-8")
    with (root / JOURNAL).open("a", encoding="utf-8") as f:
        f.write(json.dumps(journal_record(packet, argv, rc, outputs, root),
                           ensure_ascii=False) + "\n")
    if not quiet:
        print(f"==PROF== 案件目录: {root}（manifest/env/journal/next.md）", file=sys.stderr)
    return root
