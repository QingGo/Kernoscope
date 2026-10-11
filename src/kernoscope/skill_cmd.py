"""`kernoscope skill` —— 把 skill 装到 agent 的 skill 目录（I4，获客入口）。

为什么是子命令而不是"看文档自己复制"：我们用这个工具的人第一步就是"让 agent 会用"。
一个 `kernoscope skill --apply` 应当满足四条，且每一条都有测试：

  1. **先计划后执行**：默认只打印计划（目标、会新建/更新哪些文件、备份到哪、大小），
     要真写必须显式 `--apply`。（agent 不能回答交互式提问，所以用"显式开关"做同意门。）
  2. **可回滚**：写之前把已存在的目标备份成 `<file>.kernoscope.backup`（**已存在则保留
     第一份快照**，绝不覆盖旧备份）；任何一步失败按写入顺序**逆序**回滚。
  3. **写后回读校验**：写完读回来逐字节比对，不符就报错并回滚 —— 静默写坏的 skill
     会让 agent 后续每一轮都用错指令。
  4. **幂等**：内容相同就是 `unchanged`（不重写、不改 mtime），第二次 `--apply` 必须
     没有任何写入。

skill 的来源是仓库/包里的 `skills/kernoscope/`（`SKILL.md` + `references/` + `scripts/`）。
目标目录按 client 约定：
  * `claude`（默认）：`$CLAUDE_CONFIG_DIR/skills/kernoscope` 或 `~/.claude/skills/kernoscope`
  * `agents`：`~/.agents/skills/kernoscope`（其它 client 的共享目录约定）
  * `--target` 直接指定（测试与自定义布局用）
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SKILL_NAME = "kernoscope"
BACKUP_SUFFIX = ".kernoscope.backup"
# 与 effects.OPERATIONS["skill_install"] 同名：这是同一个操作，副作用声明只有一处。
OPERATION_ID = "skill_install"


def skill_source_root() -> Path | None:
    """skill 的源目录（`SKILL.md` + `references/` + `scripts/`）。

    两种布局都支持：源码 checkout（`<repo>/skills/kernoscope`）与安装后的包
    （`<site-packages>/kernoscope/skills/kernoscope`）。找不到就返回 None —— 由调用方
    报"这个安装里没带 skill"，而不是抛一个 FileNotFoundError 让人猜。
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "skills" / SKILL_NAME,          # 源码 checkout
        here.parent / "skills" / SKILL_NAME,              # 包内附带
        here.parents[1] / "skills" / SKILL_NAME,
    ]
    for c in candidates:
        if (c / "SKILL.md").is_file():
            return c
    return None


def default_target(client: str = "claude", home: str | None = None) -> Path:
    home_dir = Path(home or os.path.expanduser("~"))
    if client == "agents":
        return home_dir / ".agents" / "skills" / SKILL_NAME
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(cfg) if cfg else home_dir / ".claude"
    return root / "skills" / SKILL_NAME


@dataclass
class FileAction:
    rel: str
    operation: str            # create | update | unchanged
    bytes_: int = 0
    backup_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.rel, "operation": self.operation, "bytes": self.bytes_,
                "backup": self.backup_path}


@dataclass
class SkillPlan:
    source: str
    target: str
    actions: list[FileAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[FileAction]:
        return [a for a in self.actions if a.operation != "unchanged"]

    @property
    def requires_approval(self) -> bool:
        return bool(self.changed)

    def to_dict(self) -> dict[str, Any]:
        return {"operation": OPERATION_ID, "source": self.source, "target": self.target,
                "requires_approval": self.requires_approval,
                "actions": [a.to_dict() for a in self.actions], "notes": self.notes}


def _iter_source(src: Path) -> list[tuple[str, Path]]:
    """源目录里的文件清单（相对路径，稳定排序）—— 顺序确定，计划才可比。"""
    out: list[tuple[str, Path]] = []
    for p in sorted(src.rglob("*")):
        if p.is_file() and not p.name.endswith(BACKUP_SUFFIX):
            out.append((str(p.relative_to(src)), p))
    return out


def plan_install(source: Path | None = None, target: Path | None = None,
                 client: str = "claude", home: str | None = None) -> SkillPlan:
    """算出这次安装会做什么（纯函数：只读，不写）。"""
    src = source or skill_source_root()
    if src is None:
        raise FileNotFoundError(
            "这个安装里没有 skill：找不到 skills/kernoscope/SKILL.md。"
            "如果你是从源码跑的，请在仓库根目录下执行")
    tgt = target or default_target(client, home)
    plan = SkillPlan(source=str(src), target=str(tgt))
    for rel, sp in _iter_source(src):
        dst = tgt / rel
        data = sp.read_bytes()
        if not dst.exists():
            op = "create"
        elif dst.read_bytes() == data:
            op = "unchanged"
        else:
            op = "update"
        plan.actions.append(FileAction(
            rel=rel, operation=op, bytes_=len(data),
            backup_path=f"{dst}{BACKUP_SUFFIX}" if op == "update" else ""))
    if not plan.actions:
        plan.notes.append("源目录里一个文件都没有（这是缺陷，不是正常状态）")
    plan.notes.append(f"备份策略：已存在的文件备份成 <文件>{BACKUP_SUFFIX}，"
                      f"**已存在则保留第一份快照**（不覆盖旧备份）")
    return plan


def render_plan(plan: SkillPlan) -> str:
    lines = [f"安装计划（skill={SKILL_NAME}；默认只打印，加 --apply 才写）",
             f"  源：{plan.source}", f"  目标：{plan.target}", ""]
    for a in plan.actions:
        tag = {"create": "新建", "update": "更新", "unchanged": "不变"}[a.operation]
        line = f"  [{tag}] {a.rel}  （{a.bytes_} B）"
        if a.backup_path:
            line += f"  备份：{a.backup_path}"
        lines.append(line)
    for n in plan.notes:
        lines.append("")
        lines.append("注：" + n)
    lines.append("")
    lines.append(f"结论：{'会改动 ' + str(len(plan.changed)) + ' 个文件' if plan.changed else '已经是目标状态，无需改动'}")
    return "\n".join(lines)


def _backup_once(dst: Path) -> str:
    """备份已存在的目标文件；**已存在则保留第一份**（不覆盖旧快照）。"""
    backup = Path(f"{dst}{BACKUP_SUFFIX}")
    if backup.exists():
        return str(backup)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dst, backup)
    return str(backup)


def _write_atomic(dst: Path, data: bytes) -> None:
    """先写临时文件再 rename：中途失败不会留下半个文件。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".kernoscope.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    # 与源文件同样的可执行位（scripts/*.sh 必须可执行）
    if dst.suffix == ".sh":
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def apply_install(plan: SkillPlan, *, dry_run: bool = False) -> dict[str, Any]:
    """执行计划：备份 → 原子写 → **回读校验**；任何一步失败按写入顺序逆序回滚。"""
    if dry_run:
        return {"status": "planned", "changed": len(plan.changed), "written": 0}
    if not plan.changed:
        return {"status": "unchanged", "changed": 0, "written": 0}
    src = Path(plan.source)
    tgt = Path(plan.target)
    written: list[tuple[Path, bytes | None]] = []      # (目标, 写入前的原内容或 None)
    try:
        for a in plan.actions:
            if a.operation == "unchanged":
                continue
            dst = tgt / a.rel
            data = (src / a.rel).read_bytes()
            original = dst.read_bytes() if dst.exists() else None
            if original is not None:
                _backup_once(dst)
            _write_atomic(dst, data)
            # **先记账再校验**：回读失败时也要能回滚 —— 真机场景里"校验失败"恰恰意味着
            # 目标文件现在是坏的，而它是这次写坏的。先 append 再检查这一行，是回滚能不能
            # 生效的分界线（测试抓到的：早先版本校验失败后把坏文件留在了原地）。
            written.append((dst, original))
            if dst.read_bytes() != data:
                raise OSError(f"readback mismatch: {dst}")
    except OSError as exc:
        for dst, original in reversed(written):        # 逆序回滚
            try:
                if original is None:
                    dst.unlink(missing_ok=True)
                else:
                    dst.write_bytes(original)
            except OSError:
                pass
        return {"status": "failed", "reason": str(exc), "rolled_back": len(written)}
    return {"status": "installed", "changed": len(plan.changed), "written": len(written),
            "backups": [a.backup_path for a in plan.actions if a.backup_path]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="kernoscope skill",
        description="把 skill（SKILL.md + references/ + scripts/）安装到 agent 的 skill 目录。"
                    "默认只打印计划；要真写请加 --apply。")
    ap.add_argument("--apply", action="store_true", help="真写（默认只打印计划）")
    ap.add_argument("--target", default=None, help="目标目录（默认按 client 约定）")
    ap.add_argument("--client", default="claude", choices=["claude", "agents"],
                    help="目标目录约定：claude=~/.claude/skills（默认），agents=~/.agents/skills")
    ap.add_argument("--json", action="store_true", help="机器可读（agent 用这个）")
    ap.add_argument("--source", default=None, help="skill 源目录（默认自动发现）")
    return ap


def skill_main(argv: list[str]) -> int:
    import json

    args = build_parser().parse_args(argv)
    try:
        plan = plan_install(source=Path(args.source) if args.source else None,
                           target=Path(args.target).expanduser() if args.target else None,
                           client=args.client)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.json:
        payload = plan.to_dict()
        payload["applied"] = False
        if args.apply:
            payload["applied"] = True
            payload["result"] = apply_install(plan)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_plan(plan))
        if args.apply:
            res = apply_install(plan)
            if res["status"] == "installed":
                print(f"\n已安装：{res['written']} 个文件写入 {plan.target}")
                if res.get("backups"):
                    print(f"备份：{len(res['backups'])} 个（<文件>{BACKUP_SUFFIX}）")
            elif res["status"] == "unchanged":
                print("\n已是最新：没有文件需要改动。")
            else:
                print(f"\nerror: 安装失败并已回滚：{res.get('reason')}", file=sys.stderr)
                return 1
    return 0
