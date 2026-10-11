"""MCP（stdio）—— 只给"没有 shell 的 agent runtime"用的那一层。

**决定**（`docs/agent-first-design.md` §2.5 与 `NON-GOALS.md` N10）：**可选、零常驻、带止损**。
- 它**不进**默认安装路径，也**不进** 4 KB 常驻入口预算（`AGENTS.md` / `SKILL.md` frontmatter /
  `llms.txt` 里不许提它 —— 有测试钉住）。只有显式 `kernoscope mcp --stdio` 才启动。
- 为什么不把它当主接口：MCP 的工具定义**每一轮都在上下文里**（单个工具约 200–500 token、
  GitHub 全套 93 工具约 55k token/轮），而 skill frontmatter 只有约 120 词常驻。
  取舍是"能不能被想起" vs "每轮都付费"，所以主路径仍然是 **CLI + skill**。

止损条件（写死在决定里）：若一段时间内没有任何真实环境用它，就删掉这个模块。

设计约束：
- **零第三方依赖**，只用标准库（JSON-RPC over stdin/stdout）。
- 协议：stdio 上**按行分隔的 JSON-RPC 2.0**（MCP 规范对 stdio 的定义）。
- 工具**不自己实现测量**：它们只是把 CLI 的既有子命令叫起来并回传文本 —— 这样"证据从哪来"
  只有一处（`--packet` / `capabilities` / `diff` / `selftest`），MCP 不会变成第二条口径。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any, Callable, Iterable

PROTOCOL_VERSION = "2024-11-05"

#: 工具集合（与 `docs/agent-first-design.md` §2.5 的最小形态一致：4 工具 + 3 资源）
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "capabilities_brief",
        "description": "这台机器能答什么/不能答什么（约 1 KB），含「下一步做什么」与代价。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "profile",
        "description": "跑一次采集并返回决策包（判词 + 前提 + 结构化拒绝 + 代价，≤4 KB）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "array", "items": {"type": "string"},
                            "description": "被测命令（放在 `--` 之后）"},
                "extra_args": {"type": "array", "items": {"type": "string"},
                               "description": "额外的 kernoscope 开关"},
                "synthetic": {"type": "boolean",
                              "description": "没有 GPU 时用合成数据跑通流程（会被标注 SYNTHETIC）"},
            },
            "required": ["program"],
            "additionalProperties": False,
        },
    },
    {
        "name": "diff",
        "description": "比较两次运行：improved / regressed / indistinguishable / incomparable。",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"], "additionalProperties": False,
        },
    },
    {
        "name": "selftest",
        "description": "用已知答案负载算本机误差带与最小可分辨差异（MDD）。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
)

#: 资源：全部是**仓库里既有的静态文件**，没有隐藏状态（不假装有会话状态）
RESOURCES: tuple[dict[str, str], ...] = (
    {"uri": "kernoscope://refusal-codes", "name": "拒绝码参考",
     "description": "每个不给出数字的 code、人话、怎么办与代价（由代码生成）。",
     "path": "skills/kernoscope/references/refusal-codes.md", "mimeType": "text/markdown"},
    {"uri": "kernoscope://provenance", "name": "出处规则",
     "description": "五个出处等级与四条强制规则。",
     "path": "PROVENANCE.md", "mimeType": "text/markdown"},
    {"uri": "kernoscope://skill", "name": "给 agent 的入口",
     "description": "触发条件与三步闭环。",
     "path": "skills/kernoscope/SKILL.md", "mimeType": "text/markdown"},
)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _default_runner(argv: list[str]) -> tuple[int, str, str]:
    """把 CLI 叫起来（**唯一**的数据来源；MCP 不自己算任何东西）。"""
    p = subprocess.run([sys.executable, "-m", "kernoscope", *argv],
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _err(id_: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _ok(id_: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _text(content: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": content}], "isError": is_error}


def _tool_argv(name: str, args: dict[str, Any]) -> list[str] | None:
    """工具 → CLI argv。返回值 `None` 表示参数不合法（回 JSON-RPC 错误）。"""
    if name == "capabilities_brief":
        return ["capabilities", "--brief"]
    if name == "profile":
        prog = args.get("program")
        if not isinstance(prog, list) or not prog or not all(isinstance(x, str) for x in prog):
            return None
        extra = args.get("extra_args") or []
        if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
            return None
        base = ["--packet", "-", *extra]
        if args.get("synthetic"):
            base.append("--ks-synthetic")
        return [*base, "--", *prog]
    if name == "diff":
        a, b = args.get("a"), args.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            return None
        return ["diff", a, b, "--json"]
    if name == "selftest":
        return ["selftest", "--json"]
    return None


def handle(request: Any, *, runner: Callable[[list[str]], tuple[int, str, str]] | None = None,
           root: pathlib.Path | None = None) -> dict[str, Any] | None:
    """处理一条 JSON-RPC 消息；通知（无 id）返回 None。"""
    run = runner or _default_runner
    base = root or _repo_root()
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return _err(None, -32600, "invalid request：需要 jsonrpc=2.0 的对象")
    id_ = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "initialize":
        return _ok(id_, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "kernoscope", "version": _version()},
            # 常驻成本的取舍写在这里，客户端可据此决定要不要挂
            "instructions": "主路径是 CLI + skill；这一层只服务没有 shell 的 runtime。"
                            "所有数字来自 kernoscope 的决策包，判词 status=refused 或 "
                            "actionable=false 时不得作为行动依据。",
        })
    if method == "ping":
        return _ok(id_, {})
    if method == "tools/list":
        return _ok(id_, {"tools": [dict(t) for t in TOOLS]})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        argv = _tool_argv(str(name), args if isinstance(args, dict) else {})
        if argv is None:
            return _err(id_, -32602, f"invalid params：未知工具或参数不合法（{name!r}）")
        rc, out, err = run(argv)
        body = out.strip() or err.strip() or f"（没有输出；rc={rc}）"
        if rc != 0:
            body += f"\n\n[rc={rc}]\n{err.strip()}"
        return _ok(id_, _text(body, is_error=rc != 0))
    if method == "resources/list":
        return _ok(id_, {"resources": [{k: v for k, v in r.items() if k != "path"}
                                       for r in RESOURCES]})
    if method == "resources/read":
        uri = params.get("uri")
        for r in RESOURCES:
            if r["uri"] == uri:
                p = base / r["path"]
                try:
                    text = p.read_text(encoding="utf-8")
                except OSError as e:
                    return _err(id_, -32603, f"resource unreadable：{r['path']}（{e}）")
                return _ok(id_, {"contents": [{"uri": uri, "mimeType": r["mimeType"],
                                               "text": text}]})
        return _err(id_, -32602, f"invalid params：未知资源 {uri!r}")
    return _err(id_, -32601, f"method not found：{method!r}")


def _version() -> str:
    from . import __version__

    return __version__


def serve(stdin: Iterable[str], stdout, *,
          runner: Callable[[list[str]], tuple[int, str, str]] | None = None) -> int:
    """逐行读 JSON-RPC、逐行写响应。坏行回 `-32700` 但**不退出**（长会话不能因一行坏掉）。"""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response: dict[str, Any] | None = _err(None, -32700, f"parse error：{e}")
        else:
            response = handle(request, runner=runner)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    """`kernoscope mcp --stdio`（只有这一个传输：进程内的 stdio，不监听端口）。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] != "--stdio":
        print("error: mcp 目前只支持 `--stdio`（不监听端口、不启动服务）", file=sys.stderr)
        return 2
    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":                      # pragma: no cover - 手动调试入口
    raise SystemExit(main())
