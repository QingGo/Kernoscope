"""MCP 层的门禁（N10：**可选、零常驻、带止损**）。

这一层的价值是"没有 shell 的 runtime 也能用上"，它的**代价**是工具定义每轮都进上下文。
所以测试守三件事：
  1. 协议正确（JSON-RPC 2.0：initialize/tools/resources/ping、错误码、坏行不崩）；
  2. **零常驻**：常驻入口层（AGENTS.md / SKILL.md frontmatter / llms.txt）里不许出现 mcp ——
     否则每次会话都要为它付费，那就违背了做它的理由；
  3. 工具**不自己算数**：所有输出都来自 CLI 既有子命令（用注入的 runner 断言 argv）。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from kernoscope import mcp

ROOT = Path(__file__).resolve().parents[1]

ENTRY_LAYER = ("AGENTS.md", "llms.txt")


def _runner(record: list[list[str]], rc: int = 0, out: str = "OK", err: str = ""):
    def run(argv):
        record.append(list(argv))
        return rc, out, err
    return run


# --------------------------------------------------------------------------
# 1) 协议
# --------------------------------------------------------------------------
def test_initialize_reports_the_protocol_and_the_cost_tradeoff():
    r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert r["id"] == 1 and r["result"]["protocolVersion"] == mcp.PROTOCOL_VERSION
    info = r["result"]["serverInfo"]
    assert info["name"] == "kernoscope" and info["version"]
    assert "skill" in r["result"]["instructions"], "要写清主路径是 CLI + skill，而不是这一层"


def test_tools_and_resources_are_listed():
    tools = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in tools} == {"capabilities_brief", "profile", "diff", "selftest"}
    for t in tools:
        assert t["description"] and t["inputSchema"]["type"] == "object"
    res = mcp.handle({"jsonrpc": "2.0", "id": 3,
                      "method": "resources/list"})["result"]["resources"]
    assert {r["uri"] for r in res} == {"kernoscope://refusal-codes",
                                       "kernoscope://provenance", "kernoscope://skill"}
    # 资源条目不许把仓库内部相对路径漏给客户端（那是实现细节，不是接口）
    assert all("path" not in r for r in res)


def test_tools_call_goes_through_the_cli_and_reports_the_rc():
    seen: list[list[str]] = []
    r = mcp.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "capabilities_brief", "arguments": {}}},
                   runner=_runner(seen))
    assert seen == [["capabilities", "--brief"]], seen
    assert r["result"]["isError"] is False and r["result"]["content"][0]["text"] == "OK"

    seen.clear()
    mcp.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "profile",
                           "arguments": {"program": ["python", "bench.py"],
                                         "extra_args": ["--set", "basic"]}}},
               runner=_runner(seen))
    assert seen == [["--packet", "-", "--set", "basic", "--", "python", "bench.py"]], seen

    seen.clear()
    mcp.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": {"name": "diff", "arguments": {"a": "a.json", "b": "b.json"}}},
               runner=_runner(seen))
    assert seen == [["diff", "a.json", "b.json", "--json"]], seen

    seen.clear()
    mcp.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": {"name": "selftest", "arguments": {}}}, runner=_runner(seen))
    assert seen == [["selftest", "--json"]], seen


def test_failing_tool_call_is_marked_as_error_and_keeps_stderr():
    r = mcp.handle({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                    "params": {"name": "selftest", "arguments": {}}},
                   runner=_runner([], rc=2, out="", err="error: 用法错误"))
    assert r["result"]["isError"] is True
    text = r["result"]["content"][0]["text"]
    assert "rc=2" in text and "用法错误" in text


@pytest.mark.parametrize("params", [
    {"name": "nope", "arguments": {}},
    {"name": "profile", "arguments": {}},                       # 缺 program
    {"name": "profile", "arguments": {"program": "python"}},     # 类型错
    {"name": "diff", "arguments": {"a": "a.json"}},              # 缺 b
])
def test_bad_tool_arguments_are_usage_errors(params):
    r = mcp.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": params},
                   runner=_runner([]))
    assert r["error"]["code"] == -32602, r


def test_unknown_method_and_bad_request():
    assert mcp.handle({"jsonrpc": "2.0", "id": 10,
                       "method": "no/such"})["error"]["code"] == -32601
    assert mcp.handle({"id": 11, "method": "initialize"})["error"]["code"] == -32600
    assert mcp.handle("not-an-object")["error"]["code"] == -32600


def test_notifications_get_no_response():
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert mcp.handle({"jsonrpc": "2.0", "method": "initialized"}) is None


def test_resources_read_returns_the_generated_reference():
    r = mcp.handle({"jsonrpc": "2.0", "id": 12, "method": "resources/read",
                    "params": {"uri": "kernoscope://refusal-codes"}})
    text = r["result"]["contents"][0]["text"]
    assert "refusal" in text.lower() or "拒绝" in text
    assert mcp.handle({"jsonrpc": "2.0", "id": 13, "method": "resources/read",
                       "params": {"uri": "kernoscope://nope"}})["error"]["code"] == -32602


# --------------------------------------------------------------------------
# 2) 会话循环：逐行 JSON-RPC；一行坏掉不能把长会话带走
# --------------------------------------------------------------------------
def test_serve_answers_each_line_and_survives_a_bad_one():
    seen: list[list[str]] = []
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        "{ this is not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "capabilities_brief", "arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),   # 无响应
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
    ]
    out = io.StringIO()
    rc = mcp.serve(iter(lines), out, runner=_runner(seen))
    assert rc == 0
    responses = [json.loads(ln) for ln in out.getvalue().splitlines() if ln.strip()]
    assert [r.get("id") for r in responses] == [1, None, 2, 3]
    assert responses[1]["error"]["code"] == -32700          # 坏行只影响它自己
    assert seen == [["capabilities", "--brief"]]


def test_cli_entry_refuses_anything_but_stdio(capsys):
    # 未知开关是 argparse 的用法错误（SystemExit 2）——与仓库其它子命令同一条约定
    with pytest.raises(SystemExit) as e:
        mcp.main(["--http"])
    assert e.value.code == 2
    # 而不给 --stdio 是**自家的**一行错误 + rc=2（不是静默地什么都不做）
    assert mcp.main([]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "stdio" in err


# --------------------------------------------------------------------------
# 3) 零常驻（这条是"做它"与"不做它"之间那条线的机械化）
# --------------------------------------------------------------------------
def test_mcp_is_not_in_the_resident_entry_layer():
    """常驻入口层里不许提 mcp：工具定义每轮都进上下文，常驻就等于每次会话都付费。"""
    for name in ENTRY_LAYER:
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        assert "mcp" not in text, f"{name} 提到了 mcp —— 那就不是「零常驻」了"
    front = (ROOT / "skills" / "kernoscope" / "SKILL.md").read_text(encoding="utf-8")
    fm = front.split("---")[1] if front.startswith("---") else ""
    assert "mcp" not in fm.lower(), "SKILL.md 的 frontmatter 是唯一常驻部分，不许提 mcp"


def test_mcp_module_is_import_safe():
    """import 不得启动服务、不得读 stdin（否则任何 `import kernoscope.*` 都会挂住）。"""
    import subprocess
    import sys as _sys

    p = subprocess.run([_sys.executable, "-c", "import kernoscope.mcp; print('ok')"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60,
                       env={**_env(), "PYTHONPATH": str(ROOT / "src")})
    assert p.returncode == 0 and "ok" in p.stdout, (p.stdout, p.stderr)


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)
