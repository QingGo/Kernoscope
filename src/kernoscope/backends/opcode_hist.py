"""NVBit 官方 `opcode_hist` 输出的解析器（指令级动态计数）。

为什么用它：动态指令数是唯一能在受限环境里拿到的"CPU 侧看不到、又确实来自硬件"的量。
它让我们给出**发射槽利用率的上界**，从而补上"主要瓶颈方向"这一类结论。

官方工具的输出格式（`tools/opcode_hist`，实测 NVBit 1.8）：

    kernel 0 - copy_k(float const*, float*, long) - #thread-blocks 1024,  kernel instructions 1835008, total instructions 1835008
      LDG.E.128 = 65536
      FADD = 4096
      ...

关键语义（读源码确认）：`count_warp_level` 默认为 1，即**每个 warp 的每条指令执行只记 1 次**
（`inject_funcs.cu`：只有首个 active lane 做一次 atomicAdd），而不是每线程各记一次。
这正好是"消耗了多少个发射槽"，所以可以直接用来算发射槽利用率。

⚠️ 工具是 NVIDIA 专有许可，**不随本项目分发**；这里只解析它产出的文本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# kernel 0 - <name> - #thread-blocks 1024,  kernel instructions 1835008, total instructions 1835008
KERNEL_RE = re.compile(
    r"^kernel\s+(?P<id>\d+)\s+-\s+(?P<name>.+?)\s+-\s+#thread-blocks\s+(?P<blocks>\d+)\s*,"
    r"\s*kernel\s+instructions\s+(?P<instrs>\d+)\s*,\s*total\s+instructions\s+(?P<total>\d+)\s*$")
OPCODE_RE = re.compile(r"^\s+(?P<op>[A-Za-z][A-Za-z0-9._]*)\s*=\s*(?P<count>\d+)\s*$")


class OpcodeHistError(ValueError):
    """轨迹/输出格式不对时抛出（调用方给可读错误，不抛裸异常）。"""


@dataclass
class LaunchHistogram:
    kernel_id: int
    kernel: str
    thread_blocks: int
    warp_instrs: int            # 这条 launch 消耗的 warp 级指令数（= 发射槽占用）
    total_instrs: int
    opcode_counts: dict[str, int] = field(default_factory=dict)
    n_launches: int = 1         # 合并后代表多少次 launch（解析单条时为 1）

    @property
    def top_opcodes(self) -> list[tuple[str, int]]:
        return sorted(self.opcode_counts.items(), key=lambda kv: -kv[1])


@dataclass
class OpcodeHist:
    launches: list[LaunchHistogram] = field(default_factory=list)

    @property
    def kernels(self) -> list[str]:
        seen: list[str] = []
        for l in self.launches:
            if l.kernel not in seen:
                seen.append(l.kernel)
        return seen

    def for_kernel(self, needle: str) -> "OpcodeHist":
        """按名字子串筛选（与 -k 的语义一致：不区分大小写）。"""
        n = (needle or "").lower()
        return OpcodeHist([l for l in self.launches if n in l.kernel.lower()])

    def merged(self) -> LaunchHistogram:
        """把多条 launch 合并成一个直方图（同名 kernel 的多次调用）。"""
        if not self.launches:
            raise OpcodeHistError("没有解析到任何 kernel launch 记录")
        ops: dict[str, int] = {}
        for l in self.launches:
            for k, v in l.opcode_counts.items():
                ops[k] = ops.get(k, 0) + v
        first = self.launches[0]
        return LaunchHistogram(
            kernel_id=first.kernel_id, kernel=first.kernel,
            thread_blocks=sum(l.thread_blocks for l in self.launches),
            warp_instrs=sum(l.warp_instrs for l in self.launches),
            total_instrs=sum(l.total_instrs for l in self.launches),
            opcode_counts=ops, n_launches=len(self.launches))


def parse(text: str) -> OpcodeHist:
    """解析 `opcode_hist` 的 stdout。

    容错策略：官方工具的输出里还混着 NVBit 横幅（`NVBit: ...`、`----------------`），
    所以只认严格匹配的行；**一行都没匹配到时**报错（而不是返回空直方图）——
    空直方图会被下游当成"指令数为 0"，那是假数据。
    """
    hist = OpcodeHist()
    cur: LaunchHistogram | None = None
    for line in text.splitlines():
        m = KERNEL_RE.match(line.strip())
        if m:
            cur = LaunchHistogram(
                kernel_id=int(m.group("id")), kernel=m.group("name"),
                thread_blocks=int(m.group("blocks")),
                warp_instrs=int(m.group("instrs")), total_instrs=int(m.group("total")))
            hist.launches.append(cur)
            continue
        mo = OPCODE_RE.match(line)
        if mo and cur is not None:
            cur.opcode_counts[mo.group("op")] = int(mo.group("count"))
    if not hist.launches:
        raise OpcodeHistError(
            "没有解析到任何 'kernel N - <name> - #thread-blocks ...' 行。"
            "请用 NVBit 官方 opcode_hist 工具采集（NO_EAGER_LOAD=0），"
            "并把它的完整 stdout 存成文件")
    return hist


def load(path: str) -> OpcodeHist:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return parse(f.read())
    except OSError as e:
        raise OpcodeHistError(f"读不了文件 {path}: {e}") from e
