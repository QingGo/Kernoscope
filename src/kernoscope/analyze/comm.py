"""传输 / 通信账本 —— 把"哪些量可信、哪些不可信"钉在代码里，而不是写在文档里。

## 为什么有这个文件

`docs/STRATEGY.md` 的 O1（训练·多机的 step 账本）与 O2（分层卸载账本）共用同一块硬骨头：
**传输时间的口径**。而 2026-10-12 在 2×RTX 2080 Ti 上的真机取证把这块骨头定死了：

| 量 | 出处等级 | 依据 |
|---|---|---|
| 拷贝**字节数** | `exact` | CUPTI 逐行相加 = 真值 |
| 拷贝**时长** | **拒绝求和** | 约一半的行无论多大都报 80–95 µs；求和把 5.49 GB/s 变成 **3184 GB/s（错 ~580 倍）** |
| 单次拷贝时长 | `estimated` | 按 `(copyKind, bytes)` 组取 **MAX** —— 真机上 MAX 恰好捞回诚实那一行（1 GiB: 89.996 ms → 11.93 GB/s，与 cudaEvent 实测 11.96 GB/s 吻合） |
| 通信 kernel 时间 | `exact` | NCCL 的传输时间**全部**在 kernel 里（每 rank 只有 24 KB 的 bootstrap 拷贝），kernel 时长来自 CUPTI activity，可靠 |
| 通信与计算的重叠 | `exact` | 用**通信 kernel 的区间**与计算 kernel 的区间求交（不碰不可信的拷贝时长） |
| NVLink 流量 | `unavailable` | 硬件本身不存在（`all links are inActive`） |
| PCIe 链路代数/宽度、Replays | `measured` | 免 admin 可读（比原计划"链路计数器 = unavailable"更好） |

这张表就是 `docs/STRATEGY.md` §7 要求"先定死"的出处规则；本模块是它的可执行版本，
`tests/test_comm.py` 逐条对拍。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: 通信 kernel 的识别（真机实测的 NCCL 名：`ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)`）。
#: 这是**形状匹配、不是语义推断**：匹配不到就诚实说"没识别出通信 kernel"，不假装通信为 0。
COMM_KERNEL_MARKERS = ("nccl", "allreduce", "allgather", "reducescatter", "reduce_scatter",
                       "broadcast", "sendrecv", "p2p_", "c10d")


@dataclass
class CopyGroup:
    """一组 `(copyKind, bytes)` 相同的拷贝 —— MAX 口径的最小单位。"""

    copy_kind: int
    bytes: int
    count: int = 0
    max_duration_ns: int = 0

    @property
    def usable(self) -> bool:
        """这一组的时长**能不能**用来算带宽。

        R-141：`max_duration_ns <= 0`（时长为 0/负，实测 1 GiB / 0 ns 可达）时，
        `achieved_gbs` 会算成 0 并把 `0 GB/s [estimated]` 发布出去 —— 那是**没有测量**
        而不是"测到 0"。这一档必须拒绝，不能发布。
        """
        return self.max_duration_ns > 0 and self.bytes > 0

    @property
    def achieved_gbs(self) -> float:
        """按 MAX 时长算的达成带宽（`estimated`）。`usable` 为假时返回 0，调用方必须先判它。"""
        if not self.usable:
            return 0.0
        return self.bytes / (self.max_duration_ns / 1e9) / 1e9


@dataclass
class CommLedger:
    """一次采集的传输/通信账本。"""

    #: 拷贝总字节（`exact`：逐行相加）
    bytes_total: int = 0
    bytes_by_kind: dict[int, int] = field(default_factory=dict)
    #: 设备号 → 字节。键可以是 `None`（deviceId 为 NULL）：不并进 0 号卡（R-140）。
    bytes_by_device: dict[int | None, int] = field(default_factory=dict)
    copies: int = 0
    #: 按 (copyKind, bytes) 分组的 MAX 口径时长
    groups: list[CopyGroup] = field(default_factory=list)
    #: **不可信**的时长求和 —— 只为了把它连同证据一起报成拒绝，绝不当成结论
    duration_sum_ns: int = 0
    #: 通信 kernel（NCCL 等）的时间与占比（`exact`）
    comm_kernel_ns: int = 0
    total_kernel_ns: int = 0
    comm_kernel_names: list[str] = field(default_factory=list)
    #: 通信 kernel 与计算 kernel 的重叠（`exact`，区间求交）
    comm_busy_ns: int = 0
    overlap_ns: int = 0
    #: R-140：`bytes` 为 NULL 的行数。它们的字节**不计入**任何合计 —— 静默当成 0 会把
    #: "没读到"说成"搬了 0 字节"，而两个合计都被标成 `exact`。
    null_bytes_rows: int = 0
    #: R-140：`deviceId` 为 NULL 的行数。它们单列成 `unknown`，**不并进 0 号卡**。
    unknown_device_rows: int = 0
    #: R-141：时长 ≤ 0 的行数（"没有测量" ≠ "测到 0"）。
    nonpositive_duration_rows: int = 0

    @property
    def has_any(self) -> bool:
        return bool(self.copies or self.comm_kernel_ns)

    @property
    def comm_kernel_share_pct(self) -> float:
        return (100.0 * self.comm_kernel_ns / self.total_kernel_ns
                if self.total_kernel_ns else 0.0)

    @property
    def implied_gbs_if_summed(self) -> float:
        """如果把逐行时长加起来会得到的带宽（**故意算出来当反面证据**）。"""
        if self.duration_sum_ns <= 0 or self.bytes_total <= 0:
            return 0.0
        return self.bytes_total / (self.duration_sum_ns / 1e9) / 1e9

    @property
    def overlap_pct_of_comm(self) -> float:
        return 100.0 * self.overlap_ns / self.comm_busy_ns if self.comm_busy_ns else 0.0


def _is_comm_kernel(name: str) -> bool:
    low = (name or "").lower()
    return any(m in low for m in COMM_KERNEL_MARKERS)


def _merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """把区间按起点排序并合并重叠（半开区间；空区间丢弃）。"""
    out: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _union_len(intervals: Iterable[tuple[int, int]]) -> int:
    """区间的并集长度（重叠只算一次）。"""
    return sum(e - s for s, e in _merge(intervals))


def _intersection_len(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """两组区间的交集总长。

    **两边都必须先去重**（R-139）：`comm_busy_ns` 用的是真并集，若这里按重复计数做交集，
    分子就能大于分母 —— 实测 `comm=[(0,1000),(0,1000)]`（两张卡各一个完全重合的 NCCL kernel）
    会给出 `ks__comm_compute_overlap_pct = 200%`，而且还标着 `exact`。
    一个"比例"超过 100% 是没有意义的。所以先各自合并，再求交。
    """
    i = j = 0
    total = 0
    a, b = _merge(a), _merge(b)
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def build_ledger(memcpys: list[Any], kernels: list[Any]) -> CommLedger:
    """从拷贝 activity 与 kernel launch 算出账本。

    参数只要求有对应属性（`MemcpyRecord` / `KernelLaunch`），便于单测直接喂夹具。
    """
    led = CommLedger()
    by_group: dict[tuple[int, int], CopyGroup] = {}
    for m in memcpys:
        led.copies += 1
        nbytes = getattr(m, "bytes", None)
        dur = int(getattr(m, "duration_ns", 0) or 0)
        dev = getattr(m, "device_id", None)
        led.duration_sum_ns += dur
        if dur <= 0:
            led.nonpositive_duration_rows += 1
        if dev is None:
            # R-140：NULL deviceId 不是 0 号卡 —— 并进 0 会把"不知道哪张卡"说成"卡 0"
            led.unknown_device_rows += 1
        if nbytes is None:
            # R-140：NULL 字节**不计入任何合计**（合计标的是 exact，不能把"没读到"当 0 搬）
            led.null_bytes_rows += 1
            continue
        nbytes = int(nbytes)
        led.bytes_total += nbytes
        led.bytes_by_kind[int(m.copy_kind)] = (
            led.bytes_by_kind.get(int(m.copy_kind), 0) + nbytes)
        led.bytes_by_device[dev if dev is None else int(dev)] = (
            led.bytes_by_device.get(dev if dev is None else int(dev), 0) + nbytes)
        key = (int(m.copy_kind), nbytes)
        g = by_group.get(key)
        if g is None:
            g = by_group[key] = CopyGroup(copy_kind=key[0], bytes=nbytes)
        g.count += 1
        g.max_duration_ns = max(g.max_duration_ns, dur)
    led.groups = sorted(by_group.values(), key=lambda g: (-g.bytes, g.copy_kind))

    comm_iv: list[tuple[int, int]] = []
    comp_iv: list[tuple[int, int]] = []
    names: list[str] = []
    for k in kernels:
        dur = int(getattr(k, "duration_ns", 0) or 0)
        led.total_kernel_ns += dur
        iv = (int(getattr(k, "start_ns", 0)), int(getattr(k, "end_ns", 0)))
        if _is_comm_kernel(getattr(k, "name", "")):
            led.comm_kernel_ns += dur
            comm_iv.append(iv)
            if k.name not in names:
                names.append(k.name)
        else:
            comp_iv.append(iv)
    led.comm_kernel_names = names
    led.comm_busy_ns = _union_len(comm_iv)
    # 只对**通信 kernel** 算重叠：拷贝那一侧的时长不可信（见模块 docstring），
    # 拿它算重叠等于把 580 倍的错引进另一个数。
    led.overlap_ns = _intersection_len([(s, e) for s, e in comm_iv], comp_iv)
    return led
