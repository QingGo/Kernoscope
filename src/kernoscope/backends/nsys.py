"""nsys 后端：CUPTI activity → LaunchStats / 时长（全部 exact）。

nsys 走的是 CUPTI 的 activity 通道，**不需要**性能计数器权限 —— 这正是受限环境里
唯一保留的官方数据源。我们直接吃它的 SQLite 导出：

    nsys profile --trace=cuda,nvtx ... -o rep -- <cmd>
    nsys export --type sqlite -o rep.sqlite rep.nsys-rep

`CUPTI_ACTIVITY_KIND_KERNEL` 表自带 ncu LaunchStats 所需的全部字段（实测确认）：
registersPerThread / gridX..Z / blockX..Z / staticSharedMemory / dynamicSharedMemory /
localMemoryPerThread / gridId。
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Sequence

from ..report import DeviceInfo, KernelLaunch, MemcpyRecord

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
#: 拷贝 activity。**它是"通信/传输账本"的唯一数据源**（真机实测：在这台无 P2P 的机器上，
#: 卡间传输只能经主机内存中转，每一次都过 PCIe，所以每一次都被它记下来 —— 不需要任何硬件计数器）。
MEMCPY_TABLE = "CUPTI_ACTIVITY_KIND_MEMCPY"

#: CUPTI 的 `copyKind` → 人话。**只登记有可靠语义的那几个**，其余的按 `kind_<n>` 原样报出，
#: 不猜（猜一个名字会让下游把它当成已知口径）。
#: 真机实测（2×2080 Ti，无 P2P）：只出现 HTOD(1) 与 DTOH(2)；**PTOP(10) 一次都没出现** ——
#: 这正是"这台机器没有 PCIe P2P"的结构性证据（`cudaDeviceCanAccessPeer` 也返回 0）。
COPY_KIND_NAMES: dict[int, str] = {
    1: "htod", 2: "dtoh", 3: "htoa", 4: "atoh", 5: "atoa",
    6: "atod", 7: "dtoa", 8: "dtod", 9: "htoh", 10: "ptop",
}


class NsysError(RuntimeError):
    pass


class TargetFailed(NsysError):
    """被测**目标程序**非零退出（nsys 会把目标的退出码透传回来）。

    为什么不能并入 `NsysError`（第七轮 X-1）：旧实现把 rc≠0 一律报成
    "nsys profile failed (rc=3)"，于是**目标自己退出 3** 这件事被讲成"nsys 失败"，
    再挂上"nsys 随 CUDA Toolkit 提供（或单独安装）"的提示 —— 把人送去装一个已经装好的工具，
    而真正的原因（目标失败）一个字都没说。
    """

    def __init__(self, rc: int, *, rep_exists: bool, detail: str = "") -> None:
        super().__init__(f"target exited with rc={rc}")
        self.rc = rc
        self.rep_exists = rep_exists
        self.detail = detail


def find_nsys() -> str | None:
    """多路径发现（与 doctor 用同一份逻辑，避免"同一台机器两个答案"）。"""
    from ..toolchain import find

    p = find("nsys")
    if p:
        return p
    for cand in ("/opt/nvidia/nsight-systems/latest/nsys",
                 "/opt/nvidia/nsight-systems/*/bin/nsys"):
        import glob

        for hit in glob.glob(cand):
            if os.path.exists(hit):
                return hit
    return None


def run_profile(cmd: Sequence[str], out_prefix: str, nsys: str | None = None,
                target_processes: str = "all", extra_args: Sequence[str] = ()) -> str:
    """运行 nsys profile，返回 .nsys-rep 路径。"""
    nsys = nsys or find_nsys()
    if not nsys:
        raise NsysError("nsys not found in PATH (install Nsight Systems, bundled with CUDA >= 11.1)")
    args = [
        nsys, "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        "-o", out_prefix,
    ]
    if target_processes == "all":
        args.append("--trace-fork-before-exec=true")
    args += list(extra_args)
    args += list(cmd)
    proc = subprocess.run(args, capture_output=True, text=True)
    rep = out_prefix + ".nsys-rep"
    if proc.returncode != 0 or not os.path.exists(rep):
        # nsys 把**目标程序**的退出码透传回来。目标非零退出不是"nsys 失败"（X-1）：
        # 目标已经解析过（cli 侧先做 `_resolve_target`），所以这一条可以确定地按目标归因。
        # 只有**没有产出 repo** 时才把"nsys 自身也可能失败"作为次要可能写进 detail。
        if proc.returncode != 0:
            raise TargetFailed(
                proc.returncode, rep_exists=os.path.exists(rep),
                detail=(proc.stderr[-300:] or proc.stdout[-300:]).strip())
        raise NsysError(
            f"nsys 没有产出 .nsys-rep（它自己返回 0，这一点是 nsys 的问题）：{rep}\n"
            f"stdout tail: {proc.stdout[-800:]}\nstderr tail: {proc.stderr[-800:]}"
        )
    return rep


def export_sqlite(rep: str, db: str | None = None, nsys: str | None = None) -> str:
    nsys = nsys or find_nsys()
    if not nsys:
        raise NsysError("nsys not found")
    db = db or (rep[:-len(".nsys-rep")] + ".sqlite" if rep.endswith(".nsys-rep") else rep + ".sqlite")
    proc = subprocess.run(
        [nsys, "export", "--force-overwrite=true", "--type", "sqlite", "-o", db, rep],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.exists(db):
        raise NsysError(f"nsys export failed (rc={proc.returncode}): {proc.stderr[-500:]}")
    return db


def _unreadable_db_message(db: str, exc: sqlite3.DatabaseError) -> str:
    """坏 sqlite 的人话（V-25）：说清"这不是我们能读的数据库" + 怎么拿到对的。

    以前这里是裸 `sqlite3.DatabaseError: file is not a database`（rc=1、无自家 error 行）。
    """
    return (f"{db} 不是可读的 nsys SQLite 导出：{exc}。"
            f"确认它是 `nsys export --type sqlite -o <db> <rep.nsys-rep>` 的产物 —— "
            f".nsys-rep 本身不是 SQLite，要先 export；文件被截断就重新导出")


def has_kernel_table(db: str) -> bool:
    try:
        with sqlite3.connect(db) as con:
            row = con.execute(
                "select name from sqlite_master where type='table' and name=?", (KERNEL_TABLE,)
            ).fetchone()
    except sqlite3.DatabaseError as e:
        raise NsysError(_unreadable_db_message(db, e)) from e
    return row is not None


def has_memcpy_table(db: str) -> bool:
    """有没有拷贝 activity 表。

    没有它**不等于没有传输** —— 只有 `--trace=cuda` 才采 CUPTI 的拷贝活动；
    调用方必须把这一点当成"拿不到"，而不是当成"传输为 0"。
    """
    try:
        with sqlite3.connect(db) as con:
            row = con.execute(
                "select name from sqlite_master where type='table' and name=?", (MEMCPY_TABLE,)
            ).fetchone()
    except sqlite3.DatabaseError as e:
        raise NsysError(_unreadable_db_message(db, e)) from e
    return row is not None


def parse_memcpys(db: str) -> list[MemcpyRecord]:
    """读取拷贝 activity（start/end/bytes/copyKind/deviceId/streamId）。

    **口径警告（真机实测，2×2080 Ti / 2026-10-12）**：在这台无 P2P 的机器上，运行时把一次
    逻辑拷贝拆成多行，且其中约一半的行的 `end-start` **无论多大都报 80–95 µs**
    （原文一行：`313 | 0 | 7 | 2 | 1073741824 | 6481416834 | 6481497428 | 80594` —— 1 GiB / 80 µs
    = 13.3 TB/s，物理上不可能）。所以：

      * **`bytes` 可信**（逐行相加 = 真值）；
      * **`end-start` 不可信**（逐行相加把 5.49 GB/s 变成 3184 GB/s，错约 580 倍）。

    解析层照原样把两样都读出来，**判定交给 `analyze/comm.py`**，那里会把"对时长求和"这条路
    显式拒绝掉（而不是给一个偏小的数）。
    """
    if not os.path.exists(db):
        raise NsysError(f"sqlite not found: {db}")
    if not has_memcpy_table(db):
        raise NsysError(
            f"{db} has no {MEMCPY_TABLE} table - was the profile collected with --trace=cuda?")
    q = f"""
        select start, end, deviceId, streamId, copyKind, bytes, correlationId
        from {MEMCPY_TABLE}
        order by start
    """
    out: list[MemcpyRecord] = []
    with sqlite3.connect(db) as con:
        for row in _rows_or_nsys_error(con, q, db):
            start, end, dev, stream, kind, nbytes, corr = row
            # R-140：`bytes` / `deviceId` 为 NULL 时**保留 None**，不要 `or 0` ——
            # 静默折算会把"没读到"说成"0 字节 / 0 号卡"，而下游把合计标成 exact。
            out.append(MemcpyRecord(
                start_ns=int(start), end_ns=int(end), duration_ns=int(end) - int(start),
                copy_kind=int(kind or 0),
                bytes=None if nbytes is None else int(nbytes),
                device_id=None if dev is None else int(dev), stream_id=int(stream or 0),
                correlation_id=int(corr or 0),
            ))
    return out


def _string_ids(con: sqlite3.Connection) -> dict[int, str]:
    """nsys SQLite 把 kernel 名存成 StringIds 表的整数外键，需要自己 join。

    实测踩坑：列名看着是 `demangledName`，但**值是整数 id**；不解引用会得到
    `TypeError: expected string or bytes-like object, got 'int'`（而且只在真数据上出现）。
    少数版本可能内联字符串，因此两种都支持。
    """
    ids: dict[int, str] = {}
    try:
        for i, v in con.execute("select id, value from StringIds"):
            ids[int(i)] = v
    except sqlite3.OperationalError:
        pass
    return ids


def _resolve(value, ids: dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return ids.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def _rows_or_nsys_error(con: sqlite3.Connection, q: str, db: str):
    """流式读行（不把整表读进内存）；读到一半发现坏页/截断也要变成 `NsysError`。

    V-25：`has_kernel_table` 通过之后文件仍可能在后半段坏掉。
    """
    try:
        yield from con.execute(q)
    except sqlite3.DatabaseError as e:
        raise NsysError(_unreadable_db_message(db, e)) from e


def parse_kernels(db: str) -> list[KernelLaunch]:
    """读取 kernel activity 记录。时间单位 ns（nsys SQLite 约定）。"""
    if not os.path.exists(db):
        raise NsysError(f"sqlite not found: {db}")
    if not has_kernel_table(db):
        raise NsysError(
            f"{db} has no {KERNEL_TABLE} table - was the profile collected with --trace=cuda?"
        )
    q = f"""
        select start, end, deviceId, streamId, demangledName, shortName, mangledName,
               registersPerThread, gridX, gridY, gridZ, blockX, blockY, blockZ,
               staticSharedMemory, dynamicSharedMemory, localMemoryPerThread,
               gridId, correlationId
        from {KERNEL_TABLE}
        order by start
    """
    out: list[KernelLaunch] = []
    with sqlite3.connect(db) as con:
        ids = _string_ids(con)
        for row in _rows_or_nsys_error(con, q, db):
            (start, end, dev, stream, demangled, short, mangled, regs,
             gx, gy, gz, bx, by, bz, ssmem, dsmem, lmem, grid_id, corr) = row
            short_s = _resolve(short, ids)
            demangled_s = _resolve(demangled, ids)
            mangled_s = _resolve(mangled, ids)
            out.append(KernelLaunch(
                name=demangled_s or short_s or mangled_s or "<unknown>",
                demangled=demangled_s or short_s or mangled_s or "<unknown>",
                short_name=short_s, mangled=mangled_s,
                start_ns=int(start), end_ns=int(end),
                grid=(int(gx), int(gy), int(gz)), block=(int(bx), int(by), int(bz)),
                registers_per_thread=int(regs or 0),
                static_smem_bytes=int(ssmem or 0), dynamic_smem_bytes=int(dsmem or 0),
                local_mem_per_thread=int(lmem or 0),
                stream_id=int(stream or 0), device_id=int(dev or 0),
                grid_id=int(grid_id or 0), correlation_id=int(corr or 0),
            ))
    return out


def sqlite_from_rep(rep: str, workdir: str | None = None) -> str:
    d = workdir or tempfile.mkdtemp(prefix="kernoscope-nsys-")
    return export_sqlite(rep, os.path.join(d, os.path.basename(rep) + ".sqlite"))


# --------------------------------------------------------------------------
# 设备信息（nvidia-smi，免权限）
# --------------------------------------------------------------------------
SMI_FIELDS = ("name,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit")


def device_info_from_smi() -> DeviceInfo:
    from ..toolchain import find

    smi = find("nvidia-smi")
    if not smi:
        return DeviceInfo()
    q = ("--query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,"
         "l2CacheSize")
    proc = subprocess.run([smi, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return DeviceInfo()
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    while len(parts) < 7:
        parts.append("0")
    name, driver, cc, mem, clk, pwr, l2 = parts[:7]

    sm_count = 0
    ccq = subprocess.run([smi, "--query-gpu=clocks.max.sm", "--format=csv,noheader"],
                         capture_output=True, text=True)
    del ccq  # sm_count 由 torch/CUPTI 侧补充；这里保持 0 表示未知
    return DeviceInfo(
        name=name, driver=driver, compute_capability=cc,
        memory_total_mib=_f(mem), max_sm_clock_mhz=_f(clk), power_limit_w=_f(pwr),
        l2_mib=_f(l2) / 1024.0 if _f(l2) > 4096 else _f(l2),  # 有的驱动给 bytes，有的给 MiB
    )


def _f(s: str) -> float:
    try:
        return float(str(s).split()[0])
    except Exception:
        return 0.0
