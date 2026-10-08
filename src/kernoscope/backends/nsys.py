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
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Sequence

from ..report import DeviceInfo, KernelLaunch

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"


class NsysError(RuntimeError):
    pass


def find_nsys() -> str | None:
    for cand in ("nsys", "/usr/local/bin/nsys", "/opt/nvidia/nsight-systems/latest/nsys"):
        p = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if p:
            return p
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
        raise NsysError(
            f"nsys profile failed (rc={proc.returncode})\n"
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


def has_kernel_table(db: str) -> bool:
    with sqlite3.connect(db) as con:
        row = con.execute(
            "select name from sqlite_master where type='table' and name=?", (KERNEL_TABLE,)
        ).fetchone()
    return row is not None


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
        for row in con.execute(q):
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
    smi = shutil.which("nvidia-smi")
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
