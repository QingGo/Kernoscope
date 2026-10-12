"""传输 / 通信账本的口径契约。

守的是 2026-10-12 在 2×RTX 2080 Ti（无 NVLink、无 PCIe P2P ⇒ 卡间传输只能经主机内存中转）
上取到的真机事实。它们把一个很容易做错的数变成了**必须拒绝**的数：

  * **字节可信**：CUPTI `CUPTI_ACTIVITY_KIND_MEMCPY` 的 `bytes` 逐行相加就是真值；
  * **时长不可信**：同一张表里约一半的行无论多大都报 80–95 µs
    （原文行：`313 | 0 | 7 | 2 | 1073741824 | 6481416834 | 6481497428 | 80594` ——
    1 GiB / 80 µs = 13.3 TB/s，物理上不可能）。真正的危险是**这些坏行恰好也携带了大部分字节**
    （peer.sqlite：132 行短行声称搬了 34.06 GB 却只花 12.3 ms）⇒ 逐行相加把真实 5.49 GB/s
    变成约 **3184 GB/s（错约 580 倍）**；
  * 所以单次时长只能按 `(copyKind, bytes)` 组取 **MAX** —— 真机上 MAX 恰好捞回诚实那一行
    （1 GiB 的 copyKind=1 最大 89.996 ms → 11.93 GB/s，与 cudaEvent 实测 11.96 GB/s 吻合）；
  * **通信时间要从 kernel 拿**：NCCL 的传输全在 `ncclDevKernel_*` 里，每 rank 只有 24 KB 的
    bootstrap 拷贝 —— 去 memcpy 表里找通信量会得到接近 0 的错答案。
"""

from __future__ import annotations

import pytest

from kernoscope.analyze.comm import build_ledger
from kernoscope.report import KernelLaunch, MemcpyRecord


def _copy(kind: int, nbytes: int, dur_ns: int, *, dev: int = 0, start: int = 0,
          corr: int = 0) -> MemcpyRecord:
    return MemcpyRecord(start_ns=start, end_ns=start + dur_ns, duration_ns=dur_ns,
                        copy_kind=kind, bytes=nbytes, device_id=dev, stream_id=7,
                        correlation_id=corr)


def _kernel(name: str, dur_ns: int, *, start: int = 0, dev: int = 0) -> KernelLaunch:
    return KernelLaunch(name=name, demangled=name, start_ns=start, end_ns=start + dur_ns,
                        grid=(1, 1, 1), block=(1, 1, 1), registers_per_thread=0,
                        static_smem_bytes=0, dynamic_smem_bytes=0, local_mem_per_thread=0,
                        stream_id=7, device_id=dev, grid_id=0, correlation_id=0, short_name=name)


# 真机那一组数字：1 GiB H2D 有诚实行（89.996 ms）也有坏行（80 µs）
REAL_1GIB = [
    _copy(1, 1 << 30, 89_996_000, corr=1),   # 诚实行 → 11.93 GB/s
    _copy(1, 1 << 30, 80_594, corr=2),       # 坏行 → 13.3 TB/s（物理上不可能）
    _copy(1, 1 << 30, 82_000, corr=3),       # 坏行
]


#: 真机那种**病态分布**：大多数行是坏行（短时长），而且它们携带了大部分字节。
#: 100 个 1 GiB 的坏行只花 8.5 ms（= 12 TB/s，物理上不可能）+ 1 个诚实行 180 ms。
PATHOLOGICAL = ([_copy(1, 1 << 30, 85_000, corr=i) for i in range(100)]
                + [_copy(1, 1 << 30, 180_000_000, corr=999)])


def test_bytes_are_summed_but_durations_are_not():
    led = build_ledger(REAL_1GIB, [])
    assert led.bytes_total == 3 * (1 << 30)             # 字节：逐行相加 = 真值
    assert led.copies == 3
    assert led.duration_sum_ns == 89_996_000 + 80_594 + 82_000
    # 而 MAX 口径把时长拉回物理上合理的位置（求和**低估**时长 ⇒ 高估带宽）
    biggest = [g for g in led.groups if g.bytes == (1 << 30)][0]
    assert biggest.max_duration_ns == 89_996_000
    assert biggest.achieved_gbs == pytest.approx(11.93, abs=0.05)
    assert led.implied_gbs_if_summed > biggest.achieved_gbs, "求和口径必然高估带宽"


def test_summing_durations_inflates_bandwidth_by_orders_of_magnitude():
    """病态分布下，求和口径的错会大到"数量级"级别。

    真机实测（peer.sqlite）：求和 3184 GB/s vs 实测 5.49 GB/s（约 580 倍）。这里用同一形状的
    夹具断言"至少 20 倍"——**不把 580 写死成本夹具的性质**，那是那一次采集的量。
    """
    led = build_ledger(PATHOLOGICAL, [])
    honest = [g for g in led.groups if g.bytes == (1 << 30)][0].achieved_gbs
    assert honest == pytest.approx(5.97, abs=0.05)      # 1 GiB / 180 ms
    assert led.implied_gbs_if_summed > 20 * honest


def test_max_caliber_matches_the_independently_measured_bandwidth():
    """真机对拍：CUPTI MAX 口径 11.93 vs cudaEvent 实测 11.96 GB/s（同一台机、同一次传输）。"""
    led = build_ledger(REAL_1GIB, [])
    assert led.groups[0].achieved_gbs == pytest.approx(11.96, abs=0.1)


def test_bytes_by_kind_and_device_are_both_segmented():
    led = build_ledger([
        _copy(1, 1 << 20, 90_000, dev=0),
        _copy(2, 1 << 20, 80_000, dev=0),
        _copy(1, 1 << 20, 88_000, dev=1),
    ], [])
    assert led.bytes_by_kind == {1: 2 << 20, 2: 1 << 20}
    assert led.bytes_by_device == {0: 2 << 20, 1: 1 << 20}


def test_comm_kernel_time_comes_from_kernels_not_from_memcpy_rows():
    """真机：NCCL 的传输全在 kernel 里，memcpy 表只有 24 KB —— 所以通信时间必须从 kernel 拿。"""
    kernels = [
        _kernel("ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)", 10_000_000),
        _kernel("volta_sgemm_128x128_tn", 30_000_000),
    ]
    led = build_ledger([_copy(1, 24_000, 90_000)], kernels)
    assert led.comm_kernel_ns == 10_000_000
    assert led.total_kernel_ns == 40_000_000
    assert led.comm_kernel_share_pct == pytest.approx(25.0)
    assert led.comm_kernel_names == ["ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)"]


def test_overlap_uses_comm_kernel_intervals_only():
    """重叠只对**通信 kernel** 算：拷贝时长不可信，拿它算重叠等于把 580 倍的错引进第二个数。"""
    kernels = [
        _kernel("ncclDevKernel_AllReduce_x", 100, start=0),      # comm 0–100
        _kernel("sgemm", 100, start=50),                          # compute 50–150 → 重叠 50
        _kernel("sgemm2", 100, start=400),                        # 不重叠
    ]
    led = build_ledger([_copy(1, 1 << 20, 999_999_999, start=0)], kernels)
    assert led.comm_busy_ns == 100
    assert led.overlap_ns == 50
    assert led.overlap_pct_of_comm == pytest.approx(50.0)


def test_empty_inputs_do_not_invent_a_ledger():
    led = build_ledger([], [])
    assert not led.has_any
    assert led.bytes_total == 0 and led.comm_kernel_ns == 0


# ---------------------------------------------------------------------------
# section 层：拒绝条目必须带上反面证据
# ---------------------------------------------------------------------------
def _comm_section(led):
    from kernoscope.sections import BuildOptions, build_comm_section

    return build_comm_section(BuildOptions(comm=led))


def test_section_refuses_the_summed_duration_and_carries_the_evidence():
    sec = _comm_section(build_ledger(PATHOLOGICAL, []))
    m = [x for x in sec.metrics if x.name == "ks__comm_time_sum_ns"]
    assert m and m[0].provenance.value == "unavailable"
    assert m[0].refusal is not None
    assert m[0].refusal.code.value == "memcpy_duration_untrusted"
    # 反面证据必须随拒绝一起报出来（否则读者不知道"求和"错在哪里）
    obs = m[0].refusal.observed or {}
    assert obs.get("bytes") == 101 * (1 << 30)
    # 反面证据：求和口径会得到多少 GB/s（比诚实口径高一个数量级以上）
    assert obs.get("implied_gbs_if_summed", 0) > 20 * 5.97
    assert obs.get("sum_ns") == 100 * 85_000 + 180_000_000


def test_section_reports_bytes_as_exact_and_duration_as_estimated():
    sec = _comm_section(build_ledger(REAL_1GIB, []))
    by = {x.name: x for x in sec.metrics}
    assert by["ks__comm_bytes_total"].provenance.value == "exact"
    assert by["ks__comm_bytes_htod"].value == 3 * (1 << 30)
    est = [x for x in sec.metrics if x.name.startswith("est__copy_htod_")]
    assert est and est[0].provenance.value == "estimated"


def test_no_section_when_nothing_was_collected():
    """没采到拷贝/通信 kernel ⇒ 不给 section（而不是给一堆 0）。"""
    assert _comm_section(build_ledger([], [])) is None
