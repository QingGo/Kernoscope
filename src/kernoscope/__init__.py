"""Kernoscope - provenance-honest CUDA kernel profiler for locked-down GPUs.

在拿不到硬件性能计数器的机器（共享租用容器、`RmProfilingAdminOnly=1` 的环境）上，
ncu 只能报 `ERR_NVGPUCTRPERM`。Kernoscope 用"换物理观测量"的方式拿回大部分决策所需的信息，
并且强制给每个数字标注出处：

    exact        与 ncu 同源（CUPTI activity / cubin ELF / 解析算术）
    derived      由实测/精确量再算出来的统计量与判词（采样窗覆盖率、账本误差、区间判词）
    measured     我们真的读到的量（NVML 管理计数器、探针实测的达到 occupancy、轨迹直接计数）
    estimated    模型或代理（roofline、DRAM 字节代理、缓存模型命中率）
    unavailable  需要 GPU 硬件性能计数器 —— 只报 n/a 与原因，绝不编数字

命名纪律：`meas__` 前缀只能在出处是 measured 时使用（R-12），
`est__` 是 estimated/模型量，`ks__` 是我们自有的量（可以是 derived/exact）。
详见 PROVENANCE.md。

用法与 ncu 兼容：

    kernoscope --set basic -k regex:gemm -c 3 -o rep -- python train.py
    kernoscope -i rep.json --csv
    kernoscope --strict -k regex:gemm -- python train.py    # 缺数据即失败，供 CI 卡门
"""

__version__ = "0.2.3"

__all__ = ["__version__"]
