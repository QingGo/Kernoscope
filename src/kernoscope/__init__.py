"""Kernoscope - provenance-honest CUDA kernel profiler for locked-down GPUs.

在拿不到硬件性能计数器的机器（共享租用容器、`RmProfilingAdminOnly=1` 的环境）上，
ncu 只能报 `ERR_NVGPUCTRPERM`。Kernoscope 用"换物理观测量"的方式拿回大部分决策所需的信息，
并且强制给每个数字标注出处：

    exact        与 ncu 同源（CUPTI activity / cubin ELF / 解析算术）
    derived      由本机实测标定或解析模型推出（如 roofline）
    measured     由我们自己的时间戳 / 插桩实测
    estimated    采样或代理近似（附误差来源）
    unavailable  需要 GPU 硬件性能计数器 —— 只报 n/a 与原因，绝不编数字

用法与 ncu 兼容：

    kernoscope --set basic -k regex:gemm -c 3 -o rep -- python train.py
    kernoscope -i rep.json --csv
    kernoscope --strict -k regex:gemm -- python train.py    # 缺数据即失败，供 CI 卡门
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
