"""后端：每个模块负责一类免权限数据源。

    nsys.py   CUPTI activity（exact）：LaunchStats / 时长
    nvml.py   设备级管理计数器（免 admin）：能耗 / 显存忙碌 / 时钟功耗
    cubin.py  cubin ELF（exact，不需要 GPU）：寄存器、静态 smem、SASS 静态混合
"""

from . import cubin, nsys, nvml  # noqa: F401
