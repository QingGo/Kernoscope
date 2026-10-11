# 能力覆盖（三态）与能力分层

本文件由 `kernoscope capabilities` 生成（脚本 `tools/gen_coverage.py` 调用
`kernoscope.capabilities`）；**改动请改 `src/kernoscope/capabilities.py` 后重跑，不要手改本文件**。

要看你**这台机器**的实际状态，请现场跑 `kernoscope capabilities --json`（机器可读）或 `kernoscope capabilities`（一屏）。

<!-- question ids（与 kernoscope.capabilities.QUESTIONS 同源）: kernel_time, launch_configs, static_resources, theoretical_occupancy, device_busy, cost_per_token, dram_bytes, roofline, achieved_occupancy, dram_bytes_trace, issue_slot, l2_locality, static_mix, stall_breakdown, exact_dram, pipe_utilization, nvlink -->

## 三态怎么读

| 标记 | 含义 |
|---|---|
| 能答 | 口径就绪：这一层的数据源与前提都在，数字可以引用 |
| 更弱 | 能答，但口径更弱或有前提（例如缺本机标定、缺工作量），引用时必须同时引用前提 |
| 不答 | **不给数字**，只报 `n/a` + 结构化拒绝（code + 原因 + 替代口径） |

二态会逼我们在「能答但弱」时撒谎，所以这里是三态。

## 四层能力分层

| 层 | 名字 | 需要的依赖 | 这一层能给 |
|---|---|---|---|
| 0 | `demo` | 无（不需要 GPU） | 用合成数据看输出形状与拒绝格式；任何数字都不是测量 |
| 1 | `device` | NVIDIA GPU + NVML（nvidia-smi / pynvml） | 设备级实测：显存忙碌占比、SM 时钟、功耗、能量；给出 DRAM 字节代理（需标定峰值） |
| 2 | `trace` | nsys（CUPTI activity） | kernel 级精确量：时长、launch 配置、寄存器/共享内存、occupancy 上界 |
| 3 | `instrumented` | NVBit 自行编译（+ nvcc） | 指令级：访存轨迹（sectors/request、L2 命中模型）、动态指令数、发射槽利用率上界 |
| 4 | `counters` | ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0） | 与 ncu 同源的硬件计数器：精确 DRAM 字节、各 pipe 利用率、stall 周期细分 |

层是可以叠加的：层 N 可用意味着层 < N 也可用。最高可用层由**探测结果**决定，不由用户选择；缺件只会降级，不会假装可用。

## 三态覆盖矩阵（问题 × 能力层）

| 问题 id | 问的是什么 | 层 0 | 层 1 | 层 2 | 层 3 | 层 4 |
|---|---|---|---|---|---|---|
| `kernel_time` | 哪个 kernel 最耗时、每次跑多久 | 不答 | 不答 | 能答 | 能答 | 能答 |
| `launch_configs` | 同名 kernel 有几组 launch 配置，各自多快 | 不答 | 不答 | 能答 | 能答 | 能答 |
| `static_resources` | 静态寄存器/共享内存，以及谁是 occupancy 限制因素 | 不答 | 不答 | 能答 | 能答 | 能答 |
| `theoretical_occupancy` | occupancy 理论上界是多少 | 不答 | 不答 | 能答 | 能答 | 能答 |
| `device_busy` | 这段时间 GPU 显存子系统有多忙（实测） | 不答 | 能答 | 能答 | 能答 | 能答 |
| `cost_per_token` | 这次推理/训练的单位成本（每百万 token 多少钱 / 每小时多少钱） | 不答 | 能答 | 能答 | 能答 | 能答 |
| `dram_bytes` | DRAM 搬了多少字节 / 是不是 DRAM bound（零依赖的 NVML 代理口径；有 NVBit 时首选轨迹口径，精度高 4 个数量级） | 不答 | 更弱 | 能答 | 能答 | 能答 |
| `roofline` | 离 roofline 多远、compute 还是 memory bound | 不答 | 更弱 | 能答 | 能答 | 能答 |
| `achieved_occupancy` | 实际达到的 occupancy（不是上界） | 不答 | 不答 | 不答 | 能答 | 能答 |
| `dram_bytes_trace` | DRAM 字节的**首选**口径（访存轨迹：地址是算术，只有缓存替换是模型） | 不答 | 不答 | 不答 | 能答 | 能答 |
| `issue_slot` | 发射槽用了多少、瓶颈方向是 latency 还是 throughput | 不答 | 不答 | 不答 | 能答 | 能答 |
| `l2_locality` | L1/L2 命中率与工作集是否驻留 | 不答 | 不答 | 不答 | 能答 | 能答 |
| `static_mix` | 静态 SASS 指令混合（哪些指令占多数） | 不答 | 不答 | 能答 | 能答 | 能答 |
| `stall_breakdown` | warp 到底在等什么（stall 周期细分） | 不答 | 不答 | 不答 | 不答 | 能答 |
| `exact_dram` | DRAM 字节/L2 命中率的精确值（与 ncu 同源） | 不答 | 不答 | 不答 | 不答 | 能答 |
| `pipe_utilization` | 各 pipe（FP32/INT/Tensor）的动态利用率 | 不答 | 不答 | 不答 | 不答 | 能答 |
| `nvlink` | NVLink 的流量与利用率 | 不答 | 不答 | 不答 | 不答 | 不答 |

五个代表环境（只给「刚好够到该层」的事实，不是某台具体机器）：

| 层 | 这一层的样子 | 还缺什么 |
|---|---|---|
| 0 | 无 GPU：只能跑合成数据 | 任何 NVIDIA 设备 |
| 1 | GPU + NVML（设备级读数） | nsys（kernel 级）、本机标定 |
| 2 | 再加 nsys + 本机标定 + 工作量 + cuobjdump/cubin | NVBit（指令级/访存轨迹） |
| 3 | 再加 NVBit（轨迹/直方图/探针），计数器仍被拒 | 计数器权限（层 4） |
| 4 | 计数器可用（与 ncu 同源） | ——（已到最高层） |

## 必然被拒的问题（层 3：非计数器手段都装齐、但没有计数器权限）

这一层是**受限多租户实例的典型形态**：`nsys` + NVBit 都可用，但宿主 `RmProfilingAdminOnly=1` 且容器无 `CAP_SYS_ADMIN`/`CAP_PERFMON`。下面这些问题在这一层**仍然必然被拒** —— 与 ncu 同源的硬件计数器我们不给替代数字。

| 问题 id | 问的是什么 | 拒绝码 | 原因 | 替代口径 |
|---|---|---|---|---|
| `stall_breakdown` | warp 到底在等什么（stall 周期细分） | `counters_denied` | 硬件计数器不可用（状态 denied）：这一项与 ncu 同源，本工具不给替代数字 | issue_slot 的方向性判断（latency vs throughput） |
| `exact_dram` | DRAM 字节/L2 命中率的精确值（与 ncu 同源） | `counters_denied` | 硬件计数器不可用（状态 denied）：这一项与 ncu 同源，本工具不给替代数字 | dram_bytes（代理，带误差带）或 dram_bytes_trace（模型） |
| `pipe_utilization` | 各 pipe（FP32/INT/Tensor）的动态利用率 | `counters_denied` | 硬件计数器不可用（状态 denied）：这一项与 ncu 同源，本工具不给替代数字 | issue_slot 的上界口径 + roofline 的达成百分比 |
| `nvlink` | NVLink 的流量与利用率 | `no_nvlink` | NVLink 计数不可得（单卡实例上这个问题本来就不适用） | 单卡/无 NVLink 实例上这个问题不适用 |

层 0–2 被拒的更多 —— 那些是**环境缺件**（没有 GPU / 没有 nsys / 没给输入），每一行都带可执行的补救动作与代价，见 `kernoscope capabilities` 的「下一步」清单。层 4（计数器可用）会解锁上表里的计数器问题。

## 层 3 的逐问读数（含会产出的指标）

| 问题 | 三态 | 需要的最低层 | 会产出的指标 |
|---|---|---|---|
| `kernel_time` — 哪个 kernel 最耗时、每次跑多久 | yes | 2 | `gpu__time_duration.sum`, `ks__launch_time_mean` |
| `launch_configs` — 同名 kernel 有几组 launch 配置，各自多快 | yes | 2 | `ks__launch_time_mean`, `launch__grid_size`, `launch__block_size` |
| `static_resources` — 静态寄存器/共享内存，以及谁是 occupancy 限制因素 | yes | 2 | `ks__registers_used_for_occupancy`, `ks__occupancy_limiter` |
| `theoretical_occupancy` — occupancy 理论上界是多少 | yes | 2 | `ks__theoretical_occupancy_pct`, `ks__blocks_per_sm` |
| `device_busy` — 这段时间 GPU 显存子系统有多忙（实测） | yes | 1 | `meas__nvml_mem_busy_mean_pct`, `meas__device_energy_j` |
| `cost_per_token` — 这次推理/训练的单位成本（每百万 token 多少钱 / 每小时多少钱） | yes | 1 | `ks__cost_per_mtok`, `ks__tokens_per_s`, `ks__j_per_token` |
| `dram_bytes` — DRAM 搬了多少字节 / 是不是 DRAM bound（零依赖的 NVML 代理口径；有 NVBit 时首选轨迹口径，精度高 4 个数量级） | yes | 1 | `est__dram_bytes_total`, `ks__sampler_window_coverage_pct` |
| `roofline` — 离 roofline 多远、compute 还是 memory bound | yes | 1 | `est__achieved_gbs`, `est__bound`, `est__roofline_efficiency_pct` |
| `achieved_occupancy` — 实际达到的 occupancy（不是上界） | yes | 3 | `meas__achieved_occupancy_pct` |
| `dram_bytes_trace` — DRAM 字节的**首选**口径（访存轨迹：地址是算术，只有缓存替换是模型） | yes | 3 | `est__dram_bytes_per_launch_trace`, `meas__sectors_per_request` |
| `issue_slot` — 发射槽用了多少、瓶颈方向是 latency 还是 throughput | yes | 3 | `est__issue_slot_utilization_pct`, `est__stall_direction` |
| `l2_locality` — L1/L2 命中率与工作集是否驻留 | yes | 3 | `est__l2_hit_rate_pct`, `est__working_set_bytes` |
| `static_mix` — 静态 SASS 指令混合（哪些指令占多数） | yes | 2 | `ks__static_sass_instructions`, `ks__static_sass_share_*` |
| `stall_breakdown` — warp 到底在等什么（stall 周期细分） | no | 4 | `smsp__warp_issue_stalled_*_per_warp_active.pct` |
| `exact_dram` — DRAM 字节/L2 命中率的精确值（与 ncu 同源） | no | 4 | `dram__bytes.sum`, `lts__t_sector_hit_rate.pct` |
| `pipe_utilization` — 各 pipe（FP32/INT/Tensor）的动态利用率 | no | 4 | `sm__pipe_*.avg.pct_of_peak_sustained_elapsed` |
| `nvlink` — NVLink 的流量与利用率 | no | 4 | `nvlink__*` |
