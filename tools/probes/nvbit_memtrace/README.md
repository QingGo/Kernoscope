# NVBit 访存轨迹工具（Kernoscope 适配层）

把被测 kernel 的**每一条 warp 级访存指令**记录下来，产出一份 `ks-memtrace v1`
文本轨迹，交给 `kernoscope memtrace` 分析。

它补的是硬件计数器拿不到的那一块：

| 指标 | 性质 | 对应 ncu |
|---|---|---|
| `sectors/request` | 实测地址的**精确算术** | `l1tex__average_t_sectors_per_request` |
| 内层合并效率 | 实测地址的**精确算术** | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 派生 |
| 触达工作集 | 实测地址去重 | —— |
| L1/L2 命中率 | **LRU 模型的上界**（`estimated`） | `lts__t_sector_hit_rate.pct` |
| DRAM 字节/launch | 模型 L2 miss × 32B × 采样放大（`estimated`） | `dram__bytes.sum` |

## 依赖与许可

* 需要 **NVBit**（NVIDIA 专有许可）。本项目**不分发** NVBit 本体，也不分发用它构建的二进制；
  本目录只有我们自己的源码，编译时链接你自己那份 NVBit。
* NVBit 下载：<https://research.nvidia.com/labs/accelferry/nvbit/>（或 NVIDIA 分发的压缩包）。

## 构建

```bash
make NVBIT_PATH=$HOME/nvbit_release_x86_64
# 架构：ARCH=native（默认）/ all-major / sm_89 …（注入代码必须与目标 GPU 匹配）
make NVBIT_PATH=... ARCH=sm_89
```

构建走 NVBit 自带的 CMake 助手（`add_nvbit_tool`），**不要改成直连 nvcc**：
助手还做了三件必需的事 —— `--keep-device-functions`（否则注入函数被当死代码删掉，
运行时报 `instrumentation function ... not found in binary!`）、`-Xptxas -astoolspatch`、
以及工具 `.so` 的设备链接。实测用直连 nvcc 复刻官方示例工具同样失败，所以这是构建方式问题，
不是源码问题。

## 使用

```bash
CUDA_INJECTION64_PATH=$PWD/ks_memtrace.so LD_PRELOAD=$PWD/ks_memtrace.so \
  KS_MEMTRACE_KERNEL=my_kernel KS_MEMTRACE_OUT=trace.txt ./your_app

kernoscope memtrace --from trace.txt
# 或直接注入报告：
kernoscope --ks-memtrace trace.txt -k regex:my_kernel -o rep -- ./your_app
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `KS_MEMTRACE_OUT` | `ks-memtrace.txt` | 输出路径 |
| `KS_MEMTRACE_KERNEL` | 空（全部） | kernel 名子串过滤。**强烈建议指定**，否则轨迹可能极大 |
| `KS_MEMTRACE_BLOCK_STRIDE` | `1` | 只记录 `blockIdx.x % stride == 0` 的 block（采样） |
| `KS_MEMTRACE_MAX_RECORDS` | `200000` | 设备缓冲上限（每条 272 B，20 万条约 54 MB） |
| `KS_MEMTRACE_VERBOSE` | `0` | 打印插桩细节 |
| `KS_MEMTRACE_DEBUG_CALLS` | `0` | 1 = 只统计"注入调用发生了几次"，不做记录（排查用） |

## 自测（已知答案）

```bash
make selftest NVBIT_PATH=...
```

期望：`coalesced` = **4**、`strided32` = **32**、`unaligned` = **5** 个 sector/request。
三个数不对就说明插桩或地址收集有问题，不要继续用后面的模型量。

## 状态：设备侧地址捕获在本环境**未打通**（有证据，不是"还没试"）

Python 侧（解析 / 分析 / 模型 / CLI / 报告集成）已完成并有单测覆盖；**设备侧**的工具能编译、
能插桩、能写出格式正确的文件头，但记录数为 0。已定位到如下事实（A/B 对照）：

| 实验 | 结果 |
|---|---|
| 官方 `instr_count`（只传 `guard_pred_val` + `const_val32` + `const_val64`） | ✅ 正常统计（`kernel 0 - k_coalesced - 12 instructions`） |
| 我们自己写的 `opcode_hist`（同上参数模式） | ✅ 正常统计 |
| 官方 `mem_trace`（带 `nvbit_add_call_arg_mref_addr64`） | ❌ 0 条记录（同一 app、同一构建树） |
| 我们的 `ks_memtrace`（带 `mref_addr64`） | ❌ 0 条记录；`KS_MEMTRACE_DEBUG_CALLS=1` 下计数器也是 **0** → 注入的调用根本没执行 |
| 官方 `mem_trace` 用 CMake **重新构建** | 仍 0 条 |

⇒ 本环境（NVBit 1.8.1 + 驱动 580 + CUDA 13 + sm_89）下，**带地址参数（MREF）的注入不生效**，
而纯计数器注入正常。这不是本项目代码的逻辑问题（官方工具同样表现）。

**继续排查的建议顺序**：

1. 换 NVBit 版本（1.7.x / 更新的 1.8.x）重试 —— 最可能是版本与驱动的组合问题；
2. 换 CUDA Toolkit 版本（12.x）的工具链重试；
3. `NO_EAGER_LOAD=0` 重跑（NVBit 默认懒加载模块，可能与 MREF 插桩交互）；
4. 到 NVBit 的 issue/论坛搜 `mref_addr64` 或 `mem_trace empty`。

**在此之前可以怎么用**：轨迹格式（`ks-memtrace v1`）是明确契约，
你可以用任何能产出该格式的手段（自己的插桩、能正常工作的 NVBit 版本、或官方 `mem_trace`
若在你的环境可用），Python 侧照常分析。

## 已知限制（都是设计取舍，不是 bug）

1. **按 block 采样**：全量记录会让被测 kernel 明显变慢（每条 warp 指令一次原子写）。
   采样比写在文件头部，分析器按比例放大；采样 block 的代表性由使用者判断。
2. **上限即截断**：达到 `KS_MEMTRACE_MAX_RECORDS` 后停止记录，文件头部标 `truncated 1`。
   分析器此时**拒绝**给出 DRAM 字节 —— 截断值只是下界，不是实测。
3. **缓存模型是理想 LRU**：真实替换策略不是 LRU，所以命中率是乐观上界；报告同时给出
   "只过 L2" 与 "先过 per-SM L1 再过 L2" 两种口径。
4. **共享内存与常量访问不计入**（`LDS/STS/LDC/ULDC` 被过滤）：它们不产生 L2/DRAM 流量。
5. 每个进程只处理**一个上下文**（多上下文场景未覆盖）。
