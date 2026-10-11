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

## NVBit 版本与构建布局（R-72/R-77：版本、布局、构建命令、结论四件事必须配对写）

历史上这里写过"NVBit 1.8.1 是本仓库真机跑通的、1.7.x 也行" —— 第五轮验收报告 V-10 的真机
A/B 把它推翻了：**同一份 app（`nvcc -O3 -arch=sm_89`，两个 kernel：copy + atomicAdd）、
同一个环境变量（`NO_EAGER_LOAD=0`）、只换 `mem_trace.so`**，本环境（CUDA 13 + sm_89 +
驱动 580）**唯一被钉住可用**的组合是 v1.7.7.3 + 老 Makefile：

| NVBit 版本 | 构建方式（实测过的命令） | 产物布局 | 本环境（CUDA 13 + sm_89 + 驱动 580）结论 |
|---|---|---|---|
| **v1.7.7.3** | 老 Makefile：`make -C $NVBIT_PATH/tools/mem_trace ARCH=sm_89` | `tools/mem_trace/mem_trace.so` | ✅ **可用**（真机 A/B：rc=0，6 条 `MEMTRACE` 行 —— `LDG.E.CONSTANT` / `STG.E` / `RED.E.ADD.F32.FTZ.RN.STRONG.GPU`）。本仓库真机钉住的就是这一行 |
| **v1.8.1** | CMake：`cmake -S tools -B build && cmake --build build --target mem_trace`（在 `$NVBIT_PATH` 下；或新版 `make --target mem_trace`） | `tools/build/mem_trace/mem_trace.so` | ❌ **实测失败**（rc=1，**0 条** `MEMTRACE` 行）：`ERROR file ./src/nvbit_imp.cpp line 735: CUDA_ERROR_INVALID_SOURCE device kernel image is invalid`。⚠ **不要** `make` 整个 `tools`：`mem_trace_tma` 用 `%cluster_ctaid` 要求 sm_90+，会把整棵树带崩；架构变量写**裸数字**（`89`，不是 `sm_89`） |

> ⚠ **别把"版本号"当可用性凭据。** 同一台机器上还有一份更早的 v1.8.1 CMake 构建
> （`<workdir>/lab/nvbit_release_x86_64`，2026-10-08 构建）在小负载上出过轨迹 ——
> **同一个版本号、不同的那份构建，结果不同**。所以本文档钉住的是**组合**
> （版本 + 构建方式 + 产物布局），本环境唯一实测可用的组合就是上表第一行；
> 换环境后要改这张表，得先在本机把 A/B 重做一遍（同一份 app、同一 `NO_EAGER_LOAD=0`、只换 `.so`）。

配套的三条事实（都有门禁守着）：

* `collect.sh` **两种布局都找**（`tools/build/mem_trace/mem_trace.so` 与
  `tools/mem_trace/mem_trace.so`）；显式给了 `NVBIT_PATH`/`NVBIT_MEMTRACE_SO` 却找不到产物时
  **报错退出**（`exit 2`）并列出**实际找过的路径**，不再"静默跳过"、也不再让你重复指同一个变量。
* 版本从**实物**里读（别处不许手抄）：`nvbit_version.sh` 先看 `$NVBIT_PATH/core/nvbit.h` 的
  `NVBIT_VERSION`，退化时看 `.so` 里的整行版本字符串。
  `bench/artifacts/*/environment.txt` 现在同时记 `nvbit:`（路径）与 `nvbit_version:`（版本）
  —— 这一整节的结论就建立在版本上，只记路径等于没记。
* `tests/test_nvbit_smoke.py` 把本文件的矩阵**解析成结构化行**，与测试内写死的实测结论逐格比对
  （版本 ↔ 构建命令 ↔ 布局 ↔ 结论），并**按本文件的说明真采一段轨迹**。把 v1.8.1 那行的命令换回
  实测失败的老 Makefile 写法、把两行的结论对调、或删掉任一候选路径，门禁都会红；本机完全没有
  NVBit 时文档/候选路径那两层仍然跑（不 skip）。

两条都必须 `NO_EAGER_LOAD=0` —— **它是所有版本的前提**（1.8 起默认懒加载会让插桩错过时机：
程序跑完但零记录）。看到 0 条记录先查它，再查版本与布局是否配对。

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
# 推荐路径（官方工具 + 转换器）：
NVBIT_MEMTRACE_SO=/path/to/mem_trace.so KS_MEMTRACE_OUT=trace.txt ./collect.sh -- ./your_app
# 备选路径（本目录工具，见下方「状态」）：
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

## 产出轨迹的两条路径

### 推荐：官方 NVBit `mem_trace` + 本项目转换器（已端到端验证）

官方工具的输出正好是「一行 = 一条 warp 访存指令 + 全部 lane 地址 + CTA/warp/opcode」，
粒度刚好够用。一键采集：

```bash
NVBIT_MEMTRACE_SO=/path/to/mem_trace.so KS_MEMTRACE_OUT=t.txt ./collect.sh -- ./your_app
kernoscope memtrace --from t.txt --sm-count 128 --l2-bytes 75497472 --kernel <子串>
```

`collect.sh` 会在 `$NVBIT_PATH` 下**按两种布局**自动找官方工具
（`tools/mem_trace/mem_trace.so`、`tools/build/mem_trace/mem_trace.so`，见上面的版本矩阵；
也可以直接给 `NVBIT_MEMTRACE_SO`）；两处都没有时它会列出**实际找过的路径**并以 `exit 2` 退出。
`kernoscope memtrace` 缺 `--l2-bytes` 时会取本机 L2 容量。
报告路径同样吃官方输出（`--ks-memtrace <file>`，自动判别格式），
缓存模型的访问上限用 `--ks-max-accesses` 调（超限时**拒绝**给 DRAM 字节而不是给偏小的值）。

⚠️ 轨迹体积 ∝ 流量 × launch 次数，而插桩**不区分预热**：一次 80 MB 工作集 + 默认 50 次预热
能写出 10 GB 文本。采轨迹时用小负载，并把预热调小（`bench/ledger_probe.cu --warmup 2`）。

**必须 `NO_EAGER_LOAD=0`**：NVBit 1.8 默认懒加载模块，会让插桩错过时机 ——
表现是"程序正常跑完、一条记录都没有"。这是本项目在本环境踩了很久的坑，
`collect.sh` 已内置该变量。若你看到 0 条记录，先查这个。

已知答案验证（`make selftest` 用的三个内核，1 个 warp）：

| 负载 | 期望 sector/request | 实测 | 合并效率 |
|---|---|---|---|
| coalesced（连续 32×4B） | 4 | **4.000** | 100.0% |
| strided32（lane 间隔 128B） | 32 | **32.000** | 12.5% |
| unaligned（首地址 +4B） | 5 | **5.000** | 80.0% |

官方输出有两处信息缺失，转换器用**显式假设**补上并标注出来：

* 没有 SM id → 用 `cta % sm_count` 近似（头部 `# smid_approx 1`），
  因此「先过 per-SM L1」那个命中率口径只能当参考；
* 没有指令 PC → 用 opcode 建 id，legend 写在文件头部。

官方工具会打印**所有** kernel 的记录：转换器按 `LAUNCH` 行归属，
默认只取第一个 kernel（绝不把多个 kernel 混算），可用 `--kernel <子串>` 指定。

### 备选：本目录的工具（可编译、可插桩，但注入调用在本环境不执行）

| 实验（同一 app、同一构建树） | 结果 |
|---|---|
| 官方 `instr_count`（guard + const32 + const64） | ✅ 正常统计 |
| 本项目的 `opcode_hist` 式工具（同参数模式） | ✅ 正常统计 |
| 官方 `mem_trace`（带 `mref_addr64`）+ `NO_EAGER_LOAD=0` | ✅ 正常记录 |
| 本目录的 `ks_memtrace`（带 `mref_addr64`）+ `NO_EAGER_LOAD=0` | ❌ 插桩调用已插入，但设备计数器为 0 → 调用未执行 |

`NO_EAGER_LOAD` 这一项已经排除（本工具在 eager 下仍为 0），剩下的差异仍在排查。
在此之前请用上面的官方工具路径 —— 它已验证可用，且转换器已覆盖其格式。

## 已知限制（都是设计取舍，不是 bug）

1. **按 block 采样**：全量记录会让被测 kernel 明显变慢（每条 warp 指令一次原子写）。
   采样比写在文件头部，分析器按比例放大；采样 block 的代表性由使用者判断。
2. **上限即截断**：达到 `KS_MEMTRACE_MAX_RECORDS` 后停止记录，文件头部标 `truncated 1`。
   分析器此时**拒绝**给出 DRAM 字节 —— 截断值只是下界，不是实测。
3. **缓存模型是理想 LRU**：真实替换策略不是 LRU，所以命中率是乐观上界；报告同时给出
   "只过 L2" 与 "先过 per-SM L1 再过 L2" 两种口径。
4. **共享内存与常量访问不计入**（`LDS/STS/LDC/ULDC` 被过滤）：它们不产生 L2/DRAM 流量。
5. 每个进程只处理**一个上下文**（多上下文场景未覆盖）。
