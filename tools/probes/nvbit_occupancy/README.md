# NVBit 达到-occupancy 工具（被测 kernel 零源码改动）

用 NVBit 把 `(smid, 起止时间)` 两个时间戳**在 SASS 层注入**被测 kernel，产出
`ks-occupancy-records v1` 记录，交给既有的 `kernoscope occupancy` 分析
（区间重叠法重建「实际达到的 occupancy」）。

它补的是 `tools/probes/occupancy_probe.cu` 的那个硬伤：老探针要求用户**把几行代码抄进
自己的 kernel**（其 README §6.1 就是这么写的），于是

* **闭源 kernel 永远测不了**：cuBLAS / cuDNN / Inductor 生成的 / 库里的 kernel 没有源码可改；
* 用户得改**自己可能并不拥有**的代码，改完的 kernel 与线上跑的已经不是同一个二进制。

本工具的接口是「把被测程序跑一遍」，被测 kernel 一个字符都不改。

## 它测什么

| 量 | 口径 | 与 ncu 的关系 |
|---|---|---|
| 平均并发 blocks/SM | 每个 CTA 的 `(%smid, g0, g1)` 区间重叠的时间加权平均 | 与 ncu Achieved Occupancy **语义同源** |
| 达到 occupancy | 平均并发 blocks × warps/block ÷ 每 SM 的 warps | 同上（出处 `measured`，不宜跨工具直接比） |
| 峰值并发 blocks | 同一份区间里的最大重叠数 | —— |
| 理论上界 | **驱动**给的 `cuOccupancyMaxActiveBlocksPerMultiprocessor`（透传进文件头部） | 与 ncu Occupancy 面板同源 |
| 覆盖率 | `cta_rows / grid_blocks` | 本工具**独有**：判断这份数据能不能用 |

时间戳用 `%globaltimer`（ns，**跨 SM 同源**）。**不用 `%clock64`**：它是每 SM 各自原点的
计数器，老探针在真机上实测过跨 SM 极差 2e12 cycles，跨 SM 比较直接是错的
（见 `tools/probes/occupancy_probe.cu` 开头的三条坑）。

## 输出格式：与老探针逐字节同构

`kernoscope occupancy --from <file>` 的解析器
（`src/kernoscope/analyze/occupancy.py` 的 `load_probe_records()`）**一行都不用改**：
本工具写的就是同一套格式 —— 首行 `# ks-occupancy-records v1`、`# key value` 头部、
然后 `smid,g0,g1` 数据行。

```text
# ks-occupancy-records v1
# kernel my_kernel
# arch sm_89
# threads_per_block 128
# sm_count 128
# grid_blocks 12288
# theoretical_occupancy_pct 100.0
# cta_rows 12202
# coverage 0.9930
# coverage_ok 1
# min_coverage 0.9500
# launches_traced 1
# config_mismatch 0
# instrumented_instrs 2
# allow_partial 0
# source nvbit-sass-injection
# note g0 = CTA 入口 %globaltimer（ns，跨 SM 同源）；g1 = 最后一个走到「最后一条静态指令」的 warp 的时间
# note 出口钩子是近似：NVBit 没有「CTA 结束」钩子，early exit 的 CTA 会少报
smid,g0,g1
0,123456789,123512345
```

比老探针多出来的头部键（`cta_rows` / `coverage` / `coverage_ok` / `min_coverage` /
`launches_traced` / `config_mismatch` / `allow_partial` / `source`）是**附加信息**：
`load_probe_records()` 把 `# key value` 一律收进 dict，未知键不影响解析。

## 两个钩子，以及它们各自近似在哪

NVBit 没有「CTA 开始 / CTA 结束」这种钩子，只能在指令级插桩：

| 钩子 | 插在哪 | 语义 | 近似 |
|---|---|---|---|
| **入口** `ks_occ_enter` | kernel 里**偏置最小**的那条指令之前（第一条静态指令） | 每 CTA **只记一次**：第一个到达的线程用 `atomicExch` 抢门，抢到的人读 `%globaltimer` + `%smid` 写进本 CTA 的槽 | 记的是「第一个到达的线程」的时间。CTA 的各 warp 是陆续被调度的，所以 g0 可能**略晚于**该 CTA 真正被分配的时刻 |
| **出口** `ks_occ_exit` | kernel 里**偏置最大**的那条指令之前（最后一条静态指令） | 每个 warp 的 lane 0 各做一次 `atomicMax.u64`：g1 = **最后一个**走到这里的 warp 的时间 | ⚠️ **这是全工具最大的近似**（见下） |

**出口钩子的近似，必须说清楚**：

1. 它记的是**静态最后一条指令**。`if (…) return;` 这类 **early exit** 会让一部分 CTA 根本
   走不到那里 —— 这些 CTA 一行都记不到，**只会少报，不会报错**。所以覆盖率纪律是硬要求
   （下一节）。
2. 它记的是「最后一个 warp 走到收尾处」，不是「CTA 的资源被释放」。收尾指令到 CTA 真正
   退场之间还有一点点时间（warp 退休、资源回收），这部分**没有被算进区间**。
3. 反过来，取 `atomicMax`（而不是「第一个收尾的 warp」）是**刻意**的：CTA 一直占着 SM 的
   资源槽，直到它**最后一个** warp 结束为止；取 min 会系统性低估 block 的存活区间。

如果将来 NVBit 提供了文档化的「CTA 完成」钩子，这里应当换成它 —— 目前（本仓库钉住的
那版 NVBit）**没有**。找过的替代方案只有两个，都不能用：

* `nvbit_at_cuda_event()` 里的 `cuLaunchKernel` **退出**事件：那是**主机侧**的 API 返回
  事件（异步 launch，返回时 kernel 还没跑），而且是 kernel 级、不是 CTA 级；
* 挂钩 `EXIT` 指令：每个 warp 各有一条，插在它上面**同样**要求我们选「取哪个 warp 的时间」，
  语义与「最后一条静态指令」完全一样，而且 multi-block 的 kernel 里 `EXIT` 常常就是
  偏置最大的那条 —— 没有额外信息。

**为什么时间戳要「值依赖」**：老探针在真机上踩过 —— 只加 `memory` clobber 不够，纯寄存器
计算仍会被编译器挪到时间戳之后（实测差 200×），所以那边用 `GT_AFTER` 把读挂在一个无法
预先算出的值上。本工具同样把 `%globaltimer` 的读挂在一个由「设备内存里的 salt + CTA 编号」
拼出来的值上（`ks_occ_gt()`），并且整个读活在一个 `__noinline__` 的函数调用里 ——
注入的调用本身跨不过去。

## 覆盖率纪律（部分数据不许变成分布）

工具按 CTA 平铺编号**直接索引**记录槽（不用追加缓冲），所以「记录到几个 CTA」是精确的：

```
覆盖率 = 有 g0、有 g1、且 g1 >= g0 的 CTA 行数 ÷ 本次 launch 的 grid_blocks
阈值   = KS_OCC_MIN_COVERAGE（默认 0.95，即 95%）
```

覆盖率低于阈值、或同一个 kernel 被 launch 了多次、或后续 launch 的 grid/block 与第一个
不同时，工具会：

* 在 **stderr** 上把「为什么」逐条写出来（覆盖率、行数/网格、launch 次数、配置不一致）；
* 打一行机器可读的状态：`KS-OCCUPANCY: 状态 拒绝`（通过时是 `状态 可用`）；
* **默认不写记录文件**（`KS_OCC_ALLOW_PARTIAL=1` 才会强行写出，且文件里带
  `# coverage_ok 0`、状态仍然是拒绝）。

**为什么默认不写**：部分数据算出来的 occupancy 只会**偏小**，不会有任何「数据不全」的
信号 —— 一旦写成文件，`kernoscope occupancy` 就会照常给一个数。所以拒绝必须发生在
**产出文件之前**。

**分析侧应当拒绝**：`src/kernoscope/analyze/occupancy.py` 目前**看不见** `coverage_ok`
这个键（它只按 `# key value` 收进 dict，不校验覆盖率）—— 本工具不修改那条既有分析路径，
所以「覆盖率不足就直接算数」这件事，只能靠**不产出文件**来挡。请不要手工删掉/绕过
`coverage_ok` 标记，也不要把部分文件喂给 `kernoscope occupancy --from`；真要一个数，
先把覆盖率做上去（换更小的负载 / 让被测 kernel 只 launch 一次 / 提高 `KS_OCC_MAX_CTAS`）。

**为什么多 launch 也要拒绝**：区间重叠法算的是「同一段时间里的平均并发度」。把两次
launch 的记录放进同一份文件，`t_last - t_first` 会横跨两次 launch 的间隔，平均值被稀释
（达到 occupancy 被系统性拉低）。所以要测哪个 kernel，就让它在本次进程里**只 launch 一次**
（或把过滤缩到那一个 kernel）。

## 构建

需要 **NVBit**（NVIDIA 专有许可）。本项目**不分发** NVBit 本体，也不分发用它构建的二进制；
本目录只有我们自己的源码。

```bash
make NVBIT_PATH=$HOME/nvbit_release_x86_64
# 架构：ARCH=native（默认）/ all-major / sm_89 …
make NVBIT_PATH=$HOME/nvbit_release_x86_64 ARCH=sm_89
make NVBIT_PATH=$HOME/nvbit_release_x86_64 resolve   # 不需要 GPU：只报"会用哪份 .so、哪个版本"
```

构建走 NVBit 自带的 CMake 助手（`add_nvbit_tool`），**不要改成直连 nvcc**：助手还做了
三件必需的事（`--keep-device-functions`、`-Xptxas -astoolspatch`、工具 `.so` 的设备链接
+ `-lcuda`）—— 理由与实测见 `../nvbit_memtrace/README.md` 的「构建」一节。

## 使用

```bash
# 采集（被测程序零改动；KS_OCC_KERNEL 用子串选中要测的那个 kernel）
NVBIT_OCCUPANCY_SO=$PWD/tools/probes/nvbit_occupancy/ks_occupancy.so \
KS_OCC_KERNEL=my_kernel KS_OCC_OUT=ks-occupancy.txt \
  ./collect.sh -- ./your_app <你的参数>

# 分析（既有分析路径，一行都不用改）
kernoscope occupancy --from ks-occupancy.txt --write-json occ.json

# 把结果注入报告
kernoscope --ks-achieved-occupancy occ.json -k regex:my_kernel -o rep -- ./your_app
```

`collect.sh` 的纪律（与 `../nvbit_memtrace/collect.sh` 同一套）：

* 按两种布局找产物：`$NVBIT_PATH/tools/ks_occupancy/build/ks_occupancy.so`、
  `$NVBIT_PATH/tools/ks_occupancy/ks_occupancy.so`（外加脚本旁边那份，`make` 会把 `.so`
  拷到那里）；也可以直接 `NVBIT_OCCUPANCY_SO=…` 指过去。
* 显式指了 `NVBIT_PATH`/`NVBIT_OCCUPANCY_SO` 却找不到产物 ⇒ **exit 2**，并列出**实际找过的
  路径**（不是让人再指一遍同一个变量）。
* `--resolve-only` 只回答「这次会用哪份 `.so`、它是哪个版本」，机器可读两行。
* 版本**从实物读**（复用 `../nvbit_memtrace/nvbit_version.sh`），读不出写 `n/a`，**绝不编一个**。
* 采集时固定 `NO_EAGER_LOAD=0`（NVBit 默认懒加载会让插桩错过时机，表现是「程序跑完、
  一条记录都没有」）、`NOBANNER=1`、`CUDA_INJECTION64_PATH` + `LD_PRELOAD`。
* 退出码：`0` 可用；`2` 配置/依赖问题（没有 `.so`、没有 `cuobjdump`）；`3` 覆盖率纪律没过
  （文件没写出来，或只在 `KS_OCC_ALLOW_PARTIAL=1` 下写了但状态是拒绝）。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `KS_OCC_OUT` | `ks-occupancy.txt` | 记录文件路径 |
| `KS_OCC_KERNEL` | 空（第一个 launch） | kernel 名子串过滤；命中后**锁定这个名字**，别的 kernel 不插桩 |
| `KS_OCC_MAX_CTAS` | `262144` | 设备缓冲能容纳的 CTA 数（约 8 MB）；网格更大时超出的 CTA 记不到 ⇒ 覆盖率下降 ⇒ 拒绝 |
| `KS_OCC_MIN_COVERAGE` | `0.95` | 覆盖率阈值（0..1） |
| `KS_OCC_ALLOW_PARTIAL` | `0` | `1` = 纪律没过也写出（带 `# coverage_ok 0`，状态仍是拒绝） |
| `KS_OCC_VERBOSE` | `0` | 打印插桩细节（首/末指令偏置、缓冲大小） |
| `KS_OCC_DEBUG_CALLS` | `0` | `1` = 只统计「注入调用执行了几次」，不写记录（排查用） |
| `KS_OCC_LOG` | `${KS_OCC_OUT}.run.log` | `collect.sh` 把被测程序的输出写到哪 |

## 未在真机验证

**本工具没有在任何 GPU 上跑过。** 写它的时候手上没有能编译 / 加载 NVBit 的机器，具体是：

* `occupancy.cu` / `inject_funcs.cu` **没有被 nvcc 编译过**：能不能过编译、NVBit 的
  API 用法（`nvbit_insert_call` + `nvbit_add_call_arg_const_val64` 的参数表、
  `Instr::getOffset()` 的语义、`cuLaunchKernelEx_params` 的字段名）都**未经验证**；
* `.so` **没有被 NVBit 加载过**，因此「注入的调用到底有没有执行」**未知**
  （本仓库的 `nvbit_memtrace` 就真遇到过「插桩调用已插入、设备侧计数器为 0」，见
  `../nvbit_memtrace/README.md` 的「状态」一节 —— 别假设插了就一定执行）；
* CTA 首末指令的选取（按 `getOffset()` 取 min/max）、`atomicMax.u64` 的配对、
  覆盖率数字、以及出口钩子对 early-exit kernel 的具体低估幅度，全部**未经测量**；
* 目前**没有** `make selftest` 之类的自测：本目录没有"已知答案"的负载可直接跑。

**依赖是构建级的**：NVBit 的可用性不能只看版本号 —— 同一个版本号的不同构建结果相反。
本仓库真机唯一钉住可用的是**官方 `mem_trace`** 的 v1.7.7.3 + 老 Makefile 组合
（见 `../nvbit_memtrace/README.md` 的版本矩阵）；那张表**不覆盖本工具**。

所以第一次用的时候，按这个顺序做，不要跳过：

1. **小流量试采**：`KS_OCC_KERNEL=<一个最小的 kernel>`，别拿大负载开刀；
   `KS_OCC_DEBUG_CALLS=1` 先确认注入调用执行次数 **> 0**（`= 0` 就说明这份 NVBit 构建
   在本机不可用，先按 `../nvbit_memtrace/README.md` 的版本矩阵换组合）；
2. **与老探针对拍**：在一个**两种方法都能插桩**的 kernel 上（即你有源码的那个），用
   `tools/probes/occupancy_probe.cu` 与 `tools/probes/nvbit_occupancy/collect.sh` 各采一份，
   比较 `kernoscope occupancy` 给出的 `occupancy_pct` / `avg_concurrent_blocks` ——
   两者应当在几个百分点以内；差很多就说明钩子点或时间戳读法有问题，**先查再说**；
3. 覆盖率不够就**不要**用那份文件（工具默认也不会写出来）。

在这三件事做完之前，不要把本工具给出的数字写进任何结论。

## 已知限制

1. **出口是近似**：见「两个钩子」一节。early-exit 的 kernel 用本工具会缺行；这种 kernel
   目前只能用源码探针（`occupancy_probe.cu`）。
2. **只测一个 kernel**：过滤命中后锁定第一个 kernel 名，其它 kernel 不插桩；同进程里
   另一个匹配的 kernel 会在 stderr 上点名（不会静默混算）。
3. **只支持单次 launch**：多次 launch 会被拒绝（见「覆盖率纪律」）。CUDA Graph / 多流
   场景下同一个 kernel 往往 launch 多次，需要把负载裁到一次。
4. **CTA 槽是复用**：记录按 `blockIdx` 平铺编号索引，上限 `KS_OCC_MAX_CTAS`；超过上限的
   CTA 记不到（覆盖率会掉，工具据此拒绝）。
5. **每 CTA 一条记录**，不是每 warp 一条：本工具测的是 block 级存活区间，不是 warp 级调度
   （要看 warp 级要另做工具）。
6. **多上下文的进程未覆盖**：与 `nvbit_memtrace` 一样，每个进程按一个上下文处理。
