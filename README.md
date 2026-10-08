# Kernoscope 内核镜

**在拿不到硬件性能计数器的机器上，也能看清 kernel；每个数字都标注来源。**

> Provenance-honest CUDA kernel profiler for locked-down / multi-tenant GPUs.
> CLI-compatible with `ncu` (NVIDIA Nsight Compute), so existing scripts keep working.

Kernoscope is **not affiliated with, endorsed by, or sponsored by NVIDIA Corporation.**
"NVIDIA", "CUDA", "Nsight" and "ncu" are used descriptively only.

---

## 为什么需要它

很多共享租用的多租户 GPU 容器实例宿主机带着
`RmProfilingAdminOnly=1`，容器又没有 `CAP_SYS_ADMIN`/`CAP_PERFMON`，于是：

```
$ ncu --set basic -- python train.py
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
          NVIDIA GPU Performance Counters on the target device 0.
```

这不是配置问题，也不能靠用户侧技巧绕过：计数器是 host-wide 且会泄露同机租户行为，
所以平台有理由锁死。**但"读不到计数器"不等于"拿不到信息"** —— 换物理观测量即可：

| 你想要的结论 | Kernoscope 怎么给 | 出处 |
|---|---|---|
| 哪个 kernel 慢、launch 配置如何 | CUPTI activity（nsys）→ LaunchStats / 时长 | `exact` |
| 同名 kernel 的**多组** launch 配置各自的时长 | 按 (grid, block, smem) 分组，逐组给出读数 | `exact` |
| occupancy 上界与被谁限制 | 静态属性 + 架构上限，解析算术 | `exact` |
| **实际达到的 occupancy** | `(smid, globaltimer)` 区间重叠重建（需探针） | `measured` |
| **DRAM 字节数 / 是不是 DRAM bound** | NVML 显存忙碌时间 × 本机实测峰值带宽 | `estimated`（需窗口覆盖率达标） |
| **DRAM 字节数的第二条口径** | NVBit 访存轨迹 + 缓存模型 | `estimated` |
| 离 roofline 多远 | 本机实测峰值（不是规格表）+ 你给的工作量 | `estimated` |
| 发射槽用了多少、瓶颈方向 | NVBit 动态指令数 ÷ 硬件发射槽上限 | `estimated`（上界口径） |
| 每 kernel 能耗 | NVML 能量累加器（ncu 没有这个指标） | `measured` |
| L2/DRAM 精确字节、命中率、warp stall 细分 | **不给数字**，报 `n/a` + 原因 | `unavailable` |

最后一行是重点：**我们拒绝在 ncu 的计数器指标名下面填一个模型算出来的数。**
这由**类型系统**强制（规则 R1–R4 见 [PROVENANCE.md](PROVENANCE.md)），并有测试守住。

---

## 安装

```bash
# uv（推荐）
uv tool install kernoscope

# 或从源码
git clone https://github.com/kernoscope/kernoscope && cd kernoscope
uv sync && uv run kernoscope --help
```

依赖：Python ≥ 3.10。运行需要 `nsys`（随 CUDA ≥ 11.1 提供）与 `nvidia-smi`；
`cuobjdump`（静态资源/SASS）与 `pynvml`（能耗）为可选。
**工具的查找不只看 PATH**：`$CUDA_HOME/bin`、`/usr/local/cuda*/bin`、`/usr/bin` 都会找 ——
在"CUDA 装了但没加 PATH"的实例上，只看 PATH 会误报"未安装"。

> 网络提示：`uv` **不继承 `pip.conf`**，会直连 PyPI。若你的网络访问 PyPI 很慢，
> 给 uv 单独配镜像，否则 `uv sync` 可能只是"看起来卡住"：
> ```bash
> mkdir -p ~/.config/uv && cat > ~/.config/uv/uv.toml <<'TOML'
> [[index]]
> url = "https://pypi.tuna.tsinghua.edu.cn/simple"
> default = true
> TOML
> ```
> 实测（某国内 GPU 实例）：PyPI 直连 0.1 MB/s、某镜像 0.1 MB/s、TUNA 7.6 MB/s ——
> 配好之后 `uv sync` 从"分钟级卡住"变成 4 秒。

---

## 快速开始

```bash
# 第一步：看看这台机器上「能拿到什么、拿不到什么、为什么」
kernoscope doctor

# 与 ncu 用法一致：-- 之后是要 profile 的命令
kernoscope --set basic -k regex:gemm -c 3 -o rep -- python train.py

# 回读报告 / 导出 CSV（含 provenance 与 resolution 列）
kernoscope -i rep.ksreport.json --csv
kernoscope -i rep.ksreport.json --csv --metrics est__dram_vs_ledger_err_pct

# 每个 launch 一行（形状扫描、自己算分位数都方便）
kernoscope --launch-csv -k regex:my_kernel -- ./bench

# CI 卡门：请求了拿不到的数据就失败（exit 2），而不是悄悄给错数
kernoscope --strict --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -- python train.py

# 没有 GPU 也能跑通全流程（合成数据，明确标注 SYNTHETIC）
kernoscope --ks-synthetic --set detailed
```

输出四件套：

| 输出 | 开关 | 用途 |
|---|---|---|
| ncu 风格文本 | 默认 | 人眼/日志；每条指标带 `[exact]`/`[estimated]`/`[unavailable]` 标签与分辨率 |
| CSV | `--csv` | 表格分析；多出 `provenance` / `resolution` 列 |
| JSON | `-o rep` → `rep.ksreport.json` | 机器消费、回归对比（带 `schema_version`） |
| Perfetto trace | `-o rep` → `rep.perfetto.json` | 丢进 <https://ui.perfetto.dev> 看时间线，零安装 |
| 逐 launch CSV | `--launch-csv`（配 `-o` 时写 `rep.launches.csv`） | 每次 launch 的 grid/block/时长/配置键 |

`--metrics <名>` 在宽松模式下**真的筛选输出**（匹配到的 `n/a` 行照常打印、带原因）；
`--print-units base|auto`、`--print-details all|header`、`--print-metric-name name|full`
都是真实现（默认 `auto`/`all`/`name`）。

### 与 ncu 的迁移

```bash
alias ncu=kernoscope          # 或 ln -s $(which kernoscope) ~/bin/ncu
```
`argv[0]` 为 `ncu` 时会自动进入兼容模式（并打印一行说明：数据来源与 ncu 不同）。
选项分四类处理：

| 类别 | 例子 | 行为 |
|---|---|---|
| 真实现 | `-k` / `-c` / `-s` / `--devices` / `-o` / `-i` / `--csv` / `--launch-csv` / `--print-*` | 按 ncu 语义生效 |
| 映射实现 | `--set` / `--section` / `--metrics` | 映射到我们的数据源；产不出的只在 `unavailable` 里出现 |
| 空转（会告警，`--strict` 下报错） | `--replay-mode` / `--cache-control` / `--clock-control` / `--pm-sampling-*` / `--mps` 等 | 我们的数据源没有 replay / 计数器 pass，所以无效 |
| 自有扩展 | `--ks-*` | 数据源、标定、账本、窗口、synthetic |

`--strict` 覆盖面：`--metrics` 的指标、`--section`/`--set` 的 section、空转开关、
以及**请求了账本对拍但裁决行算不出来**。

---

## 本机峰值标定（强烈建议）

roofline 用**本机实测峰值**而非规格表。一台 RTX 4090（450 W 功耗墙）的实测差异：

| 指标 | 规格表 | 本机实测 |
|---|---|---|
| bf16 matmul | 165.2 TFLOP/s | **169.0**（口径离散 3.9%，可分辨） |
| fp32 matmul | 82.6 | **57.3**（只有 69%！） |
| DRAM copy / read / write | 1008 GB/s | **909.2 / 905.8 / 940.0** |
| 实测期 SM 时钟 | 3105 MHz | **1980–2775（均值 2397）** |

（数字来自 [`bench/artifacts/`](bench/artifacts/) 里那次完整标定，见「数字从哪来」。）

```bash
python tools/machine-profile.py --quick      # 写出 ./machine-profile.json
kernoscope --ks-calibration machine-profile.json --ks-gemm 4096,4096,4096 --ks-dtype bf16 \
           -k regex:gemm -- python bench.py
```

标定产物会记录**累加口径**（`acc_caliber` / `precision_settings` / `method`）与
**口径离散度**（`caliber_spread_pct` / `caliber_distinguishable`）—— 因为
"口径能不能被分辨"本身是测量结果：同一台机器相隔一天的两次标定给出 0.4% 与 5.0% 两种结论。
缺这些字段时，Kernoscope **拒绝用它算 roofline 拐点**（带宽侧照常给数）。
详见 [CALIBRATION.md](CALIBRATION.md)。

---

## 重建硬件计数器读数（ncu 迁移对照）

ncu 的硬件计数器在受限环境里拿不到，但很多读数可以用**其它物理观测量重建**。
每一行都标明出处等级，报告里带同样的标签。

**两点语义差异先讲清楚**：

1. ncu 默认**锁频**（所以脚本里常见 `--clock-control none`），并且会 **replay** 被测 kernel；
   我们的耗时来自 nsys 时间戳 —— **真实时间、不 replay、不锁频**，不同 kernel 的绝对耗时直接可比。
2. 我们**不占用 ncu 的指标名**（规则 R3）。重建量一律带自有前缀
   （`est__`/`meas__`/`ks__`），并在 `source` 字段里注明它对应哪个 ncu 指标 ——
   避免"看起来像官方数字、其实是模型值"。

| 想要的读数 | ncu 写法 | Kernoscope 写法 | 出处 |
|---|---|---|---|
| **DRAM 字节数** | `--metrics dram__bytes.sum` | `--ks-calibration … [--ks-ledger-mb <MB/launch>]` → `est__dram_bytes_total` / `est__dram_bytes_per_launch` | `estimated` |
| **DRAM 吞吐占比** | `--metrics dram__throughput.avg.pct_of_peak_sustained_elapsed` | `est__dram_throughput_pct_of_peak`（分母带来源与标定时间） | `estimated` |
| **与账本对拍** | —— | `--ks-ledger-mb`（per-launch）/ `--ks-ledger-total-mb`（总量）/ `--ks-ledger-range-mb LOW,HIGH` → `est__dram_vs_ledger_err_pct`、`est__dram_range_position_pct`、`ks__ledger_range_verdict` | `estimated` / `derived` |
| Achieved Occupancy | `--section Occupancy` | 探针 + `kernoscope occupancy --write-json` → `--ks-achieved-occupancy` → `meas__achieved_occupancy_pct` | `measured` |
| 每 kernel 耗时排序、kernel 间空隙 | `nsys stats --report cuda_gpu_kern_sum` | 内置（同一份 CUPTI activity 数据） | `exact` |
| LaunchStats（grid/block/寄存器/共享内存） | `--section LaunchStats` | 内置；寄存器还会与 cubin 交叉校验并暴露口径差异 | `exact` |
| 算力管道占比 | `--set speedoflight` 的 `SM [%]` | `est__pct_of_measured_compute_peak`（**仅 FMA 口径**，与 ncu 取多管道最大值不同） | `estimated` |
| L1/L2 命中率、sectors/request、coalescing | `l1tex__t_sector_hit_rate.pct` | NVBit 访存轨迹：`meas__sectors_per_request`（直接计数）+ **模型**命中率（明确标注上界口径） | `measured` / `estimated` |
| stall 细分、发射槽利用率 | `SchedulerStats` / `WarpStateStats` | `--ks-instr-hist`（NVBit 动态指令计数）→ `est__issue_slot_utilization_pct`、`est__stall_direction` | `estimated`（上界 + 方向） |
| 精确 L2/DRAM 字节 | `lts__*` / `dram__*` | 只报 `n/a` + 原因 | `unavailable` |
| 每 kernel 能耗 | （ncu 没有） | NVML 能量累加器 → `meas__device_energy_j` | `measured` |
| CI 门禁 | —— | `--strict` → exit 2（覆盖 set/section/账本裁决行/空转开关） | —— |
| 时间线可视化 | `.ncu-rep` + UI | Perfetto JSON，零安装 | `exact` |

---

## DRAM 代理与账本对拍

```
bytes ≈ Σ_i ( 显存忙碌_i / 100 × Δt_i ) × 本机实测峰值带宽
```

**口径纪律**（不一起报出来就是过度声称）：

- 报告里必然出现 `ks__sampler_window_coverage_pct`（窗口是否被被测负载主导，`derived`）
  与 `ks__dram_window_mem_busy_mean_pct`（**参与估算的**窗口显存忙碌均值，`derived`，
  分辨率 **1 个百分点**、采样周期 **200 ms**）。整进程的 NVML 均值另见
  `meas__nvml_mem_busy_mean_pct` —— 它是**另一个数**（含启动/空闲），两者不可互换。
- **per-launch 归因有两个前提，不是一个**：
  - 前提① `ks__sampler_window_coverage_pct ≥ 80%`：GPU 在不在干活；< 50% 连总量也不给；
  - 前提② `ks__matched_kernel_kernel_share_pct ≥ 95%`：**被 `-k` 选中的那个 kernel 是不是主角**
    （它占窗口内全部 kernel GPU 时间的比例）。空闲时段贡献 0 字节，所以分母用内核时间、
    不是墙钟；`ks__matched_kernel_window_share_pct` 是与覆盖率同量纲的对照量。
  - 只有前提①时，一个窗口里**顺序跑了多个 kernel/变体**的负载会被判成"可归因"，
    把整窗口字节除到某个 kernel 的 launch 数上，误差可达 **+147%**（真机实测）。
    `ks__attribution_bias_estimate_pct = 100×(1/份额−1)` 把这个上偏**算出来**并跟数字并列。
    95% 这个门槛是从验收线反推的：份额 80% 仍自带 +25% 的偏差，守不住 `|误差|<10%`。
  - 门槛是**代理**（假设各 kernel 字节强度可比）而不是严格上界，实测值可能更高。
    混合强度的窗口（计算密集 + 访存密集混跑）只认**轨迹口径**：访存轨迹只测被追踪的
    那个 kernel，天然不受窗口里还有谁影响。`ks__window_busy_segments` 会告诉你窗口里有
    几段连续忙碌期 —— 多段通常意味着"先跑正确性再跑性能"或"依次跑 v1/v2/v3"这类拼接。
- 显存忙碌读数低于分辨率下限（例如 0.0%）→ 不给字节数，并写明
  "无法区分『真的没搬』与『0.x% 被截断成 0』"。
- 一次采样都没有时，前提行也是 `n/a`（报"0%"等于声称量过）。
- 拒绝时那一行**照样出现**（`n/a` + 具体失败前提），不会从报告里消失；
  `--strict` 会因此 exit 2。
- 默认 `--ks-window auto`：按首个/末个忙碌采样点裁掉 `import torch`/JIT 的启动开销
  （实测这类启动开销 9–10 秒，会把覆盖率稀释到 15%）。`--ks-window full` 可关闭，
  `--ks-warmup <秒>` / `--ks-window-s <秒>` 可显式限定。
- `-c/--launch-count` 只限制**展示**：per-launch 的分母始终是该 kernel 在本次运行里的真实
  launch 数（分母被截断会带来同样的倍数级误差）。
- 账本对拍**同基比较**：`--ks-ledger-mb` 是"每次 launch"，`--ks-ledger-total-mb` 是整窗口；
  区间 `--ks-ledger-range-mb` **永远按 per-launch**（拿整窗口总量去比一个 per-launch 区间是荒谬的）。
- `--ks-ledger-basis auto|nvml|trace`：NVML 代理口径不成立时（覆盖率不足、主导性不足、
  或显存忙碌读数为 0）**自动改用访存轨迹口径**的 per-launch 字节完成对拍；
  显式指定 `nvml`/`trace` 时不做回退。

```bash
# 引用 est__dram_* 之前先看三个数：覆盖率、窗口主导性、分辨率下限
kernoscope --set basic --ks-calibration machine-profile.json \
           --ks-ledger-mb 234 -k regex:my_kernel -o rep -- ./my_bench
kernoscope -i rep.ksreport.json --csv | grep -E 'ledger|window_coverage|share|mem_busy'
```

### 代理精度实测（字节数已知的微基准）

[`bench/ledger_probe.cu`](bench/ledger_probe.cu) 的三个 kernel 每 launch 的字节数可以解析算出来，
用它当账本去对拍（RTX 4090，8 秒稳态循环，`--ks-window auto`；产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 负载 | 账本（解析值 / launch） | 估算 / launch | 误差 | 覆盖率 | 显存忙碌读数 |
|---|---|---|---|---|---|
| 拷贝 `y=x`（2N×4B） | 536.9 MB | 587.4 MB | **+9.4%** | 95.2% | 69.0% |
| 只读归约（N×4B） | 268.4 MB | 256.9 MB | **−4.3%** | 95.2% | 65.6% |
| 只写（N×4B） | 268.4 MB | 302.5 MB | **+12.7%** | 95.2% | 67.8% |

诚实结论：**这台机器上的代理误差带是 −4% ~ +13%，不是 ±4%** ——
同一批微基准在**两次运行**里给出过 copy −3.2%…+9.4%、read −4.3%…+0.5%、
write −0.1%…+12.7%（第二次的产物已入库；第一次是更早一次运行，数字记在这里、产物未留存）。
这个量级来自两件事：

1. `utilization.memory` 在高负载下会**先于** DRAM 达到可达峰值而饱和
   （读数 95% 时实际吞吐约为可达峰值的 87%），于是流量被系统性地高估；
2. 每次运行的窗口、时钟与调度状态都会动，误差本身也在动。

所以：

- 拿它做 "<10%" 这类验收线**只能算勉强**：本次三个负载里就有一个 +12.7% 越线。
  结论要写成"落在代理误差带内"，不要把两次方向一致当成独立证据；
- 想要第二条独立口径就用**访存轨迹**（下节），两条路径的比值会作为
  `ks__dram_trace_vs_nvml_pct` 一起报出来；
- 显存忙碌读数落在分辨率下限以下时，工具**不给数字**，并说明原因。

### 覆盖率不足时报告长什么样（真实输出）

```
est__dram_bytes_total      n/a   [unavailable]  窗口覆盖率 25% < 50%：窗口没被被测负载主导，
                                              忙碌时段是否饱和无从确认；把稳态循环延长到约 6 秒…
est__dram_bytes_per_launch n/a   [unavailable]  同上（per-launch 还需要覆盖率 ≥80% 且锁定单一 kernel）
ks__sampler_window_coverage_pct  25 %  [derived]
est__dram_vs_ledger_err_pct n/a  [unavailable]  误差行无法给出：…；可以加 --ks-memtrace <trace>
                                              换用**轨迹口径**的 per-launch 字节
```
`--strict` 在这份报告上返回 **2**（请求了账本裁决却产不出来）。

### 代理口径不成立时：自动改用轨迹口径（真实输出）

同一个负载在小工作集上跑（1 MB，常驻 L2），NVML 的显存忙碌读数落到分辨率下限以下：

```
est__dram_bytes_total             n/a   [unavailable]  显存忙碌读数 0.0% 低于 NVML 的分辨率下限 1 个百分点…
meas__nvml_mem_busy_mean_pct       0 %  [measured]     [resolution 1 %（整数百分比）]
est__dram_bytes_per_launch_trace 0.699 MB [estimated]  第二条口径：NVML 口径不成立时用它做账本对拍
est__dram_vs_ledger_err_pct     -66.7 % [estimated]  账本 2.1 MB；此次用**轨迹口径**，因为 NVML 代理口径不成立
```

−66.7% 不是算错，而是**换口径换出的正确结论**：轨迹模型给出 L2 命中 66.7%
（1 MB 工作集反复读写），于是只有 1/3 的账本流量真正落到 HBM。
这正是第二条口径的价值 —— 它回答的不是"账本差多少"，而是"这个问法在当前工作集下成不成立"。

---

## 达到的 occupancy（无需硬件计数器）

ncu 的 Achieved Occupancy 来自计数器；这里用**区间重叠法**重建 —— 让每个 block 记录
`(smid, 起止 %globaltimer)`，时间平均的并发 block 数 × 每 block warps ÷ 每 SM 上限：

```bash
make -C tools/probes run                      # 编译并采集 occupancy-records.txt
kernoscope occupancy --from occupancy-records.txt                      # 看结果
kernoscope occupancy --from occupancy-records.txt --write-json occ.json  # 产出注入用 JSON
kernoscope --ks-achieved-occupancy occ.json -k regex:<kernel> -- your_app
```

真机两点校验（RTX 4090，128 线程/块，产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 场景 | 理论（CUDA API） | 实测（区间重叠） |
|---|---|---|
| 无 smem | 12 blocks/SM = 100% | **11.27 → 93.9%**（差的是 ramp-up/tail） |
| 32 KB 静态 smem | 3 blocks/SM = 25% | **2.99 → 24.9%** |

要测**你自己的 kernel**，把 `tools/probes/occupancy_probe.cu` 里那几行搬进你的 kernel 即可
（文件头注释列了三个必须保留的坑：时间戳要值依赖、跨 SM 只能用 `%globaltimer`、
共享内存必须写后读否则被 ptxas 当死代码删掉）。

---

## 发射槽利用率与瓶颈方向（无计数器，但有上界）

官方 NVBit 的 `opcode_hist` 在默认模式下统计的是**每 warp 每条指令执行一次** ——
这正好是"消耗掉多少个发射槽"。于是：

```
发射槽利用率（上界口径） = 动态 warp 指令数 / (SM 数 × 4 调度器 × 时钟 × 时长)
```

```bash
# 采集（NVBit 是可选外部依赖，需自行安装；NO_EAGER_LOAD=0 是关键）
NO_EAGER_LOAD=0 CUDA_INJECTION64_PATH=$NVBIT/tools/build/opcode_hist/opcode_hist.so \
  LD_PRELOAD=$NVBIT/tools/build/opcode_hist/opcode_hist.so NOBANNER=1 \
  ./your_app > hist.txt 2>&1

kernoscope issue --from hist.txt --kernel my_kernel \
                 --sm-count 128 --clock-mhz 2450 --duration-ms 86.7 --dram-pct 3.2
# 或直接并进报告（SchedulerStats / WarpStateStats 两个 section）
kernoscope --set full --ks-instr-hist hist.txt --ks-sm-count 128 --ks-clock-mhz 2450 -- ./app
```

判词是**方向性**的：发射槽很高 ⇒ 偏发射受限；很低而显存很忙 ⇒ 偏显存受限；
两者都低 ⇒ 偏延迟受限（在等数据，不在等发射）。
`smsp__warp_issue_stalled_*` 那类 **stall 周期细分仍然报 `n/a` + 原因** ——
需要硬件计数器，我们不会用方向性判断冒充它。

真机两个负载的对照（RTX 4090，产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 负载 | 动态 warp 指令（每 launch） | 发射槽利用率（上界） | 显存忙碌 | 判词 |
|---|---|---|---|---|---|
| 256 MB 拷贝（DRAM bound） | 18.4 M | **2.0 %** | 71 % | 偏显存受限 —— 指令在等数据 |
| 2 MB 逐元素 norm（常驻 L2、循环开销占大头） | 0.91 M | **84.8 %** | 低 | 偏发射受限 —— 该减少指令数 |

⚠️ 发射槽利用率的分母是"这段时间里 SM 最多能提供的发射槽"，所以它是**上界口径**：
真实可用发射槽只会更少。分母里的时长必须取自**同等工作量**的运行（报告路径用的是
`gpu__time_duration.sum`；`kernoscope issue` 需要你自己把 CUPTI 时长传进来）。
直方图那次的 launch 数与报告不同时，会按 per-launch 平均对齐，并在 note 里写明。

---

## 访存轨迹（NVBit 官方 mem_trace）

```bash
make -C tools/probes/nvbit_memtrace NVBIT_PATH=~/nvbit_release_x86_64
NVBIT_PATH=~/nvbit_release_x86_64 bash tools/probes/nvbit_memtrace/collect.sh -- ./your_app
kernoscope memtrace --from memtrace-official.txt --l2-bytes 75497472
kernoscope --set basic --ks-memtrace memtrace-official.txt -k regex:<kernel> -o rep -- ./your_app
```

给出 `sectors/request`（直接计数）、合并效率、工作集、L1/L2 命中率（**模型上界**，不是计数器）、
以及轨迹口径的 per-launch DRAM 字节（可参与账本对拍）。

⚠️ 轨迹体积与**流量**成正比（一条记录 = 一条 warp 访存指令 ≤128 B），
几十 MB 工作集就能产出上 GB 的文本；先想清楚要看哪个 kernel，或只做小规模验证。

> **NVBit 许可是 `LicenseRef-NvidiaProprietary`（禁止再分发）**：Kernoscope **不打包、不 vendor**
> 任何 NVBit 二进制或由其构建的工具。请自行安装并遵守其许可。详见 [`NOTICE`](NOTICE)。

---

## 成本可预算

小 kernel × 长循环会产生十万级 launch，逐 launch 明细会把报告撑爆。真实数字
（同一次运行，产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 输出方式 | 报告体积 |
|---|---|
| 默认 JSON（含逐 launch 数组） | **134 MB** |
| `--ks-json-no-launches -c 50` | **15.5 KB** |

所以：`-c <N>` 限流每个 kernel 只报 N 次、`--ks-json-no-launches` 让 JSON 不带逐 launch 数组、
逐 launch 明细改用 `--launch-csv` 单独出表；报告写盘后 stderr 会打印实际体积与限流建议。

---

## 数字从哪来

README 与文档里每个"实测"数字都能追到一次可复现的运行：

```bash
bash bench/run.sh              # 全量（约 10 分钟 GPU 时间）
bash bench/run.sh --quick      # 快速版
```

`bench/run.sh` 会做：环境快照（**修订号 + 来源 + 是否脏树 + 工具版本**）→ `doctor` →
本机标定 → 编译已知答案微基准 → DRAM 代理精度 → **代理精度 vs 稳态段长度** →
账本对拍 → **覆盖率不足时的裁决行** → 形状扫描 → occupancy 探针 →
访存轨迹口径对拍 → 发射槽上界；每步的退出码写进产物目录的 `manifest.json`，
产物是机器可读文件（不是往 README 里手抄数字），最后一步自动脱敏。

`tests/test_doc_artifact_consistency.py` 把「文档数字 ↔ 产物」变成门禁：文档里公开的
每个实测数字都要能在最新产物里找到同一写法，产物必须钉住修订号与工具版本，
文档引用的每个产物目录都必须存在。数字一变、文档没跟着变，测试直接红。

| 本文引用的数字 | 产物 |
|---|---|
| bf16 / fp32 TFLOP/s、copy 带宽、口径离散度与逐变体设置 | `bench/artifacts/*/machine-profile.json` |
| 代理误差与覆盖率（三个已知答案微基准） | `bench/artifacts/*/rep-{copy,read,write}.txt` 与 `.csv` |
| **精度 vs 稳态段长度**（1.2/3/8/20 秒四个点） | `bench/artifacts/*/proxy-duration-sweep.csv` |
| 覆盖率不足 ⇒ 裁决行 `n/a` 且 `--strict` = 2 | `bench/artifacts/*/rep-lowduty.txt`、`manifest.json` |
| 形状扫描三组配置的逐组时长 | `bench/artifacts/*/rep-shape.txt`、`shape-sweep.launches.csv` |
| occupancy 探针两组读数 | `bench/artifacts/*/occupancy.json`、`occupancy-smem32k.json` |
| 报告体积对比 | `bench/artifacts/*/size-comparison.csv` |
| 发射槽利用率与两个方向的判词 | `bench/artifacts/*/rep-issue-*.txt` |
| 轨迹口径对拍与 L2 命中率 | `bench/artifacts/*/rep-tracebasis.txt`、`memtrace.json` |
| doctor 的 ncu 四态结论 | `bench/artifacts/*/doctor.json`、`doctor-path-stripped.json` |
| 每份产物是哪个修订产生的 | `bench/artifacts/*/environment.txt`、`manifest.json` |

`bench/issue_probe.sh` 另采发射槽上界的产物（需要 NVBit）。

---

## 拿不到数字时先看这里

| 你看到的现象 | 真实原因 | 怎么做 |
|---|---|---|
| `n/a` + "requires GPU hardware performance counters" | 宿主 `RmProfilingAdminOnly=1` 且容器无 `CAP_SYS_ADMIN`/`CAP_PERFMON` | 换物理观测量：本机标定 + DRAM 代理 / 访存轨迹；不要去找"绕过权限"的办法 |
| `doctor` 说"未安装 ncu"，但你 `ls` 到了它 | 工具只查了 PATH（旧行为） | 现在会给出四态结论与路径；`doctor --json` 的 `tools` 里是"找过哪些目录" |
| `est__dram_bytes_*` 全是 `n/a` | 窗口覆盖率 < 50%（启动开销把窗口稀释了），或显存忙碌读数低于分辨率下限 | 看 `n/a` 行给的具体前提；加 `--ks-window auto`（默认已开）、把稳态循环按**目标秒数**加长 |
| 账本误差行是 `n/a` 而不是数字 | per-launch 口径要求"单一 kernel + 覆盖率 ≥80%" | `-k` 锁定一个 kernel；或改用 `--ks-ledger-total-mb`；或加 `--ks-memtrace` 走轨迹口径 |
| 误差百分比大得离谱（百万级） | 口径混用（per-launch 账本 vs 整窗口估算） | 同基比较：`--ks-ledger-mb` 对 per-launch，`--ks-ledger-total-mb` 对总量，区间永远 per-launch |
| roofline 的 compute 侧是 `n/a` | 标定文件缺 `acc_caliber`/`precision_settings`/`method` | 用新版 `tools/machine-profile.py` 重新标定（带宽侧不受影响） |
| 报告 JSON 上百 MB | 逐 launch 明细随 iteration 数线性增长 | `-c <N>` 限流、`--ks-json-no-launches`、逐 launch 明细改走 `--launch-csv` |
| 轨迹文件几个 GB | 记录数 ∝ 流量 × launch 次数（插桩不区分预热） | 用小工作集 + 小的预热次数（见 [`bench/README.md`](bench/README.md)） |
| 形状扫描只剩一行均值 | 同名 kernel 的多组 launch 配置被聚合 | 现在会逐组给出 `LaunchConfigs`；`launch__grid_size` 在混合时改报 `n/a` + 取值集合 |
| `--strict` 一直红 | 请求了产不出来的数据（含"账本裁决行算不出来"） | 这正是它的用途；`--strict` 的报错会逐条说明是哪个请求、为什么 |

## 状态与路线

- ✅ **MVP**：ncu 兼容 CLI、LaunchStats/时长（exact）、理论与达到 occupancy、实测峰值 roofline、
  设备级计数器（能耗/显存忙碌）、四路输出、`--strict`、provenance 强制。
- ✅ **v1.5**：`kernoscope doctor` —— 一条命令给出「能用/不可用清单 + 原因 + 已选后端 + 可交付出处」，
  并用 ncu 实测确认计数器门（1 秒）；工具发现多路径，ncu 结论分四态。
- ✅ **v1.6**：DRAM 代理 + 账本对拍（per-launch / 总量 / 区间三种口径，同基比较，
  口径不成立必然出现 `n/a` + 原因），`--ks-window auto` 稳态窗口，轨迹口径回退。
- ✅ **v1.7**：形状扫描（多组 launch 配置逐组读数）、`--launch-csv`、`--metrics` 真实筛选、
  标定口径与离散度、报告体积可预算、发射槽上界与瓶颈方向（`kernoscope issue`）。
- 🚧 v1 剩余：region/阶段视图（对接 IKP Trace）、访存轨迹的输入侧复用分析（locality/复用距离）、
  `cuobjdump -sass` 静态指令混合细化到 per-kernel。
- 🚧 v2：因果链与 `explain`（借 ingero 思路，用 CUPTI activity + host 采样，不依赖 eBPF）、
  真 `.ncu-rep` 回读并与我们的报告对照。

## 文档

- [PROVENANCE.md](PROVENANCE.md)：五个出处等级、四条强制规则、命名纪律、拒绝给数的三种形态。
- [CALIBRATION.md](CALIBRATION.md)：本机峰值标定的字段与口径纪律（roofline 依赖它）。
- [NOTICE](NOTICE)：第三方与商标声明。许可见 [LICENSE](LICENSE)（Apache-2.0）。
