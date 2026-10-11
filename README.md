# Kernoscope 内核镜
**在拿不到硬件性能计数器的多租户 GPU 上，为算子优化提供「每个数字都标注出处」的性能证据。**

它回答三类问题：① 哪个 kernel 慢、launch 配置如何（`exact`）；② 离 roofline 多远、是不是 DRAM bound（`estimated`，带误差带）；③ 实际达到的 occupancy、发射槽上界与瓶颈方向（`measured`/`estimated`）。
它**不能回答**：精确 L2/DRAM 字节、命中率、warp stall 周期细分 —— 这些与 ncu 同源的硬件计数器我们不给数字，只报 `n/a` + 原因 + 替代口径；完整三态清单见 [COVERAGE.md](COVERAGE.md)。

30 秒上手（没有 GPU 也能跑通）：`kernoscope doctor` —— 先看这台机器能答什么、拿不到什么、为什么（`doctor --no-probe` 只做静态推断、不做计数器实测探测；`--json` 给机器读）。

与 `ncu` 的关系：CLI 兼容，但数据源不同（它读硬件计数器，我们读物理观测量，绝不占用它的指标名）；与 `nsys` 的关系：复用它的 CUPTI activity，是它的消费者而不是替代品；与 `torch.profiler` 的关系：它答"哪一步慢"，我们追"这个 kernel 为什么慢、这个数字有多可信"。

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

### 它明确不做什么

- **不做重放（replay）、不锁频。** 耗时来自 nsys 时间戳，是真实时间；ncu 默认锁频并重放被测 kernel。
- **不碰 NVIDIA 私有格式。** 不写 `.ncu-rep`，也不做 `--import-source` 的 SASS/源码逐行对照。
- **不追求"覆盖更多 ncu 功能"。** 覆盖的是"拿不到计数器"这一类受限场景。
- **不重放、不插桩去伪造计数器。** 缺哪个数据源就报告缺哪个，并给出替代口径。

---

## 安装

```bash
# uv（推荐）
uv tool install kernoscope

# 或从源码
git clone https://github.com/QingGo/Kernoscope && cd Kernoscope
uv sync && uv run kernoscope --help
```

依赖：Python ≥ 3.10，**运行时零第三方依赖**。运行需要 `nsys`（随 CUDA ≥ 11.1 提供）与
`nvidia-smi`；`cuobjdump`（静态资源/SASS）为可选；`pynvml`（能耗/显存忙碌优先走它）
用 `pip install 'kernoscope[nvml]'` 安装。
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

## 快速开始（30 秒路径）

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

机器可读的版本是 `kernoscope capabilities --json`：它列出**能回答哪些问题、哪些必然被拒、
补救动作的代价多少**，给 agent 规划调用序列用（人读版本就是 `kernoscope capabilities`，
静态总览见 [COVERAGE.md](COVERAGE.md)）。

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
| bf16 matmul | 165.2 TFLOP/s | **163.3**（口径离散 4.0%，可分辨） |
| fp32 matmul | 82.6 | **54.6**（只有 66%！；口径离散 0.7% ⇒ **不可分辨**） |
| DRAM copy / read / write | 1008 GB/s | **910.0 / 907.5 / 940.6** |
| 实测期 SM 时钟 | 3105 MHz | **1815–2625（均值 2356）** |

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

[`bench/ledger_probe.cu`](bench/ledger_probe.cu) 的三个 kernel 每 launch 的字节数可以**解析算出来**，
### 两条口径怎么选（先看这张表再往下读）

DRAM 字节有**两条独立路径**，选哪条取决于"你要多准"与"这台机器能不能装外挂"：

| | NVML 代理（默认，零依赖） | 访存轨迹（NVBit） |
|---|---|---|
| 精度 | **−4.2% ~ +12.8%**（含单边偏高） | **地址是算术**：同一负载 235 MB 上差 320 B（≈ −0.0001%） |
| 依赖 | 只要 NVML（驱动自带） | 要自己装 NVBit + 采集脚本（见 [探针目录](tools/probes/nvbit_memtrace/)），采集成本 ∝ 流量 × launch 数 |
| 什么时候**拒绝**给数 | 窗口主导性 <95%、覆盖率 <80%、**显存忙碌低于 1 个百分点分辨率**（小 kernel / 常驻 L2） | 轨迹被截断、访问数超上限、缺 L2 容量 |
| 什么时候用它 | 大 kernel 的稳态段：只要判断"是不是 DRAM bound"，±10% 的带够用，且不用付安装成本 | 要绝对字节、要分辨 <5% 的改动、或上面那三条拒绝正好命中你的负载 |

**轨迹口径是"绝对 DRAM 字节"的首选**：它的地址部分是 measured 级算术（可逐条复核），
只有"哪些访问命中 L2"是模型 —— 所以工具会同时给出**替换策略上下界**，让读者看到模型的影响范围。
代理口径仍然是**默认**：它不需要任何外挂，且在满足前提的大 kernel 上够用；
一旦前提不成立它会明确拒绝（而不是给个偏小的数）。

用它当账本去对拍（RTX 4090，8 秒稳态循环，`--ks-window auto`；产物在
[`bench/artifacts/`](bench/artifacts/)，逐项对照见「数字从哪来」）：

| 负载 | 账本（解析值 / launch） | 估算 / launch | 误差 | 覆盖率 | 窗口显存忙碌 | 整进程显存忙碌 |
|---|---|---|---|---|---|---|
| 拷贝 `y=x`（2N×4B） | 536.9 MB | 588.2 MB | **+9.6%** | 95.2% | 95.2% | 70.2% |
| 只读归约（N×4B） | 268.4 MB | 257.2 MB | **−4.2%** | 95.2% | 95.2% | 64.5% |
| 只写（N×4B） | 268.4 MB | 271.2 MB | **+1.0%** | 95.7% | 76.5% | 60.9% |

上表顺便演示了两件本轮才修好的事：**「窗口显存忙碌」与「整进程显存忙碌」是两个数**
（参与估算的永远是前者，它有自己的名字 `ks__dram_window_mem_busy_mean_pct`），
以及**两个归因前提都被打印出来**（覆盖率 95.2%、窗口主导性 100%）。

**这个误差带的口径与边界**（不给出口径就等于过度声称）：

- 它是在**字节数解析已知的合成微基准**上实测的范围，取上面三个负载的最小/最大值
  （本次 **−4.2% ~ +12.8%**）。**它不是上界**：真实负载的访存模式与这三个微基准不同，
  误差可以更大。同一批微基准在更早的三次运行里分别是 **−4.1% ~ +12.5%**、
  **−4.3% ~ +12.7%** 与 **−4.3% ~ +9.0%**（每一次的带都能在 `bench/artifacts/`
  的登记表里逐项复核），四次运行的并集是 **−4.3% ~ +12.8%**。
- 它是**单边偏高**的：DRAM 代理容易高估（见下）。所以结论要写"落在代理误差带内"，
  不要把"两次方向一致"当成独立证据。
- 不要写成对称的"±N%"，也不要把它当"精度上界"。

**精度 vs 稳态段长度**（同一个已知答案负载，四个时长各测一遍；
产物 `proxy-duration-sweep.csv`）：

| 稳态段目标 | 覆盖率 | 窗口主导性 | 估算 / launch | 误差 |
|---|---|---|---|---|
| 1.2 s | 71.4% | 100% | —（**拒绝**） | 前提①不成立 → `n/a` |
| 3 s | 88.2% | 100% | 581.5 MB | +8.3% |
| 8 s | 95.2% | 100% | 586.6 MB | +9.3% |
| 20 s | 98.1% | 100% | 591.6 MB | +10.2% |

这张表要说的是**两件常被搞混的事**：

1. **时长只决定"前提是否成立"，不决定"准不准"。** 1.2 秒时覆盖率 71.4% < 80%，
   工具直接拒绝给数（正确地）；而 3 秒到 20 秒，覆盖率从 88.2% 涨到 98.1%，
   误差却在 +8.3% ~ +10.2% 之间**不收敛**（20 秒那次比 8 秒更大，这是反方向的不收敛）。
   也就是说"多跑一会儿就更准"是错的，正确说法是"多跑一会儿才**有资格**给数"。
2. 同一个时长在不同运行里也会变（20 秒那一点在有产物的三次运行里是 +6.2%、+6.6%，
   本次 **+10.2%**）：**运行间的波动与时长的影响同量级**。所以误差带要按范围引用，
   并且要带上"几次运行"这个口径。

高估的系统性来源是 `utilization.memory` 在高负载下**先于** DRAM 达到可达峰值而饱和
（读数 95% 时实际吞吐约为可达峰值的 87%）。

所以：

- 拿它做 "<10%" 这类验收线**只能算勉强**：登记过的运行里 `write` 出现过 +12.5%，越线。
- 想要第二条独立口径就用**访存轨迹**（下节），两条路径的比值会作为
  `ks__dram_trace_vs_nvml_pct` 一起报出来。
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
# 记录头部缺信息时手动指定：--arch sm_89 --threads 256 --sm-count 128；--kernel 覆盖名字
# （默认全部从记录头部取；--json 给机器读）
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
# 发射槽分母默认按每 SM 4 个调度器（可改：--schedulers <N>）；--json 给机器读
# 或直接并进报告（SchedulerStats / WarpStateStats 两个 section）
kernoscope --set full --ks-instr-hist hist.txt --ks-sm-count 128 --ks-clock-mhz 2450 -- ./app
```

判词是**方向性**的：发射槽很高 ⇒ 偏发射受限；很低而显存很忙 ⇒ 偏显存受限；
两者都低 ⇒ 偏延迟受限（在等数据，不在等发射）。
`smsp__warp_issue_stalled_*` 那类 **stall 周期细分仍然报 `n/a` + 原因** ——
需要硬件计数器，我们不会用方向性判断冒充它。

真机两个负载的对照（RTX 4090，产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 负载 | 动态 warp 指令（每 launch） | 发射槽利用率（上界） | 显存忙碌 | 判词 |
|---|---|---|---|---|
| 256 MB 拷贝（DRAM bound） | 18.4 M | **2.8 %** | 71 % | 偏显存受限 —— 指令在等数据 |
| 2 MB 逐元素 norm（常驻 L2、循环开销占大头） | 0.91 M | **83.3 %** | 低 | 偏发射受限 —— 该减少指令数 |

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
# 缓存模型可调：--line-bytes（默认 32 = sector 口径）、--l1-bytes、--l1-ways、--l2-ways
# --max-accesses（默认 4M；超过就拒绝给 DRAM 字节，避免偏小的假值）
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
| 默认 JSON（含逐 launch 数组） | **130.6 MB** |
| `--ks-json-no-launches -c 50` | **25.4 KB** |

所以：`-c <N>` 限流每个 kernel 只报 N 次、`--ks-json-no-launches` 让 JSON 不带逐 launch 数组、
逐 launch 明细改用 `--launch-csv` 单独出表；报告写盘后 stderr 会打印实际体积与限流建议。

---

## 局限与已知失败

这一节是**我们主动交代的失败面**。它不是免责声明，而是"你会遇到什么、我们会怎么说"的清单。

### ① 计数器拿不到（根因，不是 bug）

宿主 `RmProfilingAdminOnly=1` 且容器无 `CAP_SYS_ADMIN`/`CAP_PERFMON` 时，CUPTI 直接报
`ERR_NVGPUCTRPERM`。计数器是 host-wide 的、会泄露同机租户行为，**容器内没有任何合法绕过**。
所以 `dram__bytes.sum`、`lts__t_sector_hit_rate.pct`、`smsp__warp_issue_stalled_*` 这些名字下
永远只有 `n/a` + 原因；`--strict` 请求它们时 exit 2。想要精确计数器，只能在有权限的机器上跑 ncu。

### ② NVML 代理口径是**单边偏高**的

DRAM 字节代理 = 显存忙碌时间 × 本机实测峰值带宽。`utilization.memory` 在高负载下
**先于** DRAM 达到可达峰值而饱和（读数 95% 时实际吞吐约为可达峰值的 87%），
所以误差不是对称的：本轮三个已知答案负载的实测是 **−4.2% / +12.8% / +9.6%**
（逐次运行与并集见上文「这个误差带的口径与边界」—— 那里是唯一一处列出各次带的地方，
免得同一句话在文档里出现两遍、然后各自漂）。
**它不是上界，也不是对称误差**：真实负载的访存模式与微基准不同，误差可以更大；
引用时必须写"落在代理误差带内"，而不是"两次方向一致所以可信"。

### ③ 分辨率是 **1 个百分点**

`utilization.memory` 是整数百分比，采样周期 200 ms。读数 0.0% 时，
"真的没搬数据"与"0.3% 被截断成 0"**不可区分** —— 此时工具不给字节数，并写明这一点。
别的量也有各自的分辨率（例如归因上偏估计的分辨率是 0.1 个百分点），
报告里 `resolution` 列与数字同时出现。

### ④ 跨运行噪声与 MDD

同一负载、同一时长，不同运行的误差会变（20 秒那一点在有产物的三次运行里是 +6.2%、
+6.6% 与本次 +10.2%）：**运行间波动与时长的影响同量级**。所以：

- `kernoscope selftest` 只依赖 torch，量出**本机**误差带与 **MDD**（最小可分辨差异）；
- `kernoscope diff` 用该 MDD 做三态比较：真改进 / 真退化 / **分不出来（indistinguishable）/ 不可比**；
- 没有误差带与 MDD 时，`diff` 拒绝给方向判词（`no_selftest_band`），而不是报"变了"；
- 两次运行的协议（标定文件、窗口模式、工具版本、kernel 选择）不同时，判 `protocol_mismatch`。

### ⑤ 哪些问题**必然被拒**

| 被拒的问题 | 拒绝码 | 为什么必然 | 替代 |
|---|---|---|---|
| warp stall 周期细分 | `counters_denied` | 与 ncu 同源，需要计数器 | 发射槽上界 + 方向性判断（`issue_slot`） |
| 精确 DRAM 字节 / L2 命中率 | `counters_denied` | 同上 | DRAM 代理（带误差带）或访存轨迹（模型） |
| 各 pipe（FP32/INT/Tensor）动态利用率 | `counters_denied` | 同上 | 发射槽上界 + roofline 达成百分比 |
| PC 采样 / SourceCounters | `sampling_unavailable` | 与计数器同源、同样被锁死 | 指令直方图的发射槽上界 |
| NVLink 流量与利用率 | `no_nvlink` | 单卡/无 NVLink 实例上本来就不适用 | 不适用（忽略这一行） |
| SASS/源码逐行对照 | `no_sass_source` | 本工具不实现 | `cuobjdump -sass` 自行对照 |

（完整清单与每台机器的实际三态见 `kernoscope capabilities --json`，或用
`kernoscope capabilities` 看一屏；本仓库的静态版本见 [COVERAGE.md](COVERAGE.md)。）

### ⑥ 「我们拒绝给数」的清单

按拒绝码（闭集，`kernoscope capabilities --json` 里可编程读取）：

- **物理上拿不到** → 只报 `n/a`：`counters_denied`、`sampling_unavailable`、
  `counters_not_installed`、`counters_unknown`、`no_nvlink`、`no_sass_source`。
- **口径前提不成立** → 那一行照样出现，但值是 `n/a`：
  `coverage_too_low`（窗口没被负载主导）、`share_too_low`（被选中的 kernel 不是主角）、
  `multi_kernel_window`（窗口里有多个 kernel，不可归因）、
  `below_resolution` / `mem_busy_zero`（读数落到分辨率下限以下）、
  `no_calibration`（缺本机实测峰值，不用规格表顶替）、
  `calibration_caliber_mismatch`（标定口径与本次测量不可比）、
  `unknown_arch`（未知架构算不出 occupancy 上界，不猜）、
  `trace_too_many_accesses`（访问数超上限时不给偏小的值）。
- **比较不可比** → 不给方向判词：`protocol_mismatch`、`both_refused`、
  `no_selftest_band`、`mdd_unknown`（没有 MDD 就说"分不出来"）。
- **请求的东西不存在** → `metric_absent`（不等于被拒）、`legacy_report`
  （旧报告没有结构化拒绝信息，标注 legacy 而不猜原因）。
- **兜底** → `internal_unknown`：出现即视为本工具缺陷，请按 issue 模板报告。

### ⑦ 其它已知的"做不到"

- **不重放、不锁频**：所以绝对耗时与 ncu 的锁频读数不可直接对比。
- **per-launch 归因依赖窗口构成**：混合强度的窗口只认轨迹口径（见上）。
- **轨迹体积与流量成正比**：几十 MB 工作集可能产出上 GB 文本。
- **fp32 roofline 拐点可能不可分辨**：标定的口径离散度大时，工具拒绝用它算拐点。

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
- ✅ **v1.8**：结构化拒绝（闭集 code + observed/threshold + 可执行 remedy + 代价）、
  `capabilities --json` 与三态覆盖矩阵、`selftest`（本机误差带 + MDD）、
  `diff`（真改进/真退化/分不出来/不可比）。
- 🚧 v1 剩余：region/阶段视图（对接 IKP Trace）、访存轨迹的输入侧复用分析（locality/复用距离）、
  `cuobjdump -sass` 静态指令混合细化到 per-kernel。
- 🚧 v2：因果链与 `explain`（借 ingero 思路，用 CUPTI activity + host 采样，不依赖 eBPF）、
  真 `.ncu-rep` 回读并与我们的报告对照。

## 文档与声明

- [COVERAGE.md](COVERAGE.md)：三态覆盖矩阵 + 四层能力分层 + 必然被拒的问题（由代码生成）。
- [CONTRIBUTING.md](CONTRIBUTING.md)：适配器清单与扩展点（指标映射 / 后端 / 能力问题 / 拒绝类型 / 产物）。
- [PROVENANCE.md](PROVENANCE.md)：五个出处等级、四条强制规则、命名纪律、拒绝给数的三种形态。
- [CALIBRATION.md](CALIBRATION.md)：本机峰值标定的字段与口径纪律（roofline 依赖它）。
- [NOTICE](NOTICE)：第三方与商标声明。许可见 [LICENSE](LICENSE)（Apache-2.0）。

Kernoscope is **not affiliated with, endorsed by, or sponsored by NVIDIA Corporation.**
"NVIDIA", "CUDA", "Nsight" and "ncu" are used descriptively only.
