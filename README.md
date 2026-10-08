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
| occupancy 上界与被谁限制 | 静态属性 + 架构上限，解析算术 | `exact` |
| **实际达到的 occupancy** | `(smid, globaltimer)` 区间重叠重建 | `measured` |
| **DRAM 字节数 / 是不是 DRAM bound** | NVML 显存忙碌时间 × 本机实测峰值带宽 | `estimated`（稳态；实测误差见「代理精度实测」） |
| 离 roofline 多远 | 本机实测峰值（不是规格表）+ 你给的工作量 | `derived` |
| 每 kernel 能耗 | NVML 能量累加器（ncu 没有这个指标） | `measured` |
| L2/DRAM 精确字节与命中率、warp stall 细分 | **不给数字**，报 `n/a` + 原因 | `unavailable` |

最后一行是重点：**我们拒绝在 ncu 的计数器指标名下面填一个模型算出来的数**。
这由**类型系统**强制（provenance 四规则见下），并有测试守住。

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
`cuobjdump` 与 `pynvml` 为可选（分别用于静态资源与能耗）。

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

# 回读报告 / 导出 CSV（含 provenance 列）
kernoscope -i rep.ksreport.json --csv

# CI 卡门：请求了拿不到的数据就失败（exit 2），而不是悄悄给错数
kernoscope --strict --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -- python train.py

# 没有 GPU 也能跑通全流程（合成数据，明确标注 SYNTHETIC）
kernoscope --ks-synthetic --set detailed
```

输出四件套：

| 输出 | 开关 | 用途 |
|---|---|---|
| ncu 风格文本 | 默认 | 人眼/日志；每条指标带 `[exact]`/`[estimated]`/`[unavailable]` 标签 |
| CSV | `--csv` | 表格分析；多一列 `provenance` |
| JSON | `-o rep` → `rep.ksreport.json` | 机器消费、回归对比（带 `schema_version`） |
| Perfetto trace | `-o rep` → `rep.perfetto.json` | 丢进 <https://ui.perfetto.dev> 看时间线，零安装 |

### 与 ncu 的迁移

```bash
alias ncu=kernoscope          # 或 ln -s $(which kernoscope) ~/bin/ncu
```
`argv[0]` 为 `ncu` 时会自动进入兼容模式（并打印一行说明：数据来源与 ncu 不同）。
选项分四类处理：
真实现 / 映射实现 / **接受但空转（`--strict` 下报错）** / 不适用。

---

## 本机峰值标定（强烈建议）

roofline 用**本机实测峰值**而非规格表。以下是一台 RTX 4090（450W 功耗墙）上的典型差异：

| 指标 | 规格表 | 本机实测 |
|---|---|---|
| bf16 matmul | 165.2 TFLOP/s | **171.7** |
| fp32 matmul | 82.6 | **54.0**（只有 65%！） |
| DRAM copy / read / write | 1008 GB/s | **909 / 900 / 934** |
| 满载 SM 时钟 | 3105 MHz | **2459–2495**（`nvidia-smi` 空闲时报 210） |

拿规格表算 fp32 会得出"只发挥了 65%"的错误结论，实际那台机器的 fp32 上限就是 54。

```bash
kernoscope --ks-calibration machine-profile.json --ks-gemm 4096,4096,4096 --ks-dtype bf16 \
           -k regex:gemm -- python bench.py
```
（`machine-profile.json` 由 `tools/machine-profile.py` 生成。）

---

## 重建硬件计数器读数（ncu 迁移对照）

ncu 的硬件计数器在受限环境里拿不到，但很多读数可以用**其它物理观测量重建**。
下表是迁移对照；每一行都标明出处等级，报告里也带同样的标签。

**两点语义差异先讲清楚**：

1. ncu 默认**锁频**（所以脚本里常见 `--clock-control none`），并且会 **replay** 被测 kernel；
   我们的耗时来自 nsys 时间戳 —— **真实时间、不 replay、不锁频**，不同 kernel 的绝对耗时直接可比。
2. 我们**不占用 ncu 的指标名**（规则 R3：非 exact 的指标必须使用自有前缀）。重建量一律带自有前缀
   （`est__`/`meas__`/`ks__`），并在 `source` 字段里注明它对应哪个 ncu 指标 ——
   避免"看起来像官方数字、其实是模型值"。

| 想要的读数 | ncu 写法 | Kernoscope 写法 | 出处 |
|---|---|---|---|
| **DRAM 字节数** | `--metrics dram__bytes.sum` | `--ks-calibration machine-profile.json [--ks-ledger-mb <账本 MB>]` → `est__dram_bytes_total` / `est__dram_bytes_per_launch` | `estimated` |
| **DRAM 吞吐占比** | `--metrics dram__throughput.avg.pct_of_peak_sustained_elapsed` | `est__dram_throughput_pct_of_peak` | `estimated` |
| **与账本区间对拍** | —— | `--ks-ledger-range-mb LOW,HIGH` → `est__dram_range_position_pct` + 落在哪一侧的判词 | `estimated` |
| Achieved Occupancy | `--section Occupancy` | 探针 + `kernoscope occupancy --write-json` → `--ks-achieved-occupancy` → `meas__achieved_occupancy_pct` | `measured` |
| 每 kernel 耗时排序、kernel 间空隙 | `nsys stats --report cuda_gpu_kern_sum` | 内置（同一份 CUPTI activity 数据） | `exact` |
| LaunchStats（grid/block/寄存器/共享内存） | `--section LaunchStats` | 内置；寄存器还会与 cubin 交叉校验并暴露口径差异 | `exact` |
| 算力管道占比 | `--set speedoflight` 的 `SM [%]` | `est__pct_of_measured_compute_peak`（**仅 FMA 口径**，与 ncu 取多管道最大值不同） | `estimated` |
| L1/L2 命中率、sectors/request、coalescing | `l1tex__t_sector_hit_rate.pct` | 🚧 v1：NVBit 访存轨迹 + 缓存模型（带窗口偏差警告） | —— |
| stall 细分、发射槽利用率 | `SchedulerStats` / `WarpStateStats` | 🚧 v1：NVBit 指令计数 → 发射槽上界；cuda-gdb PC 采样 → 方向性判断 | —— |

### 用法示例：把账本预测与实测放在一起

```bash
# 1) 先用你的解析模型/账本算出预测字节数（不需要 GPU）
#    例：fused elementwise kernel = 读 X + 读 W + 写 Y

# 2) 让 kernel 在紧循环里跑满窗口，再采集
kernoscope --set basic --ks-calibration machine-profile.json \
           --ks-ledger-mb 234 -k regex:my_kernel -o rep -- ./my_bench

# 3) 读回判据
kernoscope -i rep.ksreport.json --csv | grep -E 'dram|duty'
```

对应字段：`est__dram_vs_ledger_err_pct`（与账本的相对误差）、
`est__dram_throughput_pct_of_peak`（有效带宽占比）、
`meas__sampler_duty_pct`（这份估算到底能不能用）。

⚠️ **代理的适用边界**（不一起报出来就是过度声称）：`estimated` 的前提是
**被测 kernel 主导采样窗口、且忙碌时段显存接近饱和**。所以报告里必然附带
`meas__sampler_duty_pct`：<80% 时**拒绝**给出 per-launch 归因，<50% 时连总量也不给 ——
宁可不给数字，也不给一个被稀释的值。

### 代理精度实测

用**字节数已知**的三个微基准标定（每迭代的解析值作为真值），RTX 4090：

| 负载 | 每迭代真值 | 实测 per-launch | 误差 | DRAM 吞吐 | 占空比 |
|---|---|---|---|---|---|
| 拷贝 `y=x`（2N 字节） | 1073.7 MB | 1038.9 MB | **−3.2%** | 84.8% | 85% |
| 只读归约（N 字节） | 536.9 MB | 517.6 MB | **−3.6%** | 84.1% | 84% |
| L2 驻留拷贝 | 16.8 MB（标称） | **拒绝给出** | —— | —— | 15% |

结论：窗口被负载主导、显存接近饱和时，**重复测量落在 −3.6% ~ +0.5%，即代理噪声约 ±4%**
（同一负载的多次运行会随占空比在几个百分点内浮动）。而 L2 驻留负载即使"标称字节数"很大，
也会因占空比不足而拒绝给数 —— 这正是想要的行为：不要用一个被稀释的均值去回答"过了多少 HBM"。

账本对拍是**同基比较**：`--ks-ledger-mb` 是"每次 launch"口径，与 `est__dram_bytes_per_launch` 比；
要跟整窗口总量比用 `--ks-ledger-total-mb`。口径不成立（占空比 <80% 或未锁定单一 kernel）时
**不给误差数字**，改为 `n/a` + 原因 —— 否则会算出"看着像账本错了、其实是口径错了"的荒谬百分比。

## 达到的 occupancy（无需硬件计数器）

ncu 的 Achieved Occupancy 来自计数器；这里用**区间重叠法**重建 —— 让每个 block 记录
`(smid, 起止 %globaltimer)`，时间平均的并发 block 数 × 每 block warps ÷ 每 SM 上限：

```bash
make -C tools/probes run                      # 编译并采集 occupancy-records.txt
kernoscope occupancy --from occupancy-records.txt                      # 看结果
kernoscope occupancy --from occupancy-records.txt --write-json occ.json  # 产出注入用 JSON
kernoscope --ks-achieved-occupancy occ.json -k regex:<kernel> -- your_app
```

真机两点校验（RTX 4090，128 线程/块）：

| 场景 | 理论（CUDA API） | 实测（区间重叠） |
|---|---|---|
| 无 smem | 12 blocks/SM = 100% | **11.27 → 93.9%**（差的是 ramp-up/tail） |
| 32 KB 静态 smem | 3 blocks/SM = 25% | **2.99 → 24.9%** |

要测**你自己的 kernel**，把 `tools/probes/occupancy_probe.cu` 里那几行搬进你的 kernel 即可
（文件头注释列了三个必须保留的坑：时间戳要值依赖、跨 SM 只能用 `%globaltimer`、
共享内存必须写后读否则被 ptxas 当死代码删掉）。

## 可选：SASS 级插桩（NVBit / IKP）

需要指令级归因、访存轨迹、sectors/req、locality 时，装上 NVBit 后把探针结果喂回来：

```bash
kernoscope --ks-cubin my_kernel.so \
           --ks-achieved-occupancy occ_probe.json \
           -- python bench.py
```

> **NVBit 许可是 `LicenseRef-NvidiaProprietary`（禁止再分发）**：Kernoscope **不打包、不 vendor**
> 任何 NVBit 二进制或由其构建的工具。请自行安装并遵守其许可。详见 [`NOTICE`](NOTICE)。

---

## 状态与路线

- ✅ **MVP**：ncu 兼容 CLI、LaunchStats/时长（exact）、理论与达到 occupancy、实测峰值 roofline、
  设备级计数器（能耗/显存忙碌）、四路输出、`--strict`、provenance 强制。
- ✅ **v1.5 已完成**：`kernoscope doctor` 环境自检 —— 一条命令给出「能用/不可用清单 + 原因 +
  已选后端 + 可交付的 provenance 分级」，并用 ncu 实测确认计数器门（1 秒）。
- ✅ **DRAM 流量重建**：`est__dram_bytes_*`（NVML 显存忙碌 × 本机实测峰值，实测误差 ±4%，
  占空比不足时拒绝给数），并支持与账本/区间对拍。
- ✅ **v1 进行中**：`tools/probes/occupancy_probe.cu` + `kernoscope occupancy` 打通
  「探针 → 分析 → 注入报告」闭环（达到的 occupancy，出处 `measured`）。
- 🚧 v1 剩余：region/阶段视图（对接 IKP Trace）、NVBit 指令与访存轨迹分析（sectors/req、locality）、
  `cuobjdump -sass` 静态指令混合细化到 per-kernel。
- 🚧 v2：因果链与 `explain`（借 ingero 思路，用 CUPTI activity + host 采样，不依赖 eBPF）、
  真 `.ncu-rep` 回读并与我们的报告对照。

## 许可

Apache License 2.0（见 [`LICENSE`](LICENSE)）。第三方与商标声明见 [`NOTICE`](NOTICE)。
