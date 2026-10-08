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
| 是不是 DRAM bound | NVML 显存忙碌占比 × 本机实测峰值带宽 | `estimated`（稳态，误差约 6%） |
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
- 🚧 v1：region/阶段视图（对接 IKP Trace）、NVBit 指令与访存轨迹分析（sectors/req、locality）、
  `cuobjdump -sass` 静态指令混合细化。
- 🚧 v2：因果链与 `explain`（借 ingero 思路，用 CUPTI activity + host 采样，不依赖 eBPF）、
  真 `.ncu-rep` 回读并与我们的报告对照。

## 许可

Apache License 2.0（见 [`LICENSE`](LICENSE)）。第三方与商标声明见 [`NOTICE`](NOTICE)。
