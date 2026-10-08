# 本机峰值标定（machine calibration）

roofline 的分母通常是**规格表**上的峰值。在真实机器上这会让结论系统性偏乐观：一台
RTX 4090 的 fp32 matmul 实测常见 ~54 TFLOP/s，而规格表写 82.6 —— 拿规格表算会得出
"只发挥了 65%" 的错误结论，实际上那台机器的 fp32 上限就是 54（被 450 W 功耗墙限制）。

所以 Kernoscope 用**本机实测可达峰值**（achieved peak），而不是规格表。
标定脚本：`tools/machine-profile.py`。

```bash
# 推荐第一步（约 15 s）
python tools/machine-profile.py --quick            # 写出 ./machine-profile.json
# 全量（尺寸扫描更宽，约几分钟）
python tools/machine-profile.py --out bench/machine-profile.json
```

标定期间会同时采样 SM 时钟与功耗，用来确认"测到的是 boost 时钟下的数"，而不是空闲 210 MHz。

## 产物格式

```jsonc
{
  "schema_version": 2,
  "gpu": "NVIDIA GeForce RTX 4090",
  "cc": "8.9", "sm_count": 128, "l2_mib": 72.0,
  "nvidia_smi": "580.105.08, 3105, 450.00, 2, 16",   // driver, maxSM, power, pcie gen, width
  "measured_dram": {"copy_gbs": 908.5, "read_gbs": 904.5, "write_gbs": 942.0},
  "l2_info": {"l2_resident_max_mb": 64, "l2_drop_mb": 96, "l2_read_peak_gbs": 2180.1},
  "measured_matmul_tflops": {
    "torch.bfloat16": {
      "tflops": 167.3, "size": 16384, "ms": 52.5,
      "acc_caliber": "bf16 (fp32 accumulate)",
      "tflops_protocol": "default",          // 这个头条读数**就是**哪个变体测出来的
      "tflops_is_variant_reading": true,
      "precision_settings": {"allow_fp16_reduced_precision_reduction": true,
                             "fp32_precision": "none", "allow_tf32": true},
      "method": "torch.matmul, 3 iters best-of, size sweep [4096, 8192, 16384]",
      "caliber_variants_tflops": {"default": 167.3,
                                  "reduced_precision_reduction_off": 167.2,
                                  "fp32_ieee": 160.9},
      "caliber_variant_settings": {          // 每个变体自己的设置，不再只留一个快照
        "default": {"allow_fp16_reduced_precision_reduction": true,
                    "fp32_precision": "none"},
        "reduced_precision_reduction_off": {"allow_fp16_reduced_precision_reduction": false,
                                            "fp32_precision": "none"},
        "fp32_ieee": {"allow_fp16_reduced_precision_reduction": true,
                      "fp32_precision": "ieee"}
      },
      "caliber_probe_size": 16384,
      "caliber_probe_method": "torch.matmul, 3 iters best-of, 同一尺寸 16384",
      "caliber_spread_pct": 3.8,
      "caliber_distinguishable": true
    },
    "torch.float16": {
      "tflops": 166.9, "size": 4096, "ms": 0.82,
      "acc_caliber": "fp16 (fp32 accumulate; 允许 reduced-precision reduction)",
      "tflops_protocol": "fp32_ieee",        // 头条可能来自任一胜出协议，必须写清
      "tflops_is_variant_reading": true,
      "precision_settings": {"allow_fp16_reduced_precision_reduction": true,
                             "fp32_precision": "ieee", "allow_tf32": false},
      "method": "torch.matmul, 3 iters best-of, size sweep [4096, 8192, 16384]",
      "caliber_variants_tflops": {"default": 144.8,
                                  "reduced_precision_reduction_off": 159.4,
                                  "fp32_ieee": 166.9},
      "caliber_probe_size": 4096,
      "caliber_spread_pct": 13.2,
      "caliber_distinguishable": true
    },
    "torch.float32": {
      "tflops": 54.6, "size": 8192, "ms": 19.7,
      "acc_caliber": "fp32 (fp32 accumulate)",
      "tflops_protocol": "reduced_precision_reduction_off",
      "tflops_is_variant_reading": true,
      "precision_settings": {"allow_fp16_reduced_precision_reduction": false,
                             "fp32_precision": "none", "allow_tf32": false},
      "method": "torch.matmul, 3 iters best-of, size sweep [4096, 8192, 16384]",
      "caliber_variants_tflops": {"default": 54.4,
                                  "reduced_precision_reduction_off": 54.6,
                                  "fp32_ieee": 54.2},
      "caliber_probe_size": 8192,
      "caliber_spread_pct": 0.7,
      "caliber_distinguishable": false
    }
  },
  "precision_settings_at_end": { "...": "..." },
  "clocks_during_measurement": {"sm_clock_min": 1815.0, "sm_clock_max": 2625.0,
                                "sm_clock_mean": 2356.5, "power_max": 457.7,
                                "util_mean": 89.8, "n_samples": 10},
  "bandwidth_sweep": [{"size_mb": 128, "copy_gbs": 900.2, "...": "..."}],
  "torch": "2.12.1+cu130",
  "ts": "2026-10-08 23:47:41",
  "tool": "tools/machine-profile.py"
}
```

（以上是 [`bench/artifacts/20261008-235142/machine-profile.json`](bench/artifacts/) 的节选；
原始产物与 `bandwidth_sweep` 明细见同一目录。`tflops_protocol` 与
`caliber_variant_settings` 是后来补的字段，理由见下。）

## 为什么标定必须记录"累加口径"

matmul 的 TFLOP/s 不是机器常数，它取决于**累加口径**：fp16/bf16 输入用什么精度累加、
是否允许 reduced-precision reduction、fp32 是否走 TF32。同一台机器、同一个 `torch.mm`，
只改这几个设置就能差出可观的比例。

更关键的是：**"口径能不能被分辨"本身就是一个测量结果，不是一个可以假设的常量。**
同一台机器相隔一天的两次标定，一次给出三种设置最大相对差 0.4%（不可分辨），
另一次给出 5.0%（可分辨）。5% 的离散度足以让 roofline 在拐点附近翻转判定。

同一次标定里，不同 dtype 的结论也可以**相反** —— 上面那份产物就是：
bf16 三个口径差 3.8%（可分辨）、fp16 差 13.2%（可分辨）、而 fp32 只差 0.7%（**不可分辨**，
三种设置下都是 54.5 TFLOP/s 上下）。所以 fp32 那一档报"精确单值"是不诚实的，
报告里会带上 `caliber_spread_pct` 与 `caliber_distinguishable`，让读者自己判断。

所以标定产物必须带：

| 字段 | 作用 |
|---|---|
| `acc_caliber` | 这一档 TFLOP/s 是哪种累加口径下测得的（人可读） |
| `precision_settings` | 测量时刻的 torch 精度开关快照 |
| `method` | 测量方法（迭代数、取 best-of 还是 median、尺寸扫描范围） |
| `caliber_variants_tflops` | 三种口径设置各测一次（探针） |
| `caliber_spread_pct` | 三者最大相对差 |
| `caliber_distinguishable` | `spread ≥ 2%` 为可分辨；**为否时该档"精确单值"本身不可得** |
| `tflops_protocol` | 头条 `tflops` **是哪个变体**测出来的（`default` / `reduced_precision_reduction_off` / `fp32_ieee`；跳过口径探针时是 `sweep-default`） |
| `tflops_is_variant_reading` | 头条是否直接来自变体读数（否则它只来自尺寸扫描，复现不了） |
| `caliber_variant_settings` | 每个变体**自己的** precision 设置（从前 `_settings` 被丢弃，只剩一个事后快照） |
| `caliber_probe_size` / `caliber_probe_method` | 口径探针的尺寸与方法 —— `caliber_spread_pct` 只在**同协议**之间比较，读者要能复核这一点 |

**为什么头条必须能对应到一次具体测量**：真机产物曾经出现 fp16 的 `tflops` 是 167.4，
而同一尺寸下三个变体是 145.5 / 163.2 / 163.3 —— 头条**不等于任何一个变体**，
而 `precision_settings` 记的却是 `fp32_precision="ieee"`（对应 `fp32_ieee` 那一档）。
读者拿着这份产物无法复现这个数字，也无法判断该信哪一个。现在头条取"变体里最快的那一个"，
并写明它是谁：上面那份产物里 fp16 的头条 166.9 就是 `fp32_ieee` 变体的读数。

口径探针会在测完后恢复原始设置（有测试守这一点）。

## Kernoscope 怎么用它

```bash
kernoscope --set basic --ks-calibration machine-profile.json \
           --ks-gemm 4096,4096,4096 --ks-dtype bf16 -k regex:gemm -- ./bench
```

- **带宽峰值**取 `measured_dram.copy_gbs`：DRAM bound 的 kernel 基本是读写混合，
  用 write 峰值（最大的那个）会系统性高估流量。
- **算力峰值**取对应 dtype 的 `tflops`。**缺 `acc_caliber` / `precision_settings` / `method`
  字段时，Kernoscope 拒绝用它算 roofline 拐点**（`est__achieved_tflops`、`est__bound`、
  `est__pct_of_measured_compute_peak` 变成 `n/a` + 原因），带宽侧照常给数 ——
  字节数与显存忙碌代理不依赖累加口径。
- 报告里的每个峰值都会带上**来源与标定时间**（例如
  `measured peak (machine-profile.json @ 2026-10-08 22:00:00, dtype=bf16, bw=907 GB/s,
  compute=169.0 TFLOP/s, 累加口径=bf16 (fp32 accumulate), 口径离散=3.9%)`）。

⚠️ `--ks-peak-tflops` / `--ks-peak-gbps` 是**覆盖值**，不是标定：用它们时报告会明确标注来源
为覆盖值，并且不用于拐点判定。

## 无 GPU 时怎么办

标定需要真实 GPU。没有 GPU 的机器可以：

- `kernoscope --ks-synthetic` 跑通全流程（会标注 SYNTHETIC）；
- 用别人在同一型号机器上标出的 `machine-profile.json`（报告里会带该文件的来源与时间戳，
  所以"这是别处的数"是可追溯的）；
- 只做静态部分（`--ks-cubin` → LaunchStats / 理论 occupancy / SASS 指令混合），不需要 GPU。

`bench/` 目录下有完整的可复现流程（`bench/run.sh`）：环境快照、标定、已知答案的 DRAM 代理
精度、账本对拍、形状扫描、occupancy 探针，全部落成机器可读产物。
