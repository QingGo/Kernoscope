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

<!-- calib-excerpt:begin -->
<!-- 由 tools/gen_calibration_excerpt.py 生成（取自最新产物）；不要手改本块 -->

```jsonc
{
  "schema_version": 2,
  "gpu": "NVIDIA GeForce RTX 4090",
  "torch": "2.12.1+cu130",
  "ts": "2026-10-11 09:45:52",
  "tool": "tools/machine-profile.py",
  "measured_dram": {
    "copy_gbs": 910.0,
    "read_gbs": 907.5,
    "write_gbs": 940.6
  },
  "l2_info": {
    "l2_resident_max_mb": 64,
    "l2_drop_mb": 96,
    "l2_read_peak_gbs": 2195.8
  },
  "measured_matmul_tflops": {
    "torch.bfloat16": {
      "tflops": 163.3,
      "size": 16384,
      "ms": 53.8628,
      "acc_caliber": "bf16 (fp32 accumulate)",
      "precision_settings": {
        "allow_fp16_reduced_precision_reduction": true,
        "allow_tf32": false,
        "fp32_precision": "none",
        "float32_matmul_precision": null
      },
      "tflops_protocol": "default",
      "tflops_is_variant_reading": true,
      "caliber_variants_tflops": {
        "default": 163.3,
        "reduced_precision_reduction_off": 163.3,
        "fp32_ieee": 156.8
      },
      "caliber_probe_size": 16384,
      "caliber_spread_pct": 4.0,
      "caliber_distinguishable": true
    },
    "torch.float16": {
      "tflops": 159.5,
      "size": 8192,
      "ms": 6.8922,
      "acc_caliber": "fp16 (fp32 accumulate; 允许 reduced-precision reduction)",
      "precision_settings": {
        "allow_fp16_reduced_precision_reduction": false,
        "allow_tf32": false,
        "fp32_precision": "none",
        "float32_matmul_precision": null
      },
      "tflops_protocol": "reduced_precision_reduction_off",
      "tflops_is_variant_reading": true,
      "caliber_variants_tflops": {
        "default": 158.2,
        "reduced_precision_reduction_off": 159.5,
        "fp32_ieee": 159.5
      },
      "caliber_probe_size": 8192,
      "caliber_spread_pct": 0.8,
      "caliber_distinguishable": false
    },
    "torch.float32": {
      "tflops": 53.8,
      "size": 8192,
      "ms": 20.4247,
      "acc_caliber": "fp32 (fp32 accumulate)",
      "precision_settings": {
        "allow_fp16_reduced_precision_reduction": true,
        "allow_tf32": false,
        "fp32_precision": "ieee",
        "float32_matmul_precision": null
      },
      "tflops_protocol": "fp32_ieee",
      "tflops_is_variant_reading": true,
      "caliber_variants_tflops": {
        "default": 52.4,
        "reduced_precision_reduction_off": 53.1,
        "fp32_ieee": 53.8
      },
      "caliber_probe_size": 8192,
      "caliber_spread_pct": 2.6,
      "caliber_distinguishable": true
    }
  },
  "clocks_during_measurement": {
    "sm_clock_min": 1860.0,
    "sm_clock_max": 2535.0,
    "sm_clock_mean": 2331.8,
    "power_max": 455.74,
    "util_mean": 89.5,
    "n_samples": 11
  },
  "bandwidth_sweep": [
    {
      "size_mb": 2,
      "copy_gbs": 330.5636837471004,
      "read_gbs": 123.75246283167994,
      "write_gbs": 199.7693234284287
    },
    {
      "size_mb": 4,
      "copy_gbs": 771.9574267004622,
      "read_gbs": 253.06808425323084,
      "write_gbs": 408.08260487228125
    },
    {
      "…": "共 10 个尺寸，明细见产物"
    }
  ]
}
```

（以上是 [`bench/artifacts/20261011-094535/machine-profile.json`](bench/artifacts/) 的节选，**由该产物渲染**（`tools/gen_calibration_excerpt.py`）；原始产物与 `bandwidth_sweep` 明细见同一目录。新的 bench 运行会让这一段与门禁一起更新。）
<!-- calib-excerpt:end -->

## 为什么标定必须记录"累加口径"

matmul 的 TFLOP/s 不是机器常数，它取决于**累加口径**：fp16/bf16 输入用什么精度累加、
是否允许 reduced-precision reduction、fp32 是否走 TF32。同一台机器、同一个 `torch.mm`，
只改这几个设置就能差出可观的比例。

更关键的是：**"口径能不能被分辨"本身就是一个测量结果，不是一个可以假设的常量。**
几个百分点的离散度就足以让 roofline 在拐点附近翻转判定，而同一个 dtype 在不同次标定里
可以给出**相反**的结论（见下表最后一句的跨产物范围）。

同一次标定里，不同 dtype 的结论也可以**相反** —— 有的档位三种口径差到 4% 以上
（可分辨），有的只差不到 1%（**不可分辨**，此时报"精确单值"不诚实）。具体数字见下表
（由最新产物渲染，与上面的节选同源）：

<!-- calib-caliber:begin -->
<!-- 由 tools/gen_calibration_excerpt.py 生成（取自最新产物）；不要手改本块 -->

| dtype | 头条 TFLOP/s | 头条来自哪个变体 | 三个口径变体的读数 | 最大相对差 | 可分辨（≥2%） |
|---|---|---|---|---|---|
| `torch.bfloat16` | **163.3** | `default` | default 163.3 / reduced_precision_reduction_off 163.3 / fp32_ieee 156.8 | 4.0% | **是** |
| `torch.float16` | **159.5** | `reduced_precision_reduction_off` | default 158.2 / reduced_precision_reduction_off 159.5 / fp32_ieee 159.5 | 0.8% | **否** |
| `torch.float32` | **53.8** | `fp32_ieee` | default 52.4 / reduced_precision_reduction_off 53.1 / fp32_ieee 53.8 | 2.6% | **是** |

（本表由 `bench/artifacts/20261011-094535/machine-profile.json` 渲染。同一 dtype 的三口径最大相对差在不同产物里可以跨过 2% 门槛 —— 跨已登记产物看：`torch.bfloat16` 3.8%~4.0%；`torch.float16` 0.8%~13.2%；`torch.float32` 0.4%~2.6%。所以「口径能不能被分辨」本身是**测量结果**，不是常量。本产物的三个 dtype 头条**都等于**某个变体读数 ✓。）
<!-- calib-caliber:end -->

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

**为什么头条必须能对应到一次具体测量**：真机产物曾经出现过 fp16 的 `tflops`
**不等于任何一个变体读数**（而 `precision_settings` 记的却是另一档的设置）—— 那个数字
读者既复现不了、也无法判断该信哪个。
读者拿着这份产物无法复现这个数字，也无法判断该信哪一个。现在头条取"变体里最快的那一个"，
并写明它是谁 —— **上表**里每个 dtype 的"头条来自哪个变体"就是那个字段（`tflops_protocol`），
它必须等于三个变体读数里的某一个。

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
- 报告里的每个峰值都会带上**来源与标定时间**：`measured peak (<标定文件> @ <时间>,
  dtype=…, bw=… GB/s, compute=… TFLOP/s, 累加口径=…, 口径离散=…%)` —— 括号里的具体数字
  来自那次标定的产物（本页的生成块里就有当前这一份的对应值）。

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
