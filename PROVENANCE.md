# Provenance：每个数字都带出处

Kernoscope 的硬规则：**任何数字都必须能回答"这是怎么来的"**。在拿不到硬件计数器的机器上，
最危险的不是"没有数据"，而是**看起来合理的假数据** —— 一个被空闲稀释的带宽、一个把 L1 命中
误记成 L2 命中的命中率、一个少了 3 行 `continue` 就悄悄消失的裁决行。

所以"出处"不是文档里的口头约定，而是代码里的类型系统（`src/kernoscope/provenance.py`），
由测试守住。

## 五个出处等级

| 标签 | 含义 | 能否跨工具/跨机器比较 |
|---|---|---|
| `exact` | 与 ncu 同源：CUPTI activity 时间戳（nsys sqlite）、cubin/ELF 静态属性 | ✅ 可以 |
| `measured` | 我们真的读到的量：NVML 管理计数器（时钟/功耗/能量/显存忙碌）、探针实测的达到 occupancy、访存轨迹的直接计数 | ⚠️ 本机决策可以；跨机器比较要看口径 |
| `derived` | 由实测/精确量再**算**出来的统计量：采样窗覆盖率、账本误差、区间判词 | ⚠️ 同上 |
| `estimated` | 模型或代理：roofline、由 NVML 显存忙碌 × 实测峰值推出的 DRAM 字节、缓存模型给出的命中率 | ⚠️ 必须同时给出误差带与前提 |
| `unavailable` | 这台机器上拿不到，报 `n/a` + **原因** | —— |

`derived` 与 `measured` 的分界看起来挑剔，但它挡掉的是最常见的一类错误：
**把"由测量再算出来的东西"当成"测出来的东西"**。例如"采样窗里 60% 的点 GPU 在干活"是一个
统计量（`ks__sampler_window_coverage_pct`，`derived`），而"显存忙碌均值 77.3%"是一个读数
（`meas__nvml_mem_busy_mean_pct`，`measured`）。两者曾经共用一个名字 `meas__sampler_duty_pct`，
于是"窗口是否被负载主导"与"忙碌时段是否饱和"这两件不同的事被同一个数字回答 —— 这正是
"用单一表示承载多重语义"，也是本工具最想避免的错误形态。

## 四条强制规则

- **R1** `unavailable` 的指标必须 `value is None`，且必须有 `reason`。
  不允许"静默返回 0 或空"：学习者必须能区分「没有这一项」与「这一项是 0」。
- **R2** 计数器命名空间（`sm__ / smsp__ / l1tex__ / lts__ / dram__ / …`）下**不允许出现数字**。
  这些名字意味着"硬件计数器读出来的值"。当前实现更严：即使显式声明 `exact` 也不允许，
  除非该名字被登记在 `EXACT_COUNTER_ALLOWLIST`（当前为空）。
  想表达"这个量对应 ncu 的哪个指标"，请写在 `source` 字段里。
- **R3** 非 `exact` 的指标必须使用自有前缀：`est__` / `meas__` / `model__` / `ks__`。
  这条彻底堵死"在 ncu 指标名下面填一个模型算出来的数"。
- **R4** ncu 的 `launch__*` 与 `gpu__time_duration*` 允许 `exact`（我们确实同源可得）。

还有一条**命名纪律**：`meas__` 前缀的指标，出处必须是 `measured`。
名字承诺什么，出处就得是什么 —— 采样放大的轨迹量会改用 `est__` 前缀并在 note 里写明"按 N% 采样放大"。
三处输出口（主报告、`--json`、子命令的 `--json`）共用同一套前缀→出处判定
（`provenance_for_name()`），未登记的前缀绝不回落到 `exact`。

## 拒绝给数的三种形态

1. **口径不成立** → 那一行**照样出现**，值为 `n/a`，原因是**具体失败条件**
   （缺实测峰值 / 无采样点 / 读数低于分辨率下限 / 窗口覆盖率不足 80%），
   而不是复述算法定义。必要时给出补救路径（例如"把稳态循环延长到约 N 秒"）。
2. **物理上拿不到** → 例如 `dram__bytes.sum`，永远只报 `n/a` + 原因（需要计数器权限）。
3. **代理的前提被破坏** → 例如显存忙碌读数落在 NVML 的分辨率下限以下（`utilization.memory`
   是整数百分比，读数为 0.0% 时"真的是 0"与"0.3% 被截断成 0"不可区分），
   此时**不给字节数**，并明确写出这一点。

每一条代理量都会随数字给出自己的**分辨率**与采样周期；`utilization.memory` 的分辨率是
1 个百分点、采样周期 200 ms。

## CI 门禁

```bash
kernoscope --strict --set basic --ks-ledger-mb 234 -o rep -- ./bench
echo $?   # 0 = 请求的数据都拿到了；2 = 有请求的数据最终产不出来（含"裁决行算不出来"）
```

`--strict` 覆盖：`--metrics` 请求的指标、`--section` / `--set` 请求的 section、
空转的 ncu 兼容开关、以及**请求了账本对拍但裁决行没能算出来**这件事。
一个永远绿的 CI 比没有 CI 更危险。

## 常见指标速查

| 指标 | 出处 | 说明 |
|---|---|---|
| `gpu__time_duration.sum`、`ks__launch_time_*` | `exact` | CUPTI activity 时间戳；不 replay、不锁频 |
| `launch__*`（grid/block/寄存器/共享内存） | `exact` | 来自 CUPTI activity；与 cubin 交叉校验 |
| `ks__cfg_<配置>__*` | `exact` | 同一 kernel 的多组 launch 配置各自的读数（形状扫描） |
| `ks__theoretical_occupancy_pct` / `ks__occupancy_limiter` | `exact` | 静态属性 + 架构上限的解析算术 |
| `meas__achieved_occupancy_pct` | `measured` | 需要在被测 kernel 内插桩（`tools/probes/occupancy_probe.cu`） |
| `meas__nvml_mem_busy_mean_pct`、`meas__sm_clock_*`、`meas__power_max_w`、`meas__device_energy_j` | `measured` | NVML 管理计数器，免 admin |
| `est__dram_bytes_total` / `_per_launch` / `_throughput_pct_of_peak` | `estimated` | NVML 显存忙碌时间 × **本机实测峰值带宽**；per-launch 需要**两个**前提同时成立 |
| `est__dram_bytes_per_launch_trace` | `estimated` | 访存轨迹 + 缓存模型给出的第二条口径；只测被追踪的那个 kernel |
| `ks__sampler_window_coverage_pct` | `derived` | 前提①：窗口是否被被测负载主导（GPU 在不在干活） |
| `ks__matched_kernel_kernel_share_pct` | `derived` | 前提②：被 `-k` 选中的 kernel 占窗口**内核时间**的比例（判据） |
| `ks__matched_kernel_window_share_pct` | `derived` | 同上的墙钟口径；与覆盖率同量纲的对照量 |
| `ks__attribution_bias_estimate_pct` | `derived` | `100×(1/份额−1)`：归因本身引入的上偏估计（代理，不是严格上界） |
| `ks__window_busy_segments` | `derived` | 窗口里有几段连续忙碌期（多段 ⇒ 可能是多个阶段/变体拼接） |
| `ks__dram_window_mem_busy_mean_pct` | `derived` | **参与字节估算**的窗口显存忙碌均值（与整进程的 `meas__nvml_mem_busy_mean_pct` 区分开） |
| `ks__over_peak_warning` | `derived` | 达成值超过标定峰值 = 物理上不可能，显式红灯 |
| `est__dram_vs_ledger_err_pct`、`est__dram_range_position_pct`、`ks__ledger_range_verdict` | `estimated` / `derived` | 与你提供的账本对拍；区间判词 |
| `ks__static_sass_*` | `exact` | `cuobjdump -sass` 静态指令混合 |
| `sm__*`、`dram__*`、`l1tex__*`、`lts__*` 等计数器名 | `unavailable` | 只报 `n/a` + 原因，永不填数 |

## 相关文档

- [CALIBRATION.md](CALIBRATION.md)：本机峰值标定的字段与口径纪律（roofline 依赖它）。
- [README.md](README.md)：能力对照表与用法。
