# demo —— 30 秒看清这个工具能给你什么

```bash
bash demo/quickstart.sh                    # 默认：两个场景（GEMM + 一步训练），约 1 分钟
bash demo/quickstart.sh --scene gemm       # 只跑 GEMM：单 kernel，结论最干净（适合做封面图）
bash demo/quickstart.sh --scene train_step # 只跑一步训练：多 kernel + 阶段归因 + 每 token 成本
bash demo/quickstart.sh --price-per-hour 2.5   # 用你自己的租用价（元/小时）
bash demo/quickstart.sh --kwh 0.8              # 或电价（元/kWh）
bash demo/quickstart.sh --no-ncu               # 不在这里跑 ncu
```

产物在 `demo/out/`（已 gitignore）：`onepage-*.txt`（纯文本一屏，直接截图）、
`rep-*.html`（**单文件**证据单，离线可开、可当附件）、`packet-*.json`（决策包）、
`machine-profile.json`（本机标定）、`ncu.err`（ncu 的原始报错）。

## 它演四幕

| 幕 | 做什么 | 为什么值得看 |
|---|---|---|
| ① ncu 在这台机器上能不能用 | **真的跑一次** `ncu --set basic`，把它的报错原样登出来 | 受限机器上 `ERR_NVGPUCTRPERM` 是常态；这不是我们的报错，但决定了后面所有口径 |
| ② 这台机器上我们能回答什么 | `capabilities --brief`（约 1 KB）+ `doctor` 的计数器状态说明 | 一眼看清"能答/A 拿不到/下一步"，不用读完整文档 |
| ③ 一次把能测的都测上 | **先做本机标定**（`--quick`，约 3 秒）→ 带标定、示例价格、token 数跑负载 → 一屏 + HTML | 标定是 roofline 与"能省多少"的分母；价格让"每 token 多少钱"真的出数 |
| ④ 拿不到的那些，我们给什么 | 从**决策包**现场生成（不手抄） | 计数器被拒的每一项都有替代口径或明确的下一步 |

## 三类"拿不到"，三种不同的意思

demo 的第④幕会逐条标出来，这里先讲清区别 —— 它们**不是**同一件事：

| 桶 | 例子 | 意思 |
|---|---|---|
| **平台挡住·有替代口径** | `counters_denied`（80 项） | 这台机器的策略不让读硬件计数器。不是能力缺口 —— 我们给物理观测量的替代口径（时长/带宽代理/占用率模型/能耗） |
| **补一条命令就能答** | `no_memtrace`、`no_instr_hist`、`no_occupancy_probe`、`no_static_input` | 需要你给一份文件或跑一次探针（NVBit 那两条**不需要计数器权限**）。每条都带命令与代价 |
| **需要额外采集** | `no_workload`（给 FLOPs/bytes）、`mixed_launch_config`（同名 kernel 多组配置） | 补输入或换口径就能答；不改机器权限 |

**因此"没测到 100 多项"不等于"这个工具没用"**：一屏会同时给出**本次真的测到了多少项**，
以及与 ncu 的分工 ——

| 你想知道的 | 受限机器上的 ncu | 我们 |
|---|---|---|
| 每个 kernel 的时长 / launch 配置 | ❌ 取不到计数器（这台机器实测 `ERR_NVGPUCTRPERM`） | ✅ **exact**（nsys CUPTI activity），且给逐 kernel 排序 |
| 离 roofline 多远、compute/memory bound | ❌ | ✅ 需一次本机标定（demo 里现测），给区间与**达成率** |
| DRAM/L2 **精确字节**、命中率、warp stall 细分 | ❌ | ⚠️ 只给到代理口径（`estimated`，带误差带）或明确拒绝 + 替代口径；**绝不把估计写成实测** |
| 每 token 多少钱、这次训练花了多少电 | ❌ ncu 没有这类指标 | ✅ NVML 能耗 + 你给的 token 数与价格 |
| 阶段（NVTX）各花多少时间 | ❌ | ✅ nsys 区间重叠归因 |
| 时钟是否被锁（跨运行可比性） | ncu 自己会锁频 | ✅ 读锁频状态并进协议哈希；**我们不锁**（写操作不在只读工具的授权里） |
| 硬件计数器本身 | ✅ | ❌ 明确拒绝，并说明为什么（策略/capability） |

## 想看"钱"用哪个场景

`--scene gemm` 会带上工作量输入（`--ks-gemm M,N,K --ks-dtype bf16`），于是标定（分母）+ 价格
一起作用，处方能给出**区间与金额**，真机实测长这样：

```
[ok  ] 先改 ampere_bf16_s1688gemm_…（保守可省 87.9%~上界 93.5%）；合计可省 1.9462 CNY/h
```

`train_step` 场景在共享容器上常被 `coverage_too_low` 挡住（窗口里 GPU 忙占比 <80% ⇒
比值类口径不成立）—— 这**不是**丢结论，而是"这个窗口不足以支撑那个比值"，一屏会给出原因与
它挡住了哪条结论。想让这条口径成立：把负载跑长跑大（`DEMO_SECONDS=10 DEMO_BATCH=32`），
或者换独占机器。

⚠️ **已知缺口（真机 demo 跑出来的）**：`--scene gemm` 里 `over_peak` 可能因
`calibration_caliber_mismatch` 被拒（我们测到的峰值口径与标定不一致），而处方仍然给出区间与
金额 —— 按本仓库自己的纪律（守卫不成立 ⇒ 由它算出来的结论不可依据），**处方也该一起作废**。
这条已记入 CHANGELOG 待修。

## 常见问题

- **要 GPU 吗？** 要（第③幕要真跑负载）。只想验证"装完能不能跑"用
  `bash bench/preflight-first-run.sh`（**不需要 GPU**，约 2 秒）。
- **为什么 grep 到两个 kernoscope？** demo 永远用**它自己所在的源码树**（`src/`）；如果 `PATH` 上
  另有旧版本，它会把这个事实打出来。真机踩过：`/usr/local/bin/kernoscope` 是别人留下的包装脚本，
  指向冻结的旧版，连 `--html` 都没有 —— 按 `PATH` 跑，看到的结论会与被测代码对不上。
- **价格从哪来？** 示例值（`--price-per-hour 2.5` 元/小时）。用你自己的：`--kwh` 给电价，
  或 `--price-per-hour` 给租用价。没给价格时工具**不会编钱**，只会说"给不出"并给条件估计。
- **和 `bench/` 什么关系？** `bench/` 是**证据层**（可复现、登记、发布前门禁）；`demo/` 是
  **体验层**（陌生人在 30 秒内看到价值）。两者不重复：demo 的数字来自它自己刚跑的那一次。
- **runs 多久？** 两个场景约 1 分钟（含 3 秒标定 + 3 秒/场景的负载）。
