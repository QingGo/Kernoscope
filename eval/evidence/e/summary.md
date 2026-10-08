# Kernoscope W8 评测台结果

- agent：`(外部 agent)`
- 时间：2026-10-08T17:27:06+00:00　输出目录：`/tmp/ks-eval-c`
- 任务：5 个（失败 0 个）
- **decision_accuracy**：1.00（5/5）
- **misuse_rate（误用率）**：0.00（0 次 / 11 条 acted_on，0 个任务踩中）
- unsafe_rate（含禁止动作）：0.00（0 次 / 16 个决策位）；next_action 命中禁止动作的任务 0 个
- action_keyword_rate（行动方向命中真值关键词）：1.00（5/5）
- tokens_used 合计 0 · gpu_minutes 合计 0.00 · agent wall 合计 0.0s

> misuse_rate 的分母是 acted_on 条数：一条 acted_on 里出现「包里没有 / refused / 前提 ok=false / 真值禁止」的判词就算一次误用。

## 01_saturated_copy_dram_bound　✅

- 真值 classification：`dram_bound`；agent：`dram_bound_saturated`　一致
- acted_on：`roofline`、`dram_bytes`、`ledger_verdict`
- blocked_by：（空）
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`dram_bytes_per_launch`=soft；`ledger_verdict`=soft；`occupancy`=ok；`roofline`=soft
- 包被尺寸预算裁剪过：['unmeasured_examples', 'verdict_metrics']（被裁掉的内容不能当成读过）
- 误用明细：无
- 动作关键词命中：['访存', 'DRAM']（期望其中之一：['访存', '带宽', 'DRAM', '合并']）
- next_action：访存侧已无空间：roofline 判为 memory bound、达成 919 GB/s 离本机峰值仅 3.3%（效率 96.7%），先核对 --ks-gemm/--ks-flops 等工作量输入并用 --ks-ledger-total-mb 对拍账本（当前 -1.2%）确认口径无误，之后唯一还有收益的方向是减少 DRAM 总字节（算子融合 / 去掉这次纯拷贝 / 提高 L2 复用），不要再调访存 pattern 或 occupancy。
- 真值理由：三个归因前提（窗口覆盖率 ≥80%、被匹配 kernel 份额 ≥95%、窗口被负载主导）全部成立，且两条口径（NVML 代理、roofline）都指向 memory bound。这时按访存优化是有依据的 —— 反过来说，用'别急，先改测量口径'去回避它，就是把一个已经成立的结论当成没有结论。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 02_multi_kernel_window_unattributable　✅

- 真值 classification：`window_unattributable`；agent：`unattributable_window`　一致
- acted_on：`kernel_time`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`、`ledger_verdict`
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`dram_bytes_per_launch`=refused(前提 matched_kernel_kernel_share_pct 不成立)[share_too_low]；`ledger_verdict`=refused[share_too_low]；`occupancy`=soft
- 误用明细：无
- 动作关键词命中：['-k', '重跑']（期望其中之一：['缩小窗口', '-k', '重跑', '测量口径', '换轨迹']）
- next_action：先重跑把窗口缩到只含 cutlass_gemm_f16（kernoscope --ks-window-s 10 -k regex:cutlass_gemm_f16 -- python ...，约 20 s，见 refusals.share_too_low），拿到该 kernel 自己的 per-launch DRAM / occupancy / roofline 判词后再决定优化方向；不要拿窗口级的 5.0 GB 或 +146.9% 账本误差作为这个 kernel 的依据。
- 真值理由：per-launch 归因的两个前提里，'被匹配 kernel 占窗口 kernel 时间 ≥95%'不成立（实测 41.0%）。于是 per-launch、窗口总量、与账本的差值**都不能归给这个 kernel**；+146.9% 的偏差本身就是归因错了的证据。唯一安全的下一步是先改测量口径（把窗口缩到只含目标 kernel，或换只覆盖它的轨迹口径），拿到可归因的读数再谈改代码。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 03_l2_resident_below_resolution　✅

- 真值 classification：`not_dram_bound`；agent：`dram_proxy_below_resolution`　一致
- acted_on：`kernel_time`、`l2_locality`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`
- 包内判词：`kernel_time`=ok；`dram_bytes`=refused[below_resolution]；`dram_bytes_per_launch`=refused[below_resolution]；`occupancy`=ok；`l2_locality`=soft
- 包被尺寸预算裁剪过：['unmeasured_examples']（被裁掉的内容不能当成读过）
- 误用明细：无
- 动作关键词命中：['测量口径']（期望其中之一：['测量口径', '分辨率', '驻留', '延长窗口', '轨迹口径', '采样']）
- next_action：先修测量口径再谈方向：把该 kernel 放进能维持显存忙碌 ≥1% 的饱和窗口重跑（kernoscope --ks-window auto -- python bench_loop.py，约 20 s；现在窗口仅 0.021 s、忙碌读数被截断成 0.0%），并顺手 -k 单跑取它的 occupancy/roofline 判词，拿到可积分的 DRAM 代理之前不要下结论。
- 真值理由：显存忙碌读数低于仪表分辨率 ⇒ 这个口径**没有**给出'DRAM 流量很小'，它给出的是'我量不出 DRAM'。把'量不出'读成'很小'、再顺势把方向定到计算侧，正是本项目失败史上最贵的一类误读。L2 驻留这一条是可信的（工作集 3.2 MiB vs L2 72 MiB），所以安全动作是：先改测量口径（延长窗口 / 用访存轨迹口径）拿到可信 DRAM 读数，在此之前不要按任何一个方向改代码。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 04_compute_bound_gemm　✅

- 真值 classification：`compute_bound`；agent：`compute_bound`　一致
- acted_on：`roofline`、`issue_slot`、`dram_bytes`、`occupancy`
- blocked_by：（空）；额外依据（不在真值建议里）：`dram_bytes`
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`occupancy`=ok；`roofline`=soft；`issue_slot`=soft
- 误用明细：无
- 动作关键词命中：['计算', '发射槽', '指令', 'compute']（期望其中之一：['计算', 'roofline', '发射槽', '指令', 'compute']）
- next_action：沿 compute 方向走：加 --ks-instr-hist <opcode_hist.txt> 拿动态指令混合（occupancy 探针可省），针对发射槽上界 71.5%、方向 throughput 做指令精简——削掉占比高的地址计算/类型转换/谓词指令、提高每条指令的有效工作量，改完再跑 diff 验证；别动访存（显存忙碌 8.0%、71 GB/s）与 occupancy（55% 已接近 62.5% 上界、受 registers 限制）。
- 真值理由：roofline 判词给出了 bound=compute（有 --ks-gemm 的工作量与实测峰值支撑），且显存忙碌只有 8%。dram_bytes 有数字、也不算被拒，但在'bound 已经判定为 compute'的前提下把它当**本次行动依据**就是方向性错误：真机上正是这样把 compute bound 的 GEMM 当访存问题改了两周。它可以被引用来排除访存，但不能出现在 acted_on 里。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 05_dram_caliber_unusable　✅

- 真值 classification：`dram_caliber_unusable`；agent：`dram_proxy_no_nvml_samples`　一致
- acted_on：`kernel_time`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`
- 包内判词：`kernel_time`=ok；`dram_bytes`=refused[no_nvml_samples]；`dram_bytes_per_launch`=refused[no_nvml_samples]；`occupancy`=ok
- 误用明细：无
- 动作关键词命中：['采样', '口径', 'NVML', '重跑']（期望其中之一：['测量口径', '采样', '计数器', '口径', 'NVML', '重跑']）
- next_action：先修口径、不要动代码：去掉 --ks-no-nvml 用同一窗口重跑 fused_attention_fwd（capabilities 显示 has_nvml=true、device_busy/dram_bytes 均为 available，但本窗口 samples=0），确认 NVML 采样生效、显存忙碌能被积分后再重新生成 packet 回答“DRAM 还有多少空间”。
- 真值理由：两条 DRAM 口径都被结构化拒绝（no_nvml_samples + counters_denied），能行动的只有 kernel 级结论。'拿不到 DRAM 数'不等于'DRAM 没问题'，也不等于'那就去优化计算'：唯一有依据的动作是先把测量口径修好（恢复 NVML 采样 / 拿计数器权限 / 装 NVBit 采轨迹）。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

