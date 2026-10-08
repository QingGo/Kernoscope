# Kernoscope W8 评测台结果

- agent：`(外部 agent)`
- 时间：2026-10-08T17:27:06+00:00　输出目录：`/tmp/ks-eval-a`
- 任务：5 个（失败 0 个）
- **decision_accuracy**：1.00（5/5）
- **misuse_rate（误用率）**：0.00（0 次 / 14 条 acted_on，0 个任务踩中）
- unsafe_rate（含禁止动作）：0.05（1 次 / 19 个决策位）；next_action 命中禁止动作的任务 1 个
- action_keyword_rate（行动方向命中真值关键词）：1.00（5/5）
- tokens_used 合计 0 · gpu_minutes 合计 0.00 · agent wall 合计 0.0s

> misuse_rate 的分母是 acted_on 条数：一条 acted_on 里出现「包里没有 / refused / 前提 ok=false / 真值禁止」的判词就算一次误用。

## 01_saturated_copy_dram_bound　✅

- 真值 classification：`dram_bound`；agent：`dram_bound`　一致
- acted_on：`kernel_time`、`dram_bytes`、`dram_bytes_per_launch`、`ledger_verdict`、`roofline`、`occupancy`
- blocked_by：（空）
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`dram_bytes_per_launch`=soft；`ledger_verdict`=soft；`occupancy`=ok；`roofline`=soft
- 包被尺寸预算裁剪过：['unmeasured_examples', 'verdict_metrics']（被裁掉的内容不能当成读过）
- 误用明细：无
- 动作关键词命中：['DRAM']（期望其中之一：['访存', '带宽', 'DRAM', '合并']）
- next_action：别在带宽上找空间：roofline 已 96.7%、离本机实测峰值只差 3.3%，occupancy 46.9% 对比上界 50% 也已贴顶；接下来只做减少总字节的工作——把相邻 pass 融合、或让中间结果靠 L2 复用（分块/调 tile）不出 DRAM，改完用 -k 单跑 vectorized_copy_kernel 并做 diff，确认差异超过 MDD 才算真改进。
- 真值理由：三个归因前提（窗口覆盖率 ≥80%、被匹配 kernel 份额 ≥95%、窗口被负载主导）全部成立，且两条口径（NVML 代理、roofline）都指向 memory bound。这时按访存优化是有依据的 —— 反过来说，用'别急，先改测量口径'去回避它，就是把一个已经成立的结论当成没有结论。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 02_multi_kernel_window_unattributable　✅

- 真值 classification：`window_unattributable`；agent：`unattributable`　一致
- acted_on：`kernel_time`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`、`ledger_verdict`、`occupancy`
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`dram_bytes_per_launch`=refused(前提 matched_kernel_kernel_share_pct 不成立)[share_too_low]；`ledger_verdict`=refused[share_too_low]；`occupancy`=soft
- 误用明细：无
- 动作关键词命中：['-k', '重跑']（期望其中之一：['缩小窗口', '-k', '重跑', '测量口径', '换轨迹']）
- next_action：窗口没有被缩小（0.9 s 内 4 个 kernel），现在任何 DRAM 数字都不属于 cutlass_gemm_f16：先用 -k cutlass_gemm_f16（必要时配合 --ks-memtrace 轨迹口径）把窗口收敛到只含这个 kernel 再重跑，拿到它自己的 DRAM/roofline/occupancy 判词之后才决定改不改代码；在拿到可归因的数字之前，不要按 25 MB/launch 或 +146.9% 的账本误差做任何优化。
- 真值理由：per-launch 归因的两个前提里，'被匹配 kernel 占窗口 kernel 时间 ≥95%'不成立（实测 41.0%）。于是 per-launch、窗口总量、与账本的差值**都不能归给这个 kernel**；+146.9% 的偏差本身就是归因错了的证据。唯一安全的下一步是先改测量口径（把窗口缩到只含目标 kernel，或换只覆盖它的轨迹口径），拿到可归因的读数再谈改代码。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 03_l2_resident_below_resolution　✅

- 真值 classification：`not_dram_bound`；agent：`below_resolution`　一致
- acted_on：`kernel_time`、`l2_locality`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`、`occupancy`
- 包内判词：`kernel_time`=ok；`dram_bytes`=refused[below_resolution]；`dram_bytes_per_launch`=refused[below_resolution]；`occupancy`=ok；`l2_locality`=soft
- 包被尺寸预算裁剪过：['unmeasured_examples']（被裁掉的内容不能当成读过）
- **误用明细（1 条）**：
    - [action] next_action 命中禁止动作：'compute'
- 动作关键词命中：['分辨率', '轨迹口径']（期望其中之一：['测量口径', '分辨率', '驻留', '延长窗口', '轨迹口径', '采样']）
- next_action：现在还不能反过来去优化计算：NVML 显存忙碌恒为 0.0% 是低于 1 个百分点分辨率被截断，不等于 DRAM 流量为零，DRAM 总量与 per-launch 两条口径都因此被拒；先把窗口撑到可分辨（保持 GPU 饱和、加长 bench_loop：kernoscope --ks-window auto -- python bench_loop.py，约 20 s），或加 --ks-memtrace 用轨迹口径拿到这个 kernel 的真实 DRAM 字节与 roofline，再决定是攻 compute 还是 mov。
- 真值理由：显存忙碌读数低于仪表分辨率 ⇒ 这个口径**没有**给出'DRAM 流量很小'，它给出的是'我量不出 DRAM'。把'量不出'读成'很小'、再顺势把方向定到计算侧，正是本项目失败史上最贵的一类误读。L2 驻留这一条是可信的（工作集 3.2 MiB vs L2 72 MiB），所以安全动作是：先改测量口径（延长窗口 / 用访存轨迹口径）拿到可信 DRAM 读数，在此之前不要按任何一个方向改代码。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 04_compute_bound_gemm　✅

- 真值 classification：`compute_bound`；agent：`compute_bound`　一致
- acted_on：`kernel_time`、`roofline`、`issue_slot`、`occupancy`
- blocked_by：`dram_bytes`
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`occupancy`=ok；`roofline`=soft；`issue_slot`=soft
- 误用明细：无
- 动作关键词命中：['roofline', '发射槽', '指令', 'compute']（期望其中之一：['计算', 'roofline', '发射槽', '指令', 'compute']）
- next_action：往 compute 侧走：roofline 判定 compute bound、离上界 21.6%，发射槽利用率 71.5% 且方向是 throughput——先补 --ks-instr-hist（约 600 s）拿到动态指令直方图，然后按吞吐方向精简指令（减少地址/循环开销、提高每指令有效工作量、必要时改数据类型/分块），不要花时间在带宽（显存忙碌仅 8%）或 occupancy（55.0% 对理论上界 62.5%，已接近）上。
- 真值理由：roofline 判词给出了 bound=compute（有 --ks-gemm 的工作量与实测峰值支撑），且显存忙碌只有 8%。dram_bytes 有数字、也不算被拒，但在'bound 已经判定为 compute'的前提下把它当**本次行动依据**就是方向性错误：真机上正是这样把 compute bound 的 GEMM 当访存问题改了两周。它可以被引用来排除访存，但不能出现在 acted_on 里。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 05_dram_caliber_unusable　✅

- 真值 classification：`dram_caliber_unusable`；agent：`dram_caliber_unusable`　一致
- acted_on：`kernel_time`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`、`occupancy`
- 包内判词：`kernel_time`=ok；`dram_bytes`=refused[no_nvml_samples]；`dram_bytes_per_launch`=refused[no_nvml_samples]；`occupancy`=ok
- 误用明细：无
- 动作关键词命中：['采样', '口径', 'NVML', '重跑']（期望其中之一：['测量口径', '采样', '计数器', '口径', 'NVML', '重跑']）
- next_action：这个问题现在无法回答，先修采样不要改代码：本次窗口 NVML 采样数 samples=0，没有任何读数被积分（曾经看到的 88% 只是未积分的瞬时不构成数字），DRAM 总量与 per-launch 两条口径都因此被拒；按 no_nvml_samples 的处置——去掉 --ks-no-nvml 让设备级采样生效，确认 samples≥1 后重跑 fused_attention_fwd，拿到经过积分的显存忙碌/DRAM 代理与 roofline，再判断 DRAM 还有没有空间。
- 真值理由：两条 DRAM 口径都被结构化拒绝（no_nvml_samples + counters_denied），能行动的只有 kernel 级结论。'拿不到 DRAM 数'不等于'DRAM 没问题'，也不等于'那就去优化计算'：唯一有依据的动作是先把测量口径修好（恢复 NVML 采样 / 拿计数器权限 / 装 NVBit 采轨迹）。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

