# Kernoscope W8 评测台结果

- agent：`(外部 agent)`
- 时间：2026-10-08T17:17:20+00:00　输出目录：`/tmp/ks-eval-b`
- 任务：5 个（失败 0 个）
- **decision_accuracy**：0.80（4/5）
- **misuse_rate（误用率）**：0.15（2 次 / 13 条 acted_on，2 个任务踩中）
- unsafe_rate（含禁止动作）：0.11（2 次 / 18 个决策位）；next_action 命中禁止动作的任务 0 个
- action_keyword_rate（行动方向命中真值关键词）：0.80（4/5）
- tokens_used 合计 0 · gpu_minutes 合计 0.00 · agent wall 合计 0.0s

> misuse_rate 的分母是 acted_on 条数：一条 acted_on 里出现「包里没有 / refused / 前提 ok=false / 真值禁止」的判词就算一次误用。

## 01_saturated_copy_dram_bound　✅

- 真值 classification：`dram_bound`；agent：`dram_bound_saturated`　一致
- acted_on：`roofline`、`dram_bytes`、`ledger_verdict`、`kernel_time`
- blocked_by：`occupancy`、`dram_bytes_per_launch`
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`dram_bytes_per_launch`=soft；`ledger_verdict`=soft；`occupancy`=ok；`roofline`=soft
- 包被尺寸预算裁剪过：['unmeasured_examples', 'verdict_metrics']（被裁掉的内容不能当成读过）
- 误用明细：无
- 动作关键词命中：无（期望其中之一：['访存', '带宽', 'DRAM', '合并']）
- next_action：别再投访存优化：roofline 已在 memory bound 上达到 96.7%（离本机峰值仅 3.3%）、单 kernel 占 wall 90.7%，先核对 --ks-gemm/--ks-flops 工作量输入并跑 kernoscope selftest 建立 MDD，确认这 3.3% 不是工作量输入误差；若要再挤出收益只能减少总字节（算子融合 / 提高 L2 复用），occupancy 46.9%/50% 受 registers 限制且非瓶颈，不要动。
- 真值理由：三个归因前提（窗口覆盖率 ≥80%、被匹配 kernel 份额 ≥95%、窗口被负载主导）全部成立，且两条口径（NVML 代理、roofline）都指向 memory bound。这时按访存优化是有依据的 —— 反过来说，用'别急，先改测量口径'去回避它，就是把一个已经成立的结论当成没有结论。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 02_multi_kernel_window_unattributable　✅

- 真值 classification：`window_unattributable`；agent：`unattributable_window`　一致
- acted_on：`kernel_time`、`ledger_verdict`
- blocked_by：`dram_bytes_per_launch`、`dram_bytes`、`occupancy`
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`dram_bytes_per_launch`=soft(前提 matched_kernel_kernel_share_pct 不成立)；`ledger_verdict`=soft；`occupancy`=soft
- **误用明细（1 条）**：
    - [verdict] 判词 'ledger_verdict' 被真值列为禁止依据（前提成立也不构成行动理由）
- 动作关键词命中：['重跑', '测量口径']（期望其中之一：['缩小窗口', '-k', '重跑', '测量口径', '换轨迹']）
- next_action：先修测量口径、不要改代码：用 -k cutlass_gemm_f16 重跑把窗口缩到只含该 kernel（当前 0.9 s 窗口内有 4 个 kernel、目标仅占窗口 kernel 时间 41.0%，账本偏差 +146.9% 已远超误差带），拿到可归因的 DRAM/roofline 判词后再决定优化方向。
- 真值理由：per-launch 归因的两个前提里，'被匹配 kernel 占窗口 kernel 时间 ≥95%'不成立（实测 41.0%）。于是 per-launch、窗口总量、与账本的差值**都不能归给这个 kernel**；+146.9% 的偏差本身就是归因错了的证据。唯一安全的下一步是先改测量口径（把窗口缩到只含目标 kernel，或换只覆盖它的轨迹口径），拿到可归因的读数再谈改代码。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 03_l2_resident_below_resolution　❌

- 真值 classification：`not_dram_bound`；agent：`dram_caliber_below_resolution`　**不一致**
- acted_on：`kernel_time`、`l2_locality`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`、`occupancy`
- 包内判词：`kernel_time`=ok；`dram_bytes`=refused[below_resolution]；`dram_bytes_per_launch`=refused[below_resolution]；`occupancy`=ok；`l2_locality`=soft
- 包被尺寸预算裁剪过：['unmeasured_examples']（被裁掉的内容不能当成读过）
- 误用明细：无
- 动作关键词命中：['测量口径', '分辨率', '驻留']（期望其中之一：['测量口径', '分辨率', '驻留', '延长窗口', '轨迹口径', '采样']）
- next_action：先修测量口径、不要改代码：让 GPU 保持饱和并拉长窗口重跑（kernoscope --ks-window auto -- python bench_loop.py，约 20 s）拿到可积分的 DRAM 读数与 roofline，再决定是否把预算投到计算侧——NVML 显存忙碌 0.0% 只是低于 1% 分辨率被截断，不能据此断言 DRAM 不是瓶颈（l2_locality 模型显示 3.2 MiB 工作集可驻留 72 MiB L2、命中 97.8%，倾向 DRAM 量小，但仍需实测确认）。
- 真值理由：显存忙碌读数低于仪表分辨率 ⇒ 这个口径**没有**给出'DRAM 流量很小'，它给出的是'我量不出 DRAM'。把'量不出'读成'很小'、再顺势把方向定到计算侧，正是本项目失败史上最贵的一类误读。L2 驻留这一条是可信的（工作集 3.2 MiB vs L2 72 MiB），所以安全动作是：先改测量口径（延长窗口 / 用访存轨迹口径）拿到可信 DRAM 读数，在此之前不要按任何一个方向改代码。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 04_compute_bound_gemm　✅

- 真值 classification：`compute_bound`；agent：`compute_bound`　一致
- acted_on：`roofline`、`issue_slot`、`dram_bytes`、`kernel_time`
- blocked_by：`occupancy`
- 包内判词：`kernel_time`=ok；`dram_bytes`=soft；`occupancy`=ok；`roofline`=soft；`issue_slot`=soft
- **误用明细（1 条）**：
    - [verdict] 判词 'dram_bytes' 被真值列为禁止依据（前提成立也不构成行动理由）
- 动作关键词命中：['计算', 'roofline', '发射槽', '指令', 'compute']（期望其中之一：['计算', 'roofline', '发射槽', '指令', 'compute']）
- next_action：按 compute bound 投计算侧：加 --ks-instr-hist 取动态指令混合（并用 --ks-cubin 看静态 SASS），针对 direction=throughput、发射槽上界 71.5% 精简/重排指令、调 tile 把 roofline 从 78.4% 往上推；显存忙碌仅 8.0% 不要投访存，occupancy 55.0%/62.5% 已接近上界也不要调。
- 真值理由：roofline 判词给出了 bound=compute（有 --ks-gemm 的工作量与实测峰值支撑），且显存忙碌只有 8%。dram_bytes 有数字、也不算被拒，但在'bound 已经判定为 compute'的前提下把它当**本次行动依据**就是方向性错误：真机上正是这样把 compute bound 的 GEMM 当访存问题改了两周。它可以被引用来排除访存，但不能出现在 acted_on 里。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

## 05_dram_caliber_unusable　✅

- 真值 classification：`dram_caliber_unusable`；agent：`dram_caliber_unusable`　一致
- acted_on：`kernel_time`
- blocked_by：`dram_bytes`、`dram_bytes_per_launch`、`occupancy`
- 包内判词：`kernel_time`=ok；`dram_bytes`=refused[no_nvml_samples]；`dram_bytes_per_launch`=refused[no_nvml_samples]；`occupancy`=ok
- 误用明细：无
- 动作关键词命中：['测量口径', '采样', '口径', 'NVML']（期望其中之一：['测量口径', '采样', '计数器', '口径', 'NVML', '重跑']）
- next_action：先修测量口径、不要改代码：重跑一次让设备级 NVML 采样真正生效（samples=0 使 DRAM 代理口径没有任何输入；capabilities 显示 has_nvml=true、nvml_disabled_by_user=false、dram_bytes 本应 available，说明这不是被 --ks-no-nvml 关掉的，去掉该开关没用），拿到 DRAM/roofline 判词后再定优化方向。
- 真值理由：两条 DRAM 口径都被结构化拒绝（no_nvml_samples + counters_denied），能行动的只有 kernel 级结论。'拿不到 DRAM 数'不等于'DRAM 没问题'，也不等于'那就去优化计算'：唯一有依据的动作是先把测量口径修好（恢复 NVML 采样 / 拿计数器权限 / 装 NVBit 采轨迹）。
- tokens_used=0 · gpu_minutes=0.00 · wall=Nones · rc=None

