<!-- 由 tools/gen_playbook.py 生成；改动请改代码后重跑，不要手改本文件 -->

# 受限环境诊断手册（signal → cause → fix）

这份手册给的是**受限机器**（拿不到硬件性能计数器、`ncu` 跑不起来）上的处置路径：
看到哪个拒绝码、它是什么意思、下一步跑什么、要花多久。

本文件由 `tools/gen_playbook.py` 从 `src/kernoscope/refusals.py`、`route_cmd.py`、
`capabilities.py`、`sections.py` **现算生成** —— **散文只是视图，代码才是真源**；
手改本文件会在 `tools/dev_index.py --gates` 的 `playbook` 门禁上红。

三条读法：

* 拒绝码是**闭集**（51 个）：见到 `n/a` 先看 `code`，再照 remedy 做 —— 不要猜原因。
* `cost_kind=unavailable` 的 code 单独成节：它们**不是**「环境缺件」，
  在这台机器上没有任何合法补救路径（容器内 `sudo`/`setuid` 也突破不了）。
* 每一列都注明来源；代码里算不出来的列写 `——`，不编（这一条比表格好看更重要）。

## 一、signal → cause → fix（按拒绝码）

覆盖面：闭集里的 51 个 code 全部登记，其中 48 个在本机有补救路径（下表），3 个没有（见下一节）。**不做「受限机器上不可能发生」的人工筛选** —— 那需要判断，而判断题不该藏在生成器里；唯一的筛选规则就是 `cost_kind=unavailable`。

「观测到什么」这一列只给**触发路由**：数值阈值（`Refusal.threshold`）由各 builder 现算，代码里**没有**按 code 索引的静态阈值表，所以这一列给不出数字 —— 这是缺口，不是省略。

| code | 人话（CODE_TITLES） | 观测到什么（阈值/条件） | 怎么办（remedy.action + command） | 代价 |
|---|---|---|---|---|
| `no_gpu` | 本机没有 NVIDIA GPU | 1 条路由会报它（`device_busy`） | 换一台有 NVIDIA GPU 的机器；或在没有 GPU 时用 --ks-synthetic 跑通流程看输出形状；命令：`kernoscope --ks-synthetic --set basic` | 需人工 |
| `pytorch_missing` | 没有安装 PyTorch | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 装 PyTorch（自检的已知答案负载只需要 torch）；命令：`pip install torch` | 需人工 |
| `no_nvml` | nvidia-smi / pynvml 不可用 | 1 条路由会报它（`device_busy`） | 装 pynvml（pip install 'kernoscope[nvml]'）或确认 nvidia-smi 可用；命令：`pip install 'kernoscope[nvml]'` | 需人工 |
| `no_nvml_samples` | 没有 NVML 采样点（关掉采样／不可用／稳态段短于 200 ms） | 1 条路由会报它（`device_busy`） | 去掉 --ks-no-nvml 让设备级采样生效（代价接近 0：采样在后台线程里做） | ≈0 |
| `nsys_missing` | 没有 nsys：kernel 级数据不可得 | 1 条路由会报它（`kernel_time`） | 安装 Nsight Systems（随 CUDA Toolkit 提供），或提供已有的 .nsys-rep/.sqlite；命令：`kernoscope --ks-sqlite <kernels.sqlite>` | 需人工 |
| `cuobjdump_missing` | 没有 cuobjdump：静态资源/SASS 不可得 | 1 条路由会报它（`static_resources`） | 安装 CUDA Toolkit（cuobjdump 随 toolkit 提供），或用已有报告里的静态字段 | 需人工 |
| `no_static_input` | 没有提供 cubin/.so | 3 条路由会报它（`static_resources`、`theoretical_occupancy`、`static_mix`） | 用 --ks-cubin <cubin/.so> 指向被测二进制；没有它就没有寄存器/共享内存/SASS；命令：`kernoscope --ks-cubin build/app.so ...` | 需人工 |
| `nvbit_missing` | 未配置 NVBit：指令级/访存轨迹不可得 | 2 条路由会报它（`dram_bytes_trace`、`issue_slot`） | 自行安装 NVBit（许可证禁止本项目随包分发）并设 NVBIT_PATH | 需人工 |
| `no_memtrace` | 没有提供访存轨迹文件 | 2 条路由会报它（`dram_bytes_trace`、`l2_locality`） | 加 --ks-memtrace <trace>：轨迹口径只测被追踪的 kernel，不受窗口构成影响；命令：`kernoscope --ks-memtrace ks-memtrace.txt ...` | ≈10min |
| `no_instr_hist` | 没有提供指令直方图 | 1 条路由会报它（`issue_slot`） | 加 --ks-instr-hist <opcode_hist.txt>：给出动态指令数与发射槽利用率上界；命令：`kernoscope --ks-instr-hist opcode_hist.txt ...` | ≈10min |
| `no_occupancy_probe` | 没跑 occupancy 探针 | 1 条路由会报它（`achieved_occupancy`） | 跑 occupancy 探针并把 JSON 用 --ks-achieved-occupancy 传进来（需要 NVBit）；命令：`kernoscope occupancy --records <probe-records> -o occ.json` | ≈10min |
| `no_sass_source` | SASS/源码逐行对照未实现 | 1 条路由会报它（`static_mix`） | SASS/源码逐行对照需要 --import-source，本项目不实现：改用 cuobjdump -sass 自行对照 | 需人工 |
| `counters_unknown` | 无法判定计数器是否可用（不等于被拒绝） | 1 条路由会报它（`exact_dram`） | 装 ncu 后重跑 `kernoscope doctor --json` 才能判定计数器是否可用；命令：`kernoscope doctor --json` | 需人工 |
| `counters_not_installed` | 没有 ncu，无法判定计数器是否可用 | 1 条路由会报它（`exact_dram`） | 装 ncu（CUDA Toolkit 的一部分）后重跑 doctor：「没装」不等于「被权限限制」，两者不能混；命令：`kernoscope doctor --json` | 需人工 |
| `instrumentation_missing` | 需要插桩/多趟重放的通道不可得 | 2 条路由会报它（`stall_breakdown`、`pipe_utilization`） | 这一类需要额外的采集通道（插桩/多趟重放）：本工具不重放，用 NVBit 轨迹（--ks-memtrace）或指令直方图（--ks-instr-hist）替代 | ≈10min |
| `not_applicable` | 该指标在本设备上无定义 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 该指标在本设备/本平台上无定义（不是数据缺失）：忽略这一行即可 | ≈0 |
| `mixed_launch_config` | 同名 kernel 有多组 launch 配置，单值会误导 | 1 条路由会报它（`launch_configs`） | 同名 kernel 有多组 launch 配置时我们不报单一数值（会误导多数 launch）：逐组读数见 LaunchConfigs section；只关心一组时用 -c/-k 固定它 | ≈0 |
| `no_clock_reading` | 没有 SM 时钟读数 | 1 条路由会报它（`issue_slot`） | 加 --ks-clock-mhz <MHz>（或让 NVML 采样开着）：没有 SM 时钟就算不出发射槽上限；命令：`kernoscope --ks-clock-mhz 2520 ...` | ≈0 |
| `no_launch_activity` | 没有采到任何 kernel 活动 | 1 条路由会报它（`kernel_time`） | 确认被测命令真的在 GPU 上跑 kernel，且采集时带上了 CUDA activity（--ks-sqlite 指向的库要有 CUPTI_ACTIVITY_KIND_KERNEL 表）；命令：`nsys profile --trace=cuda -o run -- python app.py` | ≈30s |
| `no_kernel_match` | 过滤后没有匹配的 kernel | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 放宽 -k 过滤（用更短的正则片段），或先不加 -k 看有哪些 kernel 名字；命令：`kernoscope --ks-sqlite kernels.sqlite --launch-csv` | ≈5s |
| `no_kernels` | 报告里没有 kernel | 2 条路由会报它（`kernel_time`、`launch_configs`） | 先确认采集到了 kernel 活动（见 no_launch_activity 的补救办法） | ≈30s |
| `multi_kernel_window` | 窗口里不止一个 kernel，per-launch 不可归因 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 用 -k 只选一个 kernel，让窗口里只有它；否则只能给整窗口总量；命令：`kernoscope -k regex:<kernel> --set detailed -- python app.py` | ≈0 |
| `coverage_too_low` | 采样窗口没被被测负载主导 | 1 条路由会报它（`dram_bytes`） | 把稳态循环跑长（建议 ≥20 秒），或加 --ks-window auto 只统计稳态段；覆盖率不足时窗口里大部分时间是空闲，忙碌时段是否饱和无从确认；命令：`kernoscope --ks-window auto -- python bench_loop.py` | ≈20s |
| `share_too_low` | 被匹配的 kernel 不是窗口主角 | 1 条路由会报它（`dram_bytes`） | 份额按整段运行的 launch 表算，缩采样窗口改不动它。三条有效路径：① 让被测区间里只有这一个 kernel（改 bench 脚本单独跑它）；② 改走整窗口口径 --ks-ledger-total-mb（不要求归因）；③ 加 --ks-memtrace 换轨迹口径（只测被追踪的那个 kernel，不受窗口影响） | ≈20s |
| `share_unknown` | 逐 launch 时长为空，份额无法判定 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 逐 launch 时长为空：确认 nsys 的 CUPTI kernel 表可用（kernel 时长是 exact 的，通常不缺）；缺时用 --ks-sqlite 直接指定库；命令：`kernoscope --ks-sqlite kernels.sqlite` | ≈5s |
| `window_launch_unaligned` | 窗口内的 launch 数没法和采样窗口对齐 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | per-launch 要求分子（窗口内字节）与分母（窗口内 launch 数）覆盖**同一个窗口**；窗口被裁过、而 launch 时间戳与采样时钟又对不上时，分母不可信，所以不给这个数。把窗口放开（--ks-window full；代价是启动/JIT 会稀释覆盖率，可能被 coverage_too_low 挡），或改走整窗口口径 --ks-ledger-total-mb / 轨迹口径 --ks-memtrace —— 后两条不需要这个分母；命令：`kernoscope --ks-window full -- python bench_loop.py` | ≈20s |
| `below_resolution` | 读数低于仪器分辨率下限 | 2 条路由会报它（`cost_per_token`、`dram_bytes`） | 让 GPU 保持饱和：NVML 显存忙碌是整数百分比，低于 1 个百分点的读数会被截断成 0；把测量窗口对准真正在搬数据的时段（--ks-window auto / --ks-warmup）；命令：`kernoscope --ks-window auto -- python bench_loop.py` | ≈20s |
| `mem_busy_zero` | 显存忙碌恒为 0 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 显存忙碌恒为 0：确认被测负载真的在搬显存（而不是 L2 命中），且采样周期（200 ms）没有整段错过忙碌期 | ≈20s |
| `trace_unavailable` | 访存轨迹给不出字节数 | 1 条路由会报它（`dram_bytes_trace`） | 轨迹口径给不出字节时，改用 NVML 代理口径（--ks-calibration + --ks-window auto）或检查轨迹是否覆盖了被匹配的 kernel | ≈20s |
| `trace_too_many_accesses` | 访问数超过分析上限 | 1 条路由会报它（`dram_bytes_trace`） | 提高 --ks-max-accesses（内存换精度），或缩小被追踪 kernel 的迭代数：超上限时我们拒绝给一个偏小的字节数；命令：`kernoscope --ks-max-accesses 20000000 --ks-memtrace t.txt ...` | 需人工 |
| `ledger_basis_unavailable` | 账本对拍口径不成立 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 换对拍口径：--ks-ledger-total-mb 走整窗口，--ks-ledger-mb 走 per-launch（后者要求窗口主导性达标）；命令：`kernoscope --ks-ledger-total-mb 512 -- python bench_loop.py` | ≈0 |
| `zero_baseline` | 基线为 0，相对误差带不适用 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 基线是 0 时相对差无定义（除以 0），所以不能用相对误差带判「变了多少」。要么给出一个非零基线，要么按绝对差值自行判断 —— 我们只把绝对差值报出来，不下方向 | 需人工 |
| `no_calibration` | 缺少本机实测峰值（未标定） | 2 条路由会报它（`dram_bytes`、`roofline`） | 跑一次本机标定并把结果传进来（roofline 与 DRAM 代理都依赖它）；命令：`python tools/machine-profile.py --out machine-profile.json` | ≈5min |
| `calibration_caliber_mismatch` | 标定口径与本次测量不可比 | 1 条路由会报它（`roofline`） | 标定口径与本次测量不可比（例如 fp32 用 reduced-precision reduction 跑出来的峰值）：重跑标定并让口径一致；命令：`python tools/machine-profile.py --quick` | ≈5min |
| `no_workload` | 没有给工作量（FLOPs/bytes） | 1 条路由会报它（`roofline`） | 给出每次 launch 的工作量：--ks-gemm M,N,K 或 --ks-flops/--ks-bytes；命令：`kernoscope --ks-gemm 4096,4096,4096 -- python app.py` | ≈0 |
| `no_tokens` | 没有给每次迭代的 token/样本数：算不出单位成本 | 1 条路由会报它（`cost_per_token`） | 给出每次迭代（step）处理的 token 数：推理填一次 forward 的 token 数，训练填一步的 token 数（packing 后的实际数）—— 有了它才有 tokens/s 与单位成本；命令：`kernoscope --ks-tokens 8192 -- python app.py` | ≈0 |
| `no_price` | 没有给电价或租用价：算不出钱 | 1 条路由会报它（`cost_per_token`） | 给出价格：--ks-price-per-kwh（电费口径）或 --ks-price-per-hour（整机租约口径），两者都给就相加；币种随你给的价格，不做汇率换算；命令：`kernoscope --ks-tokens 8192 --ks-price-per-hour 2.5 -- python app.py` | ≈0 |
| `no_energy` | 拿不到窗口能耗读数（NVML 能量/功率不可用） | 1 条路由会报它（`cost_per_token`） | 打开 NVML 采样（别加 --ks-no-nvml）并让稳态段长于采样周期；拿不到功率读数时能耗口径不可用，我们不按 TDP 顶替；命令：`kernoscope --set basic -o rep -- python bench_loop.py` | ≈0 |
| `ncu_report_unreadable` | 读不了这份 .ncu-rep（缺 Nsight Compute 的 Python 模块，或文件损坏） | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 读 .ncu-rep 只需要 Nsight Compute 的 Python 模块（不需要 GPU、不需要计数器权限）：把 `<nsight-compute>/extras/python` 加进 PYTHONPATH；若文件本身损坏，重新导出或重采；命令：`PYTHONPATH=<nsight-compute>/extras/python kernoscope --ks-ncu-rep report.ncu-rep --set basic` | 需人工 |
| `unknown_arch` | 未知架构，算不出上界 | 2 条路由会报它（`theoretical_occupancy`、`achieved_occupancy`） | 用 --ks-sm-count 指定 SM 数，或报告里带上 compute capability：未知架构算不出 occupancy 上界，我们不猜 | 需人工 |
| `no_selftest_band` | 没有本机误差带与 MDD | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 跑一次 selftest 得到本机误差带与最小可分辨差异（MDD）；没有它就无法判断两次运行的差别是真差别还是噪声；命令：`kernoscope selftest --out machine-selfcheck.json` | ≈2min |
| `mdd_unknown` | 误差带里没有这个指标的 MDD | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 误差带里没有这个指标（MDD 按指标族给）：用 --metric 指定有 MDD 的指标，或重跑 selftest 覆盖该负载族；命令：`kernoscope selftest --out machine-selfcheck.json` | ≈2min |
| `protocol_mismatch` | 两次运行的协议不同，不可比 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 两次比较必须同协议：同标定文件、同窗口模式、同工具版本、同 kernel 选择；协议不同时我们拒绝给方向判词（宁可说不可比） | ≈0 |
| `both_refused` | 两侧都被同一个前提挡住，没有数字可比 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 两侧都被同一种原因拒绝（例如都没有标定）：先按该原因的补救办法解除前提，这次比较才有意义 —— 没有数字时 diff 不下任何结论；命令：`kernoscope capabilities --json` | 需人工 |
| `premise_failed` | 判词的前提之一不成立（数字不可引用） | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 先让前提成立（按该前提对应的 code 处理）：前提不成立时那个数字不能用，把它当结论会得到一个方向性错误；命令：`kernoscope capabilities --json` | 需人工 |
| `metric_absent` | 请求的指标不存在 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 用 --json 看本次报告里实际存在哪些指标，或用 `kernoscope capabilities --json` 看哪些问题能答（不存在 ≠ 被拒）；命令：`kernoscope capabilities --json` | ≈0 |
| `legacy_report` | 旧报告没有结构化拒绝信息 | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 报告是结构化拒绝之前生成的（没有 code 字段）：重跑一次采集即可带上；本工具按 legacy 标注，不猜原因 | ≈5s |
| `internal_unknown` | 未分类的拒绝（属于本工具缺陷） | ——（`route_cmd.FAILURE_HINTS` 里没有登记会报出它的路由） | 未分类的拒绝 —— 这属于本工具的缺陷，请按 issue 模板报告 | 需人工 |

## 永久拿不到（本例无法补救）

下面 3 个 code 的 remedy 声明 `cost_kind=unavailable`：**这不是「环境缺件」，而是本机没有这条合法路径**。它们与上面的表分开，是因为处置完全不同 ——
缺件可以去装/去开；这里只能换成上表里的替代口径，或换一台有权限的机器。

| code | 人话（CODE_TITLES） | 观测到什么（阈值/条件） | 怎么办（remedy.action + command） | 代价 |
|---|---|---|---|---|
| `counters_denied` | 硬件计数器被权限/驱动策略拒绝 | 4 条路由会报它（`exact_dram`、`stall_breakdown`、`pipe_utilization` 等 4 条） | 计数器是 host-wide 的，容器内无法突破（不要用 sudo/setuid 试）；换用本工具的非计数器口径，或在有权限的机器上跑 ncu | 本机不可行 |
| `sampling_unavailable` | PC 采样（PmSampling/SourceCounters）不可得 | 2 条路由会报它（`l2_locality`、`stall_breakdown`） | PC 采样（PmSampling/SourceCounters）与计数器同源、同样被权限锁死；需要 PC 级归因时用 --ks-instr-hist 的发射槽上界替代（方向性结论，非周期细分）；命令：`kernoscope --ks-instr-hist opcode_hist.txt ...` | 本机不可行 |
| `no_nvlink` | 没有 NVLink 硬件/NVML 支持 | 1 条路由会报它（`nvlink`） | 需要 NVLink 硬件与 NVML 支持；单卡实例上这个 section 本来就不适用 | 本机不可行 |

### 同一件事在 ncu 报告里的样子（`sections.COUNTER_SECTIONS`）

从别人的 ncu 报告里读到时，这些 section 会以 `n/a` 出现，对应的 code 如下（10 个 section，与 `sections.py` 同源）：

| ncu section | code | 原因（原样引用代码里的字符串） |
|---|---|---|
| `ComputeWorkloadAnalysis` | `counters_denied` | requires GPU hardware performance counters, which are restricted here (RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN/CAP_PERFMON in the container); CUPTI reports ERR_NVGPUCTRPERM |
| `MemoryWorkloadAnalysis` | `counters_denied` | requires GPU hardware performance counters, which are restricted here (RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN/CAP_PERFMON in the container); CUPTI reports ERR_NVGPUCTRPERM |
| `MemoryWorkloadAnalysis_Tables` | `counters_denied` | requires GPU hardware performance counters, which are restricted here (RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN/CAP_PERFMON in the container); CUPTI reports ERR_NVGPUCTRPERM |
| `SchedulerStats` | `counters_denied` | requires GPU hardware performance counters, which are restricted here (RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN/CAP_PERFMON in the container); CUPTI reports ERR_NVGPUCTRPERM |
| `WarpStateStats` | `counters_denied` | requires GPU hardware performance counters, which are restricted here (RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN/CAP_PERFMON in the container); CUPTI reports ERR_NVGPUCTRPERM |
| `SourceCounters` | `sampling_unavailable` | requires PC sampling, which needs the profiling permission here |
| `PmSampling` | `sampling_unavailable` | requires GPU hardware performance counters, which are restricted here (RmProfilingAdminOnly=1 and no CAP_SYS_ADMIN/CAP_PERFMON in the container); CUPTI reports ERR_NVGPUCTRPERM |
| `Nvlink_Tables` | `no_nvlink` | requires NVLink hardware with NVML counter support (not present / not supported) |
| `NumaAffinity` | `not_applicable` | not applicable on this device |
| `WorkloadDistribution` | `instrumentation_missing` | requires in-kernel instrumentation (NVBit / IKP trace) which is not enabled; install NVBit and pass an existing trace with --ks-memtrace <file> (see tools/probes/nvbit_memtrace/collect.sh) |

## 二、任务 → 该跑哪条检查

**规则（先看这条，再看表）**：最简单的问题**直接跑命令**，不要先跑诊断 —— 命令自己会在缺件时报出拒绝码，多跑一次 `doctor` 只是白等。只有层 ≥3 的（要 NVBit 插桩或计数器权限，装齐/开权限不是一条命令能解决的事）才值得先花一次 `kernoscope capabilities --json` 确认前提。

表由 `capabilities.QUESTIONS`（问题与层）+ `route_cmd.COMMANDS`（命令模板）+ `route_cmd.FAILURE_HINTS`（会被拒的 code）生成，17 条问题全覆盖。

| 任务（问题） | 问题 id | 命令（COMMANDS） | 先跑诊断还是直接跑 | 可能被拒的 code |
|---|---|---|---|---|
| 哪个 kernel 最耗时、每次跑多久 | `kernel_time` | `kernoscope --set basic -k regex:<kernel> -c 20 -o rep -- <你的命令>` | **直接跑**（层 2 的数据源在；缺件时命令会报 code） | `no_kernels`、`no_launch_activity`、`nsys_missing` |
| 同名 kernel 有几组 launch 配置，各自多快 | `launch_configs` | `kernoscope --set basic -c 50 --launch-csv -o rep -- <你的命令>` | **直接跑**（层 2 的数据源在；缺件时命令会报 code） | `mixed_launch_config`、`no_kernels` |
| 静态寄存器/共享内存，以及谁是 occupancy 限制因素 | `static_resources` | `kernoscope --set basic --ks-cubin <kernel.cubin> -o rep（不需要 GPU）` | **直接跑**（层 2 的数据源在；缺件时命令会报 code） | `no_static_input`、`cuobjdump_missing` |
| occupancy 理论上界是多少 | `theoretical_occupancy` | `kernoscope occupancy --from <occupancy-records.txt> --json` | **直接跑**（层 2 的数据源在；缺件时命令会报 code） | `unknown_arch`、`no_static_input` |
| 这段时间 GPU 显存子系统有多忙（实测） | `device_busy` | `kernoscope --set basic -o rep -- <你的命令>（只要有 NVML）` | **直接跑**（层 1 的数据源在；缺件时命令会报 code） | `no_nvml`、`no_nvml_samples`、`no_gpu` |
| 这次推理/训练的单位成本（每百万 token 多少钱 / 每小时多少钱） | `cost_per_token` | `kernoscope --set basic --ks-tokens <每步 token 数> --ks-price-per-kwh <元/kWh> --ks-price-per-hour <元/小时> -o rep -- <你的命令>（价格给哪个就算哪个载体）` | **直接跑**（层 1 的数据源在；缺件时命令会报 code） | `no_energy`、`no_tokens`、`no_price`、`below_resolution` |
| DRAM 搬了多少字节 / 是不是 DRAM bound（零依赖的 NVML 代理口径；有 NVBit 时首选轨迹口径，精度高 4 个数量级） | `dram_bytes` | `kernoscope --set basic --ks-calibration machine-profile.json --ks-ledger-mb <每次 launch 的 MB> --ks-window auto -o rep -- <你的命令>` | **直接跑**（层 1 的数据源在；缺件时命令会报 code） | `share_too_low`、`coverage_too_low`、`below_resolution`、`no_calibration` |
| 离 roofline 多远、compute 还是 memory bound | `roofline` | `kernoscope --set basic --ks-gemm 4096,4096,4096 --ks-dtype bf16 --ks-calibration machine-profile.json -o rep -- <你的命令>` | **直接跑**（层 1 的数据源在；缺件时命令会报 code） | `no_workload`、`no_calibration`、`calibration_caliber_mismatch` |
| 实际达到的 occupancy（不是上界） | `achieved_occupancy` | `kernoscope --set basic --ks-achieved-occupancy <occ.json> -o rep -- <你的命令>` | 先 `kernoscope capabilities --json`（层 3 需要 NVBit 自行编译（+ nvcc）） | `no_occupancy_probe`、`unknown_arch` |
| DRAM 字节的**首选**口径（访存轨迹：地址是算术，只有缓存替换是模型） | `dram_bytes_trace` | `kernoscope memtrace --from <trace.txt> --l2-bytes <L2字节> --json` | 先 `kernoscope capabilities --json`（层 3 需要 NVBit 自行编译（+ nvcc）） | `no_memtrace`、`nvbit_missing`、`trace_unavailable`、`trace_too_many_accesses` |
| 发射槽用了多少、瓶颈方向是 latency 还是 throughput | `issue_slot` | `kernoscope issue --from <opcode_hist.txt> --sm-count <N> --clock-mhz <MHz> --duration-ms <ms>` | 先 `kernoscope capabilities --json`（层 3 需要 NVBit 自行编译（+ nvcc）） | `no_instr_hist`、`nvbit_missing`、`no_clock_reading` |
| L1/L2 命中率与工作集是否驻留 | `l2_locality` | `kernoscope --set basic --ks-memtrace <trace.txt> --l2-bytes <L2字节> -o rep -- <你的命令>` | 先 `kernoscope capabilities --json`（层 3 需要 NVBit 自行编译（+ nvcc）） | `no_memtrace`、`sampling_unavailable` |
| 静态 SASS 指令混合（哪些指令占多数） | `static_mix` | `kernoscope --set basic --ks-cubin <kernel.cubin> -o rep（不需要 GPU）` | **直接跑**（层 2 的数据源在；缺件时命令会报 code） | `no_sass_source`、`no_static_input` |
| warp 到底在等什么（stall 周期细分） | `stall_breakdown` | `kernoscope issue --from <opcode_hist.txt> --sm-count <N> --clock-mhz <MHz> --duration-ms <ms>（只有上界+方向，stall 周期细分原理上拿不到）` | 先 `kernoscope capabilities --json`（层 4 需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0）） | `counters_denied`、`instrumentation_missing`、`sampling_unavailable` |
| DRAM 字节/L2 命中率的精确值（与 ncu 同源） | `exact_dram` | `kernoscope doctor --json（先确认是「没装 ncu」还是「被权限拒绝」——两者处置不同）` | 先 `kernoscope capabilities --json`（层 4 需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0）） | `counters_denied`、`counters_not_installed`、`counters_unknown` |
| 各 pipe（FP32/INT/Tensor）的动态利用率 | `pipe_utilization` | `kernoscope issue --from <opcode_hist.txt> --sm-count <N> --clock-mhz <MHz> --duration-ms <ms>（每 pipe 的**可计算下界**，不是计数器）` | 先 `kernoscope capabilities --json`（层 4 需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0）） | `counters_denied`、`instrumentation_missing` |
| NVLink 的流量与利用率 | `nvlink` | `kernoscope doctor --json（NVLink 计数器的可用性随驱动/权限变化）` | 先 `kernoscope capabilities --json`（层 4 需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0）） | `no_nvlink`、`counters_denied` |

## 三、缺哪一趟采集，就永久补不了

有些测量**不是「分析得不够」，而是采集当时没打开采集通道**：已有的产物里根本不含这份数据，事后只能重跑负载（而且这台机器上多半跑不成）。下表由两条**可从代码复算**的规则挑出来，没有第三条：

* 规则 A：`remedy.cost_kind == "unavailable"` —— 本机没有任何合法补救路径。
* 规则 B：该 code 关联到**层 ≥3** 的路由（NVBit 插桩 / 计数器权限），且 remedy 声明要再等 ≥60 秒补一趟测量（`cost_kind=wall_seconds`、`cost_s >= 60`）。

落在规则外的**不列进来**（例如事后补一个 `--ks-clock-mhz` 就能算的，不属于「缺一趟采集」）。所以这张表是**下界**，不是「补不回来的完整清单」 —— 清单的完整性受限于代码里有没有把「需要哪趟采集」写成数据。

| code（谁在告诉你） | 缺的是哪一趟采集 | 事后为什么补不了（remedy.action） | 代价 |
|---|---|---|---|
| `counters_denied`（硬件计数器被权限/驱动策略拒绝） | ——（不是缺一趟采集：`cost_kind=unavailable`，本机没有这条合法路径）；层 4 `counters`：需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0） | 计数器是 host-wide 的，容器内无法突破（不要用 sudo/setuid 试）；换用本工具的非计数器口径，或在有权限的机器上跑 ncu | 本机不可行 |
| `sampling_unavailable`（PC 采样（PmSampling/SourceCounters）不可得） | ——（不是缺一趟采集：`cost_kind=unavailable`，本机没有这条合法路径）；层 4 `counters`：需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0） | PC 采样（PmSampling/SourceCounters）与计数器同源、同样被权限锁死；需要 PC 级归因时用 --ks-instr-hist 的发射槽上界替代（方向性结论，非周期细分） | 本机不可行 |
| `no_nvlink`（没有 NVLink 硬件/NVML 支持） | ——（不是缺一趟采集：`cost_kind=unavailable`，本机没有这条合法路径）；层 4 `counters`：需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0） | 需要 NVLink 硬件与 NVML 支持；单卡实例上这个 section 本来就不适用 | 本机不可行 |
| `no_memtrace`（没有提供访存轨迹文件） | 层 3 `instrumented`：需要 NVBit 自行编译（+ nvcc） | 加 --ks-memtrace <trace>：轨迹口径只测被追踪的 kernel，不受窗口构成影响 | ≈10min |
| `no_instr_hist`（没有提供指令直方图） | 层 3 `instrumented`：需要 NVBit 自行编译（+ nvcc） | 加 --ks-instr-hist <opcode_hist.txt>：给出动态指令数与发射槽利用率上界 | ≈10min |
| `no_occupancy_probe`（没跑 occupancy 探针） | 层 3 `instrumented`：需要 NVBit 自行编译（+ nvcc） | 跑 occupancy 探针并把 JSON 用 --ks-achieved-occupancy 传进来（需要 NVBit） | ≈10min |
| `instrumentation_missing`（需要插桩/多趟重放的通道不可得） | 层 4 `counters`：需要 ncu 可用权限（CAP_SYS_ADMIN / RmProfilingAdminOnly=0） | 这一类需要额外的采集通道（插桩/多趟重放）：本工具不重放，用 NVBit 轨迹（--ks-memtrace）或指令直方图（--ks-instr-hist）替代 | ≈10min |

## 怎么用这份表

* 本文件是**生成物**：真源是 `src/kernoscope/refusals.py`（code / 人话 / remedy / 代价）、`route_cmd.py`（路由、命令模板、会被拒的 code）、`capabilities.py`（问题与能力层）、`sections.py`（ncu section → code）。这里的每一句散文都只是**它们的视图**，不是另一个真源。
* 改了上面任何一份代码，重跑 `uv run python tools/gen_playbook.py`；`uv run python tools/dev_index.py --gates` 里的 `playbook` 门禁会跑 `--check`，不一致就红（带 diff）。
* 只想知道**一条 code** 怎么办：看第一节里那一行，或 `scripts/remedy.sh <code>`；不要因为一条 `n/a` 通读全文。
* 本机实际能答什么、缺什么，现场跑 `kernoscope capabilities --json`；不知道怎么走时先 `kernoscope route "<你的问题>"`。
