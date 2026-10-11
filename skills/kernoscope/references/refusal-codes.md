<!-- 由 tools/gen_agent_refs.py 生成；改动请改代码后重跑，不要手改本文件 -->

# 拒绝码（`refusal.code`）参考

每条 `n/a` 都带一个**闭集 code**（判决在 code 里，散文只解释）；56 个 code 里 37 个带可直接执行的命令、19 个只给动作（多为本机走不通的路，故意不给命令免得让人白试）。观测值与门槛只有在**该前提本身可观测**时才给。
**硬规则：`status=refused` 的判词不得作为行动依据**；先按 remedy 解除前提，
或换一条口径，再下结论。改 code / 加 code 请改 `src/kernoscope/refusals.py`。

共 56 个 code（本文件由代码生成，与 `DEFAULT_REMEDIES` 同源）。

| code | 什么意思 | 怎么办 | 命令 | 代价 |
|---|---|---|---|---|
| `no_gpu` | 本机没有 NVIDIA GPU | 换一台有 NVIDIA GPU 的机器；或在没有 GPU 时用 --ks-synthetic 跑通流程看输出形状 | `kernoscope --ks-synthetic --set basic` | 需人工 |
| `pytorch_missing` | 没有安装 PyTorch | 装 PyTorch（自检的已知答案负载只需要 torch） | `pip install torch` | 需人工 |
| `no_nvml` | nvidia-smi / pynvml 不可用 | 装 pynvml（pip install 'kernoscope[nvml]'）或确认 nvidia-smi 可用 | `pip install 'kernoscope[nvml]'` | 需人工 |
| `no_nvml_samples` | 没有 NVML 采样点（关掉采样／不可用／稳态段短于 200 ms） | 去掉 --ks-no-nvml 让设备级采样生效（代价接近 0：采样在后台线程里做） |  | ≈0 |
| `nsys_missing` | 没有 nsys：kernel 级数据不可得 | 安装 Nsight Systems（随 CUDA Toolkit 提供），或提供已有的 .nsys-rep/.sqlite | `kernoscope --ks-sqlite <kernels.sqlite>` | 需人工 |
| `cuobjdump_missing` | 没有 cuobjdump：静态资源/SASS 不可得 | 安装 CUDA Toolkit（cuobjdump 随 toolkit 提供），或用已有报告里的静态字段 |  | 需人工 |
| `no_static_input` | 没有提供 cubin/.so | 用 --ks-cubin <cubin/.so> 指向被测二进制；没有它就没有寄存器/共享内存/SASS | `kernoscope --ks-cubin build/app.so ...` | 需人工 |
| `nvbit_missing` | 未配置 NVBit：指令级/访存轨迹不可得 | 自行安装 NVBit（许可证禁止本项目随包分发）并设 NVBIT_PATH |  | 需人工 |
| `no_nvtx` | 没有 NVTX 区间（采集时没带 --trace=nvtx） | 重采时带 `--trace=nvtx`（`nsys profile --trace=cuda,nvtx -o run -- <你的命令>`）：NVTX 区间是采集当时写进 trace 的标注，事后补不了 —— 已有的库里即使有完整的 kernel 时间，也没有「哪一段」这个维度 | `nsys profile --trace=cuda,nvtx -o run -- python app.py` | ≈60s |
| `no_memtrace` | 没有提供访存轨迹文件 | 加 --ks-memtrace <trace>：轨迹口径只测被追踪的 kernel，不受窗口构成影响 | `kernoscope --ks-memtrace ks-memtrace.txt ...` | ≈10min |
| `no_instr_hist` | 没有提供指令直方图 | 加 --ks-instr-hist <opcode_hist.txt>：给出动态指令数与发射槽利用率上界 | `kernoscope --ks-instr-hist opcode_hist.txt ...` | ≈10min |
| `no_occupancy_probe` | 没跑 occupancy 探针 | 跑 occupancy 探针并把 JSON 用 --ks-achieved-occupancy 传进来（需要 NVBit） | `kernoscope occupancy --from <probe-records> --write-json occ.json` | ≈10min |
| `no_sass_source` | SASS/源码逐行对照未实现 | SASS/源码逐行对照需要 --import-source，本项目不实现：改用 cuobjdump -sass 自行对照 |  | 需人工 |
| `counters_denied` | 硬件计数器被权限/驱动策略拒绝 | 计数器是 host-wide 的，容器内无法突破（不要用 sudo/setuid 试）；换用本工具的非计数器口径，或在有权限的机器上跑 ncu |  | 本机不可行 |
| `counters_unknown` | 无法判定计数器是否可用（不等于被拒绝） | 装 ncu 后重跑 `kernoscope doctor --json` 才能判定计数器是否可用 | `kernoscope doctor --json` | 需人工 |
| `counters_not_installed` | 没有 ncu，无法判定计数器是否可用 | 装 ncu（CUDA Toolkit 的一部分）后重跑 doctor：「没装」不等于「被权限限制」，两者不能混 | `kernoscope doctor --json` | 需人工 |
| `sampling_unavailable` | PC 采样（PmSampling/SourceCounters）不可得 | PC 采样（PmSampling/SourceCounters）与计数器同源、同样被权限锁死；需要 PC 级归因时用 --ks-instr-hist 的发射槽上界替代（方向性结论，非周期细分） | `kernoscope --ks-instr-hist opcode_hist.txt ...` | 本机不可行 |
| `instrumentation_missing` | 需要插桩/多趟重放的通道不可得 | 这一类需要额外的采集通道（插桩/多趟重放）：本工具不重放，用 NVBit 轨迹（--ks-memtrace）或指令直方图（--ks-instr-hist）替代 |  | ≈10min |
| `no_nvlink` | 没有 NVLink 硬件/NVML 支持 | 需要 NVLink 硬件与 NVML 支持；单卡实例上这个 section 本来就不适用 |  | 本机不可行 |
| `not_applicable` | 该指标在本设备上无定义 | 该指标在本设备/本平台上无定义（不是数据缺失）：忽略这一行即可 |  | ≈0 |
| `mixed_launch_config` | 同名 kernel 有多组 launch 配置，单值会误导 | 同名 kernel 有多组 launch 配置时我们不报单一数值（会误导多数 launch）：逐组读数见 LaunchConfigs section；只关心一组时用 -c/-k 固定它 |  | ≈0 |
| `no_clock_reading` | 没有 SM 时钟读数 | 加 --ks-clock-mhz <MHz>（或让 NVML 采样开着）：没有 SM 时钟就算不出发射槽上限 | `kernoscope --ks-clock-mhz 2520 ...` | ≈0 |
| `env_shared_gpu` | 同卡还有别的计算进程：读数被他人负载稀释 | 等到同卡其他进程结束，或换一张独占的卡再测：多租户下读数含他人负载，比值类口径（每 token 成本、显存带宽达成率）会被系统性稀释；拿不到独占环境时，结论按 env_gpu_exclusive 降级为不可引用 |  | 需人工 |
| `env_throttled` | 窗口内有降频（功率封顶/热降频）：能效读数被扰动 | 等降频结束再测（温度回落/功率回到上限内），或提高功耗上限/换散热更好的实例；降频期间的时间与能耗数字仍然真实，但把它当作能效结论会偏乐观（同一份工作花了更多电） | `nvidia-smi -q -d PERFORMANCE` | ≈30s |
| `env_reading_unknown` | 环境读数读不到（未知）：不能当成"没有" | 环境读数读不到时我们不猜：确认 NVML/nvidia-smi 可用（`kernoscope doctor`）、驱动在采样期间没有重启，或换一个能读到管理计数器的实例；"未知"与"零"是两件事，结论按未知降级 | `kernoscope doctor` | ≈5s |
| `clock_not_locked` | 时钟没被锁住（或锁频状态读不到）：跨运行不可比 | 把两次运行放在**同一个时钟状态**下再比：锁住 SM 时钟（`nvidia-smi -lgc`），跑完用 `nvidia-smi -rgc` 复位；或者两次都在同样的未锁状态下测，并把这条当作已知扰动。**我们不会替你锁**：改设备时钟需要 root、且会改设备状态 —— 那是写操作，不在只读工具的授权里（见 NON-GOALS「不改设备状态」）。 | `sudo nvidia-smi -lgc <min>,<max>    # 跑完：sudo nvidia-smi -rgc` | ≈5s |
| `no_launch_activity` | 没有采到任何 kernel 活动 | 确认被测命令真的在 GPU 上跑 kernel，且采集时带上了 CUDA activity（--ks-sqlite 指向的库要有 CUPTI_ACTIVITY_KIND_KERNEL 表） | `nsys profile --trace=cuda -o run -- python app.py` | ≈30s |
| `no_kernel_match` | 过滤后没有匹配的 kernel | 放宽 -k 过滤（用更短的正则片段），或先不加 -k 看有哪些 kernel 名字 | `kernoscope --ks-sqlite kernels.sqlite --launch-csv` | ≈5s |
| `no_kernels` | 报告里没有 kernel | 先确认采集到了 kernel 活动（见 no_launch_activity 的补救办法） |  | ≈30s |
| `multi_kernel_window` | 窗口里不止一个 kernel，per-launch 不可归因 | 用 -k 只选一个 kernel，让窗口里只有它；否则只能给整窗口总量 | `kernoscope -k regex:<kernel> --set detailed -- python app.py` | ≈0 |
| `coverage_too_low` | 采样窗口没被被测负载主导 | 把稳态循环跑长（建议 ≥20 秒），或加 --ks-window auto 只统计稳态段；覆盖率不足时窗口里大部分时间是空闲，忙碌时段是否饱和无从确认 | `kernoscope --ks-window auto -- python bench_loop.py` | ≈20s |
| `share_too_low` | 被匹配的 kernel 不是窗口主角 | 份额按整段运行的 launch 表算，缩采样窗口改不动它。三条有效路径：① 让被测区间里只有这一个 kernel（改 bench 脚本单独跑它）；② 改走整窗口口径 --ks-ledger-total-mb（不要求归因）；③ 加 --ks-memtrace 换轨迹口径（只测被追踪的那个 kernel，不受窗口影响） |  | ≈20s |
| `share_unknown` | 逐 launch 时长为空，份额无法判定 | 逐 launch 时长为空：确认 nsys 的 CUPTI kernel 表可用（kernel 时长是 exact 的，通常不缺）；缺时用 --ks-sqlite 直接指定库 | `kernoscope --ks-sqlite kernels.sqlite` | ≈5s |
| `window_launch_unaligned` | 窗口内的 launch 数没法和采样窗口对齐 | per-launch 要求分子（窗口内字节）与分母（窗口内 launch 数）覆盖**同一个窗口**；窗口被裁过、或忙碌段里混了不在 launch 表里的工作（斜率离 1 超出锚点不确定度）时，分母不可信，所以不给这个数。两条不需要这个分母的替代口径：整窗口口径 --ks-ledger-total-mb（对拍总量）或轨迹口径 --ks-memtrace（只测被追踪的 kernel）。要让 per-launch 口径本身成立：让采样窗口覆盖整段运行（别用 --ks-window-s 切得比 launch 表还短）并给出逐 launch 时间戳（--ks-sqlite）——**不是**把窗口切来切去（`--ks-window full` 只是换了窗口，分母口径不变，第五轮 V-32 之前这里恰恰推荐错了方向） | `kernoscope --ks-ledger-total-mb 440000 -- python bench_loop.py` | ≈20s |
| `below_resolution` | 读数低于仪器分辨率下限 | 让 GPU 保持饱和：NVML 显存忙碌是整数百分比，低于 1 个百分点的读数会被截断成 0；把测量窗口对准真正在搬数据的时段（--ks-window auto / --ks-warmup） | `kernoscope --ks-window auto -- python bench_loop.py` | ≈20s |
| `mem_busy_zero` | 显存忙碌恒为 0 | 显存忙碌恒为 0：确认被测负载真的在搬显存（而不是 L2 命中），且采样周期（200 ms）没有整段错过忙碌期 |  | ≈20s |
| `trace_unavailable` | 访存轨迹给不出字节数 | 轨迹口径给不出字节时，改用 NVML 代理口径（--ks-calibration + --ks-window auto）或检查轨迹是否覆盖了被匹配的 kernel |  | ≈20s |
| `trace_too_many_accesses` | 访问数超过分析上限 | 提高 --ks-max-accesses（内存换精度），或缩小被追踪 kernel 的迭代数：超上限时我们拒绝给一个偏小的字节数 | `kernoscope --ks-max-accesses 20000000 --ks-memtrace t.txt ...` | 需人工 |
| `ledger_basis_unavailable` | 账本对拍口径不成立 | 换对拍口径：--ks-ledger-total-mb 走整窗口，--ks-ledger-mb 走 per-launch（后者要求窗口主导性达标） | `kernoscope --ks-ledger-total-mb 512 -- python bench_loop.py` | ≈0 |
| `zero_baseline` | 基线为 0，相对误差带不适用 | 基线是 0 时相对差无定义（除以 0），所以不能用相对误差带判「变了多少」。要么给出一个非零基线，要么按绝对差值自行判断 —— 我们只把绝对差值报出来，不下方向 |  | 需人工 |
| `no_calibration` | 缺少本机实测峰值（未标定） | 跑一次本机标定并把结果传进来（roofline 与 DRAM 代理都依赖它） | `python tools/machine-profile.py --out machine-profile.json` | ≈5min |
| `calibration_caliber_mismatch` | 标定口径与本次测量不可比 | 标定口径与本次测量不可比（例如 fp32 用 reduced-precision reduction 跑出来的峰值）：重跑标定并让口径一致 | `python tools/machine-profile.py --quick` | ≈5min |
| `no_workload` | 没有给工作量（FLOPs/bytes） | 给出每次 launch 的工作量：--ks-gemm M,N,K 或 --ks-flops/--ks-bytes | `kernoscope --ks-gemm 4096,4096,4096 -- python app.py` | ≈0 |
| `no_tokens` | 没有给每次迭代的 token/样本数：算不出单位成本 | 给出每次迭代（step）处理的 token 数：推理填一次 forward 的 token 数，训练填一步的 token 数（packing 后的实际数）—— 有了它才有 tokens/s 与单位成本 | `kernoscope --ks-tokens 8192 -- python app.py` | ≈0 |
| `no_price` | 没有给电价或租用价：算不出钱 | 给出价格：--ks-price-per-kwh（电费口径）或 --ks-price-per-hour（整机租约口径），两者都给就相加；币种随你给的价格，不做汇率换算 | `kernoscope --ks-tokens 8192 --ks-price-per-hour 2.5 -- python app.py` | ≈0 |
| `no_energy` | 拿不到窗口能耗读数（NVML 能量/功率不可用） | 打开 NVML 采样（别加 --ks-no-nvml）并让稳态段长于采样周期；拿不到功率读数时能耗口径不可用，我们不按 TDP 顶替 | `kernoscope --set basic -o rep -- python bench_loop.py` | ≈0 |
| `ncu_report_unreadable` | 读不了这份 .ncu-rep（缺 Nsight Compute 的 Python 模块，或文件损坏） | 读 .ncu-rep 只需要 Nsight Compute 的 Python 模块（不需要 GPU、不需要计数器权限）：把 `<nsight-compute>/extras/python` 加进 PYTHONPATH；若文件本身损坏，重新导出或重采 | `PYTHONPATH=<nsight-compute>/extras/python kernoscope --ks-ncu-rep report.ncu-rep --set basic` | 需人工 |
| `unknown_arch` | 未知架构，算不出上界 | 用 --ks-sm-count 指定 SM 数，或报告里带上 compute capability：未知架构算不出 occupancy 上界，我们不猜 |  | 需人工 |
| `no_selftest_band` | 没有本机误差带与 MDD | 跑一次 selftest 得到本机误差带与最小可分辨差异（MDD）；没有它就无法判断两次运行的差别是真差别还是噪声 | `kernoscope selftest --out machine-selfcheck.json` | ≈2min |
| `mdd_unknown` | 误差带里没有这个指标的 MDD | 误差带里没有这个指标（MDD 按指标族给）：用 --metric 指定有 MDD 的指标，或重跑 selftest 覆盖该负载族 | `kernoscope selftest --out machine-selfcheck.json` | ≈2min |
| `protocol_mismatch` | 两次运行的协议不同，不可比 | 两次比较必须同协议：同标定文件、同窗口模式、同工具版本、同 kernel 选择；协议不同时我们拒绝给方向判词（宁可说不可比） |  | ≈0 |
| `both_refused` | 两侧都被同一个前提挡住，没有数字可比 | 两侧都被同一种原因拒绝（例如都没有标定）：先按该原因的补救办法解除前提，这次比较才有意义 —— 没有数字时 diff 不下任何结论 | `kernoscope capabilities --json` | 需人工 |
| `premise_failed` | 判词的前提之一不成立（数字不可引用） | 先让前提成立（按该前提对应的 code 处理）：前提不成立时那个数字不能用，把它当结论会得到一个方向性错误 | `kernoscope capabilities --json` | 需人工 |
| `metric_absent` | 请求的指标不存在 | 用 --json 看本次报告里实际存在哪些指标，或用 `kernoscope capabilities --json` 看哪些问题能答（不存在 ≠ 被拒） | `kernoscope capabilities --json` | ≈0 |
| `legacy_report` | 旧报告没有结构化拒绝信息 | 报告是结构化拒绝之前生成的（没有 code 字段）：重跑一次采集即可带上；本工具按 legacy 标注，不猜原因 |  | ≈5s |
| `internal_unknown` | 未分类的拒绝（属于本工具缺陷） | 未分类的拒绝 —— 这属于本工具的缺陷，请按 issue 模板报告 |  | 需人工 |

## 常见组合

- 拿到 `no_nvml_samples` / `no_calibration`：**先补测量前提**，不要改代码。
- 拿到 `share_too_low` / `coverage_too_low` / `multi_kernel_window`：
  窗口不成立 —— 缩小窗口、换轨迹口径，或走整窗口口径。
- 拿到 `counters_denied`：与 ncu 同源的那类精确量在这台机器上不可得；
  容器内没有合法绕过，换物理观测量口径。
