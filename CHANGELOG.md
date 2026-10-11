# 变更记录

本文件记录**接口层面**的变化（命令行、决策包 schema、输出字段），不逐条列内部重构。
判断标准：**如果一个外部消费者（脚本、agent、CI）需要改代码或改解析逻辑，就必须写在这里。**

## Unreleased

第五轮验收报告（`study/mimo-rl-study/Kernoscope验收报告-第五轮.md`，基线 `01f650b`）的修复。
**其中若干条会改变外部消费者看到的东西**，逐条列在下面。

### 结论不许静默消失（V-1，P0）

- 预算裁剪的顺序改成"**先裁拒绝散文、最后裁判词**"（旧顺序与函数自己的 docstring 相反：
  真机 9 条判词 → 包 6 条，被删的正是 `occupancy`/`roofline`/`over_peak`，
  而 8 条拒绝的处置散文一条不少）。
- 预算裁掉的判词 id **并入 `verdicts_more.ids`**（`count` 是 `MAX_VERDICTS` 截断与预算截断之和）：
  一个判词要么在包里，要么在 `verdicts_more` 里，没有第三种下场。
  新增 4 条测试钉住（顺序、留痕、`facts` 两级裁剪的分界、`verdicts_more` 计数）。

### `gate` 静默放绿（V-2/V-3，P0）

- `--max-metric`/`--min-metric` 改为**逐 kernel** 判：只要一个真 kernel 带该指标，
  每个真 kernel 都必须带且达标；缺项 / `n/a` / 超限 → 不通过，并**点名 kernel**。
  只有 `(device)` 伪 kernel 带的指标（如 `ks__cost_per_mtok`）按设备级判。
- `--forbid-code` 与 `--require` 的取值限定为闭集（`refusals.py` 的 `RefusalCode` / `ok,soft,refused`）：
  拼错 ⇒ **用法错误 rc=2** 并打印合法取值；不再"拼错就等于没开"。

### 口径正确性（V-5/V-6/V-7/V-8/V-9/V-31）

- **全局原子算全局访存（R-78）**：`RED.E.ADD…`/`ATOMG`/`REDG`/`CAS`（含 `.STRONG.GPU`）
  进入全局口径；`ATOMS`/`REDS`/`.SHARED`/`.LOCAL` 不进。修前真机原子 kernel
  （256 条 `RED.E.ADD.F32.FTZ.RN.STRONG.GPU`）拿到的是「全局记录 0/256、`sectors/request`
  `0.0 [measured]`、DRAM 拒绝理由甩锅"缺 l2_bytes"（而用户给了 `--l2-bytes`）」；
  修后与验收方独立重算逐项一致：**256/256、32.0、12.5%、0.262 MB、262,144 B**。
  同时：`records_global == 0` 时全局口径一族输出 `unavailable` + `code=`（**0 不是测到 0**）；
  拒绝理由逐条列出真正不成立的前提；`memtrace` 的 CLI/JSON/主报告都不再静默跳过拒绝行。
  自带探针 `tools/probes/nvbit_memtrace/memtrace.cu` 的 `is_dram_relevant()` 同步收进全局原子
  （否则纯原子 kernel 采出 0 条记录）。
- **判词绑定到 kernel（V-6）**：`roofline`/`over_peak` 改为按 kernel 取值并在 `facts` 里点名
  kernel —— 一个便宜 kernel 的超峰不再把另一个 kernel 合法的 91% roofline 判成 refused。
- **窗口分母（V-7）**：分母（窗口内等效 launch 数）与"窗口是否被裁"**无关**，始终按时间重叠加权；
  旧实现未裁窗口时直接除以整段 launch 数（采样 20 s / launch 表 40 s ⇒ per-launch 静默偏小 50%、
  `attributable=True`、无告警，而 `window_launch_unaligned` 永不可达）。现在这类情况**拒绝给数**。
- **对齐必须用锚点（V-8）**：launch 表"看起来落在窗口附近"不再被当成同源时钟；有忙碌锚点就用它对齐。
  分母的区间从"锚点 ±1 采样周期 + 锚点推出的斜率容差"给出；若斜率离 1 超出锚点能解释的范围
  （忙碌段里混了不在 launch 表里的工作，例如 memcpy），**拒绝**而不是报一个不成立的窄区间。
- **偏差口径（V-31）**：`window_bias_pct` 的分母改成 **launch 表跨度**（旧实现用进程 wall，
  会报出并不存在的偏差）。
- **`.ncu-rep` 单位换算的安全带（V-9）**：换算系数只在"我们的单位确实是 `from_units` 里那个"时成立；
  单位不符 ⇒ 该行**不写**（并进"因命名空间/单位规则未写入"清单），`convert()` 直接抛
  `UnitMismatch` 兜底。旧实现无条件 ×10⁶，上游哪天改成 ns 就会静默写出 1000× 的数。

### 文档不实与门禁自我满足（V-10…V-13、V-19/V-20/V-21、§4.6）

- **NVBit 版本矩阵**：按真机 A/B 更正为"本环境（CUDA 13 + sm_89 + 驱动 580）唯一实测可用的
  是 **v1.7.7.3 + 老 Makefile**（产物 `tools/mem_trace/`）；按文档 CMake 命令构建的 v1.8.1
  实测失败（rc=1、0 条记录、`CUDA_ERROR_INVALID_SOURCE`）"。`collect.sh` 两种布局都找、
  删掉重复候选、显式指了路径却找不到产物 ⇒ `exit 2` + 实际找过的路径清单；
  版本从实物读（`nvbit_version.sh`）并记进 `environment.txt` 的 `nvbit_version:`。
  门禁改为**结构化矩阵比对**（旧版是"版本号字符串在文档里出现过"，被注释满足）。
- **`--page details` 的口径**：署名散文不再宣称会渲染表格行（真 ncu 只出节标题 + 规则文本）；
  旁路署名里"因命名空间未写入"的例子改成真的会被丢弃的名字（`gpu__local_mem_per_thread` 类），
  不再拿已经写进去的 `gpu__time_duration.sum` 当例子（V-11/V-12）。
- **处方命令（V-13）**：`no_occupancy_probe` 的 `kernoscope occupancy --records … -o occ.json`
  被真解析器拒绝，改为 `--from … --write-json …`；新增 `tests/test_remedy_commands.py`：
  把 `refusals.py` 全部 28 条 kernoscope 处方逐条交给真解析器（+ AST 扫全 `src/` 的
  `Remedy(command=…)` 共 32 条），并重跑受影响的生成物（参考层文档、评测材料、入库产物）。
- **门禁自我满足**：
  - CI 门禁改为解析工作流里的 **`run:` 步骤**（注释里的命令不算）——旧实现整文件子串匹配，
    把真步骤换成 no-op、只在注释里留一句命令照样全绿（V-19）；
  - `bench/artifacts/README.md` 的「现行/历史」标签现在由 `_newest()` 推导并门禁校验
    （有且仅有最新产物是"现行"）；顺带修掉两份产物标签对调的错（V-20）；
  - `by_namespace` 分支补了正向测试：把 `select_metrics` 的过滤去掉 ⇒ 测试必红（V-21）。
- **入口层预算（§4.6 真机专属失败）**：`capabilities --brief` 在有 GPU 的机器上是 1301 B >
  1300 的门禁（CI 无 CUDA 永远看不到）。现在**由代码执行预算**：超了就按有序步骤裁剪
  （命令字符串 → 不可行依赖 → 弱项 → 列表尾部），并把步骤名记进 `trimmed`。
  另修 `tests/test_toolchain.py` 的三条"只在工具不在 PATH 的机器上才绿"的用例
  （显式屏蔽 PATH 命中，并新增一条钉住文档契约"PATH 优先"）。
- 顺修脚本层 `$VAR（中文括号` 展开（非 UTF-8 locale 下 `unbound variable`）：
  `collect.sh` / `bench/run.sh` / `bench/degradation-check.sh` / `remedy.sh`。

### P2 收尾（第五轮 V-4、R-75/R-76）

- **per-kernel 报告不再借用别的 kernel 的读数（V-4）**：`report_md._metric_index` 此前从
  **所有** kernel 兜底取数，于是主体 `synthetic_gemm` 的报告里会出现 `synthetic_elementwise`
  的 `ks__over_peak_warning=1`（16361.8 TFLOPS = 峰值 9779.9%），读者会去查错 kernel。
  现在本 kernel 优先、缺失时**只允许 `(device)` 兜底**（NVML/窗口级读数是"这次测量"的属性）。
- **`ks-memtrace v1` 轨迹补齐指纹与截断判定（R-75/R-76）**：v1 路径此前只认工具自己写的
  `# truncated 1` —— 手动重定向/被 `ulimit` 截断的轨迹照样给数，且不写指纹。
  现在与官方路径同一纪律：内容指纹进报告与协议哈希；**结尾不是换行 ⇒ `truncated=1` ⇒
  拒绝给 DRAM 字节**；只有显式结束标记（`KS-TRACE-COMPLETE` 或 `# complete 1`）才算完整。
  真机复核：截断的 v1 轨迹现在给 `n/a [unavailable] code=trace_unavailable` + 截断原因。

### P2 二批（第五轮 V-18、V-22…V-30、R-60/R-62/R-68）

- **`--plan` 的声明必须等于磁盘实得（V-18）**：`effects.OPERATIONS` 9 → 14，新增
  `write_report_md` / `write_ncu_rep`（含 `.ATTRIBUTION.txt`）/ `write_ncu_truth` /
  `write_name_map` / `write_log_file`；`write_outputs` 从一句散文改成**逐条具体路径**
  （案件目录逐文件、`--launch-csv`、perfetto、`--packet`）。旧声明 5 项 vs 实跑 14 个文件。
  顺带修掉两处原有不实声明（nsys 中间产物的落点与保留语义、`--ks-nsys-rep` 的 sqlite 位置）。
- **不再有裸 traceback（V-25）**：新增 `CliInputError` + 一个**窄**的顶层兜底
  （`OSError`/`JSONDecodeError`/`sqlite3.DatabaseError` → `error: <人话>（<怎么修>）` + rc=1，
  保留异常类型名）；点名接线：只读目录写 `-o`/`--packet`/`--report-md`/`--export-ncu-rep`/
  `--export-name-map`/`--log-file`、`--ks-calibration`/`-i`/`--ks-achieved-occupancy` 的不存在或坏
  JSON、坏 sqlite。`ValueError` 等程序 bug **继续裸抛**（不把 bug 说成用户错误）。
  另：`error:` 行**不再被 `--quiet` 吞**（quiet 是"别啰嗦"，不是"别报错"）。
- **门禁不再自我满足（V-22/V-23/V-24、R-68）**：
  - V-22：开关门禁改成**结构性**校验（围栏代码块里的命令行，或「命令|开关|作用」表的开关列
    + 同行说明列非空）；散文里提一句不再算登记；
  - V-23：文档命令扫描从 `bash`/`sh` 扩到 `console`/`text`/无标签等围栏，并写清三条明确排除
    （数据语言标签、`{`/`[` 开头的数据块、行首锚定 + 中文标点=散文）；
  - V-24：空输入集（无 `.git`/无 tracked 文档/扫不到围栏）**fail 并说清原因**，
    唯一出口 `KS_ALLOW_EMPTY_DOC_SCAN=1`（只有字面 "1" 算同意）；旧行为是整族静默 skip；
  - R-68：门禁清单从 `EXPECTED_GATES` 派生 + 显式集合断言（旧参数化漏了 `playbook`）。
- **子命令级 P2（V-26…V-30）**：案件目录 `manifest.protocol_inputs` 真的带输入指纹
  （新增 `packet.protocol.calibration/memtrace`，扁平形状与消费者一致）、`next_actions`
  与 `next.md` 同源、argv 路径脱敏、`--screen off`/`--csv`/`--launch-csv` 不再静默不写 case；
  `issue` 缺时钟打 `n/a` 而不是 `0`；**新增拒绝码 `no_nvtx`**（无 NVTX 区间不再错报
  `metric_absent`，处方指向"重采时带 `--trace=nvtx`"）；`route` 的"会被拒的条件"从真源派生
  并全部渲染（旧实现只列 4 条、只印前 3 条，读者第一脚就踩到没列出的 `no_nvml_samples`）；
  `skill --apply --source <不存在>` 报错而不是 rc=0 说"已经是目标状态"；
  案件 `README.md` 压到文档承诺的 ≤1 KB（原 1245 B）。
  **接口变更**：`case/manifest.json` 的 `next_actions` 由**整数**改成
  `{count, items}`（内容与 `next.md` 同源）——读它的脚本要改；仓内无其它消费者。
- **R-60 的第三个旁路**：带 `<占位符>` 的文档命令此前在**真解析之前**就放行，同一行的
  结构性/值级错误一律不查；现在先把占位符换成形状合法的夹具值（枚举开关取真解析器的
  第一个合法值），再交给真解析器。
- **`--plan` + 子命令的形状陷阱**：`kernoscope --plan skill --apply` 会把 `skill` 当成
  "要被运行的程序"（计划里写成 `运行目标 = skill --apply`，去掉 `--plan` 就真的会 exec 它）。
  现在当场 `rc=2` 并给出正确写法（`kernoscope skill --help`）。
- **R-62**（部分）：新增 L2 读峰值 needle（`l2_read_peak_gbs` 此前没有任何 needle）；
  完全穷尽的"每个公开数字都能追到产物"仍未做到，如实记在这里。

### 新增能力：窗口的环境证据

- `meas__sm_clock_p50_mhz`（只在**活跃**采样点上统计时钟驻留，含空闲点中位数同时给出）、
  `meas__throttle_limited_pct`（降频位图，空闲位不算受限，未知位照实写 `bitN`，
  驱动只给 Active 时记 `unspecified`）、`meas__gpu_idle_pct`、`meas__other_procs_on_gpu`
  （同卡其他计算进程，排除我们自己拉起的负载进程树）。
- 决策包新增前提 `env_gpu_exclusive` / `env_gpu_busy_pct` / `env_no_throttle_pct`：
  环境不干净时 `cost_per_token` 自动降级，而不是假装独占。

### 逆向政策

- 从"不逆向"改为"**逆向允许、冒充计数器语义不允许**"：可以用没有文档的面（驱动位图、
  NVML/nvidia-smi 查询面、各家工具输出格式）与通用逆向工具（形如 REA 的本地分析桥）；
  边界在发布出去的值上 —— 同源量按 ncu 的单位与语义写、代理量只留自有命名空间、每份文件带署名。
- **2026-10 收紧一条（同一条政策的修正，不是新政策）**：**不反汇编/反编译 NVIDIA 专有二进制**。
  依据：EULA §2.1.3 明文禁止反向工程（**合同**层面的禁止，DMCA 豁免不豁免合同），
  且 DMCA 第九次三年期豁免（89 FR 85437，2024-10-28 生效）的类别里**没有**覆盖 GPU 性能计数器/
  驱动 ABI 的任何一类。同时它也是**无效**路径：能拿到的 C ABI 本来就公开导出
  （`nvmlDeviceGetFieldValues` 与整个 `NVML_FI_DEV_*` 家族就在公开 `nvml.h` 里），
  头文件里没有的 enum 值要靠**调用并观察返回码**枚举，不靠反汇编；
  通用逆向工具在 NVIDIA 专有面上与我们的问题是零交集。**读**公开导出面与运行时返回**不受影响**。
  另把风险分成相反的两类写清：读公开 ABI 不是规避访问控制，绕过 `RmProfilingAdminOnly` 才是。

### 错误路径与可发现性（第六轮 W-2 ＋ 一次易用性实测）

- **模块级函数不再直接调用 `_main` 的 `log_error` 闭包**：`collect()` / `_attach_truth()` /
  `_export_substitute()` 都是模块级函数，历史上直接调 `log_error(...)`，于是**6 个调用点**
  在命中时抛 `NameError`。其中两条落在**最高频**路径上：「没给目标命令 / 忘记写 `--` /
  拼错子命令」—— 源码里已经写好的那句提示永远打不出来（第六轮 W-2 记的是同一根因的另一处）。
  现在统一走模块级 `_log_error(log, …)`；`_main()`/`_stderr_log()` 把 `log_error` 挂在 `log`
  上，所以"错误不受 `--quiet` 影响"这条纪律不变，没挂上时也退化成普通日志而不是崩。
- **「没有目标命令」与「目标命令无法执行」改用 `CliInputError`**（原来在进程内调用 `main()`
  会抛 `SystemExit`，只有 console-script 才把它变成 rc=1；这与其余输入错误的通路不一致）。
  现在 `main()` 直接返回 1、打一行 `error:`，文案给出三种写法（`--` 之后 / `--ks-sqlite` /
  `--ks-synthetic`）。
- **顶层 `--help` 现在列出全部 10 个子命令**：一次易用性实测发现 12,993 B 的帮助里
  `route`/`capabilities`/`gate`/`skill` 各出现 **0 次** —— 四个核心入口在帮助里不存在。
  说明表 `SUBCOMMAND_HELP` 与 `SUBCOMMANDS` 由测试对拍（加了子命令不写一句说明就红）；
  不做 argparse subparsers 大改，避免动到 `--` 之后目标命令的语义。

### 决策包：结论不许静默消失（第六轮 R-106/R-110 + W-4/W-6 收口）

- **判词身份不再用"截断显示串"**（第六轮 **W-4**）：`over_peak` 与 `roofline` 判断
  "是不是同一个 kernel"改用**未截断名**（`Verdict.owner_key`，不进包）。旧实现比较
  `facts["kernel"]` —— 那是 `_clip(name, 48)` 的显示串，而 C++ 模板实例化名前 48 字撞车是常态，
  于是一个便宜 kernel 的超峰会否掉另一个 kernel 合法的 91% roofline。
- **判词硬上限改为"先保结论四条"**（`kernel_time`/`cost_per_token`/`saving_estimate`/
  `truth_vs_substitute`）：旧实现是 `vs[:n]`，靠位置，只因 `BUILDERS` 的顺序恰好把结论排在
  前面才没丢。同时 `verdicts_more.ids` 改为**按判词在包里的原序**输出（预算裁剪分两轮
  6 条→4 条时，旧实现把两轮的顺序首尾相接，不再等于原序）。
- **决策包新增 `cost.trimmed_ids`**（`"<步骤>:<判词 id>"`）：facts 被裁以前只有步骤名，
  2400 个场景里 facts 被裁 663 次却不知道是谁的 facts 没了（**R-106**）。
- **`cost.trimmed` 的语义收窄**：只记**真正改动了包**的步骤。旧实现连 no-op 步骤也记，
  于是"裁剪步骤"不等于"真的裁了什么"；判据用内容比较而不是字节数 —— 因为"裁判词"那一步会
  同时删判词、加 `verdicts_more` 记账，净字节可能变大。
- **新拒绝码两个**（闭集扩大，外部消费者可分支）：`env_shared_gpu`（同卡还有别的计算进程：
  读数被他人负载稀释）、`env_throttled`（窗口内有降频：能效读数被扰动）。三条环境前提
  `env_gpu_exclusive` / `env_gpu_busy_pct` / `env_no_throttle_pct` 现在有可读标签与 code ——
  此前降级文案**直接印 id**（第六轮 **W-6**）。
- **生成物 `references/verdicts.md` 的「需要的前提」列**改为从 `packet.VERDICT_PREMISES`
  单一真源生成（旧实现读生成器自己的手写表，于是 `cost_per_token` 在文档里写「无强制前提」
  而代码给它挂三条环境前提，`--gates` 却 9/9）。新增门禁：代码里出现的每个
  `Premise("…")` 都必须在标签表里，且标了标签的前提必须真的有人发。
- **README §8.2 补齐两条判据的语义**：`--max-metric`/`--min-metric` 是**逐 kernel** 判
  （缺项 / `n/a` 都算不通过），`--forbid-code` 搜索包里**任何**能携带 code 的位置。

### 导出路径与设备级判词（第六轮 R-110 收尾）

- **`unit=""` 不再被当成"按假设换算"**：`select_metrics` 此前用 `m.unit or None` 把空串
  变成"调用方拿不到单位"，于是 `gpu__time_duration.sum` 会被无条件 ×1e6 写进 `.ncu-rep`
  （端到端曾写出 `9.51e11` ns）。现在空串按"不知道"处理 ⇒ 这一行**不写**，并出现在
  "因命名空间/单位规则未写入"的名单里；`from_units` 本来就含空串的行（`launch__*`）不受影响。
- **设备级超峰判词不再静默消失**（W-12）：超峰警告只挂在 `(device)` 伪 kernel 上时
  （`-i` 回读老报告可达），`kernel_with` 会跳过 `(device)` ⇒ 旧实现让 `over_peak` 判词消失。
  现在退回设备级读取，照常产出 `refused` + `actionable=false` 的判词。

### 不做什么：从散文变成可机检的决策记录

- 新增仓库根 **`NON-GOALS.md`**：13 条「不做」的**单一真源**，每条写全六件事
  （声明 / 保护什么 / 代价 / 前提 / 择中形态 / 判定）。此前它散落在四五份工作文档里，
  于是同一件事在四个文件里有三种说法（`.ncu-rep` 一处写"明确不做写入"、一处写"不碰私有格式"，
  而代码已经在导出），且前提变化后没人知道该推翻哪条。它有能红的门禁
  （`tests/test_non_goals.py`：六件事缺一、判定脱离闭集、实施状态引用不存在的条目、
  README 不再指向真源 —— 四种变异都必须被抓到）。README 的不做清单改成摘要表 + 指向真源，
  并且**只写已实现的形态**（"已决定、未实现"的逐条列在 `NON-GOALS.md` §3.1）。
- **口径更正（会影响按 README 取数的读者/agent）**：README 能力表「每 kernel 能耗」的出处
  由 `measured` 改为 `estimated`。能耗计在**设备**上，按窗口内时间占比分摊是**代理口径**
  （`est__kernel_energy_j`，note 里写着"假设窗口内功率大致均匀"）；设备级/窗口级能耗
  （`meas__device_energy_j` / `meas__window_energy_j`）仍是 `measured`。
  依据：CUPTI **没有** `CUPTI_ACTIVITY_KIND_ENERGY`，所以"每 kernel 的实测能耗"在任何机器上都不存在。
  需求文档附录 A 的同一行也需要同样更正。


### 导出的 `.ncu-rep`：同源行改用 ncu 的单位/标签，并让 `--page details` 显示出来

- **会改变外部消费者看到的东西**：`gpu__time_duration.sum` 现在**会**写进导出文件，
  单位是 **ns**（我们的内部值是 ms，写文件时 ×10⁶，只换算一次）。同时新增
  `launch__thread_count`、`launch__shared_mem_per_block_dynamic` 两行。
  以前这几行被整体略过（"单位不同就不写"）——现在按"同源 + 换算"的规则写。
- 新增 `ncrep/ncu_rows.py`：白名单行表（名字 → ncu 显示标签 / 单位 / 换算系数 / 所属节）
  与节表（`SpeedOfLight`、`LaunchStats`）。**标签必须逐字取自 ncu 自带的节定义**，
  否则 `--page details` 会整行不渲染（真机二分实验结论）。
- 文件结构新增：`RuleResults`（每个节一条 + 一条署名 `Warning`）、`SourceData.CudaSmVersion`
  与 `SessionDetails.DeviceAttributes`。`--page details` 是**按 rule 驱动**的页面：
  没有 rule 指向某个节，那一节连标题都不显示；所以每个节都配了一条 rule，
  同源值同时以文本形式写进 rule 的 Message（表格行在替代口径报告上无法渲染，
  原因见 README §11⑧）。
- 旁路署名（`<path>.ATTRIBUTION.txt`）新增 `unit` 与"表格"两列，并给出换算说明
  （`37,501,500 ns（我们的值 37.5015 ms × 1e6）`）。
- 读这些文件的脚本若按 `gpu__time_duration.sum` 取值，需按 **ns** 理解（此前该名字不存在）。


修掉第三轮验收报告里的阻塞缺陷。**两处会改变外部消费者看到的东西**，逐条列在下面。

### 决策包（`--packet`）

- **v1.2（追加式）**：新增 `verdicts_more`（判词超过上限时，说明少了几条、少了哪些）
  与 `refusals_omitted`（拒绝条目被裁剪时，给出数量与 code 列表，列表过长时标
  `codes_truncated`）。`schema_version` 仍是 `1`。
- **`dram_bytes` / `dram_bytes_per_launch` / `dram_bytes_trace` 的 `value` 单位修正**：
  真机指标的单位是 `MB`，而包里过去直接把该数值当**字节**用 —— 7.27 TB 被写成 7.3 MB
  （`value` / `facts.mb` / 自然语言 `claim` 三处同时差 10⁶）。现在统一折算成字节，
  `unit` 恒为 `B`，`facts.mb` 与 `value` 同源。**读这三个判词的脚本要按字节重新理解数值。**
- **拒绝条目参与尺寸裁剪**：受限环境（无 nsys/NVBit/cubin/计数器）天然产生十几个拒绝码，
  过去只裁 `cmd` 子字段、条目本身不裁，包会常态化突破 4096 B 预算。现在有序降级里加了
  「处置文本截短 → 拒绝条目上限」，并把裁掉的 code 列进 `refusals_omitted`。
- 裁剪顺序的注释与实现对齐；`write_packet` 不再额外追加换行，**`--packet <路径>` 的落盘字节数
  == `packet_bytes`**（`--packet -` 走 stdout，是流，按惯例补一个终止换行 ⇒ 重定向出来的
  文件会多 1 B；两种路径都有测试钉住）。

### `kernoscope diff`

- **schema v2**：`not_judged`（只给差值、不下方向）从前被序列化成 `incomparable`，与
  「协议不同 / 一侧缺失」混在同一个 token 里，机器按 `status` 计数必然对不上。现在它是
  独立状态，且计入 `summary.not_judged`。
- **判据改为「相对差 vs 相对误差带」**：`%` 单位的判词过去拿**百分点**去比**相对** MDD，
  于是 roofline 20% → 15%（真实退化 25%）被判成噪声。
- **0 基线不再算成"没变化"**：`a=0, b≠0` 时相对差无定义，现在走 `incomparable` +
  新拒绝码 `zero_baseline`，并把绝对差值报出来。

### 命令与管线

- `protocol_hash` 现在包含**标定身份**（文件名 + 内容哈希）、峰值、kernel 选择、
  warmup/window 秒数。以前这些键从不被写入，于是"换一份标定""把窗口调短"都不算协议变化，
  `diff` 会照报"真改进"。标定身份不含路径与时间戳（否则同口径每次都不可比）。
- `--strict` 的覆盖范围写进 `--help`/docstring（它卡的是**请求面**；低覆盖率导致的
  DRAM 口径不可用不在其中，要卡请同时给 `--ks-ledger-*`）。
- `capabilities --brief` 的 `next_layer.bytes` 现在等于**它自己命名的那个命令的真实输出
  字节数**（以前用紧凑序列化算，低估 35%）；新增 `blocked_by_unavailable`。
- `capabilities --json` 的输出不再多一个换行。
- 文档命令门禁：**全部**可检命令（当前 65 条）都交给**真实解析器**（11 个子命令各有自己的
  `build_parser()`，是 `--help`／门禁／测试共用的唯一真源；续行、`sudo`、`$( )`、
  `python -m kernoscope`、`xargs … kernoscope` 都覆盖，且**不许静默丢弃**）；
  其中**读者面 11 条**逐字执行并断言结论（比例 11/65，不是"每条都断言"）。
  注：这三个数字随代码走动，若与 `tests/test_documented_commands.py` 的断言不一致，以测试为准。

### 访存轨迹口径（NVBit）

- **按访存类型分流（R-73）**：全局口径只用 `LDG/STG`（含 PTX 的 `.global`）；
  `LDS/STS/LDSM`（共享）与 `LDL/STL`（本地）不再进入 `sectors/request`、合并效率、
  工作集与 L2 模型 —— Triton 的 `tl.dot` 会产生海量共享访存（一个 FlashAttention 轨迹里
  1,282,048 条记录只有 8,192 条是全局的），混进来会把合并效率算成 **>100%**（物理不可能）。
  新增指标 `meas__trace_records_global`（全局口径的记录数），并保留 `min(100, …)` 兜底 +
  `coalescing_clamped` 标记。
- **opcode 尺寸解析（R-73）**：SASS 的 `STS.U16` / `LDG.E.U8` 与 PTX 的 `v4.f32` 都能算对
  （旧实现只认 PTX 的 `.<宽度>`：`STS.U16` 回落 4 B、`v4.f32` 漏乘向量宽度）。
- **替换策略上下界（R-77）**：新增 `est__dram_bytes_per_launch_trace_bounds`（强制缺失下界
  ~ 完全不复用上界），并在正文里点明"**地址部分是 measured 级算术，只有谁命中 L2 是模型**"。
- **`--ks-max-accesses`（R-74）**：拒绝文案给出**具体该设多大**（"把 --ks-max-accesses 提到 ≥ N"）。
- **轨迹自证（R-75/R-76）**：解析器给每条轨迹记 `trace_sha256` 与完整性
  （`yes`/`no`/`unknown`）；`collect.sh` 追加 `KS-TRACE-COMPLETE` 哨兵（官方格式没有结束标记，
  被 `ulimit`/磁盘截断时察觉不到，报告里两次因此拿到低于物理下界的数）；结尾不是换行一律按
  截断处理；**协议哈希新增 `memtrace` 身份**（文件名 + 指纹），换了轨迹不再被当成同协议。
- 依赖说明（`tools/probes/nvbit_memtrace/README.md` + `collect.sh`）改成**版本×构建矩阵**。
  ~~v1.8.1 走 CMake（本仓库真机验证）~~ —— **按第五轮验收报告 V-10 的真机 A/B 更正**：
  本环境（CUDA 13 + sm_89 + 驱动 580）唯一实测可用的是 **v1.7.7.3 + 老 Makefile**
  （产物在 `tools/mem_trace/mem_trace.so`）；按文档 CMake 命令构建的 v1.8.1 实测失败
  （rc=1、0 条记录、`CUDA_ERROR_INVALID_SOURCE device kernel image is invalid`）。
  原句"v1.8.1 真机验证"不实，已删；另写明"不要 make 整个 tools / 架构变量用裸数字 /
  必须 NO_EAGER_LOAD=0 / 别把版本号当可用性凭据（同版本不同构建结果可以不同）"。
  新增 `tests/test_nvbit_smoke.py`：按文档真采一段轨迹并核对版本号（无 NVBit 时跳过）。

### 零源码改动的达到 occupancy（NVBit 插桩，新增工具）

- 新增 `tools/probes/nvbit_occupancy/`（NVBit 工具 + `collect.sh` + `Makefile` + README）：
  把 `(smid, g0, g1)` 两个 `%globaltimer` 读**在 SASS 层注入**被测 kernel，产出与
  `occupancy_probe.cu` **同一种** `ks-occupancy-records v1` 记录 —— 闭源 kernel
  （cuBLAS / Inductor 生成 / 库里的）第一次能测达到 occupancy，被测 kernel **零源码改动**；
  `kernoscope occupancy --from …` 与 CLI **一个字节都没改**。
- 入口钩子 = 偏置最小的指令（每 CTA 一次，`atomicExch` 抢门）；出口钩子 = 偏置最大的指令
  （每 warp 一次 `atomicMax.u64`，取**最后一个**收尾的 warp）。出口是**近似**：NVBit 没有
  「CTA 结束」钩子，early exit 的 kernel 会少报。
- **覆盖率纪律**：文件头写 `grid_blocks` / `cta_rows` / `coverage` / `coverage_ok`；
  覆盖率低于 `KS_OCC_MIN_COVERAGE`（默认 95%）、同一个 kernel 被 launch 多次、或前后
  grid/block 不一致时，工具在 stderr 逐条说明并**默认不写记录文件**
  （`KS_OCC_ALLOW_PARTIAL=1` 才强行写出，文件带 `# coverage_ok 0`）；`collect.sh` 据此 exit 3。
  部分数据不许静默变成分布（既有分析路径看不见 `coverage_ok`，所以拒绝发生在产出之前）。
- ⚠️ **未在真机验证**：本工具在任何 GPU / 任何 NVBit 版本上都没有编译或运行过（本仓库只钉住过
  官方 `mem_trace` 的 v1.7.7.3 + 老 Makefile 组合）。该目录 README 有专门的「未在真机验证」一节。
- 新增 `tests/test_nvbit_occupancy.py`（19 条，无 GPU/无 NVBit 也能跑）。

### 测量口径（重要）

- **per-launch 的分母落在积分窗口内**：过去分子是窗口内字节、分母是**整段运行**的 launch
  数，窗口一被裁就等比偏小（真机实测同一负载 `--ks-window auto` 1.06 GB、
  `--ks-warmup 10` 859.86 MB、`--ks-warmup 20` 517.26 MB，而 share=100%、无告警、rc=0）。
  现在按**时间重叠加权**算窗口内等效 launch 数；两个时钟对不上（nsys 的时间戳是
  **会话相对**的，不是 monotonic）时用窗口内忙碌段**推导偏移**并自检速率比，推不出来就
  拒绝（新拒绝码 `window_launch_unaligned`，并报可算偏差）。
- `SHARE_TOO_LOW` 的处方不再推荐"缩窗口"：份额按整段 launch 表算，缩采样窗口改不动它，
  照着做既解不开前提、又会在能出数时把 per-launch 弄错。
- `over_peak` 成立时，由它算出的 `roofline` 判词继承拒绝（不再出现"物理上不可能的
  24041.6% 仍可依据"）。
- 100%~102% 容忍带不再静默：`near_peak` 会给一句提示（"先核对峰值口径与工作量，别当达标"）。
- `selftest` 的 matmul 峰值带上**尺寸**与适用范围（它是单一尺寸采样，不要当
  `--ks-calibration` 用）；`mdd.floor_evidence` 指向本产物自己的 `run_to_run_sd_pct`，
  不再引用一份不含该数字的 bench 产物。
- 标定工具的口径变体改为**测量时刻的真实设置快照**（过去三个变体其实在同一套设置下测，
  记录的是字面量）。

### 产物身份、真值口径与测试卫生（第六轮 R-102/R-107/R-108/R-109）

- **`environment.txt` 现在同时记两处 NVBit 身份**（R-102）：`nvbit:` + `nvbit_version:`
  （`NVBIT_PATH` 那棵树）与 `memtrace_so:` + `memtrace_so_version:`（**这次真正被注入的那份
  `.so`** 及其实物版本）。二者可以不是同一棵树：`collect.sh` 优先用环境里预置的
  `NVBIT_MEMTRACE_SO`，而旧版 `bench/run.sh` 记的版本取自 `NVBIT_PATH` —— 真机上那棵树里 3 份
  v1.8.1 全挂、实际采到轨迹的却是另一棵树里的构建，产物因此解释不了自己的轨迹。
  新增 `collect.sh --resolve-only` 作为"用哪份 .so + 它的版本"的**唯一真源**（`bench/run.sh`
  问它，不再抄候选路径）；版本一律从实物读，读不出写 `n/a`（**不编**）。口径是**构建级**：
  版本号只用来指认构建，能不能用要看实物 A/B。
- **"真值对拍"降级为"包装逻辑已验；真报告口径未验证"**（R-107/R-90）：仓库里没有真 ncu 报告
  （唯一那份 `.ncu-rep` 的头是 `kernoscope-substitute-v1`），采一份需要 `ncu` 能收集计数器的
  机器。README §4.3/§5/§8/§11 同步改口，并新增测试钉住这句话不被静默改回。
- **署名文件的两处对齐**（R-108/R-93③④）：署名表里的整数按整数印（`1048576` 不再是
  `1.04858e+06`；与 details 页共用 `ncrep.profile.format_value`，换算系数也写 `1,000,000`）；
  文件名固定为**原样拼接**的 `<path>.ATTRIBUTION.txt`（`out.ncu-rep` →
  `out.ncu-rep.ATTRIBUTION.txt`），不再有 `with_suffix` 那类把 `.ncu-rep` 吃掉、让人按约定找不到的写法。
- **测试卫生**（R-109）：`pyproject` 的 `[tool.pytest.ini_options]` 加 `pythonpath = ["src"]`
  —— editable 的 `.pth` 钉的是"安装那一刻"的源码树，单跑某个测试文件可能 import 到**别的**
  `src` 而假绿（第六轮一个复核集群在改 `src/` 的变异上就是这样被骗的）。新增
  `tests/test_src_authority.py`：断言 `kernoscope.__file__` 落在本仓库 `src/` 下、且进程里
  没有第二棵树进来的子模块。

### `gate` 的门禁契约（第六轮 W-3/W-7，R-98/R-103）

- **`--forbid-code` 的搜索范围扩到"包里任何能携带 code 的位置"**：旧实现只查判词的
  `refusal.code`、`refusals` 索引的键、`environment_refusals[].code`，于是 code 被预算裁进
  `unmeasured[].code` / `refusals_omitted.codes` / `environment_refusals_more.codes`
  （受限机上十几个 code 时的**常态**）或是出现在 `next[].code` / `env.blocked_by` 时，
  **拼对的 `--forbid-code` 也 rc=0 放绿**。现在覆盖这些位置（实现是扫全树，不枚举载体）。
  闭集校验不变：未知 code 仍是用法错误 rc=2。
- **指标规则写错 ⇒ rc=2（此前 rc=1）**：`--max-metric X=abc` / `--max-metric foo` 这类
  **语法/数值错误**与未知 code / 未知 status 同类，退用法错误 rc=2；同时不再额外吐一句
  假的失败「一条规则都没给」（真·没给规则时那句保留、仍 rc=1）。
- **用法错误 + `--json` 现在有输出**：此前用法错误只往 stderr 打印、`return 2`，stdout **全空**，
  agent 拿到的 `usage_errors` 字段永不可达。现在 `--json` 在用法错误时也输出
  `{passed, packet, checked, usage_errors, failures}`，**rc 仍是 2**（CI 照样红）。
- **`--require-actionable` 只接受真布尔**：包里的 `actionable` 若是字符串 `"false"`
  （手写/外部 JSON 的常见形态），旧实现 `bool("false")` 强转成 True ⇒ **rc=0 放绿**。
  现在类型不对是**用法错误 rc=2**（不是"判词不合格 rc=1"），绝不静默当通过。

### 环境证据：读不到不许变成"干净"（W-1/W-5，P0；R-96/R-100）

- **降频位图读不到 ⇒ 未知，不再写成"没有降频"**：pynvml 的
  `nvmlDeviceGetCurrentClocksThrottleReasons` 抛异常、或 nvidia-smi 给 `n/a`/认不出的 token，
  一律记 `THROTTLE_UNKNOWN`(-1)。修前"读不到"被吞成 `mask=0` ⇒
  `meas__throttle_limited_pct=0.0 [measured]` ⇒ 前提 `env_no_throttle_pct observed=0.0 ok=True`
  ⇒ `cost_per_token` 保持 `actionable=True`（假绿）。
- **新增拒绝码 `env_reading_unknown`**：整段窗口的降频读数都读不到时，
  `meas__throttle_limited_pct` 报 `unavailable` + 该 code —— **不再**印一个
  `100.0 [measured]`（"未知"既不是"没有降频"、也不是"一直掉频"，两种都是编），
  前提 `env_no_throttle_pct` 因此**不成立**，`cost_per_token` 被降级为 `refused`。
- **`gpu_idle_pct` 加 `sm_util==0` 兜底**：位图拿不到时不再把真空闲报成 0.0%
  （那条会给 `env_gpu_busy_pct observed=100.0 ok=True`）；兜底口径在 note 与 source 里写明。
- **多租户探测有"未知"通道**（R-96④）：`Sampler.environment()` 新增 `other_procs_probe`
  （`not_run`/`ok`/`failed`）；探测失败**不写** `other_procs`，section 改报
  `meas__other_procs_on_gpu = unavailable` + `env_reading_unknown`，前提 `env_gpu_exclusive`
  按**不成立**处理 —— "没测到"不再与"0 个别的租户"同形。
- **`--query-compute-apps` 与两条 `--query-gpu` 都带 `--id=<本卡>`**：多卡机器不再把别的卡的进程
  算成"同卡"，也不再因多行输出静默丢光所有采样；`Sampler` 新增 `device_index`（默认 0）。
- **`decode_throttle(-1)` 不再死循环**；负数按"未知"返回 `unspecified`。
- **环境前提的门槛变成具名常量**（R-100③）：`ENV_OTHER_PROCS_MAX=0.0`、
  `ENV_GPU_BUSY_MIN_PCT=80.0`、`ENV_NO_THROTTLE_MAX_PCT=5.0`（README §9 已列出处与观测不足时的行为）。
- **两条时钟指标互相说明**（R-100②）：`meas__sm_clock_mean_mhz`（含空闲）的 note 写清它与活跃口径
  `meas__sm_clock_p50_mhz`（只在干活时统计）的差别；`meas__sm_clock_max_mhz` 写明"曾经跑到过"。
- **降级文案不再印 `None`**：前提读不到时写「**读不到**（未知，不是 0）」。

### 发现层第一级：不需要标定与价格的优化头寸（M13）

- **`saving_estimate` 不再在缺标定/价格时消失**：修前它依赖 `meta["prescription"]`（需要本机标定
  + 工作量输入），第一次上手的人/agent 通常都没有 ⇒ 最有价值的那半句「钱在哪、先改什么」
  **整条判词不出现**（实测合成包）。现在两级：
  **第一级**按 **CUPTI activity 的实测时长占比**排出 top-N 头寸（`exact`，零前提）；
  **第二级**（保守可省区间 + 钱）在有标定与价格时照旧给出。
- **第一级带三个诚实的字段**：`facts.basis="time_share"`（明说这是时长占比）、
  `facts.money="unavailable"`、`facts.needs=[...]`（升级到第二级还缺 `calibration` / `price`）。
  数字是占比，**不是**节省比例、也不是钱 —— 有测试钉住（不许出现 `total_per_hour`/`total_per_mtok`）。
- **拒绝也带上第一级排序**：有 `prescription` 但 `ranked` 为空时，`refused` 判词同样携带
  `basis/time_share` 与 `top[]`，让"拒绝"仍然给出可行动的方向。

### 公开 alpha 前的首用路径（接口变化；README 已加 Alpha 声明）

- **`doctor` 的退出码改了语义**：0 = **诊断完成**（结论可以是"这台机器拿不到计数器"），
  1 = **诊断本身没完成**（探针异常，此时**必须**往 stderr 打一行），2 = 用法错误。
  修前 `return 0 if rep.ok else 1`：一台没有 GPU/nsys 的笔记本上 rc=1 且 stderr **0 B** ——
  而它是 README 推荐的第一条命令，陌生人与 CI 第一眼看到"失败且没有解释"。
  `doctor --json` 新增 `diagnosed` 字段（把"结论"与"过程"分开），文本模式结论行也写明这点。
  **按旧语义写脚本的人**：如果你用 `kernoscope doctor` 的 rc 判断"这台机器能不能跑主路径"，
  请改读 `--json` 的 `ok`（rc 现在只表示诊断是否完成）。
- **README 的第一分钟换成三条都返回 0 的命令**：`capabilities --brief`（约 1 KB，带 `next[]`
  与每条的代价）→ `doctor` → `--packet -`；人读的"一屏"显式写成 `--screen only`
  （约 4.4 KB / 40 行），并给出"跟 main"的安装方式 `uv tool install git+https://…`。
- **新增 `bench/preflight-first-run.sh` + `tests/test_preflight.py`**：把开放门槛 G1
  「陌生人首跑」变成**可重跑、能变红**的检查 —— 现场构建 wheel、装进隔离目录
  （等价 `uv tool install`）、逐条断言 rc=0 与产出非空、并给"一屏"一个字节/行数预算
  （≤8 KB / ≤60 行）。脚本第一版是**假绿**（装了上一个提交的 `dist/*.whl`），已改成默认现场构建。

### 首用「一屏」：终端上默认只给一屏（M14）

- **`--screen` 的默认行为变了（只在终端上）**：不给 `--screen` 时，若 stdout 是终端，
  默认等价于 `--screen only`（只打一屏，约 4–5 KB / 40 行）；**管道、重定向、`--csv`、
  `--launch-csv`、`--packet -` 一律不变**（仍然是一屏 + 完整报告或纯机器格式）。
  为什么改：README 承诺"一屏"，而旧默认 `auto` 的语义是"一屏 **+** 完整报告"——
  实测 `--set basic` 是 21.6 KB / 200+ 行，和承诺不符（易用性实测 #3）。
  **要老行为就显式写 `--screen auto`**（显式给值时行为完全不变）。
- `--screen` 的帮助文本同步写清三种取值的差别与"默认随 stdout 是不是终端变化"。

### 单文件 HTML 证据单（N9）

- 新增 `--html <path>`（`-` = stdout）：把决策包渲染成一个**自包含** `.html` —— 内联 CSS、
  零外部 URL、零 JS、无服务器（它不是 Web UI，是"JSON → 一个文件"的渲染器；所以能截图、
  能当附件、能离线打开）。渲染入口是 `kernoscope.html_view.render_html(packet, *, title, onepage_text)`，
  也可 `python -m kernoscope.html_view packet.json > out.html`。
- **出处与数字同行**：`exact`/`measured`/`estimated`/`derived` 各有颜色角标与图例；
  `unavailable` 渲染成 `n/a` + 拒绝码，**不渲染 0**，也不渲染 `facts.uncitable_value`。
  拒绝表给 `code` + 人话 + remedy + 代价 + 命令；未测项按 code 归组；代价块含三个 `*_more` 留痕。
- 尺寸预算：正常包渲染 ≤ 64 KB（有测试）；包里的文本全部 HTML 转义；
  `--html -` 与 `--packet -` 同时给出时报用法错误（stdout 上一次只能要一种格式）。
- `--plan` 会声明这个文件（新操作 `write_html_view`，`_OP_BINARIES` 登记为空集）。

### 同源命名从散文政策变成类型系统（N3）

- **`source_class` 四档 + 机检**：`NcuRow` 新增 `source_class`（默认 `documented`，今天 7 行行为
  逐位不变）。规则：**只有 `documented`（公开文档/公开头文件定义了该量与单位）且单位相符，
  才允许写进 ncu 名字空间** —— 这正是"同源才写 ncu 名下"的可执行形式。其余三档只进
  `ks__`/`est__`/`meas__`，或被丢弃并记机器可读原因（新列表 `dropped_by_source_class`）。
  `ncu_rows.consensus_check()` 拒绝"登记在表里却 source_class 非 documented"的行（表与规则脱节必须炸）。
- **机器可读署名（外部消费者注意）**：`.ATTRIBUTION.txt` 末尾**追加**一个 JSON 块
  （标记 `kernoscope-attribution-json-v1`，`schema=kernoscope.attribution/1`），逐条给出
  `{name, value, ncu_name, unit, our_unit, source_class, provenance, origin}` 与
  `not_written_by_source_class`。人读表格一行没删，但**表头多了两列**：
  `| metric | value | our unit | unit (written) | source_class | provenance | 表格 |`
  —— 旧表头里 `unit` 的语义（写进文件的单位）现在叫 `unit (written)`，按旧列名解析的脚本要改。
  读取请用 `substitute.parse_attribution_json()`（坏块抛 `ValueError`，不返回空字典）。
- **Python API 变化**：`ncrep.substitute.select_metrics()` 由 6 元组改为 **8 元组**
  （前 6 位含义与顺序不变，档位丢弃记录插在第 6 位，`over` 移到第 8 位）。
- `.ncu-rep` 字节格式**未变**（仍 v7）；机器块只在旁路署名里。

### 可选 MCP 层（N10：可选、零常驻、带止损）

- 新增子命令 `kernoscope mcp --stdio`：在 stdin/stdout 上跑**逐行 JSON-RPC 2.0**
  （`initialize` / `tools/list` / `tools/call` / `resources/list` / `resources/read` / `ping`），
  只服务**没有 shell** 的 agent runtime。它是**可选**的：不进默认安装路径、**不进 4 KB 常驻入口层**
  （有测试断言 `AGENTS.md` / `llms.txt` / `SKILL.md` frontmatter 不许出现 mcp），
  止损条件写死在决定里 —— 一段时间内没有真实环境用它就删。
- 4 个工具（`capabilities_brief` / `profile` / `diff` / `selftest`）+ 3 个资源
  （拒绝码参考 / 出处规则 / skill 入口）。**工具不自己算任何数**：它们把 CLI 的既有子命令
  叫起来并回传文本，所以"MCP 里的数字"与 CLI/包的证据模型只有一处真源。
- 坏的一行回 `-32700` 但**不终止长会话**；工具参数不合法回 `-32602`；未知方法回 `-32601`。
- `--stdio` 是唯一传输：不监听端口、不启动服务、不发任何写请求。

```bash
kernoscope mcp --stdio
```

### 反事实视图：缺用户输入时给条件答案（拒绝不再 dead-end）

- 缺价格 / 缺 token 数时，`cost_per_token` 判词的 `facts` 里**追加**一个条件答案：
  `assumed_price_per_kwh`（示例 0.8 元/kWh）→ `if_cost_per_hour` / `if_cost_per_mtok`；
  缺 token 数时 `assumed_tokens_per_step`（示例 1.2M）→ `if_tokens_s` / `if_cost_per_mtok`。
  条件值一律带 `if_basis=conditional`、`if_provenance=estimated`、`if_missing=<code>`
  （原拒绝码仍在 `refusal.code`），并且**不占** `value`。
- 原始判决不变：`no_price` / `no_tokens` 照旧出现、remedy 照旧指路；给了真输入时行为与旧版
  完全一致（不产生任何 `if_*` 键，包字节不变 —— 有测试）。
- 外部消费者注意：`refused` 判词现在**可能**带 `provenance: "estimated"` —— 它描述的是 facts 里的
  条件答案，不是拒绝本身；没有条件答案时不写这个键。

### 有 nsys 的机器上不再把目标的问题讲成 nsys 的问题（第七轮 X-1，P1）

- **交给 nsys 之前先解析目标可执行文件**：`kernoscope doctr`（拼错的子命令被当成程序）现在报
  `error: 目标命令无法执行：doctr 不存在（…）`，与**无 nsys 路径逐字相同**；旧实现在有 nsys 的
  机器上走 `nsys profile … doctr` → `nsys profile failed (rc=1)` + "nsys 随 CUDA Toolkit 提供
  （或单独安装）" —— 把人送去装一个已经装好的工具，而真实原因一个字都没说（真机实测）。
- **目标退出码与 nsys 自身失败分开**（`backends/nsys.py` 新增 `TargetFailed`）：nsys 会把目标的
  退出码透传回来。现在报 `error: 目标命令退出码 N`；有 `.nsys-rep` 时说明"采集文件仍在、可以继续
  分析"，没有时才补一句"也可能是 nsys 自身失败"并附 stderr 尾部。只有 nsys 自己返回 0 却没产出
  才报 nsys 的问题。
- **输入文件的存在性提前到采集之前**（`--ks-calibration` / `--ks-memtrace` / `--ks-instr-hist` /
  `--ks-achieved-occupancy` / `--ks-selfcheck`）：旧实现里"标定文件不存在"会被随后的采集错误盖住，
  从头到尾没被报出来。
- 新增 5 条测试（`tests/test_error_paths.py`），含"nsys 存在时也必须是目标错误"与"采集根本不该开始"。

### M13 的两级拆成两个判词 id（第七轮 X-19，P1 口径级）

- **新增判词 id `saving_estimate_share`**（第一级：按**实测时长占比**排序的头寸，单位 `%`）；
  `saving_estimate` 只用于第二级（保守可省区间与**钱**，单位如 `CNY/h`）。旧实现两级共用
  `saving_estimate`，于是 `diff` 把 `96.6 %` 与 `1.9801 CNY/h` 直接相减、吐出
  `delta_pct=-97.95%` 并把单位标成 `CNY/h` —— 正是本项目最老的错误族（用单一表示承载多重语义）。
  **外部消费者**：按 id 取"先改哪里"的脚本要区分两级（`saving_estimate_share` 只在缺标定/价格时
  出现，零前提、`provenance=exact`、`facts.basis=time_share`）。
- **第二道防线在 `diff`**：同一个判词 id 的两侧若**单位**或 `facts.basis` 不同，判 `incomparable`
  + `protocol_mismatch`，**不给差值也不给百分比**；单位与 basis 都相同时照常给差值
  （有测试钉住这条防线不是"一律不给"）。

### 门禁与脚本在真机上可用（第七轮 X-2 / X-4）

- **`tools/dev_index.py --gates` 不再依赖 `uv`**（X-2）：9 条门禁原本全部硬编码 `uv run …`，
  而**租来的 GPU 机器**（NVbit 冒烟、真 nsys/ncu 用例唯一有分辨力的环境）通常没有 `uv` ——
  那时 `--gates` 是 0/9，"提交前跑一遍门禁"这条工作流在目标机器上等于不存在。
  现在：有 `uv` 走 `uv run`（CI 一字不变）；没有 `uv`（或显式 `--python <exe>`）时把
  `uv run pytest` 翻成 `<py> -m pytest`、`uv run python X` 翻成 `<py> X`、`uv build` 翻成
  `<py> -m build`，并**打印用了哪个解释器**。实测本机无 uv 路径 **8/9**（唯一失败是
  venv 里没装 `build` 模块，报文直接说"`pip install build` 或装 uv"）。
- **`_ks.sh` 的解释器解析修对了两件事**（X-4 真机 127）：
  ① 先找仓库/相邻 `.venv`，再找 conda 与系统绝对路径（`$HOME/miniconda3/bin/python3`、
  `/root/miniconda3/bin/python3`、`/opt/conda/bin/python3`、`/usr/bin/python3`…），最后才是 PATH；
  ② **每个候选都要过版本关**（`>=` `requires-python` 声明的 3.10）—— 不加这一关时 macOS 的
  `/usr/bin/python3`（3.9）会排在仓库 `.venv` 前面，把本来能用的解释器挡住（这就是真机上
  `setup-check.sh` 报"kernoscope 装好了吗？"的原因）。新增测试把版本关与 `requires-python` 对拍。
- **`bench/preflight-first-run.sh` 自带 `UV_CACHE_DIR`**：陌生机器/沙箱里 uv 的默认缓存位置
  不一定可写，而"缓存不可写"会让 `uv build` 失败、看起来像产品坏了。

### `gate` 的三条放绿/崩溃路径（第七轮 X-13 / X-14 / X-16）

- **数值阈值打到字符串取值的指标 ⇒ 用法错误 rc=2**（X-13）：`--max-metric est__bound=1`
  （`est__bound` 的值是 `"memory"`、`ks__occupancy_limiter` 是 `"registers"`）旧实现是
  **裸 `ValueError` traceback**。现在归类为"规则对这类指标无定义" ⇒ rc=2 + `usage_errors`，
  报文点名指标与值（数值阈值只适用于数值型指标）。
- **没有归属的超峰要保守传播**（X-14）：`ks__over_peak_warning` 只挂在 `(device)` 伪 kernel 上时
  （`-i` 回读老报告可达），它的 `owner_key="(device)"` 与真 kernel 名必然不同 ⇒ 旧逻辑判
  "不是同一个 kernel"、不传播，于是 `--require-actionable roofline` 对一份"达成率物理不可能"的包
  rc=0。现在 `(device)`/缺身份视为**未知归属 ⇒ 保守传播**（宁可拒绝一个可疑的达成率）。
- **`--forbid-code` 在 code 列表被截断时判"无法判定 ⇒ 不通过"**（X-16）：
  `refusals_omitted.codes` 最多列 4 个（再多只给 `count`），受限期机器上十几个 code 时是常态；
  旧实现"可见部分里没找到 ⇒ 通过"，于是 13 个真实 code 查不到就被放绿。现在截断时给
  `无法判定：…（还有约 N 个 code 未列出）` + 最便宜的补法，rc=1；**没有截断且确实没出现时照常通过**
  （有反面对照测试，防止这条防线变成"一律不通过"）。

### 进程级重复执行 `--repeat N`（M5 / N1）

- 新增 `--repeat N`（引擎 `kernoscope.repeat`，此前只有引擎、不可达）：把**整条命令**跑 N 次
  （每次独立进程：自己的采集、报告与身份），给出 `distributions`（`n / not_produced / min / max /
  mean / sd`，只含成功运行）、`mdd_lower_bound`（`max(3×sd/|mean|×100, 0.5)`，与
  `selftest.SD_FACTOR` 同源并有测试钉住不漂移）、逐指标 `refusals` 与 `cost`（GPU 秒；不编价格）。
  实现是**自我调用**：子进程走的是与单次运行完全相同的那段代码，所以重复执行不引入第二条口径。
- 输出：`--packet -` 给**一个** JSON（agent 用）；`-o <前缀>` 另写 `<前缀>.repeat.json`；
  否则打一屏人视图（若干次成功、分布表、逐指标 n/a + remedy、成本）。
- **失败的那次照样记、但不参与分布**（不完整的报告会把 sd 压小 ⇒ 把噪声判成真改进）；
  只有在没有任何指标能给出 MDD 时 rc=1 并说明"没有可用分布"。
- 拒绝码**全部复用既有闭集**（`mdd_unknown` / `protocol_mismatch` / `zero_baseline`），未新增。
- 诚实边界（人视图与文档里都写）：不是 kernel 级重放（不插桩、不锁时钟、不碰其他租户），
  也**不控制 warm/cold 缓存** —— 喂给 `diff` 时与误差带取最大值，只能抬高门槛。
- `--plan` 会声明这次会重复 N 次与（给了 `-o` 时）那个 `.repeat.json`；新操作 `repeat_runs`
  在 effects 表里登记。`-i` + `--repeat` 是用法错误（rc=2），不静默忽略。

### 授权遥测：消费平台已经开给你的端点（N7）

- 新增 `--ks-telemetry <url|file>`（+ `--ks-telemetry-timeout-s`，默认 5 s）：读 dcgm-exporter 的
  Prometheus 文本端点（`http(s)://` 或本地文件），把 `DCGM_FI_DEV_*` / `DCGM_FI_PROF_*` 按
  **字段自己的真名与真单位**发布到 `meas__telemetry_dcgm_*`（带端点署名与一行语义），报告里是
  独立的 `Telemetry` 节；缺席字段以 `unavailable` + code 出现（**不是 0**）。
  **这不是绕过计数器权限**：`PROF_*` 出现在端点里只可能是管理员授权了；端点里没有就只说「没有」，
  不推「被拒绝」，也不从 `DEV_*` 反推 `PROF_*`。
- **两条"百分比 ≠ 字节"的陷阱在构造期就不可能发生**：`DCGM_FI_PROF_DRAM_ACTIVE`（旧名
  `DCGM_FI_PROF_DRAM_UTIL_RATIO`）是带宽/周期占比（0–1）、`DCGM_FI_DEV_MEM_COPY_UTIL` 是时间占比；
  名字/单位/维度三者不相容就抛 `TelemetryTrapError`，另有发布前复核，**没有**任何
  "比例 × 峰值 ⇒ 字节"的入口。`B/s` 只出现在确实是字节速率的 PCIe 两条上。
- `probe` 三态（`ok` / `not_run` / `failed`）与 `missing` 的原因码把"没跑 / 读不到 / 端点里没有 /
  实测到 0"分开；网络、IO、超时、HTTP 错误与"一行序列都解析不出"只报 `failed` + 原因，
  不抛异常、也不给"空的 ok"（报告里以 `n/a` + code 出现，整次运行不失败）。
- 复用既有拒绝码闭集（缺席 → `metric_absent`；读不到/未知 → `env_reading_unknown`），未新增 code。
- `--plan` 新增独立声明 `read_telemetry`（联网读标 `external`）—— 它的副作用性质与"读本地数据文件"
  不同，混在一起会让计划的读者以为只是又一个本地文件。

### 测试卫生：一条"只在本地成立"的假绿 + CI 对等检查

- **修掉一条本地绿、CI 红的测试**（`test_ks_python_…`）：它依赖"仓库里有没有 `.venv`"
  —— 本机有、CI 由 `uv sync` 创建 ⇒ CI 红；更糟的是它在本机**假绿**：测试把 `PATH` 清空，
  脚本里的 `dirname` 于是失效、技能树根目录解析错，"找不到 `.venv`"纯属巧合。现在这条测试
  **自带一棵技能树**（把 `_ks.sh` 拷进临时目录）+ 保留 coreutils 的 PATH，并把两条真实契约都钉住：
  有 `.venv` 时它优先、没有时找 conda 布局。
- `skills/kernoscope/scripts/_ks.sh` 的根目录解析改用 bash 参数展开（不再调外部 `dirname`）：
  极简容器/裁剪过 PATH 的环境里，脚本不该因为少一个 coreutils 就"看起来跑通、其实找错解释器"。
- 新增 `bench/ci-parity.sh`：把 HEAD 的干净快照导出、`uv sync`、跑与 CI 相同的门禁命令
  （可传 `<commit>` 复现历史提交的 CI 结果）。这条脚本的价值就是本轮的两次红：
  "本地工作树 + 本地环境"与"刚 clone 的某次提交"不是同一个东西。

### 时钟状态进协议哈希 + 锁频处方（择中项 ⑥）

- **锁频状态成为协议的一部分**：新增 `ks__clock_lock_state`（`已锁频` / `未锁频` /
  `读不到锁频状态`）与顶层 `clock_lock` 字段；协议哈希里放的是**粗粒度状态**
  （`locked` / `unlocked` / `unknown`），**不含**实测时钟 —— 后者每次都变，放进去会让每一次比较
  都判"协议不同"（把噪声当口径）。有测试同时钉住这两条（状态变 ⇒ 哈希变；只有实测值变 ⇒ 哈希不变）。
- **`diff` 点名**：锁频状态不同时不可比，并在协议差异里给出
  `clock_lock: locked → unlocked（锁频状态不同 ⇒ 两次运行的时长/能效不可相减）`
  —— 只说"协议不同"太含糊，读者要的是"怎么才可比"。
- **三态且"读不到"≠"没锁"**：只有 pynvml 能问 `nvmlDeviceGetGpuLockedClocks`；
  `NotSupported`/`(0,0)` ⇒ `unlocked`，查不动 ⇒ `unknown`，没有 pynvml ⇒ `unknown`。
  不拿时钟读数去猜"看起来像锁了"（那是把观测当身份）。
- **锁频处方**：新增拒绝码 `clock_not_locked`（+ `DEFAULT_REMEDIES`）：
  `sudo nvidia-smi -lgc <min>,<max>`，跑完 `sudo nvidia-smi -rgc` 复位，并写明
  **我们不替你锁**（改设备时钟是写操作，不在只读工具的授权里，见 NON-GOALS N2）。
  只有**确定未锁**时才挂这条告示（避免每次真实运行都被"去锁频"刷屏）；`unknown` 时状态照样进
  协议哈希与 `ks__clock_lock_state`，只是不冒充确定结论。
- 该 metric 与**采样**无关（驱动查询），因此放在设备节的采样门之外：采样被关/一次都没采到时
  这条证据依然成立。`references/refusal-codes.md`、`diagnosis-playbook.md` 与
  `eval/tasks/*.json` 已按新 code 重生成。

### 两处"空转的绿"与一处手写计数（第七轮 X-10 余项 / X-12）

- `tests/test_src_authority.py`：配置断言在"`[tool.pytest.ini_options]` 段不存在"或
  "Python 3.10 没有 `tomllib`"时**静默返回空字典**，调用方再 `if not ini: return` ⇒
  这条断言一步没做却看起来通过。现在：缺 `tomllib` **显式 skip 并写明原因**（不是空转），
  段不存在/为空 ⇒ 直接判失败。顺手修掉同一改动引入的隐患：该文件此前**没有** `import pytest`
  （只在 3.10 那条分支上才会用到，于是 3.13 上永远不炸、3.10 上会 NameError）。
- `tests/test_meta_gates.py` 的模块 docstring 里"8 条门禁 + 400 多个测试"是手写计数（实际已是
  9 条 / 950+）—— 改成不带数字的说法。手写计数在这个仓库已经红过好几次，能删就删。

### 又一条"只在某种机器上成立"的测试（第七轮 X-3）

`test_export_substitute_empty_launch_table_has_its_own_error` 原来直接跑 `-- true`，于是它的
结果取决于**机器上有没有 nsys**：没有 nsys ⇒ 走设备级降级（空 launch 表）到导出分支；有 nsys
⇒ nsys 先报 sqlite 里没有 `CUPTI_ACTIVITY_KIND_KERNEL` 表，走的是另一条分支（验收方在真机上
就红在这条）。现在拆成两条**机器无关**的用例：一条打桩成"没有 nsys"（覆盖 W-2 的
`没有可导出的 kernel`），一条打桩成"装了 nsys 但这次零 kernel"（覆盖它自己的
`没有采到任何 kernel 活动`）。两条产品分支都测到，且都不依赖环境。

### 包预算与前提门禁的三条不变量（第七轮 X-15 / X-17 / X-20 / X-22）

- **守卫判词的身份与结论四条同列保护级**（X-15）：`over_peak` 的 facts 装的是"这次超峰属于
  哪个 kernel"——它是那条结论的**身份**。旧实现在 `facts_secondary` 这一级就把它删光
  （常态 3.6 KB 包实测 `over_peak.facts=null`），而 refused 判词的 `next`/`metrics` 早已被清空
  ⇒ 包里**零归属**，正好退化成 V-6 要消灭的"把某个 kernel 的超峰当成整份报告的结论"。
  现在它与 `PRIMARY_FACT_IDS` 同列，要裁就一起到最后一档 `facts_primary` 才裁。
- **`VERDICT_KEEP_HARD` 从优先级集合大小推导**（X-17）：旧值硬编码 4，而优先级集合是 5
  （结论四条 + `over_peak`）⇒ **对抗顺序**下第五条（`truth_vs_substitute`）会整条消失
  （验收方 8400 场景里 3776 例；自然顺序打不中，因为 `over_peak` 排在第 13 位）。
  现在 `VERDICT_KEEP_HARD = len(PRIMARY_FACT_IDS) + len(GUARD_VERDICT_IDS)`，并有测试把
  "硬上限 ≥ 优先级集合"与"对抗顺序下一条都不许被挤掉"两件事都钉住。
- **M13 第一级的 `facts.top` 改成结构化列表**（X-20）：它曾经是拼接字符串，于是
  `FACT_STR_LIMIT=48` 会把它截断 —— 真实 C++ 模板实例化名（一个名字就 60+ 字）下 top-N 直接
  退化成 top-1，而夹具用短名所以永远绿。现在每项是 `{kernel, pct}` 对象且**名字不截断**
  （`to_dict()` 只截字符串 fact，嵌套结构原样带出）；渲染层逐项印，一屏契约按"列表里每一项的
  每个值都要出现"扩展（契约没放松，只是从标量扩到列表）。
- **前提检查成了逐判词的、并且落在 `agentrefs` 门禁里**（X-22）：新增
  `tools/premise_audit.py` —— 解析 `packet.py`，对每个 builder **传递展开**它调用的 helper，
  要求"这个判词能发出的前提 ⊆ `VERDICT_PREMISES[该判词]`"。旧实现只做"生成物 == 生成器输出"，
  外加 pytest 里一个**全局并集**检查，于是验收方的变异 A4（给 `dram_bytes` 挂
  `_environment_premises`：三个 id 都在标签表里）在 `agentrefs` 门禁 rc=0、116 条相关测试全绿。
  现在 `gen_agent_refs.py --check` 会红，并且有一条测试**复刻 A4** 证明它真的会红。

### CALIBRATION 散文门禁改成"全部产物字段"（第七轮 X-9，最后一个 P2）

- 旧门禁是一份**手写白名单**：`ts`、`measured_dram.{copy,read,write}_gbs`、
  `l2_read_peak_gbs`、matmul 的 `tflops`/`caliber_spread_pct`。而产物有 90 多个叶子，
  漏掉的那片（`clocks_during_measurement.*`、`measured_matmul_tflops.<dtype>` 的其它读法、
  `l2_info.*`、`bandwidth_sweep[*].*`…）里新旧值确实不同的有一大批 —— 散文手抄它们的旧值时
  **没有任何门禁会红**，而"公开数字要能追到产物"正是这个仓库的核心承诺。
- 现在**递归取全部标量叶子**（路径带出来，报文点名到 `clocks_during_measurement.sm_clock_mean`
  这一层），由"这个值有没有被散文引用"决定要不要管。两条防噪声规则是实测踩出来的：
  **数字至少 3 位**（`128`/`96`/`1.5` 会在正文里合法出现），**字符串必须带数字**
  （`reduced_precision_reduction_off`、`ieee` 这类是**协议标签**，散文引用它们讲的是"用哪种
  口径"，与产物版本无关 —— 第一版递归实现就是被它们报了两条假阳性）；同时把产物字符串
  **内部**的数字也登记为"现行产物也有"（`method` 里的 sweep 参数 `[4096, 8192, 16384]`）。
- 负样本测试同步加强：不只要求五类老字段能被抓，还要求
  `clocks_during_measurement.sm_clock_mean` 与 `measured_matmul_tflops.torch.bfloat16.tflops`
  这**两族**都有旧值注入并变红（并写清"值从没变过的字段产生不了负样本"这条前提），
  另加一条点名 X-9 例子的变异测试。

## 0.2.0

首次把"给 agent 用"当成一等公民。没有破坏性变更；决策包 schema 追加到 **v1.1**。

### 决策包（`--packet`）

- **v1.1（追加式，向后兼容）**：每条判词新增 `actionable`（能不能作为行动依据），
  顶层新增 `schema_revision`。`schema_version` 仍是 `1` —— 读 v1.0 包的代码依然能读 v1.1。
  读包的人应当用 `actionable` 判断可依据性；v1.0 的包没有这个字段，回退规则是
  「不是 `refused` 且前提全部成立」（实现见 `kernoscope.packet.verdict_is_actionable`）。
- 尺寸预算仍是 4096 B，硬约束：超预算时按固定顺序裁剪，并把**裁掉了什么**写进
  `cost.trimmed`（`cost.packet_bytes` 始终等于真实字节数）。
- 判词带前提（`premises[].observed/threshold/ok`）；前提不成立时该判词**必须**是
  `refused` 且带结构化 `refusal.code`，不允许以 `soft` 出现。

### 子命令开关一览（0.2.0 起就有，本轮补进文档 —— P5）

只列**接口层面**的：这些开关会出现在读者的命令与 CI 里，缺席就等于"消费者必须去读源码"。

| 命令 | 开关 | 作用 |
|---|---|---|
| `capabilities` | `--json` / `--brief` / `--probe` / `--out <f>` / `--quiet` | 参考层 JSON / 入口层（<1.2 KB，带下一层字节价格）/ 实测探测 / 写文件 / 静音 |
| `selftest` | `-o, --out <f>` / `--json` / `--size-mb` / `--duration` / `--runs` / `--dtype` / `--only` / `--cross-process` / `--quick` / `--strict` / `--quiet` | 纯 torch 的已知答案自检：写 `--ks-selfcheck` 用的文件（**不是** `--ks-calibration`，见 `PEAK_CAVEAT`）；`--cross-process` 把跨进程复测算进 MDD |
| `gate` | `--packet <f>` / `--report <f>` / `--require ID[=STATUS]` / `--require-actionable ID` / `--forbid-code CODE` / `--max-metric NAME=V` / `--min-metric NAME=V` / `--json` | **CI 门禁**：判词状态、可依据性、禁止的拒绝码、指标阈值；只读已产出的包与报告（不跑测量）。退出码 0=通过 1=不通过 2=用法错误；一条规则都没给时**失败**（门禁不许凭空绿灯） |
| `skill` | `--apply` / `--target <dir>` / `--client {claude,agents}` / `--json` / `--source <dir>` | 把 skill 装到 agent 的 skill 目录：默认只打印计划；备份（保留第一份）、原子写、写后回读校验、失败逆序回滚、内容相同即幂等 |
| `route` | `--list` / `--json` / `--probe` | **先路由再动手**：把一条问题（中文口语也行）转成「走哪条口径 + 命令 + 期望输出 + **会被拒的条件**」；只输出计划，不跑测量；匹配不上就说匹配不上（不许硬套） |
| `diff` | `--selfcheck <f>` / `--json` / `--fail-on {none,regressed,change}` / `--strict` / `--quiet` | 误差带对拍；**`--fail-on` 是唯一能拿去卡 CI 退出码的开关**（`regressed`/`change` 命中即非零退出） |

### 路由与顺序（A1）

- 新增子命令 **`kernoscope route "<问题>"`**：回答"我该走哪条口径、跑哪条命令、会看到什么、
  什么条件下会被拒"。路由表是**代码**（`route_cmd.ROUTES` + `capabilities.QUESTIONS`，
  `--json` 是给 agent 的真源），文档只是视图；匹配不上会明说"不要硬套"。
  口径决策规则（有 NVBit 走轨迹、否则走代理）跟着路由一起输出，避免 agent 自己猜。
- `route` **只输出计划，不跑任何测量**（有测试守着它的实现里不出现采集调用）。

### CI 门禁与 ncu 名字（C4/D4）

- 新增子命令 **`kernoscope gate`**：`--require <id>[=ok|soft|refused]`、
  `--require-actionable <id>`、`--forbid-code <code>`、`--max-metric NAME=VALUE`、
  `--min-metric NAME=VALUE`、`--json`。判据全落在**可编程的东西**上（状态、拒绝码、
  `actionable`、指标值+出处）；`n/a` 的指标一律算不通过（"没有数"不是"没超阈值"）；
  一条规则都没给时**失败**，因为门禁凭空通过是它最坏的失败模式。退出码 0/1/2。
- **D4① 名字是发现键**：`--metrics dram__bytes.sum` 这类 ncu 计数器名不再得到
  "指标不存在"，而是解析成替代口径（`est__dram_bytes_total`）并打印说明；`--strict`
  下仍然失败（我们确实给不出 ncu 名下那个数），但错误信息里会写明**替代口径叫什么**。
  映射表里标为"给不出"的（如动态 Tensor Core 利用率、warp stall 细分）说清为什么。
  原来的硬规则一秒没松：数值仍然只出现在 `ks__`/`est__`/`meas__` 名下。
- **D4② 兼容导出**：`--export-name-map <path>`（`.json`/`.csv`）写一份
  "ncu 名 → 我们给什么 / 为什么给不出"的对照表；它是**声明**，不是数据替换。
  映射表是代码（`ncu_names.NCU_DISCOVERY_MAP`），文档只是视图。

### 决策包：facts 分两级裁剪（真机回归）

- 真机上（一处 GEMM 负载 + 8 个拒绝码）包被裁到**全部判词的 `facts` 丢失**，于是人视图出现
  「最耗时的是 ?（None ms）」「成本账」这种把结论裁没了的输出。现在 `facts` 分两级：
  先裁次要判词，`kernel_time` / `cost_per_token` / `saving_estimate` / `truth_vs_substitute`
  这四条的 facts（"什么最贵、多少钱、先改哪里、真值偏差"的结论本身）最后才动；
  裁剪顺序写进 `cost.trimmed`（`facts_secondary` → `facts`）。
- `phrase_verdict` 增加兜底：facts 确实被裁掉时不再印 `?`/`None`，而是用仍在包里的
  `value`/`unit` 说人话。

### 处方：先改哪里、能省多少钱（C2）

- 新增 section **`Prescription`** 与判词 **`saving_estimate`**。头寸 = 本次达成值 vs
  **本机实测峰值**：保守端「把差距关掉一半」= u/(2−u)（u = 差距比例），乐观端 = u 且
  **明确标注为上界、不是预测**；`--ks-peak-*` 覆盖值或规格表**算不出**头寸
  （`calibration_caliber_mismatch`），因为「规格 vs 实测」的差距正是本工具的理由。
- 钱的规则：没有价格就没有钱（比例照给）；价格有但无法按 kernel 份额归因 →
  `premise_failed`（与"没给价格"处置不同）；per-token 节省额外要求被匹配 kernel 主导窗口
  （份额 ≥ 50%）——否则省下的是 GPU 时间，不是 token 成本。
- **重叠头寸不相加**：同 kernel 的多个头寸按最大者封顶，被丢弃的量写进 note
  （"全部相加会得到 X"），跨 kernel 同资源的相加会显式标注它们其实在争同一份资源。
- 一条都不可辩护时：`ranked == []` + 一条结构化拒绝（不是空输出）。

### 真值对拍与写 `.ncu-rep`（I1/I2）

- 新增 **`--ks-ncu-rep <file>`**（别名 `--from-ncu-report`）：用官方只读 API 导入一份
  `.ncu-rep` 作真值，指标以 **`ncu__` 前缀 + `exact`** 出现（与 ncu 同源，但不落计数器
  命名空间、不触发允许表），并逐条与我们的替代口径对拍 → `ks__truth_vs_substitute_pct`。
  判词 **`truth_vs_substitute`**：一条都对不上时也是 `refused` + 原因（沉默会被读成"对上了"）。
  有 `-o` 时另写 **`<base>.ncu-truth.json`**（可归档、可 diff 的真值对照产物）。
  读不了报新拒绝码 **`ncu_report_unreadable`** + 处置（缺的往往只是 PYTHONPATH）。
- **读 `.ncu-rep` 走子进程隔离**：真机实测在主进程里读会 Segmentation fault（rc=139）——
  主进程 import 了 torch，其 CUDA 运行时与 `_ncu_report` 同进程调 CUDA；同一文件单独读正常。
  段错误抓不住，所以隔离是唯一可靠做法（`ncu_rep.load_isolated`，子进程不 import torch）。
- 新增 **`--export-ncu-rep <path>`**：把替代口径写成一个合法的 `.ncu-rep`
  （官方 reader 可按名字读回值，`ncu --import` 能打开），**同时**写
  `<path>.ATTRIBUTION.txt` 旁路署名。三条门禁：文件内署名（provider/comment）为空即拒绝；
  前提不成立的指标（决策包里被拒判词引用的）不写入；过滤后一个值都不剩时**拒绝生成**。
- 写进去的名字**只允许自有命名空间**（`ks__`/`est__`/`meas__`）与"两边单位语义一致"的
  计数量白名单（`launch__registers_per_thread` 等）。**不写 `gpu__time_duration.sum`**：
  我们是 ms、ncu 是 ns，写进去会让读的人差 10⁶（R-51 同类错误在源头拦截）。

### 人读报告与阶段归因（I3/I8）

- 新增 **`--report-md <path>`**：按生态里通用的 per-kernel 报告结构写出 Markdown
  （`## 0` 采集设置 → `## 1` 头部数字**带 Source 列** + `**One-line read:**` → `## 2` 分维度
  （没有数据的维度也保留并写明拒绝码）→ `## 3` 诊断 → `## 4` 按影响排序的方向（**最多 5 条**，
  每条必须带 `**Evidence:**`）→ `## 5` 置信与限制（三段）→ `## 6` 复现）。
- 新增 **`--ks-nvtx-phases`**：从 `--ks-sqlite`/`--ks-nsys-rep` 的 NVTX 区间归因 kernel 时间
  （新 section `NvtxPhases`）。跨区间的 kernel 按**重叠部分**计入，嵌套区间各自计入
  （份额之和可 >100%，明说不是划分）；没有 NVTX 数据时那一节仍以 `n/a` + code 存在。

### skill 安装（I4）

- 新增子命令 **`kernoscope skill`**：`--apply`（默认只打印计划）/ `--target <dir>` /
  `--client {claude,agents}` / `--json` / `--source <dir>`。把 `SKILL.md` + `references/` +
  `scripts/` 装到 `~/.claude/skills/kernoscope`（或 `~/.agents/skills/kernoscope`）。
  备份 `<文件>.kernoscope.backup`（**已存在则保留第一份快照**）、原子写、**写后回读校验**、
  失败**逆序回滚**、内容相同即 `unchanged`（幂等）。

### 副作用账与 `--plan`（A2）

- 新增 **`--plan` / `--plan-json`**：只打印这次调用会做什么（起哪些进程、跑哪条命令、写哪些
  文件、读哪些输入、哪些是第三方软件、是否需要你点头），**不执行任何东西、不写任何文件**。
  `--plan-json` 里每条操作带 8 项 `effects` 与 4 项由 effects **纯推导**的 annotations。
- 新增 `effects.OPERATIONS`（9 个操作）+ `effects.effects_for()`：任何会起进程/写文件/联网的
  动作必须在表里登记，**查不到就抛 `UndeclaredEffect`**（副作用纪律进类型系统，不进文档）。
- **行为修正（由"--plan 声明 vs 实际观察"的对账测试抓出）**：
  ① 计划曾漏声明 doctor 的能力探测（`nvidia-smi --query-gpu` 与各工具的 `<tool> --version`）
  —— 连 `--ks-synthetic` 都会跑，现在登记为 `probe_env`；
  ② `-o out/rep.json`（父目录不存在）过去抛裸 `FileNotFoundError`，采集几分钟后白费，
  现在自动创建输出目录并记一行 note；
  ③ 计划里写的输出文件名与实际不一致（写 `out/rep.json` vs 实际 `out/rep.ksreport.json`），
  现在两处用同一套规则算。
- 测试用 PATH 上的记录桩做**声明 vs 观察**对账：计划说会起 nsys 就必须起、说这条运行不采集
  就不许出现 `nsys profile`、计划列出的文件必须真的出现。

### 单位成本（C1）

- 新增主命令行开关：`--ks-tokens <每步 token 数>`、`--ks-price-per-kwh <电价>`、
  `--ks-price-per-hour <租约价>`、`--ks-idle-power-w <W>`、`--ks-currency <标签>`。
  产出新 section **`CostLedger`**：`meas__window_energy_j` / `meas__window_power_mean_w`
  （实测）→ `ks__tokens_per_s`、`ks__j_per_token`、`ks__cost_per_hour`、`ks__cost_per_mtok`
  （及 `_energy` / `_rental` 两个载体分量，`derived`）→ 按 kernel 归因
  `est__kernel_energy_share_pct` / `est__kernel_energy_per_launch_j` / `est__kernel_cost_per_hour`
  （`estimated`，代理口径）。能耗口径与 DRAM 账**共用同一个积分窗口**，所以"窗口内字节"
  与"窗口内能耗"说的是同一段负载。
- 新增三个拒绝码（含 remedy）：`no_tokens`（缺每步 token 数）、`no_price`（缺价格）、
  `no_energy`（拿不到功率/能量读数）。**给一半输入只给一半结论**：缺 token 数时
  `ks__cost_per_hour` 照给、单位成本报 `n/a` + `no_tokens`；缺价格时能耗与 J/token 照给。
  窗口短于 0.5 s 用 `below_resolution` 拒绝给能耗（不按 TDP 顶替）。
- 决策包新增判词 **`cost_per_token`**（`VERDICT_IDS` 已含）：有数 → `ok`（facts 带
  tokens/s、J/token、按 kernel 归因）；缺输入 → `refused` + 对应 code + 处置（agent 据此
  知道该补哪个开关）。**代价：包在新判词加入后仍在 4096 B 预算内**（有序裁剪未变）。
- **判词的前提降级与 `over_peak` 传播改为在 `MAX_VERDICTS` 截断之前执行** ——
  以前被截断挤掉的 `over_peak` 会让带着"物理不可能"前提的 roofline 判词留在包里。
- `--strict` 覆盖成本账：给了 `--ks-price-*` 却算不出 `ks__cost_per_hour`、或给了
  `--ks-tokens` 却算不出 `ks__tokens_per_s` / `ks__cost_per_mtok` ⇒ exit 2。
- `kernoscope route "每百万 token 花多少钱"` 可路由到新问题 `cost_per_token`
  （`capabilities.QUESTIONS` 新增，tier 1：零标定、零计数器、零插桩）。

### 输出与案件目录（C5）

- 一屏（`--screen only`）改为**三段式**，段落标题固定可解析：
  （**更正**：早期条目写过"`-o` 时也落 `*.onepage.txt`"，但仓库里从来没有这个写入点 ——
  一屏只走 stdout/`--screen`；要文件请重定向或看决策包/报告。）
  `结论：` 一行 → `-- 发现（判词与前提）--` → `-- 路径（下一步，按挡住决策的程度排序）--`
  → `本次代价：`。「路径」段的排序是确定的：**先列被拒的判词**（它们挡住了决策），
  再列成立判词自带的提示，最后列未测到的组；同一 code/提示只出现一次。
- 新增**案件目录**（`-o <前缀>` 时默认写 `<前缀>.case/`，`--no-case` 关闭）：
  `manifest.json`（工具版本、协议哈希、**输入指纹**、参数、输出清单）、`env.json`、
  `journal.jsonl`（**追加式**：每次调用一条，含 rc/判词状态/拒绝码/包大小）、
  `next.md`（可直接交给下一个会话的"下一步"）、`README.md`。
  目的是让一次优化任务**跨会话可继续、可交接**；案件目录写入失败不影响主产物（只 warn）。

### 命令行

- 新增子命令：`capabilities`（`--json` / `--brief` / `--probe`）、`selftest`、`diff`。
- 新增选项：`--packet <path|->`、`--screen auto|only|off`、`--ks-selfcheck <file>`。
- `capabilities --brief` 是**入口层**（<1.2 KB，只有 id/code/计数，且带下一层的字节价格）；
  `capabilities --json` 是**参考层**（含解释、remedy、逐问 observed/threshold）。
- 缺依赖（如没有 `nsys`）不再是非零退出：主流程退到设备级，报告与决策包里带上
  `environment_refusals`，退出码 0；`--strict` 时才按失败处理。
- `--csv` / `--launch-csv` / `--packet -` 会自动关闭一屏输出，保证机器格式纯净。

### 拒绝（`n/a`）的表示

- 每个 `n/a` 都带闭集 `code` + `observed` + `threshold` + 可执行 `remedy`（含代价）。
  旧报告（没有 `refusal` 字段）回读时标 `legacy_report`，不猜原因。

### 未变

- 运行时依赖仍然为零；`.ncu-rep` 只读、从不写入；NVBit 不随包分发。
- `--strict` 的退出码约定（请求了不可得的数据 → 2）不变。
