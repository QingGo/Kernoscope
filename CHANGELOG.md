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
  `skill --apply --source <不存在>` 报错而不是 rc=0 说"已经是目标状态"。
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
- 文档命令门禁：**全部**可检命令（当前 59 条）都交给**真实解析器**（10 个子命令各有自己的
  `build_parser()`，是 `--help`／门禁／测试共用的唯一真源；续行、`sudo`、`$( )`、
  `python -m kernoscope`、`xargs … kernoscope` 都覆盖，且**不许静默丢弃**）；
  其中**读者面 11 条**逐字执行并断言结论（比例 11/59，不是"每条都断言"）。
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
