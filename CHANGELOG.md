# 变更记录

本文件记录**接口层面**的变化（命令行、决策包 schema、输出字段），不逐条列内部重构。
判断标准：**如果一个外部消费者（脚本、agent、CI）需要改代码或改解析逻辑，就必须写在这里。**

## Unreleased

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
- 文档命令门禁：**全部**可检命令（当前 34 条）都交给**真实解析器**（7 个子命令各有自己的
  `build_parser()`，是 `--help`／门禁／测试共用的唯一真源；续行、`sudo`、`$( )`、
  `python -m kernoscope`、`xargs … kernoscope` 都覆盖，且**不许静默丢弃**）；
  其中**读者面 9 条**逐字执行并断言结论（比例 9/34，不是"每条都断言"）。

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
- 依赖说明（`tools/probes/nvbit_memtrace/README.md` + `collect.sh`）改成**版本×构建矩阵**：
  v1.8.1 走 CMake（本仓库真机验证）、v1.7.7.3 走老 Makefile（第四轮真机验证），
  并写明"不要 make 整个 tools / 架构变量用裸数字 / 必须 NO_EAGER_LOAD=0"。
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
| `route` | `--list` / `--json` / `--probe` | **先路由再动手**：把一条问题（中文口语也行）转成「走哪条口径 + 命令 + 期望输出 + **会被拒的条件**」；只输出计划，不跑测量；匹配不上就说匹配不上（不许硬套） |
| `diff` | `--selfcheck <f>` / `--json` / `--fail-on {none,regressed,change}` / `--strict` / `--quiet` | 误差带对拍；**`--fail-on` 是唯一能拿去卡 CI 退出码的开关**（`regressed`/`change` 命中即非零退出） |

### 路由与顺序（A1）

- 新增子命令 **`kernoscope route "<问题>"`**：回答"我该走哪条口径、跑哪条命令、会看到什么、
  什么条件下会被拒"。路由表是**代码**（`route_cmd.ROUTES` + `capabilities.QUESTIONS`，
  `--json` 是给 agent 的真源），文档只是视图；匹配不上会明说"不要硬套"。
  口径决策规则（有 NVBit 走轨迹、否则走代理）跟着路由一起输出，避免 agent 自己猜。
- `route` **只输出计划，不跑任何测量**（有测试守着它的实现里不出现采集调用）。

### 输出与案件目录（C5）

- 一屏（`--screen only`，`-o` 时也落 `*.onepage.txt`）改为**三段式**，段落标题固定可解析：
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
