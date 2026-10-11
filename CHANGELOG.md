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
- 裁剪顺序的注释与实现对齐；`write_packet` 不再额外追加换行（落盘字节数 == `packet_bytes`）。

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
- `promise`：文档命令门禁改为把每条文档命令交给**真实解析器**（覆盖全部 7 个子命令、
  续行、`sudo`、`$( )`），并登记逐字执行的文档命令。

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
