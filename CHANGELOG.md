# 变更记录

本文件记录**接口层面**的变化（命令行、决策包 schema、输出字段），不逐条列内部重构。
判断标准：**如果一个外部消费者（脚本、agent、CI）需要改代码或改解析逻辑，就必须写在这里。**

## Unreleased

（下一版的改动写在这里。小改动只升末位：0.2.1 → 0.2.2 → …；只有 schema 变更或破坏性接口变更才动中位。）

## 0.2.1 —— 2026-10-12（补丁级：真机验证与修复）

**已发布（PyPI）**：由 tag `v0.2.1` 触发 CI 以 Trusted Publisher（OIDC，零密钥）上传，
并**首次带 attestation（PEP 740）**（0.2.0 用的是 `uv publish`，那条路不生成 attestation）。
本版登记一份**由 0.2.1 在 4090 上产生**的 `bench/artifacts/` 产物（经 `bench/prune_artifacts.py`
裁到 KB–MB 级）。内容：处方接进 over_peak 传播链、NVBit occupancy 探针真机结论（标签保留）、
demo（五幕，含与 ncu 的 CLI/产物兼容）、以及一批只在真机才暴露的一屏文案与判词修复。

## 0.2.0

**已发布（PyPI）**：`2026-10-11` 由 tag `v0.2.0` 触发 CI 以 **Trusted Publisher（OIDC，零密钥）**
上传；wheel 518 784 B / sdist 1 800 015 B；发布后验证 `bash bench/release-dry-run.sh --from-pypi`
通过（隔离安装 + 首用三连 + 预算）。**该版本无 attestation**（用 `uv publish` 发的；0.2.1 起改
`pypa/gh-action-pypi-publish`）。

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
