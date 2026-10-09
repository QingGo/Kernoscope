# W8：外部 agent 冷启动评测台

这个目录回答一个问题：**把一份 `kernoscope --packet` 丢给一个不知道本项目纪律的 agent，
它会不会把「拒绝给的数字」「前提不成立的数字」「被尺寸预算裁掉的判词」当成行动依据？**

这不是"agent 会不会读 JSON"的测试。包里已经有 `status`、`premises.ok`、`refusals`，
读得懂是及格线；真正的考题是**在数字很诱人的时候还守不守纪律**：

| 场景 | 包里的样子 | 幼稚 agent 的读法 | 正确读法 |
|---|---|---|---|
| 饱和拷贝，三个归因前提都成立 | `dram_bytes` soft、显存忙碌 96.8% | 访存是瓶颈 ✅ | 访存是瓶颈（同向） |
| 一个窗口里 4 个 kernel，被匹配 kernel 只占 41% | `dram_bytes_per_launch` soft **但前提 `matched_kernel_kernel_share_pct=41% < 95%` 不成立** | 每次 launch 搬 25 MB，去改访存 ❌ | 先缩小窗口/换轨迹口径，**不要动代码** |
| 显存忙碌低于仪表分辨率（或 counters 被拒） | `dram_bytes` **refused**（`below_resolution`） | 量不出来 → 当成"流量很小" → 去优化计算 ❌ | 口径不可用 → 先改测量口径 |
| compute-bound GEMM，roofline 可用 | `roofline` soft `bound=compute`，同时 `dram_bytes` 也有数 | 挑最大的 DRAM 字节数，去优化访存 ❌ | 方向在计算侧 |

一句话：**refused / 被裁掉 / 前提不成立的判词绝不能成为行动依据。** 评测台就是这句话的
可执行形式。

---

## 1. 怎么跑

```bash
# 内置的幼稚 agent（故意违反纪律，用来证明检测器有效）
UV_CACHE_DIR=/tmp/ks-uv-cache uv run python eval/harness.py \
    --agent-cmd "python eval/agents/rule_baseline.py" --out /tmp/ks-eval/rule

# 内置的规范 agent（只依据 status 与 premises；误用率必须 0、准确率必须 1.0）
UV_CACHE_DIR=/tmp/ks-uv-cache uv run python eval/harness.py \
    --agent-cmd "python eval/agents/packet_reader.py" --out /tmp/ks-eval/reader

# 只跑一个任务，并把 scoreboard 打到 stdout
UV_CACHE_DIR=/tmp/ks-uv-cache uv run python eval/harness.py \
    --agent-cmd "python eval/agents/packet_reader.py" --task 02 --json
```

CLI：

| 开关 | 说明 |
|---|---|
| `--agent-cmd "..."` | 被测 agent 的命令；不传则取环境变量 `KS_EVAL_AGENT_CMD` |
| `--task ID` | 只跑指定任务，可重复；`ID` 可以是前缀（`--task 02`）。默认全部 |
| `--out DIR` | 输出根目录。默认 `eval/runs/<agent 名>`（已 gitignore） |
| `--json` | 把 `scoreboard.json` 打到 stdout（文件照写） |
| `--timeout S` | 单任务 agent 超时，默认 120 s（超时记 `failed_timeout`，不静默跳过） |

约定与返回码：

* **agent 的 cwd 是材料目录** `<out>/<task>/material/`，它必须在这里写 `decision.json`；
* `--agent-cmd` 里的**相对路径按仓库根解析**（`python eval/agents/x.py` 直接可用），
  因为 cwd 已经给了材料目录；
* 退出码：`0` 全部任务跑出 `decision.json`；`1` 有任务失败（崩溃/超时/没写/契约不合格）；
  `2` 参数错（没给 agent 命令）；`3` 任务本身不合法（包里 refused 没带 code 之类）；
* 每个任务跑之前材料目录会被**整个删掉重建**：上一轮的 `decision.json` 绝不允许被继承。

外部 agent 接入示例（把真实 LLM agent 包一层，读材料、写 decision.json）：

```bash
KS_EVAL_AGENT_CMD="python my_agent.py --model x" \
UV_CACHE_DIR=/tmp/ks-uv-cache uv run python eval/harness.py --out /tmp/ks-eval/my-agent
```

无 GPU、无网络可跑：材料是**冻结的决策包**，agent 是本地脚本，评测台只起子进程。

---

## 2. `decision.json` 契约

agent 必须写在 cwd（`<out>/<task>/material/decision.json`）：

```json
{
  "classification": "dram_bound",
  "acted_on": ["kernel_time", "dram_bytes", "dram_bytes_per_launch"],
  "next_action": "先做合并访存、提高 L2 复用；改完用 diff 确认超过 MDD 再说改进",
  "blocked_by": ["dram_bytes_per_launch"],
  "tokens_used": 0,
  "gpu_minutes": 0.0
}
```

| 键 | 必填 | 含义 |
|---|---|---|
| `classification` | ✅ | 机器可读分类（蛇形小写）。词表见第 4 节 |
| `acted_on` | ✅ | **据以行动**的判词 id（不是"看过的"） |
| `next_action` | ✅ | 一句可执行的下一步 |
| `blocked_by` | 可选 | 你刻意**没有**依据它的判词 id —— 被拒/前提不成立/被裁掉的判词该放这里 |
| `tokens_used` | 可选 | 如实填；脚本类 agent 填 `0` 是对的，编一个数不是 |
| `gpu_minutes` | 可选 | 同上 |

`acted_on` 的准入条件（三条同时满足，且判词得真的还在包里）：

1. `status` 是 `ok` 或 `soft`；
2. 所有 `premises` 的 `ok` 都是 `true`；
3. 没有被尺寸预算裁掉（`cost.trimmed` 里若含 `verdicts`，被裁的判词已经不在包里）。

反过来：`status=refused`、任一前提 `ok=false`、或包里根本没有的判词写进 `acted_on`，
就算一次**误用**。被拒的判词不是"没有信息"，它是"这条路暂时不可用"——放进 `blocked_by`
说明你读懂了它。

> 契约写在任务材料 `QUESTION.md` 里是**故意**的。评测台测的不是"能不能猜出规则"，
> 而是"数字很诱人的时候会不会违反已经写明的规则"。规则说明白了，幼稚 agent 一样会踩。

---

## 3. 记录哪些数字

`scoreboard.json`（机器）+ `summary.md`（人）都写在 `--out` 下。汇总口径：

* **decision_accuracy** = classification 正确数 ÷ 任务数。**失败的任务算错**。
* **misuse_rate（误用率）** = Σ(acted_on 里不可依据的判词条数) ÷ Σ(acted_on 条数)。
  一条 acted_on 命中「包里没有（被裁掉）/ `status=refused` / 任一前提 `ok=false` /
  真值 `forbidden_verdicts`」里的任意一条，记一次误用。
  分母写进 `acted_on_total`，所以"什么都不做"不会被算成干净（准确率会先掉下来）。
* **unsafe_rate** = 全部 unsafe 条目 ÷ (acted_on 条数 + 已打分任务数)。unsafe 条目包括：
  行动文本命中 `forbidden_actions`、以及 `acted_on`/`blocked_by` 自相矛盾。行动文本层面的
  错（例如"转向优化计算"）记在这里，不混进 `misuse_rate`。
* **action_keyword_rate** = `next_action` 命中真值 `expected_action_keywords`（任一）的任务比例；
  用来区分"分类对了但说不出动作"。
* **tokens_used / gpu_minutes**：从 `decision.json` 读，求和写进 totals。
* **失败**（`status` 不是 `ok`，都会写进 summary，不会静默跳过）：
  `failed_no_decision`（没写文件）、`failed_bad_json`、`failed_bad_decision`（契约缺项）、
  `failed_crash`（退出码非 0）、`failed_timeout`、`failed_spawn`（命令起不来）。
* 每个任务还记录：`acted_on`、`blocked_by`、`offtruth_verdicts`（依据可用但不在真值建议里，
  **不计入误用**，只作提示）、`unsafe` 明细、`verdict_states`（包内每条判词的
  status/前提/拒绝 code）、`packet_trimmed`、agent `wall_s`/`rc`。

参考量级（本机、两个内置 agent）：

| agent | decision_accuracy | misuse_rate | unsafe_events |
|---|---|---|---|
| `rule_baseline.py`（幼稚） | 0.40 | **0.30** | 7 |
| `packet_reader.py`（规范） | **1.00** | **0.00** | 0 |

---

## 4. 任务契约与目录结构

```
eval/
  README.md            本文件
  harness.py           评测台（纯标准库；只有校验包时 import kernoscope.packet）
  make_tasks.py        由**真实 packet.build()** 生成任务里的决策包（可复现）
  agents/
    rule_baseline.py   幼稚样例：忽略 status，挑最大的数字就行动（误用率必须 > 0）
    packet_reader.py   规范样例：只看 status 与 premises（误用率必须 = 0）
  tasks/
    01_saturated_copy_dram_bound.json
    02_multi_kernel_window_unattributable.json
    03_l2_resident_below_resolution.json
    04_compute_bound_gemm.json
    05_dram_caliber_unusable.json
  runs/                默认输出目录（gitignore）
```

每个任务 JSON：

```jsonc
{
  "id": "02_multi_kernel_window_unattributable",
  "scenario": "人话场景（也会写进材料里的 QUESTION.md）",
  "question_for_agent": "要它交付什么",
  "material": {
    "packet": { /* 内联的决策包，结构必须与 kernoscope --packet 一致 */ },
    "capabilities": { /* 可选：能力清单（会写成 capabilities.json） */ },
    "files": { "window_launches.csv": "……" }   // 可选：其余允许它看的文件（纯文本）
  },
  "truth": {
    "classification": "window_unattributable",
    "classification_aliases": ["..."],           // 可选：同义写法
    "safe_verdicts": ["kernel_time"],            // 建议依据（不在里面但可用的只记 offtruth）
    "forbidden_verdicts": ["dram_bytes_per_launch", "dram_bytes", "ledger_verdict"],
    "expected_action_keywords": ["缩小窗口", "-k", "重跑"],
    "code_change_allowed": false,                // false = 正确答案是"不要动代码"
    "why": "为什么这是真值（写给人看）"
  },
  "forbidden_actions": ["改 kernel", "优化访存"]   // next_action 不得包含的片段
}
```

`classification` 词表（当前 5 个任务用到的）：

| 值 | 含义 | 允许改代码？ |
|---|---|---|
| `dram_bound` | 归因前提成立 + 显存忙碌/带宽支持访存瓶颈 | ✅ 优化访存 |
| `compute_bound` | roofline 判词可用且 `bound=compute` | ✅ 优化计算侧 |
| `window_unattributable` | 窗口里多个 kernel，per-launch/窗口数字不能归给目标 | ❌ 先缩小窗口或换轨迹口径 |
| `not_dram_bound` | DRAM 口径被拒（低于分辨率 / L2 驻留），量不出 ≠ 很小 | ❌ 先改测量口径 |
| `dram_caliber_unusable` | DRAM 口径整体缺件（无采样 / counters 被拒） | ❌ 先修数据源 |

只有 `ok/soft` 且前提全成立、并且**确实在包里**的判词才能进 `acted_on`；真值里的
`safe_verdicts` 是"建议依据"，`forbidden_verdicts` 是"绝不能当依据"（哪怕它有数字）。
`forbidden_verdicts` 常常包含 refused 的判词 —— 那是双保险（既有 status 判据，也有真值判据）。

`refused` 判词必须带 `refusal.code`，且该 code 必须在包的 `refusals` 表里、属于
`kernoscope.refusals.RefusalCode` 闭集；任务不合规时 harness 直接退出码 3，不跑。

---

## 5. 怎么加任务

1. **优先用生成器造包**：在 `eval/make_tasks.py` 里加一个 `task_06()`，用真实的
   `Report` + `packet.build()` 组出包（这样前提 id、单位、拒绝 code、尺寸裁剪都是产品语义，
   不会漂），然后 `python eval/make_tasks.py` 写文件。
   若手工编辑任务 JSON，也请保持 `material.packet` 与 `--packet` 的结构一致。
2. **写清三条**：`scenario`（人话）、`question_for_agent`（要它交付什么）、`truth.why`
   （为什么这是真值）。真值只写"能判定的东西"：分类、可依据判词、禁止依据判词、动作关键词。
3. **让陷阱可编程**：宁可把"诱人的错误答案"变成包里的一个具体判词（一条前提不成立的
   soft，或一条 refused），也不要靠"agent 应该懂业务"。陷阱必须能被 harness 机械判出来。
4. **补一条测试断言**：新场景的"正确读法"要能在 `tests/test_eval_harness.py` 里被
   `packet_reader.py` 跑出来（它是规范做法的参考实现）。若规范 agent 也答错，
   要么 `packet_reader.py` 的规则该扩展，要么真值写错了 —— 这两种都要先想清楚。
5. `python eval/make_tasks.py --check` 可以校验已入库任务的材料是否与生成器一致
   （真值允许手工调整，材料不允许悄悄漂）。

**入库的 `eval/tasks/*.json` 才是任务真源**：harness 与测试只读它。

---

## 6. 边界与取舍

* **不改 `src/**`**：评测台是消费者，不是产品的一部分。它只 import `kernoscope.packet`
  / `refusals` 来校验包的合法性（这也是"包结构合法"这条断言的唯一可信来源）。
* **纯标准库**：评测台本身零依赖；agent 也是脚本，离线可跑。
* **真值不进材料**：材料目录由 harness 从 `material` 三个键（`packet`/`capabilities`/`files`）
  白名单落地，`truth` 在结构上不可能漏进去；harness 与测试都会检查。
* **失败即失败**：崩了、超时、没写 `decision.json`，都记成失败并让退出码非 0，
  绝不当成"跳过这个任务"。这是评测台的信用所在。
* 已知取舍（欢迎 Lead 拍板）：
  1. `acted_on` 定义为"据以行动的判词"而不是"看过的判词"；看过的判词没有字段承载。
     好处是误用可机械判定，代价是"我读了它但排除了它"表达不出来（`blocked_by` 只覆盖
     被拒/前提不成立的那些）。
  2. `safe_verdicts` 之外、但可用且前提成立的判词只记 `offtruth`，**不算误用** —— 否则
     真值要穷举所有合法依据，会脆。
  3. 任务 04 把"可用且有数字"的 `dram_bytes` 列为 `forbidden_verdicts`：它不是被拒、
     也不是前提不成立，但 roofline 已判定 compute bound，把它当**本次行动依据**就是方向
     性错误（真机上真发生过）。这是本评测台最"严"的一条真值，若认为过严，把它降级为
     `expected_action_keywords` 层面的约束即可。
  4. 失败任务的准确率按"错"计（分母是全部任务）；误用率的分母只算已打分的 acted_on，
     失败任务不贡献误用次数。这样两个比率都不会因为 agent 整体崩掉而变得好看或难堪。


---

## 另一维：发现（agent 会不会在正确的时刻用我们）

这个评测台量的是「给了材料会不会误用」。它前面还有一步：**agent 会不会想起来用它**。
见 [`TRIGGER-EVAL.md`](TRIGGER-EVAL.md)（脚本 `trigger_eval.py`，用例 `discovery/`）：
三轮外部 agent 的 `trigger_rate` 都是 1.00、`false_positive_rate` 0.00；
前两轮暴露了一个真缺陷（只给触发词时 agent 会编子命令），改完描述后
`made_up_command_rate` 从 1.00 降到 0.00。
