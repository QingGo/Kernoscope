# 外部 agent 冷启动评测（G4 的证据）

这个目录回答的问题是：**一个此前不了解 Kernoscope 的 agent，只看它能看到的材料（README 摘要、
能力清单、决策包），能不能做出正确决策 —— 以及会不会照着站不住的判词去改代码。**

评测台本身在 [`../README.md`](../README.md)；任务与真值在 [`../tasks/`](../tasks/)。
本文件是**一次真实外部评测的记录**：三次独立的冷启动（c / d / e），每次 5 个任务，全部离线、无 GPU。

## 1 结果

| 轮次 | agent | 决策正确率 | **误用率** | acted_on 总数 | 误用事件 | 动作层禁止命中 | 材料版本 |
|---|---|---|---|---|---|---|---|
| c | 外部 agent（临时叫来的性能工程师） | 1.00 (5/5) | **0.00** | 14 | 0 | 1（见 §3.3） | 修漏洞前 |
| d | 外部 agent（结构化判读程序） | 0.80 (4/5) | **0.15** | 13 | **2** | 2 | 修漏洞前 |
| e | 外部 agent（性能工程师） | 1.00 (5/5) | **0.00** | 11 | 0 | 0 | 修漏洞后 |

原始记分板：`c/scoreboard.json`、`d/scoreboard.json`、`e/scoreboard.json`；逐任务决策：
`c/<task>/decision.json`、`e/<task>/decision.json`（d 的逐任务决策未入库，因为它触发的那次修复
让材料变了；它的两个误用事件在 §3 里逐条写明）。

**误用率定义**（与评测台一致）：`Σ(acted_on 里不可作为行动依据的判词数) ÷ Σ(acted_on 数)`，
其中"不可作为依据" = 判词已被尺寸预算裁掉 ∪ `status=refused` ∪ 任一前提 `ok=false` ∪ 真值
`forbidden_verdicts`。动作文本命中 `forbidden_actions` 记入辅助指标 `unsafe_rate`，不混进误用率。

## 2 它们看到的是什么

每个任务一个"干净房间"：只有该任务目录下的 `QUESTION.md`（场景 + decision.json 契约）、
`packet.json`（`kernoscope --packet` 的真实输出）、`onepage.txt`（同一份判词的人视图）、
`capabilities.json`、个别任务的 csv。**真值结构上不可能进入材料**（评测台有断言）。
被测 agent 被明确禁止读仓库源码与任务定义 —— 这三轮里它们都只读了材料。

## 3 这一轮外部评测抓到了什么

### 3.1 真漏洞：前提不成立，`status` 却还是 `soft`（d 踩中）

任务 02 的场景是一个 0.9 s 的窗口里顺序跑了 4 个 kernel，用户只关心其中一个 ——
它只占窗口内 kernel 时间的 **41%（门槛 95%）**。修漏洞前的包里：

```
dram_bytes_per_launch   status=soft   premises=[ok=true, ok=false(41% < 95%)]
ledger_verdict          status=soft   与账本相差 +146.9%
```

d 把这两条都当成了依据（`acted_on=[kernel_time, ledger_verdict]`），结论是"按 +146.9% 去改访存"。
**这不是 agent 的错**：判词自己标着 `soft`（可用），前提那一行虽然写着 `ok=false`，但契约里
"前提不成立"与"status"是两个信号，只读 status 的 agent 必然踩中。

修复（已入库）：

* `build()` 统一降级：**前提不成立 ⇒ `status=refused`**，并按前提给出闭集 code
  （`share_too_low` / `coverage_too_low` / …），原值降级为 `facts.uncitable_value`（历史可见、明确不可引用）；
* `to_dict()` 断言：任何非 refused 的判词只要带不成立的前提就抛错，堵住旁路；
* 账本裁决**继承它依据的口径**：per-launch 数字不可归因 ⇒ 账本误差百分比也不可引用；
* 新增 `premise_failed` code（闭集 + remedy + 静态标题）。

修复后同一个任务（e 轮）的包变成：

```
dram_bytes_per_launch   status=refused  refusal.code=share_too_low
ledger_verdict          status=refused  refusal.code=share_too_low
```

三个 agent（c、d、e）在这一点上的判断本来就都对 —— 差别在于**修复前只有一个信号**（status）与
事实矛盾，修复后两个信号一致。

### 3.2 评分公平性：用词不同不等于判断错

第一轮里 c 的两条判断理由完全正确，却因为分类标签用了 `unattributable` / `below_resolution`
而不是真值词表里的 `window_unattributable` / `not_dram_bound` 被判错。真值里的词是**我们的**，
冷启动 agent 没有理由猜中。所以：
* 真值增加 `classification_aliases`（放在生成器 `make_tasks.py` 里，避免重跑生成器冲掉）；
* 评分改为"别名内视为同一判断"。

### 3.3 行动层匹配：否定语境不算命中

**修的是什么**：`_mentions()` 以前是纯子串匹配，于是「**不要**反过来去优化计算」这句
**否定式**建议会被算成命中了禁止动作「优化计算」。现在它跳过否定/不确定语境
（`不要/别/无需/不是/是否/avoid/whether…`）。这一项只影响辅助指标 `unsafe_rate`。

**别把这句读成"c 轮残留的那 1 个命中是这个 bug"**：c 轮看板里 03 号任务记的事件是
`next_action 命中禁止动作：'compute'`，来源是**正面**措辞「再决定是攻 compute 还是 mov」——
那是真命中（那条建议确实提到了 compute 方向），与否定式误报无关。上文的归因讲的是
**修之前**那一版评分器的行为；修复是真的，归因只对那个版本成立。

### 3.4 真值过严：可用判词当作"排除依据"不该算误用（d / e 都这么做）

任务 04 是 compute-bound GEMM，d 与 e 都把 `dram_bytes`（`soft`、前提全成立）放进 `acted_on`，
用来**排除**访存方向（"显存忙碌 8% ⇒ 别动访存"），然后沿 compute 侧行动。这被真值的
`forbidden_verdicts` 判成误用。

判断：**这是真值过严，不是 agent 错。** 拿一份有效的测量去排除一个方向是正确工程做法；
该被禁止的是"在 compute bound 时提议访存优化"，那由 `forbidden_actions` 覆盖。
所以 04 的 `forbidden_verdicts` 改为空，保留 `forbidden_actions`。
（记录在此以便复核：d 的 2 个误用事件里，1 个来自 §3.1 的真漏洞、1 个来自这条过严真值。）

## 4 结论与边界

* **G4 判据**「陌生 agent 只读 README + capabilities 能规划正确调用序列；误用率 0；≥2 个不同 agent 通过」
  —— 在修掉 §3.1 之后，两个独立冷启动 agent（c、e）均为 **决策正确率 1.00 / 误用率 0.00**。
* 这是**三个 agent 的五任务小样本**，不是统计结论。它证明的是"这几条失败模式已被堵住"，
  不是"没有失败模式"。真值、任务、评分口径都在本目录里可复核、可反驳。
* 评测台**不改产品代码**：它只 import `kernoscope.packet` / `kernoscope.refusals` 校验包结构，
  所以"包结构合法"这条断言的可信来源与产品本身是同一份代码。
* 复现：`python eval/harness.py --agent-cmd "<你的 agent 命令>"`；
  外部 agent 不是本地命令时用 `--prepare-only` 落地材料、`--score-only` 打分（本文件的 c/d/e 就是这样跑的）。
