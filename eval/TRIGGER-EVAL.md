# 发现维度评测（trigger）：agent 会不会在正确的时刻用我们

W8 的评测台（见 [`EXTERNAL-EVAL.md`](EXTERNAL-EVAL.md)）量的是"给了材料会不会**误用**"；
这一份量的是它前面那一步：**给一段真实诉求 + 一份工具目录，agent 会不会选择 Kernoscope
（以及有没有选错的时候）**。评测脚本是 [`trigger_eval.py`](trigger_eval.py)，用例在
[`discovery/`](discovery/)。

为什么它值得单独一轮：skill 的 `description` 是唯一常驻上下文的触发词（约 120 个词），
它决定采纳率里的"可发现性"。**它属于产品表面，不是文案** —— 所以要有数，而且要能量到改动。

## 1 三轮结果

工具目录里的 `kernoscope` 条目 = `skills/kernoscope/SKILL.md` 的 frontmatter `description`
（评测脚本现读，不抄一份：调文案才等于调产品）。被测 agent 只看到 `TASK.md` + `CATALOG.md`。

| 轮次 | 被测 agent | trigger_rate | false_positive_rate | entry_layer_rate | **made_up_command_rate** |
|---|---|---|---|---|---|
| A | 外部 agent（性能工程师） | 1.00 | 0.00 | 0.00 | **1.00** |
| B | 外部 agent（自动化运维） | 1.00 | 0.00 | 0.00 | **0.50** |
| C（改描述后复测） | 外部 agent（性能工程师） | 1.00 | 0.00 | 1.00 | **0.00** |

- `trigger_rate` = 应触发时真的选了 kernoscope 的比例（用例 01/02）
- `false_positive_rate` = 不该用时选了我们（或把性能工具当答案）的比例（用例 03/04/05）
- `entry_layer_rate` = 触发时第一条命令走入口层（`capabilities --brief` / `--help`）的比例
- `made_up_command_rate` = 触发时第一条命令**是编的**（写了不存在的子命令）的比例

原始记分板与逐用例 `choice.json`：`../evidence/trigger-{a,b,c}/`。

## 2 抓到的两件事

### 2.1 触发没问题（三轮都是 1.00 / 0.00）

五个用例构成三个判断陷阱，三轮的外部 agent 全部判对：

| 用例 | 正确判断 | 陷阱 |
|---|---|---|
| `01_ncu_perm_denied` | kernoscope | ncu 被 `ERR_NVGPUCTRPERM` 拒绝 —— 描述里点名了这个报错 |
| `02_regression_compare` | kernoscope | 要的是"真改进 vs 噪声"，不是"谁快" |
| `03_precise_counters_available` | **ncu（不要用我们）** | 有权限、要精确计数器；我们的描述写了"这时不要用它" |
| `04_cpu_dataloader_hotspot` | **py-spy（不要用我们）** | CPU 侧热点，kernel 级证据帮不上忙 |
| `05_loss_not_decreasing` | **不用工具** | 收敛问题不是性能问题 |

用例 03 尤其重要：它是**反向陷阱**。三轮 agent 都没有因为"主推工具"而误选自己 ——
说明"什么时候不要用它"这句话在触发词里是有用的，不是免责声明。

### 2.2 真缺陷：`description` 只够触发，不够调用（A/B 两轮都编了命令）

A 轮与 B 轮给的第一条命令：

```
kernoscope profile -- python train.py
kernoscope compare base_run vec_run
kernoscope compare --baseline a.nsys-rep --candidate b.nsys-rep --report out.json
```

`profile` 与 `compare` **都不存在**（真名是 `--packet -` 与 `diff`）。两位 agent 都诚实
报告了"目录没写怎么调用，这是我按形态推断的"。这不是 agent 的错，是触发面的缺口：

> 只给一句 `description` 时，agent 会**凭印象拼子命令** —— 而"编接口"正是我们最不想要的
> 失败模式（它会让用户以为工具坏了，或者更糟：照着一条不存在的命令去写脚本）。

### 2.3 修法与复测

在 `description` 末尾加了**一句入口命令**（不是把命令塞满，只给四步骨架）：

```
入口命令：kernoscope capabilities --brief → kernoscope --packet - -- <你的命令> →
kernoscope selftest / kernoscope diff。不要凭印象拼子命令（先看 kernoscope --help）。
```

C 轮复测（同一批用例、只给这条改过的 `description`）：`made_up_command_rate` 从 **1.00 → 0.00**，
第一条命令变成 `kernoscope capabilities --brief`（入口层）与 `kernoscope --help`（不知道参数时
先看帮助 —— 这正是那句话教它的行为），`trigger_rate` 与 `false_positive_rate` 不变。

代价：入口层（AGENTS.md + llms.txt + SKILL.md frontmatter）从 3.8 KB 涨到 **3.85 KB**，
仍在 4 KB 预算内（有测试守）。

## 3 边界与限制（必须一起读）

1. **材料只给触发面**：真实 skill 运行时，触发之后正文（SKILL.md 的"三步闭环"）会被加载，
   那时命令是明确的。所以 §2.2 量的是一个**更严格**的场景（只看到 description）。
   这一轮的结果说明：即便在只给触发面的情况下，加一句入口命令也能消除编造 —— 属于净收益。
2. **不是统计结论**：三轮 × 五用例，样本很小；它证明的是"这几条失败模式被堵住了"。
3. **未做真机执行**：被测 agent 所在环境没有装 `kernoscope`/`ncu`/`nsys`，`first_command`
   是纸面判断；命令的**可执行性**由别的测试守（`tests/test_documented_commands.py`
   校验文档里的命令真的存在）。
4. **其他工具的目录条目比我们短**：这是刻意的（真实的工具列表就是"名字 + 一句话"），
   而我们的条目就是 skill 的 `description`。用例 03 的 agent 反馈说 ncu 那条没给调用示例；
   这是**用例侧**的公平性问题，不影响对我们的判定（那一项不以我们为答案）。

## 4 复现

```bash
# 只落地材料（干净房间：truth 绝不落地）
python eval/trigger_eval.py --prepare-only --out /tmp/tr

# 外部 agent 不是本地命令时：让它写 <case>/choice.json，再打分
python eval/trigger_eval.py --score-only --out /tmp/tr --json

# 本地命令形式的 agent
python eval/trigger_eval.py --agent-cmd "python my_agent.py" --out /tmp/tr
```

改 `description` 之后**必须**重跑一轮：这个指标就是用来抓"听起来更好但触发变差/开始编命令"的。
