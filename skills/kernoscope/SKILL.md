---
name: kernoscope
description: >-
  在受限/多租户 GPU 上为 CUDA kernel 优化取证：哪个 kernel 慢、是不是 DRAM bound、离 roofline
  多远、occupancy 上界与实际达到值、发射槽上界与瓶颈方向，每个数字都带出处与误差带，并能对两次
  运行做「真改进 / 真退化 / 分不出来」的判定，并把结果折算成**每百万 token 的成本**与
  按影响排序的优化方向。当 ncu 报 ERR_NVGPUCTRPERM、只有 nsys 的
  kernel 时长、或需要可复核的证据、降本方向与回归对比时使用。但**不要**因此放弃 DRAM 字节：
  有 NVBit 时**不要用它**猜，走轨迹口径（地址算术 ~0.001%，需外挂）；没有时走 NVML
  代理（±10% 带内，小 kernel/常驻 L2 会被明确拒绝）。只有 warp stall 周期细分拿不到。
  入口命令：kernoscope capabilities --brief → kernoscope --packet - -- <你的命令> →
  kernoscope selftest / kernoscope diff。不要凭印象拼子命令（先看 kernoscope --help）。
---

# Kernoscope：取证式 kernel 性能分析

## 什么时候用

- `ncu` 因为权限失败（`ERR_NVGPUCTRPERM`）或根本没有计数器权限；
- 想知道「这个 kernel 是不是 DRAM bound、离 roofline 多远、瓶颈是 latency 还是 throughput」；
- 需要**可复核**的结论：每个数字有出处（exact/measured/estimated）、误差带、以及它成立的前提；
- 要判断一次改动是不是真的更快（而不是噪声）。

## 什么时候不要用

- 要精确的 L2/DRAM 字节、缓存命中率、warp stall 周期细分 —— 这些与 ncu 同源，
  在这台机器上拿不到，工具会报 `n/a` + code + 替代口径；
- 要 SASS/源码逐行对照或跨进程时间线（那是 `nsys` 的活）。

## 三步闭环

```bash
# ① 这台机器能答什么（入口层：<1 KB，只有 id/code/计数；要解释再用 --json）
kernoscope capabilities --brief

# ② 测一次，拿决策包（判词 + 前提 + 结构化拒绝 + 代价；<4 KB）
kernoscope --packet - -- python bench_loop.py

# ②' 动手前先看它会做什么（起哪些进程/写哪些文件/读哪些输入；只打印，不执行）
kernoscope --plan -o rep -- python bench_loop.py

# ③ 建立本机误差带与 MDD（一次即可；之后所有比较都复用这个文件）
kernoscope selftest --out machine-selfcheck.json

# ④ 比较两次运行（difference 小于 MDD ⇒ indistinguishable，不要当成改进）
kernoscope diff before.json after.json --selfcheck machine-selfcheck.json --json

# ⑤ 单位成本（能耗实测 + 你给的 token 数与价格）与"先改哪里、能省多少"
kernoscope --set basic --ks-tokens 8192 --ks-price-per-kwh 0.6 --packet - -- python serve.py
#    -> 判词 cost_per_token（ks__cost_per_mtok / ks__j_per_token）与 saving_estimate
#       （保守~上界区间 + ¥/h；分母必须是本机实测峰值，重叠头寸不相加）

# ⑥ 想要 ncu 那边的东西：--metrics 支持 ncu 计数器名（它会告诉你替代口径叫什么）；
#    有 .ncu-rep 就导入做真值对拍，或把我们的结论写成合法 .ncu-rep（带署名）
kernoscope --metrics dram__bytes.sum --packet - -- python serve.py
kernoscope --ks-ncu-rep other.ncu-rep --ks-synthetic --set basic -o rep
kernoscope --export-ncu-rep out.ncu-rep -o rep -- python serve.py

# ⑦ CI 门禁：只读已产出的包与报告，n/a 一律算不通过（一条规则都没给时故意失败）
kernoscope gate --packet rep.packet.json --report rep.ksreport.json \
  --require kernel_time --require-actionable cost_per_token --max-metric ks__cost_per_mtok=0.5
```

需要人看时把 `--packet -` 换成 `--screen only`；需要完整报告时加 `-o rep`（会写
`rep.ksreport.json`，体积可达上百 MB —— 不要把它当默认输入）。

## 硬规则

1. **只有 `actionable=true` 的判词能当行动依据**。`status=refused`、任一
   `premises[].ok=false`、或被尺寸预算裁掉的判词一律不行（v1.0 的包没有 `actionable`
   字段，回退规则：不是 refused 且列出的前提全部成立）。
2. 拒绝不是错误：`refusals[code].action` 给出**可执行的补救办法**与代价（`cost_kind`
   = `wall_seconds` / `manual` / `unavailable`）。先按它解除前提，再下结论。
3. 比较必须同协议（`protocol.hash` 相同）且差值 > MDD；否则 `indistinguishable` 或
   `incomparable`。**不要把噪声当改进。**
4. 引用数字必须带出处与误差带：`estimated`（代理/模型）不得当作实测；
   `exact` 只用于与 ncu 同源的量（kernel 时长、launch 配置、cubin 静态属性）。
5. 缺依赖不是失败：`capabilities --brief` 的 `next` 告诉你缺什么、补上能多答哪些问题、要多少代价。
6. **CI 用 `kernoscope gate`**（退出码 0/1/2）：它判判词状态、`actionable`、拒绝码与指标阈值；
   `n/a` 的指标一律算不通过 —— "没有数"不是"没超阈值"。
7. **有副作用前先 `--plan`**：它会列出这次调用会起哪些进程、写哪些文件、读哪些输入（含第三方
   软件 nsys/cuobjdump）。`--plan` 本身不执行任何东西；`--plan-json` 给 agent 读（每条操作带
   8 项 `effects` 与 4 项推导出来的 `annotations`）。我们用你自己的权限跑你给的命令，不注入代码。
8. **把结论交给下游**：`--report-md rep.md` 出人读报告（0–6 固定结构、每条方向必须带 Evidence）；
   `kernoscope skill --apply` 把这份 skill 装进 agent 的 skill 目录（默认只打印计划，先备份再写、
   写完回读校验、失败回滚）。

## 参考层（按需读，别默认全读）

<!-- refs-index:begin -->
<!-- 由 tools/gen_agent_refs.py 生成（字节数是现算的）；改动请改那里的 REF_DOCS -->

| 文件 | 字节 | 里面有什么 |
|---|---|---|
| `references/refusal-codes.md` | ~14 KB（生成） | 每个拒绝码什么意思、怎么办、代价 |
| `references/verdicts.md` | ~4 KB（生成） | 判词目录、方向、前提、MDD 键 |
| `references/diagnosis-playbook.md` | ~30 KB（生成） | 受限环境诊断手册：signal→cause→fix、任务路由、补不回来的采集 |
| `references/provenance-rules.md` | ~2 KB | 出处纪律的短版 |
| `references/recipes.md` | ~3 KB | 四个闭环配方 |

<!-- refs-index:end -->

单个 code 的处置**不要**读整张表：用 `scripts/remedy.sh <code>`。
