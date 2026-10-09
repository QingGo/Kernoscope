# Kernoscope —— 给 agent 的入口

一句话：在受限/多租户 GPU 上，用**物理观测量**替代被锁死的硬件性能计数器，为 kernel
优化给出「每个数字都标注出处」的证据。CLI 与 `ncu` 兼容；缺依赖时优雅降级并给出下一步。

## 三步

```bash
kernoscope capabilities --brief                       # ① 这台机器能答什么（<1 KB，只有 id/code/计数）
kernoscope --packet - -- <你的命令>                    # ② 测一次，拿决策包（<4 KB）
kernoscope selftest --out machine-selfcheck.json       # ③ 建立本机误差带（一次即可，之后复用）
kernoscope diff a.json b.json --selfcheck machine-selfcheck.json   # ④ 比较两次运行
```

## 硬规则（不要绕过）

- `actionable=false`、`status=refused`、或任一 `premises[].ok=false` 的判词**不得作为行动依据**；
  先按 `refusals[code].action` 解除前提，或换一条口径。
- 两次运行必须 `protocol.hash` 相同**且**差值 > MDD；否则一律"分不出来"（`indistinguishable`）。
- 引用数字要同时给出处（`exact`/`measured`/`estimated`）与误差带；`estimated` 不得当作实测。
- 默认读 `--packet`（约 1.2K token），**不要**读完整报告 JSON（可达上百 MB）。
- 计数器类指标（精确 L2/DRAM 字节、命中率、warp stall 细分）在这台机器上拿不到，
  工具会明确报 `n/a` + 原因 + 替代口径 —— 那不是失败。

细节：`skills/kernoscope/SKILL.md`（触发条件与闭环脚本）、`COVERAGE.md`（三态覆盖矩阵）、
`PROVENANCE.md`（出处体系）、`CHANGELOG.md`（接口层变化）。
