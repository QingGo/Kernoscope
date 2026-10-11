# Kernoscope —— 给**开发这个仓库**的 agent 的入口

一句话：在拿不到硬件性能计数器的受限 GPU 上，用物理观测量替代计数器，为 kernel 优化给出
「每个数字都标注出处」的证据。**这份是改代码用的入口**；只想用它分析 kernel 的 agent
请走 `skills/kernoscope/SKILL.md`（那里有触发条件与闭环命令）。

## 开发闭环

```bash
uv sync
uv run python tools/dev_index.py --brief   # 改哪类东西 → 改哪个文件 → 跑哪条门禁（≤1.5 KB）
uv run python tools/dev_index.py --gates   # 提交前：全部门禁一把跑完（全绿才算完）
```

## 改代码的硬规则

- **一个真源**：非 `exact` 的值不许挂在 `sm__`/`dram__` 这类计数器名下（R2/R3）；
  `unavailable` 必须带 `code=`。数字的出处判定只在 `provenance.py` 里。
- **生成物不许手改**：`COVERAGE.md`、`skills/kernoscope/references/*.md`、`eval/tasks/*.json`。
  改代码后按索引重跑生成器 —— 漏跑会在 `--gates` 里红。
- **公开数字要能追到产物**：`bench/artifacts/` 里那一次可复现的运行；**已登记**的数字由
  测试逐条对拍（登记范围见 `test_doc_artifact_consistency.py`），没登记的要自己确认。
- **常驻入口层有 4096 B 硬预算**：加内容就要减内容（否则每个会话都付全量上下文的钱）。
- **别把机器身份写进仓库**：主机名/实例名/裸 IP/绝对路径不进提交信息、注释与文档；
  不 `git add -f`；提交动作与身份由维护者负责。
- 测试比文档权威：**冲突时以测试为准**。

细节：`CONTRIBUTING.md`（首屏就是索引表：改哪类东西 → 改哪个文件 → 跑哪条门禁）、
`PROVENANCE.md`（四个出处等级与四条规则）、`COVERAGE.md`（三态能力矩阵）、`tests/`（契约）。
