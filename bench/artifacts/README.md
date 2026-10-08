# bench/artifacts/ —— 入库产物的登记表

这个目录里的每一份产物都是**一次真实运行**的结果，不是手写的示例。
登记的理由只有一条：文档里的每个"实测"数字都要能追到某一份产物，
而且读者要能判断**这份产物是哪个修订产生的、能不能当证据用**。

`tests/test_doc_artifact_consistency.py` 会检查两件事：本文件登记了每一个产物目录，
以及**最新**产物钉住了修订号、工具版本与脱敏步骤。新增目录时请在这里补一行。

| 产物目录 | 修订号 | 工具版本 | 状态 |
|---|---|---|---|
| `20261009-012257` | `fc1cdba`（`environment.txt` / `manifest.json` 里的 `revision`） | 0.2.0 | **现行**：文档里引用的数字以它为准。42 个步骤里只有 `strict-lowduty` 非零退出，那是**预期**的（请求了账本对拍但覆盖率不足，`--strict` 就该 exit 2）。这一轮除了既有实验，还多出四类新证据：`capabilities*.json`（能力分层与三态矩阵）、`machine-selfcheck.json` + `selftest.txt`（本机误差带与 MDD）、`rep-*.packet.json` + `rep-copy.onepage.txt`（决策包与一屏，包 3934 B ≤ 4 KB 预算）、`diff-same-protocol.*` 与 `diff-protocol-mismatch.*`（同协议两次运行全部 `indistinguishable`；换窗口模式则 `incomparable`，只给差值不给方向） |
| `20261008-235142` | `0f296f3`（`environment.txt` / `manifest.json` 里的 `revision`） | 0.2.0 | **历史**：仍保留可查（误差带的另一侧 −4.3% ~ +12.7% 来自它与更早那次的并集），但**不要再**拿它当现行证据 —— 它比现行产物早一轮，没有决策包/自检/对拍那几类文件 |
| `20261008-225555` | 未留存（`environment.txt` 写 `git: n/a` —— 仓库同步到租用机器时没带 `.git`，也没有 `REVISION` 文件） | 0.1.0（报告头） | **历史**：保留它是为了让"上一次运行是多少"可查（例如误差带的另一侧）。它**不能**当现行证据：既钉不住修订号，报告头版本也早于当前代码。R-40 之后不再产生这种产物 |

## 怎么读这些文件

- `environment.txt` / `manifest.json`：这次运行的实验条件与**每个步骤的退出码**。
  非零退出不隐藏，也不影响其它步骤 —— 但引用数字前要先看它。
- `rep-*.txt`：报告原文（含 provenance、resolution、以及**每个前提**的实际值）。
- `rep-*.csv`：同一份报告的 CSV（每个指标一行，适合对拍）。
- `proxy-duration-sweep.csv`：**精度 vs 稳态段长度**。`err_pct` 为空表示该时长下
  前提不成立、工具**拒绝**给 per-launch —— 那是预期行为，不是"精度变差"，
  也不是"跑失败了"。
- `probe-*.log`：微基准的解析真值（`ANALYTIC` 行，不需要 GPU 也能重算）。
- `machine-profile.json`：本机峰值 + 累加口径 + 口径离散度 + 逐变体设置。
### 现行产物里的两处裁剪（如实说明）

- 逐 launch 的大 JSON、`.nsys-rep`、`.sqlite` 与 NVBit 的原始轨迹**不入库**（体积大、按需重采），
  所以 `rep-*.ksreport.json` 不在本目录里；体积数字由 `size-comparison.csv` 给出（它是机器读的）。
- `diff-same-protocol.json` 里 `dram_bytes` 的 MDD 是 14.0%：这一轮的自检在跨进程复测里
  量到了约 4.7% 的漂移（`machine-selfcheck.json` 的 `mdd.cross_process_samples` 有原始样本），
  于是 `dram_bytes_per_launch` / `dram_bytes` 分别只差 −3.6% / −3.6% 时被判为
  `indistinguishable`。这正是想要的结论：同协议同命令的两次运行**不该**报"改进"。

### `20261008-235142` 的一处补算（如实说明）

这个目录里的 `occupancy-records.smem32k.txt` 与 `occupancy-smem32k.json` 不在该次
`manifest.json` 的步骤表里：探针**本来就同时写两份记录**（`--smem 0` 与 32 KB 静态共享内存），
但当时 `run.sh` 只分析了第一份，于是 README 引用的第二个读数不在产物里。
补算是用**同一次运行留下的原始探针记录**、按同一命令（`kernoscope occupancy --from … --json`）
在本机（不需要 GPU）重放的，输入不是新采的数据；`run.sh` 已加上这一步，
下一次运行会自动产出并记录。之所以不重跑整轮：补算的输出与运行环境无关（纯区间重叠算术），
而重跑会改动其它所有已引用的数字。

- 原始轨迹与逐 launch 的大 JSON **不入库**（体积大、按需重采）：
  `.ksreport.json`、`.perfetto.json`、`*_nsys.*`、`memtrace-official.txt` 都在 `.gitignore` 里。
