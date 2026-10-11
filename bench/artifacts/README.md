# bench/artifacts/ —— 入库产物的登记表

这个目录里的每一份产物都是**一次真实运行**的结果，不是手写的示例。
登记的理由只有一条：文档里的每个"实测"数字都要能追到某一份产物，
而且读者要能判断**这份产物是哪个修订产生的、能不能当证据用**。

`tests/test_doc_artifact_consistency.py` 会检查两件事：本文件登记了每一个产物目录，
以及**最新**产物钉住了修订号、工具版本与脱敏步骤。新增目录时请在这里补一行。

| 产物目录 | 修订号 | 工具版本 | 状态 |
|---|---|---|---|
| `20261011-094535` | `2b9a7e3`（`environment.txt` / `manifest.json` 的 `revision`；`revision_source: git`、`revision_dirty: no`，另附 `source_sha256: dcbe29605b3340c0`） | 0.2.0 | **现行**：文档里引用的数字以它为准。46 个步骤里只有 `strict-lowduty` 非零退出，那是**预期**的（请求账本对拍但覆盖率不足，`--strict` 就该 exit 2）。这一轮多出一类证据：**R-47 的窗口分母验收**（`window-denominator.txt` + `window-auto|warmup1|warmup2.packet.json`）—— 同一个单 kernel 负载（每次 launch 读+写 1073.74 MB，解析可知）换三种窗口：1057.64 / 1056.22 / 1047.88 MB，相对离散 **0.93%**（判据 ≤5%），与解析真值最大偏差 2.41%。同一份代码在旧实现下是 1.06 GB / 0.86 GB / 0.52 GB 且**无任何告警**。决策包 `rep-copy.packet.json` 内的 `cost.packet_bytes` 是 4088 B（≤ 4096 预算，裁剪 3 步），磁盘文件 4089 B；`rep-copy.packet-only.json` 是 3842 / 3843 B。**多了 1 B 是因为该产物由 `80825e9` 之前的写入器产生**（那时落盘会多写一个换行）；从 `80825e9` 起 `write_packet` 保证「落盘字节数 == `packet_bytes`」（有测试守）。 |
| `20261009-012257` | `fc1cdba`（`environment.txt` / `manifest.json` 里的 `revision`） | 0.2.0 | **历史**（曾经的现行，保留可查）：42 个步骤里只有 `strict-lowduty` 非零退出，那是**预期**的（请求了账本对拍但覆盖率不足，`--strict` 就该 exit 2）。这一轮除了既有实验，还多出四类新证据：`capabilities*.json`（能力分层与三态矩阵）、`machine-selfcheck.json` + `selftest.txt`（本机误差带与 MDD）、`rep-*.packet.json` + `rep-copy.onepage.txt`（决策包与一屏，包 3934 B ≤ 4 KB 预算）、`diff-same-protocol.*` 与 `diff-protocol-mismatch.*`（同协议两次运行全部 `indistinguishable`；换窗口模式则 `incomparable`，只给差值不给方向） |
| `20261008-235142` | `807a025`（`environment.txt` / `manifest.json` 里的 `revision`） | 0.2.0 | **历史**：仍保留可查（误差带的另一侧 −4.3% ~ +12.7% 来自它与更早那次的并集），但**不要再**拿它当现行证据 —— 它比现行产物早一轮，没有决策包/自检/对拍那几类文件 |
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
### `w1-degradation/`：同一台机器、同一修订的「缺 nsys」降级证据

W1 的验收条件是"缺件时也要成功"。这一份是 `bench/degradation-check.sh`（把 nsys 临时
移开、trap 保证恢复、需要显式同意）在同一台机器、同一修订上跑出来的：

* `capabilities-full.json` / `capabilities-no-nsys.json`：移开前后各一份能力清单
  （`has_nsys` 从 true 变 false；`kernel_time` 从能答变被拒并带 `nsys_missing`；
  `device_busy` 仍然可用）；
* `degraded.packet.json`：决策包（3759 B），里面 `environment_refusals=[nsys_missing]`、
  `env.blocked_by` 列出缺哪些依赖 —— 机器可读地说明"这台机器少了什么"；
* `degraded.onepage.txt`：一屏，结论优先 + "怎么办"；
* `degraded.err`：整个过程只有 note/warn，没有裸错误；**退出码 0**；
* `first-run-timing.txt`：首跑时长（doctor 2.87 s / capabilities 2.11 s / 带实测探测 2.59 s /
  合成首跑 2.32 s）—— 这是 G1「陌生人首跑 < 5 分钟」的直接度量。

它不在 `manifest.json` 的步骤表里（那是 `run.sh` 的步骤），因为"把 nsys 移开"这一步
不适合放进任何人都会跑的复现脚本里；产物本身（能力清单与包）与其它证据同样可复核。

### 现行产物里的两处裁剪（如实说明）

- 逐 launch 的大 JSON、`.nsys-rep`、`.sqlite` 与 NVBit 的原始轨迹**不入库**（体积大、按需重采），
  所以 `rep-*.ksreport.json` 不在本目录里；体积数字由 `size-comparison.csv` 给出（它是机器读的）。
- `diff-same-protocol.json` 里 `dram_bytes` 的 MDD 是 **5.0%**（`machine-selfcheck.json` 的
  `mdd.dram_bytes_pct`；本次自测的 run-to-run SD 是 0.31%，另有 5% 的保守下限常量兜底），
  于是 `dram_bytes` / `dram_bytes_per_launch` 分别只差 **−0.025% / −0.202%** 时被判为
  `indistinguishable`（`kernel_time` 差 0.059 ms、`occupancy` 差 0.0% 同样如此）。
  这正是想要的结论：同协议同命令的两次运行**不该**报"改进"。

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
