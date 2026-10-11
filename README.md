# Kernoscope 内核镜

**先告诉你钱在哪，再证明你改完省了多少。** 在多租户/受限 GPU 上（`ncu` 因权限跑不了）给出「每个数字都标注出处」的性能证据：按实测占比排序的优化头寸 + 成本折算，结论可被人与 agent 独立复核。

它回答：**先改哪里能省多少**（按实测占比排序的头寸，不需要先标定/先给价格）、哪个 kernel 慢（`exact`）、离 roofline 多远/是不是 DRAM bound（`estimated` 带误差带）、实际达到的 occupancy 与发射槽上界、**每百万 token 花多少钱**；它**不能回答**精确 L2/DRAM 字节、命中率、warp stall 周期细分 —— 与 ncu 同源的硬件计数器只报 `n/a` + 原因 + 替代口径（完整三态见 [COVERAGE.md](COVERAGE.md)）。

30 秒上手（没有 GPU 也能跑通）：先 `kernoscope doctor` 看这台机器能答什么、拿不到什么、为什么（`--no-probe` 只做静态推断、不做计数器实测探测；`--json` 给机器读），再 `kernoscope --packet - -- <你的命令>` 拿一份 <4 KB 决策包。

与 `ncu` 的关系：CLI 兼容但数据源不同（它读硬件计数器，我们读物理观测量，**绝不占用它的指标名**）；与 `nsys` 的关系：复用它的 CUPTI activity，是消费者不是替代品；与 `torch.profiler` 的关系：它答"哪一步慢"，我们追"这个 kernel 为什么慢、这个数字有多可信、改它值多少钱"。

---

## 目录

1. [为什么需要它](#1-为什么需要它) · [它明确不做什么](#它明确不做什么)
2. [安装与第一分钟](#2-安装与第一分钟)
3. [证据模型：每个数字都带出处](#3-证据模型每个数字都带出处)
4. [与 ncu / nsys / torch.profiler 的关系](#4-与-ncu--nsys--torchprofiler-的关系)
5. [DRAM 字节：两条口径与实测精度](#5-dram-字节两条口径与实测精度)
6. [无计数器的其它能力](#6-无计数器的其它能力)
7. [成本与处方：先改哪里、能省多少](#7-成本与处方先改哪里能省多少)
8. [比较与 CI](#8-比较与-ci)
9. [输出形态：一屏、决策包、案件目录、报告](#9-输出形态一屏决策包案件目录报告)
10. [副作用与同意：`--plan`](#10-副作用与同意--plan)
11. [局限与已知失败](#11-局限与已知失败)
12. [数字从哪来](#12-数字从哪来)
13. [拿不到数字时先看这里](#13-拿不到数字时先看这里)
14. [状态与路线](#14-状态与路线)
15. [文档与声明](#15-文档与声明)

---

## 1. 为什么需要它

很多共享租用的多租户 GPU 容器实例宿主机带着
`RmProfilingAdminOnly=1`，容器又没有 `CAP_SYS_ADMIN`/`CAP_PERFMON`，于是：

```
$ ncu --set basic -- python train.py
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
          NVIDIA GPU Performance Counters on the target device 0.
```

这不是配置问题，也不能靠用户侧技巧绕过：计数器是 host-wide 且会泄露同机租户行为，
所以平台有理由锁死。**但"读不到计数器"不等于"拿不到信息"** —— 换物理观测量即可：

| 你想要的结论 | Kernoscope 怎么给 | 出处 |
|---|---|---|
| 哪个 kernel 慢、launch 配置如何 | CUPTI activity（nsys）→ LaunchStats / 时长 | `exact` |
| 同名 kernel 的**多组** launch 配置各自的时长 | 按 (grid, block, smem) 分组，逐组给出读数 | `exact` |
| occupancy 上界与被谁限制 | 静态属性 + 架构上限，解析算术 | `exact` |
| **实际达到的 occupancy** | `(smid, globaltimer)` 区间重叠重建（需探针） | `measured` |
| **DRAM 字节数 / 是不是 DRAM bound** | NVML 显存忙碌时间 × 本机实测峰值带宽 | `estimated`（需窗口覆盖率达标） |
| **DRAM 字节数的第二条口径** | NVBit 访存轨迹 + 缓存模型（地址是算术） | `estimated` |
| 离 roofline 多远 | 本机实测峰值（不是规格表）+ 你给的工作量 | `estimated` |
| 发射槽用了多少、瓶颈方向 | NVBit 动态指令数 ÷ 硬件发射槽上限 | `estimated`（上界口径） |
| 训练/推理各阶段占多少 kernel 时间 | nsys 的 NVTX 区间按重叠时长归因 | `derived` |
| 窗口/设备级能耗与功率 | NVML 能量累加器 / 窗口内功率积分（ncu 没有这个指标） | `measured` |
| **每 kernel** 能耗与成本 | 窗口能耗 × **该 kernel 窗口时间占比**（代理口径，假设窗口内功率大致均匀） | `estimated` |
| **每百万 token 多少钱、先改哪里** | 实测能耗 + 你给的 token 数与价格 + 实测峰值头寸 | `measured`/`derived`/`estimated` |
| 真值校准我们的替代口径 | 导入一份 `.ncu-rep`（官方只读 API）逐条对拍 —— **包装逻辑已验；真报告口径未验证**（见 §4.3） | `exact` / `derived` |
| 让已有 ncu 工作流继续能用 | 导出**带署名**的替代口径 `.ncu-rep` | `estimated` / `derived` |
| L2/DRAM 精确字节、命中率、warp stall 细分 | **不给数字**，报 `n/a` + 原因 | `unavailable` |

最后一行是重点：**我们拒绝在 ncu 的计数器指标名下面填一个模型算出来的数。**
这由**类型系统**强制（规则 R1–R4 见 [PROVENANCE.md](PROVENANCE.md)），并有测试守住。

### 它明确不做什么

**逆向是允许的（含没有文档的面）；边界不在"能不能看"，而在"发布出去的值"上。**
每条「不做」都写明它在**保护什么**、**代价是什么**、**前提是什么**、**我们改成了什么形态** ——
单一真源见 [NON-GOALS.md](NON-GOALS.md)（13 条，逐条带实施状态与复审触发信号）。
下表第三列**只写已经实现的形态**；已决定但尚未实现的，一律不许出现在这里。

| 不做 | 保护什么 | 已经改成的形态 |
|---|---|---|
| **kernel 级计数重放**（为 multiplex 计数器反复跑被测 kernel） | 不引入特权控制面、不扰动同机他人；耗时是 nsys 时间戳里的**真实时间** | `--ks-warmup` / `--ks-window-s` 只统计稳态段（重放的弱形式） |
| **默认锁频**（不改设备时钟） | 时钟是**设备级**状态，锁了会影响同机所有进程 | 记录时钟驻留与降频原因（公开位名解码，未知位照实写 `bitN`）作为环境证据；**读不到一律记未知（`unspecified` / `env_reading_unknown`），不当作"没有降频"** |
| **绕过计数器权限** | 计数器是 host-wide 观测面，能泄露同机租户；平台锁它有正当理由 | 报 `counters_denied` + 物理原因（策略/capability），并给替代口径 |
| **冒充计数器语义** | 可审计性是唯一不可让渡的资产 | 同源量按 ncu 同名同单位写（`gpu__time_duration.sum`：ms × 10⁶ = ns；单位不符就不写）；代理/模型/成本量只进 `ks__`/`est__`/`meas__`；每份文件带署名 |
| **向被测进程注入我们的指令** | 不产生观察者效应、不需要用户改代码 | 复用厂商通道；NVBit 插桩由 `tools/probes/nvbit_memtrace/collect.sh` 在你调用它时执行，本工具只读它产出的文件 |
| **自研插桩框架** | 不重复造轮子 | 用 NVBit，并从**实物**读它的版本（版本号不是可用性凭据） |
| **覆盖更多 ncu 功能** | 不与 NVIDIA 正面对撞 | 两条可度量的线：**同源覆盖**（同源量就该写同名同单位）+ **决策覆盖**（真实裁决问题的三态比例，见 [COVERAGE.md](COVERAGE.md)） |
| **需要特权计数器的精确值** | 不猜、不编 | 报 `n/a` + `code` + 原因 + 替代口径。「不做」的对象是**精确值**，不是**那个问题** |
| **没有前提的数字** | 可判定性 | 结构化拒绝：`code` + `observed` + `threshold` + 可执行的 remedy + 代价（见 §3） |
| **GUI / Web UI** | 不做重资产 | 零安装的 Perfetto 输出；单文件 HTML 渲染器**尚未实现**（见 NON-GOALS.md §3.1） |

有 `ncu`/`rocprof` 可用时，我们的定位是**编排它们**（`route` 给出该用哪个）并对结果做经济口径的
融合；没有它们（容器/多租户/异构卡）时才降级到替代口径。缺哪个数据源就报告缺哪个，并给出
"最便宜的补法"。

---

## 2. 安装与第一分钟

```bash
# uv（推荐）
uv tool install kernoscope

# 或从源码
git clone https://github.com/QingGo/Kernoscope && cd Kernoscope
uv sync && uv run kernoscope --help
```

依赖：Python ≥ 3.10，**运行时零第三方依赖**。运行需要 `nsys`（随 CUDA ≥ 11.1 提供）与
`nvidia-smi`；`cuobjdump`（静态资源/SASS）为可选；`pynvml`（能耗/显存忙碌优先走它）
用 `pip install 'kernoscope[nvml]'` 安装。
**工具的查找不只看 PATH**：`$CUDA_HOME/bin`、`/usr/local/cuda*/bin`、`/usr/bin` 都会找 ——
在"CUDA 装了但没加 PATH"的实例上，只看 PATH 会误报"未安装"。

想让它自动出现在 agent 的上下文里（一条命令装好）：

```bash
kernoscope skill                  # 只打印计划：装到哪、会新建/更新哪些文件、备份到哪
kernoscope skill --apply          # 真的写；先备份、原子写、写后回读校验、失败逆序回滚
kernoscope skill --client agents --apply   # 或装到 ~/.agents/skills（其它 client 的共享约定）
```

```bash
# 第一分钟：这台机器能答什么、拿不到什么、为什么
kernoscope doctor            # 默认会实测确认计数器门（1 秒）
kernoscope doctor --no-probe # 只做静态推断：不跑 ncu 实测探测
kernoscope capabilities --brief      # agent 用的入口层（<1.2 KB，只有 id/状态/计数）
kernoscope capabilities --json       # 完整三态矩阵 + 补救动作 + 代价

# 带问题来的时候先路由（只输出计划，不跑测量）
kernoscope route "为什么这个 kernel 慢"
kernoscope route "每百万 token 花多少钱" --json

# 测一次，拿决策包（判词 + 前提 + 结构化拒绝 + 代价；<4 KB）
kernoscope --packet - -- python bench_loop.py

# 没有 GPU 也能跑通全流程（合成数据，明确标注 SYNTHETIC）
kernoscope --ks-synthetic --set detailed

# 动手之前先看副作用（起哪些进程、写哪些文件、读哪些输入；只打印不执行）
kernoscope --plan -o rep -- python bench_loop.py
```

> 网络提示：`uv` **不继承 `pip.conf`**，会直连 PyPI。若你的网络访问 PyPI 很慢，
> 给 uv 单独配镜像，否则 `uv sync` 可能只是"看起来卡住"：
> ```bash
> mkdir -p ~/.config/uv && cat > ~/.config/uv/uv.toml <<'TOML'
> [[index]]
> url = "https://pypi.tuna.tsinghua.edu.cn/simple"
> default = true
> TOML
> ```
> 实测（某国内 GPU 实例）：PyPI 直连 0.1 MB/s、某镜像 0.1 MB/s、TUNA 7.6 MB/s ——
> 配好之后 `uv sync` 从"分钟级卡住"变成 4 秒。

### 先把本机峰值标定出来（强烈建议）

roofline 与处方头寸都用**本机实测峰值**而非规格表。一台 RTX 4090（450 W 功耗墙）的实测差异：

| 指标 | 规格表 | 本机实测 |
|---|---|---|
| bf16 matmul | 165.2 TFLOP/s | **163.5**（口径离散 3.9%，可分辨） |
| fp32 matmul | 82.6 | **54.6**（只有 66%！；口径离散 0.7% ⇒ **不可分辨**） |
| DRAM copy / read / write | 1008 GB/s | **911.8 / 904.4 / 940.0** |
| 实测期 SM 时钟 | 3105 MHz | **1815–2625（均值 2356）** |

```bash
python tools/machine-profile.py --quick      # 写出 ./machine-profile.json
kernoscope --ks-calibration machine-profile.json --ks-gemm 4096,4096,4096 --ks-dtype bf16 \
           -k regex:gemm -- python bench.py
```

标定产物记录**累加口径**（`acc_caliber` / `precision_settings` / `method`）与
**口径离散度**（`caliber_spread_pct` / `caliber_distinguishable`）——
因为"口径能不能被分辨"本身是测量结果：同一台机器相隔一天的两次标定给出 0.4% 与 5.0% 两种结论。
缺这些字段时，Kernoscope **拒绝用它算 roofline 拐点与处方头寸**（带宽侧照常给数）。
详见 [CALIBRATION.md](CALIBRATION.md)。

> 数字来自 [`bench/artifacts/`](bench/artifacts/) 里那次完整标定，见 §12。

---

## 3. 证据模型：每个数字都带出处

这是整个项目的地基：**在受限环境里，最危险的不是"没有数据"，而是"看起来合理的假数据"。**
所以出处分四级，且**写在指标名里**（规则 R1–R4 由 `provenance.py` 的类型校验强制）：

| 出处 | 含义 | 命名 |
|---|---|---|
| `exact` | 与 ncu 同源可得（CUPTI activity 时间戳、launch 配置、cubin 静态属性） | ncu 的名字（`gpu__time_duration.sum`、`launch__*`） |
| `measured` | 我们的探针**实测读数**（NVML 管理计数器、区间重叠探针） | `meas__*` |
| `derived` | 由实测/精确量**再算出来**的统计量与判词（均值、占比、成本、判词） | `ks__*` |
| `estimated` | 模型或采样推算（代理口径、访存轨迹模型、roofline 达成率、处方头寸） | `est__*` / `model__*` |
| `unavailable` | 给不出，只报 `n/a` | 任何名字，必须带 `code=` |

两条硬纪律：

1. **计数器命名空间（`sm__`/`dram__`/`lts__`/…）下面只允许 `exact`，且名字必须登记在允许表**
   （当前允许表是空的）。所以"在 ncu 的指标名下填一个模型算出来的数"在**类型层面**不可能发生。
2. **非 `exact` 的值必须用自有前缀**（`ks__`/`est__`/`meas__`/`model__`）。
   连"由实测样本再算出来的占比"都不许叫 `meas__`（命名承诺什么，出处就得是什么）。

**拒绝不是错误，是输出。** 每个 `n/a` 行都带结构化拒绝：闭集 `code` + 观测值 + 门槛 + 可执行
处置 + 代价（`wall_seconds`/`manual`/`unavailable`）。散文只解释，判决在 code 里 ——
所以 agent 可以编程处理（`kernoscope gate`、`refusals[code].action`、`scripts/remedy.sh <code>`）。

**判词带前提。** 决策包里的每条结论都列出支撑它的前提（`observed` vs `threshold`），
前提不成立时该判词**必须是 `refused`**，不允许以 `soft` 出现（有测试守着）。
只有 `actionable=true` 的判词能当行动依据。

`--strict` 面向 CI：**请求了产不出来的数据就 exit 2**（覆盖 `--metrics`、`--section`/`--set`、
空转的 ncu 开关、账本裁决行、以及成本输入）。宁可直接红，也不悄悄给错数：

```bash
# CI 卡门：请求了拿不到的数据就失败（exit 2），而不是悄悄给错数
kernoscope --strict --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -- python train.py
```

---

## 4. 与 ncu / nsys / torch.profiler 的关系

### 4.1 当 `ncu` 的替代品用（CLI 兼容）

```bash
alias ncu=kernoscope          # 或 ln -s $(which kernoscope) ~/bin/ncu
```
`argv[0]` 为 `ncu` 时会自动进入兼容模式（并打印一行说明：数据来源与 ncu 不同）。
选项分四类处理：

| 类别 | 例子 | 行为 |
|---|---|---|
| 真实现 | `-k` / `-c` / `-s` / `--devices` / `-o` / `-i` / `--csv` / `--launch-csv` / `--print-*` | 按 ncu 语义生效 |
| 映射实现 | `--set` / `--section` / `--metrics` | 映射到我们的数据源；产不出的只在 `unavailable` 里出现 |
| 空转（会告警，`--strict` 下报错） | `--replay-mode` / `--cache-control` / `--clock-control` / `--pm-sampling-*` / `--mps` 等 | 我们的数据源没有 replay / 计数器 pass，所以无效 |
| 自有扩展 | `--ks-*` | 数据源、标定、账本、窗口、成本、导出的 `.ncu-rep` |

**两点语义差异先讲清楚**：① ncu 默认**锁频**并 replay 被测 kernel，我们的耗时来自 nsys
时间戳（真实时间、不 replay、不锁频），所以绝对耗时与 ncu 的锁频读数不可直接比较；
② 我们不占用 ncu 的指标名（规则 R3），重建量一律带自有前缀并在 `source` 里注明它对应哪个 ncu 指标。

### 4.2 ncu 名字是**发现键**，不是身份

带着 `--metrics dram__bytes.sum` 来，不会得到一句"指标不存在"：

```
==PROF== note: --metrics dram__bytes.sum → 替代口径 est__dram_bytes_total（零依赖的 NVML 代理口径…）；
             数值不会挂在 dram__bytes.sum 名下
```

- 已知的 ncu 名字会解析成"我们给什么"（替代口径的真实名字）或"为什么给不出"
  （动态 Tensor Core 利用率、warp stall 周期细分确实拿不到）。`--strict` 下仍然 exit 2，
  但错误信息里写明**替代口径叫什么**。
- 硬规则一秒没松：数值仍然只出现在 `ks__`/`est__`/`meas__` 名下。
- `--export-name-map names.json`（或 `.csv`）导出"ncu 名 → 我们给什么 / 为什么给不出"的对照表，
  供下游脚本翻译（它是**声明**，不是数据替换）。

### 4.3 导入真值：校准我们的替代口径（只读）

> ⚠ **状态：包装逻辑已验；真报告口径未验证。** 这条路径的**代码与契约**有测试守着
> （`tests/test_ncu_rep.py` 用假 `ncu_report` 模块把包装逻辑跑一遍、`tests/test_truth_and_substitute.py`
> 用与真实 dataclass 同构的替身跑对拍算术），但本仓库里**没有一份真 ncu 报告**：
> 唯一入库的 `.ncu-rep` 是**我们自己写的替代口径**（头是 `kernoscope-substitute-v1`）。
> 采一份真报告需要一台 `ncu` 能收集计数器的机器 —— 那正是本项目要绕开的前提
> （验收方在受限机上实测 `ERR_NVGPUCTRPERM`）。所以下面表里那些对拍行**一次都没在真报告上出现过**：
> "真值对拍"目前只能按"包装逻辑已验"读，等有脱敏的真报告可入库时再升级这句话。

```bash
kernoscope --ks-ncu-rep other.ncu-rep --ks-synthetic --set basic -o rep
```

用官方只读 API 读一份别处采到的 `.ncu-rep`：真值以 **`ncu__` 前缀 + `exact`** 出现
（与 ncu 同源，但不落计数器命名空间），并与我们的替代口径**逐条对拍**：

| 对拍 | 真值 | 我们 | 结果 |
|---|---|---|---|
| 每次 launch 时长 | `ncu__gpu__time_duration.sum`（ns） | `ks__launch_time_mean`（ms） | `ks__truth_vs_substitute_pct` |
| 每线程寄存器 / 静态共享内存 | 同名 | 同名 | 自检行：两边同源，应当逐位相同 |
| 每次 launch DRAM 字节 | `ncu__dram__bytes.sum` | `est__dram_bytes_per_launch` | 代理精度的直接证据 |
| L2 命中率 | `ncu__lts__t_sector_hit_rate.pct` | `est__l2_hit_rate_pct`（模型上界） | 只应作为上界比较 |

一条都对不上时也是 `refused` + 原因（沉默会被读成"对上了"），并用
`--json` 的 `ncu_truth.comparisons[].pairs[].reason` 逐条说明。
有 `-o` 时另写 `<base>.ncu-truth.json`（可归档、可 diff 的真值对照产物）。
读不了报 `ncu_report_unreadable` + 处置（缺的往往只是把 Nsight Compute 的 `extras/python`
放进 `PYTHONPATH`）：**读报告不需要 GPU、也不需要计数器权限**。

### 4.4 导出替代口径：让已有 ncu 工作流继续能用

```bash
kernoscope --export-ncu-rep out.ncu-rep -o rep -- python bench.py
```

写出一份**合法的 `.ncu-rep`**（官方 reader 能按名字读回值、`ncu --import` / `ncu-ui` 能打开），
**同时**写 `out.ncu-rep.ATTRIBUTION.txt` 旁路署名。三条门禁：

1. **署名是强制的**：文件内 `provider`/`comment` 为空 → 拒绝生成（`build_report` 抛错）；
   `comment` 里写明"这不是硬件计数器读数"、每个值的出处、以及它的前提。
2. **前提不成立的值不写**：决策包里任何 `status=refused` 的判词所引用的指标名都不许进文件。
3. **一个值都不剩时拒绝生成**：生成一个空的/全是垃圾的"ncu 报告"比不生成更糟。

#### 哪些名字写 ncu 名下、哪些留在自有命名空间

规则只有一条：**同源才写 ncu 名下，且必须按 ncu 的单位与标签写**。

| 类型 | 例子 | 写在 | 谁能看到 |
|---|---|---|---|
| 同源（CUPTI activity / cubin，语义与 ncu 一致） | `gpu__time_duration.sum`、`launch__grid_size`、`launch__block_size`、`launch__registers_per_thread`、`launch__shared_mem_per_block_{static,dynamic}`、`launch__thread_count` | ncu 的节（`SpeedOfLight` / `LaunchStats`），名字、单位、显示标签都按 ncu | `--page details` 的节标题与文本、`--page raw`、ncu-ui、Python API |
| 替代口径（NVML 采样 / 轨迹 / 模型 / 成本） | `ks__*`、`est__*`、`meas__*` | 自有节 `KernoscopeMetrics` | `--page raw`、Python API、旁路署名、文件内 RuleResult |

**单位换算发生在写文件那一刻，且只发生一次**（`ncrep/ncu_rows.py` 是唯一的白名单与系数表）：
`gpu__time_duration.sum` 我们是 ms、ncu 是 ns ⇒ **×10⁶** 后写入。
之所以要写而不是略过：略过是把**单位问题**当成了**身份问题** —— 同源量只要换算正确，
写在 ncu 名下恰恰是它该在的地方（用户要的就是在 ncu-ui 里看到 Duration）。
旁路署名会同时给出"文件里的值"和"我们的原值"（如 `37,501,500 ns（我们的值 37.5015 ms × 1e6）`）。

两条纪律仍然不变：**绝不占用硬件计数器命名空间**（`dram__`/`sm__`/`lts__`… 一律不写），
**绝不为没有的量编造行**（ncu 节里只列我们真有值的行，缺的行不写而不是写 0）。

### 4.5 与 nsys / NVTX

nsys 的 CUPTI activity 是我们 `exact` 级数据的来源；此外可以直接吃它的 sqlite：

```bash
kernoscope --ks-sqlite run.sqlite --ks-nvtx-phases --set basic -o rep
kernoscope --ks-nsys-rep run.nsys-rep --set basic -o rep    # 先 nsys export，再照常分析
```

`--ks-nvtx-phases` 把 kernel 时间按 **NVTX 区间重叠**归因到训练/推理阶段
（新 section `NvtxPhases`）：跨两个区间的 kernel 按重叠部分**切开**计入，嵌套区间各自计入
（所以各阶段之和可 >100% —— 表里明说"这不是划分"）；没有 NVTX 数据时那一节仍以 `n/a` + code 存在。

---

## 5. DRAM 字节：两条口径与实测精度

```
bytes ≈ Σ_i ( 显存忙碌_i / 100 × Δt_i ) × 本机实测峰值带宽
```

**口径纪律**（不一起报出来就是过度声称）：

- 报告里必然出现 `ks__sampler_window_coverage_pct`（窗口是否被被测负载主导，`derived`）
  与 `ks__dram_window_mem_busy_mean_pct`（**参与估算的**窗口显存忙碌均值，`derived`，
  分辨率 **1 个百分点**、采样周期 **200 ms**）。整进程的 NVML 均值另见
  `meas__nvml_mem_busy_mean_pct` —— 它是**另一个数**（含启动/空闲），两者不可互换。
- **per-launch 归因有两个前提，不是一个**：
  - 前提① `ks__sampler_window_coverage_pct ≥ 80%`：GPU 在不在干活；< 50% 连总量也不给；
  - 前提② `ks__matched_kernel_kernel_share_pct ≥ 95%`：**被 `-k` 选中的那个 kernel 是不是主角**
    （它占窗口内全部 kernel GPU 时间的比例）。空闲时段贡献 0 字节，所以分母用内核时间、
    不是墙钟；`ks__matched_kernel_window_share_pct` 是与覆盖率同量纲的对照量。
  - 只有前提①时，一个窗口里**顺序跑了多个 kernel/变体**的负载会被判成"可归因"，
    把整窗口字节除到某个 kernel 的 launch 数上，误差可达 **+147%**（真机实测）。
    `ks__attribution_bias_estimate_pct = 100×(1/份额−1)` 把这个上偏**算出来**并跟数字并列。
    95% 这个门槛是从验收线反推的：份额 80% 仍自带 +25% 的偏差，守不住 `|误差|<10%`。
  - 门槛是**代理**（假设各 kernel 字节强度可比）而不是严格上界，实测值可能更高。
    混合强度的窗口（计算密集 + 访存密集混跑）只认**轨迹口径**：访存轨迹只测被追踪的
    那个 kernel，天然不受窗口里还有谁影响。`ks__window_busy_segments` 会告诉你窗口里有
    几段连续忙碌期 —— 多段通常意味着"先跑正确性再跑性能"或"依次跑 v1/v2/v3"这类拼接。
- 显存忙碌读数低于分辨率下限（例如 0.0%）→ 不给字节数，并写明
  "无法区分『真的没搬』与『0.x% 被截断成 0』"。
- 一次采样都没有时，前提行也是 `n/a`（报"0%"等于声称量过）。
- 拒绝时那一行**照样出现**（`n/a` + 具体失败前提），不会从报告里消失；`--strict` 会因此 exit 2。
- 默认 `--ks-window auto`：按首个/末个忙碌采样点裁掉 `import torch`/JIT 的启动开销
  （实测这类启动开销 9–10 秒，会把覆盖率稀释到 15%）。`--ks-window full` 可关闭，
  `--ks-warmup <秒>` / `--ks-window-s <秒>` 可显式限定。
- `-c/--launch-count` 只限制**展示**：per-launch 的分母始终是该 kernel 在本次运行里的真实
  launch 数（分母被截断会带来同样的倍数级误差）。
- 账本对拍**同基比较**：`--ks-ledger-mb` 是"每次 launch"，`--ks-ledger-total-mb` 是整窗口；
  区间 `--ks-ledger-range-mb` **永远按 per-launch**（拿整窗口总量去比一个 per-launch 区间是荒谬的）。
- `--ks-ledger-basis auto|nvml|trace`：NVML 代理口径不成立时（覆盖率不足、主导性不足、
  或显存忙碌读数为 0）**自动改用访存轨迹口径**的 per-launch 字节完成对拍；
  显式指定 `nvml`/`trace` 时不做回退。

```bash
# 引用 est__dram_* 之前先看三个数：覆盖率、窗口主导性、分辨率下限
kernoscope --set basic --ks-calibration machine-profile.json \
           --ks-ledger-mb 234 -k regex:my_kernel -o rep -- ./my_bench
kernoscope -i rep.ksreport.json --csv | grep -E 'ledger|window_coverage|share|mem_busy'
```

### 5.1 代理精度实测（字节数已知的微基准）

[`bench/ledger_probe.cu`](bench/ledger_probe.cu) 的三个 kernel 每 launch 的字节数可以**解析算出来**，
于是可以给代理口径一个真实误差（RTX 4090，8 秒稳态循环，`--ks-window auto`；
产物在 [`bench/artifacts/`](bench/artifacts/)，逐项对照见 §12）：

| 负载 | 账本（解析值 / launch） | 估算 / launch | 误差 | 覆盖率 | 窗口显存忙碌 | 整进程显存忙碌 |
|---|---|---|---|---|---|---|
| 拷贝 `y=x`（2N×4B） | 536.9 MB | 589.3 MB | **+9.6%** | 95.2% | 95.2% | 70.2% |
| 只读归约（N×4B） | 268.4 MB | 257.7 MB | **−4.0%** | 95.2% | 95.2% | 64.5% |
| 只写（N×4B） | 268.4 MB | —（**拒绝**） | n/a | 95.2% | 76.6% | 60.7% |

上表顺便演示了两件事：**「窗口显存忙碌」与「整进程显存忙碌」是两个数**
（参与估算的永远是前者，它有自己的名字 `ks__dram_window_mem_busy_mean_pct`），
以及**两个归因前提都被打印出来**（覆盖率 95.2%、窗口主导性 100%）。

**这个误差带的口径与边界**（不给出口径就等于过度声称）：

- 它是在**字节数解析已知的合成微基准**上实测的范围，取上面各负载的最小/最大值
  （本次 **−4.0% ~ +9.6%**；本次只有两行给数，第三行被拒绝）。**它不是上界**：
  真实负载的访存模式与这三个微基准不同，误差可以更大。同一批微基准在更早的四次运行里
  分别是 **−4.1% ~ +12.5%**、**−4.3% ~ +12.7%**、**−4.3% ~ +9.0%** 与
  **−4.2% ~ +12.8%**（每一次的带都能在 `bench/artifacts/` 的登记表里逐项复核），
  五次运行的并集是 **−4.3% ~ +12.8%**。
- 它是**单边偏高**的：DRAM 代理容易高估（见下）。所以结论要写"落在代理误差带内"，
  不要把"两次方向一致"当成独立证据。
- 它**不是**对称的 ±4% 那种误差，也不要把它当"精度上界"。

**精度 vs 稳态段长度**（同一个已知答案负载，四个时长各测一遍；产物 `proxy-duration-sweep.csv`）：

| 稳态段目标 | 覆盖率 | 窗口主导性 | 估算 / launch | 误差 |
|---|---|---|---|---|
| 1.2 s | 71.4% | 100% | —（**拒绝**） | 前提①不成立 → `n/a` |
| 3 s | 88.2% | 100% | 583.2 MB | +8.6% |
| 8 s | 95.2% | 100% | 587.4 MB | +9.4% |
| 20 s | 98.0% | 100% | 590.1 MB | +9.9% |

这张表要说的是**两件常被搞混的事**：

1. **时长只决定"前提是否成立"，不决定"准不准"。** 1.2 秒时覆盖率 71.4% < 80%，
   工具直接拒绝给数（正确地）；而 3 秒到 20 秒，覆盖率从 88.2% 涨到 98.0%，
   误差却在 +8.6% ~ +9.9% 之间**不收敛**（20 秒那次比 8 秒更大，这是反方向的不收敛）。
   也就是说"多跑一会儿就更准"是错的，正确说法是"多跑一会儿才**有资格**给数"。
2. 同一个时长在不同运行里也会变（20 秒那一点在有产物的三次运行里是 +6.2%、+6.6%，
   本次 **+10.2%**）：**运行间的波动与时长的影响同量级**。所以误差带要按范围引用，
   并且要带上"几次运行"这个口径。

高估的系统性来源是 `utilization.memory` 在高负载下**先于** DRAM 达到可达峰值而饱和
（读数 95% 时实际吞吐约为可达峰值的 87%）。所以：

- 拿它做 "<10%" 这类验收线**只能算勉强**：登记过的运行里 `write` 出现过 +12.5%，越线。
- 想要第二条独立口径就用**访存轨迹**（§5.2），两条路径的比值会作为
  `ks__dram_trace_vs_nvml_pct` 一起报出来。
- 显存忙碌读数落在分辨率下限以下时，工具**不给数字**，并说明原因。

### 5.2 两条口径怎么选

| | NVML 代理（默认，零依赖） | 访存轨迹（NVBit） |
|---|---|---|
| 精度 | 登记过的运行里 **−4.3% ~ +12.8%**（单边偏高） | **地址是算术**：同一负载 235 MB 上差 320 B（≈ −0.0001%） |
| 依赖 | 只要 NVML（驱动自带） | 要自己装 NVBit + 采集脚本（见[探针目录](tools/probes/nvbit_memtrace/)），采集成本 ∝ 流量 × launch 数 |
| 什么时候**拒绝**给数 | 窗口主导性 <95%、覆盖率 <80%、**显存忙碌低于 1 个百分点分辨率**（小 kernel / 常驻 L2） | 轨迹被截断、访问数超上限、缺 L2 容量 |
| 什么时候用它 | 大 kernel 的稳态段：只要判断"是不是 DRAM bound"，带内误差够用，且不用付安装成本 | 要绝对字节、要分辨 <5% 的改动、或上面那三条拒绝正好命中你的负载 |

**轨迹口径是"绝对 DRAM 字节"的首选**：它的地址部分是 measured 级算术（可逐条复核），
只有"哪些访问命中 L2"是模型 —— 所以工具会同时给出**替换策略上下界**，让读者看到模型的影响范围。
代理口径仍然是**默认**：它不需要任何外挂，且在满足前提的大 kernel 上够用；
一旦前提不成立它会明确拒绝（而不是给个偏小的数）。

### 5.3 拒绝长什么样（真实输出）

覆盖率不足（窗口被启动开销稀释）：

```
est__dram_bytes_total      n/a   [unavailable]  窗口覆盖率 25% < 50%：窗口没被被测负载主导，
                                              忙碌时段是否饱和无从确认；把稳态循环延长到约 6 秒…
est__dram_bytes_per_launch n/a   [unavailable]  同上（per-launch 还需要覆盖率 ≥80% 且锁定单一 kernel）
ks__sampler_window_coverage_pct  25 %  [derived]
est__dram_vs_ledger_err_pct n/a  [unavailable]  误差行无法给出：…；可以加 --ks-memtrace <trace>
                                              换用**轨迹口径**的 per-launch 字节
```

`--strict` 在这份报告上返回 **2**（请求了账本裁决却产不出来）。

代理口径不成立时**自动改用轨迹口径**（1 MB 工作集常驻 L2，显存忙碌落到分辨率下限以下）：

```
est__dram_bytes_total             n/a   [unavailable]  显存忙碌读数 0.0% 低于 NVML 的分辨率下限 1 个百分点…
meas__nvml_mem_busy_mean_pct       0 %  [measured]     [resolution 1 %（整数百分比）]
est__dram_bytes_per_launch_trace 0.699 MB [estimated]  第二条口径：NVML 口径不成立时用它做账本对拍
est__dram_vs_ledger_err_pct     -66.7 % [estimated]  账本 2.1 MB；此次用**轨迹口径**，因为 NVML 代理口径不成立
```

−66.7% 不是算错，而是**换口径换出的正确结论**：轨迹模型给出 L2 命中 66.7%
（1 MB 工作集反复读写），于是只有 1/3 的账本流量真正落到 HBM。
这正是第二条口径的价值 —— 它回答的不是"账本差多少"，而是"这个问法在当前工作集下成不成立"。

---

## 6. 无计数器的其它能力

### 6.1 达到的 occupancy（区间重叠法）

ncu 的 Achieved Occupancy 来自计数器；这里用**区间重叠法**重建 —— 让每个 block 记录
`(smid, 起止 %globaltimer)`，时间平均的并发 block 数 × 每 block warps ÷ 每 SM 上限：

```bash
make -C tools/probes run                      # 编译并采集 occupancy-records.txt
kernoscope occupancy --from occupancy-records.txt                      # 看结果
kernoscope occupancy --from occupancy-records.txt --write-json occ.json  # 产出注入用 JSON
# 记录头部缺信息时手动指定：--arch sm_89 --threads 256 --sm-count 128；--kernel 覆盖名字
# （默认全部从记录头部取；--json 给机器读）
kernoscope --ks-achieved-occupancy occ.json -k regex:<kernel> -- your_app
```

真机两点校验（RTX 4090，128 线程/块，产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 场景 | 理论（CUDA API） | 实测（区间重叠） |
|---|---|---|
| 无 smem | 12 blocks/SM = 100% | **11.27 → 93.9%**（差的是 ramp-up/tail） |
| 32 KB 静态 smem | 3 blocks/SM = 25% | **2.99 → 24.9%** |

要测**你自己的 kernel**，把 `tools/probes/occupancy_probe.cu` 里那几行搬进你的 kernel 即可
（文件头注释列了三个必须保留的坑：时间戳要值依赖、跨 SM 只能用 `%globaltimer`、
共享内存必须写后读否则被 ptxas 当死代码删掉）。

### 6.2 发射槽利用率与瓶颈方向（上界口径）

官方 NVBit 的 `opcode_hist` 在默认模式下统计的是**每 warp 每条指令执行一次** ——
这正好是"消耗掉多少个发射槽"。于是：

```
发射槽利用率（上界口径） = 动态 warp 指令数 / (SM 数 × 4 调度器 × 时钟 × 时长)
```

```bash
# 采集（NVBit 是可选外部依赖，需自行安装；NO_EAGER_LOAD=0 是关键）
NO_EAGER_LOAD=0 CUDA_INJECTION64_PATH=$NVBIT/tools/build/opcode_hist/opcode_hist.so \
  LD_PRELOAD=$NVBIT/tools/build/opcode_hist/opcode_hist.so NOBANNER=1 \
  ./your_app > hist.txt 2>&1

kernoscope issue --from hist.txt --kernel my_kernel \
                 --sm-count 128 --clock-mhz 2450 --duration-ms 86.7 --dram-pct 3.2
# 发射槽分母默认按每 SM 4 个调度器（可改：--schedulers <N>）；--json 给机器读
# 或直接并进报告（SchedulerStats / WarpStateStats 两个 section）
kernoscope --set full --ks-instr-hist hist.txt --ks-sm-count 128 --ks-clock-mhz 2450 -- ./app
```

判词是**方向性**的：发射槽很高 ⇒ 偏发射受限；很低而显存很忙 ⇒ 偏显存受限；
两者都低 ⇒ 偏延迟受限（在等数据，不在等发射）。
`smsp__warp_issue_stalled_*` 那类 **stall 周期细分仍然报 `n/a` + 原因** ——
需要硬件计数器，我们不会用方向性判断冒充它。

真机两个负载的对照（RTX 4090，产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 负载 | 动态 warp 指令（每 launch） | 发射槽利用率（上界） | 显存忙碌 | 判词 |
|---|---|---|---|---|
| 256 MB 拷贝（DRAM bound） | 18.4 M | **2.8 %** | 71 % | 偏显存受限 —— 指令在等数据 |
| 2 MB 逐元素 norm（常驻 L2、循环开销占大头） | 0.91 M | **77.0 %** | 低 | 偏发射受限 —— 该减少指令数 |

⚠️ 发射槽利用率的分母是"这段时间里 SM 最多能提供的发射槽"，所以它是**上界口径**：
真实可用发射槽只会更少。分母里的时长必须取自**同等工作量**的运行（报告路径用的是
`gpu__time_duration.sum`；`kernoscope issue` 需要你自己把 CUPTI 时长传进来）。
直方图那次的 launch 数与报告不同时，会按 per-launch 平均对齐，并在 note 里写明。

### 6.3 访存轨迹（NVBit 官方 mem_trace）

```bash
bash tools/probes/nvbit_memtrace/collect.sh -- <你的命令>   # 产出轨迹（体积 ∝ 流量）
kernoscope --set basic --ks-memtrace trace.txt --ks-l2-bytes 50331648 -k regex:<kernel> -o rep -- <你的命令>
# 缓存模型可调：--line-bytes（默认 32 = sector 口径）、--l1-bytes、--l1-ways、--l2-ways
# --max-accesses（默认 4M；超过就拒绝给 DRAM 字节，避免偏小的假值）
```

产出 `meas__sectors_per_request`（直接计数，合并效率 ≤ 100% 有测试守着）、
模型给出的 L1/L2 命中率与工作集、以及第二条口径的 per-launch DRAM 字节。
按采样比放大的轨迹**不是实测**：那时的前缀是 `est__` + `estimated`（命名承诺什么，出处就得是什么）。

### 6.4 静态分析（不需要 GPU）

```bash
kernoscope --set basic --ks-cubin kernel.cubin -o rep          # 寄存器/共享内存/occupancy 上界
kernoscope --set detailed --ks-cubin kernel.cubin -o rep       # 加 SASS 指令混合
```

静态分析**不执行**被测二进制，也不证明任何运行行为。静态混合只说明代码长相，
动态占比要靠 §6.2 的指令直方图。

---

## 7. 成本与处方：先改哪里、能省多少

优化算子是为了**降本**。所以除了"快了多少毫秒"，工具直接给钱的维度。
能耗是**实测**，量与价格由你给 —— 没给就拒绝给那一半，不按"典型值"猜。

```bash
kernoscope --set basic --ks-tokens 8192 --ks-price-per-kwh 0.6 --ks-price-per-hour 2.5 \
           --ks-calibration machine-profile.json --ks-gemm 4096,4096,4096 --ks-dtype bf16 \
           -k regex:gemm -o rep -- python serve.py
```

| 指标 | 含义 | 出处 |
|---|---|---|
| `meas__window_energy_j` / `meas__window_power_mean_w` | 窗口内能耗与平均功耗（NVML 功率读数在窗口内梯形积分） | `measured` |
| `ks__steps_in_window` / `ks__tokens_in_window` | 窗口内等效 step 数与 token 数（分母与能耗**同一个窗口**） | `derived` |
| `ks__tokens_per_s`、`ks__j_per_token` | 吞吐与每 token 焦耳（能耗口径，不需要价格） | `derived` |
| `ks__cost_per_hour`、`ks__cost_per_mtok`（含 `_energy` / `_rental` 分量） | 钱 | `derived` |
| `est__kernel_energy_share_pct`、`est__kernel_energy_per_launch_j`、`est__kernel_cost_per_hour` | 按 kernel 归因（**代理口径**：窗口内时间占比 × 能耗） | `estimated` |
| `kind=Prescription` 的 `ks__prescription_*` | 头寸与排序：保守/上界区间 + ¥/h + ¥/Mtoken | `derived`/`estimated` |

**给一半输入就拿一半结论，缺什么一望可知**：只给功率+电价 → 给 `ks__cost_per_hour`；
再加 `--ks-tokens` → 给单位成本；不给 token 数就**拒绝**给单位成本（`no_tokens`）；
不给价格就拒绝给钱（`no_price`），能耗照给；窗口短于 0.5 s 直接拒绝给能耗
（`below_resolution`），不按 TDP 顶替；半路缺功率采样点报 `no_energy`。
电费与租约是两种**载体**，分别报、都给才相加 —— 所以 `ks__cost_per_mtok` 永远能拆回它的组成部分。

### 处方（`saving_estimate` 判词 + `Prescription` section）

代价是"能省多少"，但它必须建立在**本机实测峰值**上，否则就是一个看起来很像预测的数：

- 头寸 = 本次达成值 vs **本机实测峰值**。保守端是"把差距关掉一半"（`u/(2−u)`，`u` = 差距比例），
  乐观端是"完全填满"（`u`），**明确标注为上界、不是预测**。
- `--ks-peak-*` 覆盖值或规格表**算不出**头寸（`calibration_caliber_mismatch`）——
  "规格 vs 实测"的差距正是本工具存在的理由。
- 没有价格就没有钱（比例照给）；价格有但无法按 kernel 份额归因 → `premise_failed`
  （与"没给价格"的处置不同）。按 token 计的节省**额外**要求被匹配 kernel 主导窗口
  （份额 ≥ 50%）：否则省下的是 GPU 时间，不是 token 成本。
- **重叠的头寸不相加**：同一 kernel 的多个头寸按最大者封顶，被丢弃的量写进 note
  （"全部相加会得到 X"）；跨 kernel 同资源的头寸会显式标注它们其实在争同一份资源。
- 一条都不可辩护时：`ranked == []` + 一条结构化拒绝 —— 不是空输出，也不是一个假数字。

处方排序进决策包（`saving_estimate`，`ok`/`soft` + 拒绝码），也进一屏的「路径」段：
**先改哪个 kernel、保守能省多少、上界是多少、有多确定**。

---

## 8. 比较与 CI

### 8.1 比较两次运行：`diff` + MDD

```bash
kernoscope selftest --out machine-selfcheck.json        # 一次即可：本机误差带 + MDD
kernoscope diff before.json after.json --selfcheck machine-selfcheck.json --json
kernoscope diff a.json b.json --selfcheck machine-selfcheck.json --fail-on regressed   # CI 卡门
```

`diff` 用 MDD 做四态判定：**真改进 / 真退化 / 分不出来（indistinguishable）/ 不可比**。
没有误差带与 MDD 时拒绝给方向（`no_selftest_band`）；两次运行的协议（标定文件、窗口模式、
工具版本、kernel 选择）不同时判 `protocol_mismatch`；基线为 0 时走 `zero_baseline`
（相对差无定义，只给绝对差值）。判据是**相对差 vs 相对误差带**（`%` 单位不再拿百分点去比 MDD）。

### 8.2 接进 CI：`kernoscope gate`

```bash
kernoscope --packet rep.packet.json -o rep -- python serve.py
kernoscope gate --packet rep.packet.json --report rep.ksreport.json \
    --require kernel_time --require-actionable cost_per_token \
    --forbid-code counters_denied --max-metric ks__cost_per_mtok=0.5
```

判据全是**可编程的东西**，不是散文：

| 判据 | 语义 |
|---|---|
| `--require ID[=ok\|soft\|refused]` | 判词必须存在且状态达标（默认 `ok`）；不存在即失败（被尺寸预算裁掉也算） |
| `--require-actionable ID` | 判词可作为行动依据（`actionable=true`；v1.0 包回退到"不是 refused 且前提全成立"）。包里的 `actionable` 必须是**真布尔**：字符串 `"false"` 是**用法错误 rc=2**，绝不被强转成通过 |
| `--forbid-code CODE` | 出现该拒绝码即失败。搜索范围是**包里任何能携带 code 的位置**：判词、`refusals` 索引、`unmeasured`、`refusals_omitted`、环境拒绝与 `environment_refusals_more`（只查其中一处会让门禁在受限机上静默放绿） |
| `--max-metric NAME=V` / `--min-metric NAME=V` | 指标阈值，**逐 kernel** 判：只要一个真 kernel 带该指标，每个真 kernel 都必须带且达标；缺项 / `n/a` / 超限一律不通过（"没有数"不是"没超阈值"）。只有 `(device)` 伪 kernel 带的指标（如 `ks__cost_per_mtok`）按设备级判 |

退出码 `0` 通过 / `1` 不通过 / `2` 用法错误；**一条规则都没给时失败** ——
门禁凭空绿灯是它最坏的失败模式。指标规则写错（`--max-metric X=abc`）也退 `2`；
用法错误配 `--json` 时 stdout 仍给 `usage_errors`（rc 不变）。`--json` 给出逐条失败原因。门禁只消费已经产出的包与报告，
不跑测量（要新数据先跑一次采集，再对新的包重跑门禁）。

---

## 9. 输出形态：一屏、决策包、案件目录、报告

| 输出 | 开关 | 用途 |
|---|---|---|
| ncu 风格文本 | 默认 | 人眼/日志；每条指标带 `[exact]`/`[estimated]`/`[unavailable]` 标签与分辨率 |
| 一屏（三段式） | `--screen only`（默认铺在报告前） | 结论 → 发现（判词与前提）→ 路径（按挡住决策的程度排序）→ 本次代价 |
| 决策包 | `--packet <f>`（`-` = stdout） | agent 用：判词 + 前提 + 结构化拒绝 + 代价，**≤4 KB 预算** |
| Markdown 报告 | `--report-md rep.md` | 人读的 per-kernel 报告（0–6 固定结构） |
| CSV | `--csv` | 表格分析；多出 `provenance` / `resolution` 列 |
| JSON | `-o rep` → `rep.ksreport.json` | 机器消费、回归对比（带 `schema_version`） |
| Perfetto trace | `-o rep` → `rep.perfetto.json` | 丢进 <https://ui.perfetto.dev> 看时间线，零安装 |
| 逐 launch CSV | `--launch-csv`（配 `-o` 时写 `rep.launches.csv`） | 每次 launch 的 grid/block/时长/配置键 |
| **案件目录** | `-o rep` → `rep.case/` | 跨会话/交接：`manifest.json`（任务与输入指纹）、`journal.jsonl`（追加式调用日志）、`next.md`（下一步）、`env.json` |
| 真值对照（**包装逻辑已验；真报告口径未验证**，见 §4.3） | `--ks-ncu-rep …` + `-o rep` | `rep.ncu-truth.json`：真值、对拍偏差、每条不可比的 reason |

一屏的顺序是稳定接口（`结论：` → `-- 发现（判词与前提）--` → `-- 路径（下一步…）--` →
`本次代价：`），便于人和 agent 直接解析。

决策包里还有**三条环境前提**（它们决定"这次的数字在什么环境下成立"）与它们的门槛：

| 前提 | 门槛（`packet.py` 的具名常量） | 观测不足时的行为 |
|---|---|---|
| `env_gpu_exclusive` | `ENV_OTHER_PROCS_MAX = 0` 个其他计算进程 | 探测**失败**时报 `meas__other_procs_on_gpu = unavailable` + `env_reading_unknown`，前提按**不成立**处理（"没测到"≠"0 个别的租户"） |
| `env_gpu_busy_pct` | `ENV_GPU_BUSY_MIN_PCT = 80 %` | 位图读不到时用 SM 利用率兜底判空闲，并在 note/source 里写明是兜底口径 |
| `env_no_throttle_pct` | `ENV_NO_THROTTLE_MAX_PCT = 5 %` 降频时间 | 整段读数都读不到时报 `unavailable` + `env_reading_unknown`，前提**不成立**（不印 `100 %`） |

任一前提不成立 ⇒ 依赖它的判词降级为 `refused` 并带着对应 code（`cost_per_token` 因此不会在
环境不干净时还保持 `actionable`）。「路径」段的排序是确定的：**先列被拒的判词**
（它们挡住了决策），再列成立判词自带的提示，最后列未测到的组。

案件目录解决的是"一次优化要跨会话做"：`next.md` 说下一步做什么，`journal.jsonl` 留下
每次调用的 rc 与拒绝码，`manifest.json` 让你能确认"用的是同一份标定/轨迹"。`--no-case` 可关。

决策包是最小可交接单元：它只放**每次都变**的东西（判词 id、状态、动态数值、前提观测值、代价），
静态文本按 id 查表；`cost.trimmed` 如实记录为了压进预算裁掉了什么。真机教训写在实现里：
facts 分两级裁，`kernel_time`/`cost_per_token`/`saving_estimate`/`truth_vs_substitute`
这四条的 facts（"什么最贵、多少钱、先改哪里、真值偏差"）**最后才裁**。

### 9.1 输出体积可预算

小 kernel × 长循环会产生十万级 launch，逐 launch 明细会把报告撑爆。真实数字
（同一次运行，产物在 [`bench/artifacts/`](bench/artifacts/)）：

| 输出方式 | 报告体积 |
|---|---|
| 默认 JSON（含逐 launch 数组） | **138.1 MB** |
| `--ks-json-no-launches -c 50` | **25.4 KB** |

所以：`-c <N>` 限流每个 kernel 只报 N 次、`--ks-json-no-launches` 让 JSON 不带逐 launch 数组、
逐 launch 明细改用 `--launch-csv` 单独出表；报告写盘后 stderr 会打印实际体积与限流建议。

---

## 10. 副作用与同意：`--plan`

profiler 的副作用不平凡：它会**跑你的命令**、启动 `nsys`/`cuobjdump`、在临时目录写中间产物、
在你的目录写报告。所以先看计划再决定：

```bash
kernoscope --plan -o rep -- python serve.py      # 人读：每一步会起什么进程/写什么文件
kernoscope --plan-json -o rep -- python serve.py # agent 读：8 项 effects + 4 项推导出的 annotations
```

- **`--plan` 只打印**：不跑你的命令、不采集、不写任何文件（有测试守着）。
- 每个操作都必须有 effects 声明，**缺了就抛**（不是文档里提一句）；四个 MCP 风格
  annotation（`read_only_hint`/`destructive_hint`/`idempotent_hint`/`open_world_hint`）
  由 effects **纯推导**，禁止按命令名猜。
- 测试还会用 PATH 上的记录桩把"实际起了哪些子进程"记下来与计划对照 ——
  这条对账第一次跑就抓到三个真缺陷（漏声明 doctor 的只读探测、输出目录不存在时裸 traceback、
  计划里的输出文件名与实际写的不一致）。
- 需要你点头的只有写文件与改变被测对象；只读的外部查询（`nvidia-smi --query-gpu`、
  `<tool> --version`）不算，但仍然会如实标出"它会起进程"。
- 我们**不注入**代码到被测进程；静态分析不执行被测二进制。

---

## 11. 局限与已知失败

这一节是**我们主动交代的失败面**。它不是免责声明，而是"你会遇到什么、我们会怎么说"的清单。

### ① 计数器拿不到（根因，不是 bug）

宿主 `RmProfilingAdminOnly=1` 且容器无 `CAP_SYS_ADMIN`/`CAP_PERFMON` 时，CUPTI 直接报
`ERR_NVGPUCTRPERM`。计数器是 host-wide 的、会泄露同机租户行为，**容器内没有任何合法绕过**。
所以 `dram__bytes.sum`、`lts__t_sector_hit_rate.pct`、`smsp__warp_issue_stalled_*` 这些名字下
永远只有 `n/a` + 原因；`--strict` 请求它们时 exit 2。想要精确计数器，只能在有权限的机器上跑 ncu。

### ② NVML 代理口径是**单边偏高**的

DRAM 字节代理 = 显存忙碌时间 × 本机实测峰值带宽。`utilization.memory` 在高负载下
**先于** DRAM 达到可达峰值而饱和（读数 95% 时实际吞吐约为可达峰值的 87%），
所以误差不是对称的：本次三个已知答案负载里两行给数（**−4.0% 与 +9.6%**），
第三行（只写）因**对齐自检没通过而拒绝**给 per-launch（`window_launch_unaligned`，见产物
`rep-write.txt`）—— 那正是 R-47 的修法在做的事：宁可拒绝，也不给一个分母不落在窗口里的数。
（逐次运行与并集见 §5.1 —— 那里是唯一一处列出各次带的地方，免得同一句话在文档里出现两遍、
然后各自漂。）**它不是上界，也不是对称误差**：真实负载的访存模式与微基准不同，误差可以更大；
引用时必须写"落在代理误差带内"，而不是"两次方向一致所以可信"。

### ③ 分辨率是 **1 个百分点**

`utilization.memory` 是整数百分比，采样周期 200 ms。读数 0.0% 时，
"真的没搬数据"与"0.3% 被截断成 0"**不可区分** —— 此时工具不给字节数，并写明这一点。
别的量也有各自的分辨率（例如归因上偏估计的分辨率是 0.1 个百分点），
报告里 `resolution` 列与数字同时出现。

### ④ 跨运行噪声与 MDD

同一负载、同一时长，不同运行的误差会变（20 秒那一点在有产物的三次运行里是 +6.2%、
+6.6% 与本次 +9.9%）：**运行间波动与时长的影响同量级**。所以：

- `kernoscope selftest` 只依赖 torch，量出**本机**误差带与 **MDD**（最小可分辨差异）；
- `kernoscope diff` 用该 MDD 做三态比较：真改进 / 真退化 / **分不出来（indistinguishable）/ 不可比**；
- 没有误差带与 MDD 时，`diff` 拒绝给方向判词（`no_selftest_band`），而不是报"变了"；
- 两次运行的协议（标定文件、窗口模式、工具版本、kernel 选择）不同时，判 `protocol_mismatch`。

### ⑤ 哪些问题**必然被拒**

| 被拒的问题 | 拒绝码 | 为什么必然 | 替代 |
|---|---|---|---|
| warp stall 周期细分 | `counters_denied` | 与 ncu 同源，需要计数器 | 发射槽上界 + 方向性判断（`issue_slot`） |
| 精确 DRAM 字节 / L2 命中率 | `counters_denied` | 同上 | DRAM 代理（带误差带）或访存轨迹（模型） |
| 各 pipe（FP32/INT/Tensor）动态利用率 | `counters_denied` | 同上 | 发射槽上界 + roofline 达成百分比 |
| PC 采样 / SourceCounters | `sampling_unavailable` | 与计数器同源、同样被锁死 | 指令直方图的发射槽上界 |
| NVLink 流量与利用率 | `no_nvlink` | 单卡/无 NVLink 实例上本来就不适用 | 不适用（忽略这一行） |
| SASS/源码逐行对照 | `no_sass_source` | 本工具不实现 | `cuobjdump -sass` 自行对照 |

（完整清单与每台机器的实际三态见 `kernoscope capabilities --json`，或用
`kernoscope capabilities` 看一屏；本仓库的静态版本见 [COVERAGE.md](COVERAGE.md)。）

### ⑥ 「我们拒绝给数」的清单

按拒绝码（闭集，`kernoscope capabilities --json` 里可编程读取）：

- **物理上拿不到** → 只报 `n/a`：`counters_denied`、`sampling_unavailable`、
  `counters_not_installed`、`counters_unknown`、`no_nvlink`、`no_sass_source`。
- **口径前提不成立** → 那一行照样出现，但值是 `n/a`：
  `coverage_too_low`（窗口没被负载主导）、`share_too_low`（被选中的 kernel 不是主角）、
  `multi_kernel_window`（窗口里有多个 kernel，不可归因）、
  `below_resolution` / `mem_busy_zero`（读数落到分辨率下限以下）、
  `no_calibration`（缺本机实测峰值，不用规格表顶替）、
  `calibration_caliber_mismatch`（标定口径与本次测量不可比）、
  `unknown_arch`（未知架构算不出 occupancy 上界，不猜）、
  `trace_too_many_accesses`（访问数超上限时不给偏小的值）。
- **缺的是**输入**不是环境** → 说清该补哪个开关：`no_tokens`（每步 token 数）、
  `no_price`（电价/租约）、`no_energy`（窗口功率读数）、`no_workload`（FLOPs/bytes）、
  `no_memtrace`、`no_instr_hist`、`no_occupancy_probe`、`no_static_input`。
- **比较不可比** → 不给方向判词：`protocol_mismatch`、`both_refused`、
  `no_selftest_band`、`mdd_unknown`（没有 MDD 就说"分不出来"）、`zero_baseline`。
- **请求的东西不存在** → `metric_absent`（不等于被拒）、`legacy_report`
  （旧报告没有结构化拒绝信息，标注 legacy 而不猜原因）、
  `ncu_report_unreadable`（`.ncu-rep` 读不了：缺 Nsight Compute 的 Python 模块或文件损坏）。
- **兜底** → `internal_unknown`：出现即视为本工具缺陷，请按 issue 模板报告。

完整清单（含每个 code 的处置与代价）由代码生成到
[`skills/kernoscope/references/refusal-codes.md`](skills/kernoscope/references/refusal-codes.md)，
人读的策略表见
[`skills/kernoscope/references/diagnosis-playbook.md`](skills/kernoscope/references/diagnosis-playbook.md)
（信号→原因→处置、永久拿不到、任务→该跑哪条检查）。

### ⑦ 其它已知的"做不到"

- **不重放、不锁频**：所以绝对耗时与 ncu 的锁频读数不可直接对比。
- **per-launch 归因依赖窗口构成**：混合强度的窗口只认轨迹口径（见 §5.1）。
- **轨迹体积与流量成正比**：几十 MB 工作集可能产出上 GB 文本。
- **fp32 roofline 拐点可能不可分辨**：标定的口径离散度大时，工具拒绝用它算拐点。
- **处方只给"这个 kernel 的时间/钱"的区间**，不给端到端收益预测：端到端要自己用
  `diff` + MDD 复核（kernel 变快不等于端到端变快，除非它是关键路径）。

### ⑧ 导出的 `.ncu-rep` 的已知边界

- 官方 Python API 能按**名字与值**读回我们写的指标，`ncu --import` 能认出 kernel 行与几何，
  ncu-ui 能在无头 X（Xvfb + 软件渲染）里打开我们的文件并在结果表里显示
  `Duration`/`# Registers`/`Grid Size`（M5 已用人眼截图验收）。
- `--page details` 的机制在原厂报告上是"**按 rule 驱动**"：某个节只有当 `RuleResults[].SectionIdentifier`
  指向它时才会被渲染（我们按这个机制为每个节写了一条 rule，所以**节标题与我们的数都能显示成文本**）。
  但那个页面的**表格行**只在报告的指标集"完整到能被 ncu 自己的节定义解析"时才出现 ——
  在替代口径报告上不可能满足，**也没打算靠编造计数器行去满足**。所以详情页给的是
  "节标题 + 同源值文本 + 署名/WRN"，完整的逐指标数值在 `--page raw`、Python API 与旁路署名里。
  这三条路径都有真机输出钉住（README §12 的运行记录）。
- `ncu` 首页的设备行会显示 `Device 255, CC 0.0`：那个值来自 `ProfileResult.PartitionDeviceInfo*`
  —— proto 里只是一个**不透明的 `bytes`**（PerfWorks 私有编码，随包 proto 里没有它的定义），
  我们**不编造**它。真实设备与计算能力写在文件内的 `SessionDetails.DeviceAttributes`、
  RuleResult 文本与旁路署名里（"device: NVIDIA GeForce RTX 4090, compute capability 8.9"）。
  结果块与源码块之间的 `SourceData.Reference` 我们按 proto 语义写一致（真报告也这么做），
  但仅此不足以让设备行有值。
- 官方 `IMetric.unit()` 对我们写的指标返回空串：**单位在文件里，但不参与回读** ——
  所以"可回读的那一对"是名字 + 值，单位要看旁路署名。
- 文件内的 `Comment`/`Provider` 是我们**唯一**的署名载体之一（Python API 读不到 comment），
  所以同时写 `<path>.ATTRIBUTION.txt` 旁路署名；两条路都留。

### ⑨ NVTX 阶段归因的 schema 假设

`NVTX_EVENTS` 的列集、`end` 非空（push/pop 区间）、`eventType` 数值不解释、
`StringIds` 文本解析 —— 这些是按 nsys 的通用形状写的，并有合成 schema 的测试
（跨区间切开、嵌套各自计入、缺 NVTX 时给 reason + 零数字），但**没有在真实 nsys 导出上跑过**。
真机验证只覆盖了 `CUPTI_ACTIVITY_KIND_KERNEL` 那一侧。

---

## 12. 数字从哪来

README 与文档里每个"实测"数字都能追到一次可复现的运行：

```bash
bash bench/run.sh              # 全量（约 10 分钟 GPU 时间）
bash bench/run.sh --quick       # 快速版
```

`bench/run.sh` 会做：环境快照（**修订号 + 来源 + 是否脏树 + 工具版本**）→ `doctor` →
本机标定 → 编译已知答案微基准 → DRAM 代理精度 → **代理精度 vs 稳态段长度** →
账本对拍 → **覆盖率不足时的裁决行** → 形状扫描 → occupancy 探针 →
访存轨迹口径对拍 → 发射槽上界；每步的退出码写进产物目录的 `manifest.json`，
产物是机器可读文件（不是往 README 里手抄数字），最后一步自动脱敏。

`tests/test_doc_artifact_consistency.py` 把「文档数字 ↔ 产物」变成门禁：文档里公开的
每个实测数字都要能在最新产物里找到同一写法，产物必须钉住修订号与工具版本，
文档引用的每个产物目录都必须存在。数字一变、文档没跟着变，测试直接红 ——
所以本文里那些看起来"太具体"的数字都不是散文，是**被测试钉住的产物读数**。

本文引用的数字与它们的产物：

| 本文引用的数字 | 产物 |
|---|---|
| bf16 / fp32 TFLOP/s、copy 带宽、口径离散度与逐变体设置 | `bench/artifacts/*/machine-profile.json` |
| 代理误差与覆盖率（三个已知答案微基准） | `bench/artifacts/*/rep-{copy,read,write}.txt` 与 `.csv` |
| **精度 vs 稳态段长度**（1.2/3/8/20 秒四个点） | `bench/artifacts/*/proxy-duration-sweep.csv` |
| 覆盖率不足 ⇒ 裁决行 `n/a` 且 `--strict` = 2 | `bench/artifacts/*/rep-lowduty.txt`、`manifest.json` |
| 形状扫描三组配置的逐组时长 | `bench/artifacts/*/rep-shape.txt`、`shape-sweep.launches.csv` |
| occupancy 探针两组读数 | `bench/artifacts/*/occupancy.json`、`occupancy-smem32k.json` |
| 报告体积对比 | `bench/artifacts/*/size-comparison.csv` |
| 发射槽利用率与两个方向的判词 | `bench/artifacts/*/rep-issue-*.txt` |
| 轨迹口径对拍与 L2 命中率（**是哪份 `mem_trace.so` 采的**看 `environment.txt` 的 `memtrace_so` / `memtrace_so_version`，别拿 `nvbit_version` 猜） | `bench/artifacts/*/rep-tracebasis.txt`、`memtrace.json`、`environment.txt` |
| doctor 的 ncu 四态结论 | `bench/artifacts/*/doctor.json`、`doctor-path-stripped.json` |
| 自检误差带与 MDD（含跨进程复测） | `bench/artifacts/*/machine-selfcheck.json` |
| 同协议两次运行的三态判词 | `bench/artifacts/*/diff-same-protocol.json` |
| 每份产物是哪个修订产生的 | `bench/artifacts/*/environment.txt`、`manifest.json` |

`bench/issue_probe.sh` 另采发射槽上界的产物（需要 NVBit）。
本轮新增功能（成本/处方/`.ncu-rep` 双向/`gate`/报告/skill）的真机验证在 RTX 4090 +
Nsight Compute 2025.3.1 上做过，但**没有作为产物入库**：它们的数字不要当作登记过的证据引用，
要引用就在你的机器上自己跑一遍（工具会给出同样的口径与拒绝条件）。

**哪些证据不入库、因而不构成可复核证据**：本地工作目录下的 `acceptance/` 一整套 m6 真机证据
（`acceptance/m6-real-gemm.ncu-rep` 与它的 `.ATTRIBUTION.txt` 署名、`m6-raw.txt`、`m6-details.txt`、
`m6-api.txt`、ncu-ui 截图）**克隆者与 CI 都看不到，也不参与任何门禁** —— 它们只能当"当时跑过"
的本地记录，不能当可复核证据；要复核请在自己的机器上按同一条命令重采。
（原因：工作目录**整目录**在 `.gitignore` 里被忽略，`tests/test_repo_hygiene.py` 守着"不许有取反行"。）

---

## 13. 拿不到数字时先看这里

| 你看到的现象 | 真实原因 | 怎么做 |
|---|---|---|
| `n/a` + "requires GPU hardware performance counters" | 宿主 `RmProfilingAdminOnly=1` 且容器无 `CAP_SYS_ADMIN`/`CAP_PERFMON` | 换物理观测量：本机标定 + DRAM 代理 / 访存轨迹；不要去找"绕过权限"的办法 |
| `doctor` 说"未安装 ncu"，但你 `ls` 到了它 | 工具只查了 PATH（旧行为） | 现在会给出四态结论与路径；`doctor --json` 的 `tools` 里是"找过哪些目录" |
| `est__dram_bytes_*` 全是 `n/a` | 窗口覆盖率 < 50%（启动开销把窗口稀释了），或显存忙碌读数低于分辨率下限 | 看 `n/a` 行给的具体前提；加 `--ks-window auto`（默认已开）、把稳态循环按**目标秒数**加长 |
| 账本误差行是 `n/a` 而不是数字 | per-launch 口径要求"单一 kernel + 覆盖率 ≥80%" | `-k` 锁定一个 kernel；或改用 `--ks-ledger-total-mb`；或加 `--ks-memtrace` 走轨迹口径 |
| 误差百分比大得离谱（百万级） | 口径混用（per-launch 账本 vs 整窗口估算） | 同基比较：`--ks-ledger-mb` 对 per-launch，`--ks-ledger-total-mb` 对总量，区间永远 per-launch |
| roofline 的 compute 侧是 `n/a` | 标定文件缺 `acc_caliber`/`precision_settings`/`method` | 用新版 `tools/machine-profile.py` 重新标定（带宽侧不受影响） |
| 单位成本是 `n/a` + `no_tokens` / `no_price` | 缺"每次迭代的 token 数"或价格（我们**不猜**典型值） | 加 `--ks-tokens <N>` 与 `--ks-price-per-kwh` / `--ks-price-per-hour` |
| 处方一条头寸都没有 | 没有本机实测峰值（标定），或本次没算出达成值 | 跑 `tools/machine-profile.py` 并加 `--ks-calibration`；工作负载侧加 `--ks-gemm`/`--ks-flops` |
| `--metrics dram__bytes.sum` 没给那个名字 | 我们不占用计数器指标名（规则 R3） | 看它给的替代口径（`est__dram_bytes_total`）；`--export-name-map` 拿完整对照表 |
| `.ncu-rep` 读不了（`ncu_report_unreadable`） | 缺 Nsight Compute 的 Python 模块（或文件损坏） | 把 `<nsight-compute>/extras/python` 加进 `PYTHONPATH`；读报告不需要 GPU 与计数器权限 |
| 报告 JSON 上百 MB | 逐 launch 明细随 iteration 数线性增长 | `-c <N>` 限流、`--ks-json-no-launches`、逐 launch 明细改走 `--launch-csv` |
| 轨迹文件几个 GB | 记录数 ∝ 流量 × launch 次数（插桩不区分预热） | 用小工作集 + 小的预热次数（见 [`bench/README.md`](bench/README.md)） |
| 形状扫描只剩一行均值 | 同名 kernel 的多组 launch 配置被聚合 | 现在会逐组给出 `LaunchConfigs`；`launch__grid_size` 在混合时改报 `n/a` + 取值集合 |
| `--strict` 一直红 | 请求了产不出来的数据（含"账本裁决行算不出来"、成本输入算不出对应数字） | 这正是它的用途；`--strict` 的报错会逐条说明是哪个请求、为什么 |
| `gate` 报"一条规则都没给" | 门禁不许凭空通过 | 至少给一条 `--require`/`--forbid-code`/`--max-metric` |

日常先跑这两条定位：`kernoscope doctor`（这台机器能答什么）与
`kernoscope capabilities --json`（哪些问题被拒、code 是什么、补救代价多少）。
单个 code 的处置用 `scripts/remedy.sh <code>`，不要通读整张表。

---

## 14. 状态与路线

- ✅ **MVP**：ncu 兼容 CLI、LaunchStats/时长（exact）、理论与达到 occupancy、实测峰值 roofline、
  设备级计数器（能耗/显存忙碌）、四路输出、`--strict`、provenance 强制。
- ✅ **v1.5**：`kernoscope doctor` —— 一条命令给出「能用/不可用清单 + 原因 + 已选后端 + 可交付出处」，
  并用 ncu 实测确认计数器门；工具发现多路径，ncu 结论分四态。
- ✅ **v1.6**：DRAM 代理 + 账本对拍（per-launch / 总量 / 区间三种口径，同基比较，
  口径不成立必然出现 `n/a` + 原因），`--ks-window auto` 稳态窗口，轨迹口径回退。
- ✅ **v1.7**：形状扫描、`--launch-csv`、`--metrics` 真实筛选、标定口径与离散度、
  报告体积可预算、发射槽上界与瓶颈方向（`kernoscope issue`）。
- ✅ **v1.8**：结构化拒绝（闭集 code + observed/threshold + remedy + 代价）、
  `capabilities --json` 与三态覆盖矩阵、`selftest`（本机误差带 + MDD）、`diff` 四态比较。
- ✅ **v1.9**：`route`（先路由再动手）、三段式一屏与案件目录、
  **单位成本与处方**（能耗实测 + 你给的量与价格 → ¥/Mtoken 与排序后的头寸）、
  **`--plan` 副作用账**、**`gate` CI 门禁**、
  **`.ncu-rep` 双向**（导入真值对拍 —— **包装逻辑已验；真报告口径未验证** / 导出带署名的替代口径）、
  **NVTX 阶段归因**、`--report-md` per-kernel 报告、`kernoscope skill` 安装。
- 🚧 v1 剩余：访存轨迹的输入侧复用分析（locality/复用距离）、`cuobjdump -sass` 静态指令混合
  细化到 per-kernel、真机 nsys 导出上的 NVTX 归因验证。
- 🚧 v2：因果链与 `explain`（借 ingero 思路，用 CUPTI activity + host 采样，不依赖 eBPF）、
  MCP 形态（把同一份判词暴露成工具调用）、多客户端安装矩阵。

---

## 15. 文档与声明

- [COVERAGE.md](COVERAGE.md)：三态覆盖矩阵 + 四层能力分层 + 必然被拒的问题（由代码生成）。
- [CONTRIBUTING.md](CONTRIBUTING.md)：适配器清单与扩展点（指标映射 / 后端 / 能力问题 / 拒绝类型 / 产物）。
- [PROVENANCE.md](PROVENANCE.md)：五个出处等级、四条强制规则、命名纪律、拒绝给数的形态。
- [CALIBRATION.md](CALIBRATION.md)：本机峰值标定的字段与口径纪律（roofline 与头寸都依赖它）。
- [`skills/kernoscope/SKILL.md`](skills/kernoscope/SKILL.md)：给 agent 的入口（触发条件 + 三步闭环）。
- [`skills/kernoscope/references/diagnosis-playbook.md`](skills/kernoscope/references/diagnosis-playbook.md)：
  受限环境诊断手册（由代码生成：拒绝码 → 处置 → 代价；哪些是永久拿不到）。
- [NOTICE](NOTICE)：第三方与商标声明。许可见 [LICENSE](LICENSE)（Apache-2.0）。

Kernoscope is **not affiliated with, endorsed by, or sponsored by NVIDIA Corporation.**
"NVIDIA", "CUDA", "Nsight" and "ncu" are used descriptively only.
