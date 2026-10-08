# 贡献指南（适配器清单与扩展点）

Kernoscope 的形态是「**探测 → 选最高可用层 → 采集 → 每个数字带出处**」。
所以"扩展"几乎总是落在下面六类适配器之一。每类都只有一个单一真源，
**改代码就不会漏**；漏了会有测试直接红。

| 适配器 | 单一真源 | 你会动到的文件 | 守它的测试 |
|---|---|---|---|
| ① 指标映射（ncu 名 → 我们的名字 + 出处） | `src/kernoscope/provenance.py` | `provenance.py`、`sections.py` | `tests/test_provenance.py`、`tests/test_output_consistency.py` |
| ② 后端（nsys / NVML / cuobjdump / NVBit） | `src/kernoscope/backends/` | 对应后端模块、`doctor.py`、`capabilities.py` | `tests/test_*_parse.py`、`tests/test_doctor.py` |
| ③ 已知答案负载（selftest 那一族） | `src/kernoscope/selftest.py` | `selftest.py` | `tests/test_diff_selftest.py` |
| ④ 能力问题 | `src/kernoscope/capabilities.py` | `capabilities.py`、重跑 `tools/gen_coverage.py` | `tests/test_capabilities.py`、`tests/test_opening_package.py` |
| ⑤ 拒绝类型 | `src/kernoscope/refusals.py` | `refusals.py` | `tests/test_refusals.py` |
| ⑥ bench 产物 / 公开数字 | `bench/run.sh` + 产物 | `bench/`、`bench/artifacts/README.md` | `tests/test_doc_artifact_consistency.py`、`tests/test_repo_hygiene.py` |

开始之前：

```bash
uv sync
uv run pytest -q                    # 必须全绿
uv run kernoscope --help            # 确认你的开关真的存在
uv run python tools/gen_coverage.py --check   # 文档与代码是否同源
```

---

## ① 指标映射：把 ncu 指标名映射到我们的名字与出处

**目标不是"多给一个数"，而是"这个数怎么来的能被读者复核"。** 所以每条映射同时要定三件事：
**名字**、**出处**、**来源字符串**。

### 命名与出处规则（改映射前先读一遍）

- **R1** `unavailable` 的指标必须 `value is None` 且有 `reason`：区分"没有这一项"与"这一项是 0"。
- **R2** 计数器命名空间（`sm__` / `smsp__` / `l1tex__` / `lts__` / `dram__` / `nvlink__` …）
  下**不允许出现数字**。当前实现更严：这些名字即使声明 `exact` 也不放行，
  除非登记进 `EXACT_COUNTER_ALLOWLIST`（当前为空）。
- **R3** 非 `exact` 的指标必须使用自有前缀：`est__` / `meas__` / `model__` / `ks__`。
  这条堵死"在 ncu 指标名下面填一个模型算出来的数"。
- **R4** ncu 的 `launch__*` 与 `gpu__time_duration*` 允许 `exact`（确实同源可得）。
- **R-12（命名纪律）** `meas__` 前缀的指标，出处必须是 `measured`。由实测样本**再算**出来的
  统计量（占比、均值、判词）用 `ks__` + `derived`，不要用 `meas__`。

**ncu 的名字写在 `source` 字段里，不写进 `name`。** 例：

```python
# 对应 ncu --metrics dram__bytes.sum；我们只有代理口径，所以名字是 est__，
# 出处是 estimated，来源字符串写清"我们是怎么算的"。
s.add(Metric.estimated(
    "est__dram_bytes_total", value, "MB",
    source="NVML utilization.memory × 本机实测峰值带宽（对应 ncu dram__bytes.sum）",
    note="代理口径；分辨率与采样周期随行给出",
))
```

现成的构造入口在 `provenance.py` 的 `Metric`：`exact` / `derived` / `measured` / `estimated` /
`unavailable`。`unavailable` 必须带 `code=`（拒绝码，见 ⑤）。构造时 `validate()` 会强制
R1–R4 与 R-12；违规直接抛 `ProvenanceError`，不会"先写进去再修"。

子命令（`issue` / `memtrace` / `occupancy`）的 `--json` 指标对象用
`provenance_for_name(name, value)` 定级，与主报告同构（未登记前缀一律回落到 `estimated`，
**绝不回落到 `exact`**）。

### 加一条映射的步骤

1. 在产出该读数的 section builder（`src/kernoscope/sections.py`）里用上面的构造入口加指标；
2. `source=` 必须写清**两层**来源：我们的物理观测量 + 它对应哪个 ncu 指标；代理量还要给
   `note=`（口径/前提）与 `resolution=`（读数下限，例如 NVML 显存忙碌是 1 个百分点、200 ms）；
3. 若这个名字落在计数器命名空间，唯一的合法写法是 `Metric.unavailable(..., code=RefusalCode.NO_COUNTERS)`
   —— **不要**为它编一个 `est__` 名字来绕过：那会让读者以为拿到的就是那个读数；
4. `tests/test_provenance.py` 与 `tests/test_output_consistency.py` 会检查四个输出口的
   provenance 一致（文本 / CSV / JSON / `-i` 回读）。

---

## ② 后端：每一族数据源加什么、怎么加

一个后端 = 一个模块 + 一次探测登记 + 若干指标。后端**只负责取原始数据**，
判定（能不能用、缺什么）在 `capabilities.py`，出处判定在 `provenance.py`。

| 后端 | 文件 | 数据源 | 加什么 | 加它需要 |
|---|---|---|---|---|
| nsys | `backends/nsys.py` | CUPTI activity（`CUPTI_ACTIVITY_KIND_KERNEL` 表） | kernel 级 exact 量：时长、launch 配置、寄存器/共享内存 | `nsys`；输入 `--ks-sqlite` / `--ks-nsys-rep` |
| NVML | `backends/nvml.py` | NVML / `nvidia-smi`（管理计数器，免 admin） | 设备级 measured 量：显存忙碌、SM 时钟、功耗、能量 | `nvidia-smi`，或 `pynvml`（`kernoscope[nvml]`） |
| cuobjdump | `backends/cubin.py` | cubin/ELF 静态 | 静态 exact 量：寄存器、静态 smem、SASS 指令分类 | `cuobjdump`；输入 `--ks-cubin`；**不需要 GPU** |
| NVBit | `backends/memtrace.py`、`backends/opcode_hist.py` | 用户自行编译的 NVBit 工具产出的**文本** | 访存轨迹（sectors/request、命中模型）、动态指令数（发射槽上界） | 用户自己的 NVBit；我们**不打包、不 vendor** |

### 加一个后端的检查清单

1. **模块**：在 `backends/<name>.py` 里把原始格式解析成 dataclass，格式错误抛**可读**异常
   （不要裸 `KeyError`）；解析器要能校验工具侧写错的字段（见 `memtrace.py` 用 `addrs` 重算 `sectors`）。
2. **发现**：路径发现一律走 `toolchain.find()` / `discover()`（PATH → `$CUDA_HOME/bin` →
   `/usr/local/cuda*/bin` → 常见目录）。自己写一套 PATH 查找会出现"doctor 说没有、后端却找到了"。
3. **探测**：在 `doctor.py` 的工具发现字典里登记它（`doctor --json` 的 `tools` 会列出
   "找过哪些目录"与四态结论）。
4. **能力**：`capabilities.py` 的 `Env` 加一个字段，`Env.tier` 里体现它属于哪一层，
   并在 `CHECKS` 里让相关问题消费它；缺件时给 `Refusal(code, ...)`。
5. **指标**：在 `sections.py` 里产出（遵守 ①）。需要注入的读数（JSON/轨迹文件）
   通过 `--ks-*` 开关传入，不要偷偷读环境变量里的隐式路径。
6. **测试**：解析器用固定样例测（`tests/fixtures/`），不要依赖真机；再接一条 `bench/run.sh` 步骤（见 ⑥）。

---

## ③ 已知答案负载（selftest 那一族）

`selftest` 是唯一**不依赖外部真值**的自证方式：负载的搬运字节数/工作量由张量形状**解析可得**，
于是"我们测出来的"能直接与"物理上应该是的"对拍。它只依赖 torch（不需要 nsys / nvcc / NVBit）。

单一真源在 `src/kernoscope/selftest.py`：

- `WORKLOAD_IDS` —— 负载族（当前 `copy` / `read` / `write` / `matmul`）；
- `_analytic_bytes(wid, nbytes, mm)` —— **解析真值**（例如 copy = 2×N×itemsize）；
- `_one_iter(wid, t, a, b, c, mm)` —— 一次迭代怎么跑；
- `_allocate(...)` / `_matmul_shape(...)` —— 张量分配与 matmul 形状反推；
- `_fill_band_and_mdd(res)` —— 误差带 = 各负载偏差的并集；MDD = 重复间离散度 ×2 与
  NVML 1 个百分点分辨率下限取大。这两个数被 `kernoscope diff` 消费（没有它们只能报"变了"）。

### 加一个负载的步骤

1. 写出它的**解析字节公式**并加进 `_analytic_bytes`（公式错了，整个误差带就错了）；
2. 在 `_one_iter` 里实现执行路径；
3. 把 id 加进 `WORKLOAD_IDS`（`to_dict`/`render`/MDD 会自动带上）；
4. 若形状需要反推，补 `_matmul_shape` 之类的辅助；
5. 饱和检查会看 `mem_util_pct`：低于 `MIN_SATURATION_MEM_UTIL_PCT` 的测量会让误差带偏乐观，
   会在 note 里明说 —— **不要为了让数字好看而关掉这个检查**。

---

## ④ 新增一个能力问题

能力的单位是**问题**（用户的词），不是指标名（我们的词）。加一个问题要动两处：

```python
# capabilities.py
Question("<id>", "<用户会怎么问>", <tier>, ("<会产出的指标>", ...), alt="被拒时改问什么")

CHECKS = {
    ...
    "<id>": _q_<id>,
}
```

判定函数是纯函数 `Env -> (state, why, Refusal | None)`：

- **缺件**（这台机器没有这个数据源）→ `STATE_REFUSED` + `Refusal(code, message)`，
  code 取 ⑤ 里已登记的闭集元素；
- **能答但前提缺失**（例如没有本机标定、没有给工作量）→ `STATE_DEGRADED`，
  让读者知道"能用但口径更弱"，而不是把弱口径说成能答；
- **能答** → `STATE_AVAILABLE`。

`tier` 必须与 `TIERS` 的层语义一致。漏登记判定函数不会被静默放过：
`evaluate()` 遇到没有 `CHECKS` 条目的问题会立刻 `raise KeyError`，
`tests/test_capabilities.py::test_every_question_has_a_check` 也会红。

**加完必须重跑**：

```bash
uv run python tools/gen_coverage.py     # 重新生成 COVERAGE.md（与代码同源）
uv run python tools/gen_coverage.py --check
```

`COVERAGE.md` 是生成物：问题 id 集合由 `tests/test_opening_package.py` 与
`kernoscope.capabilities.QUESTIONS` 对拍，手改会在下一次生成时被覆盖。

---

## ⑤ 新增一个拒绝类型

拒绝是**可编程**的：判决在 `code` 里，散文只解释。所以新增一种"我们不给数"的理由时，
`src/kernoscope/refusals.py` 里**三处都要加**（少一处就有测试红）：

1. `RefusalCode`（闭集）—— 命名规则 `<被拒绝的东西>_<为什么>`：
   `_MISSING` 是环境缺件（可安装解决），`_LOW` / `_UNKNOWN` / `_MISMATCH` 是前提不成立
   （可改测量方式解决）；
2. `DEFAULT_REMEDIES[code]` —— `Remedy(action, command=..., cost_s=..., cost_kind=..., then_answers=...)`。
   `cost_kind` 取 `wall_seconds` / `manual` / `unavailable`；`unavailable` 表示"这条路走不通，
   别让用户白试"（例如计数器权限）；
3. `CODE_TITLES[code]` —— 一句话人话（人视图用它把 code 翻译出来；静态、不进决策包）。

自检函数 `missing_remedies()` 与 `missing_titles()` 必须返回空列表
（`tests/test_refusals.py` 守着）。另外想清楚两件事：

- 哪个能力问题会用到它（`capabilities.py` 的 `CHECKS`）；
- `--strict` 是否要因为它 exit 2（请求了产不出来的数据就该失败，而不是悄悄给错数）。

构造拒绝时把 `observed`（实测到的值）与 `threshold`（门槛）**同单位**填上 ——
两者相减才知道差多少，agent 也才能据此规划。复合拒绝用 `combined_refusal([...])`，
不要只留第一条。

---

## ⑥ 新增一个 bench 产物（脱敏 + 文档数字门禁）

规则只有一条：**文档里每个"实测"数字都要能追到一次可复现的运行**，所以数字不写进 README
就算数，而是落成机器可读产物。

1. **采集**：在 `bench/run.sh` 里加一步（写清真值从哪来），产物落
   `bench/artifacts/<YYYYmmdd-HHMMSS>/`；每步的退出码写进 `manifest.json`，**非零不隐藏**。
   能解析算出真值的实验优先（见 `bench/ledger_probe.cu` 的 `ANALYTIC` 行）。
2. **脱敏**（复用机器上采的产物前必须做）：
   ```bash
   python bench/sanitize_artifacts.py bench/artifacts          # 就地替换
   python bench/sanitize_artifacts.py bench/artifacts --check  # 只检查
   ```
   只替换**主机名与绝对路径**；GPU 型号/驱动版本/日期/误差数字一律保留（它们是实验条件，
   不是身份）。脱敏门禁必须**可绿**：已经脱敏过的产物再跑 `--check` 必须是 0 处需要替换。
3. **登记**：在 `bench/artifacts/README.md` 里登记每个产物目录（它是现行的还是历史的），
   `tests/test_doc_artifact_consistency.py` 会检查目录都被登记、修订号与工具版本被钉住、
   `sanitize` 步骤在 `manifest.json` 里。
4. **公开数字**：把引用它的文档与取值函数登记到该测试的 `DOC_NUMBER_RULES`
   （`(文档, 产物文件, 取值函数, 展示模板)`）—— 这是**追加登记**，不改动既有断言；
   取值函数只读产物，所以数字一变、文档没跟着变就会红，并直接告诉你期望串。
5. 引用产物目录时用 `bench/artifacts/*/...` 这类**通配**写法，不要写死某一次的目录名；
   真要写目录名，它必须真的存在。

---

## ⑦ 测试与提交规则

提交前跑：

```bash
uv run pytest -q                       # 全量必须绿
uv run python tools/gen_coverage.py --check   # 生成物与代码同源
uv build                               # sdist + wheel 能生成
```

**提交动作与提交者身份由维护者负责**；贡献者不要自己 `git commit` / `git push`。
提交内容遵守这几条：

- **产物先脱敏**再入库；带平台身份（主机名、绝对路径、实例名）的文件一律不入库。
- **提交信息、代码注释、文档里不得出现**：平台/实例/主机名、`ssh -p … user@…` 这类命令、
  裸 IPv4、绝对路径，以及"维护者本地有哪些文档"的线索（文档名、目录名、数量）。
  这些正是 `tests/test_repo_hygiene.py` 会扫的模式。
- **不要 `git add -f`** 任何被 `.gitignore` 忽略的路径；`git ls-files docs | wc -l` 必须恒为 0。
  维护者侧另有一套同步策略自检（那套约定与脚本不进仓库），它必须报"一致"。
- 不提交构建产物：`dist/`、`*.pyc`、`__pycache__/`、编译中间件；`REVISION` 是每次部署产生的，不入库。
- 新接口要有测试；改了公开数字要同步产物或文档（数字门禁会告诉你差在哪）。
- 提交信息只描述这次改动**本身**（做了什么、为什么），不要描述你所在的机器或维护者的工作目录。
