# bench/ —— 可复现的实测流程

这个目录的存在理由只有一条：**文档里的每个"实测"数字都要能追到一次可复现的运行**，
而不是手抄进 README 的散文里。

```bash
bash bench/run.sh              # 全量（约 10 分钟 GPU 时间）
bash bench/run.sh --quick      # 快速版
OUTDIR=/tmp/bench bash bench/run.sh

# 装了 NVBit 时，run.sh 会自动多跑两步：访存轨迹口径对拍 + 发射槽上界
NVBIT_PATH=~/nvbit_release_x86_64 bash bench/run.sh

# 只想采发射槽上界的产物：
NVBIT_PATH=~/nvbit_release_x86_64 bash bench/issue_probe.sh bench/artifacts/<stamp>/issue
```

产物落在 `bench/artifacts/<YYYYmmdd-HHMMSS>/`：

| 文件 | 内容 |
|---|---|
| `environment.txt` | 日期/内核/nvcc/nsys/驱动/GPU，以及**修订号 + 修订来源 + 是否脏树 + 工具版本** —— 实验条件 |
| `manifest.json` | 修订号/工具版本 + 每个步骤的名字与退出码（**非零不隐藏**，就是给人看的） |
| `doctor.json` / `doctor-path-stripped.json` | 工具多路径发现与 ncu 四态结论（只改 PATH 跑两次） |
| `machine-profile.json` / `.log` | 本机峰值 + 累加口径 + 口径离散度 |
| `probe-*.log` | 微基准的**解析真值**（`ANALYTIC` 行，不需要 GPU） |
| `rep-*.txt` / `rep-*.csv` | 各负载的报告文本与 CSV（含 provenance / resolution 列） |
| `proxy-duration-sweep.csv` | **精度 vs 稳态段长度**：同一个已知答案负载在 1.2/3/8/20 秒各测一遍；`err_pct` 为空 = 该时长下前提不成立、工具**拒绝**给 per-launch（不是精度变差） |
| `rep-lowduty.*` | 覆盖率不足时的裁决行（必须是 `n/a` + 原因，且 `--strict` = 2） |
| `shape-sweep.launches.csv` | 每个 launch 一行的形状扫描原始数据 |
| `occupancy.json` / `occupancy-records.txt` | 达到 occupancy 的区间重叠重建 |
| `memtrace.json` / `memtrace-sample.txt` | 访存轨迹分析（原始轨迹按需重采，不入库） |
| `issue/*.json` | 发射槽利用率上界与瓶颈方向 |
| `rep-l2resident{,-full}.*`、`size-comparison.csv` | 报告体积对比（逐 launch 数组 vs `--ks-json-no-launches`） |
| `rep-issue-*.txt` | 发射槽利用率上界与瓶颈方向（时长取自 CUPTI，不是手填秒数） |

## 被采纳为"文档数字"的产物

README / PROVENANCE / CALIBRATION 里引用的数字都来自 `bench/artifacts/` 里保留的那一次运行；
对照关系写在 README 的「数字从哪来」一节，每个产物目录的登记见 `bench/artifacts/README.md`。
改了工具就要重跑并把新产物一起提交 —— 数字与产物不一致时，以产物为准，
而且 `tests/test_doc_artifact_consistency.py` 会把不一致直接变成失败。

**同步到远端时务必带上修订号。** 租用机器上的仓库常常没有 `.git`，那种情况下
产物只能写 `revision: n/a`，事后没人能判断它由哪一版代码产生。做法是同步后写一个文件：

```bash
git rev-parse HEAD > REVISION      # 远端仓库根目录
```

`run.sh` 按 `KS_REVISION` → `.git` → `REVISION` 的顺序取修订号，并把**来源**与
**是否脏树**一起记进产物；三者都没有时明确告警，绝不冒充"已钉住"。

## 两个必须知道的坑（都真机上踩过）

1. **迭代数要按目标秒数反推，不要手填。**
   DRAM 代理只在"窗口被被测负载主导"时才可用。`--mb 64 --iters 500` 看起来很多，
   实际只有 0.09 秒 GPU 活动，覆盖率会掉到 0%，报告就会（正确地）拒绝给数 ——
   那是实验设计问题，不是工具问题。`bench/ledger_probe.cu --target-s 8` 会先量一次
   每-launch 耗时再定迭代数。

2. **NVBit 轨迹的体积 ∝ 流量 × launch 次数。**
   一条记录 = 一条 warp 访存指令（≤128 B 流量），而插桩**不区分预热**：
   一个 80 MB 工作集 + 默认 50 次预热能写出 **10 GB** 的文本（真事，把磁盘写满过一次）。
   采轨迹时用小负载 + `--warmup 2`，采完立刻分析、只留分析产物。
   实测产物的体积对比（同一次运行）见 `size-comparison.csv`（默认 JSON 138.1 MB vs
   `--ks-json-no-launches -c 50` 25.4 KB，两行都以字节为单位写给机器读）。

## 入库前脱敏

`run.sh` 的最后一步会自动对自己的产物脱敏（路径规则按仓库位置生成），
手工处理时这样跑：

```bash
python bench/sanitize_artifacts.py bench/artifacts          # 就地替换
python bench/sanitize_artifacts.py bench/artifacts --check  # 只检查
```

替换规则（只动这几类模式，谁都能复核）：

| 原文 | 替换为 |
|---|---|
| 实例主机名（从 `environment.txt` 读） | `<host>` |
| 仓库根目录的绝对路径（`--repo-path`） | `<repo>` |
| 仓库根目录的父目录 | `<workdir>` |
| 解释器的绝对路径（`…/bin/python*`） | `<python>` |

规则本身不含任何具体平台字符串 —— 路径规则按**脚本运行的位置**生成。

GPU 型号、驱动版本、CUDA 路径、日期、误差数字**一律保留** —— 它们是实验条件，不是身份。

## 已知限制

- `run.sh` 的 DRAM 代理精度只覆盖三种**合成**负载（字节数可解析算出的拷贝/只读/只写）。
  真实算子的账本依赖使用者自己的解析模型，工具无法替你判断账本对不对。
- 访存轨迹与发射槽上界都需要自行安装 NVBit（专有许可，本项目不分发）。
- `bench/artifacts/` 里保留的是**筛选过的**产物：逐 launch 明细（可达上百 MB）只保留体积对比，
  原始 `.nsys-rep`/`.sqlite` 与完整轨迹不入库。
