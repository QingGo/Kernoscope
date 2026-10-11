# RELEASING —— 公开与发布

> 状态：**公开 alpha**。接口会变（见 [CHANGELOG.md](CHANGELOG.md)）；不承诺稳定 API，版本号跟
> schema 走。这份文件写清"谁来做什么"——其中两步**只有维护者能执行**（需要 GitHub 与 PyPI 的账号），
> 其余步骤任何人都能跑、也应当跑。

## 0 公开前的三件必做（已做完，且都是可重跑的）

| # | 事 | 现状 | 怎么复核 |
|---|---|---|---|
| 1 | README 的安装命令**真的能跑** | `uv tool install kernoscope` 在 PyPI 有包之后为真；同时给了"跟 main"的等价命令 `uv tool install git+https://github.com/QingGo/Kernoscope` 与源码安装 | 照 README §2 抄一遍 |
| 2 | 第一分钟的第一步**不再是"失败且没有解释"** | `doctor` 的退出码表示**诊断是否完成**（0=完成，结论可以是"这台机器没有 GPU"）；1 只留给"诊断本身没完成"，且必须往 stderr 打一行。README 的第一条命令换成任何机器都 rc=0 的 `capabilities --brief` | `kernoscope doctor; echo $?` |
| 3 | 「陌生人首跑」**可重跑、能变红** | `bash bench/preflight-first-run.sh`：现场构建 wheel → 装进隔离目录（等价 `uv tool install`）→ 逐条断言 rc=0、产出非空、一屏 ≤8 KB/≤60 行 | 直接跑（约 2 秒） |

```bash
bash bench/preflight-first-run.sh          # 期望：PREFLIGHT OK 6/6
```

## 1 谁做什么

| 步骤 | 谁 | 命令 / 动作 |
|---|---|---|
| 1 | **维护者** | 仓库转公开：GitHub → Settings → Danger Zone → Change visibility → Public（或 `gh repo edit QingGo/Kernoscope --visibility public --accept-visibility-change-consequences`） |
| 2 | 任何人 | 公开后立刻复核：`git clone` 到一个干净目录、按 README §2 走一遍、确认 `bench/`、`skills/`、`NON-GOALS.md`、`PROVENANCE.md`、`COVERAGE.md` 都在 |
| 3 | **维护者** | 在 PyPI 建 API token（**只授 `kernoscope` 这一个项目**，不要账号级 token），存到 `~/.pypirc` 或 `UV_PUBLISH_TOKEN` |
| 4 | **维护者** | 打 tag 并发布：`git tag -a v0.2.0 -m "0.2.0 alpha"` → `git push origin v0.2.0` → `uv build && uv publish`（或 `twine upload dist/*`） |
| 5 | 任何人 | 发布后验证**发布物**而不是工作树：`uv tool install kernoscope`（新环境）→ `kernoscope doctor; echo $?` → `bash bench/preflight-first-run.sh` |
| 6 | 维护者 | GitHub Release：贴 §0 的 preflight 结果 + CHANGELOG 的 Unreleased 段 + 「接口会变」声明 |

## 2 版本与兼容策略

- **版本号跟 schema 走**：`packet.schema_version` / `schema_revision` 变了就要动次版本号。
- **接口变化必须进 `CHANGELOG.md`**（命令行、决策包字段、输出字段、退出码语义）。
  例子：`doctor` 的退出码语义与 `--json` 的 `diagnosed` 字段就记在 Unreleased 里。
- **alpha 期间允许破坏性变更**，但必须：① 写清"按旧行为写的脚本要改哪一行"；
  ② 在 PR/Release 里给出迁移一句；③ 不留"没写进 CHANGELOG 的静默变更"。

## 3 发布前检查单（每条都能在本机跑）

```bash
uv run python tools/dev_index.py --gates     # 9 条门禁（含 sanitize 与 sdist/wheel 构建）
bash bench/preflight-first-run.sh            # 陌生首跑路径（G1）
uv run kernoscope doctor; echo "rc=$?"       # 期望 rc=0（诊断完成）
uv run kernoscope capabilities --brief       # 期望 ≤1.2 KB 且带 next[]
git status --porcelain                       # 期望为空（发布物必须来自被跟踪的提交）
```

- **元数据**：`pyproject.toml` 的 `urls`（Homepage/Issues/Documentation/Repository）必须指向
  **真实存在且公开**的地址 —— PyPI 页面上的链接就是它们。
- **`docs/` 不入库**（`.gitignore` 整目录忽略）：README 已披露"其中的证据不构成可复核证据"。
  **不要**用 `git add -f` 把它塞进去 —— 那会与 `.gitignore` 的断言门禁打架，而且那些证据本来就是
  "某一次本地运行"，不是可复核产物（可复核的是 `bench/artifacts/**`，它是入库的）。
- **不要**在仓库还是私有时发 PyPI：Homepage 会是死链、wheel 无法核对来源、也没有 issue 入口。
- **NVBit 不可再分发**：仓库里只有我们自己的探针源码，构建产物（`*.so`）在 `.gitignore` 里；
  发布物不得包含它。`NOTICE` 里有"与 NVIDIA 无关联"与第三方声明。

## 4 发布之后要看什么

| 信号 | 为什么 | 在哪看 |
|---|---|---|
| 首用时长（从零到第一条结论） | 采纳率的直接度量（C） | `bench/preflight-first-run.sh` 的输出 + 外部反馈 |
| 安装失败/报错路径 | "装不上"是第一死因 | GitHub issues（issue 模板已要求贴 `capabilities --json` 与 `doctor --json`） |
| 误用率（把 `refused` 的判词当依据） | 我们唯一不可让渡的资产的**反向**指标 | 外部 bug 报告与 `eval/` 评测台 |
| 外部引用/复现 | 开放门槛的那条"≥1 个外部引用" | 引用、博客、issue、PR |
