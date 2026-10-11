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
| 3 | **维护者** | 一次性：在 PyPI 配 **Trusted Publisher（OIDC）**，**不需要任何 token**。项目还不存在时配的是 pending publisher，四个字段必须与仓库里 `publish.yml` 逐字一致：`owner=QingGo`、`repo=Kernoscope`、`workflow=publish.yml`、`environment=pypi`（`https://pypi.org/manage/account/publishing/`），并在 GitHub 建同名 environment `pypi`（可挂保护规则：仅允许 `v*` tag、需要 reviewer） |
| 4 | **维护者** | 打 tag 即发布（**密钥为零**，由 CI 用 OIDC 换短时 token）：`git tag -a v0.2.0 -m "0.2.0 alpha"` → `git push origin v0.2.0`。⚠️ tag 指向的提交里必须**已经**有 `publish.yml`。CI 会先跑门禁与元数据断言，再 `uv publish --trusted-publishing always` |
| 5 | 任何人 | 发布后验证**发布物**而不是工作树：`bash bench/release-dry-run.sh --from-pypi`（从 PyPI 隔离安装 + 首用三连 + 预算），或手工 `uv tool install kernoscope`（新环境）→ `kernoscope doctor; echo $?` |
| 6 | 维护者 | GitHub Release：贴 §0 的 preflight 结果 + CHANGELOG 的 Unreleased 段 + 「接口会变」声明 |

### 1.1 为什么不用 API token（以及什么时候仍然用）

**日常发布走 Trusted Publisher**：没有长期密钥可泄露，每次运行由 GitHub 换一个 ~15 分钟的
短时 token，绑定 `owner/repo/workflow/environment`；而且 PyPI 会自动为这类上传附
**attestation（PEP 740）**，用户可验证"这个 wheel 确实由这个仓库的这次构建产出"。

顺带一个**首次发布的现实理由**：项目还不存在时，PyPI **建不了** project-scoped token，
只能先用**账号级** token（权限大得多：能往你任意项目发包、能建新项目）——而 pending publisher
恰恰在项目不存在时就能配好。所以"零密钥"这条路在第一次发布时反而更干净。

**仍然需要 token 的场合**（应急，不是日常）：CI 挂了但必须立刻发版、或要往 TestPyPI 演练。
那时：① 首次用账号级 token，发完**立刻**换成 project-scoped（Scope = Project: kernoscope）并删掉
账号级的；② 只放在环境变量 `UV_PUBLISH_TOKEN` 或 CI secret 里，绝不进仓库；③ token 只显示一次。

```bash
# 应急路径（日常不要用）
uv build && UV_PUBLISH_TOKEN=pypi-XXXXXXXX uv publish
```

## 2 版本与兼容策略

- **版本号跟 schema 走**：`packet.schema_version` / `schema_revision` 变了就要动次版本号。
- **接口变化必须进 `CHANGELOG.md`**（命令行、决策包字段、输出字段、退出码语义）。
  例子：`doctor` 的退出码语义与 `--json` 的 `diagnosed` 字段就记在 Unreleased 里。
- **alpha 期间允许破坏性变更**，但必须：① 写清"按旧行为写的脚本要改哪一行"；
  ② 在 PR/Release 里给出迁移一句；③ 不留"没写进 CHANGELOG 的静默变更"。

## 2.5 推之前：跑一次「CI 对等检查」

```bash
bash bench/ci-parity.sh
```

它把 **HEAD 的干净快照**导出到临时目录、`uv sync`、再跑与 CI **完全相同**的那条门禁命令。
为什么需要：本地跑的是"我的工作树 + 我的环境"，CI 跑的是"刚 clone 的某次提交" —— 已经踩过两次
（一条测试依赖仓库里有没有 `.venv`；另一条把 `PATH` 清空后脚本里的 `dirname` 失效 ⇒ **本机假绿**）。
想看某个历史提交在 CI 上会怎样：`bash bench/ci-parity.sh <commit>`。

## 3 发布前检查单（每条都能在本机跑）

```bash
uv run python tools/dev_index.py --gates     # 9 条门禁（含 sanitize 与 sdist/wheel 构建）
bash bench/preflight-first-run.sh            # 陌生首跑路径（G1）
uv run kernoscope doctor                 # 期望 rc=0（诊断完成；"诊断是否完成"是退出码的含义）
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

## 3.5 发布预演与发布后验证（同一条命令，两种来源）

```bash
bash bench/release-dry-run.sh                 # 发布前：现场构建 + 元数据 + **从本地 wheel 隔离安装** + 首用三连
bash bench/release-dry-run.sh --from-pypi     # 发布后：从 **PyPI** 装 kernoscope==<版本> 再跑同一组断言
bash bench/release-dry-run.sh --metadata-only # 最快：只构建 + 检查元数据（名字/版本/许可/Alpha/Python 下限/入口点/不含测试）
```

为什么要有它：README §2 的三条安装路径里，`uv tool install kernoscope` **只有在包真的发上去之后
才可用**（第七轮 X-21：发布前实测 404，而文档已经宣布同日发布）。所以"能不能装"不能只写在散文里
—— 它是**能跑的命令**，发布前后各跑一次：

* 发布前 `--metadata-only` / 无参数：证明"这一刻的产物装出来能跑、首屏在预算内"；
* 发布后 `--from-pypi`：证明"用户按 README 复制粘贴的那条命令真的能用"。

两种模式共用**同一段断言**（隔离安装 + `capabilities --brief` ≤1.5 KB 且带 `next[]` +
`doctor --no-probe` rc=0 + 一屏 ≤8192 B / ≤60 行），所以不存在"发布前一套清单、发布后另一套"。

发布后如果它报"隔离安装失败"，最可能的原因是 **PyPI 上还没有这个版本**或 tag 与 `pyproject`
的 version 不一致；脚本会直接把这两条提示打出来。

## 4 发布之后要看什么

| 信号 | 为什么 | 在哪看 |
|---|---|---|
| 首用时长（从零到第一条结论） | 采纳率的直接度量（C） | `bench/preflight-first-run.sh` 的输出 + 外部反馈 |
| 安装失败/报错路径 | "装不上"是第一死因 | GitHub issues（issue 模板已要求贴 `capabilities --json` 与 `doctor --json`） |
| 误用率（把 `refused` 的判词当依据） | 我们唯一不可让渡的资产的**反向**指标 | 外部 bug 报告与 `eval/` 评测台 |
| 外部引用/复现 | 开放门槛的那条"≥1 个外部引用" | 引用、博客、issue、PR |
