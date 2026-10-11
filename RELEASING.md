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

### 1.0 0.2.0 的发布事实（第一次真的发出去了）

| 事实 | 值 |
|---|---|
| 版本 / 文件 | `kernoscope 0.2.0`：`kernoscope-0.2.0-py3-none-any.whl`（518 784 B）+ `.tar.gz`（1 800 015 B） |
| 触发 | tag `v0.2.0` → GitHub Actions `publish.yml` → OIDC（**没有任何 token**） |
| 那次运行 | `38152627009` = success；之前的 `38152306470` 失败在 OIDC 一步（见下表） |
| 发布后验证 | `bash bench/release-dry-run.sh --from-pypi` → `RELEASE OK`（隔离装出 `kernoscope`+`ksc`、首用三连 rc=0、一屏 4854 B/42 行） |
| attestation | **无**（0.2.0 用 `uv publish` 发的；0.2.1 起改官方动作带 attestation） |

**第一次失败的原因（值得记下来）**：pending publisher 是**在仓库转公开之前**填的，没有生效；
重新登记后一次通过。另外四个字段必须逐字一致 —— 最常见的两个坑是
**Workflow name 填成了 `publish`（要填文件名 `publish.yml`）** 与 **Environment name 留空**
（工作流写了 `environment: pypi`，PyPI 那边空着就会在 OIDC 交换时被拒）。

失败后的重试**不需要新提交**（没能上传时也没有"版本已存在"的问题）：删 tag 重推即可 ——

```bash
git push --delete origin v0.2.0 && git push origin v0.2.0
```

### 1.1 为什么不用 API token（以及什么时候仍然用）

**日常发布走 Trusted Publisher**：没有长期密钥可泄露，每次运行由 GitHub 换一个 ~15 分钟的
短时 token，绑定 `owner/repo/workflow/environment`；上传用 PyPA 官方动作
`pypa/gh-action-pypi-publish`，它会为产物生成 **attestation（PEP 740）**，用户可验证
"这个 wheel 确实由这个仓库的这次构建产出"。

⚠️ **0.2.0 是这条规则的例外，别被它误导**：0.2.0 是用 `uv publish` 发出去的，而那个工具当时
（0.8.17）根本没有 attestation 能力 —— PyPI Integrity API 对它返回
`404 No provenance available for kernoscope-0.2.0-py3-none-any.whl`（我实测过）。
从 **0.2.1** 起走官方动作，**发布后请用 `--require-attestation` 验一次**：

```bash
bash bench/release-dry-run.sh --from-pypi --require-attestation
curl -s -o /dev/null -w '%{http_code}\n' \
  https://pypi.org/integrity/kernoscope/0.2.1/kernoscope-0.2.1-py3-none-any.whl/provenance
# 期望 200；404 = attestation 没上去（先查是不是又用 uv publish 发的）
```

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

- **默认只升末位（patch）**：小改动 —— 修 bug、补门禁、加文档、换发布机制、加一个不改语义的
  开关 —— 一律 `0.2.0 → 0.2.1 → 0.2.2`。**不要为了"看起来有进展"跳号**。
- **只有下面两类才动中位（minor）**：① `packet.schema_version` / `schema_revision` 变了；
  ② 有**破坏性**接口变更（删/改命令行开关、改退出码语义、改决策包字段含义）。
  这两类都必须按下面 §2 的规矩写清"按旧行为写的脚本要改哪一行"。
- 建议一条命令同时改版本与生成 tag 正文（版本号只有 `pyproject.toml` 一个真源）：

  ```bash
  python3 bench/release_notes.py 0.2.1 --tag > /tmp/notes.md   # 正文来自 CHANGELOG 那一节
  git tag -a v0.2.1 -F /tmp/notes.md                            # tag message = Release 正文（同源）
  git push origin v0.2.1                                        # CI 发布 + 自动建 Release
  ```
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

## 3.6 GitHub Release：谁建、怎么建（**不需要任何 PAT**）

| 情况 | 谁 | 怎么做 |
|---|---|---|
| **0.2.1 及以后**（tag 已按 §2 的流程推上去） | CI 自动 | `publish.yml` 的 `release` job：正文取 `bench/release_notes.py`（= CHANGELOG 那一节，**唯一真源**）+ `bench/pypi_digests.py` 的 sha256；用 workflow 自带的 `GITHUB_TOKEN` 建 Release，`--verify-tag` 保证**不移动 tag**。不留资产：分发渠道是 PyPI（那里才有 attestation），源码归档 GitHub 自动附 |
| **v0.2.0**（已经发出去的版本，tag **不能重打**） | 任何人，3 次点击 | Actions → **release** → Run workflow → 填 `v0.2.0`。它读 **tag 指向的提交**里的 `pyproject` 校验版本、用当前 CHANGELOG 生成正文，然后 create-or-edit（幂等） |
| 临时手工 | 维护者 | GitHub → Releases → Draft a new release → 选**已存在的** tag → 粘贴 `python3 bench/release_notes.py <版本> --tag` 的输出（例如 `0.2.1`） |

**为什么 v0.2.0 不能重打 tag**：tag 对象指向 `e0bcac0` —— 就是构建出 PyPI 上那份 wheel 的提交。
重打会把它指向别处，抹掉"这个 tag = 这份产物"的对应关系，而且会**再触发一次发布**（PyPI 已有同名
文件）。所以已发布的版本只能**针对现有 tag** 建 Release（`--verify-tag` / UI 里选已有 tag 都是这样）。

**为什么 tag ≠ Release**：推 tag 只出现在 Tags 列表；`/releases` 是另一类对象，不会因为 push 一个
tag 就出现条目（"只订阅 Releases"的人因此收不到通知）。tag message 可以装发布说明（`git show` 能读，
§2 的命令就是这么打的），但要让 Release 页也有，就得建 Release 对象 —— 而这件事**不需要你的 token**。

## 4 发布之后要看什么

| 信号 | 为什么 | 在哪看 |
|---|---|---|
| 首用时长（从零到第一条结论） | 采纳率的直接度量（C） | `bench/preflight-first-run.sh` 的输出 + 外部反馈 |
| 安装失败/报错路径 | "装不上"是第一死因 | GitHub issues（issue 模板已要求贴 `capabilities --json` 与 `doctor --json`） |
| 误用率（把 `refused` 的判词当依据） | 我们唯一不可让渡的资产的**反向**指标 | 外部 bug 报告与 `eval/` 评测台 |
| 外部引用/复现 | 开放门槛的那条"≥1 个外部引用" | 引用、博客、issue、PR |
