"""R-25/R-36/R-40：文档里公开的数字必须能追到入库产物，产物必须钉住产生它的修订。

需求方三轮验收里最顽固的一条是「文档数字 ↔ 产物」没有机制：
  * README 写「误差带 ±4%」，产物里是 −3.2%…+12.7%（早就漂了）
  * `bench/README.md` 写「147 MB / 15.6 KB」，`size-comparison.csv` 是 134 MB / 15.5 KB
  * `CALIBRATION.md` 的示例 JSON 是**另一次运行**的值（pcie gen 4 vs 2、2211.7 vs 2193.1）
  * 入库产物的 `environment.txt` 写 `git: n/a`，报告头写工具 `0.1.0` 而 HEAD 是 `0.2.0`

所以这里做两件事，都是**从产物反推期望值**（而不是把数字抄进测试里）：
  1. 产物必须钉住版本：修订号 + 修订来源 + 是否脏树 + 工具版本 + `sanitize` 步骤；
  2. 文档里公开的每个「实测数字」都必须能在产物里找到一模一样的写法。

新增公开数字时的登记方式：在 `DOC_NUMBER_RULES` 里加一条
`(文档, 产物文件, 取值函数, 展示模板)`。取值函数只读产物；文档里必须出现模板渲染出来的串。
这样数字一变、文档没跟着变，测试就会红，并且直接告诉你期望串是什么。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from kernoscope import __version__

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "bench" / "artifacts"

_M = sys.modules[__name__]


def _artifact_dirs() -> list[Path]:
    if not ARTIFACTS.exists():
        return []
    return sorted([p for p in ARTIFACTS.iterdir() if p.is_dir() and p.name[0].isdigit()],
                  key=lambda p: p.name)


def _artifact_dirs_with_profile() -> list[Path]:
    """带 `machine-profile.json` 的产物目录（按名字排序）。"""
    return [d for d in _artifact_dirs() if (d / "machine-profile.json").exists()]


def _newest() -> Path:
    """「最新的产物」= 最后一个**含 `machine-profile.json`** 的目录。

    唯一真源（W-10）：与 `tools/gen_calibration_excerpt.py:newest_profile()` 同一套判据。
    旧实现这里是"名字排序最后一个"，于是两套定义可以各指一份产物而全绿 ——
    定这份判据的人是 CALIBRATION 节选（它只能引用有 profile 的那一份）。
    """
    dirs = _artifact_dirs_with_profile()
    if dirs:
        return dirs[-1]
    if not _artifact_dirs():
        pytest.skip("bench/artifacts 下还没有产物目录")
    pytest.fail("bench/artifacts 下没有一个含 machine-profile.json 的产物目录 —— "
                "标定证据必须有 profile（否则 CALIBRATION 的节选没有可引用的产物）")


def _newest_definition_conflict() -> str | None:
    """两套「最新产物」定义是否互相矛盾（None = 一致）。W-10 的判据。

    定义 A：按目录名排序的最后一个（产物名就是时间戳，读者会这么读）。
    定义 B：最后一个**含 `machine-profile.json`** 的（`newest_profile()`，CALIBRATION
    节选由它渲染）。新增一个不含 profile 的更新目录会让 A、B 指向两份不同的产物 ——
    那时 `bench/artifacts/README.md` 的「现行」标签与 CALIBRATION 的引用互相打脸，
    而旧门禁全绿（V-20 只修了标签那一半）。
    """
    dirs = _artifact_dirs()
    if not dirs:
        return None
    by_name = dirs[-1]
    by_profile = next((d for d in reversed(dirs)
                       if (d / "machine-profile.json").exists()), None)
    if by_profile is None:
        return (f"最新产物 {by_name.name} 没有 machine-profile.json —— 没有 profile 的目录"
                f"不能当标定证据（它进不了 CALIBRATION 节选）。")
    if by_name.name != by_profile.name:
        return (f"「最新产物」有两套定义且互相矛盾：按目录名排序是 {by_name.name}，"
                f"按「最后一个含 machine-profile.json」是 {by_profile.name} —— "
                f"`bench/artifacts/README.md` 的「现行」标签与 CALIBRATION 的节选会互相打脸。"
                f"请让 {by_name.name} 也带 machine-profile.json（或把它移出产物登记）。")
    return None


def test_newest_artifact_has_one_definition(monkeypatch, tmp_path):
    """W-10：`_newest()` 与 `gen_calibration_excerpt.newest_profile()` 不许各说各话。"""
    conflict = _newest_definition_conflict()
    assert conflict is None, conflict
    # 两套实现在本仓库里必须指向同一份产物（现状全都有 profile，所以应当相等）
    mod = _load_calib_excerpt()
    assert _newest() == mod.newest_profile()[0], (
        "本测试模块的 `_newest()` 与生成器的 `newest_profile()` 指向了不同的产物")

    # 负样本：新增一个**不含 profile** 的、名字更新的产物目录 ⇒ 必须报冲突（旧实现全绿）
    older = tmp_path / "20261011-140642"
    older.mkdir()
    (older / "machine-profile.json").write_text("{}", encoding="utf-8")
    stray = tmp_path / "20261012-000000"
    stray.mkdir()                                    # 比 older 新，但没有 profile
    monkeypatch.setattr(_M, "_artifact_dirs", lambda: [older, stray])
    bad = _newest_definition_conflict()
    assert bad, "新增一个不含 profile 的更新产物目录后，两套「最新」定义没有报冲突"
    assert stray.name in bad and older.name in bad, bad
    # `_newest()` 本身也必须按 profile 判据走（而不是名字排序的那个 stray）
    assert _newest() == older


# --------------------------------------------------------------------------
# 1) 产物必须钉住"自己是哪个修订产生的"（R-40）
# --------------------------------------------------------------------------
def test_newest_artifact_pins_revision_and_tool_version():
    art = _newest()
    man = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    env = (art / "environment.txt").read_text(encoding="utf-8", errors="replace")

    assert man.get("revision") and man["revision"] != "n/a", \
        f"{art.name}: manifest 必须记修订号，不能是 n/a（R-40）"
    assert man.get("revision_source") not in (None, "", "none"), \
        f"{art.name}: 必须记下修订号是从哪来的（env/git/file）"
    # R-71/P4：最新产物必须钉住"有没有未提交改动"。旧门禁允许 `unknown`，于是两份
    # `unknown` 的产物永远不可能让它变红 —— 那不是"允许"，是"没人守"。
    assert man.get("revision_dirty") in ("yes", "no"), (
        f"最新产物的 revision_dirty={man.get('revision_dirty')!r}：要么钉住（yes/no），"
        f"要么别把它当现行证据（在登记表里标「不可钉住」）")
    assert man.get("kernoscope", "").startswith("kernoscope ") or man.get("kernoscope") not in ("", None)
    assert f"revision: {man['revision']}" in env
    assert f"revision_source: {man['revision_source']}" in env
    assert "kernoscope:" in env
    # 旧版 environment.txt 用的是 `git:` 这一行；改名之后不许再出现无来源的 n/a
    assert "git: n/a" not in env

    # 工具版本必须与仓库当前的 __version__ 一致：产物比代码旧这件事本身要可见
    header = None
    for txt in sorted(art.glob("rep-*.txt")):
        m = re.search(r"==PROF== Kernoscope (\S+)", txt.read_text(encoding="utf-8",
                                                                  errors="replace"))
        if m:
            header = m.group(1)
            break
    assert header is not None, f"{art.name}: 找不到带版本号的报告头"
    assert header == __version__, (
        f"{art.name} 的报告头写 {header}，而仓库当前是 {__version__} —— "
        f"产物比代码旧，要么重跑 bench，要么别用它当证据")


def test_newest_artifact_records_the_sanitize_step():
    """入库产物必须先跑过脱敏；manifest 里没有这一步就说明它可能是带平台身份的。"""
    art = _newest()
    man = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    steps = [s.get("step") for s in man.get("steps", [])]
    assert "sanitize" in steps, f"{art.name}: manifest 缺 sanitize 步骤（R-40）"


def test_every_artifact_dir_is_listed_and_explained():
    """每个产物目录都要被登记，且"现行/历史"标签必须与**推出来的最新产物**一致。

    第五轮 V-20：旧实现只断言"目录名在文件里出现过" —— 于是把两份产物的「现行/历史」
    标签对调，20 条测试照样全绿（而 `_newest()`、`gen_calibration_excerpt.newest_profile()`、
    CALIBRATION 节选三处都以真正最新的那一份为准）。登记表是文档对数据的**承诺**，
    承诺必须被机器校验。
    """
    readme = ARTIFACTS / "README.md"
    assert readme.exists(), "bench/artifacts/README.md 必须存在（登记每个产物目录）"
    text = readme.read_text(encoding="utf-8")
    dirs = _artifact_dirs()
    for d in dirs:
        assert d.name in text, f"{d.name} 没被 bench/artifacts/README.md 登记"

    # 每行（一个产物目录一行）必须带明确标签；**有且仅有**最新那一份是"现行"
    current = _newest().name
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|") or "`" not in line:
            continue
        for d in dirs:
            if f"`{d.name}`" in line:
                rows[d.name] = line
    missing = [d.name for d in dirs if d.name not in rows]
    assert not missing, f"这些产物目录没有自己的登记行：{missing}"
    marked_current = [name for name, row in rows.items() if "**现行**" in row]
    assert marked_current == [current], (
        f"「现行」标签必须且只能落在最新产物 {current} 上，实际：{marked_current}")
    for name, row in rows.items():
        if name != current:
            assert "**历史**" in row, f"{name} 不是最新，却既没标「历史」也没标「现行」"


# --------------------------------------------------------------------------
# 2) 文档公开的数字必须能在产物里找到（R-36）
# --------------------------------------------------------------------------
def _size_comparison(art: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (art / "size-comparison.csv").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) == 2 and parts[1].strip().isdigit():
            out[parts[0].strip()] = int(parts[1].strip())
    return out


def _ledger_error(art: Path, kernel: str) -> float | None:
    """从 rep-<kernel>.txt 里取账本误差（%）。

    两种写法都要认：摘要句里的「误差 +9.0%」，以及指标行 `est__dram_vs_ledger_err_pct -66.7 %`
    （轨迹口径那次走的是指标行 —— 只认摘要会让 README 里的 −66.7% 变成不可校验的数字）。
    """
    p = art / f"rep-{kernel}.txt"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"误差\s*([-+]?\d+\.\d+)%", text) or \
        re.search(r"est__dram_vs_ledger_err_pct\s+([-+]?\d+\.\d+)\s*%", text)
    return float(m.group(1)) if m else None


def _calibration(art: Path) -> dict:
    return json.loads((art / "machine-profile.json").read_text(encoding="utf-8"))


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def _fmt_signed_unicode(v: float) -> str:
    return f"{v:+.1f}%".replace("-", "\u2212")


def _doc_rules(art: Path):
    """(文档, 描述, 可接受的期望串) —— 期望串全部由产物算出。

    一个数字可以有多种写法（减号有 ASCII `-` 与 Unicode `−` 两种），所以允许给多个
    可接受串；只要出现其中任意一个就算对得上。
    """
    rules: list[tuple[str, str, list[str]]] = []

    sizes = _size_comparison(art)
    if sizes:
        default_bytes = max(sizes.values())
        slim_bytes = min(sizes.values())
        rules.append(("bench/README.md", "默认 JSON 体积",
                      [f"{default_bytes / 1e6:.0f} MB", f"{default_bytes / 1e6:.1f} MB"]))
        rules.append(("bench/README.md", "限流后体积",
                      [f"{slim_bytes / 1e3:.1f} KB", f"{slim_bytes / 1e3:.0f} KB"]))

    for kern in ("copy", "read", "write"):
        err = _ledger_error(art, kern)
        if err is not None:
            rules.append(("README.md", f"{kern} 的账本误差",
                          [_fmt_pct(err), _fmt_signed_unicode(err)]))

    # 标定：峰值、带宽、口径离散度、以及"头条来自哪个变体"
    cal = _calibration(art)
    dram = cal.get("measured_dram", {})
    for key in ("copy_gbs", "read_gbs", "write_gbs"):
        if dram.get(key):
            rules.append(("README.md", f"标定 {key}", [str(dram[key])]))
    # CALIBRATION.md 的**产物节选现在是生成物**（tools/gen_calibration_excerpt.py），
    # 由 test_calibration_excerpt_is_generated_from_the_current_artifact 与门禁
    # `calib` 守 —— 所以这里不再逐条登记它的字段。手抄的 needle 只能盖住"你想到的那几个"，
    # 而混合体能同时通过全部 needle（D13 就是这么活下来的）。
    # R-62：`l2_read_peak_gbs` 此前没有任何 needle —— 把它换成更旧产物的值，门禁全绿。
    l2_peak = (cal.get("l2_info") or {}).get("l2_read_peak_gbs") or cal.get("l2_read_peak_gbs")
    if l2_peak:
        rules.append(("CALIBRATION.md", "L2 读峰值", [str(l2_peak)]))
    mm = cal.get("measured_matmul_tflops", {})
    if mm:
        bf16 = mm.get("torch.bfloat16", {})
        if bf16.get("tflops"):
            rules.append(("README.md", "bf16 标定读数（README 的能力表）", [str(bf16["tflops"])]))
        if bf16.get("caliber_spread_pct"):
            rules.append(("README.md", "bf16 口径离散度", [str(bf16["caliber_spread_pct"])]))

    # occupancy 探针的两个读数（无 smem / 32 KB smem）—— 两个都必须存在：
    # 探针本来就同时写两份记录，README 也引用了两个读数，所以产物里必须都有，
    # 否则"文档引用的数字能追到产物"这条就在这一行上破了。
    for fn in ("occupancy.json", "occupancy-smem32k.json"):
        p_occ = art / fn
        if not p_occ.exists():
            raise AssertionError(
                f"{art.name} 缺 {fn}：occupancy 的两个读数都要在产物里"
                f"（run.sh 第 7 步会分别分析 occupancy-records.txt 与 .smem32k）")
        p_occ = art / fn
        occ = json.loads(p_occ.read_text(encoding="utf-8"))
        if occ.get("occupancy_pct") is not None:
            rules.append(("README.md", f"{fn} 的达到 occupancy",
                          [f"{occ['occupancy_pct']:.1f}", str(round(occ["occupancy_pct"]))]))
        if occ.get("avg_concurrent_blocks") is not None:
            rules.append(("README.md", f"{fn} 的平均并发块数",
                          [f"{occ['avg_concurrent_blocks']:.2f}"]))

    # 发射槽上界（两个方向性判词）
    for kern in ("rep-issue-dram-copy.txt", "rep-issue-l2-resident.txt"):
        p_iss = art / kern
        if not p_iss.exists():
            continue
        text = p_iss.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"est__issue_slot_utilization_pct\s+(\S+)", text)
        if m and m.group(1) not in ("n/a", "-"):
            rules.append(("README.md", f"{kern} 的发射槽利用率",
                          [f"{float(m.group(1)):.1f}", m.group(1)]))

    # 轨迹口径对拍的误差与 L2 命中率
    err_tr = _ledger_error(art, "tracebasis")
    if err_tr is not None:
        rules.append(("README.md", "轨迹口径对拍误差",
                      [_fmt_pct(err_tr), _fmt_signed_unicode(err_tr)]))
    p_mem = art / "memtrace.json"
    if p_mem.exists():
        mt = json.loads(p_mem.read_text(encoding="utf-8"))
        hit = mt.get("l2_hit_rate_l2only_pct")
        if hit is not None:
            rules.append(("README.md", "轨迹模型给出的 L2 命中率",
                          [f"{hit:.1f}", f"{round(hit)}"]))
    return rules


def test_documented_union_band_matches_all_registered_artifacts():
    """README 写的"并集 −4.3% ~ +12.7%"必须等于所有已登记产物里该范围的并集。

    单个数字由 `_doc_rules` 逐条对拍；这一条守的是"跨运行的并集"这个说法 ——
    它最容易在新增一次运行之后悄悄过期（并集只会变宽，不会自动写进文档）。
    """
    lo: float | None = None
    hi: float | None = None
    for art in _artifact_dirs():
        for kern in ("copy", "read", "write"):
            err = _ledger_error(art, kern)
            if err is None:
                continue
            lo = err if lo is None else min(lo, err)
            hi = err if hi is None else max(hi, err)
    if lo is None or hi is None:
        pytest.skip("产物里还没有可比对的历史运行")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    needle = f"{lo:+.1f}% ~ {hi:+.1f}%"
    assert needle in readme or needle.replace("-", "\u2212") in readme, (
        f"README 要写出跨全部 {len(_artifact_dirs())} 次登记的并集 {needle}")


def test_documented_numbers_exist_in_the_artifact():
    art = _newest()
    rules = _doc_rules(art)
    assert rules, f"{art.name} 里没有可对拍的数字 —— 产物可能不完整"
    missing: list[str] = []
    for rel, what, needles in rules:
        doc = ROOT / rel
        if not doc.exists():
            missing.append(f"{rel} 不存在")
            continue
        text = doc.read_text(encoding="utf-8")
        if not any(n in text for n in needles):
            missing.append(f"{rel} 缺「{what}」= {needles[0]!r}（产物 {art.name} 里的值）")
    assert not missing, "文档数字与产物不一致：\n  - " + "\n  - ".join(missing)


def test_docs_reference_only_existing_artifact_dirs():
    """文档里提到的每个 bench/artifacts/<目录> 都必须真的存在（旧版引用过 182/188 这类行号）。"""
    pattern = re.compile(r"bench/artifacts/([0-9][0-9A-Za-z\-]*)")
    docs = [ROOT / "README.md", ROOT / "CALIBRATION.md", ROOT / "PROVENANCE.md",
            ROOT / "bench" / "README.md", ARTIFACTS / "README.md"]
    bad: list[str] = []
    for doc in docs:
        if not doc.exists():
            continue
        for name in set(pattern.findall(doc.read_text(encoding="utf-8"))):
            if not (ARTIFACTS / name).is_dir():
                bad.append(f"{doc.relative_to(ROOT)} 引用了不存在的产物目录 {name}")
    assert not bad, "\n  - ".join(bad)


def _sweep_rows(art: Path) -> list[dict[str, str]]:
    p = art / "proxy-duration-sweep.csv"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","))) for ln in lines[1:] if ln.strip()]


def test_duration_sweep_is_documented_with_its_boundary():
    """R-35：文档必须写出「时长只决定前提是否成立」这个边界，而不是只给一个误差数。"""
    art = _newest()
    rows = _sweep_rows(art)
    if len(rows) < 3:
        pytest.skip(f"{art.name} 里没有时长扫描（proxy-duration-sweep.csv）")
    refused = [r for r in rows if not r["err_pct"]]
    ok = [r for r in rows if r["err_pct"]]
    assert refused, "扫描里应当至少有一个时长因前提不成立而被拒绝"
    assert ok, "扫描里应当至少有一个时长给出了误差"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # ① 被拒绝的那一行的覆盖率要点名（读者才知道"拒绝"发生在什么条件下）
    cov = refused[0]["coverage_pct"]
    assert cov in readme, f"README 要写出被拒绝那个时长的覆盖率 {cov}%"
    # ② 给出误差的那几个时长的两端也要在
    errs = [float(r["err_pct"]) for r in ok]
    lo, hi = min(errs), max(errs)
    for v in (f"{lo:+.1f}%", f"{hi:+.1f}%"):
        assert v.replace("-", "\u2212") in readme or v in readme, \
            f"README 要写出时长扫描的误差范围（本次 {lo:+.1f}% ~ {hi:+.1f}%）"
    # ③ 必须写明它不随时长收敛（否则读者会以为"多跑就更准"）
    assert "不收敛" in readme or "不随时长" in readme


def test_error_band_statement_matches_the_artifact():
    """误差带必须由产物算出，并且写成「范围」而不是「±N%」。"""
    art = _newest()
    errs = [e for e in (_ledger_error(art, k) for k in ("copy", "read", "write"))
            if e is not None]
    if len(errs) < 2:
        pytest.skip(f"{art.name} 里给数的已知答案微基准不足两个")
    low, high = min(errs), max(errs)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ud = "\u2212"
    needles = [f"{low:+.1f}%".replace("-", ud), f"{high:+.1f}%".replace("-", ud),
               f"{low:+.1f}%", f"{high:+.1f}%"]
    assert any(n in readme for n in needles), \
        f"README 必须写出由产物算出的误差带两端（{low:+.1f}% … {high:+.1f}% 附近）"
    # 「±4%」这种写法已经被真机数据推翻过两次，不许再作为**主张**出现。
    # 但「不是 ±4%」这种纠正性表述是允许的（而且正是要保留的），所以逐行判断有没有否定词。
    negation = ("不是", "不再", "不许", "推翻", "错误的", "曾被")
    claims = [ln.strip() for ln in readme.splitlines()
              if "±4%" in ln and not any(n in ln for n in negation)]
    assert not claims, "误差带不能再用 ±4% 作为主张（实测带是单边偏高的）：\n  - " + \
        "\n  - ".join(claims)


def test_source_hash_pins_the_code_that_ran(tmp_path):
    """R-40/D26：产物要能钉住"跑的是哪份源码" —— 不带 `.git` 同步的机器上也要能复核。

    `bench/source_hash.py` 只覆盖真正影响测量行为的文件（src/tools/bench 的 py 与 sh、
    pyproject.toml），所以它必须对源码改动敏感、对产物/文档不敏感。
    """
    import importlib.util
    import shutil

    path = ROOT / "bench" / "source_hash.py"
    spec = importlib.util.spec_from_file_location("ks_source_hash", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "bench").mkdir()
    (root / "tools").mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "bench" / "s.sh").write_text("echo hi\n", encoding="utf-8")
    (root / "bench" / "probe.cu").write_text("__global__ void k() {}\n", encoding="utf-8")
    before = mod.source_hash(root)
    # 源码改了 → 指纹必须变
    (root / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert mod.source_hash(root) != before
    # 探针源码（.cu）改了 → 指纹必须变（账本真值就是它算出来的）
    before_cu = mod.source_hash(root)
    (root / "bench" / "probe.cu").write_text("__global__ void k() { /* v2 */ }\n", encoding="utf-8")
    assert mod.source_hash(root) != before_cu
    # 文档/产物变了 → 指纹不变（它们不影响测量行为）
    after_src = mod.source_hash(root)
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    (root / "bench" / "artifacts").mkdir()
    (root / "bench" / "artifacts" / "x.json").write_text("{}\n", encoding="utf-8")
    assert mod.source_hash(root) == after_src
    # 本仓库自己的指纹是 16 位十六进制
    real = mod.source_hash(ROOT)
    assert len(real) == 16 and all(c in "0123456789abcdef" for c in real)
    shutil.rmtree(root, ignore_errors=True)


def _load_calib_excerpt():
    import importlib.util

    path = ROOT / "tools" / "gen_calibration_excerpt.py"
    spec = importlib.util.spec_from_file_location("ks_calib_excerpt", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_calibration_excerpt_is_generated_from_the_current_artifact():
    """R-56/D13：节选必须是**最新产物**渲染出来的，且门禁能发现不一致。

    旧节选是手抄的，慢慢变成"两轮混合体"（一部分字段来自最新产物、一部分来自上一次，
    `ts` 与 `fp32.ms` 两份都对不上），而门禁只对拍了 3 个字段（还恰好是两份相同的），
    于是一路绿灯。现在它是生成物：值来自产物、`--check` 在门禁里跑。
    """
    mod = _load_calib_excerpt()
    text = (ROOT / "CALIBRATION.md").read_text(encoding="utf-8")
    assert mod.BEGIN in text and mod.END in text
    have = text[text.index(mod.BEGIN):text.index(mod.END) + len(mod.END)]
    assert have == mod.render(), "产物节选被手改了（它是生成物）"
    # R-62：**正文**里那节口径数字也是生成物（旧正文与节选直接矛盾，而覆盖它的 needle
    # 门禁被删掉了 —— 手抄的正文必须没有藏身处）
    body_block = text[text.index(mod.CALIB_BEGIN):text.index(mod.CALIB_END) + len(mod.CALIB_END)]
    assert body_block == mod.render_caliber(), "正文里的口径表被手改了（它是生成物）"
    # 节选里的每个数字都必须能在产物里找到（不是"看起来像"）
    art, prof = mod.newest_profile()
    body = have[have.index("```jsonc") + 9:have.rindex("```")]
    for needle in (str(prof["ts"]), str(prof["measured_dram"]["read_gbs"]),
                   str(prof["l2_info"]["l2_read_peak_gbs"])):
        assert needle in body, needle
    assert art.name in have, "节选要写明取自哪一份产物"


def test_calibration_excerpt_gate_can_go_red(monkeypatch, tmp_path):
    """门禁必须能红：产物换一个值，节选就应当被判不一致。"""
    mod = _load_calib_excerpt()
    art, prof = mod.newest_profile()
    changed = json.loads(json.dumps(prof))
    changed["measured_dram"]["read_gbs"] = round(changed["measured_dram"]["read_gbs"] + 5.0, 1)
    monkeypatch.setattr(mod, "newest_profile", lambda: (art, changed))
    # 走**真实代码路径**（`sync(check=True)`），而不是自己跟自己比 ——
    # 旧版本这里写着 `assert X if hasattr(...) else True`，恒真（N7）。
    assert mod.sync(check=True) == 1, "换了产物值之后 --check 必须变红"
    # 两个块都要能红：口径表那个块同样来自产物
    changed2 = json.loads(json.dumps(prof))
    changed2["measured_matmul_tflops"]["torch.bfloat16"]["caliber_spread_pct"] = 42.0
    monkeypatch.setattr(mod, "newest_profile", lambda: (art, changed2))
    assert mod.sync(check=True) == 1, "口径表块也要能红"
    monkeypatch.undo()
    assert mod.sync(check=True) == 0, "恢复之后必须重新一致"


def _bands_by_artifact() -> dict[str, str]:
    """每份产物能算出的误差带（**有几个负载给数就用几个**）。

    放宽到 `>= 2` 是因为拒绝是正常结果：某次运行里"只写"那行的 per-launch 可能因为
    对齐自检没通过（`window_launch_unaligned`）而**拒绝给数** —— 那是 R-47 的修法在
    正常工作，不该让整份产物的带消失（否则文档会被逼着引用一个不存在的数）。
    """
    out: dict[str, str] = {}
    for art in _artifact_dirs():
        errs = [e for e in (_ledger_error(art, k) for k in ("copy", "read", "write"))
                if e is not None]
        if len(errs) >= 2:
            out[art.name] = f"{min(errs):+.1f}% ~ {max(errs):+.1f}%"
    return out


def test_every_band_quoted_in_the_readme_comes_from_a_registered_artifact():
    """R-56/D27：README 里的每一个误差带都必须能追到**某一份已登记的产物**。

    旧散文写"更早的两次运行里分别是 −4.3% ~ +9.0% 与 −4.3% ~ +12.7%"，而 −4.3~+9.0
    其实是**本次**运行；更早那次是 −4.1% ~ +12.5%。逐产物的带是可算的，所以散文里
    引用的带必须落在"已登记产物给出的带 ∪ 它们的并集"里 —— 引用不到的就得删掉或标注来源。

    只要求"引用得到"，不要求"把所有历史都列出来"：历史产物会一直留着（误差带的另一侧
    需要它），但读者并不需要每次运行都在散文里补一行。
    """
    bands = _bands_by_artifact()
    if len(bands) < 2:
        pytest.skip("只有一个可比对的产物")
    errs_all = []
    for art in _artifact_dirs():
        errs_all += [e for e in (_ledger_error(art, k) for k in ("copy", "read", "write"))
                     if e is not None]
    known = set(bands.values()) | {f"{min(errs_all):+.1f}% ~ {max(errs_all):+.1f}%"}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quoted = {m.group(0) for m in re.finditer(r"[-\u2212]\d+\.\d+%\s*~\s*[-\u2212+]?\d+\.\d+%", readme)}
    norm = {q.replace("\u2212", "-").replace(" ", "") for q in quoted}
    known_norm = {k.replace("\u2212", "-").replace(" ", "") for k in known}
    bad = sorted(norm - known_norm)
    assert not bad, ("README 里这些误差带追不到任何已登记产物（写错或引用了没入库的运行）：\n  - "
                     + "\n  - ".join(bad) + f"\n已登记产物给出的带：{sorted(known)}")


def test_a_new_bench_run_forces_the_docs_to_follow():
    """产物一变，文档就必须跟着变 —— 用"当前最新产物的带"和"上一次的带"对照说明这条链是活的。"""
    bands = _bands_by_artifact()
    if len(bands) < 2:
        pytest.skip("只有一个产物")
    newest = _newest().name
    assert newest in bands, "最新产物应当能算出误差带"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    band = bands[newest]
    assert band in readme or band.replace("-", "\u2212") in readme, \
        f"README 的「本次」误差带必须是 {band}"


# CALIBRATION 散文允许点名、且必须与**现行产物**一致的那些字段（与
# `tools/gen_calibration_excerpt.py` 的取值口径一致：标量 + 两侧峰值 + matmul 两读法）。
# 逐字段登记的理由：散文既讲道理也会点名数字，点名时只能是现行产物的值；旧门禁只登记了
# matmul 的 `tflops`/`caliber_spread_pct`，于是注入旧的 `copy_gbs 910.0` /
# `l2_read_peak_gbs 2195.8` / 旧 `ts` 都无人拦（W-8/R-62）。
def _calib_prose_needles(prof: dict) -> dict[str, set[str]]:
    """`字段路径 -> 该产物在这个字段上的字面串集合`（只收散文可能引用的标量）。"""
    out: dict[str, set[str]] = {}

    def add(what: str, val: Any) -> None:
        if val is None or isinstance(val, bool):
            return
        out.setdefault(what, set()).add(str(val))

    add("ts", prof.get("ts"))
    dram = prof.get("measured_dram") or {}
    for k in ("copy_gbs", "read_gbs", "write_gbs"):
        add(f"measured_dram.{k}", dram.get(k))
    l2_peak = (prof.get("l2_info") or {}).get("l2_read_peak_gbs") or prof.get("l2_read_peak_gbs")
    add("l2_read_peak_gbs", l2_peak)
    for dtype, v in (prof.get("measured_matmul_tflops") or {}).items():
        v = v or {}
        add(f"{dtype}.tflops", v.get("tflops"))
        pct = v.get("caliber_spread_pct")
        if pct is not None:
            add(f"{dtype}.caliber_spread_pct", f"{pct}%")
            out.setdefault(f"{dtype}.caliber_spread_pct", set()).add(str(pct))
    return out


def _calibration_prose() -> str:
    """`CALIBRATION.md` 去掉两个生成块之后的**散文**。"""
    mod = _load_calib_excerpt()
    text = (ROOT / "CALIBRATION.md").read_text(encoding="utf-8")
    for b, e in ((mod.BEGIN, mod.END), (mod.CALIB_BEGIN, mod.CALIB_END)):
        text = text[:text.index(b)] + text[text.index(e) + len(e):]
    return text


_NUM_NEEDLE_RE = re.compile(r"^\d+(?:\.\d+)?%?$")


def _prose_cites(needle: str, prose: str) -> bool:
    """散文里引用了这个字面值吗 —— **数字必须落在词边界上**。

    反例（真实的假阳性）：旧产物的 `float32.caliber_spread_pct=2.6` 会匹配正文里规格表的
    `82.6`；数字 needle 不做边界就只是"看起来像"。`2.6` 命中 `2.6`/`2.6%`，
    但不命中 `82.6`、`2.65`。
    """
    if not _NUM_NEEDLE_RE.match(needle):
        return needle in prose
    body = re.escape(needle[:-1]) + "%" if needle.endswith("%") else re.escape(needle) + "%?"
    return re.search(r"(?<![\d.])" + body + r"(?![\d.])", prose) is not None


def _calibration_prose_offenders(prose: str, newest: dict,
                                 others: list[tuple[str, dict]]) -> list[str]:
    """散文里出现的、**非现行产物**的字段值（W-8/R-62）。

    现行产物自己的值允许出现（它们本来就在生成块里）；某份**旧**产物的值出现在散文里
    就是漂了的证据 —— 除非那个字面串恰好也是现行产物的值（那时无法区分，放过）。
    """
    allowed: set[str] = set()
    for needles in _calib_prose_needles(newest).values():
        allowed |= needles
    offenders: list[str] = []
    for name, prof in others:
        for what, needles in _calib_prose_needles(prof).items():
            for needle in sorted(needles):
                if needle in allowed:
                    continue
                if _prose_cites(needle, prose):
                    offenders.append(f"{name} {what}={needle!r}")
    return offenders


def _calibration_prose_problem(prose: str | None = None) -> str | None:
    """非现行值出现在 CALIBRATION 散文里吗（None = 干净）。"""
    mod = _load_calib_excerpt()
    art, newest = mod.newest_profile()
    others = [(d.name, json.loads((d / "machine-profile.json").read_text(encoding="utf-8")))
              for d in _artifact_dirs()
              if d.name != art.name and (d / "machine-profile.json").exists()]
    text = _calibration_prose() if prose is None else prose
    offenders = _calibration_prose_offenders(text, newest, others)
    if offenders:
        return ("CALIBRATION 散文里手抄了**非现行产物**的数字（它们必须由生成块渲染，"
                "否则会像 P3/W-8 那样漂）：\n  - " + "\n  - ".join(offenders)
                + f"\n现行产物是 {art.name}；要引用数字请引用生成块。")
    return None


def test_calibration_prose_does_not_hand_copy_artifact_numbers():
    """R-62/W-8：CALIBRATION 的**散文**里不许出现"属于某份旧产物"的数字。

    第四轮验收报告 P3 的原形：正文写 bf16 3.8%（可分辨）、fp16 13.2%、fp32 0.7%/54.5、
    头条 166.9 —— 全部来自 `20261008-235142`，而紧邻上方的生成节选来自新产物，两者
    直接矛盾。第五轮只登记了 matmul 的 `tflops`/`caliber_spread_pct`，于是
    `copy_gbs`/`read_gbs`/`write_gbs`/`l2_read_peak_gbs`/`ts` 这五类旧值仍无人拦。
    """
    problem = _calibration_prose_problem()
    assert problem is None, problem


def test_calibration_prose_gate_goes_red_for_every_stale_field():
    """W-8 的负样本：**每一类**被引产物字段的旧值注进散文，门禁都必须报出来。

    （注入发生在内存里：`_calibration_prose_problem(prose)` 走的就是上面那条门禁用的
    同一段判定逻辑，所以这里红 == 那条测试会红。）
    """
    mod = _load_calib_excerpt()
    art, newest = mod.newest_profile()
    others = [(d.name, json.loads((d / "machine-profile.json").read_text(encoding="utf-8")))
              for d in _artifact_dirs()
              if d.name != art.name and (d / "machine-profile.json").exists()]
    assert others, "至少要两份产物才能证明'旧值'会被抓到"
    allowed: set[str] = set()
    for needles in _calib_prose_needles(newest).values():
        allowed |= needles
    real = _calibration_prose()
    stale: list[tuple[str, str]] = []
    for name, prof in others:
        for what, needles in _calib_prose_needles(prof).items():
            for needle in sorted(needles):
                if needle not in allowed:
                    stale.append((f"{name} {what}", needle))
    assert stale, "没有任何旧值可注入 —— 负样本退化了（历史产物都被删了？）"
    miss: list[str] = []
    fields: set[str] = set()
    for what, needle in stale:
        fields.add(what.split(" ", 1)[1])
        problem = _calibration_prose_problem(f"{real}\n\n（正文补一句：本机 {needle}。）\n")
        if not problem or needle not in problem:
            miss.append(f"{what}={needle}")
    assert not miss, ("这些旧值注入散文后门禁没有红（W-8 的字段没纳管）：\n  - "
                      + "\n  - ".join(miss))
    # 五类字段都要有负样本（copy/read/write/l2/ts），否则"每类字段"这句话是空的
    for needed in ("measured_dram.copy_gbs", "measured_dram.read_gbs",
                   "measured_dram.write_gbs", "l2_read_peak_gbs", "ts"):
        assert needed in fields, f"没有 {needed} 的旧值负样本（历史产物缺这个字段？）"


# --------------------------------------------------------------------------
# R-63：登记表里的**每个数字**都要能追到产物
# --------------------------------------------------------------------------
def _registry_rows() -> dict[str, str]:
    """产物目录名 → 登记表里那一行。"""
    reg = (ROOT / "bench" / "artifacts" / "README.md").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in reg.splitlines():
        m = re.match(r"\|\s*`(\d{8}-\d{6})`\s*\|", line)
        if m:
            out[m.group(1)] = line
    return out


def test_registry_revision_matches_both_files():
    """S1：每行写的修订号必须**真的**出现在该产物的 `environment.txt` 与 `manifest.json` 里。

    真事：`20261008-235142` 那行写 `0f296f3`（"environment.txt / manifest.json 里的 revision"），
    而两份文件里都是 `807a025` —— `0f296f3` 在那个目录的任何文件里都不存在。
    """
    rows = _registry_rows()
    bad: list[str] = []
    for art in _artifact_dirs():
        row = rows.get(art.name)
        if row is None:
            bad.append(f"{art.name}: 登记表里没有这一行")
            continue
        env = art / "environment.txt"
        man = art / "manifest.json"
        revs = []
        if env.exists():
            m = re.search(r"^revision: (\S+)", env.read_text(encoding="utf-8"), re.M)
            revs.append((env.name, m.group(1) if m else None))
        if man.exists():
            try:
                revs.append((man.name, (json.loads(man.read_text(encoding="utf-8"))
                                        or {}).get("revision")))
            except json.JSONDecodeError:
                pass
        cited = re.findall(r"`([0-9a-f]{7})`", row)
        if not cited:
            # 必须明确写"钉不住"，且文件里确实没有可用修订号
            if "n/a" not in row and "未留存" not in row:
                bad.append(f"{art.name}: 行里既没写修订号、也没说明为什么")
            continue
        for name, rev in revs:
            if rev and not any(rev.startswith(c) for c in cited):
                bad.append(f"{art.name}: {name} 里的 {rev[:7]} 不在行里 {cited}")
    assert not bad, "登记表的修订号与产物文件对不上：\n  - " + "\n  - ".join(bad)


def test_registry_packet_bytes_are_truthful():
    """S2/P6：行里引用的包字节数必须等于 `cost.packet_bytes`，或**同时**给出磁盘大小并解释。

    真事：现行产物那行写 4088/3842 B，而磁盘是 4089/3843 —— 那个产物由 `80825e9` 之前
    的写入器产生（多写一个换行），而"落盘 == packet_bytes"正是本轮刚立的不变量。
    """
    rows = _registry_rows()
    bad: list[str] = []
    for art in _artifact_dirs():
        row = rows.get(art.name, "")
        # 只校验**被点名**的包文件（登记表通常只引一个代表包，不该逼它列全）
        for m in re.finditer(r"`(rep-[A-Za-z0-9_.-]+\.packet\.json)`", row):
            pk = art / m.group(1)
            if not pk.exists():
                bad.append(f"{art.name}: 行里点名了 {m.group(1)}，但产物里没有这个文件")
                continue
            try:
                pb = json.loads(pk.read_text(encoding="utf-8"))["cost"]["packet_bytes"]
            except (KeyError, json.JSONDecodeError):
                continue
            disk = pk.stat().st_size
            cited = [int(x) for x in re.findall(r"(\d{3,5})\s*B", row)]
            if pb in cited:
                continue
            if disk in cited and "磁盘" in row and str(pb) in row:
                continue
            bad.append(f"{art.name}/{pk.name}: 行里引用的字节数 {cited} 既没有 {pb}"
                       f"（packet_bytes）也没有 {disk}（磁盘）")
    assert not bad, "登记表里的包字节数与产物对不上：\n  - " + "\n  - ".join(bad)


def test_registry_current_section_numbers_come_from_the_current_artifact():
    """A-N1：`### 现行产物…` 小节里的数字必须来自**现行**产物。

    真事：`e087d35` 把新产物提为现行，却把那一节留成上一轮的数（MDD 14.0%、漂移 4.7%、
    delta −3.6%/−3.6%），而现行产物是 MDD 5.0%、SD 0.31%、delta −0.025%/−0.202%。
    """
    reg = (ROOT / "bench" / "artifacts" / "README.md").read_text(encoding="utf-8")
    m = re.search(r"###\s*`?现行产物[^\n]*\n(.*?)(?=\n###|\n##|\Z)", reg, re.S)
    if not m:
        pytest.skip("登记表里没有「现行产物」小节")
    section = m.group(1)
    art = _newest()
    allowed: set[str] = set()
    sc = art / "machine-selfcheck.json"
    if sc.exists():
        mdd = (json.loads(sc.read_text(encoding="utf-8")).get("mdd") or {})
        for k in ("dram_bytes_pct", "kernel_time_pct", "run_to_run_sd_pct",
                  "resolution_floor_pct", "band_half_width_pct"):
            if mdd.get(k) is not None:
                allowed.add(str(mdd[k]))
                allowed.add(f"{mdd[k]}%")
    dpath = art / "diff-same-protocol.json"
    if dpath.exists():
        doc = json.loads(dpath.read_text(encoding="utf-8"))
        for c in doc.get("comparisons", []):
            allowed.add(str(c.get("delta_pct")))
        allowed.add(str((doc.get("summary") or {}).get("indistinguishable")))
    cited = {c.replace("\u2212", "-") for c in re.findall(r"[-\u2212]?\d+(?:\.\d+)?%?", section)}
    allowed = {a.replace("\u2212", "-") for a in allowed}
    # 允许结构性数字（引用包字节数、步骤数、百分比里的 100% 等）与"元"数字（1 B、2 次）
    allowed |= {str(n) for n in range(0, 6)}
    unknown = sorted(c for c in cited if c not in allowed and c.rstrip("%") not in allowed)
    assert not unknown, (f"`现行产物`小节里的 {unknown} 不在 {art.name} 的 selfcheck/diff 里"
                         f"（那一节抄了别的产物？）允许集合：{sorted(allowed)[:12]}…")


_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}


def test_union_label_matches_the_number_of_registered_artifacts():
    """R-64：「N 次运行的并集」里的 **N** 必须等于已登记且能算出带的产物数。

    真事（A-N2）：README 一处写"四次运行的并集 −4.3% ~ +12.8%"（对），另一处写
    "三次并集 −4.3% ~ +12.7%"（既错数又错值）。值有并集校验守着，**标签**没有 ——
    所以这里补上：标签错了也要红。
    """
    n = len(_bands_by_artifact())
    if n < 2:
        pytest.skip("可比对的产物不足")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    bad: list[str] = []
    for m in re.finditer(r"([一两二三四五六七八\d]+)\s*次(?:运行的)?并集", readme):
        raw = m.group(1)
        got = int(raw) if raw.isdigit() else _CN_NUM.get(raw)
        if got != n:
            bad.append(f"「{m.group(0)}」说 {got} 次，实际已登记 {n} 份能算带的产物")
    assert not bad, "\n  - ".join(bad)


def test_every_sweep_row_is_documented():
    """P6：时长扫描**每一行**的覆盖率与误差都要在 README 里出现（不只是两端）。

    真事：README `:290` 写"覆盖率从 88.2% 涨到 **98.0%**"，而表格与产物都是 98.1%。
    """
    art = _newest()
    rows = _sweep_rows(art)
    if len(rows) < 3:
        pytest.skip(f"{art.name} 里没有时长扫描")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    bad: list[str] = []
    for r in rows:
        for key, fmt in (("coverage_pct", "{}%"), ("err_pct", "{:+}%")):
            v = r.get(key)
            if not v:
                continue
            try:
                num = float(v)
            except ValueError:
                continue
            for s in (fmt.format(num), fmt.format(num).replace("-", "\u2212"),
                      fmt.format(round(num, 1))):
                if s in readme:
                    break
            else:
                bad.append(f"{r.get('target_s')}s 的 {key}={num}（README 里没有）")
    assert not bad, "时长扫描的行没被文档覆盖：\n  - " + "\n  - ".join(bad)


def test_artifacts_without_a_pinned_revision_are_marked_unpinnable():
    """R-71：钉不住修订号／脏状态的产物，登记表必须明说"不可钉住"，门禁不许替它背书。"""
    rows = _registry_rows()
    bad: list[str] = []
    for art in _artifact_dirs():
        env = art / "environment.txt"
        if not env.exists():
            continue
        text = env.read_text(encoding="utf-8")
        rev = re.search(r"^revision: (\S+)", text, re.M)
        dirty = re.search(r"^revision_dirty: (\S+)", text, re.M)
        loose = (rev is None or rev.group(1) in ("n/a", "none")
                 or dirty is None or dirty.group(1) == "unknown")
        if loose:
            row = rows.get(art.name, "")
            if not any(k in row for k in ("不可钉住", "不能当现行", "未留存", "钉不住")):
                bad.append(f"{art.name}: revision={rev.group(1) if rev else None} "
                           f"dirty={dirty.group(1) if dirty else None}，但登记表没说明它钉不住")
    assert not bad, "\n  - ".join(bad)
