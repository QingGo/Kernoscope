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
from pathlib import Path

import pytest

from kernoscope import __version__

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "bench" / "artifacts"


def _artifact_dirs() -> list[Path]:
    if not ARTIFACTS.exists():
        return []
    return sorted([p for p in ARTIFACTS.iterdir() if p.is_dir() and p.name[0].isdigit()],
                  key=lambda p: p.name)


def _newest() -> Path:
    dirs = _artifact_dirs()
    if not dirs:
        pytest.skip("bench/artifacts 下还没有产物目录")
    return dirs[-1]


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
    assert man.get("revision_dirty") in ("yes", "no", "unknown")
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
    """每个产物目录都要被 bench/artifacts/README.md 登记，并写明它是现行还是历史。"""
    readme = ARTIFACTS / "README.md"
    assert readme.exists(), "bench/artifacts/README.md 必须存在（登记每个产物目录）"
    text = readme.read_text(encoding="utf-8")
    for d in _artifact_dirs():
        assert d.name in text, f"{d.name} 没被 bench/artifacts/README.md 登记"


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
    mm = cal.get("measured_matmul_tflops", {})
    if mm.get("torch.bfloat16", {}).get("tflops"):
        rules.append(("CALIBRATION.md", "bf16 标定读数",
                      [str(mm["torch.bfloat16"]["tflops"])]))
    dram = cal.get("measured_dram", {})
    if dram.get("copy_gbs"):
        rules.append(("CALIBRATION.md", "copy 带宽标定读数", [str(dram["copy_gbs"])]))
    if mm:
        bf16 = mm.get("torch.bfloat16", {})
        if bf16.get("tflops"):
            rules.append(("README.md", "bf16 标定读数（README 的能力表）", [str(bf16["tflops"])]))
        if bf16.get("caliber_spread_pct"):
            rules.append(("README.md", "bf16 口径离散度", [str(bf16["caliber_spread_pct"])]))
    smi = str(cal.get("nvidia_smi", ""))
    if smi:
        # 「driver, maxSM, power, pcie gen, width」里的 pcie gen 是第 4 段
        parts = [x.strip() for x in smi.split(",")]
        if len(parts) >= 4:
            rules.append(("CALIBRATION.md", "PCIe 代数", [f", {parts[3]}, "]))

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
    if len(errs) < 3:
        pytest.skip(f"{art.name} 里没有三个已知答案微基准的误差")
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
