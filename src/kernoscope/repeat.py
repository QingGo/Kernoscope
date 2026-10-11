"""进程级重复执行（M5 / N1）—— 把「同一个命令跑 N 次」变成**跨运行分布**与 MDD 下界。

为什么需要它：`diff` 只会比两次运行，而「真改进还是噪声」在 N=2 上本来就不可判 ——
本机实测的跨运行噪声可以到 5 个百分点（见 `selftest.py` 里 `SD_FACTOR` / `MDD_FLOOR_DRAM_PCT`
的注释）。MDD（最小可分辨差异）必须来自**跨运行分布**，而这个分布以前只能靠用户手工跑两次
来凑。`repeat_runs()` 提供的就是这个缺失的输入：N 次**独立调用** + 每指标一个分布 + MDD 下界。

**进程级重复不是什么**（这句话要能被读到，因为它决定了这些数字能用来做什么）：

* **不是 kernel 级重放**：本模块不在一个进程内重跑 kernel，也不插桩、不多趟采集 ——
  那条路要特权、会扰动同租户，本项目明确拒绝（见 `refusals.py` 的
  `instrumentation_missing` / `sampling_unavailable` / `no_counters`）。
* **不锁时钟**：不设频率锁、不设 persistence mode、不改功耗上限。GPU 自己降频/升频，
  两次运行之间机器状态可以完全不同。
* **不碰其他租户**：不申请独占、不杀别人的进程、不做任何隔离。别人在跑就在跑，
  那份负载会进你的分布 —— 那属于"噪声"的一部分，不是本模块能消掉的东西。
* **不控制 warm/cold 缓存状态**：第一次运行通常是冷的（L2/显存/页表/编译产物都是空的），
  后面几次是热的。所以**跨运行分布里含缓存效应**：`sd` 不只是仪器噪声，也含
  「第一次比第二次慢」这种系统性差异。要把两者分开只能自己先预热再采。
* **不保证协议一致**：本模块只重复调用，不检查 N 次运行是不是同一个协议
  （同标定 / 同窗口 / 同 kernel 选择 / 同工具版本）。喂给 `diff` 之前必须确认这一点，
  否则差值里混着口径差，MDD 再准也没用。

MDD 下界（`RepeatResult.mdd_lower_bound`）的公式与它的坑：

    mdd_pct = max(MDD_SD_FACTOR × sd / |mean| × 100, MDD_FLOOR_PCT)

* `sd` 是**样本**标准差（n−1 分母，`statistics.stdev`），只在 ≥2 次**成功**运行产出该指标时存在；
* `MDD_SD_FACTOR = 3.0`，与 `selftest.SD_FACTOR` 同一个数、同一个理由：比较**两次**单次测量时，
  差值的噪声是 √2×sd，t(95%, ν≈4)·√2 ≈ 2.8~3σ，不是 2σ（测试钉住两者不漂移）；
* 相对形式（%）只在 `mean ≠ 0` 时有定义：`mean == 0` 时拒绝给相对 MDD（`zero_baseline`），
  但绝对分布照给（min/max/sd 仍可读）；
* `MDD_FLOOR_PCT = 0.5` 兜住「sd 恰好是 0」（N 次完全相同）—— 不许出现 MDD=0，
  那等于把任何抖动都判成真变化。DRAM 字节口径还有 selftest 的 5% 下限，
  **diff 侧要取各来源的最大值**：这个下界只能抬高门槛，永远不能用来下调门槛；
* 它是**下界**而不是无偏估计：N 小（3 次很常见）时样本 sd 本身下偏、只统计成功运行、
  机器状态在 N 次之间会漂移、缓存冷热没被控制 —— 真实噪声只会更大。
  `mdd_lower_bound` 这个名字就是给这个方向留的余量。

价格：本模块只出 **GPU 秒**（`RepeatResult.cost["gpu_s"]`），绝不编电价/租价 ——
要换成钱请把秒数交给 `analyze.cost` 的 `CostInputs`（价格是调用方给的，这条界线与
`refusals.NO_PRICE` 一致）。

不阻塞：模块内没有 sleep、没有重试。`runner` 是**注入的**（测试不许 shell 到真 CLI），
一次抛异常就如实记成一次失败运行，然后接着跑下一次 —— N 是调用方要的次数，
不是「跑到成功为止」。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator

from .refusals import Refusal, RefusalCode, combined_refusal

REPEAT_SCHEMA_VERSION = 1

# 比较**两次单次测量**时的最小可分辨差异：t(95%, ν≈4)·√2 ≈ 2.8~3σ（不是 2σ）。
# 与 selftest.SD_FACTOR 同一个数、同一个理由；测试钉住两者不漂移（同一种统计量，
# 分头漂移会让 diff 的门槛和这里的建议互相打架）。
MDD_SD_FACTOR = 3.0
# sd 恰好为 0（N 次完全相同）时 MDD 不能是 0：那会把任何抖动都判成真变化。
# 0.5 是本仓库既有的「仪器分辨率下限」量级（selftest.kernel_time_pct 用同一个数）；
# DRAM 字节口径另有 selftest 的 5% 下限，diff 侧取最大值 —— 本模块不替它决定。
MDD_FLOOR_PCT = 0.5
# 一个分布至少要几次**成功运行**产出该指标：1 次算不出 sd，也就谈不上分布。
MIN_RUNS_PER_METRIC = 2


@dataclass(frozen=True)
class Sample:
    """一次运行里读到的一个指标样本。"""

    name: str
    value: float
    unit: str = ""
    # 同一份报告里同名指标还出现了**不同**的（值, 单位）：这份报告自己口径不一致。
    # 记为冲突而不是"取第一个" —— 取第一个是猜，猜出来的分布没有意义。
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"value": self.value, "unit": self.unit}
        if self.conflict:
            d["conflict"] = True
        return d


@dataclass
class RunRecord:
    """一次运行的记录。**失败的那次也在这里**（`ok=False` + `rc` + 报告原样保留）。

    `report` 缺失（runner 返回 None / 抛异常）时是 None，但这条记录本身不会消失：
    「只是没成功」与「没跑过」是两件事。
    """

    index: int
    ok: bool
    rc: int
    report: Any = None
    error: str = ""
    gpu_s: float | None = None
    metrics: dict[str, Sample] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"index": self.index, "ok": self.ok, "rc": self.rc,
                             "report": self.report}
        if self.error:
            d["error"] = self.error
        if self.gpu_s is not None:
            d["gpu_s"] = self.gpu_s
        d["metrics"] = {k: v.to_dict() for k, v in sorted(self.metrics.items())}
        return d


@dataclass(frozen=True)
class Distribution:
    """一个指标的跨运行分布：**只含成功运行**产出的样本，且只含单一单位口径。

    `n` 是真正产出该指标的运行次数（不是 N），`not_produced = ok_runs − n`（成功但没产出的次数）；
    `sd` 在 n<2 时是 `None` 而不是 0 —— 「一次」算不出离散度，写成 0 会让下游以为很稳定。
    `values` 留原始样本，让分布可复核（与 selftest 的 `cross_process_samples` 同一条理由）。
    """

    name: str
    unit: str
    n: int
    not_produced: int
    values: tuple[float, ...]
    min: float
    max: float
    mean: float
    sd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "n": self.n,
                "not_produced": self.not_produced, "values": list(self.values),
                "min": self.min, "max": self.max, "mean": self.mean, "sd": self.sd}


@dataclass
class RepeatResult:
    """N 次进程级重复的全部结果。

    * `runs`：逐次记录（成功与失败都在）；
    * `distributions`：指标名 → 分布（单位不一致的指标**不进这里**，见 refusals）；
    * `mdd_lower_bound`：指标名 → MDD 下界（百分比）。值为 `None` ⇒ 有分布但相对 MDD
      算不出来（均值为 0）；**键不存在** ⇒ 连分布都没有（单位口径不一致）；
    * `refusals`：指标名 → 该指标为什么不能用于 MDD（逐指标，粒度到指标）；
    * `refusal`：**整体**拒绝 —— 只有当没有任何一个指标可算 MDD 时才非空。
      有指标可用时它是 None，此时个别指标仍可能在 `refusals` 里被拒（两者要一起看）；
      全部不可用时主 code 取首个（按名排序）被拒指标的 code，其余进 `causes`
      （`combined_refusal` 的语义：不让读者以为只有一个原因）；一个指标都没有时
      用 `mdd_unknown` 兜住，报文里点名"没有产出任何指标"；
    * `cost`：`{"gpu_s": 秒或 None, "runs": N, "runs_with_gpu_s": k, "basis": 口径}`。
      本模块不编价格：`gpu_s` 交出去之后由调用方按自己的价格换算。
    """

    runs: list[RunRecord] = field(default_factory=list)
    distributions: dict[str, Distribution] = field(default_factory=dict)
    mdd_lower_bound: dict[str, float | None] = field(default_factory=dict)
    refusals: dict[str, Refusal] = field(default_factory=dict)
    refusal: Refusal | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok_runs(self) -> int:
        return sum(1 for r in self.runs if r.ok)

    @property
    def usable_metrics(self) -> list[str]:
        """能给出 MDD 下界的指标名（排序）。"""
        return sorted(k for k, v in self.mdd_lower_bound.items() if v is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPEAT_SCHEMA_VERSION,
            "tool": "kernoscope",
            "runs": [r.to_dict() for r in self.runs],
            "distributions": {k: v.to_dict() for k, v in self.distributions.items()},
            "mdd_lower_bound": dict(self.mdd_lower_bound),
            "refusal": self.refusal.to_dict() if self.refusal is not None else None,
            "refusals": {k: v.to_dict() for k, v in self.refusals.items()},
            "cost": dict(self.cost),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """人视图（中文）。数字一律来自上面的结构，不在这里新算任何东西。"""
        out = [f"Kernoscope repeat {REPEAT_SCHEMA_VERSION}"
               f"（进程级重复 {self.cost.get('runs', len(self.runs))} 次）"]
        out.append(f"  运行：{self.ok_runs}/{len(self.runs)} 次成功"
                   f"（失败的那几次照样记在 runs 里、照样算成本，但不参与分布）")
        if self.distributions:
            out.append("  跨运行分布（同指标名、同单位；sd=None 表示样本不足 2 次）：")
            for name in sorted(self.distributions):
                d = self.distributions[name]
                sd = "None" if d.sd is None else f"{d.sd:.6g}"
                mdd = self.mdd_lower_bound.get(name)
                out.append(f"    {name:<28} n={d.n}/{d.n + d.not_produced} {d.unit:<4}"
                           f" min={d.min:.6g} max={d.max:.6g} mean={d.mean:.6g} sd={sd}"
                           + (f"  MDD下界 {mdd}%" if mdd is not None else ""))
        for name in sorted(self.refusals):
            r = self.refusals[name]
            rem = r.remedy_or_default
            out.append(f"  [n/a] {name}：{r.code.value} —— {r.message}"
                       + (f"（{rem.action}）" if rem.action else ""))
        if not self.distributions and self.refusal is not None:
            out.append(f"  没有可用分布：{self.refusal.code.value} —— {self.refusal.message}")
        gpu = self.cost.get("gpu_s")
        out.append("  成本：" + (f"GPU 秒合计 {gpu}" if gpu is not None
                                else "GPU 秒不可得（留空，不拿 wall 时间顶替）")
                   + f"；runs={self.cost.get('runs')}；本模块不编价格")
        for n in self.notes:
            out.append(f"  注：{n}")
        return "\n".join(out)


# --------------------------------------------------------------------------
# 读一份报告：指标在哪
# --------------------------------------------------------------------------
def _as_number(value: Any) -> float | None:
    """只认有限实数。bool 不是数，NaN/±Inf 也不是可用的样本（它们会毒掉 mean/sd）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _entry(obj: Any, name_key: str) -> tuple[str, Any, str] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get(name_key)
    if not isinstance(name, str) or not name.strip():
        return None
    unit = obj.get("unit")
    return name.strip(), obj.get("value"), (unit.strip() if isinstance(unit, str) else "")


def _iter_entries(report: dict[str, Any], layer: str) -> Iterator[tuple[str, Any, str]]:
    """产出 (name, value, unit)。

    layer="metrics" 走报告自己的指标表（`kernels[].sections[].metrics[]`、
    顶层 `sections[].metrics[]`、顶层 `metrics[]`）；layer="verdicts" 走决策包的判词
    （用 `id` 当指标名），因为 diff 吃的正是判词那一层。
    """
    if layer == "verdicts":
        for v in report.get("verdicts") or []:
            got = _entry(v, "id")
            if got is not None:
                yield got
        return
    holders: list[Any] = list(report.get("kernels") or [])
    holders.append(report)
    for holder in holders:
        if not isinstance(holder, dict):
            continue
        for sec in holder.get("sections") or []:
            if not isinstance(sec, dict):
                continue
            for m in sec.get("metrics") or []:
                got = _entry(m, "name")
                if got is not None:
                    yield got
    for m in report.get("metrics") or []:
        got = _entry(m, "name")
        if got is not None:
            yield got


def _collect(entries: Iterator[tuple[str, Any, str]]) -> dict[str, Sample]:
    out: dict[str, Sample] = {}
    for name, value, unit in entries:
        v = _as_number(value)
        if v is None:
            continue            # 缺失 / 被拒 / 非数值：不假装成 0，直接算"本次没产出"
        prev = out.get(name)
        if prev is None:
            out[name] = Sample(name=name, value=v, unit=unit)
        elif (prev.value, prev.unit) != (v, unit):
            out[name] = replace(prev, conflict=True)
    return out


def extract_metrics(report: Any) -> dict[str, Sample]:
    """一份报告 → {指标名: Sample}。读不了（None/不是报告）就是 `{}`，不抛异常。

    报告自己的指标表**优先于**判词层：`verdicts` 是从指标表派生出来的一层，
    同名时以指标表为准（这是**有依据的优先级**，不是"取第一个"）。
    同一层里同名指标给了两个不同的（值, 单位）时标 `conflict=True`——
    随后该指标会因口径不可比被拒，而不是悄悄挑一个。
    """
    d = _as_report_dict(report)
    if d is None:
        return {}
    out = _collect(_iter_entries(d, "verdicts"))
    out.update(_collect(_iter_entries(d, "metrics")))
    return out


# --------------------------------------------------------------------------
# 调 runner
# --------------------------------------------------------------------------
def _normalize_report(report: Any) -> Any:
    """带 `to_dict()` 的报告对象转成 dict（便于序列化）；转不了就原样保留（不丢）。"""
    if report is None or isinstance(report, dict):
        return report
    to_dict = getattr(report, "to_dict", None)
    if not callable(to_dict):
        return report
    try:
        d = to_dict()
    except Exception:           # noqa: BLE001 - 报告对象坏了不等于这次运行没发生过
        return report
    return d if isinstance(d, dict) else report


def _as_report_dict(report: Any) -> dict[str, Any] | None:
    normalized = _normalize_report(report)
    return normalized if isinstance(normalized, dict) else None


def _says_failed(report: Any) -> bool:
    """报告自己说 `ok: false` 时按失败记（只认**显式的** False，不认别的 falsy 值）。"""
    if isinstance(report, dict):
        return report.get("ok") is False
    return getattr(report, "ok", None) is False


def _invoke(runner: Callable[[], Any]) -> tuple[int, Any, str]:
    """跑一次注入的 runner → (rc, report, error)。

    认两种返回：报告本身（dict / 带 to_dict() 的对象）或 `(rc, report)`。
    抛异常**不**带走整批：它是一次失败的运行，如实记下来。
    """
    try:
        got = runner()
    except Exception as e:      # noqa: BLE001 - 一次失败不该让剩下 N-1 次都不跑
        return 1, None, f"{type(e).__name__}: {e}"
    if isinstance(got, (tuple, list)):
        if len(got) != 2:
            return 1, None, f"runner 返回了 {len(got)} 个元素的序列：只认 report 或 (rc, report)"
        rc, report = got
        if isinstance(rc, bool) or not isinstance(rc, int):
            return 1, None, f"runner 的 rc 不是整数：{rc!r}"
        return rc, _normalize_report(report), ""
    if got is None:
        return 1, None, "runner 返回了 None：既没有报告也没有 rc"
    return 0, _normalize_report(got), ""


def default_gpu_s(index: int, rc: int, report: Any) -> float | None:
    """默认口径：报告里的 `gpu_s`（顶层 / `cost.gpu_s` / `device.gpu_s`）；没有就是 None。

    **不拿 wall 时间顶替**：报告里的 `wall_time_ns` 含启动、编译、数据加载、等 I/O 的时间，
    把它当 GPU 秒会把成本算多。要 wall 口径就注入自己的 `gpu_s_of`
    （注入时 `cost["basis"]` 会写明口径来源是调用方）。
    """
    d = _as_report_dict(report)
    if d is None:
        return None
    for path in (("gpu_s",), ("cost", "gpu_s"), ("device", "gpu_s")):
        node: Any = d
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        v = _as_number(node)
        if v is not None and v >= 0:
            return v
    return None


def _gpu_seconds(fn: Callable[[int, int, Any], float | None], index: int, rc: int,
                 report: Any) -> tuple[float | None, str]:
    try:
        got = fn(index, rc, report)
    except Exception as e:      # noqa: BLE001 - 成本口径坏掉不该带走已经跑完的那几次
        return None, f"gpu_s_of 抛了 {type(e).__name__}: {e}（这次按「没给 GPU 秒」记）"
    if got is None:
        return None, ""
    v = _as_number(got)
    if v is None or v < 0:
        return None, f"gpu_s_of 返回了不可用的值 {got!r}：GPU 秒必须是非负有限数"
    return v, ""


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def repeat_runs(runner: Callable[[], Any], n: int, *,
                gpu_s_of: Callable[[int, int, Any], float | None] | None = None) -> RepeatResult:
    """把注入的 `runner` 跑 N 次，汇总成跨运行分布 + MDD 下界（公式与坑见模块 docstring）。

    契约：

    * `runner()` 返回**报告本身**（dict，或带 `to_dict()` 的对象）或 `(rc, report)`；
      返回值里的报告**原样**记进 `runs[i].report`（只做 dict 化以便序列化），
      测试因此可以完全离线（不 shell 到真 CLI）。
    * `ok` 取这次运行自己的成功：`rc == 0`（或报告自己 `ok: false` 时为 False）、
      未抛异常。失败的那次**照样记录**（`ok=False`、`rc`、`error`），
      但它产出的数字**不进分布** —— 失败的报告常常是不完整的（截断/半程），
      把它塞进分布只会把 sd 压小、把 MDD 压低，最后把噪声判成真改进。
    * 成功但报告里没有指标的那次也计入 N 与 `ok_runs`（"跑了但没数"与"没跑"是两件事）。
    * `gpu_s_of(index, rc, report) -> float | None` 默认走 `default_gpu_s`（报告里的 `gpu_s`）；
      找不到就给 None，**不猜、不顶替**（价格同理：本模块不编价格）。
    * 不 sleep、不重试：N 就是调用方要的次数。

    进程级重复**不是** kernel 级重放（不插桩、不多趟采集）、**不锁时钟**、
    **不碰其他租户**、也**不控制 warm/cold 缓存**（跨运行分布里含缓存冷热差异，
    详见模块 docstring）。
    """
    if not callable(runner):
        raise TypeError(f"runner 必须是可调用对象，收到 {type(runner).__name__}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须是 ≥1 的整数（要跑几次），收到 {n!r}")
    gpu_fn = gpu_s_of or default_gpu_s
    runs: list[RunRecord] = []
    gpu_errors: list[str] = []
    for i in range(n):
        rc, report, error = _invoke(runner)
        ok = error == "" and rc == 0
        if report is not None and _says_failed(report):
            ok = False
        gpu_s, gpu_err = _gpu_seconds(gpu_fn, i, rc, report)
        if gpu_err:
            gpu_errors.append(f"第 {i + 1} 次：{gpu_err}")
        runs.append(RunRecord(index=i, ok=ok, rc=rc, report=report, error=error,
                              gpu_s=gpu_s,
                              metrics=extract_metrics(report) if report is not None else {}))
    return _summarize(runs, n, gpu_s_of is not None, gpu_errors)


def _summarize(runs: list[RunRecord], n: int, injected_gpu_caliber: bool,
               gpu_errors: list[str]) -> RepeatResult:
    ok_runs = [r for r in runs if r.ok]
    distributions: dict[str, Distribution] = {}
    mdd_lower_bound: dict[str, float | None] = {}
    refusals: dict[str, Refusal] = {}
    notes: list[str] = []

    units: dict[str, set[str]] = {}
    conflicted: set[str] = set()
    for r in ok_runs:
        for name, s in r.metrics.items():
            units.setdefault(name, set()).add(s.unit)
            if s.conflict:
                conflicted.add(name)

    for name in sorted(units):
        us = units[name]
        values = [r.metrics[name].value for r in ok_runs if name in r.metrics]
        not_produced = len(ok_runs) - len(values)
        if len(us) > 1 or name in conflicted:
            why = ("单位不一致：" + " / ".join(sorted(us)) if len(us) > 1
                   else "同一份报告里同名指标给了两个不同的值或单位")
            bad_reports = sorted(r.index for r in ok_runs
                                 if name in r.metrics and r.metrics[name].conflict)
            refusals[name] = Refusal(
                RefusalCode.PROTOCOL_MISMATCH,
                f"指标 {name} 的口径不可比（{why}）：混在一起算出的分布没有意义"
                f"（sd 里混着两种口径的差），所以这个指标不出分布、不出 MDD；"
                f"请让 N 次运行用同一口径（同单位；同一份报告里同名指标只出现一次）",
                observed={"units": sorted(us), "n": len(values), "ok_runs": len(ok_runs),
                          "runs": n, "conflicted_reports": bad_reports},
                threshold={"units": 1, "conflicted_reports": 0})
            continue

        unit = next(iter(us))
        sd = statistics.stdev(values) if len(values) >= MIN_RUNS_PER_METRIC else None
        dist = Distribution(name=name, unit=unit, n=len(values), not_produced=not_produced,
                            values=tuple(values), min=min(values), max=max(values),
                            mean=statistics.fmean(values), sd=sd)
        distributions[name] = dist

        if len(values) < MIN_RUNS_PER_METRIC:
            mdd_lower_bound[name] = None
            failed_with = sum(1 for r in runs if not r.ok and name in r.metrics)
            refusals[name] = Refusal(
                RefusalCode.MDD_UNKNOWN,
                f"指标 {name} 只有 {len(values)}/{len(ok_runs)} 次**成功**运行产出："
                f"一次算不出 sd，跨运行分布与 MDD 都无从谈起"
                + (f"（另有 {failed_with} 次失败运行也报了它，但失败的报告不参与分布）"
                   if failed_with else ""),
                observed={"n": len(values), "ok_runs": len(ok_runs), "runs": n,
                          "failed_runs_with_metric": failed_with},
                threshold={"min_runs_per_metric": MIN_RUNS_PER_METRIC})
            continue

        if dist.mean == 0:
            mdd_lower_bound[name] = None
            refusals[name] = Refusal(
                RefusalCode.ZERO_BASELINE,
                f"指标 {name} 的均值为 0：相对 MDD（要除以均值）无定义，所以只给绝对分布"
                f"（min/max/sd 仍可读），不给 mdd_lower_bound",
                observed={"mean": 0.0, "min": dist.min, "max": dist.max, "n": dist.n},
                threshold={"mean_nonzero": True})
            continue

        mdd_lower_bound[name] = round(
            max(MDD_SD_FACTOR * float(sd or 0.0) / abs(dist.mean) * 100.0,
                MDD_FLOOR_PCT), 2)

    # ---- 拒绝：逐指标已经有了；整体只在"一个可用指标都没有"时给 ----
    failed = [r for r in runs if not r.ok]
    failed_without_report = [r.index for r in failed if r.report is None and not r.error]
    if failed:
        notes.append(
            f"{len(failed)}/{n} 次运行没有成功（rc≠0 / 报告 ok=False / 抛异常）："
            f"它们照样记在 runs 里、照样算成本，但不参与分布 —— "
            f"失败的报告可能不完整，塞进分布只会把 sd 压小、把 MDD 压低")
    if failed_without_report:
        notes.append(f"第 {'、'.join(str(i + 1) for i in failed_without_report)} 次失败时连报告"
                     f"都没拿到（rc≠0 且 report 为空）：这次能确认的只有 rc")
    ok_without_metrics = [r.index for r in ok_runs if not r.metrics]
    if ok_without_metrics:
        notes.append(f"{len(ok_without_metrics)} 次成功运行没有产出任何指标（第 "
                     f"{'、'.join(str(i + 1) for i in ok_without_metrics)} 次；"
                     f"report 原样留在 runs 里）：它们计入 N，但不贡献样本")
    if refusals:
        notes.append(f"{len(refusals)} 个指标被拒（见 refusals）："
                     f"{'、'.join(sorted(refusals))}")

    usable = [k for k, v in mdd_lower_bound.items() if v is not None]
    refusal: Refusal | None
    if usable:
        refusal = None
    elif refusals:
        refusal = combined_refusal([refusals[k] for k in sorted(refusals)])
    else:
        refusal = Refusal(
            RefusalCode.MDD_UNKNOWN,
            f"{n} 次运行没有产出任何指标：跨运行分布不存在，MDD 无从算起"
            + (f"（{len(failed)} 次失败）" if failed else ""),
            observed={"runs": n, "ok_runs": len(ok_runs), "metrics": 0},
            threshold={"min_runs_per_metric": MIN_RUNS_PER_METRIC})

    # ---- 成本：只出 GPU 秒，不编价格 ----
    measured = [r.gpu_s for r in runs if r.gpu_s is not None]
    gpu_total = round(sum(measured), 6) if measured else None
    cost: dict[str, Any] = {
        "gpu_s": gpu_total,
        "runs": n,
        "runs_with_gpu_s": len(measured),
        "basis": ("调用方注入的 gpu_s_of 口径（本模块不知道它的定义）"
                  if injected_gpu_caliber else
                  "报告里的 gpu_s 字段（顶层 / cost.gpu_s / device.gpu_s）；"
                  "wall 时间不是 GPU 秒，不顶替"),
    }
    if gpu_total is None:
        notes.append("没有任何一次运行给出 GPU 秒（报告里没有 gpu_s，也没注入 gpu_s_of）："
                     "cost.gpu_s 留空，不猜 —— 要换成钱请把秒数与价格交给 analyze.cost")
    elif len(measured) < n:
        notes.append(f"只有 {len(measured)}/{n} 次运行给出 GPU 秒：cost.gpu_s 是这几次的和，"
                     f"不是 N 次的总花费（缺的那些不按 wall 时间顶替）")
    for e in gpu_errors:
        notes.append(e)
    notes.append("跨运行分布含缓存冷热差异：本模块不锁时钟、不预热、不控制 warm/cold，"
                 "跑得越靠后的那次通常越热 —— sd 里同时含仪器噪声与缓存效应")

    return RepeatResult(runs=runs, distributions=distributions,
                        mdd_lower_bound=mdd_lower_bound, refusals=refusals,
                        refusal=refusal, cost=cost, notes=notes)


__all__ = ["MDD_FLOOR_PCT", "MDD_SD_FACTOR", "MIN_RUNS_PER_METRIC", "REPEAT_SCHEMA_VERSION",
           "Distribution", "RepeatResult", "RunRecord", "Sample", "default_gpu_s",
           "extract_metrics", "repeat_runs"]
