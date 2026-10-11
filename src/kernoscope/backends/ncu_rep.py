"""ncu-rep 后端（只读）：把在别处采集的 ``.ncu-rep`` 当作 ground truth 吃进来。

**不变量**：本模块只在**真的读到了 ncu 报告**时才产出数字。读不到就抛异常，
绝不返回"空的但看起来正常"的数据 —— 那正是本项目最怕的假数据。

身份与名字的分离（R2/R3 的落点）：
    报告里的原始名字已经带计数器命名空间（``dram__`` / ``sm__`` / …）。如果直接
    以原样发布，它会撞上 ``provenance`` 的 R2 加固（计数器名下不许有值，因为本机
    拿不到计数器）。所以 ``to_metrics()`` 一律加 ``ncu__`` 前缀：值的出处确实是
    exact（与 ncu 同源），但名字不可能被误认成"本工具自己测出来的数"。

    反过来说：本模块**不**改写、不重排、不补值。``launch__`` / ``gpu__time_duration``
    这类非计数器名字也照样加前缀，保持规则唯一（同一个来源 → 同一种命名）。

懒加载：``ncu_report`` 只在函数里 import。本模块必须在没有 Nsight Compute 的机器上
可导入（有测试守），``reader_available()`` 永远不抛异常。

Python API 的最小面（Nsight Compute 2025.3.1 实测，见 tests/test_ncu_rep.py 的假模块）::

    ctx  = ncu_report.load_report(path)      # -> IContext（自动推导 library_dir）
    ctx.get_version()                        # "2025.3.1"
    ctx.num_ranges() / ctx.range_by_idx(i)   # -> IRange
    rng.num_actions() / rng.action_by_idx(i) # -> IAction
    act.name()                               # 短名（NameBase_FUNCTION，默认）
    act.name(ncu_report.IAction.NameBase_DEMANGLED)   # 全签名
    act.metric_names() / act.metric_by_name(n)        # -> IMetric | None
    m.has_value() / m.value() / m.num_instances() / m.kind()
    act.rule_results()                       # -> tuple[IRuleResult]
    rr.name() / rr.rule_identifier() / rr.has_speedup_estimation() / rr.speedup_estimation()

实测三个坑（都写进了实现）：

1. ``load_report()`` 对**损坏文件不报错**：随机字节 / 纯文本 / 空文件都"成功"返回一个
   ``num_ranges() == 0`` 的 IContext。所以"文件能读"不能当作"数据有效" —— 见 ``load()``
   对 0 action 的处理。
2. ``has_value()`` 为 False 的 metric（例如只有 per-instance 值的 instanced metric）
   调 ``value()`` 会静默返回 ``0.0``。把 0.0 当真值发布就是凭空造数，所以必须跳过。
3. numpy/不可打印的字符串型 metric（``launch__func_cache_config`` 之类）不能塞进
   ``float``；能转成数字的字符串保留（"12.5" → 12.5），其余计入 ``skipped_metrics``。
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

NCU_REPORT_MODULE = "ncu_report"

#: 发布前缀：``dram__bytes.sum`` -> ``ncu__dram__bytes.sum``（见模块 docstring）
NCU_METRIC_PREFIX = "ncu__"

_GRID_METRICS = ("launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z")
_BLOCK_METRICS = ("launch__block_dim_x", "launch__block_dim_y", "launch__block_dim_z")


class NcuRepError(Exception):
    """读取 ``.ncu-rep`` 失败：文件不存在、损坏、或内容无法解析。"""


class NcuRepUnavailable(NcuRepError):
    """``ncu_report`` 模块（或它的 reader）不可导入 —— 本机没有 Nsight Compute。"""


@dataclass
class NcuRepAction:
    """报告里的一个 action（通常就是一个 kernel launch）。

    ``metrics`` 只放**能诚实转成 float** 的值：名字是报告里的原始 metric 名（不改名），
    顺序与 ``metric_names()`` 一致。没有可发布值的 metric 不进 dict（``{}`` 是合法的数据，
    不是错误），跳过的条数记在 ``skipped_metrics`` 里，便于调用方解释"为什么条数变少了"。

    ``rules`` 只收录**带 speedup 估计**的规则结果：``(rule name, estimated speedup)``。
    没有 speedup 的规则（例如 SOLBottleneck/Bottleneck）无法用 float 表达，不收录 ——
    用 0.0 占位就是造数。
    """

    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    metrics: dict[str, float]
    rules: list[tuple[str, float]]
    # --- 身份补充（默认值为空，老调用方按前 5 个字段构造不受影响）---
    function_name: str = ""          # IAction.name()，ncu 默认的 NameBase_FUNCTION
    demangled_name: str = ""         # IAction.name(NameBase_DEMANGLED)
    skipped_metrics: int = 0         # 有名字但没发布成 float 的 metric 条数


@dataclass
class NcuRepData:
    """一个 ``.ncu-rep`` 的读数。

    ``comment`` 恒为空字符串：Nsight Compute 的 Python API **不暴露**报告里的注释/provider
    字段（只有 ``IBaseContext.get_version()``）。这里留字段是为了如实说明"读不到"，
    而不是假装读到过。
    """

    path: str
    actions: list[NcuRepAction]
    tool_version: str = ""           # IContext.get_version()，例如 "2025.3.1"
    provider: str = ""               # 实际用的 reader（ncu_report 模块路径）
    num_ranges: int = 0
    comment: str = ""                # Python API 读不到；恒为 ""


# --------------------------------------------------------------------------- #
# reader 发现（懒加载；永远不抛）
# --------------------------------------------------------------------------- #


def _import_reader() -> Any:
    """import ``ncu_report``；失败一律转成 :class:`NcuRepUnavailable`。

    单独抽出来是为了让 ``reader_available()`` 能用同一个判定 —— 检查"模块在不在"和
    "能不能真的读报告"必须是同一个答案。
    """
    try:
        mod = importlib.import_module(NCU_REPORT_MODULE)
    except Exception as exc:  # ImportError，或坏掉的 .so 抛 OSError
        raise NcuRepUnavailable(
            f"{NCU_REPORT_MODULE} is not importable ({type(exc).__name__}: {exc}); "
            "Nsight Compute is not installed here - put its extras/python on PYTHONPATH "
            "(ncu cannot run on this machine anyway, but a .ncu-rep collected elsewhere can "
            "still be ingested)"
        ) from exc
    if getattr(mod, "load_report", None) is None:
        raise NcuRepUnavailable(
            f"{NCU_REPORT_MODULE} has no load_report(): unsupported or broken Nsight Compute build"
        )
    return mod


def reader_available() -> bool:
    """能不能 import ``ncu_report``。**永远不抛异常**。"""
    try:
        _import_reader()
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- #
# 读报告
# --------------------------------------------------------------------------- #


def _metric_float(metric: Any) -> float | None:
    """把一个 IMetric 转成 float；没有可发布的值就返回 None（**不返回 0.0**）。

    ``has_value() is False`` 必须当"没有值"处理：实测这种 metric（只剩 per-instance
    值的 instanced metric，如 ``FBSP.TriageSCG.dramc__read_throughput...``）调 ``value()``
    会返回 0.0。把 0.0 发布成真值就是编数据。
    """
    if metric is None:
        return None
    try:
        if not metric.has_value():
            return None
    except Exception:
        # 老版本可能没有 has_value()；继续尝试取值，取不到就是 None
        pass
    try:
        raw = metric.value()
    except Exception:
        return None
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None


def _read_metrics(act: Any) -> tuple[dict[str, float], int]:
    """-> (原始名 -> float, 未发布的条数)。顺序即 ``metric_names()`` 的顺序。"""
    try:
        names = list(act.metric_names())
    except Exception as exc:
        raise NcuRepError(
            f"cannot enumerate metric names (metric_names() failed: {type(exc).__name__}: {exc})"
        ) from exc
    out: dict[str, float] = {}
    skipped = 0
    for name in names:
        try:
            metric = act.metric_by_name(name)
        except Exception:
            metric = None
        value = _metric_float(metric)
        if value is None:
            skipped += 1
            continue
        out[name] = value
    return out, skipped


def _read_names(act: Any, mod: Any) -> tuple[str, str]:
    """-> (function/short name, demangled name)；取不到就是空串。"""
    function_name = ""
    try:
        function_name = str(act.name() or "")
    except Exception:
        function_name = ""
    demangled_name = ""
    base = getattr(getattr(mod, "IAction", None), "NameBase_DEMANGLED", None)
    if base is not None:
        try:
            demangled_name = str(act.name(base) or "")
        except Exception:
            demangled_name = ""
    return function_name, demangled_name


def _read_rules(act: Any) -> list[tuple[str, float]]:
    """只收录带 speedup 估计的规则结果；规则 API 缺失时返回 ``[]``（不是错误）。"""
    try:
        results = list(act.rule_results())
    except Exception:
        return []
    out: list[tuple[str, float]] = []
    for rule in results:
        try:
            if not rule.has_speedup_estimation():
                continue
            est = rule.speedup_estimation() or {}
            speedup = est.get("speedup")
            if speedup is None:
                continue
            name = str(rule.name() or "") or str(rule.rule_identifier() or "")
            out.append((name, float(speedup)))
        except Exception:
            continue
    return out


def _dim3(metrics: dict[str, float], keys: tuple[str, str, str]) -> tuple[int, int, int]:
    """从 ``launch__grid_dim_x`` 一类的 metric 里取三元组；缺一个就是 0。

    grid/block 没有独立的 Python accessor，只能走 metric。缺字段时补 0 是**如实**的
    （"这一维没读到"），不拿 ``launch__grid_size`` 去反推 —— 反推出来的是编的。
    """
    return tuple(  # type: ignore[return-value]
        int(metrics[k]) if k in metrics else 0 for k in keys
    )


def _read_action(act: Any, mod: Any) -> NcuRepAction:
    function_name, demangled_name = _read_names(act, mod)
    metrics, skipped = _read_metrics(act)
    return NcuRepAction(
        kernel_name=demangled_name or function_name or "<unknown>",
        grid=_dim3(metrics, _GRID_METRICS),
        block=_dim3(metrics, _BLOCK_METRICS),
        metrics=metrics,
        rules=_read_rules(act),
        function_name=function_name,
        demangled_name=demangled_name,
        skipped_metrics=skipped,
    )


def _read_actions(rng: Any, mod: Any) -> list[NcuRepAction]:
    try:
        count = int(rng.num_actions())
    except Exception as exc:
        raise NcuRepError(
            f"num_actions() failed: {type(exc).__name__}: {exc}"
        ) from exc
    out: list[NcuRepAction] = []
    for i in range(count):
        try:
            act = rng.action_by_idx(i)
        except Exception as exc:
            raise NcuRepError(
                f"action_by_idx({i}) failed: {type(exc).__name__}: {exc}"
            ) from exc
        out.append(_read_action(act, mod))
    return out


def load(path: str) -> NcuRepData:
    """读一个 ``.ncu-rep``。

    Raises:
        NcuRepUnavailable: 本机没有 ``ncu_report``。
        NcuRepError: 路径不存在/不是文件、reader 抛错/返回 None、或报告加载后
            **一个 action 都没有**（实测损坏文件就是这个样子：``load_report()`` 不报错，
            返回 ``num_ranges() == 0`` 的空 context）。这里宁可报错，也不返回空 ground truth。
    """
    mod = _import_reader()  # 先判能力，再判文件
    path = str(path)
    if not path:
        raise NcuRepError("empty path")
    if os.path.isdir(path):
        raise NcuRepError(f"not a file: {path}")
    if not os.path.isfile(path):
        raise NcuRepError(f"ncu-rep not found: {path}")

    try:
        ctx = mod.load_report(path)
    except Exception as exc:
        raise NcuRepError(
            f"ncu_report.load_report({path!r}) failed: {type(exc).__name__}: {exc}"
        ) from exc
    if ctx is None:
        raise NcuRepError(
            f"ncu_report.load_report({path!r}) returned None - corrupt or unsupported report"
        )

    try:
        tool_version = str(ctx.get_version() or "")
    except Exception:
        tool_version = ""

    # 新 API：report -> ranges -> actions。老 API 直接就是 range，用 num_ranges() 探测。
    try:
        num_ranges = int(ctx.num_ranges())
    except Exception:
        num_ranges = -1
    ranges: list[Any] = []
    if num_ranges >= 0:
        for i in range(num_ranges):
            try:
                ranges.append(ctx.range_by_idx(i))
            except Exception as exc:
                raise NcuRepError(
                    f"range_by_idx({i}) failed: {type(exc).__name__}: {exc}"
                ) from exc
    else:
        ranges.append(ctx)

    actions: list[NcuRepAction] = []
    for rng in ranges:
        actions.extend(_read_actions(rng, mod))

    if not actions:
        raise NcuRepError(
            f"{path}: loaded but yielded 0 actions (num_ranges={max(num_ranges, 0)}) - "
            "not a valid Nsight Compute report, or a report with no profiled kernels; "
            "refusing to hand back empty ground truth"
        )

    return NcuRepData(
        path=path,
        actions=actions,
        tool_version=tool_version,
        provider=str(getattr(mod, "__file__", "") or NCU_REPORT_MODULE),
        num_ranges=max(num_ranges, 0),
    )


# --------------------------------------------------------------------------- #
# 纯转换：NcuRepData -> provenance 三元组
# --------------------------------------------------------------------------- #


def to_metrics(data: NcuRepData) -> list[tuple[str, float, str]]:
    """``-> [(our_metric_name, value, source_string)]``（纯函数，不碰 IO/reader）。

    命名规则（已定的设计）：原始名 ``X`` -> ``ncu__X``，值照抄，``provenance=exact``
    （同源）。**不改名、不重排、不补值**：``dram__bytes.sum`` -> ``ncu__dram__bytes.sum``。

    跨 action 同名 metric 会出现多条（各带自己的 source），因为每个 action 都是真实读数；
    去重/聚合是调用方的事，这里不替它决定。
    """
    out: list[tuple[str, float, str]] = []
    for i, action in enumerate(data.actions):
        source = f"ncu-rep {data.path} action[{i}] {action.kernel_name}"
        if data.tool_version:
            source += f" (ncu {data.tool_version})"
        for raw_name, value in action.metrics.items():
            out.append((NCU_METRIC_PREFIX + raw_name, float(value), source))
    return out
