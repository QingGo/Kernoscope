"""``kernoscope.backends.ncu_rep`` 的本地契约。

本文件**不依赖** NVIDIA 的 ``ncu_report``：我们自己造一个假模块塞进 ``sys.modules``，
把包装逻辑真的跑一遍（不是 skip），另外用一个子进程证明"模块不存在时本模块照样能导入"。

**真报告口径未验证**（R-107/R-90）：这里验的是**包装逻辑**与失败路径。仓库里没有一份真
ncu 报告 —— 唯一那份 ``.ncu-rep`` 是**本工具自己写的替代口径**（头是
``kernoscope-substitute-v1``），而采一份真报告需要一台 ``ncu`` 能收集计数器的机器
（受限机上实测 ``ERR_NVGPUCTRPERM``，那正是本项目要绕开的前提）。
导出方向是另一回事：我们写出的文件被真 ncu 2025.3.1 ``--import`` 打开过（I2 的证据）。
哪天有脱敏的真报告入库，就补一条"读它 → 出真值行 → 对拍"的端到端测试，并把 README 的
措辞升级回去 —— ``tests/test_claim_maturity.py`` 会在那时变红提醒。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from kernoscope.backends import ncu_rep
from kernoscope.backends.ncu_rep import (
    NCU_METRIC_PREFIX,
    NcuRepAction,
    NcuRepData,
    NcuRepError,
    NcuRepUnavailable,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 假 ncu_report：模仿 2025.3.1 我们用到的那一小块 API
# --------------------------------------------------------------------------- #
class FakeMetric:
    """IMetric 的关键行为，含**最重要的那个坑**：has_value() 为 False 时 value() 返回 0.0。"""

    def __init__(self, value, *, has_value: bool = True, kind: int = 4,
                 unit: str = "", instances: int = 0):
        self._value = value
        self._has_value = has_value
        self._kind = kind
        self._unit = unit
        self._instances = instances

    def has_value(self, *args) -> bool:
        return self._has_value

    def value(self, idx=None):
        if not self._has_value:
            return 0.0  # 真 API 就是这样的：没值也给 0.0，不能当真值发布
        if idx is not None:
            return self._value
        return self._value

    def num_instances(self) -> int:
        return self._instances

    def kind(self, *args) -> int:
        return self._kind

    def unit(self) -> str:
        return self._unit

    def name(self) -> str:
        return "<fake>"


class FakeRule:
    def __init__(self, name: str, rule_id: str, speedup: float | None = None):
        self._name = name
        self._id = rule_id
        self._speedup = speedup

    def name(self) -> str:
        return self._name

    def rule_identifier(self) -> str:
        return self._id

    def has_speedup_estimation(self) -> bool:
        return self._speedup is not None

    def speedup_estimation(self):
        if self._speedup is None:
            return None
        return {"type": 2, "speedup": self._speedup}

    def rule_message(self):
        return {"title": self._name, "message": "...", "type": 1}


class FakeAction:
    def __init__(self, function_name: str, metrics: dict | None = None,
                 rules=(), demangled: str | None = None):
        self._fn = function_name
        self._dem = demangled if demangled is not None else function_name
        self._metrics = dict(metrics or {})
        self._rules = list(rules)

    def name(self, base=None) -> str:
        return self._dem if base == 1 else self._fn  # 1 == NameBase_DEMANGLED

    def metric_names(self):
        return tuple(self._metrics)

    def metric_by_name(self, name):
        return self._metrics.get(name)

    def rule_results(self):
        return tuple(self._rules)

    def workload_type(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"<FakeAction {self._fn}>"


class FakeRange:
    def __init__(self, actions):
        self._actions = list(actions)

    def num_actions(self) -> int:
        return len(self._actions)

    def action_by_idx(self, idx):
        return self._actions[idx]

    def __len__(self) -> int:
        return len(self._actions)


class FakeContext:
    def __init__(self, ranges, version: str = "2025.3.1"):
        self._ranges = list(ranges)
        self._version = version

    def num_ranges(self) -> int:
        return len(self._ranges)

    def range_by_idx(self, idx):
        return self._ranges[idx]

    def get_version(self) -> str:
        return self._version


def install_fake_report(monkeypatch, ranges, version: str = "2025.3.1", load=None):
    """把假 ``ncu_report`` 塞进 sys.modules（monkeypatch 会自动还原）。"""
    mod = types.ModuleType("ncu_report")

    class IAction:
        NameBase_FUNCTION = 0
        NameBase_DEMANGLED = 1
        NameBase_MANGLED = 2

    mod.IAction = IAction
    mod.IMetric = FakeMetric
    mod.load_report = load if load is not None else (lambda path: FakeContext(ranges, version))
    mod.__file__ = "/fake/ncu_report.py"
    monkeypatch.setitem(sys.modules, "ncu_report", mod)
    return mod


# --------------------------------------------------------------------------- #
# 懒加载 / 模块缺失
# --------------------------------------------------------------------------- #
_ABSENT_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, sys

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "ncu_report":
                raise ImportError("blocked: no Nsight Compute on this machine")
            return None

    sys.meta_path.insert(0, Blocker())
    import kernoscope.backends.ncu_rep as nr   # 必须在模块缺失时也能 import

    assert "ncu_report" not in sys.modules, "import ncu_rep must not import ncu_report"
    assert nr.reader_available() is False
    try:
        nr.load("/tmp/nope.ncu-rep")
    except nr.NcuRepUnavailable as exc:
        print("UNAVAILABLE:", type(exc).__name__)
    else:
        raise AssertionError("load() must raise NcuRepUnavailable when the reader is absent")
    print("OK")
    """
)


def test_importable_without_ncu_report_subprocess():
    proc = subprocess.run(
        [sys.executable, "-c", _ABSENT_SCRIPT],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout
    assert "UNAVAILABLE: NcuRepUnavailable" in proc.stdout


def test_reader_available_false_when_module_is_none(monkeypatch):
    # sys.modules[name] = None 会让 import 直接抛 ImportError；等价于"机器上没有这个模块"
    monkeypatch.setitem(sys.modules, "ncu_report", None)
    assert ncu_rep.reader_available() is False
    with pytest.raises(NcuRepUnavailable):
        ncu_rep.load("/tmp/nope.ncu-rep")


def test_reader_available_true_with_fake(monkeypatch):
    install_fake_report(monkeypatch, [FakeRange([FakeAction("k")])])
    assert ncu_rep.reader_available() is True


def test_reader_available_never_raises_on_broken_module(monkeypatch):
    mod = types.ModuleType("ncu_report")  # 没有 load_report
    monkeypatch.setitem(sys.modules, "ncu_report", mod)
    assert ncu_rep.reader_available() is False
    with pytest.raises(NcuRepUnavailable):
        ncu_rep.load("x.ncu-rep")


def test_exception_hierarchy():
    assert issubclass(NcuRepUnavailable, NcuRepError)
    assert issubclass(NcuRepError, Exception)


# --------------------------------------------------------------------------- #
# 正常读取
# --------------------------------------------------------------------------- #
def _sample_action() -> FakeAction:
    return FakeAction(
        "addConstDouble",
        metrics={
            "launch__grid_dim_x": FakeMetric(12288.0),
            "launch__grid_dim_y": FakeMetric(1.0),
            "launch__grid_dim_z": FakeMetric(1.0),
            "launch__block_dim_x": FakeMetric(256.0),
            "launch__block_dim_y": FakeMetric(1.0),
            "launch__block_dim_z": FakeMetric(1.0),
            "dram__bytes.sum": FakeMetric(1048576.0, unit="byte"),
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": FakeMetric(31.01559314119487),
        },
        rules=[
            FakeRule("Issue Slot Utilization", "IssueSlotUtilization", 9.423601439182928),
            FakeRule("Bottleneck", "SOLBottleneck", None),  # 无 speedup -> 不收录
        ],
        demangled="addConstDouble(int, double *, double, double *)",
    )


def test_load_reads_actions_and_identity(tmp_path, monkeypatch):
    install_fake_report(monkeypatch, [FakeRange([_sample_action()])], version="2025.3.1")
    path = tmp_path / "a.ncu-rep"
    path.write_bytes(b"fake")
    data = ncu_rep.load(str(path))

    assert isinstance(data, NcuRepData)
    assert data.path == str(path)
    assert data.tool_version == "2025.3.1"
    assert data.num_ranges == 1
    assert len(data.actions) == 1

    act = data.actions[0]
    assert act.kernel_name == "addConstDouble(int, double *, double, double *)"
    assert act.function_name == "addConstDouble"
    assert act.demangled_name == "addConstDouble(int, double *, double, double *)"
    assert act.grid == (12288, 1, 1)
    assert act.block == (256, 1, 1)
    assert act.metrics["dram__bytes.sum"] == 1048576.0
    assert act.rules == [("Issue Slot Utilization", 9.423601439182928)]


def test_load_flattens_multiple_ranges_and_actions(tmp_path, monkeypatch):
    ranges = [
        FakeRange([FakeAction("k0"), FakeAction("k1")]),
        FakeRange([FakeAction("k2")]),
    ]
    install_fake_report(monkeypatch, ranges)
    p = tmp_path / "m.ncu-rep"
    p.write_bytes(b"fake")
    data = ncu_rep.load(str(p))
    assert data.num_ranges == 2
    assert [a.kernel_name for a in data.actions] == ["k0", "k1", "k2"]


def test_metric_without_value_is_skipped_not_zeroed(tmp_path, monkeypatch):
    """instanced metric 的 has_value() 是 False，value() 却给 0.0 —— 不能发布成 0。"""
    act = FakeAction("k", metrics={
        "launch__grid_dim_x": FakeMetric(4.0),
        "FBSP.TriageSCG.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed":
            FakeMetric(0.0, has_value=False, instances=231),
        "launch__func_cache_config": FakeMetric("prefer_none", has_value=True, kind=3),
        "some__numeric_string": FakeMetric("12.5", kind=3),
    })
    install_fake_report(monkeypatch, [FakeRange([act])])
    p = tmp_path / "n.ncu-rep"
    p.write_bytes(b"fake")
    data = ncu_rep.load(str(p))
    m = data.actions[0].metrics
    assert "FBSP.TriageSCG.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed" not in m
    assert "launch__func_cache_config" not in m
    assert m["some__numeric_string"] == 12.5
    assert data.actions[0].skipped_metrics == 2


def test_action_without_metrics_is_data_not_error(tmp_path, monkeypatch):
    install_fake_report(monkeypatch, [FakeRange([FakeAction("kEmpty")])])
    p = tmp_path / "e.ncu-rep"
    p.write_bytes(b"fake")
    data = ncu_rep.load(str(p))
    assert len(data.actions) == 1
    assert data.actions[0].metrics == {}
    assert data.actions[0].grid == (0, 0, 0)
    assert data.actions[0].block == (0, 0, 0)
    assert data.actions[0].rules == []


def test_metric_order_is_preserved(tmp_path, monkeypatch):
    names = ["z__last", "a__first", "m__middle"]
    act = FakeAction("k", metrics={n: FakeMetric(float(i)) for i, n in enumerate(names)})
    install_fake_report(monkeypatch, [FakeRange([act])])
    p = tmp_path / "o.ncu-rep"
    p.write_bytes(b"fake")
    data = ncu_rep.load(str(p))
    assert list(data.actions[0].metrics) == names


def test_missing_demangled_base_falls_back_to_function_name(tmp_path, monkeypatch):
    mod = install_fake_report(monkeypatch, [FakeRange([FakeAction("kf", demangled="kf(long)")])])
    del mod.IAction.NameBase_DEMANGLED  # 老版本没有这个常量
    p = tmp_path / "f.ncu-rep"
    p.write_bytes(b"fake")
    data = ncu_rep.load(str(p))
    assert data.actions[0].kernel_name == "kf"
    assert data.actions[0].demangled_name == ""


# --------------------------------------------------------------------------- #
# 失败路径：绝不返回"空的但看起来正常"的数据
# --------------------------------------------------------------------------- #
def test_missing_file_raises_ncu_rep_error(tmp_path, monkeypatch):
    install_fake_report(monkeypatch, [FakeRange([FakeAction("k")])])
    missing = tmp_path / "does-not-exist.ncu-rep"
    with pytest.raises(NcuRepError) as ei:
        ncu_rep.load(str(missing))
    assert "not found" in str(ei.value)
    assert not isinstance(ei.value, NcuRepUnavailable)


def test_directory_path_raises(tmp_path, monkeypatch):
    install_fake_report(monkeypatch, [FakeRange([FakeAction("k")])])
    with pytest.raises(NcuRepError) as ei:
        ncu_rep.load(str(tmp_path))
    assert "not a file" in str(ei.value)


def test_reader_exception_is_wrapped_with_original_message(tmp_path, monkeypatch):
    def boom(path):
        raise RuntimeError("Could not load report file: bad magic")

    install_fake_report(monkeypatch, [], load=boom)
    p = tmp_path / "bad.ncu-rep"
    p.write_bytes(b"junk")
    with pytest.raises(NcuRepError) as ei:
        ncu_rep.load(str(p))
    msg = str(ei.value)
    assert "RuntimeError" in msg and "bad magic" in msg


def test_corrupt_file_loading_as_empty_context_raises(tmp_path, monkeypatch):
    """真机上损坏文件会被 load_report() "成功"读成 0 ranges —— 必须报错，不能返回空。"""
    install_fake_report(monkeypatch, [])  # num_ranges() == 0
    p = tmp_path / "corrupt.ncu-rep"
    p.write_bytes(b"\x00\x01\x02")
    with pytest.raises(NcuRepError) as ei:
        ncu_rep.load(str(p))
    assert "0 actions" in str(ei.value)


def test_load_report_returning_none_raises(tmp_path, monkeypatch):
    install_fake_report(monkeypatch, [], load=lambda path: None)
    p = tmp_path / "none.ncu-rep"
    p.write_bytes(b"junk")
    with pytest.raises(NcuRepError) as ei:
        ncu_rep.load(str(p))
    assert "returned None" in str(ei.value)


def test_range_with_zero_actions_raises(tmp_path, monkeypatch):
    install_fake_report(monkeypatch, [FakeRange([])])
    p = tmp_path / "zero.ncu-rep"
    p.write_bytes(b"junk")
    with pytest.raises(NcuRepError):
        ncu_rep.load(str(p))


# --------------------------------------------------------------------------- #
# to_metrics：纯函数、命名规则、不改不排不补
# --------------------------------------------------------------------------- #
def _data() -> NcuRepData:
    return NcuRepData(
        path="/reports/a.ncu-rep",
        tool_version="2025.3.1",
        actions=[
            NcuRepAction(
                kernel_name="addConstDouble(int, double *)",
                grid=(12288, 1, 1), block=(256, 1, 1),
                metrics={
                    "launch__grid_dim_x": 12288.0,
                    "dram__bytes.sum": 1048576.0,
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": 31.01559314119487,
                },
                rules=[("Issue Slot Utilization", 9.4)],
            )
        ],
    )


def test_to_metrics_prefixes_and_keeps_values():
    rows = ncu_rep.to_metrics(_data())
    assert [r[0] for r in rows] == [
        "ncu__launch__grid_dim_x",
        "ncu__dram__bytes.sum",
        "ncu__sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    assert [r[1] for r in rows] == [12288.0, 1048576.0, 31.01559314119487]
    assert NCU_METRIC_PREFIX == "ncu__"


def test_to_metrics_source_is_traceable():
    (name, _value, source), *_ = ncu_rep.to_metrics(_data())
    assert name.startswith("ncu__")
    assert "/reports/a.ncu-rep" in source
    assert "addConstDouble" in source
    assert "2025.3.1" in source


def test_to_metrics_is_pure_no_reader_needed(monkeypatch):
    """纯转换：模块不在（sys.modules 里是 None）也必须能跑。"""
    monkeypatch.setitem(sys.modules, "ncu_report", None)
    assert ncu_rep.reader_available() is False
    assert len(ncu_rep.to_metrics(_data())) == 3


def test_to_metrics_empty_data():
    assert ncu_rep.to_metrics(NcuRepData(path="p", actions=[])) == []


def test_to_metrics_keeps_cross_action_duplicates_in_order():
    data = NcuRepData(path="p", actions=[
        NcuRepAction("k1", (1, 1, 1), (1, 1, 1), {"dram__bytes.sum": 1.0}, []),
        NcuRepAction("k2", (1, 1, 1), (1, 1, 1), {"dram__bytes.sum": 2.0}, []),
    ])
    rows = ncu_rep.to_metrics(data)
    assert rows == [
        ("ncu__dram__bytes.sum", 1.0, "ncu-rep p action[0] k1"),
        ("ncu__dram__bytes.sum", 2.0, "ncu-rep p action[1] k2"),
    ]


def test_ncu_prefixed_names_are_publishable_as_exact():
    """设计意图的落地检查：ncu__ 前缀的名字不触发 provenance 的 R2 加固。"""
    from kernoscope.provenance import Metric, Provenance

    for name, value, source in ncu_rep.to_metrics(_data()):
        m = Metric.exact(name, value, source=source)
        assert m.provenance is Provenance.EXACT
        assert not name.startswith(("dram__", "sm__", "launch__"))
        assert m.render_value()


# --------------------------------------------------------------------------- #
# 子进程隔离读取（**CLI 用的就是这条路径**）
# --------------------------------------------------------------------------- #
# 真机实测：主进程 import 了 torch 之后再读 .ncu-rep，会在 ncu_report.get_version() 里
# Segmentation fault（rc=139）；同一份文件单独读却完全正常。段错误抓不住，所以读取被放进
# 子进程。这里用"临时目录里放一个假 ncu_report"的方式，把隔离通道真的跑一遍。
_FAKE_REPORT_MODULE = '''
class _M:
    def __init__(self, v): self._v = v
    def has_value(self, *a): return True
    def value(self, idx=None): return self._v
    def num_instances(self): return 0
    def kind(self, *a): return 4
    def unit(self): return ""


class _A:
    NameBase_FUNCTION = 0
    NameBase_DEMANGLED = 1
    NameBase_MANGLED = 2
    def __init__(self): self._m = {"sm__throughput.avg.pct_of_peak_sustained_elapsed": _M(31.5),
                                   "launch__registers_per_thread": _M(16.0)}
    def name(self, base=None): return "fake_kernel" if base != 1 else "fake_kernel(int)"
    def metric_names(self): return tuple(self._m)
    def metric_by_name(self, n): return self._m.get(n)
    def rule_results(self): return ()
    def workload_type(self): return 0


class _R:
    def num_actions(self): return 1
    def action_by_idx(self, i): return _A()
    def __len__(self): return 1


class _C:
    def num_ranges(self): return 1
    def range_by_idx(self, i): return _R()
    def get_version(self): return "2025.3.1"


def load_report(path):
    return _C()
'''


def _fake_report_dir(tmp_path):
    d = tmp_path / "fakereport"
    d.mkdir()
    (d / "ncu_report.py").write_text(_FAKE_REPORT_MODULE, encoding="utf-8")
    return d


def test_load_isolated_round_trips_through_a_clean_subprocess(tmp_path, monkeypatch):
    """隔离通道的完整往返：子进程读 -> JSON -> 父进程重建（含 grid/block/规则字段）。"""
    fake = _fake_report_dir(tmp_path)
    report = tmp_path / "r.ncu-rep"
    report.write_bytes(b"NVR\x00" + b"\x00" * 8)      # 内容由假模块决定，这里只要存在
    monkeypatch.setenv("PYTHONPATH", str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    data = ncu_rep.load_isolated(str(report))
    assert data.tool_version == "2025.3.1"
    assert len(data.actions) == 1
    action = data.actions[0]
    assert action.kernel_name == "fake_kernel"
    assert action.metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 31.5
    assert action.metrics["launch__registers_per_thread"] == 16.0


def test_load_isolated_translates_each_failure_to_its_own_exception(tmp_path, monkeypatch):
    """三种失败要能被区分：缺模块 / 文件不存在 / 报告没有 action —— 处置完全不同。"""
    empty_path = str(tmp_path / "nothing.ncu-rep")
    with pytest.raises(NcuRepUnavailable):
        ncu_rep.load_isolated(empty_path)

    fake = _fake_report_dir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with pytest.raises(NcuRepError) as e:
        ncu_rep.load_isolated(empty_path)            # 模块在，文件不在
    assert "not found" in str(e.value)


def test_cli_uses_the_isolated_reader(monkeypatch):
    """接线断言：CLI 必须走 load_isolated（走回 load 就等于把真机 rc=139 的崩溃带回来）。"""
    from kernoscope import cli as cli_mod

    calls = []
    monkeypatch.setattr(ncu_rep, "load_isolated",
                        lambda path, **kw: (calls.append(path),
                                            NcuRepData(path=path, actions=[]))[1])
    monkeypatch.setattr(ncu_rep, "load", lambda path: pytest.fail("不该在进程内读 .ncu-rep"))

    class FakeRep:
        kernels, meta = [], {}

    cli_mod._attach_truth(FakeRep(), "/tmp/whatever.ncu-rep", lambda *_: None)
    assert calls == ["/tmp/whatever.ncu-rep"]


# --------------------------------------------------------------------------- #
# N3：来源档位（`source_class`）是类型，不是注释
# --------------------------------------------------------------------------- #
# 这一节钉的是"名字与来源相容"这条规则的边界：有人写错、或者规则被回退时，
# 必须在**最底层的那一处**（唯一拿得到 ncu 行定义的地方）拒绝，而不是靠调用方自觉。
def test_source_class_enum_is_a_str_enum_with_the_four_approved_levels():
    """四档的名称与含义由 NON-GOALS 的 N3 钉住；`str` 子类便于直接进 JSON/CLI。"""
    from kernoscope.ncrep.source_class import SourceClass

    assert [c.value for c in SourceClass] == [
        "documented", "documented-adjacent", "undocumented-observed", "vendor-tool-output"]
    assert isinstance(SourceClass.DOCUMENTED, str)
    assert SourceClass("documented-adjacent") is SourceClass.DOCUMENTED_ADJACENT
    # 每一档都要能自证"为什么如此"（机器可读署名里逐档印出来）
    assert "NVML_FI_DEV_*" in SourceClass.DOCUMENTED.justification
    assert "enum" in SourceClass.DOCUMENTED_ADJACENT.justification
    assert "blob" in SourceClass.UNDOCUMENTED_OBSERVED.justification


def test_consensus_check_rejects_a_row_whose_source_class_forbids_its_own_name(monkeypatch):
    """登记在 `NCU_ROWS` 里却不许用 ncu 名 = 表与规则脱节：自查必须点名，而不是静默。

    没有这一条，降级一行的 `source_class` 会得到一个"永远写不出去、但看起来还在白名单里"
    的行 —— 那种状态最容易被下一个人误读成"这行能用"。
    """
    from dataclasses import replace

    from kernoscope.ncrep import ncu_rows
    from kernoscope.ncrep.source_class import SourceClass

    row = ncu_rows.row_for("launch__grid_size")
    assert row is not None
    monkeypatch.setitem(ncu_rows._BY_NAME, row.name,
                        replace(row, source_class=SourceClass.UNDOCUMENTED_OBSERVED))
    problems = ncu_rows.consensus_check()
    assert any("launch__grid_size" in p and "undocumented-observed" in p for p in problems), \
        problems
    # 反过来：档位回到 documented，这条问题就消失（其余自洽性检查仍然绿）
    monkeypatch.setitem(ncu_rows._BY_NAME, row.name, row)
    assert ncu_rows.consensus_check() == []


def test_row_lookup_refuses_a_name_or_unit_that_cannot_be_published(monkeypatch):
    """`allows_ncu_row_name` / `ncu_name_refusal`：**单一判定入口**，理由要能说清是哪道门。

    两个调用方（`select_metrics` 的过滤、`assert_writable` 的兜底）都走这里，
    所以回退只有一处可回退 —— 这也是"规则可测试"的前提。
    """
    from dataclasses import replace

    from kernoscope.ncrep import ncu_rows
    from kernoscope.ncrep.source_class import SourceClass

    assert ncu_rows.allows_ncu_row_name("gpu__time_duration.sum", "ms") is True
    assert ncu_rows.allows_ncu_row_name("gpu__time_duration.sum", "us") is False
    assert ncu_rows.ncu_name_refusal("gpu__time_duration.sum", "ms") == ""
    assert "单位" in ncu_rows.ncu_name_refusal("gpu__time_duration.sum", "us")
    assert "白名单" in ncu_rows.ncu_name_refusal("nope__not_a_row", "ms")

    row = ncu_rows.row_for("gpu__time_duration.sum")
    assert row is not None
    monkeypatch.setitem(ncu_rows._BY_NAME, row.name,
                        replace(row, source_class=SourceClass.DOCUMENTED_ADJACENT))
    assert ncu_rows.allows_ncu_row_name(row.name, "ms") is False
    reason = ncu_rows.ncu_name_refusal(row.name, "ms")
    assert "documented-adjacent" in reason and "只有 documented" in reason
    assert ncu_rows.source_class_of(row.name) is SourceClass.DOCUMENTED_ADJACENT
    assert ncu_rows.source_class_of("nope__not_a_row") is None
