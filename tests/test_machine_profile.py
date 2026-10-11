"""标定脚本（tools/machine-profile.py）的回归测试。

守两条需求方实测出来的缺陷：
  R-18/B7-a  默认 `--out machine-profile.json`（裸文件名）不得崩 —— 它是 README 与
             doctor 共同推荐的第一步，真机上 rc=1 并抛 `FileNotFoundError: ''`
  R-09/B9    标定产物必须记录累加口径、precision 设置、方法，以及**口径离散度**
             （同一台机器相隔一天，两个脚本对"口径可不可分辨"给出相反判断）
  R-50/D7    口径探针必须"应用 → 测量 → 恢复"：旧实现把三次 `_set_precision` 都放在
             `_variants()` 构造列表时调用，三个副作用在第一次测量前全部落地，于是三个变体
             其实是在同一个设置下测的，而产物里记的是写死的字面量。`precision_settings`
             必须是测量那一刻的实况快照（见两个探针测试）。
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_module():
    spec = importlib.util.spec_from_file_location("ks_machine_profile",
                                                 TOOLS / "machine-profile.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_without_torch_and_help_works(capsys):
    """没有 torch 的环境里也必须能 --help（否则连用法都看不到）。"""
    mod = _load_module()
    with pytest.raises(SystemExit) as e:
        mod.main.__globals__  # noqa: B018 - 只是确认模块对象可用
        sys.argv = [sys.argv[0], "--help"]
        mod.main()
    assert (e.value.code or 0) == 0
    assert "--out" in capsys.readouterr().out


def test_write_profile_accepts_bare_filename(tmp_path, monkeypatch):
    """R-18 的负样本：裸文件名（默认值）与深层目录都必须成功。"""
    mod = _load_module()
    monkeypatch.chdir(tmp_path)
    p = mod.write_profile({"a": 1}, "machine-profile.json")
    assert Path(p).exists() and Path(p).name == "machine-profile.json"
    assert json.loads(Path(p).read_text())["a"] == 1

    deep = mod.write_profile({"b": 2}, str(tmp_path / "sub" / "dir" / "x.json"))
    assert Path(deep).exists()


class _FakeProps:
    name = "Fake RTX"
    multi_processor_count = 128
    major, minor = 8, 9
    total_memory = 24 * 2 ** 30
    L2_cache_size = 72 * 2 ** 20


class _Arr:
    def __matmul__(self, other):
        return self

    def __getitem__(self, item):
        return self


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"
    __version__ = "2.12.1+cu130"

    class cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_properties(i=0):
            return _FakeProps()

        @staticmethod
        def empty_cache():
            return None


_PRECISION_FIELDS = ("allow_fp16_reduced_precision_reduction", "fp32_precision",
                     "allow_tf32")


class _RecordingMatmul:
    """假 `torch.backends.cuda.matmul`：属性可读可写，写入时留痕（R-50/D7 的探针用）。

    `writes` 是累积的 `(字段, 值)` 日志 —— 可以据此判断"应用"到底发生在"测量"之前还是之后。
    读不到的属性（如 `float32_matmul_precision`）就是普通 AttributeError，正好覆盖
    "这个 torch 版本读不到这一项"的分支。
    """

    def __init__(self, allow_reduced=True, fp32_precision="none", allow_tf32=True):
        object.__setattr__(self, "writes", [])
        object.__setattr__(self, "allow_fp16_reduced_precision_reduction", allow_reduced)
        object.__setattr__(self, "allow_tf32", allow_tf32)
        object.__setattr__(self, "fp32_precision", fp32_precision)

    def __setattr__(self, name, value):
        if name in _PRECISION_FIELDS:
            self.writes.append((name, value))
        object.__setattr__(self, name, value)


def _probe_torch(matmul):
    """只提供 `matmul_peak` 走得到的那部分 API 的假 torch。"""

    class T:
        bfloat16, float16, float32 = "bf16", "fp16", "fp32"
        __version__ = "2.12.1"
        backends = type("B", (), {"cuda": type("C", (), {"matmul": matmul})})

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def synchronize():
                return None

        @staticmethod
        def randn(*a, **k):
            return _Arr()

    return T


class _FakeClockSampler:
    def __init__(self, interval=0.25):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        return {"sm_clock_min": 2400.0, "sm_clock_max": 2495.0, "sm_clock_mean": 2450.0,
                "power_max": 440.0, "util_mean": 99.0, "n_samples": 4}


def test_main_default_out_does_not_crash(tmp_path, monkeypatch):
    """R-18 的端到端负样本：`--quick` + 默认 --out 必须 rc=0 并产出一个可读 JSON。"""
    mod = _load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "torch", _FakeTorch)
    monkeypatch.setattr(mod, "ClockSampler", _FakeClockSampler)
    monkeypatch.setattr(mod, "bandwidth_sweep",
                        lambda sizes, dtype_bytes=4, reps=5: ([], {"copy_gbs": 906.7,
                                                                   "read_gbs": 897.1,
                                                                   "write_gbs": 935.8},
                                                              {"l2_resident_max_mb": 32}))
    monkeypatch.setattr(mod, "matmul_peak",
                        lambda sizes, dtypes, caliber_probe=True, reps=3: {
                            "torch.bfloat16": {"tflops": 171.7, "acc_caliber": "bf16",
                                               "precision_settings": {}, "method": "fake"}})

    class _P:
        stdout = "580.105, 3105, 450.00, 4, 16\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _P())
    monkeypatch.setattr(sys, "argv", ["machine-profile.py", "--quick"])
    mod.main()

    out = tmp_path / "machine-profile.json"
    assert out.exists(), "默认 --out 是裸文件名，必须能落盘（R-18）"
    doc = json.loads(out.read_text())
    assert doc["measured_dram"]["copy_gbs"] == 906.7
    assert doc["schema_version"] == 2
    assert "measured_matmul_tflops" in doc


def test_caliber_fields_are_written(tmp_path, monkeypatch):
    """R-09：口径字段（acc_caliber/precision_settings/method）与离散度必须落盘。"""
    fake_torch = _FakeTorch()

    class T:
        cuda = fake_torch.cuda
        bfloat16, float16, float32 = "bf16", "fp16", "fp32"
        __version__ = "2.12.1"
        randn = staticmethod(lambda *a, **k: _Arr())

        class backends:
            class cuda:
                class matmul:
                    allow_fp16_reduced_precision_reduction = True
                    fp32_precision = "none"
                    allow_tf32 = True

    mod = _load_module()
    monkeypatch.setattr(mod, "torch", T)

    monkeypatch.setattr(mod, "timeit", lambda fn, reps=3, warmup=2: (0.00664, 0.007))

    out = mod.matmul_peak([4096], [mod.torch.bfloat16], caliber_probe=True, reps=2)
    entry = out["bf16"]
    assert entry["acc_caliber"]
    assert "precision_settings" in entry
    assert "method" in entry
    assert entry["caliber_spread_pct"] is not None
    assert isinstance(entry["caliber_distinguishable"], bool)
    assert "caliber_variants_tflops" in entry


def test_variants_declare_settings_without_side_effects(monkeypatch):
    """`_variants()` 只声明设置，构造期不得有任何写入（R-50/D7）。

    旧实现返回 `(name, restore, settings)`，并且在**构造列表时**就调用三次
    `_set_precision` —— 三个副作用在第一次测量之前全部落地。这里用写日志守住"构造期零写"。
    """
    matmul = _RecordingMatmul()          # 现场：allow=True, allow_tf32=True, fp32="none"
    mod = _load_module()
    monkeypatch.setattr(mod, "torch", _probe_torch(matmul))

    variants = mod._variants()
    assert [v[0] for v in variants] == ["default", "reduced_precision_reduction_off",
                                       "fp32_ieee"]
    assert all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict)
               for v in variants), "只返回 (名字, 设置描述)，不再返回 restore 闭包"
    assert matmul.writes == [], "构造 _variants() 不得写任何精度设置"
    assert mod._precision_settings_snapshot() == {
        "allow_fp16_reduced_precision_reduction": True, "allow_tf32": True,
        "fp32_precision": "none", "float32_matmul_precision": None}

    variants[0][1]["injected"] = 1
    assert mod.CALIBER_VARIANTS[0][1] == {}, "_variants() 必须返回副本，不能被调用方改坏"

    # 应用 → 恢复：restore 必须把现场放回去
    restore = mod._set_precision(dict(variants[2][1]))
    assert matmul.fp32_precision == "ieee"
    restore()
    assert matmul.fp32_precision == "none"


def test_caliber_probe_applies_settings_before_each_measurement(monkeypatch):
    """R-50/D7 的探针：每个变体在自己的声明设置下被测，且"记录的 == 实测的"。

    做法：
      * 假 torch 的 `backends.cuda.matmul` 属性可读可写，并在**每次写入时留痕**；
      * `timeit` 替身在**被调用的那一刻**读一次后端状态（旧测试里的
        `lambda fn, reps=3, warmup=2: (…)` 看不见状态，守不住这条）。

    于是可以同时断言：
      ① 测量前该变体要改的字段已经写进后端（写日志里能查到）；
      ② 测量时刻的实际状态与产物里的 precision_settings **逐项相等**；
      ③ 三个变体的实际状态互不相同 —— 旧实现下三者在第一次测量前就全被写成
         fp32_ieee 的设置，这一条会红。
    """
    matmul = _RecordingMatmul()          # 现场：allow=True, allow_tf32=True, fp32="none"
    mod = _load_module()
    monkeypatch.setattr(mod, "torch", _probe_torch(matmul))

    seen = []                            # 每次"测量"时刻的 (实况快照, 当时的写日志)

    def recording_timeit(fn, reps=3, warmup=2):
        seen.append((mod._precision_settings_snapshot(), tuple(matmul.writes)))
        return 0.00664, 0.007

    monkeypatch.setattr(mod, "timeit", recording_timeit)

    entry = mod.matmul_peak([4096], [mod.torch.bfloat16], caliber_probe=True,
                            reps=2)["bf16"]

    names = [name for name, _ in mod._variants()]
    declared = dict(mod._variants())
    assert len(seen) == 4, "1 次尺寸扫描 + 3 次变体测量"
    probe = seen[1:]                     # seen[0] 是尺寸扫描那一次
    recorded = entry["caliber_variant_settings"]
    assert list(recorded) == names

    for name, (actual, writes_at_measure) in zip(names, probe):
        # ① 应用发生在测量之前
        for field, value in declared[name].items():
            assert (field, value) in writes_at_measure, \
                f"{name}: 测量前必须已把 {field}={value!r} 写进后端，实得日志 {writes_at_measure}"
        # ② 记录 == 实测（逐项）
        assert set(recorded[name]) == set(actual), f"{name}: 记录的字段集必须与实况一致"
        for field, value in actual.items():
            assert recorded[name][field] == value, \
                f"{name}.{field}: 产物记的是 {recorded[name][field]!r}，测量时实况是 {value!r}"

    measured = [snap for snap, _ in probe]
    # ③ 三者互不相同。现场是 (allow=True, allow_tf32=True, fp32="none")，而每个变体各动
    #    一个与现场不同的字段（default 不改、reduced_off 把 allow 翻成 False、fp32_ieee 把
    #    fp32 换成 "ieee"），所以三项本来就该两两不同。
    assert len({json.dumps(m, sort_keys=True) for m in measured}) == 3, \
        f"三个变体必须在互不相同的设置下被测，实得 {measured}"
    # 旧实现的症状：default 其实是在 fp32_ieee 的设置下测的
    assert measured[0]["fp32_precision"] == "none"
    assert measured[1]["allow_fp16_reduced_precision_reduction"] is False
    assert measured[2]["fp32_precision"] == "ieee"

    # ④ 头条的 precision_settings 就是胜出变体那次测量的实况
    assert entry["tflops"] == max(entry["caliber_variants_tflops"].values())
    assert entry["precision_settings"] == recorded[entry["tflops_protocol"]]

    # ⑤ 探针跑完恢复现场：最后三次写入就是 baseline
    assert matmul.allow_fp16_reduced_precision_reduction is True
    assert matmul.allow_tf32 is True
    assert matmul.fp32_precision == "none"
    assert matmul.writes[-3:] == [("allow_fp16_reduced_precision_reduction", True),
                                  ("allow_tf32", True), ("fp32_precision", "none")]


def test_headline_tflops_is_reproducible_from_a_variant(monkeypatch):
    """R-37 的负样本：产物必须内部自洽。

    真机产物实测的不自洽：fp16 的 `tflops` 是 167.4，而同一尺寸下的三个口径变体是
    145.5/163.2/163.3 —— headline **不等于任何一个变体**，且 `precision_settings` 记的是
    `fp32_precision="ieee"`（对应 fp32_ieee 变体），读者无法复现这个头条数字。
    """
    class T:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                pass

        bfloat16s = "bf16"
        bfloat16, float16, float32 = "bf16", "fp16", "fp32"
        __version__ = "2.12.1"

        class backends:
            class cuda:
                class matmul:
                    allow_fp16_reduced_precision_reduction = True
                    fp32_precision = "none"
                    allow_tf32 = True

        @staticmethod
        def randn(*a, **k):
            return _Arr()

    mod = _load_module()
    monkeypatch.setattr(mod, "torch", T)
    # 每次调用返回不同的耗时 → 三个变体给出三个不同的读数（模拟真实离散）
    seq = iter([(0.0200, 0.0), (0.0080, 0.0), (0.0180, 0.0), (0.0175, 0.0),
                (0.0090, 0.0), (0.0069, 0.0)])
    monkeypatch.setattr(mod, "timeit", lambda fn, reps=3, warmup=2: next(seq))

    entry = mod.matmul_peak([4096], [mod.torch.bfloat16], caliber_probe=True, reps=3)["bf16"]
    variants = entry["caliber_variants_tflops"]
    assert len(variants) == 3
    # ① 头条必须等于某个变体的读数（同尺寸、同方法、同 reps）
    assert entry["tflops"] == pytest.approx(max(variants.values())), \
        "headline 必须能对应到一次具体测量"
    assert entry["tflops_protocol"] in variants
    assert entry["tflops_is_variant_reading"] is True
    # ② 每个变体带自己的 precision_settings（_settings 不允许被丢弃）
    settings = entry["caliber_variant_settings"]
    assert set(settings) == set(variants), "_settings 不能被丢弃"
    assert settings["fp32_ieee"]["fp32_precision"] == "ieee"
    assert settings["default"]["fp32_precision"] == "none"
    assert settings["reduced_precision_reduction_off"][
        "allow_fp16_reduced_precision_reduction"] is False
    # ③ 头条的 precision_settings 就是胜出变体的设置，不是事后的快照
    assert entry["precision_settings"] == settings[entry["tflops_protocol"]]
    # ④ 离散度只在同一协议之间比较 → 探针尺寸必须被记录下来
    assert entry["caliber_probe_size"] == 4096


def test_headline_falls_back_to_sweep_without_caliber_probe(monkeypatch):
    """跳过口径探针时也要说清头条来自哪儿（不能留下一个来历不明的 tflops）。"""
    class T:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                pass

        bfloat16, float16, float32 = "bf16", "fp16", "fp32"
        __version__ = "2.12.1"
        backends = type("B", (), {"cuda": type("C", (), {"matmul": type(
            "M", (), {"allow_fp16_reduced_precision_reduction": True,
                      "fp32_precision": "none", "allow_tf32": True})})})

        @staticmethod
        def randn(*a, **k):
            return _Arr()

    mod = _load_module()
    monkeypatch.setattr(mod, "torch", T)
    monkeypatch.setattr(mod, "timeit", lambda fn, reps=3, warmup=2: (0.00664, 0.007))
    entry = mod.matmul_peak([4096], [mod.torch.bfloat16], caliber_probe=False)["bf16"]
    assert entry["caliber_variants_tflops"] == {}
    assert entry["tflops_protocol"] == "sweep-default"
    assert entry["tflops_is_variant_reading"] is False
    assert entry["tflops"] > 0 and entry["size"] == 4096
