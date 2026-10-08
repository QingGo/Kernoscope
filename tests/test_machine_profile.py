"""标定脚本（tools/machine-profile.py）的回归测试。

守两条需求方实测出来的缺陷：
  R-18/B7-a  默认 `--out machine-profile.json`（裸文件名）不得崩 —— 它是 README 与
             doctor 共同推荐的第一步，真机上 rc=1 并抛 `FileNotFoundError: ''`
  R-09/B9    标定产物必须记录累加口径、precision 设置、方法，以及**口径离散度**
             （同一台机器相隔一天，两个脚本对"口径可不可分辨"给出相反判断）
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


def test_variants_restore_precision_settings(monkeypatch):
    """口径探针跑完必须恢复现场（否则会把用户的 torch 设置永久改掉）。"""
    class T:
        class cuda:
            @staticmethod
            def is_available():
                return True

    class M:
        allow_fp16_reduced_precision_reduction = True
        fp32_precision = "none"
        allow_tf32 = True

    T.backends = type("B", (), {"cuda": type("C", (), {"matmul": M})})
    mod = _load_module()
    monkeypatch.setattr(mod, "torch", T)
    variants = mod._variants()
    names = [v[0] for v in variants]
    assert names == ["default", "reduced_precision_reduction_off", "fp32_ieee"]
    for _name, restore, _settings in variants:
        restore()
    assert M.allow_fp16_reduced_precision_reduction is True
    assert M.fp32_precision == "none"
