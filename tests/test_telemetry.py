"""N7 授权遥测消费的契约：解析、映射、两个陷阱、三态探针。

这些测试要能在**功能被回退**时变红，所以每一条都钉在一个可观察的行为上，而不是钉在
"函数存在"。四个重点：

  ① `NaN`/`±Inf` 与"缺席"都不是 0（与 `nvml.SamplerResult.other_procs_probe` 同一套纪律）；
  ② `probe` 三态能把"没跑 / 读不到 / 读到了但没有这个字段"分开；
  ③ `DCGM_FI_PROF_DRAM_ACTIVE`（带宽占比）与 `DCGM_FI_DEV_MEM_COPY_UTIL`（时间占比）
     **在任何路径上都不会变成字节** —— 构造期拒绝 + 发布前复核两条防线；
  ④ 每个有读数的量都带端点署名与自己的（外部）真名，不挂 ncu 计数器名。
"""

from __future__ import annotations

import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from kernoscope.backends import telemetry as T
from kernoscope.provenance import COUNTER_NAMESPACES, Metric, Provenance, Section
from kernoscope.refusals import RefusalCode

# 设备级（不需要 profiling 权限）—— 单卡，带 HELP/TYPE、科学计数法、时间戳、标签。
ENDPOINT_DEV_ONLY = """\
# HELP DCGM_FI_DEV_POWER_USAGE Power usage (W)
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-0000"} 55.3
# TYPE DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION counter
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0"} 1.23456789e8 1750000000000
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 98
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 41.5
DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1980
DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 9.501e3
DCGM_FI_DEV_FB_USED{gpu="0"} 20480
DCGM_FI_DEV_FB_FREE{gpu="0"} 61000
DCGM_FI_DEV_XID_ERRORS{gpu="0"} 0
"""

# 设备级 + profiling 家族（端点里有 PROF_* ⇒ 只可能是管理员授权了这些字段）。
ENDPOINT_FULL = ENDPOINT_DEV_ONLY + """\
DCGM_FI_PROF_SM_OCCUPANCY{gpu="0"} 0.42
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0"} 0.71
DCGM_FI_PROF_PCIE_TX_BYTES{gpu="0"} 1.2e6
DCGM_FI_PROF_PCIE_RX_BYTES{gpu="0"} 3.4e6
"""

# 只留下两个"极易挂错名字"的陷阱量。
ENDPOINT_TRAPS_ONLY = """\
DCGM_FI_PROF_DRAM_ACTIVE{gpu="0"} 0.71
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 41.5
"""

MULTI_GPU = """\
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-0000"} 55.3
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-1111"} 7.05e1
"""


def _collect(tmp_path, text: str, **kw) -> T.TelemetryResult:
    path = tmp_path / "metrics.txt"
    path.write_text(text, encoding="utf-8")
    return T.collect_telemetry(str(path), **kw)


# ---------------------------------------------------------------------------
# ① Prometheus 文本解析
# ---------------------------------------------------------------------------
def test_parse_prometheus_text_reads_labels_exponents_and_ignores_comments():
    values = T.parse_prometheus_text(ENDPOINT_DEV_ONLY)
    # 值：普通小数、科学计数法、整数
    assert values[T.series_key("DCGM_FI_DEV_POWER_USAGE", {"gpu": "0", "UUID": "GPU-0000"})] == 55.3
    assert values['DCGM_FI_DEV_MEM_CLOCK{gpu="0"}'] == pytest.approx(9501.0)
    assert values['DCGM_FI_DEV_GPU_UTIL{gpu="0"}'] == 98.0
    # 规范键：标签按键名排序（大写在前）—— 同一个序列在两次抓取里必须得到同一个键
    assert 'DCGM_FI_DEV_POWER_USAGE{UUID="GPU-0000",gpu="0"}' in values
    # `# HELP` / `# TYPE` 是注释，不是数据
    assert not [k for k in values if k.startswith("#")]
    assert T.parse_prometheus_text_detailed(ENDPOINT_DEV_ONLY).comments_ignored == 3
    # 标签值里的逗号与转义引号要能原样往返
    tricky = T.parse_prometheus_text('m{a="x,y",b="q\\"z"} 1\n')
    assert tricky == {T.series_key("m", {"a": "x,y", "b": 'q"z'}): 1.0}
    # 多卡：同一字段多条序列不会被互相覆盖
    two = T.parse_prometheus_text(MULTI_GPU)
    assert two[T.series_key("DCGM_FI_DEV_POWER_USAGE", {"gpu": "0", "UUID": "GPU-0000"})] == 55.3
    assert two[T.series_key("DCGM_FI_DEV_POWER_USAGE", {"gpu": "1", "UUID": "GPU-1111"})] == 70.5


def test_parse_prometheus_text_accepts_optional_timestamps():
    parsed = T.parse_prometheus_text_detailed('x{a="1"} 5 1750000000000\n')
    assert parsed.values['x{a="1"}'] == 5.0
    assert parsed.newest_timestamp_ms == 1750000000000.0
    assert T.parse_prometheus_text("x 5\n") == {"x": 5.0}
    assert T.parse_prometheus_text("x 5\n")["x"] != 0.0


def test_nan_and_inf_are_skipped_with_a_reason_never_zeroed():
    parsed = T.parse_prometheus_text_detailed("a 1\nb NaN\nc +Inf\nd -Inf\ne -0.0\n")
    assert parsed.values == {"a": 1.0, "e": 0.0}
    assert parsed.skipped["b"] == "nan"
    assert parsed.skipped["c"] == "+inf"
    assert parsed.skipped["d"] == "-inf"
    for name in ("b", "c", "d"):
        assert name not in parsed.values, f"{name} 是 NaN/Inf，不许出现在读数里"
        assert parsed.values.get(name) != 0.0, f"{name} 不许被折算成 0"


def test_malformed_lines_are_recorded_not_silently_dropped():
    parsed = T.parse_prometheus_text_detailed('ok 1\n=bad 2\nx{a="1"} \nx{a=1} 3\n')
    assert parsed.values == {"ok": 1.0}
    assert parsed.skipped["line 2"] == "malformed_line"
    assert parsed.skipped['x{a="1"}'] == "missing_value"
    assert parsed.skipped["line 4"] == "malformed_labels"
    # 同一序列重复出现：后者覆盖前者，且记下来
    dup = T.parse_prometheus_text_detailed("x 1\nx 2\n")
    assert dup.values == {"x": 2.0} and dup.duplicates == ("x",)


# ---------------------------------------------------------------------------
# ② 映射：自己的名字 / 真单位 / 一行语义 / 端点署名
# ---------------------------------------------------------------------------
def test_collect_telemetry_maps_dev_fields_with_units_and_semantics(tmp_path):
    res = _collect(tmp_path, ENDPOINT_DEV_ONLY)
    assert res.probe == "ok" and res.ok
    rows = res.to_metrics()
    assert rows["meas__telemetry_dcgm_power_w"]["value"] == 55.3
    assert rows["meas__telemetry_dcgm_power_w"]["unit"] == "W"
    assert rows["meas__telemetry_dcgm_energy_total_mj"]["value"] == pytest.approx(1.23456789e8)
    assert rows["meas__telemetry_dcgm_energy_total_mj"]["unit"] == "mJ"
    assert rows["meas__telemetry_dcgm_fb_used_mib"]["unit"] == "MiB"
    assert rows["meas__telemetry_dcgm_sm_clock_mhz"]["unit"] == "MHz"
    assert rows["meas__telemetry_dcgm_gpu_util_pct"]["unit"] == "%"
    # 语义（一行）必须说明"这是时间占比，不是字节"
    assert "时间占比" in rows["meas__telemetry_dcgm_mem_copy_util_pct"]["note"]
    # 署名：每个量都带端点，且带端点的外部真名
    for metric in res.metrics:
        assert metric.source.startswith(str(tmp_path))
        assert metric.field.startswith("DCGM_FI_")
    # 名字全落在自有命名空间里，绝不挂 ncu 计数器名
    for name in res.names():
        assert name.startswith(("meas__", "ks__"))
        assert not name.startswith(COUNTER_NAMESPACES)
    assert res.series_total == len(T.parse_prometheus_text(ENDPOINT_DEV_ONLY))


def test_xid_errors_is_a_code_not_a_count(tmp_path):
    metric = _collect(tmp_path, ENDPOINT_DEV_ONLY).find("meas__telemetry_dcgm_last_xid_error_code")
    assert metric is not None
    assert metric.unit == "code"
    assert metric.value == 0.0                      # 实测到 0 是一条真读数
    assert "编号" in metric.semantics and "不是错误次数" in metric.semantics


def test_mapping_covers_the_required_dcgm_fields():
    required = {
        "DCGM_FI_DEV_POWER_USAGE": ("W", T.Quantity.POWER),
        "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": ("mJ", T.Quantity.ENERGY),
        "DCGM_FI_DEV_GPU_UTIL": ("%", T.Quantity.PERCENT),
        "DCGM_FI_DEV_MEM_COPY_UTIL": ("%", T.Quantity.PERCENT),
        "DCGM_FI_DEV_SM_CLOCK": ("MHz", T.Quantity.FREQUENCY),
        "DCGM_FI_DEV_MEM_CLOCK": ("MHz", T.Quantity.FREQUENCY),
        "DCGM_FI_DEV_FB_USED": ("MiB", T.Quantity.MEMORY_SIZE),
        "DCGM_FI_DEV_FB_FREE": ("MiB", T.Quantity.MEMORY_SIZE),
        "DCGM_FI_DEV_XID_ERRORS": ("code", T.Quantity.ERROR_CODE),
        "DCGM_FI_PROF_SM_OCCUPANCY": ("ratio", T.Quantity.RATIO),
        "DCGM_FI_PROF_DRAM_ACTIVE": ("ratio", T.Quantity.RATIO),
        "DCGM_FI_PROF_PCIE_TX_BYTES": ("B/s", T.Quantity.BYTES_PER_SECOND),
        "DCGM_FI_PROF_PCIE_RX_BYTES": ("B/s", T.Quantity.BYTES_PER_SECOND),
    }
    for fieldname, (unit, quantity) in required.items():
        spec = T.MAPPING[fieldname]
        assert spec.unit == unit and spec.quantity is quantity, fieldname
        assert spec.name.startswith(("meas__", "ks__"))
        assert spec.semantics and "\n" not in spec.semantics
    # requires_profiling 必须与 DCGM_FI_PROF_ 前缀一致（它决定"缺了意味着什么"）
    assert all(s.requires_profiling == s.field.startswith("DCGM_FI_PROF_")
               for s in T.DCGM_FIELDS)
    assert not [s for s in T.DCGM_FIELDS if s.quantity is T.Quantity.UNKNOWN]
    # 两个易错量的维度就是"比例"
    assert T.MAPPING["DCGM_FI_PROF_DRAM_ACTIVE"].quantity is T.Quantity.RATIO
    assert T.MAPPING["DCGM_FI_PROF_DRAM_UTIL_RATIO"].quantity is T.Quantity.RATIO
    assert T.MAPPING["DCGM_FI_PROF_DRAM_UTIL_RATIO"].alias_of == "DCGM_FI_PROF_DRAM_ACTIVE"
    # 用我们自己的名字点单时指向主字段，而不是老拼写
    assert T.FIELD_BY_NAME["meas__telemetry_dcgm_dram_active_ratio"] == "DCGM_FI_PROF_DRAM_ACTIVE"


# ---------------------------------------------------------------------------
# ③ 三态探针：缺席 / 读不到 / 实测 0 / NaN 是四件事
# ---------------------------------------------------------------------------
def test_missing_profiling_fields_are_recorded_while_probe_stays_ok(tmp_path):
    res = _collect(tmp_path, ENDPOINT_DEV_ONLY)
    assert res.probe == "ok"
    assert res.profiling_present is False
    # 一个物理量只报一条缺席：老拼写（别名）由主字段代表
    for fieldname in T.PROFILING_PRIMARY_FIELDS:
        assert res.missing[fieldname] == "absent", fieldname
    assert "DCGM_FI_PROF_DRAM_UTIL_RATIO" not in res.missing
    # 缺席 ≠ 失败：设备级量照常发布，0 也照常发布
    assert "meas__telemetry_dcgm_power_w" in res.names()
    assert res.find("meas__telemetry_dcgm_last_xid_error_code").value == 0.0
    assert "DCGM_FI_DEV_XID_ERRORS" not in res.missing
    assert res.profiling_absent == T.PROFILING_PRIMARY_FIELDS


def test_endpoint_with_profiling_fields_is_marked_as_the_authorized_channel(tmp_path):
    res = _collect(tmp_path, ENDPOINT_FULL)
    assert res.probe == "ok" and res.profiling_present is True
    ratio = res.find("meas__telemetry_dcgm_dram_active_ratio")
    assert ratio is not None and ratio.value == pytest.approx(0.71)
    assert ratio.unit == "ratio"
    assert res.missing.get("DCGM_FI_PROF_DRAM_ACTIVE") is None
    # 别名（老拼写）没出现时不该单独报一条缺席：同一个量的缺席由主字段代表
    assert res.missing.get("DCGM_FI_PROF_DRAM_UTIL_RATIO") is None


def test_nan_on_a_mapped_field_is_missing_with_a_reason_and_never_zero(tmp_path):
    text = ('DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1980\n'
            'DCGM_FI_PROF_DRAM_ACTIVE{gpu="0"} NaN\n')
    res = _collect(tmp_path, text)
    assert res.probe == "ok"
    assert res.missing["DCGM_FI_PROF_DRAM_ACTIVE"] == "nan"
    assert res.find("meas__telemetry_dcgm_dram_active_ratio") is None
    assert res.find("meas__telemetry_dcgm_sm_clock_mhz").value == 1980.0
    # 关键：它不是 0
    assert "meas__telemetry_dcgm_dram_active_ratio" not in res.to_metrics()


def test_absent_or_unreadable_source_fails_with_a_reason_and_never_raises(tmp_path):
    for source, expected in (
        (str(tmp_path / "nope.txt"), "io_error"),
        ("ftp://example.com/metrics", "unsupported_scheme"),
    ):
        res = T.collect_telemetry(source)          # 不许抛异常
        assert res.probe == "failed", source
        assert res.reason.startswith(expected), (source, res.reason)
        assert res.metrics == []
        for fieldname in T.PROFILING_PRIMARY_FIELDS:
            assert fieldname in res.missing, source
            # 缺席原因是"读不到"，不是"端点里没有"（两者必须能分开）
            assert res.missing[fieldname].startswith(expected), source


def test_network_timeout_and_unknown_errors_are_reported_not_raised(monkeypatch):
    def boom(*_a, **_k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(T, "_open_url", boom)
    res = T.collect_telemetry("http://telemetry.invalid/metrics")
    assert res.probe == "failed" and res.reason.startswith("network_error")

    def slow(*_a, **_k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(T, "_open_url", slow)
    assert T.collect_telemetry("https://telemetry.invalid/metrics", timeout_s=0.01).reason == "timeout"

    def weird(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(T, "_open_url", weird)
    res = T.collect_telemetry("https://telemetry.invalid/metrics")
    assert res.probe == "failed" and res.reason.startswith("internal_error")

    # 超时参数本身不合法也要变成"失败 + 原因"，不是异常
    assert T.collect_telemetry("https://telemetry.invalid/metrics", timeout_s=0).reason == "bad_timeout"


def test_garbage_source_is_failed_not_a_silent_empty_ok(tmp_path):
    path = tmp_path / "not-prometheus.txt"
    path.write_text("hello world\n", encoding="utf-8")
    res = T.collect_telemetry(str(path))
    assert res.probe == "failed" and res.reason == "no_prometheus_series"
    assert res.metrics == []
    # 读不出数据时，profil 量的缺席原因就是"读不出"，不是"端点里没有"
    assert res.missing
    assert set(res.missing.values()) == {"no_prometheus_series"}


def test_empty_source_is_not_run_not_failed():
    res = T.collect_telemetry("")
    assert res.probe == "not_run" and res.reason == "no_source"
    assert T.collect_telemetry("   ").probe == "not_run"
    assert res.metrics == []


def test_truncated_source_is_recorded_not_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_BYTES", 40)
    path = tmp_path / "big.txt"
    path.write_text('DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1980\n' + "x" * 200 + "\n", encoding="utf-8")
    res = T.collect_telemetry(str(path))
    assert res.probe == "ok"
    assert res.skipped[str(path)] == "truncated"


# ---------------------------------------------------------------------------
# ④ N7 的两个陷阱：比例/时间占比**永远**不会变成字节
# ---------------------------------------------------------------------------
def test_dram_active_ratio_and_mem_copy_time_fraction_never_become_bytes(tmp_path):
    res = _collect(tmp_path, ENDPOINT_TRAPS_ONLY)
    rows = res.to_metrics()
    ratio = rows["meas__telemetry_dcgm_dram_active_ratio"]
    assert ratio["value"] == pytest.approx(0.71)
    assert ratio["unit"] == "ratio"
    time_fraction = rows["meas__telemetry_dcgm_mem_copy_util_pct"]
    assert time_fraction["unit"] == "%"
    byte_units = {"B", "KB", "MB", "GB", "TB", "KIB", "MIB", "GIB", "TIB", "B/S", "KB/S",
                  "MB/S", "GB/S", "MIB/S", "GIB/S"}
    for name, row in rows.items():
        assert "byte" not in name.lower(), f"{name} 的名字像字节数（N7 事故）"
        assert row["unit"].strip().upper() not in byte_units, f"{name} 带了字节单位 {row['unit']}"
        assert row["provenance"] == "measured"
    assert T.byte_claim_violations(res.metrics) == []
    # 整条链路上不许出现 dram__bytes 之类的计数器名
    assert not [n for n in res.names() if n.startswith(COUNTER_NAMESPACES)]
    assert "dram__bytes" not in res.to_metrics()
    # 用这份端点拼一个**真的 section**（报告/决策包形状）：有值的行里不许出现字节单位
    section = Section("Telemetry")
    section.extend(res.to_provenance_metrics())
    section.extend(res.unavailable_metrics())
    section.validate()
    ratio_names = [m.name for m in section.metrics if m.provenance is not Provenance.UNAVAILABLE]
    assert "meas__telemetry_dcgm_dram_active_ratio" in ratio_names     # 断言不是空转
    for metric in section.metrics:
        if metric.provenance is Provenance.UNAVAILABLE:
            assert metric.value is None
            continue
        assert metric.unit.strip().upper() not in byte_units, f"{metric.name} 带了字节单位"
        assert "byte" not in metric.name.lower(), f"{metric.name} 像字节数"


def test_byte_rate_is_allowed_only_for_the_byte_rate_field(tmp_path):
    res = _collect(tmp_path, ENDPOINT_FULL)
    tx = res.find("meas__telemetry_dcgm_pcie_tx_bytes_per_s")
    assert tx is not None and tx.unit == "B/s"
    # 明确是速率，不是窗口总量
    assert "速率" in tx.semantics
    byteish = [n for n in res.names() if "byte" in n.lower()]
    assert byteish == [tx.name, "meas__telemetry_dcgm_pcie_rx_bytes_per_s"]
    assert T.byte_claim_violations(res.metrics) == []


def test_old_dram_util_ratio_spelling_publishes_under_the_same_ratio_name(tmp_path):
    """`DCGM_FI_PROF_DRAM_UTIL_RATIO` 是同一个量的老拼写：发布名与单位都必须一致。"""
    res = _collect(tmp_path, 'DCGM_FI_PROF_DRAM_UTIL_RATIO{gpu="0"} 0.55\n')
    assert res.profiling_present is True
    metric = res.find("meas__telemetry_dcgm_dram_active_ratio")
    assert metric is not None
    assert metric.value == pytest.approx(0.55) and metric.unit == "ratio"
    assert "DCGM_FI_PROF_DRAM_UTIL_RATIO" in metric.source
    assert "byte" not in metric.name.lower()
    # 同一个量只代表一次：两个拼写都不进 missing
    assert res.missing.get("DCGM_FI_PROF_DRAM_ACTIVE") is None
    assert res.missing.get("DCGM_FI_PROF_DRAM_UTIL_RATIO") is None


def test_ratio_field_declared_with_byte_unit_or_byte_name_is_rejected():
    with pytest.raises(T.TelemetryTrapError):
        T.FieldSpec("DCGM_FI_PROF_DRAM_ACTIVE", "meas__telemetry_dram_active_bytes", "B",
                    T.Quantity.RATIO, "x", requires_profiling=True)
    with pytest.raises(T.TelemetryTrapError):
        T.FieldSpec("DCGM_FI_PROF_DRAM_ACTIVE", "meas__telemetry_dram_active_ratio", "B/s",
                    T.Quantity.RATIO, "x", requires_profiling=True)
    with pytest.raises(T.TelemetryTrapError):
        T.TelemetryMetric("meas__telemetry_dram_active_bytes", 0.7, "ratio", "x", "endpoint",
                          T.Quantity.RATIO)
    # 单位与维度不符（写错单位 == 写错语义）
    with pytest.raises(T.TelemetryTrapError):
        T.TelemetryMetric("meas__telemetry_mem_copy_util_pct", 41.5, "ratio", "x", "endpoint",
                          T.Quantity.PERCENT)
    # ncu 计数器名一律不许（N3/R2）
    with pytest.raises(T.TelemetryTrapError):
        T.FieldSpec("DCGM_FI_PROF_PCIE_TX_BYTES", "dram__bytes.sum", "B/s",
                    T.Quantity.BYTES_PER_SECOND, "x", requires_profiling=True)
    # requires_profiling 与字段前缀不一致也拒绝
    with pytest.raises(T.TelemetryTrapError):
        T.FieldSpec("DCGM_FI_DEV_POWER_USAGE", "meas__telemetry_power_w", "W",
                    T.Quantity.POWER, "x", requires_profiling=True)
    # 没有端点署名的读数不许存在
    with pytest.raises(T.TelemetryTrapError):
        T.TelemetryMetric("meas__telemetry_power_w", 55.3, "W", "x", "", T.Quantity.POWER)


def test_publish_audit_catches_a_forged_ratio_in_bytes_metric(tmp_path):
    """构造期已经拦过一次；发布前复核也必须能拦住"别处手搓的一行"。"""
    res = _collect(tmp_path, ENDPOINT_TRAPS_ONLY)
    forged = object.__new__(T.TelemetryMetric)          # 绕过构造期校验
    for key, value in {
        "name": "meas__telemetry_dcgm_dram_active_bytes",
        "value": 0.71, "unit": "B", "semantics": "x", "source": "endpoint",
        "quantity": T.Quantity.RATIO, "field": "DCGM_FI_PROF_DRAM_ACTIVE",
    }.items():
        object.__setattr__(forged, key, value)
    violations = T.byte_claim_violations([forged])
    assert violations and "N7" in violations[0]
    with pytest.raises(T.TelemetryTrapError):
        T.TelemetryResult(metrics=[forged], probe="ok", source="endpoint").to_metrics()
    # 正常结果永远没有违规
    assert T.byte_claim_violations(res.metrics) == []


def test_module_has_no_ratio_times_peak_to_bytes_path():
    """比例 × 峰值 ⇒ 字节 这条路必须**不存在**（不是"不推荐"）。"""
    source = Path(T.__file__).read_text(encoding="utf-8")
    for token in ("UNIT_TO_BYTES", "to_bytes", "peak_gbs", "peak_gb_s", "* peak", "*peak"):
        assert token not in source, f"遥测模块里出现了 {token!r}：这条换算路径不许存在"
    assert not hasattr(T, "to_bytes")
    assert callable(T.byte_claim_violations)


# ---------------------------------------------------------------------------
# ⑤ 其它：HTTP 端点、多卡、wanted、语义未知、出处合规
# ---------------------------------------------------------------------------
class _MetricsHandler(BaseHTTPRequestHandler):
    body = ENDPOINT_FULL.encode("utf-8")

    def do_GET(self) -> None:                            # noqa: N802 - http.server 的接口
        if self.path != "/metrics":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *_args) -> None:               # 测试不要往 stderr 写日志
        return


def test_http_endpoint_is_read_with_the_given_timeout():
    server = ThreadingHTTPServer(("", 0), _MetricsHandler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        url = f"http://localhost:{port}/metrics"
        res = T.collect_telemetry(url, timeout_s=5.0)
        assert res.probe == "ok", res.reason
        assert res.source == url
        power = res.find("meas__telemetry_dcgm_power_w")
        assert power.value == 55.3
        assert power.source == (
            f"{url} [{T.series_key('DCGM_FI_DEV_POWER_USAGE', {'gpu': '0', 'UUID': 'GPU-0000'})}]")
        assert res.newest_timestamp_ms == 1750000000000.0
        bad = T.collect_telemetry(f"http://localhost:{port}/nope")
        assert bad.probe == "failed" and bad.reason == "http_error:404"
    finally:
        server.shutdown()
        server.server_close()


def test_multiple_series_do_not_collide_and_sources_carry_the_endpoint(tmp_path):
    res = _collect(tmp_path, MULTI_GPU)
    assert sorted(res.names()) == ["meas__telemetry_dcgm_power_w__gpu_0",
                                   "meas__telemetry_dcgm_power_w__gpu_1"]
    rows = res.to_metrics()
    assert rows["meas__telemetry_dcgm_power_w__gpu_0"]["value"] == 55.3
    assert rows["meas__telemetry_dcgm_power_w__gpu_1"]["value"] == 70.5
    for metric in res.metrics:
        assert metric.source.startswith(str(tmp_path))
        assert "DCGM_FI_DEV_POWER_USAGE" in metric.source
    # 多卡时**不猜**单值（不求和、不取最大）
    values = T.parse_prometheus_text(MULTI_GPU)
    assert T.single_value_for_field(values, "DCGM_FI_DEV_POWER_USAGE") is None
    assert len(T.values_for_field(values, "DCGM_FI_DEV_POWER_USAGE")) == 2


def test_wanted_fields_are_tracked_and_unknown_wanted_is_recorded(tmp_path):
    text = 'DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1980\n'
    res = _collect(tmp_path, text, wanted=("DCGM_FI_DEV_POWER_USAGE",
                                           "meas__telemetry_dcgm_fb_used_mib",
                                           "DCGM_FI_DEV_NOPE"))
    assert res.missing["DCGM_FI_DEV_POWER_USAGE"] == "absent"
    # 用我们发布的名字要，也追讨同一个外部字段（两种名字都认）
    assert res.missing["DCGM_FI_DEV_FB_USED"] == "absent"
    assert res.missing["DCGM_FI_DEV_NOPE"] == "unknown_field"
    # 没点名要、但端点里有的字段照样发布
    assert "meas__telemetry_dcgm_sm_clock_mhz" in res.names()
    # 用我们的名字点单，端点给的是老拼写：同一个量，算覆盖到，不报缺席
    other = _collect(tmp_path, 'DCGM_FI_PROF_DRAM_UTIL_RATIO{gpu="0"} 0.5\n',
                     wanted=("meas__telemetry_dcgm_dram_active_ratio",))
    assert other.missing.get("DCGM_FI_PROF_DRAM_ACTIVE") is None
    assert other.missing.get("DCGM_FI_PROF_DRAM_UTIL_RATIO") is None


def test_unmapped_dcgm_field_is_semantics_unknown_not_guessed(tmp_path):
    res = _collect(tmp_path, 'DCGM_FI_DEV_SYNTHETIC_UNMAPPED{gpu="0"} 7\n')
    assert res.probe == "ok"
    assert res.missing["DCGM_FI_DEV_SYNTHETIC_UNMAPPED"] == "semantics_unknown"
    assert not [n for n in res.names() if "SYNTHETIC" in n.upper()]
    assert "无法确认" in T.reason_text_for("semantics_unknown")
    # 点名要一个"真的在端点里、但我们不认识"的字段：原因是语义未知，不是"你点错了名字"
    res2 = _collect(tmp_path, 'DCGM_FI_DEV_SYNTHETIC_UNMAPPED{gpu="0"} 7\n',
                    wanted=("DCGM_FI_DEV_SYNTHETIC_UNMAPPED",))
    assert res2.missing["DCGM_FI_DEV_SYNTHETIC_UNMAPPED"] == "semantics_unknown"


def test_rows_pass_the_project_provenance_rules(tmp_path):
    res = _collect(tmp_path, ENDPOINT_FULL)
    for row in res.to_metrics().values():
        assert row["provenance"] == "measured"
        assert row["name"].startswith(("meas__", "ks__"))
        Metric(row["name"], row["value"], Provenance(row["provenance"]),
               unit=row["unit"], source=row["source"])       # 不抛异常即合规
    for metric in res.to_provenance_metrics():
        metric.validate()
        assert metric.provenance is Provenance.MEASURED
    # 缺席字段 → 带结构化拒绝的 n/a 行，且复用既有 code 闭集
    unavailable = _collect(tmp_path, ENDPOINT_DEV_ONLY).unavailable_metrics()
    assert unavailable
    for metric in unavailable:
        assert metric.value is None
        assert metric.refusal is not None and metric.refusal.code in set(RefusalCode)
        metric.validate()
    names = [m.name for m in unavailable]
    assert len(names) == len(set(names)), "同一个发布名不许出现两条 n/a"


def test_reasons_map_onto_the_existing_refusal_closed_set():
    for reason in ("absent", "semantics_unknown", "unknown_field", "nan", "network_error:refused",
                   "timeout", "http_error:503", "no_prometheus_series", "io_error:FileNotFoundError"):
        code = T.reason_code_for(reason)
        assert isinstance(code, RefusalCode)
        assert not code.value.startswith("telemetry_"), "本轮不新增拒绝码（复用既有闭集）"
        assert T.reason_text_for(reason)
    assert T.reason_code_for("absent") is RefusalCode.METRIC_ABSENT
    assert T.reason_code_for("network_error:refused") is RefusalCode.ENV_READING_UNKNOWN
    assert T.reason_code_for("nan") is RefusalCode.ENV_READING_UNKNOWN
    # 处方必须指向"向平台要一个只读端点"（N7 的择中形态）
    assert "只读" in T.reason_remedy_for("no_source").action
