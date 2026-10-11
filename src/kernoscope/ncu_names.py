"""ncu 名字的**发现键**与兼容导出（D4 ①②）。

背景（用户的取舍）：我们的硬规则是"非 exact 的值不许挂在 ncu 计数器名下"，这条规则保护
的是**证据身份**（读的人不会把代理值当计数器读数）。但它不该变成**使用门槛**：一个带着
`--metrics dram__bytes.sum` 来的用户，不该得到一句"指标不存在"。

所以采取的做法是"改形式、不改身份"：

  ① **名字是发现键**：`--metrics dram__bytes.sum` 会被解析成"你要的是 DRAM 字节"，
     我们给出**替代口径的真实指标名**（`est__dram_bytes_total`），并在报告里写清
     "ncu 的 X 在这台机器上拿不到；这是替代口径 + 出处 + 精度"。原来的规则一秒没松：
     数值仍然只出现在 `est__*` / `ks__*` 名下。
  ② **兼容导出**：`--export-name-map <path>` 写一份"ncu 名 → 我们给什么（或为什么给不出）"
     的对照表，供下游脚本/看板做翻译。它是**声明**，不是数据替换。

这张表是代码（`NCU_DISCOVERY_MAP`），文档只是它的视图 —— 新增一条映射就同时被
`--metrics` 解析、`--strict` 的错误信息、兼容导出与测试看到。
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Iterable

# ncu 指标名 → 替代口径说明。
#   ours:    我们给出的指标名（空字符串 = 这条我们确实给不出任何替代）
#   caliber: 替代口径的一句话（含精度/前提），随映射一起输出
#   kind:    "substitute"（有替代）| "same_source"（与 ncu 同源，我们能原样给）| "none"
@dataclass(frozen=True)
class Discovery:
    ncu_name: str
    ours: str
    caliber: str
    kind: str = "substitute"


NCU_DISCOVERY_MAP: tuple[Discovery, ...] = (
    Discovery("dram__bytes.sum", "est__dram_bytes_total",
              "零依赖的 NVML 代理口径（显存忙碌 × 本机实测峰值，±10% 带内；"
              "有 NVBit 时用轨迹口径 est__dram_bytes_per_launch_trace，精度高 4 个数量级）"),
    Discovery("dram__bytes_read.sum", "est__dram_bytes_read_per_launch",
              "NVML 代理口径的读字节（需要 NVBit 轨迹才能分量拆分）"),
    Discovery("dram__bytes_write.sum", "est__dram_bytes_write_per_launch",
              "NVML 代理口径的写字节（同上）"),
    Discovery("dram__throughput.avg.pct_of_peak_sustained_elapsed",
              "est__dram_throughput_pct_of_peak",
              "达成带宽 ÷ **本机实测**峰值（分母来自 machine-profile，不是规格表）"),
    Discovery("gpu__time_duration.sum", "gpu__time_duration.sum",
              "同源：CUPTI activity 时间戳（不需要计数器权限）", kind="same_source"),
    Discovery("launch__registers_per_thread", "launch__registers_per_thread",
              "同源：cubin/LaunchStats 静态属性", kind="same_source"),
    Discovery("launch__grid_size", "launch__grid_size",
              "同源：CUPTI activity 的 grid 维度", kind="same_source"),
    Discovery("l1tex__t_sector_hit_rate.pct", "est__l2_hit_rate_pct",
              "**模型上界**（访存轨迹的 L1/L2 模型），不是计数器读数；"
              "没有轨迹时给不出"),
    Discovery("lts__t_sector_hit_rate.pct", "est__l2_hit_rate_pct",
              "同上（L2 侧）"),
    Discovery("sm__throughput.avg.pct_of_peak_sustained_elapsed",
              "est__issue_slot_utilization_pct",
              "**上界口径**：发射槽利用率上界 + 瓶颈方向，不是 ncu 的多管道最大值；"
              "需要 --ks-instr-hist"),
    Discovery("smsp__inst_executed.sum", "ks__dynamic_instructions",
              "NVBit opcode_hist 的动态指令数（需要 --ks-instr-hist）"),
    Discovery("sm__warps_active.avg.pct_of_peak_sustained_active", "meas__achieved_occupancy_pct",
              "区间重叠法探针重建的**实际达到** occupancy（需要 occupancy 探针）"),
    Discovery("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "",
              "动态 Tensor Core 利用率在这台机器上拿不到（需要计数器）；"
              "只有静态 SASS 里 tensor 指令的占比 ks__static_sass_share_tensor",
              kind="none"),
    Discovery("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "",
              "warp stall 周期细分原理上拿不到（需要 PC 采样/计数器）；"
              "能给的只有 est__stall_direction 的方向性判断", kind="none"),
)

_BY_NCU = {d.ncu_name: d for d in NCU_DISCOVERY_MAP}

# 规范化：把 ncu 的各种后缀（`.sum`/`.avg`/`.pct`/`.per_second`/`.pct_of_peak_*`）逐层剥掉，
# 于是 `dram__bytes`、`dram__bytes.sum`、`dram__bytes.sum.per_second` 都能查到同一条映射。
# 这是一张**别名索引**（键值对唯一性由测试守），不是模糊匹配：规范化后还查不到就是查不到。
_STRIP = (".per_second", ".pct_of_peak_sustained_elapsed", ".pct_of_peak_sustained_active",
          ".pct_of_peak_sustained", ".sum", ".avg", ".max", ".min", ".pct")


def _normalize(name: str) -> str:
    out = name.strip()
    changed = True
    while changed:
        changed = False
        for suf in _STRIP:
            if out.endswith(suf) and len(out) > len(suf):
                out = out[: -len(suf)]
                changed = True
    return out


_ALIAS: dict[str, Discovery] = {}
_ALIAS_CONFLICTS: list[str] = []
for _d in NCU_DISCOVERY_MAP:
    _key = _normalize(_d.ncu_name)
    if _key in _ALIAS and (_ALIAS[_key].ours != _d.ours or _ALIAS[_key].kind != _d.kind):
        _ALIAS_CONFLICTS.append(_key)      # 同一个规范化键映射到不同的东西 => 表里有冲突
    else:
        _ALIAS.setdefault(_key, _d)


def aliases(ncu_name: str) -> list[str]:
    """一个请求名可能的等价写法：原名 + 规范化形式（去后缀）。"""
    name = ncu_name.strip()
    norm = _normalize(name)
    return [name] if norm == name else [name, norm]


def resolve(ncu_name: str) -> Discovery | None:
    """把用户写的 ncu 名字查成一条替代口径说明；查不到返回 None（**不猜**）。

    先精确查表（保留 `same_source` 那几条的原名），再查规范化别名索引。
    """
    name = ncu_name.strip()
    d = _BY_NCU.get(name)
    if d is not None:
        return d
    return _ALIAS.get(_normalize(name))


def resolve_requested(names: Iterable[str]) -> tuple[dict[str, Discovery], list[str]]:
    """解析 `--metrics` 请求：-> (命中表, 完全查不到的名字)。

    命中表是"请求名 → 说明"；`same_source` 的那种说明里 `ours` 与请求名相同（我们真给得出）。
    """
    hit: dict[str, Discovery] = {}
    miss: list[str] = []
    for n in names:
        d = resolve(n)
        if d is None:
            miss.append(n)
        else:
            hit[n] = d
    return hit, miss


def explain(ncu_name: str) -> str:
    """一句人话：这个名字我们给什么、或者为什么给不出（`--strict` 的错误信息用它）。"""
    d = resolve(ncu_name)
    if d is None:
        return f"{ncu_name}：不在已知映射里，也不存在（用 --export-name-map 看全部映射）"
    if d.kind == "none" or not d.ours:
        return f"{ncu_name}：这台机器上给不出（{d.caliber}）"
    if d.kind == "same_source":
        return f"{ncu_name}：同源可给（{d.ours}；{d.caliber}）"
    return (f"{ncu_name}：ncu 计数器读数拿不到，替代口径是 {d.ours}（{d.caliber}）——"
            f"数值只出现在 {d.ours} 名下，不会挂在 {ncu_name} 下面")


def render_name_map(fmt: str = "json") -> str:
    """兼容导出：`json` 或 `csv`（ncu 名 → 我们给什么 → 口径 → 类型）。"""
    rows = [{"ncu_name": d.ncu_name, "ours": d.ours, "kind": d.kind, "caliber": d.caliber}
            for d in NCU_DISCOVERY_MAP]
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["ncu_name", "ours", "kind", "caliber"])
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()
    return json.dumps({"schema": "kernoscope-name-map-v1", "map": rows},
                      ensure_ascii=False, indent=2) + "\n"
