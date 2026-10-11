"""量从哪来、名字能挂到哪 —— N3「同源才写 ncu 名」的**类型系统**那一半。

背景（`NON-GOALS.md` 的 N3）：本产品唯一不可让渡的资产是**可审计性**。
一个挂在 ncu 名字下、语义却不成立的数字，比没有数字更坏 ——
因为读的人会把它当成硬件计数器语义去决策。过去这条边界是**散文政策**：
"同源才写 ncu 名"由人判断，只有 `NCU_ROWS` 白名单 + 单位核对是机检的。
现在把"这个量是什么来源"也变成类型，于是"名字与来源相容"可以被机器判定，
而不是被同一个人在同一次提交里重新解释一遍。

四档（不要把相邻两档当成一回事 —— 它们之间差的正是"能不能写 ncu 名"）：

``documented``
    NVIDIA 的公开文档/公开头文件里**明确定义了这个量以及它的单位**。
    最典型的例子是 ``NVML_FI_DEV_*`` 家族（``nvmlDeviceGetFieldValues`` 的字段 id）：
    它们就在公开的 ``nvml.h`` 与 API 参考里，语义与单位都有出处。
    只有这一档的量允许写进 ncu 的名字空间（且必须单位相符）。

``documented-adjacent``
    有**公开符号名**，但没有任何语义承诺。典型形状是"头文件里没有、驱动却接受"的 enum 值：
    我们能看到调用成功、能拿到数字，但厂商没有承诺这个数字是什么、单位是什么。
    名字眼熟不等于语义成立 ⇒ 只许进 ``ks__``/``meas__``。

``undocumented-observed``
    纯实证观察：我们调用它、观察返回值/不透明 blob，能复现，但没有任何公开定义。
    它可能看起来与某个 ncu 指标"一样"，但"看起来一样"不是判据 ⇒ 只许进自有命名空间。

``vendor-tool-output``
    另一个工具的输出格式（例如它自己写下的字段/文件结构）。能读、能引用，
    但那是**那个工具的产物**，不是我们的同源量 ⇒ 引用时要写清来源，不借 ncu 名字。

发布规则的机检形式就是 :func:`allows_ncu_name`：**名字与来源相容**，且单位相符，
才允许用 ncu 名 + ncu 单位；否则必须留在 ``ks__``/``est__``/``meas__`` 下
（或者被丢弃，并带上机器可读的原因）。
"""

from __future__ import annotations

from enum import Enum


class SourceClass(str, Enum):
    """量的来源分类（`str` 子类，便于直接进 JSON / 命令行）。"""

    #: NVIDIA 文档/公开头文件定义了这个量与单位（例：整个 ``NVML_FI_DEV_*`` 家族在 `nvml.h`）。
    DOCUMENTED = "documented"
    #: 公开符号名、无语义承诺（典型：头文件没有、驱动却接受的 enum 值）。
    DOCUMENTED_ADJACENT = "documented-adjacent"
    #: 实证观察（不透明 blob 一类）：能复现，但没有公开定义。
    UNDOCUMENTED_OBSERVED = "undocumented-observed"
    #: 别的工具的输出格式：那是它的产物，不是我们的同源量。
    VENDOR_TOOL_OUTPUT = "vendor-tool-output"

    @property
    def allows_ncu_namespace(self) -> bool:
        """这一档的来源是否**有可能**挂 ncu 名（还要单位相符，见 :func:`allows_ncu_name`）。"""
        return self is SourceClass.DOCUMENTED

    @property
    def justification(self) -> str:
        """这一档为什么如此 —— 写进机器可读署名，让人不必读源码就能复核判定。"""
        return _JUSTIFICATION[self]


#: 每档的判定依据（散文，进机器可读署名；**不是**政策文本的新真源，真源是本模块 docstring）。
_JUSTIFICATION: dict[SourceClass, str] = {
    SourceClass.DOCUMENTED:
        "NVIDIA 公开文档/公开头文件定义了这个量及其单位（例：NVML_FI_DEV_* 在公开的 nvml.h 里）",
    SourceClass.DOCUMENTED_ADJACENT:
        "有公开符号名但无语义承诺（典型：头文件里没有、驱动却接受的 enum 值）",
    SourceClass.UNDOCUMENTED_OBSERVED:
        "实证观察，无公开定义（例：不透明 blob）—— 看起来像不等于同源",
    SourceClass.VENDOR_TOOL_OUTPUT:
        "另一个工具的输出格式 —— 能引用，但不借 ncu 的名字",
}


def allows_ncu_name(cls: "SourceClass | str", unit_ok: bool) -> bool:
    """这个来源 + 单位是否允许写进 **ncu 的名字空间**。

    Args:
        cls: 量的来源分类（:class:`SourceClass` 或它的字符串值）。
        unit_ok: 单位是否与 ncu 的语义/单位相符（换算系数成立）。**单位是独立的一道门**：
            来源再权威，单位对不上也不许写（V-9：宁可不写，也不写一个乘错的数）。

    规则（N3）：**只有 `documented` 且单位相符**才允许。
    `documented-adjacent` / `undocumented-observed` / `vendor-tool-output`
    即使"看起来一样"，也只许进 `ks__`/`est__`/`meas__`，或者被丢弃并记原因。
    """
    return SourceClass(cls).allows_ncu_namespace and bool(unit_ok)


def describe(cls: "SourceClass | str") -> str:
    """一行可进署名的解释：`documented` / `documented — NVIDIA 公开文档…`。"""
    resolved = SourceClass(cls)
    return f"{resolved.value} — {resolved.justification}"
