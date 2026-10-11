/* SPDX-License-Identifier: Apache-2.0 */
/* Kernoscope「达到的 occupancy」NVBit 工具：设备与主机共享的记录结构。
 *
 * 与 occupancy_probe.cu 的区别：探针要**把几行代码抄进被测 kernel**（见
 * tools/probes/occupancy_probe.cu 与它的 README §6.1），于是闭源 kernel（cuBLAS、
 * Inductor 生成、库 kernel）永远测不了。本工具把同样两个时间戳读**在 SASS 层注入**，
 * 被测 kernel **零源码改动**。
 *
 * 一条记录 = 一个 CTA（不是每 warp 一条），按 CTA 平铺编号**直接索引** —— 不用
 * atomicAdd 追加缓冲，于是「到底记录到几个 CTA」是精确的：
 *   分子 = g1_ok 为 1 的行数；分母 = 本次 launch 的 grid_blocks。
 * 追加式缓冲一旦溢出就只能报"截断"，而覆盖率的分子恰恰是判断这份数据能不能用的依据。
 */
#ifndef KS_OCC_COMMON_H
#define KS_OCC_COMMON_H

#include <stdint.h>

/* 每个 CTA 一条。字段顺序刻意让两个 uint64_t 落在 8 字节边界上：设备侧对 g1 做
 * atomicMax.u64 要求 8 字节对齐。 */
typedef struct {
    uint32_t smid;    /* 这个 CTA 落在哪个 SM（%smid） */
    uint32_t g0_ok;   /* 1 = 有线程写过 g0（入口钩子命中过这个 CTA） */
    uint64_t g0;      /* CTA 入口时间戳（%globaltimer，ns，跨 SM 同源） */
    uint32_t g1_ok;   /* 1 = 有 warp 走到过「最后一条静态指令」 */
    uint32_t _pad;
    uint64_t g1;      /* CTA 出口时间戳；取**最后一个**走到收尾处的 warp（atomicMax） */
} ks_occ_rec_t;

/* 注入调用的参数表刻意与 NVBit 官方示例同构（见 nvbit_memtrace/ks_common.h 的原因：
 * 参数一多会 CUDA_ERROR_INVALID_SOURCE）。这里只需要一个指针。 */
typedef struct {
    uint64_t recs;        /* ks_occ_rec_t*，长度 max_ctas */
    uint64_t g0_done;     /* uint32_t*，每 CTA 一次的入口门（atomicExch） */
    uint64_t calls;       /* uint32_t*，诊断计数器（KS_OCC_DEBUG_CALLS=1 时用） */
    uint64_t salt;        /* 值依赖用的盐：时间戳读挂在这条依赖链后面 */
    uint32_t max_ctas;    /* 缓冲能容纳的 CTA 数（平铺 blockIdx 的上限） */
    uint32_t debug_calls; /* 1 = 只数"注入调用发生了多少次"，不写记录 */
} ks_occ_cfg_t;

#endif
