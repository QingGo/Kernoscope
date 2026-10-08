/* SPDX-License-Identifier: Apache-2.0 */
/* Kernoscope 访存轨迹工具：设备与主机共享的记录结构。
 *
 * 一条记录 = 一条 warp 级访存指令的**一次执行**（不是每 lane 一条），
 * 由首 lane 收集全 warp 的地址后写入设备缓冲。 */
#ifndef KS_COMMON_H
#define KS_COMMON_H

#include <stdint.h>

#define KS_MAX_LANES 32

typedef struct {
    uint32_t insn_id;      /* 主机侧指派的静态指令编号（对应文件头部 legend） */
    uint32_t active_mask;  /* 本次执行的活跃 lane 位图（未活跃 lane 的地址无意义） */
    uint32_t cta_id;       /* blockIdx.x，用于统计实际被采样的 block 数 */
    uint16_t smid;         /* 该 warp 落在哪个 SM（L1 是 per-SM 的，建模需要） */
    uint8_t  lanes;        /* 活跃 lane 数 */
    uint8_t  size;         /* 每 lane 访问字节数（主机按 opcode 推断后传入） */
    uint64_t addrs[KS_MAX_LANES];
} ks_rec_t;

/* 注入调用的参数表刻意与 NVBit 官方示例同构（guard, const32, mref64, launch64, const64）：
 * 参数一多（实测 8 个）会出现 "CUDA_ERROR_INVALID_SOURCE device kernel image is invalid"，
 * 因为注入代码的调用约定放不下那么多实参。这里把其余参数收进一个设备侧结构，只传指针。 */
typedef struct {
    uint64_t recs;          /* ks_rec_t* 设备缓冲 */
    uint64_t counter;       /* 记录数计数器（设备原子） */
    int max_records;
    int block_stride;
    int debug_calls;        /* 1 = 只计数不做记录（排查"调用到底有没有发生"） */
    int _pad;
} ks_cfg_t;

#endif
