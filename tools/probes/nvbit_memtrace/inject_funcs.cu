/* SPDX-License-Identifier: Apache-2.0 */
/* Kernoscope 访存轨迹工具：注入到被测 kernel 里的设备侧代码。
 *
 * 设计要点（都是在官方 mem_trace 示例基础上按我们的格式改的）：
 *   1. 参数表与官方示例同构：guard, const32, mref64, launch64, const64 ——
 *      实测参数一多（8 个）会 CUDA_ERROR_INVALID_SOURCE，因为注入调用的
 *      调用约定放不下。多出来的配置收进 ks_cfg_t（只传一个指针）。
 *   2. 用 __shfl_sync 收集全 warp 的地址，只有**首 lane** 写设备缓冲 —— 否则
 *      每个 lane 一次原子操作，开销 32 倍。
 *   3. 记录 active_mask：被谓词关掉的 lane 也占 shfl 通道，其地址无意义，
 *      必须靠位图在主机侧剔除（否则垃圾地址会被算成 sector）。
 *   4. 按 blockIdx.x % stride 采样：全量轨迹会爆（一条记录 272B），采样比写在文件头。
 *   5. 不加 `if (!pred) return;` 这种提前返回：pred 只用来构建活跃掩码
 *      （官方 instr_count 也是这么处理的），提前 return 会让整个 warp 不记录。
 */

#include "ks_common.h"

#define KS_SIZE_SHIFT 24
#define KS_ID_MASK 0xFFFFFF

extern "C" __device__ __noinline__ void ks_record_mem(int pred, int insn_size_id, uint64_t addr,
                                                      uint64_t unused_launch_val,
                                                      uint64_t cfg_ptr) {
    (void)unused_launch_val;
    ks_cfg_t* cfg = (ks_cfg_t*)cfg_ptr;

    /* 诊断模式：只想确认"调用是否发生"时用，不做任何记录逻辑 */
    if (cfg->debug_calls) {
        if (pred) atomicAdd((unsigned int*)cfg->counter, 1u);
        return;
    }

    if (blockIdx.x % (unsigned)cfg->block_stride != 0) return;

    const unsigned active = __ballot_sync(__activemask(), 1);
    const int lane = (int)(threadIdx.x & 31u);

    /* 收集（所有活跃 lane 都必须参与 shfl，所以不能提前 return） */
    uint64_t a[KS_MAX_LANES];
#pragma unroll
    for (int i = 0; i < KS_MAX_LANES; i++) {
        a[i] = __shfl_sync(active, addr, i);
    }

    const int first = __ffs((int)active) - 1;
    if (lane != first) return;

    unsigned slot = atomicAdd((unsigned int*)cfg->counter, 1u);
    if ((int)slot >= cfg->max_records) return;

    ks_rec_t* recs = (ks_rec_t*)cfg->recs;
    ks_rec_t* r = &recs[slot];
    r->insn_id = (uint32_t)(insn_size_id & KS_ID_MASK);
    r->active_mask = active;
    r->cta_id = blockIdx.x;
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    r->smid = (uint16_t)smid;
    r->lanes = (uint8_t)__popc(active);
    r->size = (uint8_t)((unsigned)insn_size_id >> KS_SIZE_SHIFT);
#pragma unroll
    for (int i = 0; i < KS_MAX_LANES; i++) r->addrs[i] = a[i];
}
