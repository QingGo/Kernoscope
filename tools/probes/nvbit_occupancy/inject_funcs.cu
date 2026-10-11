/* SPDX-License-Identifier: Apache-2.0 */
/* Kernoscope「达到的 occupancy」NVBit 工具：注入到被测 kernel 里的设备侧代码。
 *
 * 两个钩子（都由主机侧在 SASS 上插入调用，被测 kernel 零源码改动）：
 *
 *   入口 ks_occ_enter —— 插在 kernel **偏置最小**的那条指令之前（= 第一条静态指令）。
 *                       每个 CTA **只记一次**：8.9 里第一个到达的线程用 atomicExch 抢门，
 *                       谁抢到谁读 %globaltimer + %smid 写进本 CTA 的记录槽。
 *   出口 ks_occ_exit  —— 插在 kernel **偏置最大**的那条指令之前（= 最后一条静态指令）。
 *                       **不是**每 CTA 一次，而是每个 warp 的 lane 0 各做一次
 *                       atomicMax.u64：g1 = 最后一个走到这里的 warp 的时间 ——
 *                       CTA 一直占着资源到最后一个 warp 结束，取 max 比取"第一个收尾的
 *                       warp"更接近 CTA 真正结束的时刻（用 atomicExch 会系统性偏小）。
 *
 * ⚠️ 出口钩子是**近似**：NVBit 没有「CTA 结束」这种保证性钩子。early exit 的 kernel
 *    （部分 CTA 走不到最后一条静态指令）会**少报**行数 —— 覆盖率会掉下来，主机侧据此
 *    拒绝给出数字（见 README「覆盖率纪律」）。这一条**未在真机验证**。
 *
 * 时间戳为什么用 %globaltimer 而不是 %clock64：%clock64 是**每 SM 各自原点**的计数器
 * （老探针在真机上实测跨 SM 极差 2e12 cycles），跨 SM 比较必须用 globaltimer
 * （跨 SM 同源，ns）。见 tools/probes/occupancy_probe.cu 的开头三条坑。
 */

#include "ks_occ_common.h"

/* 平铺 blockIdx -> CTA 编号，与主机侧 grid_blocks = gridDim.x*y*z 同一序号空间。
 * 用 64 位算中间量：gridDim.x*blockIdx.y 在超大 grid 上会溢出 32 位。 */
__device__ __forceinline__ unsigned ks_occ_cta_id(void) {
    const unsigned long long x = blockIdx.x, y = blockIdx.y, z = blockIdx.z;
    const unsigned long long gx = gridDim.x, gy = gridDim.y;
    return (unsigned)(x + gx * (y + gy * z));
}

/* %globaltimer 的「值依赖」读法：把 v 作为输入操作数 + memory clobber。
 *
 * 为什么不能只写 asm volatile：tools/probes/occupancy_probe.cu 在真机上实测过，
 * 纯寄存器的计算仍会被编译器挪到时间戳之后（差 200×），所以那边用 GT_AFTER 把读
 * 挂在一个无法预先算出的值上。这里 v 由设备内存里的 salt 与 CTA 编号拼出来，编译器
 * 无法折成常数。本文件的另一层保证是 __noinline__：时间戳读活在一个真实的函数调用里。 */
__device__ __forceinline__ unsigned long long ks_occ_gt(unsigned long long v) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) : "l"(v) : "memory");
    return t;
}

extern "C" __device__ __noinline__ void ks_occ_enter(uint64_t cfg_ptr) {
    ks_occ_cfg_t* cfg = (ks_occ_cfg_t*)cfg_ptr;

    /* 诊断模式：只回答"注入的调用到底有没有执行"（真机上遇到过一次"插了但设备侧
     * 计数为 0"，见 nvbit_memtrace/README.md 的「状态」一节）。 */
    if (cfg->debug_calls) {
        atomicAdd((unsigned int*)cfg->calls, 1u);
        return;
    }

    const unsigned cta = ks_occ_cta_id();
    if (cta >= cfg->max_ctas) return;   /* 网格比 KS_OCC_MAX_CTAS 大：这些 CTA 记不到 */

    /* 每 CTA 一次：atomicExch 返回旧值，只有第一个到达的线程看到 0。 */
    if (atomicExch((unsigned int*)cfg->g0_done + cta, 1u) != 0u) return;

    const unsigned long long salt = (unsigned long long)cfg->salt;
    const unsigned long long t = ks_occ_gt(salt ^ ((unsigned long long)cta << 17) ^ cfg->recs);

    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    ks_occ_rec_t* r = (ks_occ_rec_t*)cfg->recs + cta;
    r->smid = smid;
    r->g0 = t;
    /* 主机侧只在 kernel 结束之后读回，普通写不用 fence（kernel 完成本身就是全屏障）。 */
    r->g0_ok = 1u;
}

extern "C" __device__ __noinline__ void ks_occ_exit(uint64_t cfg_ptr) {
    ks_occ_cfg_t* cfg = (ks_occ_cfg_t*)cfg_ptr;

    if (cfg->debug_calls) {
        atomicAdd((unsigned int*)cfg->calls, 1u);
        return;
    }

    const unsigned cta = ks_occ_cta_id();
    if (cta >= cfg->max_ctas) return;

    /* 每 warp 只留 lane 0 做原子：每个线程一次原子是纯浪费（32 倍开销）。 */
    if ((threadIdx.x & 31u) != 0u) return;

    const unsigned long long salt = (unsigned long long)cfg->salt;
    const unsigned long long t = ks_occ_gt(salt ^ ((unsigned long long)cta << 31) ^ cfg->recs);

    ks_occ_rec_t* r = (ks_occ_rec_t*)cfg->recs + cta;
    /* atomicMax 还有第二个好处：%globaltimer 单调，多 launch 时旧的 g1 一定更小，
     * 于是"陈旧值"不会与新的 g0 配成一对（主机侧还会用 g1 >= g0 再筛一次）。 */
    atomicMax((unsigned long long*)&r->g1, t);
    atomicExch((unsigned int*)&r->g1_ok, 1u);
}
