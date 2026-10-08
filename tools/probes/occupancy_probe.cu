// =============================================================================
// occupancy_probe.cu —— 区间重叠法测「达到的 occupancy」（免硬件计数器）
//
// 为什么需要它：ncu 的 Achieved Occupancy 来自硬件计数器，在受限环境拿不到。
// 替代方案：让每个 block 记录 (自己落在哪个 SM, 开始与结束的 %globaltimer)，
// 主机侧用「区间重叠的时间平均并发度」重建 —— 语义与 ncu 同源，但出处是 measured。
//
// 输出格式 ks-occupancy-records v1（可直接喂给 `kernoscope occupancy --from`）：
//     # ks-occupancy-records v1
//     # kernel k_occ
//     # arch sm_89
//     # threads_per_block 128
//     # sm_count 128
//     # grid_blocks 3072
//     smid,g0,g1
//     0,123456789,123512345
//     ...
//
// 编译运行：
//     nvcc -O3 -arch=sm_89 tools/probes/occupancy_probe.cu -o occupancy_probe
//     ./occupancy_probe occupancy-records.txt
//     kernoscope occupancy --from occupancy-records.txt --write-json occ.json
//     kernoscope --ks-achieved-occupancy occ.json -k regex:<kernel> -- <你的命令>
//
// 三个真机踩过的坑，本文件已处理（抄到自己 kernel 里时别丢）：
//   1. 时间戳必须「值依赖」被计时的计算，只加 "memory" clobber 不够 ——
//      纯寄存器计算仍会被编译器挪到时间戳之后（实测差 200×）。见 GT_AFTER。
//   2. %clock64 是「每 SM 各自原点」的计数器（实测跨 SM 极差 2e12 cycles），
//      跨 SM 比较必须用 %globaltimer（跨 SM 极差 0，单位 ns）。
//   3. 不要把被测对象优化掉：共享内存要「写+读」，否则 ptxas 会删掉死代码，
//      occupancy 直接失真（实测 32KB smem 的 kernel 被算成 12 blocks/SM）。
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  printf("CUDA error: %s @%d\n", cudaGetErrorString(_e), __LINE__); exit(1); } } while (0)

#define GT_AFTER(t, v) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) \
                                    : "l"((unsigned long long)(v)) : "memory")
#define SMID(r)        asm volatile("mov.u32 %0, %%smid;" : "=r"(r))

struct BlockRec { int smid; unsigned long long g0, g1; };

// SMEM_BYTES = 0 表示不用共享内存；用模板参数便于对比不同的 occupancy 上限
template <int SMEM_BYTES>
__global__ void k_occ(BlockRec* rec, int iters) {
  __shared__ float dummy[(SMEM_BYTES ? SMEM_BYTES : 4) / 4];
  unsigned smid; SMID(smid);
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0) : "l"((unsigned long long)smid) : "memory");

  float x = threadIdx.x * 1e-6f + 1.0f;
#pragma unroll 4
  for (int i = 0; i < iters; i++) x = fmaf(x, 1.0000001f, 0.5f);

  if (SMEM_BYTES) {                 // 必须写后读，否则 smem 会被 ptxas 当死代码删掉
    dummy[threadIdx.x & 15] = x;
    __syncthreads();
    x += dummy[(threadIdx.x + 1) & 15];
  }
  GT_AFTER(g1, __float_as_uint(x));  // 值依赖，防止时间戳被挪到循环之前
  if (threadIdx.x == 0) rec[blockIdx.x] = BlockRec{(int)smid, g0, g1};
  if (x == 12345.0f) rec[0].smid = -1;   // 防止整段被优化掉
}

static double g_theo_pct = 0.0;   // 探针记录用（由 run_case 填充）

template <int SMEM_BYTES>
static void run_case(const char* label, BlockRec* rec, int blocks, int threads, int iters,
                     int sm_count, int arch_warps) {
  int theo_blocks = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&theo_blocks, (const void*)k_occ<SMEM_BYTES>,
                                                   threads, 0));
  k_occ<SMEM_BYTES><<<blocks, threads>>>(rec, iters);
  CK(cudaDeviceSynchronize());
  double theo = 100.0 * theo_blocks * (threads / 32.0) / arch_warps;
  g_theo_pct = theo;
  printf("  %-22s 理论 %2d blocks/SM (%.1f%%)  已采集 %d 个 block 记录\n",
         label, theo_blocks, theo, blocks);
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "occupancy-records.txt";

  int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  const int sm_count = p.multiProcessorCount;
  const int arch_warps = p.maxThreadsPerMultiProcessor / 32;
  char arch[32]; snprintf(arch, sizeof arch, "sm_%d%d", p.major, p.minor);

  printf("GPU: %s  SM=%d  warps/SM=%d  arch=%s\n", p.name, sm_count, arch_warps, arch);

  // 被测 kernel 的规模：128 线程/块（4 warps）。grid 给到理论并发的 8 倍，才有稳态可言。
  const int threads = 128, iters = 20000;
  const int blocks = sm_count * 12 * 8;
  BlockRec* rec; CK(cudaMalloc(&rec, (size_t)blocks * sizeof(BlockRec)));

  auto dump = [&](const std::string& path, const char* label, int blocks_n, double theo) {
    std::vector<BlockRec> h(blocks_n);
    CK(cudaMemcpy(h.data(), rec, (size_t)blocks_n * sizeof(BlockRec), cudaMemcpyDeviceToHost));
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { printf("无法写入 %s\n", path.c_str()); return; }
    fprintf(f, "# ks-occupancy-records v1\n");
    fprintf(f, "# kernel %s\n", label);
    fprintf(f, "# arch %s\n", arch);
    fprintf(f, "# threads_per_block %d\n", threads);
    fprintf(f, "# sm_count %d\n", sm_count);
    fprintf(f, "# grid_blocks %d\n", blocks_n);
    fprintf(f, "# theoretical_occupancy_pct %.1f\n", theo);
    fprintf(f, "smid,g0,g1\n");
    int written = 0;
    for (const auto& r : h) {
      if (r.smid >= 0) { fprintf(f, "%d,%llu,%llu\n", r.smid, r.g0, r.g1); written++; }
    }
    fclose(f);
    printf("  已写出 %s（%d 条记录，理论 %.1f%%）\n", path.c_str(), written, theo);
  };

  printf("对照实验（无 smem 理论上限 12 blocks/SM；32KB smem 上限 3 blocks/SM）：\n");
  run_case<0>("k_occ<0>（无 smem）", rec, blocks, threads, iters, sm_count, arch_warps);
  dump(std::string(out), "k_occ_smem0", blocks, g_theo_pct);
  run_case<32768>("k_occ<32768>（32KB smem）", rec, blocks, threads, iters, sm_count, arch_warps);
  dump(std::string(out) + ".smem32k", "k_occ_smem32k", blocks, g_theo_pct);

  printf("\n已写出 %s（%d 条 block 记录）\n", out, blocks);
  printf("下一步:\n");
  printf("  kernoscope occupancy --from %s --write-json occ-smem0.json\n", out);
  printf("  kernoscope occupancy --from %s.smem32k --write-json occ-smem32k.json\n", out);
  return 0;
}
