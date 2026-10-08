// SPDX-License-Identifier: Apache-2.0
// 已知访存形态的自测内核：用来验证 ks-memtrace 的 sectors/request 是否算对。
//   coalesced : 相邻 lane 访问相邻 float  -> 32 lane × 4B = 128B = 4 个 32B sector
//   strided32 : lane 间隔 32 个 float     -> 32 个 sector（每个 lane 独占一块）
//   unaligned : 首地址 +4 字节            -> 5 个 sector（跨了 5 个 32B 块）
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s@%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

__global__ void k_coalesced(const float* __restrict__ x, float* __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  y[i] = x[i] * 2.0f;
}
__global__ void k_strided32(const float* __restrict__ x, float* __restrict__ y) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
  y[i] = x[i] * 2.0f;
}
__global__ void k_unaligned(const float* __restrict__ x, float* __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  y[i] = x[i] * 2.0f;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("用法: %s coalesced|strided32|unaligned [n]\n", argv[0]); return 1; }
  const char* mode = argv[1];
  long n = (argc > 2) ? atol(argv[2]) : 1024;
  float *x, *y;
  CK(cudaMalloc(&x, n * sizeof(float) * 32 + 256));
  CK(cudaMalloc(&y, n * sizeof(float) * 32 + 256));
  CK(cudaMemset(x, 1, n * sizeof(float) * 32 + 256));
  // 只用 1 个 block、32 个线程 => 恰好一个 warp，轨迹确定、易核对
  if (mode[0] == 'c')      k_coalesced<<<1, 32>>>(x, y);
  else if (mode[0] == 's') k_strided32<<<1, 32>>>(x, y);
  else                     k_unaligned<<<1, 32>>>(x, y);
  CK(cudaDeviceSynchronize());
  printf("ran %s\n", mode);
  return 0;
}
