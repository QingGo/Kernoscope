// bench/ledger_probe.cu —— 字节数**已知**的微基准：给 DRAM 代理与账本对拍做真值。
//
// 为什么需要它：`est__dram_bytes_*` 是"显存忙碌时间 × 本机实测峰值带宽"的估算。
// 要判断它准不准，必须有一个**独立算出来的**每 launch 字节数当真值 ——
// 这个程序把真值打在 stdout 上（ANALYTIC 行），run.sh 再拿它当 --ks-ledger-mb。
//
// 用法：
//   nvcc -O3 -o ledger_probe ledger_probe.cu
//   ./ledger_probe --kernel copy  --mb 256 --target-s 6      # 按目标时长自动定迭代数
//   ./ledger_probe --kernel read  --mb 256 --iters 4000
//   ./ledger_probe --kernel write --mb 256 --iters 4000
//   ./ledger_probe --kernel norm  --mb 256 --iters 4000      # 读 x + 读 w + 写 y
//   ./ledger_probe --kernel strided --mb 256 --iters 4000    # 同样字节数，但合并很差
//   ./ledger_probe --kernel copy  --mb 64 --iters 300 --grid 64|256|1024   # 形状扫描
//   ./ledger_probe --kernel copy  --mb 256 --iters 400 --analyze-only      # 只打印真值
//   ./ledger_probe --kernel copy  --mb 1 --iters 1 --warmup 2 --grid 16 --block 128  # 采 NVBit 轨迹用的小负载
//
// 输出（stdout 上机器可读的两行）：
//   ANALYTIC kernel=copy bytes_per_launch=1073741824 iters=4000 total_bytes=4.29e12 grid=… block=… n=…
//   DONE elapsed_s=12.3
//
// 为什么要有 --target-s：DRAM 代理只有在**窗口被负载主导**时才可用，而"跑多少迭代"
// 取决于这张卡的带宽。手工填迭代数很容易跑出"3 秒窗口里只忙 0.09 秒"的结果（覆盖率 0%），
// 于是报告正确地拒绝给数 —— 但那是实验设计的问题，不是工具的问题。--target-s 让迭代数
// 按实测的每-launch 耗时反推，实验就可复现了。
//
// 三个坑（与 tools/probes/occupancy_probe.cu 同源）：
//   1. 结果必须被写回内存，否则 ptxas 会把整段循环当死代码删掉；
//   2. 用 --grid 显式控制 grid，形状扫描要的是"同一个 kernel 换 launch 配置"；
//   3. 预热必须做够，否则时钟爬升与首次分配会混进测量窗。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

__global__ void copy_k(const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) y[i] = x[i];
}

__global__ void read_k(const float* __restrict__ x, float* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += x[i];
    if (acc == -1.0f) out[0] = acc;            // 值依赖：不让编译器删掉读
}

__global__ void write_k(float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) y[i] = 1.0f;
}

// 读 x + 读 w + 写 y：账本口径 = 3N×4B（融合 norm 那类形状）
__global__ void norm_k(const float* __restrict__ x, const float* __restrict__ w,
                       float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) y[i] = x[i] * w[i % 1024];
}

// 同样量级的字节数，但按 32 个 float 跨步访问：合并效率低、sectors/request 高
__global__ void strided_k(const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    long m = n / 32;
    for (; i < m; i += stride) y[(i * 32) % n] = x[i];
}

static void launch(const char* k, int grid, int block, const float* x, const float* w,
                   float* y, float* out, long n) {
    if (!strcmp(k, "copy")) copy_k<<<grid, block>>>(x, y, n);
    else if (!strcmp(k, "read")) read_k<<<grid, block>>>(x, out, n);
    else if (!strcmp(k, "write")) write_k<<<grid, block>>>(y, n);
    else if (!strcmp(k, "norm")) norm_k<<<grid, block>>>(x, w, y, n);
    else strided_k<<<grid, block>>>(x, y, n);
}

int main(int argc, char** argv) {
    const char* kernel = "copy";
    long mb = 256;
    int iters = 4000, block = 256, grid = 0, device = 0, analyze_only = 0, warmup = 50;
    double target_s = 0.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--kernel") && i + 1 < argc) kernel = argv[++i];
        else if (!strcmp(argv[i], "--mb") && i + 1 < argc) mb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-s") && i + 1 < argc) target_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "--block") && i + 1 < argc) block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grid") && i + 1 < argc) grid = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--analyze-only")) analyze_only = 1;
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    long n = mb * 1024 * 1024 / 4;                 // float 个数
    int grid_arg = grid;

    // 每 launch 的解析字节数（真值）—— 不需要 GPU 就能算，--analyze-only 只用这一步
    double bytes_per_launch = 0;
    if (!strcmp(kernel, "copy"))         bytes_per_launch = 2.0 * n * 4;
    else if (!strcmp(kernel, "read"))    bytes_per_launch = 1.0 * n * 4;
    else if (!strcmp(kernel, "write"))   bytes_per_launch = 1.0 * n * 4;
    else if (!strcmp(kernel, "norm"))    bytes_per_launch = 3.0 * n * 4;
    else if (!strcmp(kernel, "strided")) bytes_per_launch = 2.0 * (n / 32) * 4;
    else { fprintf(stderr, "unknown kernel %s\n", kernel); return 2; }

    if (analyze_only) {
        fprintf(stdout, "ANALYTIC kernel=%s bytes_per_launch=%.0f iters=%d total_bytes=%.6g "
                        "grid=%d block=%d n=%ld\n",
                kernel, bytes_per_launch, iters, bytes_per_launch * iters, grid_arg, block, n);
        return 0;
    }

    CHECK(cudaSetDevice(device));
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, device));
    if (grid <= 0) grid = prop.multiProcessorCount * 8;

    float *x, *y, *w, *out;
    CHECK(cudaMalloc(&x, n * 4)); CHECK(cudaMalloc(&y, n * 4));
    CHECK(cudaMalloc(&w, 1024 * 4)); CHECK(cudaMalloc(&out, 4));
    CHECK(cudaMemset(x, 1, n * 4)); CHECK(cudaMemset(w, 1, 1024 * 4));

    // 预热（把 JIT / 首次分配 / 时钟爬升挪出测量窗）。
    // 采集 NVBit 轨迹时把它调小（--warmup 2）：插桩会把**每一次** launch 都写进轨迹，
    // 50 次预热足以让轨迹文件涨到 GB 级。
    for (int i = 0; i < warmup; i++) launch(kernel, grid, block, x, w, y, out, n);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    // 按目标时长反推迭代数：先用 20 次 launch 估每次耗时
    if (target_s > 0.0) {
        CHECK(cudaEventRecord(t0));
        for (int i = 0; i < 20; i++) launch(kernel, grid, block, x, w, y, out, n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double per_launch_s = (ms / 1000.0) / 20.0;
        if (per_launch_s > 0) {
            iters = (int)(target_s / per_launch_s);
            if (iters < 20) iters = 20;
            if (iters > 20000000) iters = 20000000;
        }
        fprintf(stderr, "per_launch_ms=%.4f -> iters=%d (target %.1f s)\n",
                per_launch_s * 1000.0, iters, target_s);
    }

    fprintf(stdout, "ANALYTIC kernel=%s bytes_per_launch=%.0f iters=%d total_bytes=%.6g "
                    "grid=%d block=%d n=%ld\n",
            kernel, bytes_per_launch, iters, bytes_per_launch * iters, grid, block, n);
    fflush(stdout);

    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) launch(kernel, grid, block, x, w, y, out, n);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaDeviceSynchronize());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(stdout, "DONE elapsed_s=%.3f\n", ms / 1000.0);
    return 0;
}
