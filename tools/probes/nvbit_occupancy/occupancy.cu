/* SPDX-License-Identifier: Apache-2.0 */
/* Kernoscope「达到的 occupancy」工具（NVBit 版，主机侧）—— 产出 ks-occupancy-records v1。
 *
 * 它解决的是 occupancy_probe.cu 的硬伤：那个探针要求用户**把几行代码抄进自己的 kernel**
 * （README §6.1），于是闭源 kernel（cuBLAS / Inductor 生成的 / 库里的）永远测不了，
 * 而且用户得改自己可能并不拥有的代码。本工具用 NVBit 在 **SASS 层**注入同样的两个
 * 时间戳读，被测 kernel **零源码改动**。
 *
 * 本文件是**我们的代码**，但编译时需要用户自己的 NVBit 安装（NVIDIA 专有许可，
 * 本项目不分发 NVBit 本体，也不分发用 NVBit 构建好的二进制）。
 *
 * 环境变量：
 *   KS_OCC_OUT            输出路径（默认 ks-occupancy.txt）
 *   KS_OCC_KERNEL         kernel 名子串过滤；**锁定第一个匹配到的 kernel 名**，
 *                         之后只插桩同名 kernel（避免把别的 kernel 的 CTA 混进同一份记录）
 *   KS_OCC_MAX_CTAS       设备缓冲能容纳的 CTA 数（默认 262144，约 8 MB）
 *   KS_OCC_MIN_COVERAGE   覆盖率阈值（默认 0.95）
 *   KS_OCC_ALLOW_PARTIAL  1 = 纪律没过也强行写出（文件带 coverage_ok 0，**别喂分析器**）
 *   KS_OCC_VERBOSE        1 打印插桩细节
 *   KS_OCC_DEBUG_CALLS    1 = 只统计"注入调用执行了几次"，不写记录（排查用）
 *
 * 用法：
 *   CUDA_INJECTION64_PATH=$PWD/ks_occupancy.so LD_PRELOAD=$PWD/ks_occupancy.so \
 *     KS_OCC_KERNEL=my_kernel KS_OCC_OUT=ks-occupancy.txt ./your_app
 *   kernoscope occupancy --from ks-occupancy.txt --write-json occ.json
 *
 * 输出必须与 src/kernoscope/analyze/occupancy.py 的 load_probe_records() **逐字节兼容**
 * （同一探针格式：`# ks-occupancy-records v1` + `# key value` 头 + `smid,g0,g1` 行）。
 * 那个 Python 文件是既有分析路径，本工具不许改它，也不许给它加新格式。
 */

#include "nvbit_tool.h"

#include "nvbit.h"
#include "ks_occ_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

/* ------------------------------------------------------------------ 配置 */
static std::string g_out = "ks-occupancy.txt";
static std::string g_filter;
static int g_max_ctas = 262144;
static double g_min_coverage = 0.95;
static int g_allow_partial = 0;
static int g_verbose = 0;
static int g_debug_calls = 0;

static void read_env() {
    if (const char* e = getenv("KS_OCC_OUT")) g_out = e;
    if (const char* e = getenv("KS_OCC_KERNEL")) g_filter = e;
    if (const char* e = getenv("KS_OCC_MAX_CTAS")) g_max_ctas = atoi(e);
    if (const char* e = getenv("KS_OCC_MIN_COVERAGE")) g_min_coverage = atof(e);
    if (const char* e = getenv("KS_OCC_ALLOW_PARTIAL")) g_allow_partial = atoi(e);
    if (const char* e = getenv("KS_OCC_VERBOSE")) g_verbose = atoi(e);
    if (const char* e = getenv("KS_OCC_DEBUG_CALLS")) g_debug_calls = atoi(e);
    if (g_max_ctas < 1) g_max_ctas = 1;
    if (g_min_coverage < 0.0) g_min_coverage = 0.0;
    if (g_min_coverage > 1.0) g_min_coverage = 1.0;
}

/* --------------------------------------------------------------- 设备缓冲 */
static ks_occ_rec_t* g_d_recs = nullptr;
static unsigned int* g_d_g0_done = nullptr;
static unsigned int* g_d_calls = nullptr;
static ks_occ_cfg_t* g_d_cfg = nullptr;
static ks_occ_cfg_t g_cfg_host{};
static std::unordered_set<CUfunction> g_instrumented;
static std::mutex g_mtx;
static bool g_skip = false;      /* 防止在回调里再触发回调 */

/* -------------------------------------------------------------- launch 状态 */
static bool g_have_launch = false;
static std::string g_kernel;          /* 头部里的 kernel 名 */
static std::string g_locked_name;     /* 锁定后只插桩这个名字 */
static CUfunction g_func = nullptr;   /* 第一个被插桩的 kernel（算理论上界用） */
static uint64_t g_grid = 0;           /* 第一个 launch 的 grid */
static uint64_t g_block = 0;          /* 第一个 launch 的每 block 线程数 */
static uint64_t g_dyn_smem = 0;       /* 第一个 launch 的动态共享内存 */
static uint64_t g_grid_max = 0;       /* 所有被插桩 launch 里最大的 grid（覆盖率分母） */
static int g_launches = 0;
static int g_config_mismatch = 0;     /* 后续 launch 的 grid/block 与第一个不同 */
static int g_instrumented_instrs = 0;
static int g_warned_other_kernel = 0;

/* ------------------------------------------------------------------- 钩子 */
void nvbit_at_init() {
    read_env();
    if (g_verbose) {
        printf("KS-OCCUPANCY: out=%s kernel=%s max_ctas=%d min_coverage=%.2f debug_calls=%d\n",
               g_out.c_str(), g_filter.empty() ? "(第一个 launch)" : g_filter.c_str(),
               g_max_ctas, g_min_coverage, g_debug_calls);
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    /* 这里不要做 CUDA 分配：NVBit 会警告 "Do not call CUDA memory allocation in
     * nvbit_at_ctx_init(). It will cause deadlocks." —— 统一放到 nvbit_tool_init。 */
    (void)ctx;
}

void nvbit_tool_init(CUcontext ctx) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_d_recs != nullptr) return;          /* 只分配一次 */
    g_skip = true;
    cudaMalloc((void**)&g_d_recs, (size_t)g_max_ctas * sizeof(ks_occ_rec_t));
    cudaMalloc((void**)&g_d_g0_done, (size_t)g_max_ctas * sizeof(unsigned int));
    cudaMalloc((void**)&g_d_calls, sizeof(unsigned int));
    cudaMemset(g_d_recs, 0, (size_t)g_max_ctas * sizeof(ks_occ_rec_t));
    cudaMemset(g_d_g0_done, 0, (size_t)g_max_ctas * sizeof(unsigned int));
    cudaMemset(g_d_calls, 0, sizeof(unsigned int));
    g_cfg_host.recs = (uint64_t)g_d_recs;
    g_cfg_host.g0_done = (uint64_t)g_d_g0_done;
    g_cfg_host.calls = (uint64_t)g_d_calls;
    /* 值依赖用的盐：由时钟与进程内地址拼出来，编译器折不了（见 inject_funcs.cu）。 */
    g_cfg_host.salt = (uint64_t)time(nullptr) * 2654435761ULL
                      ^ reinterpret_cast<uint64_t>(&g_cfg_host);
    g_cfg_host.max_ctas = (uint32_t)g_max_ctas;
    g_cfg_host.debug_calls = g_debug_calls ? 1u : 0u;
    cudaMalloc((void**)&g_d_cfg, sizeof(ks_occ_cfg_t));
    cudaMemcpy(g_d_cfg, &g_cfg_host, sizeof(ks_occ_cfg_t), cudaMemcpyHostToDevice);
    g_skip = false;
    if (g_verbose) {
        printf("KS-OCCUPANCY: 设备缓冲 %d 个 CTA 槽（%.1f MB）\n", g_max_ctas,
               (double)g_max_ctas * sizeof(ks_occ_rec_t) / 1e6);
    }
    (void)ctx;
}

/* 插桩一个函数：入口 = 偏置最小的指令，出口 = 偏置最大的指令。
 *
 * 为什么按 getOffset() 找首尾、而不是用 nvbit_get_instrs() 的**表序**：表序是 NVBit
 * 的实现细节（官方示例都按表序遍历），偏置才是 SASS 里的事实。取 min/max 与表序无关。
 *
 * 为什么**不**像 nvbit_memtrace 那样走 nvbit_get_related_functions：被调用的设备函数
 * 可能被多个 kernel 共用，插进它里面会让**别的** kernel 的 CTA 也来写这份记录，
 * 于是 CTA 编号空间与 kernel 对不上。这里只插桩被过滤/锁定的那个顶层 kernel。
 */
static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    if (!g_instrumented.insert(func).second) return;   /* 只插一次，后续 launch 复用 */

    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);
    const std::string fname = nvbit_get_func_name(ctx, func);

    Instr* first = nullptr;
    Instr* last = nullptr;
    uint64_t lo = 0, hi = 0;
    for (Instr* instr : instrs) {
        const uint64_t off = (uint64_t)instr->getOffset();
        if (first == nullptr || off < lo) { lo = off; first = instr; }
        if (last == nullptr || off >= hi) { hi = off; last = instr; }
    }
    if (first == nullptr) {
        fprintf(stderr, "KS-OCCUPANCY: %s 没有静态指令，插桩跳过\n", fname.c_str());
        return;
    }

    const uint64_t cfg = (uint64_t)g_d_cfg;
    /* 参数顺序必须与 inject_funcs.cu 里那两个函数的形参顺序一致（都是一个指针）。 */
    nvbit_insert_call(first, "ks_occ_enter", IPOINT_BEFORE);
    nvbit_add_call_arg_const_val64(first, cfg);
    nvbit_insert_call(last, "ks_occ_exit", IPOINT_BEFORE);
    nvbit_add_call_arg_const_val64(last, cfg);
    g_instrumented_instrs += 2;

    if (g_verbose) {
        printf("KS-OCCUPANCY: %s -> 入口@0x%llx / 出口@0x%llx（%zu 条静态指令）\n",
               fname.c_str(), (unsigned long long)lo, (unsigned long long)hi, instrs.size());
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name,
                         void* params, CUresult* pStatus) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_skip) return;
    g_skip = true;

    CUfunction func = nullptr;
    uint64_t grid = 0, block = 0, dyn_smem = 0;
    if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        func = p->f;
        grid = (uint64_t)p->gridDimX * p->gridDimY * p->gridDimZ;
        block = (uint64_t)p->blockDimX * p->blockDimY * p->blockDimZ;
        dyn_smem = (uint64_t)p->sharedMemBytes;
    } else if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        func = p->f;
        grid = (uint64_t)p->config->gridDimX * p->config->gridDimY * p->config->gridDimZ;
        block = (uint64_t)p->config->blockDimX * p->config->blockDimY * p->config->blockDimZ;
        dyn_smem = (uint64_t)p->config->sharedMemBytes;
    }

    if (func != nullptr && !is_exit) {
        const std::string kname = nvbit_get_func_name(ctx, func);
        bool match = false;
        if (g_locked_name.empty()) {
            match = g_filter.empty() || kname.find(g_filter) != std::string::npos;
        } else if (kname == g_locked_name) {
            match = true;
        } else if (!g_warned_other_kernel
                   && (g_filter.empty() || kname.find(g_filter) != std::string::npos)) {
            /* 过滤为空时锁的是"第一个 launch"；之后别的 kernel 也会匹配上，必须说清楚
             * 它们**没有**被插桩（否则报告里的 CTA 数会被误当成整个进程的）。 */
            g_warned_other_kernel = 1;
            fprintf(stderr,
                    "KS-OCCUPANCY: 只测第一个被插桩的 kernel（%s）；%s 也匹配但没有插桩。\n"
                    "  要测另一个 kernel 就用 KS_OCC_KERNEL=<子串> 缩小过滤。\n",
                    g_locked_name.c_str(), kname.c_str());
        }

        if (match) {
            if (g_locked_name.empty()) {
                g_locked_name = kname;
                g_have_launch = true;
                g_kernel = kname;
                g_func = func;
                g_grid = grid;
                g_block = block;
                g_dyn_smem = dyn_smem;
                g_grid_max = grid;
            } else {
                if (grid != g_grid || block != g_block) g_config_mismatch = 1;
                if (grid > g_grid_max) g_grid_max = grid;
            }
            instrument_function_if_needed(ctx, func);
            g_launches++;
        }
    }
    g_skip = false;
    (void)name;
    (void)pStatus;
}

/* 纪律没过时要说清**为什么**（只写"覆盖率不足"没法复核）。 */
static std::vector<std::string> discipline_problems(uint64_t rows, uint64_t grid,
                                                    double coverage) {
    std::vector<std::string> probs;
    if (grid == 0) {
        probs.push_back("grid_blocks=0：没拿到 launch 配置");
    }
    if (coverage < g_min_coverage) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "覆盖率 %.2f%% < 阈值 %.2f%%（CTA 行 %llu / grid_blocks %llu）",
                 coverage * 100.0, g_min_coverage * 100.0, (unsigned long long)rows,
                 (unsigned long long)grid);
        probs.push_back(buf);
    }
    if (g_launches > 1) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "同一个 kernel 被 launch 了 %d 次：按 CTA 槽记录会把多次 launch 揉在一份"
                 "记录里（区间重叠法会把它当成一次，达到 occupancy 会被系统性拉低）",
                 g_launches);
        probs.push_back(buf);
    }
    if (g_config_mismatch) {
        probs.push_back("后续 launch 的 grid/block 与第一个不同：同一份记录里混了不同配置");
    }
    return probs;
}

void nvbit_at_term() {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!g_have_launch) {
        fprintf(stderr,
                "KS-OCCUPANCY: 没有匹配到任何 kernel（KS_OCC_KERNEL=%s）—— 没有产出记录\n",
                g_filter.c_str());
        return;
    }

    g_skip = true;
    unsigned int calls = 0;
    if (g_d_calls) cudaMemcpy(&calls, g_d_calls, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    const uint64_t grid = g_grid_max;
    const uint64_t copy_n = std::min<uint64_t>(grid, (uint64_t)g_max_ctas);
    std::vector<ks_occ_rec_t> recs((size_t)copy_n);
    if (copy_n) {
        /* 与 nvbit_memtrace 同一套读回方式（那份在真机上跑通过）。若最后一个 kernel 还没
         * 结束就读回，行数会偏少 —— 表现就是覆盖率掉下来、工具据此拒绝，而不是给出一个
         * 偏小的分布。 */
        cudaMemcpy(recs.data(), g_d_recs, (size_t)copy_n * sizeof(ks_occ_rec_t),
                   cudaMemcpyDeviceToHost);
    }

    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    g_skip = false;

    if (g_debug_calls) {
        /* 诊断模式：不写记录 —— 只回答"注入的调用有没有真的执行"。
         *（真事：nvbit_memtrace 在本环境遇到过"插桩调用已插入但设备计数器为 0"。） */
        printf("KS-OCCUPANCY: 诊断模式（KS_OCC_DEBUG_CALLS=1）—— 注入调用执行 %u 次，"
               "kernel=%s，launches=%d（不写记录文件）\n",
               calls, g_kernel.c_str(), g_launches);
        if (calls == 0) {
            fprintf(stderr,
                    "KS-OCCUPANCY: 注入调用一次都没执行（计数器 0）—— 这份配置下测不出来。\n"
                    "  先确认 NO_EAGER_LOAD=0、工具架构与 GPU 匹配（ARCH=native 重建），\n"
                    "  并先用官方 instr_count 验证 NVBit 在本机可用。\n");
        }
        return;
    }

    uint64_t rows = 0;
    for (const ks_occ_rec_t& r : recs) {
        /* 三个条件缺一不可：入口命中过、出口命中过、配对后时间单调。
         * g1 >= g0 这一条同时挡住"多次 launch 留下的陈旧 g1"。 */
        if (r.g0_ok && r.g1_ok && r.g1 > 0 && r.g1 >= r.g0) rows++;
    }
    const double coverage = grid ? (double)rows / (double)grid : 0.0;

    std::vector<std::string> probs = discipline_problems(rows, grid, coverage);
    const bool ok = probs.empty();

    if (!ok) {
        fprintf(stderr, "KS-OCCUPANCY: 覆盖率纪律没过 —— 这份数据**不能**用来算达到 occupancy：\n");
        for (const std::string& p : probs) fprintf(stderr, "  - %s\n", p.c_str());
        fprintf(stderr,
                "  可能原因：kernel 有 early exit / 提前 return（本工具只挂钩「最后一条静态"
                "指令」，走不到那里的 CTA 记不到）；网格大于 KS_OCC_MAX_CTAS=%d；\n"
                "            被测 kernel 被 launch 了多次；KS_OCC_KERNEL 过滤到的不是"
                "真正想要的那个 kernel。\n"
                "  部分数据算出来的分布只会偏小，不会给出「数据不全」的信号 —— 所以这里"
                "默认**不写文件**。\n"
                "  只想排查用：KS_OCC_ALLOW_PARTIAL=1 重跑（文件会带 `# coverage_ok 0`，"
                "仍然**别**喂给 `kernoscope occupancy`）。\n",
                g_max_ctas);
        if (!g_allow_partial) {
            printf("KS-OCCUPANCY: 状态 拒绝（未写出文件）\n");
            return;
        }
        fprintf(stderr, "KS-OCCUPANCY: KS_OCC_ALLOW_PARTIAL=1 —— 仍然写出，但状态仍是拒绝。\n");
    }

    /* 理论上界：交给**驱动**算（cudaOccupancyMaxActiveBlocksPerMultiprocessor 的驱动版），
     * 与 occupancy_probe.cu 头部那个 `theoretical_occupancy_pct` 同源；分析器会透传它。
     * 算不出来就不写这一行（宁缺勿造）。 */
    double theo_pct = -1.0;
    if (g_func != nullptr && g_block > 0) {
        int theo_blocks = 0;
        if (cuOccupancyMaxActiveBlocksPerMultiprocessor(&theo_blocks, g_func, (int)g_block,
                                                        (size_t)g_dyn_smem) == CUDA_SUCCESS
            && theo_blocks > 0 && prop.maxThreadsPerMultiProcessor > 0) {
            const double warps_per_block = (double)((g_block + 31) / 32);
            const double max_warps = (double)(prop.maxThreadsPerMultiProcessor / 32);
            theo_pct = 100.0 * (double)theo_blocks * warps_per_block / max_warps;
        }
    }

    char arch[32];
    snprintf(arch, sizeof(arch), "sm_%d%d", prop.major, prop.minor);

    std::ofstream f(g_out);
    if (!f) {
        fprintf(stderr, "KS-OCCUPANCY: 无法写入 %s\n", g_out.c_str());
        return;
    }
    f << "# ks-occupancy-records v1\n";
    f << "# kernel " << g_kernel << "\n";
    f << "# arch " << arch << "\n";
    f << "# threads_per_block " << g_block << "\n";
    f << "# sm_count " << prop.multiProcessorCount << "\n";
    f << "# grid_blocks " << grid << "\n";
    if (theo_pct >= 0.0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1f", theo_pct);
        f << "# theoretical_occupancy_pct " << buf << "\n";
    }
    f << "# cta_rows " << rows << "\n";
    char cov[32];
    snprintf(cov, sizeof(cov), "%.4f", coverage);
    f << "# coverage " << cov << "\n";
    f << "# coverage_ok " << (ok ? 1 : 0) << "\n";
    char mincov[32];
    snprintf(mincov, sizeof(mincov), "%.4f", g_min_coverage);
    f << "# min_coverage " << mincov << "\n";
    f << "# launches_traced " << g_launches << "\n";
    f << "# config_mismatch " << (g_config_mismatch ? 1 : 0) << "\n";
    f << "# instrumented_instrs " << g_instrumented_instrs << "\n";
    f << "# allow_partial " << (g_allow_partial ? 1 : 0) << "\n";
    f << "# source nvbit-sass-injection\n";
    f << "# note g0 = CTA 入口 %globaltimer（ns，跨 SM 同源）；g1 = 最后一个走到"
         "「最后一条静态指令」的 warp 的时间\n";
    f << "# note 出口钩子是近似：NVBit 没有「CTA 结束」钩子，early exit 的 CTA 会少报\n";
    f << "smid,g0,g1\n";
    uint64_t written = 0;
    for (uint64_t i = 0; i < copy_n; i++) {
        const ks_occ_rec_t& r = recs[(size_t)i];
        if (!(r.g0_ok && r.g1_ok && r.g1 > 0 && r.g1 >= r.g0)) continue;
        f << r.smid << "," << r.g0 << "," << r.g1 << "\n";
        written++;
    }
    f.close();

    printf("KS-OCCUPANCY: 写出 %s —— CTA 行 %llu/%llu（覆盖率 %.2f%%），kernel=%s，"
           "launches=%d，注入调用=%u，插桩指令=%d\n",
           g_out.c_str(), (unsigned long long)written, (unsigned long long)grid,
           coverage * 100.0, g_kernel.c_str(), g_launches, calls, g_instrumented_instrs);
    /* 这两行是给 collect.sh（与别的脚本）grep 的**状态标记**，必须是完整字面量：
     * 用 printf 组装出来的字符串 grep 不到，"拒绝"就会静默漏过。 */
    if (ok) {
        printf("KS-OCCUPANCY: 状态 可用\n");
        printf("  下一步: kernoscope occupancy --from %s --write-json occ.json\n",
               g_out.c_str());
    } else {
        printf("KS-OCCUPANCY: 状态 拒绝\n");
    }
}
