/* SPDX-License-Identifier: Apache-2.0 */
/* Kernoscope 访存轨迹工具（主机侧）—— 产出 ks-memtrace v1 文本轨迹。
 *
 * 本文件是**我们的代码**，但编译时需要用户自己的 NVBit 安装（NVIDIA 专有许可，
 * 本项目不分发 NVBit 本体，也不分发用 NVBit 构建好的二进制）。
 *
 * 环境变量：
 *   KS_MEMTRACE_OUT           输出路径（默认 ks-memtrace.txt）
 *   KS_MEMTRACE_KERNEL        kernel 名子串过滤（空 = 全部；强烈建议指定，否则轨迹巨大）
 *   KS_MEMTRACE_BLOCK_STRIDE  只记录 blockIdx.x % stride == 0 的 block（默认 1 = 全量）
 *   KS_MEMTRACE_MAX_RECORDS   设备缓冲上限（默认 200000 条 ≈ 54 MB）
 *   KS_MEMTRACE_VERBOSE       1 打印插桩细节
 *
 * 用法：
 *   CUDA_INJECTION64_PATH=$PWD/memtrace.so LD_PRELOAD=$PWD/memtrace.so ./your_app
 *   kernoscope memtrace --from ks-memtrace.txt
 *
 * 为什么不在设备侧去重/排序：设备侧每多一次比较都会拖慢被测 kernel，
 * 而主机侧处理是免费的（而且 Python 侧能对同一份轨迹反复改口径）。
 */

#include "nvbit_tool.h"

#include "nvbit.h"
#include "ks_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* ------------------------------------------------------------------ 配置 */
static std::string g_out = "ks-memtrace.txt";
static std::string g_filter;
static int g_stride = 1;
static int g_max_records = 200000;
static int g_verbose = 0;
static int g_debug_calls = 0;

static void read_env() {
    if (const char* e = getenv("KS_MEMTRACE_OUT")) g_out = e;
    if (const char* e = getenv("KS_MEMTRACE_KERNEL")) g_filter = e;
    if (const char* e = getenv("KS_MEMTRACE_BLOCK_STRIDE")) g_stride = atoi(e);
    if (const char* e = getenv("KS_MEMTRACE_MAX_RECORDS")) g_max_records = atoi(e);
    if (const char* e = getenv("KS_MEMTRACE_VERBOSE")) g_verbose = atoi(e);
    if (const char* e = getenv("KS_MEMTRACE_DEBUG_CALLS")) g_debug_calls = atoi(e);
    if (g_stride < 1) g_stride = 1;
    if (g_max_records < 1) g_max_records = 1;
}

/* ------------------------------------------------------- 静态指令表（legend） */
struct InsnInfo {
    std::string func;
    std::string opcode;
    uint64_t offset = 0;
    int size = 4;
};

static std::unordered_map<uint32_t, InsnInfo> g_insns;
static uint32_t g_next_insn = 0;

/* 访问宽度：从 opcode 后缀推断（LDG.E.128 -> 16B，LDG.E.64 -> 8B，默认 4B） */
static int size_from_opcode(const std::string& op) {
    if (op.find(".128") != std::string::npos) return 16;
    if (op.find(".64") != std::string::npos) return 8;
    if (op.find(".32") != std::string::npos) return 4;
    if (op.find(".16") != std::string::npos) return 2;
    if (op.find(".8") != std::string::npos || op.find("U8") != std::string::npos) return 1;
    return 4;
}

/* 只关心会走到 L2/DRAM 的访存：排除共享内存(LDS/STS/ATOMS/REDS)与常量(LDC/ULDC)。
 *
 * R-78（第五轮 V-5）：**全局原子/归约也是 DRAM 流量**。旧实现只认 LDG/STG/LD/ST，
 * 于是纯原子 kernel（RL 训练里的 index_add_/MoE 路由/直方图/损失累加）采出 0 条记录，
 * 解析器只会报"没有任何访存记录"，而不是给出字节数。
 * 顺序要紧：ATOMS/REDS 必须先判，否则被 ATOM/RED 前缀吃掉；
 * 全局原子在 SASS 里是 RED.E.ADD…/ATOMG/CAS（sm_90+ 的归约是 REDG）。 */
static bool is_dram_relevant(const std::string& op) {
    if (op.rfind("LDS", 0) == 0 || op.rfind("STS", 0) == 0) return false;
    if (op.rfind("LDC", 0) == 0 || op.rfind("ULDC", 0) == 0) return false;
    if (op.rfind("ATOMS", 0) == 0 || op.rfind("REDS", 0) == 0) return false;  /* 共享原子 */
    if (op.rfind("LDG", 0) == 0 || op.rfind("STG", 0) == 0) return true;
    if (op.rfind("ATOMG", 0) == 0 || op.rfind("REDG", 0) == 0) return true;   /* 全局原子/归约 */
    if (op.rfind("ATOM", 0) == 0 || op.rfind("RED", 0) == 0) return true;
    if (op.rfind("CAS", 0) == 0) return true;
    if (op.rfind("LD", 0) == 0 || op.rfind("ST", 0) == 0) return true;        /* LDL/STL 等 */
    return false;
}

/* --------------------------------------------------------------- 设备缓冲 */
static ks_rec_t* g_d_recs = nullptr;
static unsigned int* g_d_counter = nullptr;
static ks_cfg_t* g_d_cfg = nullptr;
static ks_cfg_t g_cfg_host{};
static std::unordered_set<CUfunction> g_instrumented;
static std::mutex g_mtx;
static bool g_skip = false;      /* 防止在回调里再触发回调 */

/* -------------------------------------------------------------- launch 状态 */
static bool g_have_launch = false;
static std::string g_kernel;
static uint64_t g_grid = 0;
static uint64_t g_block = 0;
static int g_launches = 0;

/* ------------------------------------------------------------------- 钩子 */
void nvbit_at_init() {
    read_env();
    if (g_verbose) {
        printf("KS-MEMTRACE: out=%s filter=%s stride=%d max_records=%d\n",
               g_out.c_str(), g_filter.empty() ? "(all)" : g_filter.c_str(), g_stride,
               g_max_records);
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    /* 这里不要做 CUDA 分配：NVBit 会警告 "Do not call CUDA memory allocation in
     * nvbit_at_ctx_init(). It will cause deadlocks." —— 统一放到 nvbit_tool_init。 */
}

void nvbit_tool_init(CUcontext ctx) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_d_recs != nullptr) return;          /* 只分配一次 */
    g_skip = true;
    cudaMalloc((void**)&g_d_recs, (size_t)g_max_records * sizeof(ks_rec_t));
    cudaMalloc((void**)&g_d_counter, sizeof(unsigned int));
    cudaMemset(g_d_counter, 0, sizeof(unsigned int));
    g_cfg_host.recs = (uint64_t)g_d_recs;
    g_cfg_host.counter = (uint64_t)g_d_counter;
    g_cfg_host.max_records = g_max_records;
    g_cfg_host.block_stride = g_stride;
    g_cfg_host.debug_calls = g_debug_calls;
    cudaMalloc((void**)&g_d_cfg, sizeof(ks_cfg_t));
    cudaMemcpy(g_d_cfg, &g_cfg_host, sizeof(ks_cfg_t), cudaMemcpyHostToDevice);
    g_skip = false;
    if (g_verbose) {
        printf("KS-MEMTRACE: 设备缓冲 %zu 条记录（%.1f MB）\n", (size_t)g_max_records,
               (double)g_max_records * sizeof(ks_rec_t) / 1e6);
    }
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (CUfunction f : related) {
        if (!g_instrumented.insert(f).second) continue;

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        const std::string fname = nvbit_get_func_name(ctx, f);
        if (g_verbose) printf("KS-MEMTRACE: 检查 %s（%zu 条静态指令）\n", fname.c_str(), instrs.size());

        int n_insn = 0, n_call = 0;
        for (Instr* instr : instrs) {
            const std::string op = instr->getOpcode();
            if (!is_dram_relevant(op)) continue;

            /* 先数一下有几个内存引用操作数：一条指令可能引用多个地址（如 LDG.E.128） */
            int nmref = 0;
            for (int i = 0; i < instr->getNumOperands(); i++) {
                if (instr->getOperand(i)->type == InstrType::OperandType::MREF) nmref++;
            }
            if (nmref == 0) continue;

            const uint32_t insn_id = g_next_insn++;
            InsnInfo info;
            info.func = fname;
            info.opcode = op;
            info.offset = (uint64_t)instr->getOffset();
            info.size = size_from_opcode(op);
            g_insns[insn_id] = info;

            int mref_idx = 0;
            for (int i = 0; i < instr->getNumOperands(); i++) {
                if (instr->getOperand(i)->type != InstrType::OperandType::MREF) continue;

                /* 参数顺序必须与 inject_funcs.cu 里 ks_record_mem 的形参顺序一致 */
                nvbit_insert_call(instr, "ks_record_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);          /* int pred */
                /* 低 24 位指令编号，高 8 位访问宽度（避免多传参数） */
                nvbit_add_call_arg_const_val32(instr, (int)(insn_id | ((uint32_t)info.size << 24)));
                nvbit_add_call_arg_mref_addr64(instr, mref_idx);   /* uint64_t addr */
                nvbit_add_call_arg_launch_val64(instr, 0);         /* 与官方同构占位 */
                nvbit_add_call_arg_const_val64(instr, (uint64_t)g_d_cfg);
                mref_idx++;
                n_call++;
            }
            n_insn++;
        }
        if (g_verbose) {
            printf("KS-MEMTRACE: %s -> 插桩访存指令 %d 条 / 调用 %d 个\n",
                   fname.c_str(), n_insn, n_call);
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name,
                         void* params, CUresult* pStatus) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_skip) return;
    g_skip = true;

    CUfunction func = nullptr;
    uint64_t grid = 0, block = 0;
    if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        func = p->f;
        grid = (uint64_t)p->gridDimX * p->gridDimY * p->gridDimZ;
        block = (uint64_t)p->blockDimX * p->blockDimY * p->blockDimZ;
    } else if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
        func = p->f;
        grid = (uint64_t)p->config->gridDimX * p->config->gridDimY * p->config->gridDimZ;
        block = (uint64_t)p->config->blockDimX * p->config->blockDimY * p->config->blockDimZ;
    }

    if (func != nullptr && !is_exit) {
        const std::string kname = nvbit_get_func_name(ctx, func);
        const bool match = g_filter.empty() || kname.find(g_filter) != std::string::npos;
        if (match) {
            if (!g_have_launch) {
                g_have_launch = true;
                g_kernel = kname;
                g_grid = grid;
                g_block = block;
            }
            instrument_function_if_needed(ctx, func);
            g_launches++;
        }
    }
    g_skip = false;
}

void nvbit_at_term() {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!g_have_launch) {
        fprintf(stderr,
                "KS-MEMTRACE: 没有匹配到任何 kernel（KS_MEMTRACE_KERNEL=%s）—— 没有产出轨迹\n",
                g_filter.c_str());
        return;
    }

    g_skip = true;
    unsigned int used = 0;
    if (g_d_counter) cudaMemcpy(&used, g_d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    const bool truncated = used > (unsigned int)g_max_records;
    size_t n = truncated ? (size_t)g_max_records : (size_t)used;

    std::vector<ks_rec_t> recs(n);
    if (n) cudaMemcpy(recs.data(), g_d_recs, n * sizeof(ks_rec_t), cudaMemcpyDeviceToHost);

    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    std::unordered_set<uint32_t> ctas;
    for (const ks_rec_t& r : recs) ctas.insert(r.cta_id);
    g_skip = false;

    char arch[32];
    snprintf(arch, sizeof(arch), "sm_%d%d", prop.major, prop.minor);
    const double ratio = g_grid ? double(ctas.size()) / double(g_grid) : 1.0;

    std::ofstream f(g_out);
    if (!f) {
        fprintf(stderr, "KS-MEMTRACE: 无法写入 %s\n", g_out.c_str());
        return;
    }
    f << "# ks-memtrace v1\n";
    f << "# kernel " << g_kernel << "\n";
    f << "# arch " << arch << "\n";
    f << "# grid " << g_grid << "\n";
    f << "# block " << g_block << "\n";
    f << "# blocks_traced " << ctas.size() << "\n";
    f << "# sampling_ratio " << ratio << "\n";
    f << "# l2_bytes " << (uint64_t)prop.l2CacheSize << "\n";
    f << "# sm_count " << prop.multiProcessorCount << "\n";
    f << "# launches_traced " << g_launches << "\n";
    f << "# truncated " << (truncated ? 1 : 0) << "\n";
    f << "# note 一条记录 = 一条 warp 级访存指令的一次执行；addrs 为各活跃 lane 的地址\n";
    if (g_launches > 1) {
        f << "# note 多 launch 记录已求和；per-launch 口径 = 总量 / launches_traced\n";
    }
    for (const auto& kv : g_insns) {
        char off[32];
        snprintf(off, sizeof(off), "0x%llx", (unsigned long long)kv.second.offset);
        f << "# insn " << kv.first << " " << kv.second.func << " " << kv.second.opcode << " "
          << off << " size " << kv.second.size << "\n";
    }
    f << "insn,op,lanes,size,sectors,smid,addrs\n";

    size_t written = 0;
    for (const ks_rec_t& r : recs) {
        if (r.active_mask == 0) continue;
        auto it = g_insns.find(r.insn_id);
        const std::string op = (it != g_insns.end()) ? it->second.opcode : std::string("UNKNOWN");
        f << r.insn_id << "," << op << "," << (int)r.lanes << "," << (int)r.size << ",0,"
          << (int)r.smid << ",";
        bool first = true;
        for (int i = 0; i < KS_MAX_LANES; i++) {
            if (!(r.active_mask & (1u << i))) continue;      /* 被谓词关掉的 lane 地址无意义 */
            if (!first) f << ";";
            f << "0x" << std::hex << (unsigned long long)r.addrs[i] << std::dec;
            first = false;
        }
        f << "\n";
        written++;
    }
    f.close();

    /* 关键诊断：写出**原始计数器值**。debug_calls 模式下不写记录，靠它判断
     * "注入的调用到底有没有发生"（written=0 但 counter>0 说明调用发生了）。 */
    printf("KS-MEMTRACE: 写出 %s —— %zu 条记录（计数器 %u），kernel=%s，launches=%d，"
           "blocks_traced=%zu/%llu，truncated=%d\n",
           g_out.c_str(), written, used, g_kernel.c_str(), g_launches, ctas.size(),
           (unsigned long long)g_grid, truncated ? 1 : 0);
    if (written == 0) {
        fprintf(stderr,
            "\nKS-MEMTRACE: 警告 —— 插桩了 %s 但**没有任何访存记录**。\n"
            "  这不是『kernel 没有访存』：已确认该 kernel 有访存指令。可能原因：\n"
            "    · 注入的调用没有真正执行（本环境实测过；见 README 的『状态』一节）\n"
            "    · 工具架构与 GPU 不匹配（用 ARCH=native 重建）\n"
            "  排查：KS_MEMTRACE_DEBUG_CALLS=1 重跑，看计数是否 >0；\n"
            "        或先跑官方 instr_count 确认 NVBit 在本机可用。\n",
            g_kernel.c_str());
    }
    if (truncated) {
        printf("KS-MEMTRACE: 注意 —— 记录数达到上限 %d，轨迹被截断。"
               "分析器会因此拒绝给出 DRAM 字节（截断值只是下界）。"
               "请提高 KS_MEMTRACE_MAX_RECORDS 或增大 KS_MEMTRACE_BLOCK_STRIDE。\n",
               g_max_records);
    }
}
