# SPDX-License-Identifier: Apache-2.0
# 用 NVBit 自带的 CMake 助手构建本工具（与 nvbit_memtrace 完全同一套理由）。
#
# 为什么不直接 nvcc：官方 add_nvbit_tool() 还做了三件必需的事，缺一个就会失败：
#   1. inject_funcs 目标加 --keep-device-functions（否则注入函数被当死代码删掉，
#      运行时报 "instrumentation function ... not found in binary!"）
#   2. -Xptxas -astoolspatch（NVBit 的 AST 补丁路径）
#   3. 工具 .so 走设备链接（CUDA_RESOLVE_DEVICE_SYMBOLS）并同时链 -lcuda
#      （本项目还用到驱动 API cuOccupancyMaxActiveBlocksPerMultiprocessor 算理论上界，
#        它就在 libcuda 里，而 NVBit 的助手已经链了 -lcuda）
#
# 打包方式（官方约定）：把本目录拷到 $NVBIT_PATH/tools/ 下，再：
#   cd $NVBIT_PATH/tools/ks_occupancy
#   cmake -B build -DNVBIT_TOOL_CUDA_ARCHITECTURES=native
#   cmake --build build -j
# 产物：build/ks_occupancy.so（仓库里的 Makefile 会把这几步包起来）
#
# ⚠️ 未在真机验证：本组合（本工具 + 任何 NVBit 版本）**没有**在 GPU 上跑过 ——
#    本仓库真机唯一钉住可用的是官方 mem_trace 的 v1.7.7.3 + 老 Makefile 组合，
#    见 ../nvbit_memtrace/README.md 的版本矩阵。构建方式与可用性都是**构建级**的结论。

cmake_minimum_required(VERSION 3.29)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(ks_occupancy LANGUAGES CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/NVBitTool.cmake")
endif()

# 默认源 = 本目录 *.cu 去掉 inject_funcs.cu（occupancy.cu 会被自动收集）
add_nvbit_tool(ks_occupancy)
