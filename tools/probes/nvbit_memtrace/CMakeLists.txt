# SPDX-License-Identifier: Apache-2.0
# 用 NVBit 自带的 CMake 助手构建本工具。
#
# 为什么不直接 nvcc：官方 add_nvbit_tool() 还做了三件必需的事，缺一个就会失败：
#   1. inject_funcs 目标加 --keep-device-functions（否则注入函数被当死代码删掉，
#      运行时报 "instrumentation function ... not found in binary!"）
#   2. -Xptxas -astoolspatch（NVBit 的 AST 补丁路径）
#   3. 工具 .so 走设备链接（CUDA_RESOLVE_DEVICE_SYMBOLS）并同时链 -lcuda
# 实测：用直连 nvcc 复刻官方工具（instr_count/mem_trace）同样会失败，所以这不是
# 我们代码的问题，而是构建方式的问题。
#
# 打包方式（官方约定）：把本目录拷到 $NVBIT_PATH/tools/ 下，再：
#   cd $NVBIT_PATH/tools/ks_memtrace
#   cmake -B build -DNVBIT_TOOL_CUDA_ARCHITECTURES=native
#   cmake --build build -j
# 产物：build/ks_memtrace.so（仓库根目录的 Makefile 会把这几步包起来）

cmake_minimum_required(VERSION 3.29)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(ks_memtrace LANGUAGES CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/NVBitTool.cmake")
endif()

# 默认源 = 本目录 *.cu 去掉 inject_funcs.cu（memtrace.cu 会被自动收集）
add_nvbit_tool(ks_memtrace)
