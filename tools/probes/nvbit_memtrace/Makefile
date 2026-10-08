# 构建 Kernoscope 的 NVBit 访存轨迹工具。
#
#   make NVBIT_PATH=$HOME/nvbit_release_x86_64
#   make selftest NVBIT_PATH=...        # 用已知访存形态自测（期望 4 / 32 / 5）
#   make ARCH=various 不支持；用 ARCH=native|all-major|sm_89 等
#   make clean
#
# ⚠️ NVBit 是 NVIDIA 专有许可软件，本项目**不分发**它。自行获取并遵守其许可；
#    本目录只包含我们自己的源码。
#
# 构建方式说明：必须用 NVBit 的 CMake 助手（见 CMakeLists.txt 里的原因）。
# 直接把本目录拷进 $NVBIT_PATH/tools/ 再 cmake，等价于官方示例工具的构建流程。

NVBIT_PATH ?= $(HOME)/nvbit_release_x86_64
ARCH ?= native
TOOL := ks_memtrace
WORK := $(NVBIT_PATH)/tools/$(TOOL)
SO := $(TOOL).so

all: $(SO)

$(SO): CMakeLists.txt inject_funcs.cu memtrace.cu ks_common.h
	@test -d "$(NVBIT_PATH)/tools/cmake" || { \
	  echo "错误：找不到 $(NVBIT_PATH)/tools/cmake —— 请用 NVBIT_PATH=<你的 NVBit 解压目录>"; exit 1; }
	rm -rf "$(WORK)"
	mkdir -p "$(WORK)"
	cp CMakeLists.txt inject_funcs.cu memtrace.cu ks_common.h "$(WORK)/"
	cd "$(WORK)" && cmake -B build -DNVBIT_TOOL_CUDA_ARCHITECTURES=$(ARCH) >/dev/null
	cd "$(WORK)" && cmake --build build -j
	cp "$(WORK)/build/$(SO)" ./$(SO)

memtest: memtest.cu
	nvcc -O3 -arch=native memtest.cu -o $@

selftest: memtest $(SO)
	@for m in coalesced strided32 unaligned; do \
	  CUDA_INJECTION64_PATH=$(CURDIR)/$(SO) LD_PRELOAD=$(CURDIR)/$(SO) \
	    KS_MEMTRACE_KERNEL=k_$$m KS_MEMTRACE_OUT=ks-memtrace-$$m.txt \
	    ./memtest $$m 1024 2>&1 | grep -E "KS-MEMTRACE: 写出" || true; \
	  echo "--- $$m（期望：coalesced/unaligned 各见下，sector 数 coalesced=4、strided32=32、unaligned=5）---"; \
	  kernoscope memtrace --from ks-memtrace-$$m.txt 2>&1 | grep -E "sectors/request|合并效率|error" || true; \
	done

clean:
	rm -f *.o *.so memtest ks-memtrace*.txt
	rm -rf "$(WORK)"

.PHONY: all clean selftest
