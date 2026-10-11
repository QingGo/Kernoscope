# 构建 Kernoscope 的 NVBit 达到-occupancy 工具（被测 kernel 零源码改动）。
#
#   make NVBIT_PATH=$HOME/nvbit_release_x86_64
#   make ARCH=sm_89 NVBIT_PATH=...     # 注入代码必须与目标 GPU 匹配；默认 native
#   make resolve NVBIT_PATH=...        # 只回答"这次会用哪份 .so、它是哪个版本"（不需要 GPU）
#   make clean
#
# ⚠️ NVBit 是 NVIDIA 专有许可软件，本项目**不分发**它。自行获取并遵守其许可；
#    本目录只包含我们自己的源码。
#
# ⚠️ 已知可用组合是**构建级**的结论，而且是官方 mem_trace 的结论（见
#    ../nvbit_memtrace/README.md 的版本矩阵：本仓库真机唯一钉住可用的是
#    v1.7.7.3 + 老 Makefile；同版本号的另一份构建结果相反）。**本工具在任何 NVBit
#    版本上都没有在真机验证过** —— 不要只看版本号就以为能跑，先按 README 的
#    「未在真机验证」一节做小流量试采 + 与 occupancy_probe.cu 对拍。
#
# 构建方式说明：必须用 NVBit 的 CMake 助手（见 CMakeLists.txt 里的原因）。
# 直接把本目录拷进 $NVBIT_PATH/tools/ 再 cmake，等价于官方示例工具的构建流程。

NVBIT_PATH ?= $(HOME)/nvbit_release_x86_64
ARCH ?= native
TOOL := ks_occupancy
WORK := $(NVBIT_PATH)/tools/$(TOOL)
SO := $(TOOL).so

all: $(SO)

$(SO): CMakeLists.txt inject_funcs.cu occupancy.cu ks_occ_common.h
	@test -d "$(NVBIT_PATH)/tools/cmake" || { \
	  echo "错误：找不到 $(NVBIT_PATH)/tools/cmake —— 请用 NVBIT_PATH=<你的 NVBit 解压目录>"; exit 1; }
	rm -rf "$(WORK)"
	mkdir -p "$(WORK)"
	cp CMakeLists.txt inject_funcs.cu occupancy.cu ks_occ_common.h "$(WORK)/"
	cd "$(WORK)" && cmake -B build -DNVBIT_TOOL_CUDA_ARCHITECTURES=$(ARCH) >/dev/null
	cd "$(WORK)" && cmake --build build -j
	cp "$(WORK)/build/$(SO)" ./$(SO)

# 不需要 GPU：把"这次会用哪份 .so、它是哪个版本"以机器可读的两行打出来
# （版本从实物读；读不出写 n/a，绝不编）。
resolve:
	bash ./collect.sh --resolve-only

clean:
	rm -f *.o *.so ks-occupancy*.txt ks-occupancy*.log
	rm -rf "$(WORK)"

.PHONY: all resolve clean
