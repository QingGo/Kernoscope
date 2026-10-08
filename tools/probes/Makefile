# 编译区间重叠 occupancy 探针。
#
#   make                       # 自动探测当前 GPU 架构（native）
#   make ARCH=sm_89            # 指定架构
#   make run                   # 编译并采集 occupancy-records.txt
#
# 注意：NVBit 等外部插桩工具要求工具二进制与目标架构一致（默认列表可能不含你的架构），
# 这里同样建议用 native，避免"看起来编好了但架构不匹配"。

ARCH ?= native
NVCC ?= nvcc
CFLAGS ?= -O3 -arch=$(ARCH)

all: occupancy_probe

occupancy_probe: occupancy_probe.cu
	$(NVCC) $(CFLAGS) $< -o $@

run: occupancy_probe
	./occupancy_probe occupancy-records.txt

clean:
	rm -f occupancy_probe occupancy-records.txt occupancy-records.txt.smem32k occ*.json

.PHONY: all run clean
