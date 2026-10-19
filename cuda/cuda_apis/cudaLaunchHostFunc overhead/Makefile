NVCC ?= nvcc
TARGET := cuda_launch_host_func_overhead
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: all clean

all: $(TARGET)

$(TARGET): benchmark.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-pthread $< -o $@

clean:
	$(RM) $(TARGET)
