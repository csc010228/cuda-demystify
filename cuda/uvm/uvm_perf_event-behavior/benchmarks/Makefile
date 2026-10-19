# uvm_perf_event-behavior · benchmarks
# Each <name>.cu builds into an independent ./<name> binary.
NVCC    ?= nvcc
NVFLAGS ?= -O2 -std=c++17 -arch=native

BINS := $(patsubst %.cu,%,$(wildcard *.cu))

all: $(BINS)

%: %.cu common.cuh
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
