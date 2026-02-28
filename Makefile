# CUDA GEMM Makefile

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14
# Use sm_70 for Volta (V100), adjust based on your GPU:
#   sm_60: Pascal (P100)
#   sm_70: Volta (V100)
#   sm_75: Turing (RTX 2080)
#   sm_80: Ampere (A100, RTX 3090)
#   sm_90: Hopper (H100)

# Detect GPU architecture automatically if possible
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | cut -d' ' -f1)
ifneq ($(GPU_ARCH),)
    NVCC_FLAGS = -O3 -arch=sm_$(GPU_ARCH) -std=c++14
endif

# Targets
TARGET = gemm
SRC = gemm.cu

.PHONY: all clean run info

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

info:
	@echo "CUDA Compiler: $(NVCC)"
	@echo "Flags: $(NVCC_FLAGS)"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>/dev/null || echo "nvidia-smi not available"
