NVCC = nvcc
CXX = g++

# Directories
SRC_DIR = src
INC_DIR = include

# Output file
OUTPUT = vector_addition_benchmark

# Flags
NVCC_FLAGS = -std=c++11 -I$(INC_DIR)
CXX_FLAGS  = -std=c++11 -I$(INC_DIR)

# Source files
CU_SRC  = $(SRC_DIR)/vector_add_gpu.cu
CPP_SRC = $(SRC_DIR)/main.cpp $(SRC_DIR)/vector_add_cpu.cpp

# Object files
OBJ = $(CU_SRC:.cu=.o) $(CPP_SRC:.cpp=.o)

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $@

# Pattern rules
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OUTPUT) $(OBJ)
