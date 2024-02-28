CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcufft_static -lculibos -lcudart
FLAGS        := -O3 -std=c++11

all: 1d_c2c_example

1d_c2c_example: 1d_c2c_example.cpp
	nvcc -x cu $(FLAGS) $(INC) 1d_c2c_example.cpp -o 1d_c2c_example $(LIBS)

clean:
	rm -f 1d_c2c_example

.PHONY: clean all
