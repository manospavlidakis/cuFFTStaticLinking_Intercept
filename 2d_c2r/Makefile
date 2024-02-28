CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcufft_static -lculibos -lcudart
FLAGS        := -O3 -std=c++11

all: 2d_c2r_example

2d_c2r_example: 2d_c2r_example.cpp
	nvcc -x cu $(FLAGS) $(INC) 2d_c2r_example.cpp -o 2d_c2r_example $(LIBS)

clean:
	rm -f 2d_c2r_example

.PHONY: clean all
