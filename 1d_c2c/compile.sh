#!/bin/bash
nvcc 1d_c2c_example.cpp -lcudart -lculibos -lcufft_static -I ../utils/
#nvcc -O3 cublas_amax_example_extended.cu -lcublas_static -lcublasLt_static -lculibos -lcudart -I ../../utils/ -o a2.out

