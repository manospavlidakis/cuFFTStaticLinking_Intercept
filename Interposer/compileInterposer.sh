#!/bin/bash
#g++ -O3 -I${CUDA_HOME}/include -fPIC -shared -o libmylib.so intercept_cuda_launch_ptx.cpp -ldl -L${CUDA_HOME}/lib64 -lcudart -lcuda

g++ -O3 -I${CUDA_HOME}/include -fPIC -shared -o libmylib_nat.so intercept_cuda_launch_ptx_nat.cpp -ldl -L${CUDA_HOME}/lib64 -lcudart -lcuda

#g++ -O3 -I${CUDA_HOME}/include -fPIC -shared -o libmylib_mod.so intercept_cuda_launch_ptx_modified.cpp -ldl -L${CUDA_HOME}/lib64 -lcudart -lcuda

g++ -O3 -I${CUDA_HOME}/include -fPIC -shared -o libmylib_allocator.so intercept_cuda_launch_ptx_allocator.cpp -ldl -L${CUDA_HOME}/lib64 -lcudart -lcuda


