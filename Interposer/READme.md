* Working with Cuda 11.1
**nvcc -version:
	Cuda compilation tools, release 11.1, V11.1.105
	Build cuda_11.1.TC455_06.29190527_

* Install cuda11.1 to archlinux
** yay -s cuda11.1

* Intructions to run with Ampere
1. g++ -I/opt/cuda/include -fPIC -shared -o libmylib.so intercept_cuda_launch_ptx.cpp -ldl -L/opt/cuda/lib64 -lcudart -lcuda
2. nvcc simple_sgem.cpp -lcublas_static -lcublasLt_static -lculibos -lcudart
3. export CUDA_FORCE_PTX_JIT=1
4. LD_PRELOAD=./libmylib.so ./a.out

SOS if you do not add export CUDA_FORCE_PTX_JIT=1 then it uses ampere_sgemm_128x128_nn from cubin

