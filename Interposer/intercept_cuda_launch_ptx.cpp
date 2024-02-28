#include <cuda_runtime.h> //Runtime API
#include <cuda.h> //Driver API
#include <cublas_v2.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <string>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <streambuf>
#include <chrono>
#include "rapidjson/document.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <time.h>
//#define DEBUG
#define TIMERS
#define RED "\033[1;31m"
#define RESET "\033[0m"
uint64_t s_compute;
uint64_t e_compute;

uint64_t s_index_map1;
uint64_t e_index_map1;


uint64_t s_args;
uint64_t e_args;


std::unordered_map<const void*, char *> ptr2name;
// key: krnl name, values: ptx, params
std::unordered_map<std::string, std::pair<std::string, int>> kernelMap;
// key: ptx, values: krnl name
std::unordered_map<std::string, std::vector<std::string>> PTXMap;

// key: ptx, value cumodule
std::unordered_map<std::string, CUmodule> ptx2module;

// key: Kernel name, value CUfunction
std::unordered_map<std::string, CUfunction> name2CUfunc;

//Map with key: krnl_ptr and values a pair of: krnl_name, ptx, #params
//std::unordered_map<const void*, std::tuple<std::string ,std::string, int>> ptr2ptx_param;

//Map with key: krnl_ptr and values a pair of: CUfunction, #params
std::unordered_map<const void*, std::pair<CUfunction, int>> ptr2CUfunc_param;

std::chrono::high_resolution_clock::time_point s_map_chrono;
std::chrono::high_resolution_clock::time_point e_map_chrono;
/*
std::chrono::high_resolution_clock::time_point s_1;
std::chrono::high_resolution_clock::time_point e_1;

std::chrono::high_resolution_clock::time_point s_2;
std::chrono::high_resolution_clock::time_point e_2;
*/
int firstcudaLaunch = 0;

uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

CUresult err;
#define CUDA_ERROR_FATAL(err)                                                  \
  cudaErrorCheckFatal(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused))
cudaErrorCheckFatal(CUresult err, const char *func, const char *file,
                    size_t line) {
  const char* err_str = nullptr;
  if (err != CUDA_SUCCESS) {
    cuGetErrorString(err, &err_str);
    std::cerr << RED << func << " error : " << RESET << err_str<<" "<<err<< std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    exit(1);
  }
}

#define CUDA_ERROR_FATAL_RUNTIME(err)                                                  \
  cudaErrorCheckFatal_Runtime(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused))
cudaErrorCheckFatal_Runtime(cudaError_t err, const char *func, const char *file,
                    size_t line) {
  if (err != cudaSuccess) {
    std::cerr << RED << func << " error : " << RESET << cudaGetErrorString(err)
              << std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    exit(1);
  }
}
extern "C"{
static cudaError_t (*orig_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL;

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, 
		void **args, size_t sharedMem, cudaStream_t stream) {

    orig_cudaLaunchKernel = (cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t))dlsym(RTLD_NEXT,"cudaLaunchKernel");
#ifdef TIMERS
    s_compute = rdtsc();
#endif

    CUDA_ERROR_FATAL_RUNTIME(orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));

#ifdef TIMERS
    CUDA_ERROR_FATAL_RUNTIME(cudaDeviceSynchronize());
    e_compute = rdtsc();
#endif

return cudaSuccess;
}

cudaError_t (*cudaDeviceReset_original) ();
cudaError_t cudaDeviceReset() {
    std::cerr << "Intercept cudaDeviceReset" << std::endl;
    if (!cudaDeviceReset_original) {
        cudaDeviceReset_original = (cudaError_t (*)()) dlsym(RTLD_NEXT, "cudaDeviceReset");
    }
#ifdef TIMERS
    std::cerr<<"====== Inside cudaLaunchKrnl ======"<<std::endl;
    printf("Index cycles: %.3lu \n", (e_index_map1 - s_index_map1));
    printf("Args cycles: %.3lu \n", (e_args - s_args));
    printf("PTX cycles: %.3lu \n", (e_compute - s_compute));
    
   std::cerr<<"===================================="<<std::endl;                                                                                  
#endif
 
   // Call the original cudaFree function
    return cudaDeviceReset_original();
}
//cudaError_t (*cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original) (int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags);

/*cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags( int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) 
{
   printf("Intercept cudaOccupancyMaxActiveBlocks\n");
   cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original = (cudaError_t (*)(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags)) dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
   // Call the original cudaFree function
   return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original(numBlocks, func, blockSize, dynamicSMemSize, flags);
}*/

CUresult (*cuDeviceGetAttribute_original) (int*, CUdevice_attribute, CUdevice);

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
    printf("Intercept cuDeviceGetAttribute\n");
    if (!cuDeviceGetAttribute_original)
    {
        cuDeviceGetAttribute_original = (CUresult (*)(int*, CUdevice_attribute, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    }

    return cuDeviceGetAttribute_original(pi, attrib, dev);
}

}
