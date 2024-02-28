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

#define CHK(X) if ((err = X) != CUDA_SUCCESS) printf("CUDA error %d at %d\n", (int)err, __LINE__) 
extern "C"{
int firstRegisterFunc = 0;
std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: could not open file " << filename << std::endl;
	abort();
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
void createModuleMap() {
    CUdevice cuDevice; 
    CUcontext cuContext;
    int devID = 0;
    CUDA_ERROR_FATAL(cuInit(0));
    CUDA_ERROR_FATAL(cuDeviceGet(&cuDevice, devID));
    CUDA_ERROR_FATAL(cuCtxCreate(&cuContext, 0, cuDevice));

    CUmodule cuModule;

    //std::cerr<<"Size of kernelMap_param: "<<kernelMap.size()<<std::endl;
    for (const auto& entry : PTXMap) {
        const std::string& ptx = entry.first;

	std::string path = "../../Interposer/modified_cublas/" + ptx;  
	//std::string path = "../../Interposer/native_cublas/" + ptx;  
	std::ifstream my_file(path);
	std::string my_ptx((std::istreambuf_iterator<char>(my_file)), std::istreambuf_iterator<char>());

	CUDA_ERROR_FATAL(cuModuleLoadData(&cuModule, my_ptx.c_str()));
	ptx2module.emplace(ptx, cuModule);
    }
    /*
    //std::cerr<<"Size of name2Cufunc: "<<name2Cufunc.size()<<std::endl;
    for (const auto& entry : ptx2module) {
      const std::string& name = entry.first;
      std::cerr<<"name:" <<name<<std::endl;
    }
    */
}

void createFunctionMap() 
{
    CUfunction kernelFunc;
    //std::cerr<<"Size of kernelMap_param: "<<kernelMap.size()<<std::endl;
    for (auto& entry : PTXMap) {
        const std::string& ptxName = entry.first;
	//std::cerr<<"PTX: "<<ptxName<<std::endl;
	CUmodule cuModule = ptx2module.at(ptxName);
	std::vector<std::string>& values = entry.second; // get the vector of values
	for (std::string& krnl : values) {
	    const char* fname = krnl.c_str();
	    //std::cerr<<"Function name: "<<fname<<std::endl;
	    CUDA_ERROR_FATAL(cuModuleGetFunction(&kernelFunc, cuModule, fname));
	    name2CUfunc.emplace(fname, kernelFunc);
	}
    }
}

void parseJson()
{
    std::string jsonFile = "../../Interposer/klist.json";
    std::string jsonStr = read_file(jsonFile);
    //std::cerr<<"File name: "<<jsonFile<<std::endl;
    //std::cerr<<"jsonStr: "<<jsonStr<<std::endl;
    rapidjson::Document document;
    document.Parse(jsonStr.c_str());
    if (document.IsObject()) {
        auto const& sm80 = document["80"];
	if (sm80.IsObject()) {
	    for (auto it = sm80.MemberBegin(); it != sm80.MemberEnd(); ++it) {
	        for (auto const& kernel : it->value.GetArray()) {
		    auto const& name = kernel["name"].GetString();
		    auto const& params = kernel["params"].GetArray();
		    const rapidjson::SizeType paramsSize = params.Size();
		    kernelMap.emplace(name, std::make_pair(it->name.GetString(), paramsSize));
		}
	    }
	    for (auto const& ptx : sm80.GetObject()) {
                auto const& ptx_filename = ptx.name.GetString();
                auto const& kernels = ptx.value.GetArray();

                for (auto const& kernel : kernels) {
                    auto const& name = kernel["name"].GetString();
                    PTXMap[ptx_filename].push_back(name);
                }
	    }
	}
    }
}
	

void __cudaRegisterFunction(void **fatCubinHandle,
                const char *hostFun,
                char *deviceFun,
                const char *deviceName,
                int thread_limit,
                uint3 *tid,
                uint3 *bid,
                dim3 *bDim,
                dim3 *gDim,
                int *wSize){
    ptr2CUfunc_param.reserve(1500);
    //std::cerr<<"firstRegisterFunc: "<<firstRegisterFunc<<std::endl;
    if (firstRegisterFunc == 0){
        parseJson();
	//std::cerr<<"Done with Parse JSON"<<std::endl;
	createModuleMap();
	//std::cerr<<"Done with Module Map"<<std::endl;
	createFunctionMap();
	//std::cerr<<"Done with Function Map"<<std::endl;
	firstRegisterFunc=1;
    }
    //fprintf(stderr,"Intercepted register function\n");
    void (*__cudaRegisterFunction_original) (void **, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*) = (void (*)(void **, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    __cudaRegisterFunction_original(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

    //Store for every ptr the kernel name
    ptr2name[hostFun] = (char*)deviceName;
    for (const auto& entry : ptr2name) {
        const void* key = entry.first;
	const char* name = entry.second;
	auto it = kernelMap.find(name);
	if (it != kernelMap.end()) {
	    int param = it->second.second;
	    CUfunction cuFunc = name2CUfunc[name];
	    std::pair<CUfunction, int> pair(cuFunc, param);
	    ptr2CUfunc_param.emplace(key, pair);
	}
    }
    //printf("hostfun:%p, deviceName: %s\n", hostFun, deviceName);
}
#ifdef TIMERS 
unsigned long long results[40];
int count = 0;
#endif

static cudaError_t (*orig_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL;
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, 
		void **args, size_t sharedMem, cudaStream_t stream) {

#ifdef TIMERS 
    //s_map_chrono = std::chrono::high_resolution_clock::now();
    s_index_map1 = rdtsc();
#endif 

    // Get the pair associated with the key
    const auto& pair = ptr2CUfunc_param.at(func);

    // Access the kernel name in the tupple
    CUfunction kernelFunc = std::get<0>(pair);

    // Access the number of args
    int argsNum = std::get<1>(pair);
#ifdef DEBUG
    if (kernelFunc == NULL){
	    std::cerr<<"Can NOT find CUFunction. Abort!!"<<std::endl;
	    abort();
    }
    if (argsNum == 0){
	    std::cerr<<"Args is 0. Abort!"<<std::endl;
	    abort();
    }
#endif
#ifdef TIMERS 
    e_index_map1 = rdtsc();
    //e_map_chrono = std::chrono::high_resolution_clock::now();
    s_args = rdtsc();
#endif    

    printf("Intercepted launch of kernel %p with grid size %d x %d x %d and block size %d x %d x %d - NEW KRNL: %p\n", func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, kernelFunc);

    long int deviceExtraArgument = 0xfffffffffffffff;
    void **newArgs = (void **)malloc(sizeof(void *) * (argsNum + 1));
    memcpy(newArgs, args, sizeof(void *) * argsNum);
    newArgs[argsNum] = &deviceExtraArgument;
    
    orig_cudaLaunchKernel = (cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t))dlsym(RTLD_NEXT,"cudaLaunchKernel");
#ifdef TIMERS
    e_args = rdtsc();
    s_compute = rdtsc();
#endif

   CUDA_ERROR_FATAL_RUNTIME(orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
   //CUresult err = cuLaunchKernel(kernelFunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, newArgs, NULL);

#ifdef TIMERS
    //CUDA_ERROR_FATAL(cuCtxSynchronize());
    cudaDeviceSynchronize();
    e_compute = rdtsc();
    results[count] = e_compute - s_compute;
    count++;
#endif 
   if (err == CUDA_SUCCESS)
        return cudaSuccess;
    else
	return cudaErrorUnknown;
}
/*
cudaError_t (*cudaFree_original) (void *);
cudaError_t cudaFree(void *devPtr) {
    if (!cudaFree_original) {
        cudaFree_original = (cudaError_t (*)(void *)) dlsym(RTLD_NEXT, "cudaFree");
    }
   // Call the original cudaFree function
    return cudaFree_original(devPtr);
}
*/

#ifdef TIMERS
cudaError_t (*cudaDeviceReset_original) ();
cudaError_t cudaDeviceReset() {
    std::cerr << "Intercept cudaDeviceReset" << std::endl;
    if (!cudaDeviceReset_original) {
        cudaDeviceReset_original = (cudaError_t (*)()) dlsym(RTLD_NEXT, "cudaDeviceReset");
    }
    printf("Index cycles: %.3lu \n", (e_index_map1 - s_index_map1));
    printf("Args cycles: %.3lu \n", (e_args - s_args));
    unsigned long long sum = 0;
    std::cerr<<"Count: "<<count<<std::endl;
    for (int i=0; i<count; i++){
	std::cerr<<"count "<<i<<" cycles: "<<results[i]<<std::endl; 
    	sum += results[i];
    }
    printf("PTX cycles: %.3lu \n", (e_compute - s_compute));
    printf("AVG PTX cycles: %.3lu \n", sum/count); 
    std::cerr<<"===================================="<<std::endl;                                                                                   
    // Call the original cudaFree function
    return cudaDeviceReset_original();
}
#endif
//cudaError_t (*cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original) (int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags);

/*cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags( int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) 
{
   printf("Intercept cudaOccupancyMaxActiveBlocks\n");
   cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original = (cudaError_t (*)(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags)) dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
   // Call the original cudaFree function
   return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

CUresult (*cuDeviceGetAttribute_original) (int*, CUdevice_attribute, CUdevice);

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
   printf("Intercept cuDeviceGetAttribute!!!!!!!!\n");
    if (!cuDeviceGetAttribute_original)
    {
        cuDeviceGetAttribute_original = (CUresult (*)(int*, CUdevice_attribute, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    }

    return cuDeviceGetAttribute_original(pi, attrib, dev);
}

nvrtcResult nvrtcCreateProgram(const nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) {
    // Add your interception code here

    // Call the original function
    nvrtcResult res = nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames);

    // Add any additional interception code here

    return res;
}*/
}
