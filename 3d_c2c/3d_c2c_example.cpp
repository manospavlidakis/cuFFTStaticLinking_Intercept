#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cufftXt.h>

#include "cufft_utils.h"
#include "../timer.h"
#define TIMERS
std::chrono::high_resolution_clock::time_point s_handle;
std::chrono::high_resolution_clock::time_point e_handle;

using dim_t = std::array<int, 3>;

int main(int argc, char *argv[]) {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int n = 2;
    dim_t fft = {n, n, n};
    int batch_size = 2;
    int fft_size = batch_size * fft[0] * fft[1] * fft[2];

    using scalar_type = float;
    using data_type = std::complex<scalar_type>;

    std::vector<data_type> data(fft_size);

    for (int i = 0; i < fft_size; i++) {
        data[i] = data_type(i, -i);
    }

    std::printf("Input array:\n");
    for (auto &i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    cufftComplex *d_data = nullptr;

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlanMany(&plan, fft.size(), fft.data(), nullptr, 1,
                             fft[0] * fft[1] * fft[2],             // *inembed, istride, idist
                             nullptr, 1, fft[0] * fft[1] * fft[2], // *onembed, ostride, odist
                             CUFFT_C2C, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    // Create device data arrays
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(data_type) * data.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(),
                                 cudaMemcpyHostToDevice, stream));

    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
#ifdef TIMERS
    s_compute = std::chrono::high_resolution_clock::now();
    s_compute_cycles = rdtsc();
#endif
    for (int i=0; i<ITERATIONS; i++){
	    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
	    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    }
#ifdef TIMERS
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    e_compute_cycles = rdtsc();
    e_compute = std::chrono::high_resolution_clock::now();

    std::cerr<<"============"<<std::endl;
    printf("Blas func call cycles: %.3lu \n", (e_compute_cycles - s_compute_cycles));
    printf("AVG Blas func call cycles: %.3lu \n", (e_compute_cycles - s_compute_cycles)/ITERATIONS);
    std::chrono::duration<double, std::milli> compute_milli = e_compute - s_compute;
    std::cerr << "Blas func call time : " << compute_milli.count() << " ms" << std::endl;
    std::cerr << "AVG Blas func call time : " << compute_milli.count()/ITERATIONS << " ms" << std::endl;
    std::cerr<<"============"<<std::endl;
#endif
    CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    std::printf("Output array:\n");
    for (auto &i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_data))

    CUFFT_CALL(cufftDestroy(plan));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}
