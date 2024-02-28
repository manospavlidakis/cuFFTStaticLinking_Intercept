#include <complex>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cufftXt.h>

#include "cufft_utils.h"
#include "../timer.h"
#define TIMERS
std::chrono::high_resolution_clock::time_point s_handle;
std::chrono::high_resolution_clock::time_point e_handle;

int main(int argc, char *argv[]) {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int n = 8;
    int batch_size = 2;
    int fft_size = batch_size * n;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = std::complex<scalar_type>;

    std::vector<input_type> input(fft_size, 0);
    std::vector<output_type> output(static_cast<int>((fft_size / 2 + 1)));

    for (int i = 0; i < fft_size; i++) {
        input[i] = static_cast<input_type>(i);
    }

    std::printf("Input array:\n");
    for (auto &i : input) {
        std::printf("%f\n", i);
    }
    std::printf("=====\n");

    input_type *d_input = nullptr;
    cufftComplex *d_output = nullptr;

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan1d(&plan, input.size(), CUFFT_R2C, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input_type) * input.size()));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output_type) * output.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_input, input.data(), sizeof(input_type) * input.size(),
                                 cudaMemcpyHostToDevice, stream));
#ifdef TIMERS
    s_compute = std::chrono::high_resolution_clock::now();
    s_compute_cycles = rdtsc();
#endif
    for (int i=0; i<ITERATIONS; i++)
	    CUFFT_CALL(cufftExecR2C(plan, d_input, d_output));
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
    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_output, sizeof(output_type) * output.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    std::printf("Output array:\n");
    for (auto &i : output) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(plan));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}
