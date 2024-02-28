#include <array>
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


using dim_t = std::array<int, 2>;

int main(int argc, char *argv[]) {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int n = 2;
    dim_t fft = {n, n};
    int batch_size = 2;
    int fft_size = batch_size * fft[0] * fft[1];

    using scalar_type = float;
    using input_type = std::complex<scalar_type>;
    using output_type = scalar_type;

    std::vector<input_type> input(batch_size * (fft[0] * static_cast<int>(fft[1] / 2 + 1)));
    std::vector<output_type> output(batch_size * (fft[0] * fft[1]), 0);

    for (int i = 0; i < fft_size; i++) {
        input[i] = input_type(i, -i);
    }

    std::printf("Input array:\n");
    for (auto &i : input) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    cufftComplex *d_input = nullptr;
    output_type *d_output = nullptr;

    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlanMany(&plan, fft.size(), fft.data(), nullptr, 1,
                             0,             // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_C2R, batch_size));
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
	    CUFFT_CALL(cufftExecC2R(plan, d_input, d_output));

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
        std::printf("%f\n", i);
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
