# cuFFT Library - APIs Examples
I have modified the cmake using the comment in stack overflow: https://stackoverflow.com/questions/76061853/cmake-cuda-static-link-with-cublas to use static version of cufft lib. The repo also includes the interposer that intercepts all cuda runtime and driver API calls. 

## Description

This folder demonstrates cuFFT APIs usage.

[cuSOLVER API Documentation](https://docs.nvidia.com/cuda/cufft/index.html)

## cuFFT Samples

##### 1D FFT R2C example

* [cuFFT 1D R2C](1d_r2c/)

    The sample compute 1D FFT using R2C. See example for detailed description.

##### 1D FFT C2C example

* [cuFFT 1D C2C](1d_c2c/)

    The sample compute 1D FFT using C2C. See example for detailed description.
    
##### 2D FFT C2R example

* [cuFFT 2D C2R](2d_c2r/)

    The sample compute 2D FFT using C2R. See example for detailed description.

##### 3D FFT C2C example

* [cuFFT 3D C2C](3d_c2c/)

    The sample compute 3D FFT using C2C. See example for detailed description.

##### MutliGPU 1D FFT C2C example

* [cuFFT MGPU 1D C2C](1d_mgpu_c2c/)

    The sample compute MultiGPU 1D FFT using C2C. See example for detailed description.

##### MutliGPU 3D FFT C2C example

* [cuFFT MGPU 3D R2C:C2R](3d_mgpu_c2c/)

    The sample compute MultiGPU 3D FFT using C2C. See example for detailed description.

##### MutliGPU 3D FFT R2C:C2R example

* [cuFFT MGPU 3D R2C:C2R](3d_mgpu_r2c_c24/)

    The sample compute MultiGPU 3D FFT using R2C and C2R. See example for detailed description.

## Interposer

* [Interposer](Interposer/)

    A tool to intercept CUDA hidden calls performed from cuFFT high-level calls
 



