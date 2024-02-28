cmake_minimum_required(VERSION 3.18)

set(ROUTINE 1d_c2c)

project(
  "${ROUTINE}_example"
  DESCRIPTION "GPU-Accelerated Fast Fourier Transforms"
  HOMEPAGE_URL "https://docs.nvidia.com/cuda/cufft/index.html"
  LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80)
find_package(CUDAToolkit REQUIRED)

add_executable(${ROUTINE}_example)


set_target_properties(${ROUTINE}_example
	PROPERTIES
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

target_compile_features(${ROUTINE}_example
	PRIVATE cxx_std_11)

target_sources(${ROUTINE}_example
               PRIVATE ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp)

target_include_directories(${ROUTINE}_example
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
                           ${CMAKE_SOURCE_DIR}/../utils)

target_link_libraries(${ROUTINE}_example PRIVATE CUDA::cufft_static CUDA::culibos CUDA::cudart Threads::Threads dl)


