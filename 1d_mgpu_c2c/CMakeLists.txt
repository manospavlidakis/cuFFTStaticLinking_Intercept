cmake_minimum_required(VERSION 3.18)

set(ROUTINE 1d_c2c)

project(
  "${ROUTINE}_example"
  DESCRIPTION "GPU-Accelerated Fast Fourier Transforms"
  HOMEPAGE_URL "https://docs.nvidia.com/cuda/cufft/index.html"
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_ARCHITECTURES 80)
#if(CMAKE_CUDA_ARCHITECTURES LESS 60)
	#set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
	#endif()
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUFFT_LIBRARIES ${CUDA_cufft_LIBRARY} ${CUDA_culibos_LIBRARY} ${CUDA_cudart_LIBRARY})


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

target_link_libraries(${ROUTINE}_example PRIVATE CUDA::cufft_static CUDA::cudart)

