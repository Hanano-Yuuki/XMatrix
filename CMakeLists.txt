cmake_minimum_required(VERSION 3.10)

project(cuda)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 50)
message("[info]: cuda enabled, device info:")
message("[info]: CMAKE_CUDA_COMPILER_ID: ${CMAKE_CUDA_COMPILER_ID}")
message("[info]: CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} -O3 -Xcompiler ${OpenMP_CXX_FLAGS}")

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_subdirectory(src)

add_executable(main main.cpp)
target_link_libraries(main mat)