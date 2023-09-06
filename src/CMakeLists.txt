enable_language(CUDA)

add_library(mat SHARED matrix.cu)
target_link_libraries(mat CUDA::cudart CUDA::cublas OpenMP::OpenMP_CXX)