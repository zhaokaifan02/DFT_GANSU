cmake_minimum_required(VERSION 3.24)

project(dftgrid LANGUAGES CXX CUDA)

# 架构和标准
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 全局设置，方便做 .so 等
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# -------- 公共头文件路径（所有目标可见）--------
# 如果你的头都在 ${CMAKE_SOURCE_DIR}/include
# 且源码里用的是 #include "boys.hpp" 这种短路径，这里就够了
# 若你的头在 include/integrals/...，保留这行并在源码里用相对 include 路径 #include "integrals/int2c2e.hpp"
# 或者额外再添加该子目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# -------- 子库：int* （每个库也分别标注 include 和属性）--------
add_library(int1e STATIC src/int1e.cu)
target_include_directories(int1e PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(int1e PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RUNTIME_LIBRARY       Shared
)

add_library(int2e STATIC src/int2e.cu)
target_include_directories(int2e PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(int2e PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RUNTIME_LIBRARY       Shared
)

add_library(int2c2e STATIC src/int2c2e.cu)
target_include_directories(int2c2e PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(int2c2e PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RUNTIME_LIBRARY       Shared
)

add_library(int3c2e STATIC src/int3c2e.cu)
target_include_directories(int3c2e PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(int3c2e PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RUNTIME_LIBRARY       Shared
)

# -------- 主库 dftgrid --------
add_library(dftgrid STATIC
  src/dft_atomGrid.cpp
  src/dft_constant.cpp
  src/dft_lebedev.cpp
  src/dft_partition_gpu.cu
  src/basis_set.cpp
  src/shell_eval.cpp
  src/dft_genGrid.cpp
  src/lda.cu
  src/io_txt.cpp
  src/parseXYZ.cpp
  src/hf.cu
  src/rhf.cu
  src/uhf.cu
  src/rohf.cu
  src/gpu_manager.cu
  src/gpu_kernels.cu
  src/profiler.cu
  src/parameter_manager.cpp
  src/eri.cu
  src/utils.cpp
)

target_include_directories(dftgrid PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dftgrid
  PUBLIC
    Eigen3::Eigen
    int1e
    int2e
    int2c2e
    int3c2e
)
set_target_properties(dftgrid PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RUNTIME_LIBRARY       Shared
  OUTPUT_NAME                "dftgrid"
)

# -------- 可执行程序 --------
add_executable(test_atomgrid ${CMAKE_SOURCE_DIR}/test_atomgrid4.cpp)
target_link_libraries(test_atomgrid PRIVATE dftgrid Eigen3::Eigen)

if(MSVC)
  target_compile_options(dftgrid       PRIVATE /utf-8)
  target_compile_options(test_atomgrid PRIVATE /utf-8)
endif()
