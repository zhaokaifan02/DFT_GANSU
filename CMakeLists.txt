cmake_minimum_required(VERSION 3.24)

project(dftAO LANGUAGES CXX CUDA)

# ---- GPU 架构（4060 建议 89；也可留 native）----
# set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 可执行输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 关键：用官方包，拿到导入目标 CUDA::cudart / ::cublas / ::cusolver
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas cusolver)
# 如需手动指定：cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda

# 你的静态库
add_library(dftAO STATIC
  src/dft_atomGrid.cpp
  src/dft_constant.cpp
  src/dft_lebedev.cpp
  src/dft_partition_gpu.cu
  src/dft_genGrid.cpp
  src/basis_set.cpp
  src/dft_shell_eval.cpp
  src/dft_aoEVAL.cpp
  src/dft_aoEVAL_kernel.cu
  src/lda.cu
  src/utils.cpp
)

target_include_directories(dftAO PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 用导入目标链接，避免“裸名库”导致找错版本
target_link_libraries(dftAO PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver
)

# 重要：WSL 下设置 RPATH，保证运行期优先加载宿主驱动
set_target_properties(dftAO PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared
  BUILD_RPATH "/usr/lib/wsl/lib;/usr/local/cuda/lib64"
)

# 可执行程序
add_executable(test_atomgrid ${CMAKE_SOURCE_DIR}/test_atomgrid4.cpp)
target_link_libraries(test_atomgrid PRIVATE dftAO)

# 可执行同样加 RPATH（顺序同上）
set_target_properties(test_atomgrid PROPERTIES
  BUILD_RPATH "/usr/lib/wsl/lib;/usr/local/cuda/lib64"
)

# 建议开启：让构建产物保留/使用 ORIGIN 型 RPATH
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# （可选）Windows MSVC 源文件 UTF-8
if(MSVC)
  target_compile_options(dftAO PRIVATE /utf-8)
  target_compile_options(test_atomgrid PRIVATE /utf-8)
endif()

# 友好提示
message(STATUS "CUDAToolkit_ROOT = ${CUDAToolkit_ROOT}")
message(STATUS "cudart   = ${CUDAToolkit_cudart_LIBRARY}")
message(STATUS "cublas   = ${CUDAToolkit_cublas_LIBRARY}")
message(STATUS "cusolver = ${CUDAToolkit_cusolver_LIBRARY}")
