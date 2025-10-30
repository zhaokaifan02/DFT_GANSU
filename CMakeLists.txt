cmake_minimum_required(VERSION 3.24)

project(dftAO LANGUAGES CXX CUDA)

# ---- GPU 架构（4060 建议 89；也可留 native）----
# set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ===== 统一输出到与 build 同级的 work/ 目录 =====
set(WORK_DIR "${CMAKE_SOURCE_DIR}/work")
file(MAKE_DIRECTORY "${WORK_DIR}")

# 三类产物（可执行、共享库、静态库）统一输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${WORK_DIR}")  # exe
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${WORK_DIR}")  # .so
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${WORK_DIR}")  # .a

# 多配置生成器（如 Ninja Multi-Config / MSVC）也同样指向 work/
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${cfg}" CFGU)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFGU} "${WORK_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFGU} "${WORK_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFGU} "${WORK_DIR}")
endforeach()

# 依赖
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas cusolver)

# 公共源文件列表
set(DFTAO_SOURCES
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

# ---- 生成静态库（可选）----
add_library(dftAO_static STATIC ${DFTAO_SOURCES})
target_include_directories(dftAO_static PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dftAO_static PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver)
set_target_properties(dftAO_static PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared
)

# ---- 生成共享库 .so ----
add_library(dftAO SHARED ${DFTAO_SOURCES})
target_include_directories(dftAO PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dftAO PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver)
set_target_properties(dftAO PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared
  BUILD_RPATH "/usr/lib/wsl/lib;/usr/local/cuda/lib64"
)

# ---- 可执行文件（链接共享库）----
add_executable(test_atomgrid ${CMAKE_SOURCE_DIR}/test_atomgrid4.cpp)
target_link_libraries(test_atomgrid PRIVATE dftAO)
set_target_properties(test_atomgrid PROPERTIES
  BUILD_RPATH "/usr/lib/wsl/lib;/usr/local/cuda/lib64"
)

# RPATH 相关（建议保留）
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# Windows 源文件 UTF-8（可选）
if(MSVC)
  target_compile_options(dftAO PRIVATE /utf-8)
  target_compile_options(dftAO_static PRIVATE /utf-8)
  target_compile_options(test_atomgrid PRIVATE /utf-8)
endif()

# 友好提示
message(STATUS "产物将输出到: ${WORK_DIR}")
message(STATUS "CUDAToolkit_ROOT = ${CUDAToolkit_ROOT}")
message(STATUS "cudart   = ${CUDAToolkit_cudart_LIBRARY}")
message(STATUS "cublas   = ${CUDAToolkit_cublas_LIBRARY}")
message(STATUS "cusolver = ${CUDAToolkit_cusolver_LIBRARY}")
