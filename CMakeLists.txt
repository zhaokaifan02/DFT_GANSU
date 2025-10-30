cmake_minimum_required(VERSION 3.24)

project(dftAO LANGUAGES CXX CUDA)

# ---- GPU Architecture (e.g., 89 for 4060; 'native' is also fine) ----
# set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ===== Unify all outputs to ${CMAKE_SOURCE_DIR}/work/ =====
set(WORK_DIR "${CMAKE_SOURCE_DIR}/work")
file(MAKE_DIRECTORY "${WORK_DIR}")

# Set unified output directories for all three target types (exe, shared, static)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${WORK_DIR}")  # Executables (.exe)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${WORK_DIR}")  # Shared libraries (.so)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${WORK_DIR}")  # Static libraries (.a)

# Also point multi-config generators (e.g., Ninja Multi-Config / MSVC) to work/
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${cfg}" CFGU)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFGU} "${WORK_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFGU} "${WORK_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFGU} "${WORK_DIR}")
endforeach()

# Dependencies
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas cusolver)

# Common source file list
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

# ---- Build static library (optional) ----
add_library(dftAO_static STATIC ${DFTAO_SOURCES})
target_include_directories(dftAO_static PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dftAO_static PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver)
set_target_properties(dftAO_static PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared
)

# ---- Build shared library (.so) ----
add_library(dftAO SHARED ${DFTAO_SOURCES})
target_include_directories(dftAO PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dftAO PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver)
set_target_properties(dfAO PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared
  BUILD_RPATH "/usr/lib/wsl/lib;/usr/local/cuda/lib64"
)

# ---- Executable (links against the shared library) ----
add_executable(test_atomgrid ${CMAKE_SOURCE_DIR}/test_atomgrid4.cpp)
target_link_libraries(test_atomgrid PRIVATE dftAO)
set_target_properties(test_atomgrid PROPERTIES
  BUILD_RPATH "/usr/lib/wsl/lib;/usr/local/cuda/lib64"
)

# RPATH settings (recommended to keep)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# Force Windows source files to UTF-8 (optional)
if(MSVC)
  target_compile_options(dftAO PRIVATE /utf-8)
  target_compile_options(dftAO_static PRIVATE /utf-8)
  target_compile_options(test_atomgrid PRIVATE /utf-8)
endif()

# Status messages
message(STATUS "Outputs will be written to: ${WORK_DIR}")
message(STATUS "CUDAToolkit_ROOT = ${CUDAToolkit_ROOT}")
message(STATUS "cudart   = ${CUDAToolkit_cudart_LIBRARY}")
message(STATUS "cublas   = ${CUDAToolkit_cublas_LIBRARY}")
message(STATUS "cusolver = ${CUDAToolkit_cusolver_LIBRARY}")