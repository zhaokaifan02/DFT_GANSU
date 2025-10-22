cmake_minimum_required(VERSION 3.24)

# 按你的显卡改；或直接用 native
set(CMAKE_CUDA_ARCHITECTURES native)

project(dftgrid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 统一把可执行输出到 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---- 静态库 dftgrid ----
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
)
target_include_directories(dftgrid PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dftgrid PUBLIC Eigen3::Eigen)
set_target_properties(dftgrid PROPERTIES
  POSITION_INDEPENDENT_CODE ON  
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared
  OUTPUT_NAME "dftgrid"  
)

# ---- 可执行程序 test_atomgrid ----
add_executable(test_atomgrid ${CMAKE_SOURCE_DIR}/test_atomgrid4.cpp)
target_link_libraries(test_atomgrid PRIVATE dftgrid Eigen3::Eigen) # 链接静态库和 Eigen
# 可执行程序 test_atomgrid_shared (可选)

# MSVC 源文件用 UTF-8（可选）
if(MSVC)
  target_compile_options(dftgrid       PRIVATE /utf-8)
  target_compile_options(test_atomgrid PRIVATE /utf-8)
endif()