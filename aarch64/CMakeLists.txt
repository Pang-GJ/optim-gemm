cmake_minimum_required(VERSION 3.15.2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "${CROSSTOOL_ROOT}/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${CROSSTOOL_ROOT}/bin/aarch64-linux-gnu-g++")

project(how-to-optimize-gemm LANGUAGES C CXX ASM VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O2 -g -march=armv8-a -ftree-vectorize ${CMAKE_CXX_FLAGS}")

function(add_bin source_file)
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file} test_MMult.cpp compare_matrices.cpp random_matrix.cpp copy_matrix.cpp dclock.cpp REF_MMult.cpp print_matrix.cpp)
endfunction()

add_bin(MMult0.cpp)
add_bin(MMult1.cpp)
add_bin(MMult_4x4_8.cpp)