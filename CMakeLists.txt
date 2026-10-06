cmake_minimum_required(VERSION 3.20)
project(CKL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CKL_BUILD_TESTS "Build CKL bootstrap tests" ON)
option(CKL_ENABLE_MLIR "Build the CKL MLIR optimizer driver" OFF)
option(CKL_ENABLE_CUDA_RUNTIME "Build the NVIDIA execution runtime" OFF)
option(CKL_ENABLE_HIP_RUNTIME "Build the HIP execution runtime" OFF)
option(CKL_BUILD_CUDA_BENCHMARKS "Build the Milestone 0 CUDA feasibility benchmarks" OFF)
option(CKL_BUILD_HIP_BENCHMARKS "Build the HIP runtime-overhead benchmarks" OFF)

if(CKL_BUILD_HIP_BENCHMARKS)
  set(CKL_ENABLE_HIP_RUNTIME ON)
  # The Stage 1 in-process bridge links the thin runtime into a shared library.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(CKL_BUILD_CUDA_BENCHMARKS)
  enable_language(CUDA)
endif()

if(CKL_ENABLE_CUDA_RUNTIME OR CKL_BUILD_CUDA_BENCHMARKS)
  find_package(CUDAToolkit REQUIRED)
endif()

if(CKL_ENABLE_HIP_RUNTIME)
  find_path(CKL_HIP_INCLUDE_DIR hip/hip_runtime_api.h REQUIRED)
  find_library(CKL_HIP_LIBRARY amdhip64 REQUIRED)
endif()

if(CKL_BUILD_CUDA_BENCHMARKS)
  add_subdirectory(benchmarks/milestone0)
endif()

if(CKL_BUILD_HIP_BENCHMARKS)
  add_subdirectory(benchmarks/hip-runtime-overhead)
  add_subdirectory(benchmarks/flydsl-stage1)
endif()

if(CKL_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Found MLIR: ${MLIR_DIR}")
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
  add_definitions(${LLVM_DEFINITIONS})
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  add_subdirectory(include)
endif()

add_subdirectory(lib)

if(CKL_ENABLE_MLIR)
  add_subdirectory(tools)
endif()

if(CKL_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
