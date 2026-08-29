cmake_minimum_required(VERSION 3.20)
project(CKL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CKL_BUILD_TESTS "Build CKL core validation tests" ON)
option(CKL_ENABLE_MLIR "Build the future MLIR integration" OFF)

if(CKL_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Found MLIR: ${MLIR_DIR}")
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
endif()

add_subdirectory(lib)

if(CKL_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
