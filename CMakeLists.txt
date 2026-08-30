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
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include_directories(${PROJECT_BINARY_DIR}/include)
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
