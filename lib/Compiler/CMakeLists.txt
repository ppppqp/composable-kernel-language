find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE nanobind_status)
if(NOT nanobind_status EQUAL 0)
  message(FATAL_ERROR "Install nanobind in ${Python_EXECUTABLE}'s environment")
endif()
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(CKLCompiler NB_STATIC NOMINSIZE Compiler.cpp)
set_target_properties(CKLCompiler PROPERTIES
  OUTPUT_NAME "_ckl_native"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/ckl")
target_link_libraries(CKLCompiler PRIVATE
  CKLTransforms CKLAnalysis CKLDialect CKLExecDialect
  MLIRRegisterAllPasses MLIRRegisterAllDialects MLIRRegisterAllExtensions
  MLIRToLLVMIRTranslationRegistration MLIRExecutionEngine MLIRParser)
if(CKL_ENABLE_CUDA_RUNTIME)
  target_compile_definitions(CKLCompiler PRIVATE CKL_HAS_CUDA_RUNTIME=1)
  target_link_libraries(CKLCompiler PRIVATE CKLNvidiaRuntime)
endif()

# Upstream MLIR bindings are supplied by the matching LLVM build/install.
get_filename_component(ckl_mlir_python_default
  "${MLIR_DIR}/../../../tools/mlir/python_packages/mlir_core" ABSOLUTE)
set(CKL_MLIR_PYTHON_PATH "${ckl_mlir_python_default}" CACHE PATH
  "Directory containing the matching upstream mlir Python package")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
  "PYTHONPATH=${CKL_MLIR_PYTHON_PATH}:$ENV{PYTHONPATH}"
  "${Python_EXECUTABLE}" -c "from mlir import ir"
  RESULT_VARIABLE ckl_mlir_python_status)
if(NOT ckl_mlir_python_status EQUAL 0)
  message(FATAL_ERROR "MLIR Python bindings unavailable; set CKL_MLIR_PYTHON_PATH")
endif()

# A static LLVM build may also be embedded in MLIR's Python CAPI library.
# Keep our archive symbols private so their registries cannot interpose each other.
if(UNIX AND NOT APPLE)
  target_link_options(CKLCompiler PRIVATE "LINKER:--exclude-libs,ALL")
endif()
