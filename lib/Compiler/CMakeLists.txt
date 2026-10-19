find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE nanobind_status)
if(NOT nanobind_status EQUAL 0)
  message(FATAL_ERROR "Install nanobind in ${Python_EXECUTABLE}'s environment")
endif()
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(CKLCompiler NB_STATIC NOMINSIZE NativeBindings.cpp RuntimeBindings.cpp)
set_target_properties(CKLCompiler PROPERTIES
  OUTPUT_NAME "_ckl_native"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/ckl")
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

# Python and CKL must use one MLIR implementation and process-wide pass registry.
# Put the Python CAPI library before transitive static MLIR archives, so the
# extension resolves upstream symbols there instead of embedding a second copy.
find_library(CKL_MLIR_PYTHON_CAPI NAMES MLIRPythonCAPI
  PATHS "${CKL_MLIR_PYTHON_PATH}/mlir/_mlir_libs" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(CKLCompiler PRIVATE
  "${CKL_MLIR_PYTHON_CAPI}"
  CKLTransforms CKLAnalysis CKLDialect CKLExecDialect
  MLIRRegisterAllDialects MLIRRegisterAllExtensions
  MLIRToLLVMIRTranslationRegistration)

# CKL's static archive symbols remain private; upstream MLIR symbols come from
# the shared Python CAPI library above.
if(UNIX AND NOT APPLE)
  target_link_options(CKLCompiler PRIVATE "LINKER:--exclude-libs,ALL")
endif()
