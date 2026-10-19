if(CKL_ENABLE_MLIR)
  # The extension and its tests must use the same Python ABI.
  set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(ckl-memory-planner-test Core/MemoryPlannerTest.cpp)
target_link_libraries(ckl-memory-planner-test PRIVATE CKLMemoryPlanner)
target_compile_features(ckl-memory-planner-test PRIVATE cxx_std_17)
target_compile_options(ckl-memory-planner-test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME ckl-memory-planner COMMAND ckl-memory-planner-test)

add_executable(ckl-execution-plan-test Core/ExecutionPlanTest.cpp)
target_link_libraries(ckl-execution-plan-test PRIVATE CKLRuntimeCore)
target_compile_features(ckl-execution-plan-test PRIVATE cxx_std_17)
target_compile_options(ckl-execution-plan-test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME ckl-execution-plan COMMAND ckl-execution-plan-test)

set(ckl_python_test_environment "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${CKL_MLIR_PYTHON_PATH}:$ENV{PYTHONPATH}")
if(CKL_ENABLE_MLIR)
  list(APPEND ckl_python_test_environment "CKL_OPT=$<TARGET_FILE:ckl-opt>")
  list(APPEND ckl_python_test_environment "CKL_NATIVE_LIBRARY=$<TARGET_FILE:CKLCompiler>")
  if(CKL_ENABLE_CUDA_RUNTIME)
    list(APPEND ckl_python_test_environment
      "CKL_CUDA_TOOLKIT=${CUDAToolkit_BIN_DIR}/..")
  endif()
endif()

add_test(
  NAME ckl-python-compiler-utils
  COMMAND ${CMAKE_COMMAND} -E env
    ${ckl_python_test_environment}
    ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/Python -p "test_*.py"
)

if(CKL_ENABLE_MLIR)
  add_test(
    NAME ckl-mlir-analysis
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MLIR/test_semantics.py
        $<TARGET_FILE:ckl-opt> ${CMAKE_CURRENT_SOURCE_DIR}/MLIR
  )
endif()

if(CKL_ENABLE_CUDA_RUNTIME)
  add_subdirectory(Runtime)
endif()

if(CKL_ENABLE_HIP_RUNTIME AND NOT CKL_ENABLE_CUDA_RUNTIME)
  add_subdirectory(Runtime)
endif()
