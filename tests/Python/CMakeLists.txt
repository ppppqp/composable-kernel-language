find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(CKL_LLVM_BUILD "${MLIR_DIR}/../../.." ABSOLUTE)
set(CKL_MLIR_PYTHON_ROOT "${CKL_LLVM_BUILD}/tools/mlir/python_packages/mlir_core")
if(NOT EXISTS "${CKL_MLIR_PYTHON_ROOT}/mlir/ir.py")
  message(FATAL_ERROR "MLIR Python bindings not found at ${CKL_MLIR_PYTHON_ROOT}")
endif()

add_test(
  NAME ckl-python-unit
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${PROJECT_SOURCE_DIR}/python
    ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR} -p test_frontend.py
)

add_test(
  NAME ckl-python-mlir-roundtrip
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPYTHON_DIR=${PROJECT_SOURCE_DIR}/python
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/emit_copy.py
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_frontend.cmake
)

add_test(
  NAME ckl-python-memory-mlir-roundtrip
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPYTHON_DIR=${PROJECT_SOURCE_DIR}/python
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/emit_memory.py
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_memory_frontend.cmake
)

add_test(
  NAME ckl-python-native-tracing
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPYTHONPATH_VALUE=${PROJECT_SOURCE_DIR}/python:${PROJECT_BINARY_DIR}/python:${CKL_MLIR_PYTHON_ROOT}:${CMAKE_CURRENT_SOURCE_DIR}
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/emit_traced_memory.py
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_tracing_frontend.cmake
)
