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

add_test(
  NAME ckl-python-task-tracing
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPYTHONPATH_VALUE=${PROJECT_SOURCE_DIR}/python:${PROJECT_BINARY_DIR}/python:${CKL_MLIR_PYTHON_ROOT}:${CMAKE_CURRENT_SOURCE_DIR}
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/emit_traced_task.py
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_traced_task.cmake
)

add_test(
  NAME ckl-python-jit
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPYTHONPATH_VALUE=${PROJECT_SOURCE_DIR}/python:${PROJECT_BINARY_DIR}/python:${CKL_MLIR_PYTHON_ROOT}:${CMAKE_CURRENT_SOURCE_DIR}
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/emit_jit_task.py
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_jit.cmake
)

add_test(
  NAME ckl-python-device-kernel
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPYTHONPATH_VALUE=${PROJECT_SOURCE_DIR}/python:${PROJECT_BINARY_DIR}/python:${CKL_MLIR_PYTHON_ROOT}:${CMAKE_CURRENT_SOURCE_DIR}
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/emit_device_kernel.py
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_device_kernel.cmake
)

if(CKL_ENABLE_CUDA_TESTS)
  add_test(
    NAME ckl-python-nvidia-artifact
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${PROJECT_BINARY_DIR}/python:${CKL_MLIR_PYTHON_ROOT}
      CKL_OPT=$<TARGET_FILE:ckl-opt>
      CKL_NVIDIA_INPUT=${PROJECT_SOURCE_DIR}/tests/Dialect/nvidia_fragment_io.mlir
      CKL_CUDA_TOOLKIT_ROOT=${CKL_CUDA_TOOLKIT_ROOT}
      CKL_CUDA_ARCH=${CKL_CUDA_ARCH}
      CKL_CUDA_PTX_FEATURE=${CKL_CUDA_PTX_FEATURE}
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/emit_nvidia_artifact.py
  )
  set_tests_properties(ckl-python-nvidia-artifact PROPERTIES LABELS "cuda-compile")
  add_test(
    NAME ckl-python-nvidia-mma-e2e
    COMMAND ${CMAKE_COMMAND}
      -DCKL_OPT=$<TARGET_FILE:ckl-opt>
      -DCKL_CUDA_RUNNER=$<TARGET_FILE:ckl-nvidia-runtime-validation>
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
      -DPYTHONPATH_VALUE=${PROJECT_SOURCE_DIR}/python:${PROJECT_BINARY_DIR}/python:${CKL_MLIR_PYTHON_ROOT}
      -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/run_nvidia_mma_example.py
      -DEXAMPLE_FILE=${PROJECT_SOURCE_DIR}/examples/nvidia_mma.py
      -DCUDA_ROOT=${CKL_CUDA_TOOLKIT_ROOT}
      -DCUDA_ARCH=${CKL_CUDA_ARCH}
      -DCUDA_PTX_FEATURE=${CKL_CUDA_PTX_FEATURE}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_nvidia_mma_jit.cmake
  )
  set_tests_properties(ckl-python-nvidia-mma-e2e PROPERTIES LABELS "cuda")
endif()
