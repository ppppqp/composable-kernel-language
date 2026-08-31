set(output_file "${CMAKE_CURRENT_BINARY_DIR}/python-nvidia-mma-cubin.mlir")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${PYTHONPATH_VALUE}"
    "CKL_OPT=${CKL_OPT}"
    "CKL_CUDA_TOOLKIT_ROOT=${CUDA_ROOT}"
    "CKL_CUDA_ARCH=${CUDA_ARCH}"
    "CKL_CUDA_PTX_FEATURE=${CUDA_PTX_FEATURE}"
    "CUDA_HOME=${CUDA_ROOT}"
    "CKL_NVIDIA_EXAMPLE=${EXAMPLE_FILE}"
    "CKL_NVIDIA_OUTPUT=${output_file}"
    "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE compile_result
  OUTPUT_VARIABLE compile_output
  ERROR_VARIABLE compile_error
)
if(NOT compile_result EQUAL 0)
  message(FATAL_ERROR "Python NVIDIA MMA compilation failed:\n${compile_error}")
endif()
if(NOT compile_output MATCHES "frontend artifact")
  message(FATAL_ERROR "Python compilation did not report a GPU artifact")
endif()
execute_process(
  COMMAND "${CKL_CUDA_RUNNER}" "${output_file}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error
)
if(NOT run_result EQUAL 0)
  message(FATAL_ERROR "Python-generated CUBIN execution failed:\n${run_error}")
endif()
if(NOT run_output MATCHES "validated 128 results")
  message(FATAL_ERROR "unexpected runtime validation output: ${run_output}")
endif()
