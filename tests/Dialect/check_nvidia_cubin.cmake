set(output_file "${CMAKE_CURRENT_BINARY_DIR}/nvidia-cubin.mlir")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "CUDA_HOME=${CUDA_ROOT}"
    "CUDA_PATH=${CUDA_ROOT}"
    "CUDAToolkit_ROOT=${CUDA_ROOT}"
    "PATH=${CUDA_ROOT}/bin:$ENV{PATH}"
    "LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$ENV{LD_LIBRARY_PATH}"
    "${CKL_OPT}" "${INPUT_FILE}"
    --ckl-nvidia-materialize-mma-sync
    --ckl-nvidia-lower-mma-sync
    --ckl-nvidia-lower-fragment-io
    "--gpu-lower-to-nvvm-pipeline=cubin-chip=${CUDA_ARCH} cubin-features=${CUDA_PTX_FEATURE} cubin-format=bin"
    -o "${output_file}"
  RESULT_VARIABLE result
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "CKL-to-CUBIN pipeline failed:\n${error}")
endif()

file(READ "${output_file}" output)
foreach(expected "gpu.binary" "#gpu.object" "bin =")
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "serialized output is missing ${expected}")
  endif()
endforeach()
foreach(unexpected "gpu.module" "ckl." "ckl_nvidia." "nvgpu." "nvvm.mma")
  string(FIND "${output}" "${unexpected}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR "serialized output still contains ${unexpected}")
  endif()
endforeach()

execute_process(
  COMMAND "${CKL_CUDA_RUNNER}" "${output_file}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error
)
if(NOT run_result EQUAL 0)
  message(FATAL_ERROR "CUBIN runtime validation failed:\n${run_error}")
endif()
if(NOT run_output MATCHES "validated 128 results")
  message(FATAL_ERROR "unexpected runtime validation output: ${run_output}")
endif()
