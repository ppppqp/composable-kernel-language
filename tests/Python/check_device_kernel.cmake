execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHONPATH_VALUE}"
          "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE emit_result OUTPUT_VARIABLE source ERROR_VARIABLE emit_error)
if(NOT emit_result EQUAL 0)
  message(FATAL_ERROR "device kernel tracing failed:\n${emit_error}")
endif()
if(NOT source MATCHES "gpu.module @copy_kernels" OR
   NOT source MATCHES "gpu.func @copy" OR
   NOT source MATCHES "known_block_size = array<i32: 4, 1, 1>")
  message(FATAL_ERROR "missing native GPU container or launch metadata:\n${source}")
endif()
set(generated "${CMAKE_CURRENT_BINARY_DIR}/frontend-device-kernel.mlir")
file(WRITE "${generated}" "${source}")
execute_process(
  COMMAND "${CKL_OPT}" "${generated}"
  RESULT_VARIABLE parse_result ERROR_VARIABLE parse_error)
if(NOT parse_result EQUAL 0)
  message(FATAL_ERROR "device kernel failed dialect verification:\n${parse_error}")
endif()
