execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHONPATH_VALUE}"
          "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE emit_result OUTPUT_VARIABLE source ERROR_VARIABLE emit_error)
if(NOT emit_result EQUAL 0)
  message(FATAL_ERROR "native Python tracing failed:\n${emit_error}")
endif()
if(NOT source MATCHES "\"ckl.load_tile\"" OR
   NOT source MATCHES "\"ckl.store_tile\"")
  message(FATAL_ERROR "trace did not construct native generic operations:\n${source}")
endif()
set(generated "${CMAKE_CURRENT_BINARY_DIR}/frontend-traced-memory.mlir")
file(WRITE "${generated}" "${source}")
execute_process(
  COMMAND "${CKL_OPT}" "${generated}"
  RESULT_VARIABLE parse_result OUTPUT_VARIABLE output ERROR_VARIABLE parse_error)
if(NOT parse_result EQUAL 0)
  message(FATAL_ERROR "native traced CKL failed dialect verification:\n${parse_error}\n${source}")
endif()
if(NOT output MATCHES "ckl.load_tile" OR NOT output MATCHES "ckl.store_tile")
  message(FATAL_ERROR "native traced CKL did not survive round trip:\n${output}")
endif()
