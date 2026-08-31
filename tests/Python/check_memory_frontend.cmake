execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHON_DIR}"
          "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE emit_result OUTPUT_VARIABLE source ERROR_VARIABLE emit_error)
if(NOT emit_result EQUAL 0)
  message(FATAL_ERROR "Python memory frontend failed:\n${emit_error}")
endif()
set(generated "${CMAKE_CURRENT_BINARY_DIR}/frontend-memory.mlir")
file(WRITE "${generated}" "${source}")
execute_process(
  COMMAND "${CKL_OPT}" "${generated}"
  RESULT_VARIABLE parse_result OUTPUT_VARIABLE output ERROR_VARIABLE parse_error)
if(NOT parse_result EQUAL 0)
  message(FATAL_ERROR "generated memory CKL failed to parse:\n${parse_error}\n${source}")
endif()
if(NOT output MATCHES "ckl.load_tile" OR NOT output MATCHES "ckl.store_tile" OR
   NOT output MATCHES "#ckl.distribution")
  message(FATAL_ERROR "generated memory CKL did not survive round trip:\n${output}")
endif()
