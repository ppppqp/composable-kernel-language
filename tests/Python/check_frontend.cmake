execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHON_DIR}"
          "${Python3_EXECUTABLE}" "${INPUT_FILE}"
  RESULT_VARIABLE emit_result OUTPUT_VARIABLE source ERROR_VARIABLE emit_error)
if(NOT emit_result EQUAL 0)
  message(FATAL_ERROR "Python frontend failed:\n${emit_error}")
endif()
set(generated "${CMAKE_CURRENT_BINARY_DIR}/frontend.mlir")
file(WRITE "${generated}" "${source}")
execute_process(
  COMMAND "${CKL_OPT}" "${generated}"
  RESULT_VARIABLE parse_result OUTPUT_VARIABLE output ERROR_VARIABLE parse_error)
if(NOT parse_result EQUAL 0)
  message(FATAL_ERROR "generated CKL failed to parse:\n${parse_error}\n${source}")
endif()
if(NOT output MATCHES "ckl.task @copy" OR NOT output MATCHES "ckl.invoke @copy")
  message(FATAL_ERROR "generated CKL did not survive round trip:\n${output}")
endif()
