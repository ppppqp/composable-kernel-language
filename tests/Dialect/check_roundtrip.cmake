execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}"
  RESULT_VARIABLE once_result
  OUTPUT_VARIABLE once_output
  ERROR_VARIABLE once_error
)
if(NOT once_result EQUAL 0)
  message(FATAL_ERROR "first CKL parse/print failed: ${once_error}")
endif()

execute_process(
  COMMAND "${CKL_OPT}" "${INPUT_FILE}"
  COMMAND "${CKL_OPT}"
  RESULT_VARIABLE twice_result
  OUTPUT_VARIABLE twice_output
  ERROR_VARIABLE twice_error
)
if(NOT twice_result EQUAL 0)
  message(FATAL_ERROR "second CKL parse/print failed: ${twice_error}")
endif()
if(NOT once_output STREQUAL twice_output)
  message(FATAL_ERROR "CKL structured syntax is not stable after a second round trip")
endif()
